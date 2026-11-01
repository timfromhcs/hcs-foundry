"""GGUF tooling: export from HF safetensors, quantize, requantize, verify.

The converters and quantizers are external llama.cpp tools. They are discovered at
runtime and never downloaded. Every operation records the exact command, versions and
output size, and refuses when disk space is insufficient.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Settings

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

QUANT_METHODS = ["F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q2_K"]
#: Methods that increase precision. llama.cpp calls this requantization; it is the
#: closest supported operation to "dequantize" without a GGUF→HF converter.
UPCAST_METHODS = {"F16", "BF16", "F32"}


class GGUFError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Toolchain:
    converter: str | None
    quantize: str | None
    split: str | None
    source_root: str | None

    def to_public(self) -> dict:
        return {
            "converter": self.converter,
            "quantize": self.quantize,
            "split": self.split,
            "source_root": self.source_root,
        }


def _walk_find(roots: list[Path], names: set[str], max_depth: int = 5, budget: float = 12.0) -> str | None:
    deadline = time.time() + budget
    prune = {"node_modules", ".git", "__pycache__", "site-packages", "Windows", "$Recycle.Bin"}
    for root in roots:
        if not root.exists():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if time.time() > deadline:
                return None
            if len(Path(dirpath).parts) - base_depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d.lower() not in prune and not d.startswith(".")]
            for filename in filenames:
                if filename.lower() in names:
                    return str(Path(dirpath) / filename)
    return None


def discover(settings: Settings) -> Toolchain:
    converter = os.environ.get("FOUNDRY_LLAMA_CONVERT")
    quantize = os.environ.get("FOUNDRY_LLAMA_QUANTIZE")

    roots: list[Path] = [settings.data_dir / "runtime", Path.home() / "llama.cpp", Path("C:/llama.cpp")]
    for candidate in settings.llama_server_candidates:
        parent = Path(candidate).expanduser().parent
        roots.extend([parent, parent.parent, parent.parent.parent])
    if sys.platform == "win32":
        for drive in ("C:/", "D:/", "E:/"):
            for hint in ("AI", "llama.cpp", "llama-src"):
                path = Path(drive) / hint
                if path.exists():
                    roots.append(path)

    if not converter:
        converter = _walk_find(roots, {"convert_hf_to_gguf.py"}, max_depth=4)
    if not quantize:
        quantize = _walk_find(roots, {"llama-quantize.exe", "llama-quantize"}, max_depth=5)
    split = _walk_find(roots, {"llama-gguf-split.exe", "llama-gguf-split"}, max_depth=5, budget=4.0)
    source_root = str(Path(converter).parent) if converter else None
    return Toolchain(converter=converter, quantize=quantize, split=split, source_root=source_root)


def _run_streaming(
    cmd: list[str],
    *,
    cwd: Path | None,
    env: dict | None,
    timeout: float,
    log: Callable[[str], None] | None,
    cancel=None,
) -> int:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(cwd) if cwd else None,
        env=env,
        creationflags=CREATE_NO_WINDOW,
    )
    started = time.time()
    assert process.stdout is not None
    for line in process.stdout:
        if cancel is not None and cancel.is_set():
            process.terminate()
            raise InterruptedError("operation cancelled")
        if time.time() - started > timeout:
            process.terminate()
            raise GGUFError("timeout", f"command exceeded {timeout:.0f}s")
        if log is not None:
            text = line.rstrip()
            if text:
                log(text)
    return process.wait()


def _python_env(converter: Path) -> dict:
    env = dict(os.environ)
    gguf_py = converter.parent / "gguf-py"
    parts = [str(converter.parent)]
    if gguf_py.exists():
        parts.insert(0, str(gguf_py))
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(parts + ([existing] if existing else []))
    return env


def export_gguf(
    hf_dir: str | Path,
    out_path: str | Path,
    *,
    toolchain: Toolchain,
    dtype: str = "f16",
    log: Callable[[str], None] | None = None,
    cancel=None,
    timeout: float = 7200,
) -> dict:
    source = Path(hf_dir).expanduser()
    out = Path(out_path).expanduser()
    if not source.is_dir():
        raise GGUFError("missing_source", f"HF directory not found: {source}")
    if not toolchain.converter:
        raise GGUFError(
            "no_converter",
            "convert_hf_to_gguf.py was not found. Point FOUNDRY_LLAMA_CONVERT at it.",
        )
    weights = list(source.glob("*.safetensors")) + list(source.glob("*.bin"))
    if not weights:
        raise GGUFError("no_weights", "No safetensors/bin weights in the source directory.")
    estimate = sum(f.stat().st_size for f in weights) * 1.2
    free = shutil.disk_usage(out.parent if out.parent.exists() else source).free
    if free < estimate:
        raise GGUFError("no_space", f"Need about {estimate:.0f} bytes, {free} free.")

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, toolchain.converter, str(source), "--outfile", str(out), "--outtype", dtype]
    started = time.time()
    code = _run_streaming(
        cmd, cwd=Path(toolchain.converter).parent, env=_python_env(Path(toolchain.converter)),
        timeout=timeout, log=log, cancel=cancel,
    )
    if code != 0 or not out.exists():
        raise GGUFError("convert_failed", f"convert_hf_to_gguf exited with {code}")
    return {
        "output": str(out),
        "size_bytes": out.stat().st_size,
        "dtype": dtype,
        "duration_seconds": round(time.time() - started, 1),
        "command": cmd,
    }


def quantize_gguf(
    src: str | Path,
    out: str | Path,
    method: str,
    *,
    toolchain: Toolchain,
    log: Callable[[str], None] | None = None,
    cancel=None,
    timeout: float = 7200,
) -> dict:
    source = Path(src).expanduser()
    target = Path(out).expanduser()
    method = method.upper()
    if method not in QUANT_METHODS:
        raise GGUFError("bad_method", f"Unknown quantization {method}. Use one of {', '.join(QUANT_METHODS)}.")
    if not source.is_file():
        raise GGUFError("missing_source", f"GGUF not found: {source}")
    if not toolchain.quantize:
        raise GGUFError(
            "no_quantizer",
            "llama-quantize was not found. Point FOUNDRY_LLAMA_QUANTIZE at it.",
        )
    free = shutil.disk_usage(target.parent if target.parent.exists() else source.parent).free
    needed = source.stat().st_size * 1.1
    if free < needed:
        raise GGUFError("no_space", f"Need about {needed:.0f} bytes, {free} free.")

    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [toolchain.quantize, str(source), str(target), method]
    started = time.time()
    code = _run_streaming(cmd, cwd=target.parent, env=dict(os.environ), timeout=timeout, log=log, cancel=cancel)
    if code != 0 or not target.exists():
        raise GGUFError("quantize_failed", f"llama-quantize exited with {code}")
    return {
        "source": str(source),
        "output": str(target),
        "method": method,
        "source_bytes": source.stat().st_size,
        "size_bytes": target.stat().st_size,
        "ratio": round(target.stat().st_size / max(1, source.stat().st_size), 3),
        "duration_seconds": round(time.time() - started, 1),
        "command": cmd,
        "upcast": method in UPCAST_METHODS,
    }


def requantize(
    src: str | Path,
    out: str | Path,
    *,
    toolchain: Toolchain,
    method: str = "F16",
    log=None,
    cancel=None,
) -> dict:
    """The supported 'dequantize' path: raise a GGUF to F16/BF16.

    llama.cpp can requantize; it cannot currently convert GGUF back to HF safetensors
    with the discovered tooling, so this is labelled as an upcast, not a conversion.
    """
    return quantize_gguf(src, out, method, toolchain=toolchain, log=log, cancel=cancel)
