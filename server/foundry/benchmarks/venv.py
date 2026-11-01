"""Virtual-environment isolation for benchmark execution.

Each instance runs in its own venv so a benchmark cannot mutate the host Python
environment. This is process isolation, not a VM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class VenvError(RuntimeError):
    pass


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def create_venv(base: Path, *, timeout: float = 120.0) -> Path:
    venv_dir = base / "venv"
    if venv_python(venv_dir).exists():
        return venv_dir
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VenvError(f"could not create venv: {exc}") from exc
    if proc.returncode != 0 or not venv_python(venv_dir).exists():
        raise VenvError(f"venv creation failed: {(proc.stderr or proc.stdout)[-400:]}")
    return venv_dir


def run_in_venv(
    venv_dir: Path,
    script: Path,
    *,
    cwd: Path | None = None,
    timeout: float = 60.0,
    env: dict | None = None,
) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(
            [str(venv_python(venv_dir)), str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(cwd or script.parent),
            env={**os.environ, **(env or {})},
            creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s", "duration": timeout, "output": ""}
    output = (proc.stdout or "") + (proc.stderr or "")
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "output": output[-4000:],
        "duration": round(time.time() - started, 3),
    }
