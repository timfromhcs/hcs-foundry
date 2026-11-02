"""Merge an INT4 LoRA adapter back into a plain fp16 checkpoint.

The INT4 trainer freezes 4-bit-packed base weights and trains a LoRA delta on top. Until now
that adapter was a dead end: the file is a custom ``lora_int4.pt`` and nothing could consume it.
This module reconstructs the weights the forward pass actually used::

    W_merged = dequantize_int4(packed, scales) + (lora_B @ lora_A) * (alpha / r)

Non-target tensors (including a vision tower or an MTP head that the training store skipped) are
taken from the original checkpoint, so the merged directory is a complete, loadable HF model and
can go through the existing GGUF export and quantization path.

Memory: one layer is materialised at a time (fp32 for the merge, fp16 on the way out) and tensors
are flushed into shards as they accumulate, so peak memory is roughly one shard plus one layer.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable

COPY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

SHARD_PREFIX = "model"


def _torch():
    import torch

    return torch


def _load_adapter(adapter_dir: Path) -> dict:
    torch = _torch()
    path = adapter_dir / "lora_int4.pt"
    if not path.exists():
        raise ValueError(f"no lora_int4.pt in {adapter_dir}")
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    lora = payload.get("lora") or {}
    if not lora:
        raise ValueError(f"{path} contains no lora tensors")
    return {
        "lora": lora,
        "r": int(payload.get("lora_r") or 8),
        "alpha": int(payload.get("lora_alpha") or 16),
        "targets": list(payload.get("target_modules") or []),
        "converted_layers": int(payload.get("converted_layers") or 0),
    }


def _delta_for(module_path: str, lora: dict, r: int, alpha: int):
    torch = _torch()
    a = lora.get(f"{module_path}.lora_A")
    b = lora.get(f"{module_path}.lora_B")
    if a is None or b is None:
        return None
    scaling = alpha / max(1, r)
    return (b.to(torch.float32) @ a.to(torch.float32)) * scaling


def _merged_weight(store, module_path: str, adapter: dict):
    torch = _torch()
    from .int4 import dequantize_int4

    info = store.meta["tensors"][module_path]
    shape = (info["out_features"], info["in_features"], info["padded"])
    weight = dequantize_int4(store.packed(module_path), store.scales(module_path), shape, torch.float32)
    delta = _delta_for(module_path, adapter["lora"], adapter["r"], adapter["alpha"])
    if delta is not None:
        if delta.shape != weight.shape:
            raise ValueError(
                f"adapter delta for {module_path} has shape {tuple(delta.shape)}, "
                f"base has {tuple(weight.shape)}"
            )
        weight = weight + delta
        del delta
    return weight


def merge_int4_adapter(
    store_dir: str | Path,
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    base_model_dir: str | Path | None = None,
    dtype: str = "float16",
    shard_bytes: int = 1_500_000_000,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel=None,
) -> dict:
    """Write a merged checkpoint (safetensors shards + config/tokenizer) to ``out_dir``."""
    torch = _torch()
    from safetensors import safe_open
    from safetensors.torch import save_file

    from .int4 import _resolve_key, _shard_paths
    from .offload import Int4Store

    adapter = _load_adapter(Path(adapter_dir).expanduser())
    store = Int4Store(store_dir)
    expected = int(store.meta.get("counts", {}).get("quantized") or 0)
    if expected and adapter["converted_layers"] and adapter["converted_layers"] != expected:
        raise ValueError(
            f"adapter was trained on {adapter['converted_layers']} quantized layers but the store "
            f"has {expected}; refusing to merge mismatched artifacts"
        )
    if adapter["targets"]:
        missing = sorted(set(adapter["targets"]) - set(store.meta.get("targets") or []))
        if missing:
            raise ValueError(f"store was built without these adapter targets: {missing}")

    base = Path(base_model_dir).expanduser() if base_model_dir else Path(store.meta["identity"]["model"])
    if not base.is_dir():
        raise ValueError(f"base model directory not found: {base}")

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(f"{SHARD_PREFIX}-*.safetensors"):
        stale.unlink()

    available = set(store.meta["rest"].keys())
    available |= {f"{module}.weight" for module in store.meta["tensors"]}
    available |= {"lm_head.weight"}

    target_dtype = getattr(torch, dtype)
    started = time.time()
    shards: list[str] = []
    weight_map: dict[str, str] = {}
    buffer: dict[str, "torch.Tensor"] = {}
    buffered = 0
    quantized_written = 0
    tensors_written = 0

    def flush() -> None:
        nonlocal buffer, buffered
        if not buffer:
            return
        name = f"{SHARD_PREFIX}-{len(shards) + 1:05d}.safetensors"
        save_file(buffer, str(out / name), metadata={"format": "pt"})
        for key in buffer:
            weight_map[key] = name
        shards.append(name)
        if log:
            log(f"  wrote {name} ({sum(t.numel() for t in buffer.values()) / 1e6:.1f} M params)")
        buffer = {}
        buffered = 0

    def add(key: str, tensor) -> None:
        nonlocal buffered, tensors_written
        buffer[key] = tensor.to(target_dtype).contiguous()
        buffered += buffer[key].numel() * buffer[key].element_size()
        tensors_written += 1
        if buffered >= shard_bytes:
            flush()

    shard_files = _shard_paths(base)
    for shard_index, shard in enumerate(shard_files):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("merge cancelled")
        with safe_open(str(shard), framework="pt") as handle:
            keys = list(handle.keys())
            for key in keys:
                resolved = _resolve_key(key, available)
                module_path = resolved[: -len(".weight")] if resolved and resolved.endswith(".weight") else None
                if resolved and module_path in store.meta["tensors"]:
                    add(resolved, _merged_weight(store, module_path, adapter))
                    quantized_written += 1
                    continue
                add(resolved or key, handle.get_tensor(key))
        if progress:
            progress(5 + 85 * (shard_index + 1) / max(1, len(shard_files)), f"merged {shard_index + 1}/{len(shard_files)} shards")
    flush()

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum((out / s).stat().st_size for s in shards)}, "weight_map": weight_map}, indent=2),
        "utf-8",
    )
    copied = []
    for name in COPY_FILES:
        source = base / name
        if source.exists():
            shutil.copy(source, out / name)
            copied.append(name)
    if (base / "config.json").exists():
        config = json.loads((base / "config.json").read_text("utf-8"))
        config["torch_dtype"] = "float16"
        config["dtype"] = "float16"
        (out / "config.json").write_text(json.dumps(config, indent=2), "utf-8")
    else:
        raise ValueError(f"base has no config.json: {base}")

    size = sum(p.stat().st_size for p in out.glob("*") if p.is_file())
    store.close()
    if progress:
        progress(100, "merge complete")
    return {
        "output": str(out),
        "base_model": str(base),
        "store_dir": str(store.dir),
        "adapter_dir": str(Path(adapter_dir).expanduser()),
        "dtype": dtype,
        "shards": shards,
        "tensors": tensors_written,
        "quantized_weights_merged": quantized_written,
        "lora_rank": adapter["r"],
        "lora_alpha": adapter["alpha"],
        "size_bytes": size,
        "duration_seconds": round(time.time() - started, 1),
        "copied_files": copied,
    }
