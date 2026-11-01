"""Disk-backed INT4 weight store — "HDD offload, RAM as buffer".

Why this exists: a 4B checkpoint is 18.0 GiB in fp32, but this class of machine has
~6 GiB available. Packing to INT4 removes most of the *value* size, yet the fp32
embeddings alone are 2.4 GiB and the transient fp32 tensor during loading is another
2.4 GiB. Holding all of that in anonymous RAM is what breaks the run.

The store writes the quantized model to disk once:

* ``weights.u8``  packed 4-bit nibbles for every target linear,
* ``scales.f32``  one fp32 scale per 64-value group,
* ``rest.f32``    everything that is not quantized (embeddings, norms, lm_head),
* ``meta.json``   byte offsets, shapes, model file identity, skipped tensors.

Loading memory-maps those three files. The pages are file-backed, so Windows can evict
them under pressure instead of swapping or killing the process, and the file cache acts
as the "RAM buffer": the hot layers stay resident, the cold ones are read again. Only the
LoRA parameters, one dequantized layer and the activations occupy anonymous memory.

Honest limits: the first pass over the weights reads from disk (page faults), so it is
slower than the in-RAM path; the store costs disk space equal to the quantized weights
plus the fp32 leftovers; and it is a memory trade, not a speed feature.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import time
import warnings
from pathlib import Path
from typing import Callable

STORE_VERSION = 1
STORE_DIRNAME = "int4-store"
WRAPPER_SEGMENTS = ("language_model", "text_model", "text_transformer")


def _torch():
    import torch

    return torch


def store_identity(model_path: Path) -> dict:
    """Identity of the checkpoint: file names, sizes and mtimes (cheap, no hashing)."""
    files = []
    for path in sorted(Path(model_path).glob("*.safetensors")):
        stat = path.stat()
        files.append({"name": path.name, "size": stat.st_size, "mtime": int(stat.st_mtime)})
    return {"model": str(model_path), "files": files}


def _same_identity(a: dict, b: dict) -> bool:
    return a.get("files") == b.get("files")


def _build_meta_model(model_path: Path):
    """Meta shell + its state-dict keys (see ``int4.build_meta_model``)."""
    from .int4 import build_meta_model

    model, config = build_meta_model(model_path)
    return model, config, set(model.state_dict().keys())


def _resolve_key(key: str, available: set[str]) -> str | None:
    if key in available:
        return key
    parts = key.split(".")
    for index, part in enumerate(parts[:-1]):
        if index == 0 or part not in WRAPPER_SEGMENTS:
            continue
        candidate = ".".join(parts[:index] + parts[index + 1 :])
        if candidate in available:
            return candidate
    return None


def _skip_group(key: str) -> str:
    parts = key.split(".")
    return ".".join(parts[:2]) if len(parts) > 2 else parts[0]


def _write_f32(fileobj, tensor, rows: int = 2048) -> int:
    """Write a tensor to ``fileobj`` as little-endian float32, a few rows at a time.

    Chunking matters: a 248k-vocabulary embedding is 2.4 GiB in fp32, and converting it in
    one step would need the bf16 tensor *and* the fp32 copy *and* the byte string at once.
    """
    torch = _torch()
    written = 0
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    if tensor.dim() == 1:
        tensor = tensor.reshape(-1, 1)
        rows = max(rows, 8192)
    for start in range(0, tensor.shape[0], rows):
        block = tensor[start : start + rows].to(torch.float32).contiguous()
        payload = block.numpy().tobytes()
        fileobj.write(payload)
        written += len(payload)
        del block, payload
    return written


def _detect_targets(model, requested: list[str] | None) -> set[str]:
    from .int4 import DEFAULT_TARGETS

    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = set(requested or DEFAULT_TARGETS)
    if targets & module_names:
        return targets
    for candidate in (DEFAULT_TARGETS, ["c_attn", "c_proj"], ["query", "key", "value", "dense"]):
        if all(name in module_names for name in candidate):
            return set(candidate)
    return targets


def build_store(
    model_path: str | Path,
    store_dir: str | Path,
    *,
    targets: list[str] | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel=None,
    force: bool = False,
) -> dict:
    """Quantize ``model_path`` into a memory-mappable store. Reuses a matching store."""
    torch = _torch()
    from safetensors import safe_open

    from .int4 import DEFAULT_TARGETS, GROUP_SIZE, quantize_int4, _shard_paths

    path = Path(model_path).expanduser()
    out = Path(store_dir).expanduser()
    meta_path = out / "meta.json"
    identity = store_identity(path)
    requested = sorted(targets or DEFAULT_TARGETS)
    if meta_path.exists() and not force:
        try:
            existing = json.loads(meta_path.read_text("utf-8"))
            if (
                existing.get("version") == STORE_VERSION
                and _same_identity(existing.get("identity"), identity)
                and existing.get("requested_targets") == requested
            ):
                if log:
                    log(f"int4 store reused: {out}")
                return {**existing, "reused": True, "store_dir": str(out)}
        except (OSError, json.JSONDecodeError):
            pass

    model, _config, available = _build_meta_model(path)
    target_suffixes = _detect_targets(model, targets)
    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / "weights.u8"
    scales_path = out / "scales.f32"
    rest_path = out / "rest.f32"

    tensors: dict[str, dict] = {}
    rest: dict[str, dict] = {}
    skipped: dict[str, int] = {}
    converted = 0
    assigned = 0
    started = time.time()
    weight_offset = scale_offset = rest_offset = 0

    with open(weights_path, "wb") as wf, open(scales_path, "wb") as sf, open(rest_path, "wb") as rf:
        for shard in _shard_paths(path):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("int4 store build cancelled")
            with safe_open(str(shard), framework="pt") as handle:
                keys = list(handle.keys())
                for key in keys:
                    resolved = _resolve_key(key, available)
                    if resolved is None:
                        group = _skip_group(key)
                        skipped[group] = skipped.get(group, 0) + 1
                        continue
                    if resolved.endswith(".weight"):
                        module_path = resolved[: -len(".weight")]
                        if module_path.rsplit(".", 1)[-1] in target_suffixes and not module_path.endswith("lm_head"):
                            parent_path, _, attribute = module_path.rpartition(".")
                            parent = model.get_submodule(parent_path) if parent_path else model
                            module = getattr(parent, attribute)
                            weight = handle.get_tensor(key)
                            out_features = int(
                                getattr(module, "out_features", None)
                                or getattr(module, "nf", None)
                                or weight.shape[0]
                            )
                            in_features = int(
                                getattr(module, "in_features", None)
                                or getattr(module, "nx", None)
                                or weight.shape[1]
                            )
                            if weight.shape != (out_features, in_features):
                                if weight.shape == (in_features, out_features):
                                    weight = weight.t().contiguous()
                                else:
                                    skipped[_skip_group(key)] = skipped.get(_skip_group(key), 0) + 1
                                    del weight
                                    continue
                            packed, scales, shape = quantize_int4(weight)
                            del weight
                            packed_bytes = packed.numpy().tobytes()
                            scales_bytes = scales.numpy().tobytes()
                            wf.write(packed_bytes)
                            sf.write(scales_bytes)
                            tensors[module_path] = {
                                "kind": "int4",
                                "out_features": shape[0],
                                "in_features": shape[1],
                                "padded": shape[2],
                                "packed_offset": weight_offset,
                                "packed_bytes": len(packed_bytes),
                                "scales_offset": scale_offset,
                                "scales_bytes": len(scales_bytes),
                                "scales_per_row": int(scales.shape[-1]),
                            }
                            weight_offset += len(packed_bytes)
                            scale_offset += len(scales_bytes)
                            converted += 1
                            del packed, scales, packed_bytes, scales_bytes
                            continue
                    tensor = handle.get_tensor(key).contiguous()
                    written = _write_f32(rf, tensor)
                    rest[resolved] = {
                        "shape": list(tensor.shape),
                        "offset": rest_offset,
                        "bytes": written,
                    }
                    rest_offset += written
                    assigned += 1
                    del tensor
            if progress:
                progress(10 + 40 * converted / max(1, len(tensors) + 1), f"store: {converted} quantized tensors")

    meta = {
        "version": STORE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": identity,
        "targets": sorted(target_suffixes),
        "requested_targets": requested,
        "group_size": GROUP_SIZE,
        "tensors": tensors,
        "rest": rest,
        "skipped": skipped,
        "bytes": {
            "weights_u8": weight_offset,
            "scales_f32": scale_offset,
            "rest_f32": rest_offset,
            "total": weight_offset + scale_offset + rest_offset,
        },
        "counts": {"quantized": converted, "assigned": assigned, "skipped": sum(skipped.values())},
        "duration_seconds": round(time.time() - started, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2), "utf-8")
    meta["reused"] = False
    meta["store_dir"] = str(out)
    if log:
        log(
            "int4 store built: "
            f"{converted} quantized, {assigned} fp32 tensors, "
            f"{(weight_offset + scale_offset + rest_offset) / 1024**3:.2f} GiB on disk"
            + (f", skipped {skipped}" if skipped else "")
        )
    return meta


class Int4Store:
    """Memory-mapped view of a store built by :func:`build_store`."""

    def __init__(self, store_dir: str | Path) -> None:
        torch = _torch()
        self.dir = Path(store_dir).expanduser()
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no int4 store at {self.dir}")
        self.meta = json.loads(meta_path.read_text("utf-8"))
        self._handles: list[tuple] = []
        self._weights = self._map("weights.u8", torch.uint8)
        self._scales = self._map("scales.f32", torch.float32)
        self._rest = self._map("rest.f32", torch.float32)

    def _map(self, name: str, dtype):
        torch = _torch()
        path = self.dir / name
        if not path.exists() or path.stat().st_size == 0:
            return torch.empty(0, dtype=dtype)
        handle = open(path, "rb")
        # ACCESS_READ (not ACCESS_COPY): a copy-on-write section reserves Windows commit
        # charge for the whole mapping, which defeats the point of the store. The weights are
        # frozen, so a read-only view is also the safer contract: an accidental in-place write
        # fails loudly instead of silently diverging from the file.
        try:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
        self._handles.append((handle, mapped))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-writable.*")
            return torch.frombuffer(mapped, dtype=dtype)

    def packed(self, module_path: str):
        info = self.meta["tensors"][module_path]
        start = info["packed_offset"]
        view = self._weights[start : start + info["packed_bytes"]]
        return view.view(info["out_features"], info["padded"] // 2)

    def scales(self, module_path: str):
        info = self.meta["tensors"][module_path]
        start = info["scales_offset"] // 4
        count = info["scales_bytes"] // 4
        view = self._scales[start : start + count]
        return view.view(info["out_features"], info["scales_per_row"])

    def tensor(self, key: str):
        info = self.meta["rest"][key]
        start = info["offset"] // 4
        count = info["bytes"] // 4
        view = self._rest[start : start + count]
        return view.view(*info["shape"])

    def close(self) -> None:
        for handle, mapped in self._handles:
            try:
                mapped.close()
            except (BufferError, ValueError):
                pass
            try:
                handle.close()
            except OSError:
                pass
        self._handles.clear()

    @property
    def disk_bytes(self) -> int:
        return int(self.meta.get("bytes", {}).get("total", 0))


def store_fingerprint(store_dir: str | Path) -> str:
    meta = Path(store_dir).expanduser() / "meta.json"
    if not meta.exists():
        return ""
    return hashlib.sha256(meta.read_bytes()).hexdigest()[:16]


def attach_store(model, store: Int4Store, *, r: int, alpha: int, dropout: float = 0.0) -> dict:
    """Load fp32 leftovers from the store and wrap target linears with INT4 + LoRA."""
    torch = _torch()
    from accelerate.utils import set_module_tensor_to_device

    from .int4 import Int4Linear, LoRALinear

    assigned = 0
    for key in store.meta["rest"]:
        set_module_tensor_to_device(model, key, "cpu", value=store.tensor(key), dtype=torch.float32)
        assigned += 1

    converted = 0
    for module_path in store.meta["tensors"]:
        parent_path, _, attribute = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        current = getattr(parent, attribute)
        bias = getattr(current, "bias", None)
        info = store.meta["tensors"][module_path]
        int4 = Int4Linear.from_store(
            store.packed(module_path),
            store.scales(module_path),
            (info["out_features"], info["in_features"], info["padded"]),
            bias=bias,
        )
        setattr(parent, attribute, LoRALinear.build(int4, r, alpha, dropout))
        converted += 1

    try:
        model.tie_weights()
    except Exception:  # noqa: BLE001 - best effort for non-standard configs
        pass

    leftovers = [name for name, parameter in model.named_parameters() if parameter.device.type == "meta"]
    leftovers += [name for name, buffer in model.named_buffers() if buffer.device.type == "meta"]
    if leftovers:
        raise RuntimeError(
            f"INT4 store load left {len(leftovers)} tensors on meta, e.g. {leftovers[:3]}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)

    try:
        model.config.use_cache = False
    except AttributeError:
        pass

    model._int4_store = store  # keep the mappings alive for the lifetime of the model
    resident = 0
    for parameter in model.parameters():
        resident += parameter.numel() * parameter.element_size()
    for buffer in model.buffers():
        if getattr(buffer, "is_meta", False) or buffer.device.type == "meta":
            continue
        resident += buffer.numel() * buffer.element_size()
    return {
        "quantized_layers": converted,
        "assigned_tensors": assigned,
        "resident_bytes_estimate": resident,
        "skipped": store.meta.get("skipped") or {},
        "store_dir": str(store.dir),
        "store_disk_bytes": store.disk_bytes,
        "store_version": store.meta.get("version"),
        "fingerprint": store_fingerprint(store.dir),
        "offload": True,
    }


def load_store_model(
    model_path: str | Path,
    store_dir: str | Path,
    *,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    log: Callable[[str], None] | None = None,
):
    """Build the meta model, attach the store, return ``(model, report)``."""
    path = Path(model_path).expanduser()
    model, _config, _available = _build_meta_model(path)
    store = build_store(path, store_dir, log=log)
    loaded = Int4Store(store["store_dir"])
    report = attach_store(model, loaded, r=r, alpha=alpha, dropout=dropout)
    report["reused"] = bool(store.get("reused"))
    report["converted_layers"] = report["quantized_layers"]
    report["targets"] = list(store.get("targets") or [])
    report["store_counts"] = store.get("counts") or {}
    return model, report


def store_size_bytes(store_dir: str | Path) -> int:
    base = Path(store_dir).expanduser()
    total = 0
    for name in ("weights.u8", "scales.f32", "rest.f32"):
        path = base / name
        if path.exists():
            total += path.stat().st_size
    return total


def clear_store(store_dir: str | Path) -> None:
    base = Path(store_dir).expanduser()
    for name in ("weights.u8", "scales.f32", "rest.f32", "meta.json"):
        path = base / name
        if path.exists():
            try:
                os.remove(path)
            except OSError:
                pass
