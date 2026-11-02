"""INT4 (QLoRA-style) training without bitsandbytes.

The frozen base weights are packed to 4 bits with group-wise symmetric absmax
quantization and dequantized per layer inside the forward pass. Only the LoRA
adapters are trained and stored in fp32. This works on CPU and needs no CUDA.

Honest limits:

* the non-streaming loader materialises the model in fp32 once, so the peak memory
  during *loading* is the fp32 size; ``streaming=True`` avoids that peak by quantizing
  while reading the shards (measured peak RSS is written to the manifest),
* dequantization costs throughput; it is a memory trade, not a speed feature.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psutil

from ..db import Database, new_id, utc_now
from .memory import PeakRSS
from .sft import SFTConfig, build_eval_texts, build_examples, estimate_resources

GROUP_SIZE = 64
# Dense attention + MLP, plus the hybrid/linear-attention projections used by
# Qwen3-Next/Qwen3.5-style models (in_proj_qkv, in_proj_z, out_proj). Names that do not
# exist in a given architecture are simply ignored.
DEFAULT_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
]


def _torch():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"PyTorch is not importable: {exc}") from exc
    return torch, nn, F


def quantize_int4(weight):
    """Pack a 2-D float weight into uint8 nibbles + per-group scales."""
    torch, _, _ = _torch()
    w = weight.detach().float()
    out_features, in_features = w.shape
    group = GROUP_SIZE
    padded = int(math.ceil(in_features / (group * 2)) * group * 2)
    if padded != in_features:
        w = torch.nn.functional.pad(w, (0, padded - in_features))
    reshaped = w.reshape(out_features, padded // group, group)
    scale = reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.round(reshaped / scale).clamp(-8, 7).to(torch.int8)
    q = q.reshape(out_features, padded)
    low = (q[:, 0::2] & 0x0F).to(torch.uint8)
    high = (q[:, 1::2] & 0x0F).to(torch.uint8)
    packed = low | (high << 4)
    scales = scale.squeeze(-1).contiguous()
    return packed.contiguous(), scales, (out_features, in_features, padded)


def dequantize_int4(packed, scales, shape, dtype=None):
    torch, _, _ = _torch()
    out_features, in_features, padded = shape
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    interleaved = torch.stack((low, high), dim=-1).reshape(out_features, padded)
    group = GROUP_SIZE
    dequant = interleaved.reshape(out_features, padded // group, group).float() * scales.unsqueeze(-1)
    dequant = dequant.reshape(out_features, padded)[:, :in_features]
    return dequant.to(dtype) if dtype is not None else dequant


def int4_size_bytes(weight) -> dict:
    packed, scales, _ = quantize_int4(weight)
    return {
        "fp32_bytes": int(weight.numel() * 4),
        "int4_bytes": int(packed.numel() + scales.numel() * 4),
    }


_INT4_MATMUL = None
DEFAULT_BLOCK = 256


def int4_block_size() -> int:
    """Rows of the output matrix dequantized at once (``FOUNDRY_INT4_BLOCK``)."""
    try:
        value = int(os.environ.get("FOUNDRY_INT4_BLOCK", DEFAULT_BLOCK))
    except (TypeError, ValueError):
        return DEFAULT_BLOCK
    return max(1, value)


def blocked_linear(x, packed, scales, shape, bias=None, block: int | None = None, cache=None, module_path: str | None = None):
    """``x @ dequant(packed)^T + bias`` one row block at a time.

    Dequantizing the full matrix first costs ``out_features * in_features * 4`` bytes per layer
    (94 MB for a 9216x2560 projection) and drives the allocator into multi-gigabyte peaks over a
    long run. Blocking keeps one block resident and writes straight into the output buffer.

    With a ``cache`` the blocks come from the tiered store (RAM → disk arena → dequantize), and
    the next block is prefetched while this one is multiplied.
    """
    torch, _, F = _torch()
    out_features, in_features, padded = shape
    step = block or int4_block_size()
    if step >= out_features:
        if cache is not None and module_path:
            weight = cache.get(module_path, 0, out_features)
            return F.linear(x, weight.to(x.dtype), bias)
        weight = dequantize_int4(packed, scales, shape, x.dtype)
        return F.linear(x, weight, bias)
    out = None
    for start in range(0, out_features, step):
        stop = min(start + step, out_features)
        if cache is not None and module_path:
            w = cache.get(module_path, start, stop).to(x.dtype)
            if stop < out_features:
                cache.prefetch([_block_key(module_path, stop, min(stop + step, out_features))])
        else:
            w = dequantize_int4(packed[start:stop], scales[start:stop], (stop - start, in_features, padded), x.dtype)
        part = F.linear(x, w)
        if out is None:
            out = torch.empty(*part.shape[:-1], out_features, dtype=part.dtype, device=part.device)
        out[..., start:stop] = part
        del w, part
    if bias is not None:
        out = out + bias
    return out


def _block_key(module_path: str, start: int, stop: int):
    from .vram import BlockKey

    return BlockKey(module_path, start, stop)


def blocked_grad_input(grad_output, packed, scales, shape, block: int | None = None, cache=None, module_path: str | None = None):
    """``grad_output @ dequant(packed)`` accumulated block by block into one buffer."""
    torch, _, _ = _torch()
    out_features, in_features, padded = shape
    step = block or int4_block_size()
    if step >= out_features:
        if cache is not None and module_path:
            return grad_output @ cache.get(module_path, 0, out_features).to(grad_output.dtype)
        weight = dequantize_int4(packed, scales, shape, grad_output.dtype)
        return grad_output @ weight
    grad = None
    for start in range(0, out_features, step):
        stop = min(start + step, out_features)
        if cache is not None and module_path:
            w = cache.get(module_path, start, stop).to(grad_output.dtype)
        else:
            w = dequantize_int4(packed[start:stop], scales[start:stop], (stop - start, in_features, padded), grad_output.dtype)
        part = grad_output[..., start:stop] @ w
        if grad is None:
            grad = part
        else:
            grad.add_(part)
        del w, part
    return grad


def _int4_matmul():
    """Build (once) the autograd function that recomputes the dequantization in backward.

    Dequantizing inside a normal forward pass would retain every fp32 weight matrix until the
    backward pass finishes — 14 GiB for a 4B model. This function saves only the packed INT4
    tensors and dequantizes again during backward, so peak memory stays at roughly one block
    while the base weights remain frozen.
    """
    global _INT4_MATMUL
    if _INT4_MATMUL is not None:
        return _INT4_MATMUL
    torch, _, _ = _torch()

    class _Int4Matmul(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, packed, scales, shape, bias, block, cache, module_path):
            ctx.save_for_backward(x, packed, scales)
            ctx.shape = shape
            ctx.block = block
            ctx.cache = cache
            ctx.module_path = module_path
            return blocked_linear(x, packed, scales, shape, bias, block, cache=cache, module_path=module_path)

        @staticmethod
        def backward(ctx, grad_output):
            x, packed, scales = ctx.saved_tensors
            grad = blocked_grad_input(
                grad_output, packed, scales, ctx.shape, ctx.block, cache=ctx.cache, module_path=ctx.module_path
            )
            return grad, None, None, None, None, None, None, None

    _INT4_MATMUL = _Int4Matmul
    return _INT4_MATMUL


class Int4Linear:
    """A frozen 4-bit linear layer.

    ``packed`` and ``scales`` may be ordinary in-RAM tensors (quantized at load time) or
    read-only views onto a memory-mapped INT4 store; both share the same forward pass.
    """

    @staticmethod
    def from_store(packed, scales, shape, bias=None, cache=None, module_path: str | None = None):
        torch, nn, F = _torch()

        class _Int4Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("packed", packed)
                self.register_buffer("scales", scales)
                self.shape = shape
                self.in_features = int(shape[1])
                self.out_features = int(shape[0])
                self.cache = cache
                self.module_path = module_path
                if bias is not None:
                    self.register_buffer("bias", bias.detach().clone())
                else:
                    self.bias = None

            def forward(self, x):
                if torch.is_grad_enabled() and x.requires_grad:
                    return _int4_matmul().apply(
                        x,
                        self.packed,
                        self.scales,
                        self.shape,
                        self.bias,
                        int4_block_size(),
                        self.cache,
                        self.module_path,
                    )
                return blocked_linear(
                    x, self.packed, self.scales, self.shape, self.bias, cache=self.cache, module_path=self.module_path
                )

            def extra_repr(self):
                return f"in={self.in_features}, out={self.out_features}, int4"

        return _Int4Linear()

    @staticmethod
    def from_parts(weight, bias, out_features: int, in_features: int):
        w = weight.detach()
        if w.shape[0] != out_features:
            w = w.t().contiguous()
        packed, scales, shape = quantize_int4(w)
        return Int4Linear.from_store(packed, scales, shape, bias=bias)

    @staticmethod
    def build(linear, group: int = GROUP_SIZE):
        weight = linear.weight.detach()
        out_features = int(getattr(linear, "out_features", None) or getattr(linear, "nf", weight.shape[1]))
        in_features = int(getattr(linear, "in_features", None) or weight.shape[0])
        return Int4Linear.from_parts(weight, linear.bias, out_features, in_features)


class LoRALinear:
    @staticmethod
    def build(base, r: int, alpha: int, dropout: float = 0.0):
        torch, nn, _ = _torch()

        class _LoRALinear(nn.Module):
            def __init__(self, wrapped, rank: int, lora_alpha: int):
                super().__init__()
                self.base = wrapped
                for parameter in self.base.parameters():
                    parameter.requires_grad_(False)
                self.lora_A = nn.Parameter(torch.randn(rank, wrapped.in_features) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(wrapped.out_features, rank))
                self.scaling = lora_alpha / max(1, rank)
                self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            def forward(self, x):
                delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
                return self.base(x) + delta * self.scaling

        return _LoRALinear(base, r, alpha)


def convert_model(model, *, targets: list[str] | None = None, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> dict:
    torch, nn, _ = _torch()
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:  # noqa: BLE001 - older/newer transformers
        Conv1D = ()  # type: ignore[assignment]
    targets = set(targets or DEFAULT_TARGETS)
    converted = 0
    skipped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, (nn.Linear, Conv1D)):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in targets or suffix == "lm_head":
            skipped += 1
            continue
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        int4 = Int4Linear.build(module)
        setattr(parent, attribute, LoRALinear.build(int4, r, alpha, dropout))
        converted += 1
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
    return {"converted_layers": converted, "skipped_linear_layers": skipped}


def _tokenize(tokenizer, text: str, max_len: int):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    return encoded["input_ids"]


def _shard_paths(model_path: Path) -> list[Path]:
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text("utf-8"))
        names = sorted(set((data.get("weight_map") or {}).values()))
        return [model_path / name for name in names if (model_path / name).exists()]
    single = model_path / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise ValueError("no safetensors shards found")
    return shards


# Wrapper segments that multimodal / nested checkpoints put around the language model.
WRAPPER_SEGMENTS = ("language_model", "text_model", "text_transformer", "language_model.model")


def _resolve_key(key: str, available: set[str]) -> str | None:
    """Map a checkpoint tensor name onto a key of the instantiated model.

    Vision-language checkpoints store the language model under ``model.language_model.*``
    while a text-only ``...ForCausalLM`` instantiates ``model.*``. Dropping the wrapper
    segment lets the same loader handle both layouts. Anything that still does not match
    (vision tower, multi-token-prediction head) is reported as skipped, and the hard
    meta-device check afterwards guarantees that no needed weight is silently missing.
    """
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


def _skip_prefix(key: str) -> str:
    parts = key.split(".")
    if len(parts) > 2:
        return ".".join(parts[:2])
    return parts[0]


def build_meta_model(model_path: str | Path, dtype=None):
    """Instantiate the architecture shell on the ``meta`` device.

    ``dtype`` is forced on the config (and its nested text/vision configs) because some
    checkpoints declare bf16; without this, accelerate would convert every stored tensor to
    bf16 on assignment, which both copies the mmap'd data into anonymous RAM and changes the
    numerics of the quantized path.
    """
    torch, _, _ = _torch()
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = dtype or torch.float32
    config = AutoConfig.from_pretrained(str(model_path))
    for candidate in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if candidate is None:
            continue
        try:
            candidate.dtype = dtype
        except (AttributeError, TypeError, ValueError):
            pass
    with init_empty_weights():
        try:
            model = AutoModelForCausalLM.from_config(config, dtype=dtype)
        except TypeError:  # older transformers only knows torch_dtype
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    return model, config


def load_int4_streaming(
    model_path: str | Path,
    *,
    targets: list[str] | None = None,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    dtype=None,
):
    """Load a model shard by shard and quantize target linears while reading.

    The model is created on the meta device, so the fp32 weights are never all
    materialised at once.
    """
    torch, nn, _ = _torch()
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    path = Path(model_path).expanduser()
    targets = set(targets or DEFAULT_TARGETS)
    dtype = dtype or torch.float32
    model, config = build_meta_model(path, dtype=dtype)

    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    if not (targets & module_names):
        for candidate in (DEFAULT_TARGETS, ["c_attn", "c_proj"], ["query", "key", "value", "dense"]):
            if all(name in module_names for name in candidate):
                targets = set(candidate)
                break

    converted = 0
    assigned = 0
    skipped: dict[str, int] = {}
    available = set(model.state_dict().keys())
    for shard in _shard_paths(path):
        with safe_open(str(shard), framework="pt") as handle:
            keys = list(handle.keys())
            for key in keys:
                resolved = _resolve_key(key, available)
                if resolved is None:
                    group = _skip_prefix(key)
                    skipped[group] = skipped.get(group, 0) + 1
                    continue
                if resolved.endswith(".weight"):
                    module_path = resolved[: -len(".weight")]
                    suffix = module_path.rsplit(".", 1)[-1]
                    if suffix in targets and suffix != "lm_head":
                        parent_path, _, attribute = module_path.rpartition(".")
                        parent = model.get_submodule(parent_path) if parent_path else model
                        target_module = getattr(parent, attribute)
                        bias_key = module_path + ".bias"
                        bias = handle.get_tensor(bias_key) if bias_key in keys else None
                        weight = handle.get_tensor(key)
                        out_features = int(
                            getattr(target_module, "out_features", None)
                            or getattr(target_module, "nf", weight.shape[1])
                        )
                        in_features = int(
                            getattr(target_module, "in_features", None)
                            or getattr(target_module, "nx", weight.shape[0])
                        )
                        int4 = Int4Linear.from_parts(weight, bias, out_features, in_features)
                        setattr(parent, attribute, LoRALinear.build(int4, r, alpha, dropout))
                        del weight
                        converted += 1
                        continue
                if resolved.endswith(".bias"):
                    module_path = resolved[: -len(".bias")]
                    if module_path.rsplit(".", 1)[-1] in targets:
                        continue  # consumed with the weight
                set_module_tensor_to_device(model, resolved, "cpu", value=handle.get_tensor(key), dtype=dtype)
                assigned += 1

    try:
        model.tie_weights()
    except Exception:  # noqa: BLE001 - tying is best effort for non-standard configs
        pass

    leftovers = [name for name, parameter in model.named_parameters() if parameter.device.type == "meta"]
    leftovers += [name for name, buffer in model.named_buffers() if buffer.device.type == "meta"]
    if leftovers:
        raise RuntimeError(f"streaming INT4 load left {len(leftovers)} tensors on meta, e.g. {leftovers[:3]}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
    try:
        model.config.use_cache = False
    except AttributeError:
        pass
    return model, {
        "converted_layers": converted,
        "assigned_tensors": assigned,
        "targets": sorted(targets),
        "streaming": True,
        "skipped": skipped,
    }


def estimate_int4_resources(model_path: str | Path, config: SFTConfig) -> dict:
    """Memory estimate for the streaming path (INT4 weights + fp32 leftovers)."""
    path = Path(model_path).expanduser()
    weights = sum(f.stat().st_size for f in list(path.glob("*.safetensors")) + list(path.glob("*.bin")))
    model_config = {}
    config_path = path / "config.json"
    if config_path.exists():
        try:
            model_config = json.loads(config_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            model_config = {}
    text_config = model_config.get("text_config") if isinstance(model_config.get("text_config"), dict) else model_config
    hidden = int(text_config.get("hidden_size") or model_config.get("hidden_size") or model_config.get("n_embd") or 2048)
    layers = int(text_config.get("num_hidden_layers") or model_config.get("num_hidden_layers") or model_config.get("n_layer") or 24)
    vocab = int(text_config.get("vocab_size") or model_config.get("vocab_size") or 0)
    quantized = int(weights * 0.25)  # 4 bits + scales, roughly
    # Everything that stays fp32: the (usually tied) embedding matrix dominates, the rest
    # of the non-target tensors are approximated from the checkpoint size.
    embed = vocab * hidden * 4 if vocab else 0
    others = int(weights * 0.12)
    vision = 0
    vision_config = model_config.get("vision_config")
    if isinstance(vision_config, dict):
        depth = int(vision_config.get("depth") or 0)
        v_hidden = int(vision_config.get("hidden_size") or 0)
        v_inter = int(vision_config.get("intermediate_size") or 0)
        vision = depth * (4 * v_hidden**2 + 2 * v_hidden * v_inter) * 4
    leftovers = max(embed + others, 900 * 1024**2)
    activations = config.micro_batch * config.max_seq_len * hidden * layers * 2 * 4
    low = quantized + leftovers + vision + activations
    high = int(low * 1.35)
    available = psutil.virtual_memory().available
    return {
        "weights_bytes": weights,
        "quantized_bytes": quantized,
        "leftovers_bytes": leftovers,
        "vision_bytes": vision,
        "activation_bytes": activations,
        "estimated_low_bytes": int(low),
        "estimated_high_bytes": high,
        "available_bytes": available,
        "fits": high < available * 0.9,
        "streaming": True,
        "confidence": "low",
        "note": (
            "Streaming estimate: INT4 weights + fp32 embeddings/norms (and a vision tower "
            "if the checkpoint has one) + activations. When it does not fit, the disk-backed "
            "INT4 store keeps the same weights file-backed instead of resident."
        ),
    }


def train_int4(
    db: Database,
    config: SFTConfig,
    *,
    output_dir: Path,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancel=None,
    streaming: bool = False,
    offload: bool | None = None,
    store_dir: str | Path | None = None,
    vram_gb: float | None = None,
    vram_mode: str = "auto",
    vram_dir: str | Path | None = None,
) -> dict:
    """Train a LoRA adapter on INT4-packed base weights and record the measured peak RSS.

    ``offload=None`` picks the disk-backed INT4 store automatically when the RAM estimate
    does not fit; ``True`` forces it, ``False`` keeps everything in RAM. ``vram_gb`` sizes the
    tiered block cache (RAM + disk arena); ``vram_mode="off"`` disables it.
    """
    sampler = PeakRSS().start()
    holders: dict = {}
    try:
        return _train_int4(
            db,
            config,
            output_dir=output_dir,
            progress=progress,
            log=log,
            cancel=cancel,
            streaming=streaming,
            offload=offload,
            store_dir=store_dir,
            vram_gb=vram_gb,
            vram_mode=vram_mode,
            vram_dir=vram_dir,
            sampler=sampler,
            holders=holders,
        )
    finally:
        sampler.stop()
        cache = holders.get("vram")
        if cache is not None:
            try:
                cache.close()
            except Exception:  # noqa: BLE001 - cleanup must not mask the real error
                pass


def _train_int4(
    db: Database,
    config: SFTConfig,
    *,
    output_dir: Path,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancel=None,
    streaming: bool = False,
    offload: bool | None = None,
    store_dir: str | Path | None = None,
    vram_gb: float | None = None,
    vram_mode: str = "auto",
    vram_dir: str | Path | None = None,
    sampler: PeakRSS,
    holders: dict | None = None,
) -> dict:
    torch, nn, _ = _torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"training stack not importable: {exc}") from exc

    model_path = Path(config.model_path).expanduser()
    if not model_path.is_dir():
        raise ValueError(f"base model directory not found: {model_path}")

    resources = estimate_int4_resources(model_path, config) if streaming else estimate_resources(model_path, config)
    if offload is True and not streaming:
        raise ValueError("disk offload needs the streaming INT4 loader (streaming=True)")
    offload_mode = bool(streaming and (offload if offload is not None else not resources["fits"]))
    if not resources["fits"] and not offload_mode and not config.__dict__.get("force"):
        raise ValueError(
            "this configuration does not fit the safe memory envelope "
            f"({resources['estimated_high_bytes'] / 1024**3:.1f} GiB estimated, "
            f"{resources['available_bytes'] / 1024**3:.1f} GiB available)."
            + (
                " Enable streaming INT4 to avoid the fp32 load peak."
                if not streaming
                else " Use offload to keep the weights on disk instead of in RAM."
            )
        )

    if progress:
        progress(5, "Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    examples, dataset_meta = build_examples(db, config.dataset_id, config.max_seq_len, config.extra_texts)
    eval_texts = build_eval_texts(db, config.dataset_id, config.eval_samples) if config.eval_every else []

    if progress:
        progress(
            12,
            "Loading base model (streaming INT4, disk-backed)" if offload_mode
            else "Loading base model (streaming INT4)" if streaming
            else "Loading base model (fp32 peak)",
        )
    store_meta = None
    vram_meta = None
    vram_cache = None
    if offload_mode:
        from .offload import STORE_DIRNAME, Int4Store, load_store_model
        from .vram import DEFAULT_ARENA_GB, BlockCache, resolve_budget

        target_store = Path(store_dir).expanduser() if store_dir else output_dir / STORE_DIRNAME
        budget = resolve_budget(vram_gb, vram_mode)
        arena_dir = Path(vram_dir).expanduser() if vram_dir else output_dir / "vram"

        def cache_factory(store_view: Int4Store):
            """Called once, after the store exists, so the callback can slice its mmap."""
            nonlocal vram_cache
            from .int4 import dequantize_int4

            def _dequantize_block(module_path: str, start: int, stop: int):
                info = store_view.meta["tensors"][module_path]
                block_shape = (stop - start, info["in_features"], info["padded"])
                return dequantize_int4(
                    store_view.packed(module_path)[start:stop],
                    store_view.scales(module_path)[start:stop],
                    block_shape,
                    torch.float32,
                )

            vram_cache = BlockCache(
                dequantize=_dequantize_block,
                budget_bytes=budget,
                arena_dir=arena_dir,
                arena_bytes=int(DEFAULT_ARENA_GB * 1024**3),
                block_rows=int4_block_size(),
                # measured: prefetching the very next block mostly duplicates work
                prefetch=os.environ.get("FOUNDRY_VRAM_PREFETCH", "0") == "1",
                log=log,
            )
            if holders is not None:
                holders["vram"] = vram_cache
            if log:
                log(
                    f"vram: cache budget {budget / 1024**2:.0f} MB, arena "
                    f"{vram_cache.arena_limit / 1024**3:.1f} GiB at {arena_dir}"
                )
            return vram_cache

        model, report = load_store_model(
            model_path,
            target_store,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            log=log,
            cache_factory=cache_factory if budget > 0 else None,
        )
        if vram_cache is not None:
            vram_meta = {
                "mode": vram_mode,
                "budget_bytes": vram_cache.budget_bytes,
                "arena_bytes": vram_cache.arena_limit,
                "arena_dir": str(vram_cache.arena_dir),
            }
        store_meta = {
            "store_dir": report["store_dir"],
            "disk_bytes": report["store_disk_bytes"],
            "fingerprint": report["fingerprint"],
            "reused": bool(report.get("reused")),
            "skipped_tensors": report.get("skipped") or {},
            "note": (
                "Packed INT4 weights and fp32 leftovers live in memory-mapped files; the OS "
                "page cache is the RAM buffer. Only LoRA, one dequantized layer and "
                "activations are anonymous memory."
            ),
        }
    elif streaming:
        model, report = load_int4_streaming(
            model_path,
            targets=config.target_modules,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32, low_cpu_mem_usage=True)
        model.config.use_cache = False
        if progress:
            progress(22, "Quantizing to INT4 and attaching LoRA")
        module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
        targets = [name for name in config.target_modules if name in module_names]
        if not targets:
            for candidate in (DEFAULT_TARGETS, ["c_attn", "c_proj"], ["query", "key", "value", "dense"]):
                if all(name in module_names for name in candidate):
                    targets = candidate
                    break
        if not targets:
            raise ValueError("no LoRA target modules matched this architecture")
        report = convert_model(model, targets=targets, r=config.lora_r, alpha=config.lora_alpha, dropout=config.lora_dropout)

    targets = report.get("targets") or list(config.target_modules)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    if log:
        log(f"INT4: {report['converted_layers']} layers quantized, {trainable_count:,} LoRA parameters")
    if not trainable:
        raise RuntimeError("no LoRA parameters were attached; check target modules")

    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    model.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    eval_history: list[dict] = []
    best = {"loss": None, "step": 0}
    started = time.time()
    step = 0
    accum = max(1, config.grad_accum)
    data_index = 0

    while step < config.max_steps:
        if cancel is not None and cancel.is_set():
            raise InterruptedError("INT4 training cancelled")
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for _ in range(accum):
            example = examples[data_index % len(examples)]
            data_index += 1
            ids = _tokenize(tokenizer, example["text"], config.max_seq_len)
            if ids.shape[-1] < 3:
                continue
            outputs = model(input_ids=ids, labels=ids)
            loss = outputs.loss / accum
            loss.backward()
            batch_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        step += 1
        value = batch_loss
        if not math.isfinite(value):
            raise RuntimeError("INT4 training loss became non-finite")
        losses.append(value)
        if log:
            log(f"int4 step {step}: loss={value:.4f}")
        if progress:
            progress(24 + (step / max(1, config.max_steps)) * 66, f"int4 step {step}/{config.max_steps}")

        if eval_texts and config.eval_every > 0 and step % config.eval_every == 0:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for text in eval_texts:
                    ids = _tokenize(tokenizer, text, config.max_seq_len)
                    if ids.shape[-1] < 3:
                        continue
                    out = model(input_ids=ids, labels=ids)
                    if math.isfinite(float(out.loss)):
                        total += float(out.loss)
                        count += 1
            model.train()
            if count:
                eval_loss = total / count
                eval_history.append({"step": step, "eval_loss": round(eval_loss, 6)})
                if best["loss"] is None or eval_loss < float(best["loss"]) - 1e-4:
                    best["loss"] = eval_loss
                    best["step"] = step
                    torch.save(
                        {"lora": {name: p.detach() for name, p in model.named_parameters() if "lora_" in name}},
                        output_dir / "best_lora.pt",
                    )
                if log:
                    log(f"int4 eval step {step}: {eval_loss:.4f}")

    adapter = output_dir / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora": {name: p.detach() for name, p in model.named_parameters() if "lora_" in name},
            "converted_layers": report["converted_layers"],
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "target_modules": list(targets),
        },
        adapter / "lora_int4.pt",
    )
    tokenizer.save_pretrained(str(adapter))

    memory = sampler.report()
    if vram_cache is not None and vram_meta is not None:
        vram_meta["stats"] = vram_cache.stats_public()
        vram_meta["note"] = (
            "Tiered weight cache: fp32 blocks stay in a bounded RAM tier, spill to a memory-mapped "
            "arena file, and are recomputed from the INT4 store only on a miss."
        )
    manifest = {
        "kind": "int4_lora_sft",
        "base_model": str(model_path),
        "dataset": dataset_meta,
        "config": config.to_public(),
        "converted_layers": report["converted_layers"],
        "lora_parameters": trainable_count,
        "target_modules": list(targets),
        "steps": step,
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "eval_history": eval_history,
        "best_eval_loss": round(float(best["loss"]), 6) if best["loss"] is not None else None,
        "best_eval_step": best["step"],
        "duration_seconds": round(time.time() - started, 2),
        "adapter_dir": str(adapter),
        "quantization": {"scheme": "int4_groupwise_absmax", "group_size": GROUP_SIZE},
        "streaming": bool(streaming),
        "offload": store_meta,
        "vram": vram_meta,
        "resources": resources,
        "memory": memory,
        "note": (
            "Base weights are packed to 4 bits and dequantized per layer in the forward pass. "
            "Only LoRA adapters are trained. "
            + (
                "Weights are memory-mapped from the disk-backed INT4 store, so they occupy "
                "file-backed pages instead of anonymous RAM; peak_rss_mb is the measured result."
                if offload_mode
                else "The model was quantized shard by shard while loading, so the full fp32 model "
                "was never resident; peak_rss_mb is the measured result."
                if streaming
                else "Loading materialises the fp32 model once, so peak_rss_mb includes that load peak."
            )
        ),
        "finished_at": utc_now(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    if progress:
        progress(100, "INT4 adapter saved")
    return manifest


def create_int4_run(
    db: Database,
    config: SFTConfig,
    *,
    job_id: str | None,
    output_dir: Path,
    progress=None,
    log=None,
    cancel=None,
    streaming: bool = False,
    offload: bool | None = None,
    store_dir: str | Path | None = None,
    vram_gb: float | None = None,
    vram_mode: str = "auto",
    vram_dir: str | Path | None = None,
) -> dict:
    run_id = new_id("trn")
    db.insert(
        "training_runs",
        {
            "id": run_id,
            "job_id": job_id,
            "dataset_id": config.dataset_id,
            "base_model": config.model_path,
            "backend": "int4_lora_sft",
            "status": "running",
            "config": json.dumps({**config.to_public(), "streaming": streaming, "offload": offload}),
            "metrics": "{}",
            "checkpoint_dir": str(output_dir),
            "created_at": utc_now(),
        },
    )
    try:
        manifest = train_int4(
            db,
            config,
            output_dir=output_dir,
            progress=progress,
            log=log,
            cancel=cancel,
            streaming=streaming,
            offload=offload,
            store_dir=store_dir,
            vram_gb=vram_gb,
            vram_mode=vram_mode,
            vram_dir=vram_dir,
        )
    except Exception as exc:
        db.execute(
            "UPDATE training_runs SET status='failed', metrics=?, finished_at=? WHERE id=?",
            (json.dumps({"error": str(exc)}), utc_now(), run_id),
        )
        raise
    db.execute(
        "UPDATE training_runs SET status='completed', metrics=?, finished_at=? WHERE id=?",
        (json.dumps(manifest), utc_now(), run_id),
    )
    db.activity("training.int4", f"INT4 LoRA run completed: {manifest['steps']} steps", {"run_id": run_id})
    return {"run_id": run_id, "manifest": manifest}
