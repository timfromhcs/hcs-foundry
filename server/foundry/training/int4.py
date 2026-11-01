"""INT4 (QLoRA-style) training without bitsandbytes.

The frozen base weights are packed to 4 bits with group-wise symmetric absmax
quantization and dequantized per layer inside the forward pass. Only the LoRA
adapters are trained and stored in fp32. This works on CPU and needs no CUDA.

Honest limits:

* the current loader still materialises the model in fp32 once, so the peak memory
  during *loading* is the fp32 size; the INT4 saving applies to the training phase,
* dequantization costs throughput; it is a memory trade, not a speed feature.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psutil

from ..db import Database, new_id, utc_now
from .sft import SFTConfig, build_eval_texts, build_examples, estimate_resources

GROUP_SIZE = 64
DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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


class Int4Linear:
    """Factory for a frozen 4-bit linear layer."""

    @staticmethod
    def build(linear, group: int = GROUP_SIZE):
        torch, nn, F = _torch()

        class _Int4Linear(nn.Module):
            def __init__(self, source):
                super().__init__()
                weight = source.weight.detach()
                out_features = int(getattr(source, "out_features", None) or getattr(source, "nf", weight.shape[1]))
                in_features = int(getattr(source, "in_features", None) or weight.shape[0])
                if weight.shape[0] != out_features:
                    weight = weight.t().contiguous()
                packed, scales, shape = quantize_int4(weight)
                self.register_buffer("packed", packed)
                self.register_buffer("scales", scales)
                self.shape = shape
                self.in_features = in_features
                self.out_features = out_features
                if source.bias is not None:
                    self.register_buffer("bias", source.bias.detach().clone())
                else:
                    self.bias = None

            def forward(self, x):
                weight = dequantize_int4(self.packed, self.scales, self.shape, x.dtype)
                return F.linear(x, weight, self.bias)

            def extra_repr(self):
                return f"in={self.in_features}, out={self.out_features}, int4"

        return _Int4Linear(linear)


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


def train_int4(
    db: Database,
    config: SFTConfig,
    *,
    output_dir: Path,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancel=None,
) -> dict:
    torch, nn, _ = _torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"training stack not importable: {exc}") from exc

    model_path = Path(config.model_path).expanduser()
    if not model_path.is_dir():
        raise ValueError(f"base model directory not found: {model_path}")

    resources = estimate_resources(model_path, config)
    if not resources["fits"] and not config.__dict__.get("force"):
        raise ValueError(
            "loading this model still needs its fp32 size in RAM before INT4 conversion "
            f"({resources['estimated_high_bytes'] / 1024**3:.1f} GiB estimated, "
            f"{resources['available_bytes'] / 1024**3:.1f} GiB available). "
            "Use a smaller base model; a streaming INT4 loader is not implemented yet."
        )

    if progress:
        progress(5, "Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    examples, dataset_meta = build_examples(db, config.dataset_id, config.max_seq_len, config.extra_texts)
    eval_texts = build_eval_texts(db, config.dataset_id, config.eval_samples) if config.eval_every else []

    if progress:
        progress(12, "Loading base model (fp32 peak)")
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
        "resources": resources,
        "note": (
            "Base weights are packed to 4 bits and dequantized per layer in the forward pass. "
            "Only LoRA adapters are trained. Loading still needs the fp32 peak."
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
            "config": json.dumps(config.to_public()),
            "metrics": "{}",
            "checkpoint_dir": str(output_dir),
            "created_at": utc_now(),
        },
    )
    try:
        manifest = train_int4(db, config, output_dir=output_dir, progress=progress, log=log, cancel=cancel)
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
