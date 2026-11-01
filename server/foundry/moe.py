"""Dense → Mixture-of-Experts conversion.

This is a real structural conversion, not a rename:

* every transformer block's feed-forward network is cloned into N experts,
* expert 0 keeps the dense weights, the others start from the same weights plus noise,
* a linear router maps the hidden state to expert logits,
* the forward pass uses top-k routing with a softmax over the selected experts,
* the router is initialised to favour expert 0 so the converted model starts close to
  the dense model instead of from noise.

It is not a trained MoE. Routing quality requires fine-tuning afterwards, which the
manifest states explicitly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import psutil


class MoEError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class MoEConfig:
    experts: int = 4
    top_k: int = 2
    noise: float = 0.01
    router_bias_to_dense: float = 2.0
    seed: int = 42

    def to_public(self) -> dict:
        return asdict(self)


def _torch():
    try:
        import torch
        from torch import nn
    except Exception as exc:  # noqa: BLE001
        raise MoEError("no_torch", f"PyTorch is not importable: {exc}") from exc
    return torch, nn


def _blocks(model):
    """Return (container, attribute) for the transformer block list."""
    candidates = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("transformer.h", getattr(getattr(model, "transformer", None), "h", None)),
        ("gpt_neox.layers", getattr(getattr(model, "gpt_neox", None), "layers", None)),
    ]
    for name, value in candidates:
        if value is not None and len(value) > 0:
            return value
    raise MoEError("unsupported_arch", "Could not locate transformer blocks for this architecture.")


def _mlp_attr(block) -> str:
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(block, name):
            return name
    raise MoEError("unsupported_arch", "Could not locate the feed-forward block.")


class _SparseMoE:
    """Factory that builds an nn.Module with N expert copies of a dense MLP."""

    @staticmethod
    def build(mlp, config: MoEConfig, hidden_size: int):
        torch, nn = _torch()

        class SparseMoE(nn.Module):
            def __init__(self, dense, cfg: MoEConfig):
                super().__init__()
                self.cfg = cfg
                self.experts = nn.ModuleList([copy.deepcopy(dense) for _ in range(cfg.experts)])
                self.router = nn.Linear(hidden_size, cfg.experts, bias=False)
                with torch.no_grad():
                    for index in range(1, cfg.experts):
                        for parameter in self.experts[index].parameters():
                            parameter.add_(torch.randn_like(parameter) * cfg.noise)
                    self.router.weight.zero_()
                    self.router.weight[:, 0] = cfg.router_bias_to_dense

            def forward(self, hidden_states, *args, **kwargs):
                logits = self.router(hidden_states)
                probabilities = torch.softmax(logits, dim=-1)
                top_values, top_indices = torch.topk(probabilities, self.cfg.top_k, dim=-1)
                top_values = top_values / top_values.sum(dim=-1, keepdim=True)
                output = None
                for slot in range(self.cfg.top_k):
                    expert_index = top_indices[..., slot]
                    weight = top_values[..., slot].unsqueeze(-1)
                    for expert_id in range(self.cfg.experts):
                        mask = (expert_index == expert_id).unsqueeze(-1).to(hidden_states.dtype)
                        if not bool(mask.any()):
                            continue
                        expert_out = self.experts[expert_id](hidden_states, *args, **kwargs)
                        if isinstance(expert_out, tuple):
                            expert_out = expert_out[0]
                        contribution = expert_out * mask * weight
                        output = contribution if output is None else output + contribution
                if output is None:
                    output = self.experts[0](hidden_states, *args, **kwargs)
                    if isinstance(output, tuple):
                        output = output[0]
                return output

        return SparseMoE(mlp, config)


def estimate(model_dir: str | Path, config: MoEConfig) -> dict:
    path = Path(model_dir).expanduser()
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    weight_bytes = sum(f.stat().st_size for f in weights)
    config_json = {}
    config_path = path / "config.json"
    if config_path.exists():
        try:
            config_json = json.loads(config_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            config_json = {}
    layers = int(config_json.get("num_hidden_layers") or config_json.get("n_layer") or 0)
    intermediate = int(config_json.get("intermediate_size") or 0)
    hidden = int(config_json.get("hidden_size") or config_json.get("n_embd") or 0)
    ffn_bytes = 0
    if layers and intermediate and hidden:
        # gate + up + down, 3 matrices per layer, 2 bytes per parameter at fp16
        ffn_bytes = layers * 3 * intermediate * hidden * 2
    projected = weight_bytes + ffn_bytes * max(0, config.experts - 1) + layers * hidden * config.experts * 4
    available = psutil.virtual_memory().available
    return {
        "weight_bytes": weight_bytes,
        "ffn_bytes_per_expert": ffn_bytes,
        "projected_bytes": int(projected),
        "available_bytes": available,
        "fits_ram": projected < available * 0.8,
        "layers": layers,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "note": "Projection assumes fp16 on disk and fp32 in memory during conversion.",
    }


def moeify(
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    config: MoEConfig | None = None,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    force: bool = False,
) -> dict:
    torch, nn = _torch()
    config = config or MoEConfig()
    if config.experts < 2:
        raise MoEError("bad_config", "At least two experts are required.")
    if config.top_k < 1 or config.top_k > config.experts:
        raise MoEError("bad_config", "top_k must be between 1 and the expert count.")

    source = Path(model_dir).expanduser()
    target = Path(out_dir).expanduser()
    if not source.is_dir():
        raise MoEError("missing_source", f"Model directory not found: {source}")
    resources = estimate(source, config)
    if not resources["fits_ram"] and not force:
        raise MoEError(
            "does_not_fit",
            f"Projected {resources['projected_bytes'] / 1024**3:.1f} GiB exceeds the safe RAM envelope "
            f"({resources['available_bytes'] / 1024**3:.1f} GiB available). Use a smaller model, fewer experts, or --force.",
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if progress:
        progress(8, "Loading dense model")
    torch.manual_seed(config.seed)
    model = AutoModelForCausalLM.from_pretrained(str(source), torch_dtype=torch.float32, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(str(source))

    blocks = _blocks(model)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
    if not hidden_size:
        raise MoEError("unsupported_arch", "Could not determine the hidden size.")

    replaced = 0
    for index, block in enumerate(blocks):
        attribute = _mlp_attr(block)
        dense = getattr(block, attribute)
        setattr(block, attribute, _SparseMoE.build(dense, config, hidden_size))
        replaced += 1
        if progress:
            progress(10 + (index / max(1, len(blocks))) * 55, f"Converting block {index + 1}/{len(blocks)}")

    if log:
        log(f"converted {replaced} blocks into {config.experts}-expert MoE (top_k={config.top_k})")

    if progress:
        progress(70, "Forward check")
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    if not bool(torch.isfinite(logits).all()):
        raise MoEError("forward_failed", "MoE forward pass produced non-finite logits.")

    target.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(82, "Saving model")
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))

    digest = hashlib.sha256()
    for path in sorted(target.glob("*.safetensors")):
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)

    manifest = {
        "kind": "moe_dense_conversion",
        "source": str(source),
        "experts": config.experts,
        "top_k": config.top_k,
        "noise": config.noise,
        "router": "linear, initialised to favour expert 0",
        "blocks_converted": replaced,
        "hidden_size": hidden_size,
        "resources": resources,
        "trained": False,
        "note": "Routing is initialised, not trained. Fine-tune the router and experts before use.",
        "output_sha256": digest.hexdigest(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.to_public(),
    }
    (target / "moe_manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    if progress:
        progress(100, "MoE model saved")
    return {"output": str(target), "manifest": manifest, "size_bytes": sum(p.stat().st_size for p in target.glob("*") if p.is_file())}


def load_moe(model_dir: str | Path, *, force: bool = False):
    """Rebuild the MoE structure and load saved weights."""
    torch, _ = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_dir).expanduser()
    manifest_path = path / "moe_manifest.json"
    if not manifest_path.exists():
        raise MoEError("not_moe", "No moe_manifest.json in this directory.")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    config = MoEConfig(**manifest["config"])

    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32, low_cpu_mem_usage=True)
    blocks = _blocks(model)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
    for block in blocks:
        attribute = _mlp_attr(block)
        setattr(block, attribute, _SparseMoE.build(getattr(block, attribute), config, hidden_size))
    state = None
    weights_path = path / "model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(weights_path))
    if state is not None:
        model.load_state_dict(state, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model.eval()
    return model, tokenizer, manifest
