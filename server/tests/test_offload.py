from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="training extras are not installed")

from foundry.training.offload import (  # noqa: E402
    Int4Store,
    attach_store,
    build_store,
    load_store_model,
    store_size_bytes,
)
from foundry.training.sft import SFTConfig  # noqa: E402
from foundry.training.int4 import train_int4  # noqa: E402

from .test_sft import make_tiny_model, seed_dataset  # noqa: E402


def _nested_copy(model_dir: Path, target: Path) -> Path:
    """Copy a tiny model and wrap every key in ``model.`` → ``model.language_model.``.

    This reproduces the layout of multimodal checkpoints such as Qwen3.5-VL, where the
    language model sits one level deeper than a text-only ``...ForCausalLM`` expects.
    """
    from safetensors.torch import load_file, save_file

    weights = load_file(str(model_dir / "model.safetensors"))
    nested = {}
    for key, value in weights.items():
        if key.startswith("transformer."):
            nested["transformer.language_model." + key[len("transformer.") :]] = value
        else:
            nested[key] = value
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_dir / "config.json", target / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
        src = model_dir / name
        if src.exists():
            shutil.copy(src, target / name)
    save_file(nested, str(target / "model.safetensors"), metadata={"format": "pt"})
    return target


def test_store_roundtrip_matches_in_ram_int4(tmp_path: Path):
    import torch

    model_dir = make_tiny_model(tmp_path / "tiny")
    store_dir = tmp_path / "store"
    meta = build_store(model_dir, store_dir, targets=["c_attn", "c_proj"])
    assert meta["reused"] is False
    assert meta["counts"]["quantized"] >= 2
    assert store_size_bytes(store_dir) > 0

    store = Int4Store(store_dir)
    key = next(iter(meta["tensors"]))
    packed = store.packed(key)
    scales = store.scales(key)
    info = meta["tensors"][key]
    assert packed.shape == (info["out_features"], info["padded"] // 2)
    assert scales.shape == (info["out_features"], info["scales_per_row"])
    assert packed.dtype == torch.uint8

    # reload the same store → reused, byte-identical metadata
    again = build_store(model_dir, store_dir, targets=["c_attn", "c_proj"])
    assert again["reused"] is True
    store.close()


def test_mmap_model_matches_in_ram_int4_forward(tmp_path: Path):
    import torch

    from foundry.training.int4 import load_int4_streaming

    model_dir = make_tiny_model(tmp_path / "tiny")
    reference, _ = load_int4_streaming(model_dir, targets=["c_attn", "c_proj"], r=2, alpha=4)
    offloaded, report = load_store_model(model_dir, tmp_path / "store", r=2, alpha=4)
    assert report["offload"] is True
    assert report["quantized_layers"] >= 2
    assert report["store_disk_bytes"] > 0

    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    reference.eval()
    offloaded.eval()
    with torch.no_grad():
        expected = reference(input_ids=ids).logits
        actual = offloaded(input_ids=ids).logits
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=2e-2), (actual - expected).abs().max()


def test_store_handles_nested_language_model_keys(tmp_path: Path):
    import torch

    model_dir = make_tiny_model(tmp_path / "tiny")
    nested = _nested_copy(model_dir, tmp_path / "nested")
    meta = build_store(nested, tmp_path / "store", targets=["c_attn", "c_proj"])
    assert meta["counts"]["quantized"] >= 2, "wrapper segments must be dropped, not skipped"

    model, report = load_store_model(nested, tmp_path / "store", r=2, alpha=4)
    assert report["quantized_layers"] >= 2
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert bool(torch.isfinite(logits).all())


def test_store_reports_skipped_subtrees(tmp_path: Path):
    from safetensors.torch import load_file, save_file

    model_dir = make_tiny_model(tmp_path / "tiny")
    weights = load_file(str(model_dir / "model.safetensors"))
    weights["model.visual.fake.weight"] = weights[next(iter(weights))].clone()
    save_file(weights, str(model_dir / "model.safetensors"), metadata={"format": "pt"})

    meta = build_store(model_dir, tmp_path / "store", targets=["c_attn", "c_proj"])
    assert meta["skipped"], "unknown subtrees must be reported, not silently dropped"
    assert any("visual" in group for group in meta["skipped"])


def test_offload_training_records_store_and_memory(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    config = SFTConfig(
        model_path=str(model_dir),
        dataset_id=dataset_id,
        max_steps=1,
        max_seq_len=24,
        grad_accum=1,
        lora_r=2,
        lora_alpha=4,
        learning_rate=1e-3,
        target_modules=["c_attn", "c_proj"],
    )
    manifest = train_int4(db, config, output_dir=tmp_path / "run", streaming=True, offload=True)
    assert manifest["offload"]["disk_bytes"] > 0
    assert manifest["offload"]["store_dir"].endswith("int4-store")
    assert manifest["memory"]["peak_rss_mb"] > 0
    assert "memory-mapped" in manifest["note"]
    saved = json.loads((tmp_path / "run" / "manifest.json").read_text("utf-8"))
    assert saved["offload"]["fingerprint"]
    adapter = Path(manifest["adapter_dir"]) / "lora_int4.pt"
    assert adapter.exists()


def test_offload_requires_streaming(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    config = SFTConfig(model_path=str(model_dir), dataset_id=dataset_id, max_steps=1, target_modules=["c_attn"])
    with pytest.raises(ValueError, match="streaming"):
        train_int4(db, config, output_dir=tmp_path / "run", streaming=False, offload=True)


def test_forward_and_backward_recompute_instead_of_retaining_fp32_weights(tmp_path: Path, monkeypatch):
    """Backward must dequantize again; otherwise every fp32 layer stays alive in the graph."""
    import torch

    from foundry.training import int4 as int4_module

    model_dir = make_tiny_model(tmp_path / "tiny")
    model, report = load_store_model(model_dir, tmp_path / "store", r=2, alpha=4)
    layers = report["quantized_layers"]
    assert layers >= 2

    calls = {"count": 0}
    original = int4_module.dequantize_int4

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(int4_module, "dequantize_int4", counting)

    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = model(input_ids=ids, labels=ids)
    assert torch.isfinite(out.loss)
    assert calls["count"] == layers, f"forward dequantized {calls['count']} of {layers} layers"
    out.loss.backward()
    assert calls["count"] >= layers * 2 - 1, "backward did not recompute the dequantization"

    grads = [p.grad for name, p in model.named_parameters() if "lora_" in name]
    assert any(g is not None for g in grads), "LoRA parameters received no gradient"


def test_bf16_config_is_forced_to_fp32_so_mmap_stays_shared(tmp_path: Path):
    """A checkpoint that declares bf16 must not make accelerate copy every mapped tensor."""
    import torch

    model_dir = make_tiny_model(tmp_path / "tiny")
    config_path = model_dir / "config.json"
    payload = json.loads(config_path.read_text("utf-8"))
    payload["dtype"] = "bfloat16"
    payload["torch_dtype"] = "bfloat16"
    config_path.write_text(json.dumps(payload, indent=2), "utf-8")

    model, report = load_store_model(model_dir, tmp_path / "store", r=2, alpha=4)
    dtypes = {parameter.dtype for parameter in model.parameters()}
    assert dtypes == {torch.float32}, dtypes

    store = model._int4_store
    key = next(k for k in store.meta["rest"] if k.endswith("wte.weight"))
    assert store.tensor(key).data_ptr() == dict(model.named_parameters())[key].data_ptr()
