from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="training extras are not installed")

from foundry.training.int4 import (  # noqa: E402
    dequantize_int4,
    estimate_int4_resources,
    int4_size_bytes,
    load_int4_streaming,
    quantize_int4,
    train_int4,
)
from foundry.training.sft import SFTConfig  # noqa: E402

from .test_sft import make_tiny_model, seed_dataset  # noqa: E402


def test_quantize_roundtrip_error_is_small():
    import torch

    torch.manual_seed(0)
    weight = torch.randn(32, 128) * 0.05
    packed, scales, shape = quantize_int4(weight)
    restored = dequantize_int4(packed, scales, shape)
    error = (restored - weight).abs().mean() / weight.abs().mean()
    assert error < 0.15, f"relative error too high: {error}"
    assert packed.dtype == torch.uint8
    assert packed.shape[1] == 128 // 2


def test_int4_is_more_than_three_times_smaller():
    import torch

    weight = torch.randn(64, 256)
    sizes = int4_size_bytes(weight)
    assert sizes["int4_bytes"] * 3 < sizes["fp32_bytes"]


def test_int4_training_runs_and_saves_adapter(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    config = SFTConfig(
        model_path=str(model_dir),
        dataset_id=dataset_id,
        max_steps=2,
        max_seq_len=24,
        grad_accum=1,
        lora_r=2,
        lora_alpha=4,
        learning_rate=1e-3,
        target_modules=["c_attn", "c_proj"],
    )
    manifest = train_int4(db, config, output_dir=tmp_path / "int4")
    assert manifest["kind"] == "int4_lora_sft"
    assert manifest["steps"] == 2
    assert manifest["converted_layers"] >= 2
    assert manifest["lora_parameters"] > 0
    assert manifest["loss_first"] is not None
    adapter = Path(manifest["adapter_dir"]) / "lora_int4.pt"
    assert adapter.exists()
    saved = json.loads((tmp_path / "int4" / "manifest.json").read_text("utf-8"))
    assert saved["quantization"]["scheme"] == "int4_groupwise_absmax"


def test_int4_refuses_missing_model(db, tmp_path: Path):
    dataset_id = seed_dataset(db)
    config = SFTConfig(model_path=str(tmp_path / "nope"), dataset_id=dataset_id, max_steps=1)
    with pytest.raises(ValueError):
        train_int4(db, config, output_dir=tmp_path / "int4")


def test_streaming_loader_quantizes_without_meta_leftovers(tmp_path: Path):
    import torch

    model_dir = make_tiny_model(tmp_path / "tiny")
    model, report = load_int4_streaming(model_dir, targets=["c_attn", "c_proj"], r=2, alpha=4)
    assert report["converted_layers"] >= 2
    assert report["streaming"] is True
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    assert bool(torch.isfinite(logits).all())


def test_streaming_training_runs(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    config = SFTConfig(
        model_path=str(model_dir),
        dataset_id=dataset_id,
        max_steps=2,
        max_seq_len=24,
        grad_accum=1,
        lora_r=2,
        lora_alpha=4,
        learning_rate=1e-3,
        target_modules=["c_attn", "c_proj"],
    )
    manifest = train_int4(db, config, output_dir=tmp_path / "int4s", streaming=True)
    assert manifest["kind"] == "int4_lora_sft"
    assert manifest["resources"]["streaming"] is True
    assert manifest["streaming"] is True
    assert manifest["steps"] == 2


def test_streaming_manifest_records_measured_memory_and_honest_note(db, tmp_path: Path):
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
    manifest = train_int4(db, config, output_dir=tmp_path / "int4m", streaming=True)
    memory = manifest["memory"]
    assert memory["peak_rss_mb"] > 0
    assert memory["samples"] >= 1
    assert "never resident" in manifest["note"]
    assert "fp32 peak" not in manifest["note"]


def test_non_streaming_note_still_discloses_the_fp32_load_peak(db, tmp_path: Path):
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
    manifest = train_int4(db, config, output_dir=tmp_path / "int4f")
    assert manifest["streaming"] is False
    assert "fp32 model once" in manifest["note"]


def test_peak_rss_sampler_observes_a_growing_process():
    from foundry.training.memory import PeakRSS

    sampler = PeakRSS(interval=0.01).start()
    baseline = sampler.peak_rss_mb
    try:
        import torch

        ballast = torch.ones(32 * 1024 * 1024, dtype=torch.float32)  # 128 MB
        assert float(ballast.sum()) == float(32 * 1024 * 1024)
        for _ in range(200):
            if sampler.peak_rss_mb > baseline + 8:
                break
            time.sleep(0.01)
        report = sampler.stop().report()
    finally:
        sampler.stop()
    assert report["peak_rss_mb"] > baseline, "sampler did not observe the allocation"
    assert report["min_available_mb"] is not None
    assert "peak_private_mb" in report


def test_int4_resource_estimate_is_smaller_than_fp32(tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    config = SFTConfig(model_path=str(model_dir), dataset_id="x", max_seq_len=32)
    estimate = estimate_int4_resources(model_dir, config)
    assert estimate["quantized_bytes"] < estimate["weights_bytes"]
    assert estimate["streaming"] is True


def test_blocked_linear_matches_monolithic():
    import torch

    from foundry.training.int4 import blocked_linear, quantize_int4

    torch.manual_seed(0)
    weight = torch.randn(37, 96) * 0.1
    bias = torch.randn(37) * 0.01
    packed, scales, shape = quantize_int4(weight)
    x = torch.randn(2, 5, 96)

    reference = blocked_linear(x, packed, scales, shape, bias, block=10_000)
    for block in (1, 3, 16, 36):
        got = blocked_linear(x, packed, scales, shape, bias, block=block)
        assert got.shape == reference.shape
        assert torch.allclose(got, reference, atol=1e-5), f"block={block} differs"


def test_blocked_gradients_match_monolithic():
    import torch

    from foundry.training.int4 import blocked_grad_input, blocked_linear, quantize_int4

    torch.manual_seed(1)
    weight = torch.randn(23, 64) * 0.1
    packed, scales, shape = quantize_int4(weight)

    def run(block, base):
        x = base.detach().clone().requires_grad_(True)
        y = blocked_linear(x, packed, scales, shape, None, block=block)
        y.sum().backward()
        return y.detach(), x.grad

    torch.manual_seed(7)
    base = torch.randn(3, 64)
    exact, grad_exact = run(10_000, base)
    for block in (2, 7):
        got, grad = run(block, base)
        assert torch.allclose(got, exact, atol=1e-5)
        assert torch.allclose(grad, grad_exact, atol=1e-4)

    grad_out = torch.randn(3, 23)
    ref = blocked_grad_input(grad_out, packed, scales, shape, block=10_000)
    for block in (2, 7):
        assert torch.allclose(blocked_grad_input(grad_out, packed, scales, shape, block=block), ref, atol=1e-4)


def test_block_size_env_is_parsed_defensively(monkeypatch):
    from foundry.training.int4 import DEFAULT_BLOCK, int4_block_size

    monkeypatch.setenv("FOUNDRY_INT4_BLOCK", "64")
    assert int4_block_size() == 64
    monkeypatch.setenv("FOUNDRY_INT4_BLOCK", "not-a-number")
    assert int4_block_size() == DEFAULT_BLOCK
    monkeypatch.setenv("FOUNDRY_INT4_BLOCK", "0")
    assert int4_block_size() == 1
