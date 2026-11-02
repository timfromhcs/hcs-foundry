from __future__ import annotations

from pathlib import Path

import pytest

from foundry.train.base import TrainContext
from foundry.train.conformance import (
    check_manifest_valid,
    check_preflight_honesty,
    check_real_optimisation,
    check_source_untouched,
    hash_model_files,
)
from foundry.train.manifest import from_legacy
from foundry.train.registry import get_trainer, list_trainers, load_builtin_trainers
from foundry.train.spec import Backend, Method, Profile, TrainSpec, spec_for_profile

pytest.importorskip("transformers", reason="the conformance matrix runs real trainers")


def test_registry_covers_the_documented_methods():
    load_builtin_trainers()
    keys = set(list_trainers())
    assert (Method.SFT, Backend.FP32_LORA) in keys
    assert (Method.INT4, Backend.INT4_STREAM) in keys
    assert (Method.INT4, Backend.INT4_STORE) in keys
    assert (Method.DPO, Backend.FP32_LORA) in keys
    assert (Method.DISTILL, Backend.FP32_LORA) in keys


def _spec(tmp_path: Path, model_dir: Path, dataset_id: str, method: Method, backend: Backend) -> TrainSpec:
    return spec_for_profile(
        method=method,
        backend=backend,
        model_path=str(model_dir),
        dataset_id=dataset_id,
        profile=Profile.CI_TINY,
        hyper={"seq_len": 24, "max_steps": 1, "lora_r": 2, "lora_alpha": 4, "lr": 1e-3},
        method_args={"eval_every": 0},
    )


def _context(tmp_path: Path, db, settings) -> TrainContext:
    import threading

    logs: list[str] = []
    return TrainContext(
        out_dir=tmp_path / "run",
        progress=lambda pct, message: None,
        log=logs.append,
        cancel=threading.Event(),
        db=db,
        settings=settings,
    )


@pytest.mark.parametrize(
    "method,backend,requires_adapter",
    [
        (Method.SFT, Backend.FP32_LORA, True),
        (Method.INT4, Backend.INT4_STREAM, True),
        (Method.INT4, Backend.INT4_STORE, True),
        (Method.DPO, Backend.FP32_LORA, True),
    ],
)
def test_conformance_matrix_on_a_tiny_model(db, settings, tmp_path: Path, method, backend, requires_adapter):
    from tests.test_sft import make_tiny_model, seed_dataset

    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    spec = _spec(tmp_path, model_dir, dataset_id, method, backend)
    trainer = get_trainer(spec)
    ctx = _context(tmp_path, db, settings)

    before = hash_model_files(spec)
    preflight = check_preflight_honesty(trainer, spec, ctx)
    assert preflight.status == "passed", preflight.detail
    assert trainer.preflight(spec, ctx).ok is True

    result = trainer.run(spec, ctx)
    manifest = result.manifest

    assert check_manifest_valid(manifest).status == "passed"
    assert check_source_untouched(trainer, spec, ctx, before).status == "passed"
    if requires_adapter:
        assert manifest.param_delta_max is not None
        assert check_real_optimisation(manifest).status == "passed", check_real_optimisation(manifest).detail
    assert manifest.spec_hash == spec.spec_hash
    assert manifest.steps >= 1


def test_preflight_refuses_a_missing_model(db, settings, tmp_path: Path):
    spec = _spec(tmp_path, tmp_path / "nope", "ds", Method.SFT, Backend.FP32_LORA)
    trainer = get_trainer(spec)
    ctx = _context(tmp_path, db, settings)
    preflight = trainer.preflight(spec, ctx)
    assert preflight.ok is False
    assert any("not found" in blocker for blocker in preflight.blockers)


def test_preflight_refuses_an_unknown_dataset(db, settings, tmp_path: Path):
    from tests.test_sft import make_tiny_model

    model_dir = make_tiny_model(tmp_path / "tiny")
    spec = _spec(tmp_path, model_dir, "ds_missing", Method.SFT, Backend.FP32_LORA)
    preflight = get_trainer(spec).preflight(spec, _context(tmp_path, db, settings))
    assert preflight.ok is False
    assert any("rows" in blocker for blocker in preflight.blockers)


def test_from_legacy_keeps_the_numbers():
    legacy = {
        "kind": "int4_lora_sft",
        "base_model": "/models/qwen",
        "base_model_hash": "sha256:abc",
        "dataset": {"dataset_id": "ds_1", "dataset_hash": "sha256:def", "records_used": 53},
        "config": {"dataset_id": "ds_1", "model_path": "/models/qwen"},
        "converted_layers": 200,
        "lora_parameters": 7_618_560,
        "steps": 2,
        "loss_first": 4.453481,
        "loss_last": 3.545403,
        "duration_seconds": 51.11,
        "adapter_dir": "/runs/x/adapter",
        "quantization": {"scheme": "int4_groupwise_absmax", "group_size": 64},
        "streaming": True,
        "offload": {"store_dir": "/runs/x/int4-store", "disk_bytes": 4567773184},
        "memory": {"peak_rss_mb": 9450.7, "peak_private_mb": 6343.5, "min_available_mb": 1410.0},
        "tuning": {"block_rows": 256, "torch_threads_used": 8},
        "unknown_future_key": {"kept": True},
    }
    manifest = from_legacy(legacy, run_id="trn_x")
    assert manifest.legacy_kind == "int4_lora_sft"
    assert manifest.steps == 2
    assert manifest.losses == [4.453481, 3.545403]
    assert manifest.memory.peak_rss_mb == 9450.7
    assert manifest.extras["converted_layers"] == 200
    assert manifest.extras["offload"]["disk_bytes"] == 4567773184
    assert manifest.extras["unknown_future_key"] == {"kept": True}
    assert manifest.tuning["block_rows"] == 256
