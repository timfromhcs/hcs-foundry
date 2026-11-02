"""Checks C3-C6, C9-C11: every one must be able to fail.

The matrix is worthless if a check can only pass. Each test below drives one check into its
failing branch with a constructed manifest (fast, deterministic) and, where a real run is the
honest fixture, with the tiny model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foundry.train.base import Preflight, TrainContext
from foundry.train.conformance import (
    check_adapter_reload,
    check_budget,
    check_cancellation,
    check_learning,
    check_no_false_success,
    check_repeatability,
    check_resume_equivalence,
    run_matrix,
)
from foundry.train.manifest import EnvInfo, RunManifest
from foundry.train.registry import get_trainer, load_builtin_trainers
from foundry.train.spec import Backend, Method, Profile, TrainSpec, spec_for_profile

ENV = EnvInfo(python="3.14", platform="win32")


def manifest(**overrides) -> RunManifest:
    payload = {
        "run_id": "trn_test",
        "env": ENV,
        "base_model": "/models/tiny",
        "status": "completed",
        "steps": 4,
        "losses": [4.0, 3.5, 3.0, 2.5],
        "param_delta_max": 0.01,
    }
    payload.update(overrides)
    return RunManifest(**payload)


# -- C3 learns something ----------------------------------------------------


def test_c3_passes_when_the_loss_window_drops():
    result = check_learning(manifest(losses=[4.0, 3.8, 3.6, 3.2, 3.0, 2.8]))
    assert result.status == "passed"
    assert "loss 3.900000 -> 2.900000" in result.detail


def test_c3_fails_on_a_flat_or_rising_loss():
    flat = check_learning(manifest(losses=[3.0, 3.0, 3.0, 3.0]))
    assert flat.status == "failed"
    rising = check_learning(manifest(losses=[2.0, 2.4, 2.9, 3.3]))
    assert rising.status == "failed"


def test_c3_refuses_to_judge_a_one_step_run():
    result = check_learning(manifest(losses=[4.0]))
    assert result.status == "not_applicable"
    assert "need >= 3" in result.detail


def test_c3_fails_on_a_non_finite_loss():
    result = check_learning(manifest(losses=[4.0, float("nan"), 3.0, 2.5]))
    assert result.status == "failed"


# -- C4 repeatability -------------------------------------------------------


def test_c4_passes_identical_runs_and_prints_the_delta():
    result = check_repeatability(manifest(), manifest())
    assert result.status == "passed"
    assert "delta 0.000e+00" in result.detail


def test_c4_fails_when_the_adapter_differs_beyond_tolerance():
    result = check_repeatability(manifest(param_delta_max=0.01), manifest(param_delta_max=0.02))
    assert result.status == "failed"
    assert "delta 1.000e-02" in result.detail


def test_c4_fails_when_only_the_losses_drift():
    result = check_repeatability(manifest(losses=[4.0, 3.0]), manifest(losses=[4.0, 3.5]))
    assert result.status == "failed"


# -- C5 resume equivalence --------------------------------------------------


def test_c5_is_not_applicable_without_a_resume_path():
    result = check_resume_equivalence(None, None)
    assert result.status == "not_applicable"
    assert "resume" in result.detail


def test_c5_fails_when_the_resumed_run_has_a_different_step_count():
    result = check_resume_equivalence(manifest(steps=4), manifest(steps=3, status="resumed"))
    assert result.status == "failed"
    assert "steps differ" in result.detail


# -- C6 cancellation --------------------------------------------------------


def test_c6_is_not_applicable_when_cancel_was_never_exercised():
    assert check_cancellation(manifest(), observed_latency_s=None).status == "not_applicable"


def test_c6_fails_when_a_cancelled_run_claims_completion():
    result = check_cancellation(manifest(status="completed"), observed_latency_s=1.0)
    assert result.status == "failed"
    assert "status=completed" in result.detail


def test_c6_fails_when_cancel_takes_too_long():
    result = check_cancellation(manifest(status="cancelled"), observed_latency_s=120.0)
    assert result.status == "failed"
    assert "cancel took" in result.detail


def test_c6_passes_a_timely_cancel():
    result = check_cancellation(manifest(status="cancelled"), observed_latency_s=2.5)
    assert result.status == "passed"


# -- C9 adapter reload ------------------------------------------------------


def test_c9_is_not_applicable_without_an_adapter():
    assert check_adapter_reload(manifest(adapter_dir=None)).status == "not_applicable"


def test_c9_reports_the_int4_store_format_as_not_applicable(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "lora_int4.pt").write_bytes(b"\0")
    result = check_adapter_reload(manifest(adapter_dir=str(adapter)))
    assert result.status == "not_applicable"
    assert "merge tests" in result.detail


# -- C10 no false success ---------------------------------------------------


def test_c10_fails_a_completed_run_without_a_parameter_update():
    result = check_no_false_success(manifest(param_delta_max=0.0))
    assert result.status == "failed"
    assert "LoRA-B is still zero" in result.detail


def test_c10_passes_a_real_update_and_ignores_non_completed_runs():
    assert check_no_false_success(manifest(param_delta_max=0.005)).status == "passed"
    assert check_no_false_success(manifest(status="cancelled")).status == "not_applicable"


# -- C11 budget -------------------------------------------------------------


def test_c11_is_not_applicable_without_an_estimate_or_budget():
    spec = spec_for_profile(method=Method.SFT, backend=Backend.FP32_LORA, model_path="/m", dataset_id="ds")
    preflight = Preflight(ok=True, est_step_seconds=(0.0, 0.0))
    assert check_budget(preflight, spec, max_wall_s=60).status == "not_applicable"
    assert check_budget(preflight, spec, max_wall_s=None).status == "not_applicable"


def test_c11_refuses_a_run_that_cannot_fit_the_budget():
    spec = spec_for_profile(
        method=Method.SFT,
        backend=Backend.FP32_LORA,
        model_path="/m",
        dataset_id="ds",
        hyper={"max_steps": 100},
    )
    preflight = Preflight(ok=True, est_step_seconds=(2.0, 3.0))
    result = check_budget(preflight, spec, max_wall_s=60)
    assert result.status == "failed"
    assert "budget 60s" in result.detail


# -- the matrix on the tiny model -------------------------------------------


def _spec(tmp_path: Path, model_dir: Path, dataset_id: str) -> TrainSpec:
    return spec_for_profile(
        method=Method.SFT,
        backend=Backend.FP32_LORA,
        model_path=str(model_dir),
        dataset_id=dataset_id,
        profile=Profile.CI_TINY,
        hyper={"seq_len": 24, "max_steps": 6, "lora_r": 2, "lora_alpha": 4, "lr": 5e-2},
        method_args={"eval_every": 0},
    )


def test_matrix_runs_c3_c9_c10_for_sft_on_the_tiny_model(db, settings, tmp_path: Path):
    import threading

    from tests.test_sft import make_tiny_model, seed_dataset

    pytest.importorskip("transformers")
    load_builtin_trainers()
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    spec = _spec(tmp_path, model_dir, dataset_id)
    trainer = get_trainer(spec)
    ctx = TrainContext(out_dir=tmp_path / "run", cancel=threading.Event(), db=db, settings=settings)

    report = run_matrix(trainer, spec, ctx)
    statuses = {check.check: check.status for check in report.checks}

    assert report.ok, [check.to_public() for check in report.failures]
    assert statuses["C3"] == "passed", statuses
    assert statuses["C9"] == "passed", statuses
    assert statuses["C10"] == "passed", statuses
    # checks that need a second run/resume/cancel say so instead of passing silently
    assert statuses["C4"] == "not_applicable"
    assert statuses["C5"] == "not_applicable"
    assert statuses["C6"] == "not_applicable"


def test_matrix_records_the_preflight_refusal(db, settings, tmp_path: Path):
    import threading

    from tests.test_sft import make_tiny_model

    pytest.importorskip("transformers")
    load_builtin_trainers()
    model_dir = make_tiny_model(tmp_path / "tiny")
    spec = _spec(tmp_path, model_dir, "ds_missing")
    trainer = get_trainer(spec)
    ctx = TrainContext(out_dir=tmp_path / "run", cancel=threading.Event(), db=db, settings=settings)

    report = run_matrix(trainer, spec, ctx)
    assert report.ok is True  # nothing failed: it refused honestly
    statuses = {check.check: check.status for check in report.checks}
    assert statuses["C1"] == "passed"
    assert statuses["C2"] == "not_applicable"
