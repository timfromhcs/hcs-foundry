from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="training extras are not installed")

from foundry.training.unified import decide  # noqa: E402

from .test_sft import make_tiny_model, seed_dataset  # noqa: E402


def test_decide_picks_sft_for_a_small_model(db, settings, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    stats = {"record_count": 2, "dataset_hash": "abc"}
    decision = decide(db, settings, model_path=str(model_dir), dataset_id=dataset_id, dataset_stats=stats)
    assert decision["ok"] is True
    assert decision["algorithm"] == "sft"
    assert decision["config"]["max_steps"] >= 1


def test_decide_refuses_missing_model(db, settings, tmp_path: Path):
    decision = decide(db, settings, model_path=str(tmp_path / "nope"), dataset_id="ds", dataset_stats={"record_count": 2})
    assert decision["ok"] is False
    assert "not found" in decision["reason"]


def test_decide_refuses_tiny_dataset(db, settings, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    decision = decide(db, settings, model_path=str(model_dir), dataset_id="ds", dataset_stats={"record_count": 1})
    assert decision["ok"] is False
    assert "two records" in decision["reason"]


def test_decide_switches_to_offload_when_ram_does_not_fit(db, settings, tmp_path: Path, monkeypatch):
    """A model that only fits on disk must still be trainable, not refused."""
    from foundry.training import unified

    model_dir = make_tiny_model(tmp_path / "tiny")
    blocked = {
        "fits": False,
        "estimated_high_bytes": 20 * 1024**3,
        "available_bytes": 6 * 1024**3,
        "streaming": True,
    }
    monkeypatch.setattr(unified, "estimate_resources", lambda *a, **k: dict(blocked, streaming=False))
    monkeypatch.setattr(unified, "estimate_int4_resources", lambda *a, **k: dict(blocked))
    monkeypatch.setattr(unified, "INT4_MIN_BYTES", 0)
    decision = decide(db, settings, model_path=str(model_dir), dataset_id="ds", dataset_stats={"record_count": 3})
    assert decision["ok"] is True
    assert decision["algorithm"] == "int4"
    assert decision["offload"] is True
    assert "disk-backed" in decision["reason"]


def test_unified_int4_merges_before_export_and_reports_every_stage(db, settings, tmp_path: Path, monkeypatch):
    """The INT4 branch must not stop at 'cannot be merged' any more."""
    from foundry.training import unified

    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    monkeypatch.setattr(
        unified,
        "decide",
        lambda *a, **k: {
            "ok": True,
            "algorithm": "int4",
            "offload": True,
            "reason": "test-forced",
            "config": {
                "max_steps": 1,
                "max_seq_len": 24,
                "lora_r": 2,
                "learning_rate": 1e-3,
                "eval_every": 0,
                "patience": 3,
            },
            "signature": "sig",
            "playbook_seeded": False,
            "weights_bytes": 1,
            "estimate": {},
        },
    )
    report = unified.run_unified(
        db, settings, model_path=str(model_dir), dataset_id=dataset_id, options={"max_steps": 1}
    )
    stages = {stage["stage"]: stage for stage in report["stages"]}
    assert stages["train"]["status"] == "ok"
    assert stages["merge"]["status"] == "ok", stages["merge"]
    assert Path(report["artifacts"]["merged_dir"]).is_dir()
    # GGUF export depends on a discovered llama.cpp checkout; it must fail visibly, not silently
    assert stages["export"]["status"] in {"ok", "failed"}


def test_unified_int4_without_a_store_says_why_it_cannot_merge(db, settings, tmp_path: Path, monkeypatch):
    from foundry.training import unified

    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    monkeypatch.setattr(
        unified,
        "decide",
        lambda *a, **k: {
            "ok": True,
            "algorithm": "int4",
            "offload": False,
            "reason": "test-forced",
            "config": {"max_steps": 1, "max_seq_len": 24, "lora_r": 2, "learning_rate": 1e-3, "eval_every": 0, "patience": 3},
            "signature": "sig",
            "playbook_seeded": False,
            "weights_bytes": 1,
            "estimate": {},
        },
    )
    report = unified.run_unified(
        db, settings, model_path=str(model_dir), dataset_id=dataset_id, options={"max_steps": 1}
    )
    stages = {stage["stage"]: stage for stage in report["stages"]}
    assert stages["merge"]["status"] == "skipped"
    assert "disk-backed store" in stages["merge"]["reason"]

    from foundry.training import unified

    model_dir = make_tiny_model(tmp_path / "tiny")
    blocked = {"fits": False, "estimated_high_bytes": 20 * 1024**3, "available_bytes": 6 * 1024**3, "streaming": False}
    roomy = {"fits": True, "estimated_high_bytes": 3 * 1024**3, "available_bytes": 6 * 1024**3, "streaming": True}
    monkeypatch.setattr(unified, "estimate_resources", lambda *a, **k: dict(blocked))
    monkeypatch.setattr(unified, "estimate_int4_resources", lambda *a, **k: dict(roomy))
    # the tiny model is below the INT4 threshold, so raise it to exercise the INT4 branch
    monkeypatch.setattr(unified, "INT4_MIN_BYTES", 0)
    decision = decide(db, settings, model_path=str(model_dir), dataset_id="ds", dataset_stats={"record_count": 3})
    assert decision["algorithm"] == "int4"
    assert decision["offload"] is False
