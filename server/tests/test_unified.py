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
