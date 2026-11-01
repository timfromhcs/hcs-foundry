from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="training extras are not installed")

from foundry.training import playbook  # noqa: E402
from foundry.training.algorithms import run_auto, run_distill, run_dpo  # noqa: E402
from foundry.training.sft import SFTConfig  # noqa: E402

from .test_sft import make_tiny_model, seed_dataset  # noqa: E402


def _config(model_dir: Path, dataset_id: str, **overrides) -> SFTConfig:
    base = dict(
        model_path=str(model_dir),
        dataset_id=dataset_id,
        max_steps=4,
        max_seq_len=32,
        micro_batch=1,
        grad_accum=1,
        lora_r=2,
        lora_alpha=4,
        learning_rate=1e-3,
        save_every=2,
    )
    base.update(overrides)
    return SFTConfig(**base)


def test_auto_run_records_eval_and_playbook(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    result = run_auto(
        db,
        _config(model_dir, dataset_id, auto=True, eval_every=1, eval_samples=2, patience=2),
        job_id=None,
        output_dir=tmp_path / "auto",
    )
    manifest = result["manifest"]
    assert manifest["auto"] is True
    assert manifest["eval_history"], "OpenEval must produce at least one measurement"
    assert manifest["best_eval_loss"] is not None
    assert manifest["playbook_signature"] == result["signature"]

    stored = playbook.entries(db, result["signature"])
    assert len(stored) == 1
    assert stored[0]["improved"] is True
    assert stored[0]["best_eval_loss"] == manifest["best_eval_loss"]


def test_playbook_suggestion_is_reused(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    first = run_auto(
        db,
        _config(model_dir, dataset_id, auto=True, eval_every=1, eval_samples=2),
        job_id=None,
        output_dir=tmp_path / "auto1",
    )
    suggestion = playbook.suggest(db, first["signature"])
    assert suggestion is not None
    second = run_auto(
        db,
        _config(model_dir, dataset_id, auto=True, eval_every=1, eval_samples=2, learning_rate=5e-4),
        job_id=None,
        output_dir=tmp_path / "auto2",
    )
    assert second["manifest"]["playbook_seeded"] is True


def test_distill_uses_teacher_answers(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)

    def teacher(messages):
        return "the patch handles it tests pass"

    result = run_distill(
        db,
        _config(model_dir, dataset_id, max_steps=2),
        teacher_complete=teacher,
        prompts=["fix the failing test", "explain the patch"],
        job_id=None,
        output_dir=tmp_path / "distill",
    )
    manifest = result["manifest"]
    assert manifest["kind"] == "distillation"
    assert manifest["teacher_samples"] == 2
    assert manifest["dataset"]["extra_texts"] == 2
    assert Path(manifest["adapter_dir"], "adapter_config.json").exists()


def test_dpo_runs_and_saves_adapter(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    result = run_dpo(
        db,
        _config(model_dir, dataset_id, max_steps=2, max_seq_len=24),
        beta=0.1,
        output_dir=tmp_path / "dpo",
    )
    manifest = result["manifest"]
    assert manifest["kind"] == "dpo"
    assert manifest["steps"] == 2
    assert manifest["loss_first"] is not None
    assert (Path(manifest["adapter_dir"]) / "adapter_config.json").exists()
    saved = json.loads((tmp_path / "dpo" / "manifest.json").read_text("utf-8"))
    assert saved["pairs"] >= 2
