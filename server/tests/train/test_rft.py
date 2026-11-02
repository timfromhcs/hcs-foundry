"""RFT: sample only train tasks, verify, train on the accepted pairs, refuse a zero-accept run."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from foundry.train.adapters import rft
from foundry.train.base import TrainContext, TrainingError
from foundry.train.conformance import run_matrix
from foundry.train.registry import get_trainer, load_builtin_trainers
from foundry.train.spec import Backend, Method, Profile, spec_for_profile
from foundry.tasks import generators as gen

pytest.importorskip("transformers")


def _tasks_file(tmp_path: Path, *, seed: int = 1) -> tuple[Path, list[gen.TaskItem]]:
    items = gen.generate_all(seed=seed, per_template=12)
    path = tmp_path / "tasks.jsonl"
    gen.to_jsonl(path, items)
    return path, items


def _spec(tmp_path: Path, model_dir: Path, tasks: Path, **method_args) -> "object":
    args = {"tasks_path": str(tasks), "samples_per_task": 1, "min_accepted": 4, "max_candidates": 128, "seed": 3}
    args.update(method_args)
    return spec_for_profile(
        method=Method.RFT,
        backend=Backend.FP32_LORA,
        model_path=str(model_dir),
        dataset_id="tasks_jsonl",
        profile=Profile.CI_TINY,
        hyper={"seq_len": 24, "max_steps": 6, "lora_r": 2, "lora_alpha": 4, "lr": 5e-2},
        method_args=args,
    )


def _oracle(items: list[gen.TaskItem]):
    answers = {item.prompt: item.answer for item in items if item.split == "train"}

    def sampler(prompt: str, n: int) -> list[str]:
        return [answers.get(prompt, "no answer")] * n

    return sampler


def _context(tmp_path: Path, db, settings) -> TrainContext:
    return TrainContext(
        out_dir=tmp_path / "run",
        log=lambda message: None,
        cancel=threading.Event(),
        db=db,
        settings=settings,
    )


def test_rft_reaches_the_matrix_with_accepted_samples(monkeypatch, db, settings, tmp_path: Path):
    from tests.test_sft import make_tiny_model

    load_builtin_trainers()
    model_dir = make_tiny_model(tmp_path / "tiny")
    tasks, items = _tasks_file(tmp_path)
    monkeypatch.setattr(rft, "make_sampler", lambda spec: _oracle(items))

    spec = _spec(tmp_path, model_dir, tasks, min_accepted=6)
    trainer = get_trainer(spec)
    report = run_matrix(trainer, spec, _context(tmp_path, db, settings))

    statuses = {check.check: check.status for check in report.checks}
    assert report.ok, [check.to_public() for check in report.failures]
    assert statuses["C3"] == "passed", statuses
    assert statuses["C9"] == "passed", statuses
    assert statuses["C10"] == "passed", statuses
    # checks that need a second run / resume / cancel still say why
    assert statuses["C4"] == "not_applicable"
    assert statuses["C5"] == "not_applicable"
    assert statuses["C6"] == "not_applicable"


def test_rft_manifest_records_counts_versions_and_seed(monkeypatch, db, settings, tmp_path: Path):
    from tests.test_sft import make_tiny_model

    load_builtin_trainers()
    model_dir = make_tiny_model(tmp_path / "tiny")
    tasks, items = _tasks_file(tmp_path)
    monkeypatch.setattr(rft, "make_sampler", lambda spec: _oracle(items))

    spec = _spec(tmp_path, model_dir, tasks, min_accepted=6)
    result = get_trainer(spec).run(spec, _context(tmp_path, db, settings))
    extras = result.manifest.extras

    assert extras["accepted_count"] >= 6
    assert extras["rejected_count"] == 0
    assert extras["seed"] == 3
    assert extras["generator_version"] == rft.GENERATOR_VERSION
    assert extras["verifier_version"] == rft.VERIFIER_VERSION
    assert extras["accepted_dataset_hash"].startswith("sha256:")
    assert extras["used_test_items"] > 0  # the file has a test split; it was never sampled
    assert result.manifest.dataset_id == extras["accepted_dataset_id"]
    assert result.manifest.param_delta_max and result.manifest.param_delta_max > 0

    stored = db.query_one("SELECT rights_status, provider, status FROM datasets WHERE id=?", (result.manifest.dataset_id,))
    assert stored["rights_status"] == "confirmed"
    assert stored["provider"] == "synthetic"
    assert stored["status"] == "frozen"


def test_rft_fails_when_no_candidate_passes(monkeypatch, db, settings, tmp_path: Path):
    from tests.test_sft import make_tiny_model

    load_builtin_trainers()
    model_dir = make_tiny_model(tmp_path / "tiny")
    tasks, _ = _tasks_file(tmp_path)
    monkeypatch.setattr(rft, "make_sampler", lambda spec: (lambda prompt, n: ["i do not know"] * n))

    spec = _spec(tmp_path, model_dir, tasks)
    ctx = _context(tmp_path, db, settings)
    with pytest.raises(TrainingError) as excinfo:
        get_trainer(spec).run(spec, ctx)
    assert "passed the verifier" in str(excinfo.value)

    manifest = json.loads((ctx.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["extras"]["accepted_count"] == 0
    assert manifest["extras"]["rejected_count"] > 0
    assert manifest["extras"]["rejection_reasons"]["verifier_rejected"] > 0


def test_a_plausible_but_wrong_answer_is_rejected(monkeypatch, db, settings, tmp_path: Path):
    from tests.test_sft import make_tiny_model

    load_builtin_trainers()
    model_dir = make_tiny_model(tmp_path / "tiny")
    tasks, items = _tasks_file(tmp_path)

    def almost(prompt: str, n: int) -> list[str]:
        answers = {item.prompt: item.answer for item in items if item.split == "train"}
        value = answers.get(prompt, "0")
        return [value + "1"] * n

    monkeypatch.setattr(rft, "make_sampler", lambda spec: almost)
    spec = _spec(tmp_path, model_dir, tasks)
    with pytest.raises(TrainingError):
        get_trainer(spec).run(spec, _context(tmp_path, db, settings))


def test_leaked_templates_are_refused_before_sampling(monkeypatch, db, settings, tmp_path: Path):
    from tests.test_sft import make_tiny_model

    load_builtin_trainers()
    model_dir = make_tiny_model(tmp_path / "tiny")
    tasks, items = _tasks_file(tmp_path)
    leaked = items + [
        gen.TaskItem(
            task_id="leak",
            family=items[0].family,
            template_id=items[0].template_id,
            split="test" if items[0].split != "test" else "train",
            prompt="leaked prompt",
            answer="x",
        )
    ]
    leaked_path = tmp_path / "leaked.jsonl"
    gen.to_jsonl(leaked_path, leaked)

    spec = _spec(tmp_path, model_dir, leaked_path)
    preflight = get_trainer(spec).preflight(spec, _context(tmp_path, db, settings))
    assert preflight.ok is False
    assert any("templates appear in more than one split" in blocker for blocker in preflight.blockers)

    with pytest.raises(TrainingError):
        get_trainer(spec).run(spec, _context(tmp_path, db, settings))


def test_the_candidate_budget_is_enforced_in_preflight(db, settings, tmp_path: Path):
    from tests.test_sft import make_tiny_model

    load_builtin_trainers()
    model_dir = make_tiny_model(tmp_path / "tiny")
    tasks, _ = _tasks_file(tmp_path)
    spec = _spec(tmp_path, model_dir, tasks, max_candidates=5)
    preflight = get_trainer(spec).preflight(spec, _context(tmp_path, db, settings))
    assert preflight.ok is False
    assert any("candidate budget" in blocker for blocker in preflight.blockers)
