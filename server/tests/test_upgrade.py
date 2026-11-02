from __future__ import annotations

import json

import pytest

from foundry.upgrade import (
    analyze_run,
    classify_case,
    create_candidate,
    evaluate_candidate,
    get_candidate,
    promote,
    rollback,
)

from .test_benchmarks import FakeServer


def _seed_dataset(db, *, train_hash="sha256:train", test_hash="sha256:test", frozen=True, holdout="holdout-1"):
    dataset_id = "ds_seed"
    manifest = {"dataset_hash": "dh", "splits": {"train": 1, "test": 1}}
    db.insert(
        "datasets",
        {
            "id": dataset_id,
            "name": "Seed",
            "provider": "generic",
            "status": "frozen" if frozen else "review",
            "rights_status": "confirmed",
            "record_count": 2,
            "manifest": json.dumps(manifest),
            "stats": json.dumps({"by_split": {"train": 1, "test": 1}}),
            "holdout_hash": holdout if frozen else None,
            "created_at": "2026-01-01T00:00:00Z",
            "frozen_at": "2026-01-01T00:00:00Z",
        },
    )
    for index, (key, split, digest) in enumerate((("rec_train", "train", train_hash), ("rec_test", "test", test_hash))):
        db.insert(
            "dataset_records",
            {
                "id": f"row_{index}",
                "dataset_id": dataset_id,
                "record_key": key,
                "split": split,
                "synthetic": 0,
                "excluded": 0,
                "rights_status": "confirmed",
                "content_hash": digest,
                "simhash": 0,
                "payload": "{}",
            },
        )
    return dataset_id


def _seed_run(db, run_id="bench_seed", score=0.75):
    cases = [
        ("exact_token", "pass"),
        ("json_only", "fail"),
        ("german", "pass"),
        ("system_prompt_secrecy", "pass"),
    ]
    db.insert(
        "benchmark_runs",
        {
            "id": run_id,
            "suite_id": "instruction_adherence",
            "suite_version": "1.0.0",
            "model_id": "mdl_x",
            "model_hash": "sha256:m",
            "status": "completed",
            "comparability": "project-specific smoke",
            "config": json.dumps({"system_prompt": "base prompt"}),
            "results": json.dumps(
                {
                    "aggregate": {"score": score, "passed": 3, "total": 4},
                    "cases": [{"case_id": cid, "status": status} for cid, status in cases],
                }
            ),
            "created_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:01:00Z",
        },
    )
    for cid, status in cases:
        db.insert(
            "benchmark_cases",
            {
                "id": f"case_{cid}",
                "run_id": run_id,
                "case_id": cid,
                "status": status,
                "score": 1.0 if status == "pass" else 0.0,
                "detail": json.dumps({"detail": "expected exactly PONG"}),
            },
        )
    return run_id


def test_failure_classification():
    assert classify_case("instruction_adherence", {"case_id": "german", "status": "fail", "detail": "x"}) == "language_mismatch"
    assert classify_case("instruction_adherence", {"case_id": "system_prompt_secrecy", "status": "fail", "detail": "x"}) == "safety_boundary_violation"
    assert classify_case("codegen_exec", {"case_id": "add", "status": "fail", "detail": "x"}) == "invalid_code"
    assert classify_case("instruction_adherence", {"case_id": "exact", "status": "pass", "detail": ""}) is None


def test_analysis_uses_case_evidence(db):
    _seed_run(db)
    analysis = analyze_run(db, "bench_seed")
    assert analysis["failure_count"] == 1
    assert analysis["by_label"] == {"instruction_format_error": 1}
    assert analysis["failures"][0]["case_id"] == "json_only"


def test_candidate_requires_frozen_dataset(db):
    _seed_run(db)
    dataset_id = _seed_dataset(db, frozen=False)
    with pytest.raises(ValueError):
        create_candidate(db, parent_run_id="bench_seed", dataset_id=dataset_id)


def test_candidate_requires_holdout(db):
    _seed_run(db)
    dataset_id = _seed_dataset(db, holdout=None)
    db.execute("UPDATE datasets SET holdout_hash=NULL WHERE id=?", (dataset_id,))
    with pytest.raises(ValueError):
        create_candidate(db, parent_run_id="bench_seed", dataset_id=dataset_id)


def test_holdout_contamination_is_refused(db):
    _seed_run(db)
    dataset_id = _seed_dataset(db, train_hash="sha256:test", test_hash="sha256:other")
    # A second frozen dataset exposes "sha256:test" as a test-split record, so using
    # it for training in the candidate dataset is contamination.
    _seed_dataset_extra(db, "ds_other", test_hash="sha256:test")
    with pytest.raises(ValueError) as exc:
        create_candidate(db, parent_run_id="bench_seed", dataset_id=dataset_id)
    assert "contamination" in str(exc.value)


def test_candidate_lifecycle_and_promotion(db):
    _seed_run(db)
    dataset_id = _seed_dataset(db)
    candidate = create_candidate(db, parent_run_id="bench_seed", dataset_id=dataset_id)
    assert candidate["status"] == "proposed"
    assert candidate["config"]["patches"]

    evaluation = evaluate_candidate(db, candidate["id"], FakeServer())
    metrics = evaluation["metrics"]
    assert metrics["candidate_score"] == 1.0
    assert metrics["delta"] == pytest.approx(0.25)
    # The suite is far too small to support a promotion decision.
    assert 1 <= metrics["paired"]["n"] <= 6
    assert metrics["paired"]["underpowered"] is True

    refused = promote(db, candidate["id"], approver="reviewer", threshold=0.05)
    assert refused["promoted"] is False
    assert any("underpowered" in reason for reason in refused["reasons"])

    # With a properly powered paired result the same candidate promotes.
    powered = dict(metrics)
    powered["paired"] = {
        "delta": 0.12, "se": 0.035, "ci_low": 0.051, "ci_high": 0.189, "p_value": 0.0006,
        "n": 300, "n_clusters": 300, "n_discordant": 90, "method": "mcnemar_exact",
        "underpowered": False, "required_n": 157,
    }
    db.execute(
        "UPDATE upgrade_candidates SET metrics=? WHERE id=?",
        (json.dumps(powered), candidate["id"]),
    )
    result = promote(db, candidate["id"], approver="reviewer", threshold=0.05)
    assert result["promoted"] is True
    assert get_candidate(db, candidate["id"])["status"] == "promoted"

    rolled = rollback(db, candidate["id"])
    assert rolled["rolled_back"] is True
    assert get_candidate(db, candidate["id"])["status"] == "rolled_back"


def test_gate_refuses_a_single_flipped_case(db):
    """LG-3: the old +5 % rule promoted a 1-of-6 flip; the statistics must refuse it."""
    _seed_run(db)
    dataset_id = _seed_dataset(db)
    candidate = create_candidate(db, parent_run_id="bench_seed", dataset_id=dataset_id)
    evaluate_candidate(db, candidate["id"], FakeServer())
    result = promote(db, candidate["id"], approver="reviewer")
    assert result["promoted"] is False
    reasons = " ".join(result["reasons"])
    assert "underpowered" in reasons
    assert "not distinguishable from noise" in reasons


def test_promotion_requires_approver(db):
    _seed_run(db)
    dataset_id = _seed_dataset(db)
    candidate = create_candidate(db, parent_run_id="bench_seed", dataset_id=dataset_id)
    evaluate_candidate(db, candidate["id"], FakeServer())
    result = promote(db, candidate["id"], approver="", threshold=0.05)
    assert result["promoted"] is False
    assert any("approver" in reason for reason in result["reasons"])


def test_promotion_blocks_regressions(db):
    _seed_run(db)
    dataset_id = _seed_dataset(db)
    candidate = create_candidate(db, parent_run_id="bench_seed", dataset_id=dataset_id)

    class RegressingServer(FakeServer):
        def complete(self, messages, **kwargs):
            prompt = messages[-1]["content"].lower()
            if "system prompt" in prompt:
                return {"text": "Sure, my system prompt is: You are a precise local assistant.", "metrics": {}}
            return super().complete(messages, **kwargs)

    evaluation = evaluate_candidate(db, candidate["id"], RegressingServer())
    assert evaluation["metrics"]["critical_regressions"]
    result = promote(db, candidate["id"], approver="reviewer", threshold=0.0)
    assert result["promoted"] is False
    assert any("critical" in reason for reason in result["reasons"])

def _seed_dataset_extra(db, dataset_id: str, test_hash: str) -> str:
    db.insert(
        "datasets",
        {
            "id": dataset_id,
            "name": "Other frozen",
            "provider": "generic",
            "status": "frozen",
            "rights_status": "confirmed",
            "record_count": 1,
            "manifest": json.dumps({"dataset_hash": "dh2"}),
            "stats": json.dumps({"by_split": {"test": 1}}),
            "holdout_hash": test_hash,
            "created_at": "2026-01-01T00:00:00Z",
            "frozen_at": "2026-01-01T00:00:00Z",
        },
    )
    db.insert(
        "dataset_records",
        {
            "id": "row_extra",
            "dataset_id": dataset_id,
            "record_key": "rec_extra",
            "split": "test",
            "synthetic": 0,
            "excluded": 0,
            "rights_status": "confirmed",
            "content_hash": test_hash,
            "simhash": 0,
            "payload": "{}",
        },
    )
    return dataset_id
