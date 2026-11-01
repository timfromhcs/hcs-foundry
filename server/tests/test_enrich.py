from __future__ import annotations

import json

from foundry.data.enrich import (
    annotation_report,
    deterministic_annotation,
    enrich_dataset,
    validate_annotation,
)
from foundry.data.pipeline import ImportOptions, import_dataset
from foundry.data.schema import CanonicalRecord, TraceStep


def _record(outcome_text: str | None = "1 passed in 0.4s") -> CanonicalRecord:
    trace = [
        TraceStep(kind="user_message", content="Fix the failing test."),
        TraceStep(kind="assistant_message", content="I changed the null branch."),
    ]
    if outcome_text:
        trace.append(TraceStep(kind="tool_result", content=outcome_text))
    return CanonicalRecord(
        record_id="rec_x",
        source={"provider": "generic", "adapter_version": "test"},
        task={"user_request": "Fix the failing test.", "task_type": "debugging"},
        trace=trace,
    )


def test_deterministic_annotation_quotes_evidence():
    record = _record()
    annotation = deterministic_annotation(record)
    assert annotation["decision_summary"]
    assert "1 passed in 0.4s" in annotation["decision_summary"]
    ok, reason = validate_annotation(record, annotation["decision_summary"])
    assert ok, reason


def test_validator_rejects_invented_quotes():
    record = _record()
    ok, reason = validate_annotation(record, 'The test said "everything is perfect now".')
    assert not ok
    assert "not found" in reason


def test_annotation_without_evidence_is_unknown():
    record = _record(outcome_text=None)
    annotation = deterministic_annotation(record)
    assert annotation["confidence"] == "low"
    assert "unknown" in annotation["decision_summary"].lower()


def test_enrichment_adds_synthetic_rows_without_touching_originals(db, fixtures_dir):
    result = import_dataset(
        db,
        paths=[str(fixtures_dir / "claude_session.jsonl")],
        import_roots=[fixtures_dir],
        options=ImportOptions(name="Enrich", provider="auto", rights_status="confirmed"),
    )
    before = db.query_one("SELECT COUNT(*) AS c FROM dataset_records WHERE dataset_id=?", (result["dataset_id"],))["c"]
    enrichment = enrich_dataset(db, result["dataset_id"], mode="deterministic", limit=10)
    after = db.query_one("SELECT COUNT(*) AS c FROM dataset_records WHERE dataset_id=?", (result["dataset_id"],))["c"]
    assert enrichment.created > 0
    assert after == before + enrichment.created

    rows = db.query("SELECT payload FROM dataset_records WHERE dataset_id=? AND synthetic=1", (result["dataset_id"],))
    for row in rows:
        payload = json.loads(row["payload"])
        assert payload["metadata"]["synthetic"] is True
        assert payload["teaching"]["annotation_origin"] == "synthetic"
        assert payload["metadata"]["teacher"]
        assert payload["metadata"]["source_record_ids"]

    report = annotation_report(db, result["dataset_id"])
    assert report["synthetic_rows"] == enrichment.created
    assert report["grounded"] >= 1


def test_frozen_dataset_refuses_enrichment(db, fixtures_dir):
    from foundry.data.pipeline import freeze_dataset

    result = import_dataset(
        db,
        paths=[str(fixtures_dir / "claude_session.jsonl")],
        import_roots=[fixtures_dir],
        options=ImportOptions(name="Frozen", provider="auto", rights_status="confirmed"),
    )
    freeze_dataset(db, result["dataset_id"])
    try:
        enrich_dataset(db, result["dataset_id"])
    except ValueError as exc:
        assert "frozen" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected frozen datasets to refuse enrichment")