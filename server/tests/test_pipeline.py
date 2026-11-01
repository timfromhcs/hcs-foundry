from __future__ import annotations

from pathlib import Path

import pytest

from foundry.data.pipeline import (
    ImportOptions,
    export_jsonl,
    freeze_dataset,
    get_dataset,
    import_dataset,
    list_records,
    preview,
    set_dataset_rights,
    split_for_group,
)


def _import(db, fixtures_dir: Path, name="Test import", rights="needs_review", paths=None):
    return import_dataset(
        db,
        paths=paths or [str(fixtures_dir / "claude_session.jsonl")],
        import_roots=[fixtures_dir],
        options=ImportOptions(name=name, provider="auto", rights_status=rights),
    )


def test_preview_redacts_secrets(db, fixtures_dir):
    data = preview([str(fixtures_dir / "claude_session.jsonl")], [fixtures_dir])
    assert data["inventory"]["file_count"] == 1
    sample = data["samples"][0]
    assert "sk-ant-" not in sample["user_request"]
    assert "[REDACTED" in sample["user_request"]


def test_import_redacts_and_splits(db, fixtures_dir):
    result = _import(db, fixtures_dir)
    assert result["record_count"] >= 1
    records = list_records(db, result["dataset_id"])["items"]
    payload = records[0]["payload"]
    assert "sk-ant-" not in payload["task"]["user_request"]
    assert payload["privacy"]["contains_sensitive_data"] is True
    assert payload["split"] in {"train", "validation", "test"}
    assert payload["metadata"]["split_group"]


def test_split_assignment_is_deterministic():
    assert split_for_group("session-a") == split_for_group("session-a")
    values = {split_for_group(f"session-{i}") for i in range(200)}
    assert {"train", "validation", "test"} <= values


def test_freeze_requires_confirmed_rights(db, fixtures_dir):
    result = _import(db, fixtures_dir, rights="needs_review")
    with pytest.raises(ValueError):
        freeze_dataset(db, result["dataset_id"])


def test_freeze_and_export(db, fixtures_dir):
    result = _import(db, fixtures_dir, rights="confirmed")
    manifest = freeze_dataset(db, result["dataset_id"])
    assert manifest["dataset_hash"]
    assert manifest["record_count"] >= 1
    text = export_jsonl(db, result["dataset_id"])
    assert text.strip()
    dataset = get_dataset(db, result["dataset_id"])
    assert dataset["status"] == "frozen"


def test_export_requires_rights(db, fixtures_dir):
    result = _import(db, fixtures_dir, rights="needs_review")
    with pytest.raises(ValueError):
        export_jsonl(db, result["dataset_id"])


def test_freeze_without_test_split_warns(db, fixtures_dir):
    result = _import(db, fixtures_dir, rights="confirmed")
    manifest = freeze_dataset(db, result["dataset_id"])
    if manifest["holdout_hash"] is None:
        assert manifest.get("warning") == "no_test_split"


def test_duplicate_import_is_deduplicated(db, fixtures_dir):
    first = _import(db, fixtures_dir, name="A")
    second = import_dataset(
        db,
        paths=[str(fixtures_dir / "claude_session.jsonl"), str(fixtures_dir / "claude_session.jsonl")],
        import_roots=[fixtures_dir],
        options=ImportOptions(name="B", provider="auto", rights_status="confirmed"),
    )
    assert second["stats"]["duplicates_removed"] >= 1
    assert second["record_count"] == first["record_count"]


def test_excluded_record_not_exported(db, fixtures_dir):
    result = import_dataset(
        db,
        paths=[str(fixtures_dir / "generic_messages.jsonl")],
        import_roots=[fixtures_dir],
        options=ImportOptions(name="Two", provider="auto", rights_status="confirmed"),
    )
    items = list_records(db, result["dataset_id"])["items"]
    assert len(items) >= 2
    excluded = items[0]
    db.execute(
        "UPDATE dataset_records SET excluded=1 WHERE dataset_id=? AND record_key=?",
        (result["dataset_id"], excluded["record_id"]),
    )
    freeze_dataset(db, result["dataset_id"])
    text = export_jsonl(db, result["dataset_id"])
    assert excluded["record_id"] not in text


def test_rights_update_propagates(db, fixtures_dir):
    result = _import(db, fixtures_dir)
    set_dataset_rights(db, result["dataset_id"], "confirmed")
    assert get_dataset(db, result["dataset_id"])["rights_status"] == "confirmed"
    assert db.query_one(
        "SELECT COUNT(*) AS c FROM dataset_records WHERE dataset_id=? AND rights_status!='confirmed'",
        (result["dataset_id"],),
    )["c"] == 0
