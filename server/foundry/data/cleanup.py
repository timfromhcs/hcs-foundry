"""One-command chat-log cleanup.

discover → import all sources → redact → dedupe (pipeline) → quality filter →
rights → freeze → quarantine copy → report.

Originals are never modified. Excluded records are marked, not deleted, so the
decision stays auditable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..db import Database, utc_now
from .pipeline import export_jsonl, freeze_dataset, get_dataset, set_dataset_rights
from .sources import import_all

MIN_ANSWER_CHARS = 20


def _answer_text(payload: dict) -> str:
    for step in reversed(payload.get("trace") or []):
        if step.get("kind") == "assistant_message" and (step.get("content") or "").strip():
            return step["content"].strip()
    return ""


def cleanup_all(
    db: Database,
    settings: Settings,
    *,
    providers: list[str] | None = None,
    rights_status: str = "confirmed",
    quality_filter: bool = True,
    quarantine: bool = True,
    progress: Callable[[float, str], None] | None = None,
    cancel=None,
) -> dict:
    providers = providers or ["claude_code", "codex", "antigravity", "opencode"]
    if progress:
        progress(2, "importing sources")
    imported = import_all(
        db,
        settings,
        providers,
        rights_status=rights_status,
        progress=(lambda pct, msg: progress(2 + pct * 0.6, msg)) if progress else None,
        cancel=cancel,
    )
    dataset_id = imported["dataset_id"]

    excluded = 0
    if quality_filter:
        if progress:
            progress(65, "quality filter")
        rows = db.query("SELECT id, payload FROM dataset_records WHERE dataset_id=? AND excluded=0", (dataset_id,))
        for row in rows:
            payload = json.loads(row["payload"])
            answer = _answer_text(payload)
            request = ((payload.get("task") or {}).get("user_request") or "").strip()
            reason = None
            if not request:
                reason = "empty request"
            elif len(answer) < MIN_ANSWER_CHARS:
                reason = f"answer shorter than {MIN_ANSWER_CHARS} characters"
            if reason:
                db.execute(
                    "UPDATE dataset_records SET excluded=1, exclusion_reason=? WHERE id=?",
                    (reason, row["id"]),
                )
                excluded += 1

    set_dataset_rights(db, dataset_id, rights_status)

    frozen = None
    freeze_error = None
    if rights_status == "confirmed":
        try:
            frozen = freeze_dataset(db, dataset_id)
        except ValueError as exc:
            freeze_error = str(exc)
    else:
        freeze_error = "rights are not confirmed; freeze skipped"

    quarantine_path = None
    if quarantine and rights_status == "confirmed":
        if progress:
            progress(85, "quarantine copy")
        target = settings.data_dir / "quarantine" / f"{dataset_id}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(export_jsonl(db, dataset_id), encoding="utf-8")
            quarantine_path = str(target)
        except ValueError:
            quarantine_path = None

    dataset = get_dataset(db, dataset_id) or {}
    stats = dataset.get("stats") or {}
    included = db.query_one(
        "SELECT COUNT(*) AS c FROM dataset_records WHERE dataset_id=? AND excluded=0", (dataset_id,)
    )["c"]
    report = {
        "dataset_id": dataset_id,
        "created_at": utc_now(),
        "providers": providers,
        "files_by_source": imported.get("files_by_source"),
        "total_files": imported.get("total_files"),
        "records_imported": imported.get("record_count"),
        "records_included": included,
        "records_excluded": excluded,
        "duplicates_removed": (imported.get("stats") or {}).get("duplicates_removed"),
        "near_duplicates_removed": (imported.get("stats") or {}).get("near_duplicates_removed"),
        "redactions": (imported.get("stats") or {}).get("redactions"),
        "splits": (frozen or {}).get("splits") or stats.get("by_split"),
        "dataset_hash": (frozen or {}).get("dataset_hash"),
        "holdout_hash": (frozen or {}).get("holdout_hash"),
        "frozen": frozen is not None,
        "freeze_error": freeze_error,
        "quarantine": quarantine_path,
        "originals_modified": False,
    }
    target = settings.data_dir / "reports" / f"cleanup-{dataset_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), "utf-8")
    report["report"] = str(target)
    db.activity("logs.cleanup", f"Cleaned {imported.get('record_count')} records into {dataset_id}", {"excluded": excluded})
    if progress:
        progress(100, "cleanup finished")
    return report
