"""Import pipeline: inventory → preview → parse → normalize → redact → dedupe →
segment → label → split → freeze.

Every stage is deterministic and recorded, so a dataset can be rebuilt from the same
source files with the same manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..db import Database, new_id, utc_now
from ..security import resolve_within
from .adapters import ADAPTER_VERSIONS, parse_file
from .redact import redact_record
from .schema import CanonicalRecord

MAX_FILE_BYTES = 512 * 1024 * 1024  # refuse absurdly large single files
PREVIEW_RECORDS = 5


# ---------------------------------------------------------------------------
# Inventory / preview
# ---------------------------------------------------------------------------

def inventory(paths: Iterable[str], import_roots: Iterable[Path]) -> dict:
    files = []
    problems = []
    total = 0
    for raw in paths:
        try:
            path = resolve_within(raw, import_roots)
        except ValueError as exc:
            problems.append({"path": str(raw), "error": str(exc)})
            continue
        if path.is_dir():
            candidates = sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".markdown"}
            )
        else:
            candidates = [path]
        for candidate in candidates:
            try:
                size = candidate.stat().st_size
            except OSError as exc:
                problems.append({"path": str(candidate), "error": str(exc)})
                continue
            if size > MAX_FILE_BYTES:
                problems.append({"path": str(candidate), "error": f"file too large ({size} bytes)"})
                continue
            from .adapters import detect_provider

            files.append(
                {
                    "path": str(candidate),
                    "name": candidate.name,
                    "size_bytes": size,
                    "suffix": candidate.suffix.lower(),
                    "provider_guess": detect_provider(candidate),
                    "modified_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(candidate.stat().st_mtime)
                    ),
                }
            )
            total += size
    return {"files": files, "problems": problems, "total_bytes": total, "file_count": len(files)}


def preview(paths: Iterable[str], import_roots: Iterable[Path], provider: str = "auto", limit: int = PREVIEW_RECORDS) -> dict:
    from .redact import redact_text

    inv = inventory(paths, import_roots)
    samples: list[dict] = []
    errors: list[str] = []
    stats = Counter()
    for entry in inv["files"][:25]:
        result = parse_file(Path(entry["path"]), provider)
        errors.extend(result.errors[:5])
        stats.update({k: v for k, v in result.stats.items() if isinstance(v, int)})
        for record in result.records[: max(0, limit - len(samples))]:
            request = redact_text(record.task.get("user_request") or "")
            trace = [redact_text(step.content[:300]).text for step in record.trace[:3]]
            samples.append(
                {
                    "record_id": record.record_id,
                    "provider": record.source.get("provider"),
                    "task_type": record.task.get("task_type"),
                    "user_request": request.text[:800],
                    "trace_kinds": [step.kind for step in record.trace][:20],
                    "trace_preview": trace,
                    "outcome": record.outcome.get("status"),
                    "has_tool_calls": any(step.kind == "tool_call" for step in record.trace),
                    "redactions": request.findings,
                }
            )
        if len(samples) >= limit:
            break
    return {
        "inventory": inv,
        "samples": samples,
        "errors": errors,
        "stats": dict(stats),
        "preview_redacted": True,
        "detected_schema": {
            "record_fields": ["source", "task", "trace", "outcome", "teaching", "privacy", "split"],
            "schema_version": "1.0",
        },
    }


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------

def simhash(text: str, bits: int = 64) -> int:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0
    width = 3
    shingles = [" ".join(tokens[i : i + width]) for i in range(max(1, len(tokens) - width + 1))]
    vector = [0] * bits
    for shingle in shingles:
        digest = int(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(bits):
            vector[bit] += 1 if (digest >> bit) & 1 else -1
    value = 0
    for bit in range(bits):
        if vector[bit] > 0:
            value |= 1 << bit
    return value


def hamming(a: int, b: int) -> int:
    mask = 0xFFFFFFFFFFFFFFFF
    return bin(((a or 0) & mask) ^ ((b or 0) & mask)).count("1")


def to_sqlite_int(value: int) -> int:
    """SQLite INTEGER is signed 64-bit; 64-bit hashes must be wrapped."""
    value = int(value) & 0xFFFFFFFFFFFFFFFF
    return value - (1 << 64) if value >= (1 << 63) else value


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def split_for_group(group_key: str) -> str:
    digest = int(hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if digest < 80:
        return "train"
    if digest < 90:
        return "validation"
    return "test"


# ---------------------------------------------------------------------------
# Language / label helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "en": {"the", "and", "is", "to", "of", "in", "that", "it", "for", "with"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "mit", "für", "auf", "ein"},
    "fr": {"le", "la", "les", "et", "est", "des", "une", "pour", "dans", "que"},
}


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zA-Zäöüßàâçéèêëîïôûùüÿœ]+", text.lower())
    if not words:
        return "unknown"
    scores = {lang: sum(1 for w in words if w in stops) for lang, stops in STOPWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

@dataclass
class ImportOptions:
    name: str
    provider: str = "auto"
    rights_status: str = "needs_review"
    do_redact: bool = True
    do_dedupe: bool = True
    near_duplicate_distance: int = 3
    task_type_override: str | None = None


def import_dataset(
    db: Database,
    *,
    paths: list[str],
    import_roots: Iterable[Path],
    options: ImportOptions,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
) -> dict:
    inv = inventory(paths, import_roots)
    if not inv["files"]:
        raise ValueError("no importable files found in the selected paths")

    dataset_id = new_id("ds")
    started = time.time()
    records: list[CanonicalRecord] = []
    errors: list[str] = []
    parse_stats: Counter = Counter()

    for index, entry in enumerate(inv["files"]):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("import cancelled")
        if progress is not None:
            progress(index / max(1, len(inv["files"])) * 70, f"Parsing {entry['name']}")
        result = parse_file(Path(entry["path"]), options.provider)
        parse_stats.update({k: v for k, v in result.stats.items() if isinstance(v, int)})
        errors.extend(result.errors[:20])
        records.extend(result.records)

    if not records:
        raise ValueError("the selected files produced no usable records")

    if progress is not None:
        progress(72, f"Normalising {len(records)} records")

    # -- redact ---------------------------------------------------------
    redaction_totals: Counter = Counter()
    if options.do_redact:
        redacted_records = []
        for index, record in enumerate(records):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("import cancelled")
            redacted = redact_record(record)
            for item in redacted.privacy.get("redactions", []):
                redaction_totals[item["type"]] += item["count"]
            redacted.source["rights_status"] = options.rights_status
            redacted_records.append(redacted)
            if progress is not None and index % 25 == 0:
                progress(72 + index / max(1, len(records)) * 12, "Redacting sensitive content")
        records = redacted_records
    else:
        for record in records:
            record.source["rights_status"] = options.rights_status

    # -- label / segment ------------------------------------------------
    for record in records:
        if options.task_type_override:
            record.task["task_type"] = options.task_type_override
        record.metadata["language"] = detect_language(
            (record.task.get("user_request") or "") + " " + record.text_for_embedding[:2000]
        )
        record.metadata["turn_count"] = len(record.trace)
        record.metadata["has_tool_calls"] = any(step.kind == "tool_call" for step in record.trace)
        record.metadata["source_name"] = record.metadata.get("source_name") or Path(record.source.get("source_file_hash") or "unknown").name
        group = record.source.get("source_session_id") or record.task.get("task_id") or record.record_id
        record.metadata["split_group"] = group
        record.split = split_for_group(str(group))

    # -- dedupe ---------------------------------------------------------
    if progress is not None:
        progress(86, "Detecting duplicates")
    kept: list[CanonicalRecord] = []
    seen_hashes: set[str] = set()
    seen_sim: list[tuple[int, str]] = []
    duplicates = 0
    near_duplicates = 0
    for record in records:
        digest = record.canonical_hash()
        if digest in seen_hashes:
            duplicates += 1
            continue
        if options.do_dedupe:
            value = simhash(record.text_for_embedding)
            if any(hamming(value, other) <= options.near_duplicate_distance for other, _ in seen_sim):
                near_duplicates += 1
                continue
            seen_sim.append((value, record.record_id))
        else:
            value = simhash(record.text_for_embedding)
        seen_hashes.add(digest)
        record.metadata["simhash"] = value
        kept.append(record)

    if progress is not None:
        progress(92, "Writing dataset")

    # -- persist --------------------------------------------------------
    providers = sorted({str(record.source.get("provider")) for record in kept})
    dataset_provider = options.provider if options.provider != "auto" else ", ".join(providers)
    db.execute("BEGIN")
    try:
        db.insert(
            "datasets",
            {
                "id": dataset_id,
                "name": options.name or "Imported dataset",
                "provider": dataset_provider,
                "status": "review",
                "rights_status": options.rights_status,
                "record_count": len(kept),
                "manifest": "{}",
                "stats": "{}",
                "created_at": utc_now(),
            },
        )
        rows = []
        for record in kept:
            rows.append(
                (
                    new_id("row"),
                    dataset_id,
                    record.record_id,
                    record.split,
                    1 if record.is_synthetic else 0,
                    0,
                    None,
                    record.source.get("rights_status", "needs_review"),
                    record.canonical_hash(),
                    to_sqlite_int(int(record.metadata.get("simhash") or 0)),
                    record.to_json(),
                )
            )
        db.executemany(
            "INSERT INTO dataset_records(id, dataset_id, record_key, split, synthetic, excluded, exclusion_reason, "
            "rights_status, content_hash, simhash, payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise

    stats = compute_stats(kept)
    stats.update(
        {
            "files": inv["file_count"],
            "source_bytes": inv["total_bytes"],
            "parsed_records": len(records),
            "duplicates_removed": duplicates,
            "near_duplicates_removed": near_duplicates,
            "redactions": dict(redaction_totals),
            "parse_errors": len(errors),
            "parse_stats": dict(parse_stats),
            "duration_seconds": round(time.time() - started, 2),
            "stages": [
                "inventory", "preview", "parse", "normalize", "redact", "dedupe",
                "segment", "label", "split",
            ],
        }
    )
    db.execute(
        "UPDATE datasets SET stats=?, record_count=? WHERE id=?",
        (json.dumps(stats), len(kept), dataset_id),
    )
    db.activity("dataset.import", f"Imported {len(kept)} records into {options.name}", {"dataset_id": dataset_id})

    return {
        "dataset_id": dataset_id,
        "record_count": len(kept),
        "stats": stats,
        "errors": errors[:50],
    }


def compute_stats(records: list[CanonicalRecord]) -> dict:
    by_task = Counter(r.task.get("task_type") for r in records)
    by_provider = Counter(r.source.get("provider") for r in records)
    by_language = Counter((r.metadata or {}).get("language") for r in records)
    by_outcome = Counter(r.outcome.get("status") for r in records)
    by_split = Counter(r.split for r in records)
    synthetic = sum(1 for r in records if r.is_synthetic)
    lengths = sorted(len(r.text_for_embedding) for r in records) or [0]
    return {
        "record_count": len(records),
        "by_task": dict(by_task),
        "by_provider": dict(by_provider),
        "by_language": dict(by_language),
        "by_outcome": dict(by_outcome),
        "by_split": dict(by_split),
        "synthetic_count": synthetic,
        "synthetic_ratio": round(synthetic / len(records), 4) if records else 0.0,
        "median_length_chars": lengths[len(lengths) // 2],
        "max_length_chars": lengths[-1],
    }


# ---------------------------------------------------------------------------
# Review / freeze / export
# ---------------------------------------------------------------------------

def list_datasets(db: Database) -> list[dict]:
    rows = db.query("SELECT * FROM datasets ORDER BY created_at DESC")
    return [_dataset_row(r) for r in rows]


def get_dataset(db: Database, dataset_id: str) -> dict | None:
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    return _dataset_row(row) if row else None


def _dataset_row(row) -> dict:
    data = dict(row)
    data["manifest"] = json.loads(data.get("manifest") or "{}")
    data["stats"] = json.loads(data.get("stats") or "{}")
    return data


def list_records(
    db: Database,
    dataset_id: str,
    *,
    offset: int = 0,
    limit: int = 50,
    split: str | None = None,
    synthetic: bool | None = None,
    include_excluded: bool = True,
) -> dict:
    clauses = ["dataset_id=?"]
    params: list = [dataset_id]
    if split:
        clauses.append("split=?")
        params.append(split)
    if synthetic is not None:
        clauses.append("synthetic=?")
        params.append(1 if synthetic else 0)
    if not include_excluded:
        clauses.append("excluded=0")
    where = " AND ".join(clauses)
    total = db.query_one(f"SELECT COUNT(*) AS c FROM dataset_records WHERE {where}", tuple(params))["c"]
    rows = db.query(
        f"SELECT * FROM dataset_records WHERE {where} ORDER BY rowid LIMIT ? OFFSET ?",
        tuple(params + [limit, offset]),
    )
    items = []
    for row in rows:
        payload = json.loads(row["payload"])
        items.append(
            {
                "id": row["id"],
                "record_id": row["record_key"],
                "split": row["split"],
                "synthetic": bool(row["synthetic"]),
                "excluded": bool(row["excluded"]),
                "exclusion_reason": row["exclusion_reason"],
                "rights_status": row["rights_status"],
                "content_hash": row["content_hash"],
                "payload": payload,
            }
        )
    return {"total": total, "offset": offset, "limit": limit, "items": items}


def set_record_excluded(db: Database, dataset_id: str, record_id: str, excluded: bool, reason: str | None = None) -> dict:
    row = db.query_one(
        "SELECT id FROM dataset_records WHERE dataset_id=? AND (id=? OR record_key=?)",
        (dataset_id, record_id, record_id),
    )
    if not row:
        raise KeyError(record_id)
    db.execute(
        "UPDATE dataset_records SET excluded=?, exclusion_reason=? WHERE id=?",
        (1 if excluded else 0, reason, row["id"]),
    )
    return {"id": row["id"], "excluded": excluded, "reason": reason}


def set_dataset_rights(db: Database, dataset_id: str, rights_status: str) -> dict:
    if rights_status not in {"confirmed", "needs_review", "restricted", "excluded"}:
        raise ValueError("invalid rights status")
    db.execute("UPDATE datasets SET rights_status=? WHERE id=?", (rights_status, dataset_id))
    db.execute("UPDATE dataset_records SET rights_status=? WHERE dataset_id=?", (rights_status, dataset_id))
    return {"dataset_id": dataset_id, "rights_status": rights_status}


def freeze_dataset(db: Database, dataset_id: str) -> dict:
    dataset = get_dataset(db, dataset_id)
    if not dataset:
        raise KeyError(dataset_id)
    if dataset["rights_status"] != "confirmed":
        raise ValueError("rights must be confirmed before freezing")
    rows = db.query("SELECT record_key, content_hash, split, synthetic FROM dataset_records WHERE dataset_id=? AND excluded=0", (dataset_id,))
    if not rows:
        raise ValueError("dataset has no included records")
    test_hashes = sorted(row["content_hash"] for row in rows if row["split"] == "test")
    dataset_hash = hashlib.sha256("\n".join(sorted(row["content_hash"] for row in rows)).encode()).hexdigest()
    holdout_hash = hashlib.sha256("\n".join(test_hashes).encode()).hexdigest() if test_hashes else None
    manifest = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "name": dataset["name"],
        "record_count": len(rows),
        "splits": dict(Counter(row["split"] for row in rows)),
        "synthetic_count": sum(1 for row in rows if row["synthetic"]),
        "dataset_hash": dataset_hash,
        "holdout_hash": holdout_hash,
        "holdout_record_count": len(test_hashes),
        "frozen_at": utc_now(),
        "adapter_versions": ADAPTER_VERSIONS,
    }
    if not test_hashes:
        manifest["warning"] = "no_test_split"
        manifest["warning_detail"] = (
            "No record landed in the test split, so there is no frozen holdout. "
            "This dataset cannot be used to create an upgrade candidate until a holdout exists."
        )
    db.execute(
        "UPDATE datasets SET status='frozen', frozen_at=?, manifest=?, holdout_hash=? WHERE id=?",
        (manifest["frozen_at"], json.dumps(manifest), holdout_hash, dataset_id),
    )
    db.activity("dataset.freeze", f"Froze dataset {dataset['name']}", {"dataset_hash": dataset_hash})
    return manifest


def export_records(db: Database, dataset_id: str, *, include_synthetic: bool = True) -> list[dict]:
    dataset = get_dataset(db, dataset_id)
    if not dataset:
        raise KeyError(dataset_id)
    if dataset["rights_status"] != "confirmed":
        raise ValueError("rights must be confirmed before export")
    clauses = ["dataset_id=?", "excluded=0"]
    params: list = [dataset_id]
    if not include_synthetic:
        clauses.append("synthetic=0")
    rows = db.query(
        f"SELECT payload FROM dataset_records WHERE {' AND '.join(clauses)} ORDER BY split, rowid",
        tuple(params),
    )
    return [json.loads(row["payload"]) for row in rows]


def export_jsonl(db: Database, dataset_id: str, **kwargs) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in export_records(db, dataset_id, **kwargs))


def delete_dataset(db: Database, dataset_id: str) -> None:
    db.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    db.activity("dataset.delete", "Deleted dataset", {"dataset_id": dataset_id})
