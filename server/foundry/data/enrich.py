"""Synthetic teaching enrichment — evidence-grounded, never reconstructed reasoning.

Two modes:

* ``deterministic`` — templates that quote observable evidence (tool results, test
  output, diffs). Always available, fully reproducible.
* ``model`` — the loaded local model writes the summary; a validator then requires
  every quoted fragment to exist in the record's trace, otherwise the annotation is
  rejected and the row is labelled ``unknown``.

Synthetic rows never replace originals. They carry ``synthetic: true``, the teacher
identity, template version and source record hashes.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Callable

from ..db import Database, new_id, utc_now
from .pipeline import get_dataset, list_records
from .schema import CanonicalRecord, TraceStep, content_hash

TEMPLATE_VERSION = "1.0.0"
QUOTE_RE = re.compile(r"[\"“”'`]([^\"“”'`]{8,300})[\"“”'`]")


@dataclass
class EnrichmentResult:
    created: int
    rejected: int
    skipped: int
    teacher: str
    mode: str


def _evidence_steps(record: CanonicalRecord) -> list[TraceStep]:
    return [s for s in record.trace if s.kind in {"tool_result", "test_result", "artifact"}]


def _test_steps(record: CanonicalRecord) -> list[TraceStep]:
    return [s for s in record.trace if s.kind == "test_result"]


def _clip(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def deterministic_annotation(record: CanonicalRecord) -> dict:
    tests = _test_steps(record)
    evidence = _evidence_steps(record)
    assistant = [s for s in record.trace if s.kind == "assistant_message"]

    if tests:
        signal = _clip(tests[-1].content)
        summary = (
            f"The change was accepted because the test output reports: \"{signal}\". "
            "The passing test is the decisive evidence, not the model's own claim of success."
        )
        confidence = "high"
        negative = "Do not treat a successful build or a plausible explanation as proof; only the test result decides."
        supporting = [tests[-1].content_hash]
    elif evidence:
        signal = _clip(evidence[-1].content)
        summary = (
            f"The action was grounded in observable tool output: \"{signal}\". "
            "The conclusion is limited to what that output shows."
        )
        confidence = "medium"
        negative = "Do not generalise beyond the tool output that was actually observed."
        supporting = [evidence[-1].content_hash]
    elif assistant:
        summary = (
            "No independent evidence (test, tool result, artifact) is present in the trace, "
            "so the outcome is recorded as unknown rather than asserted."
        )
        confidence = "low"
        negative = "Do not present an unverified answer as a verified result."
        supporting = []
    else:
        return {"decision_summary": None, "confidence": None, "negative_example": None, "supporting_evidence": []}

    return {
        "decision_summary": summary,
        "method_summary": _method_summary(record),
        "negative_example": negative,
        "supporting_evidence": supporting,
        "confidence": confidence,
    }


def _method_summary(record: CanonicalRecord) -> str:
    tool_calls = [s for s in record.trace if s.kind == "tool_call"]
    if tool_calls:
        names = []
        for step in tool_calls:
            name = step.tool_name or "tool"
            if name not in names:
                names.append(name)
        return f"Observed approach: {len(tool_calls)} tool call(s) using {', '.join(names[:5])}."
    turns = len(record.trace)
    return f"Observed approach: a {turns}-step conversation without tool calls."


def validate_annotation(record: CanonicalRecord, summary: str | None) -> tuple[bool, str]:
    """Reject annotations that quote text not present in the trace."""
    if not summary:
        return False, "empty summary"
    trace_text = "\n".join(step.content for step in record.trace)
    quotes = QUOTE_RE.findall(summary)
    if not quotes:
        return False, "annotation contains no quotable evidence"
    for quote in quotes:
        fragment = quote.strip().strip("…")
        if len(fragment) < 8:
            continue
        if fragment not in trace_text:
            return False, f"quoted evidence not found in trace: {fragment[:60]!r}"
    return True, "ok"


MODEL_PROMPT = (
    "You are writing a short, factual teaching note about a recorded software task. "
    "Use ONLY the evidence in the trace below. Quote the decisive evidence verbatim in double quotes. "
    "Do not invent steps, do not describe hidden reasoning, do not claim anything the trace does not show. "
    "Answer with one paragraph of at most 60 words."
)


def model_annotation(record: CanonicalRecord, llm, timeout_note: str | None = None) -> dict:
    trace_text = "\n".join(f"[{step.kind}] {step.content[:1500]}" for step in record.trace[:40])
    messages = [
        {"role": "system", "content": MODEL_PROMPT},
        {
            "role": "user",
            "content": f"User request:\n{record.task.get('user_request')}\n\nTrace:\n{trace_text}",
        },
    ]
    try:
        result = llm.complete(messages, temperature=0.2, max_tokens=180)
    except Exception as exc:  # noqa: BLE001 - enrichment must not break an import
        return {"decision_summary": None, "confidence": None, "negative_example": None, "supporting_evidence": [], "error": str(exc)}
    summary = (result.get("text") or "").strip()
    ok, reason = validate_annotation(record, summary)
    return {
        "decision_summary": summary if ok else None,
        "method_summary": _method_summary(record),
        "negative_example": deterministic_annotation(record).get("negative_example"),
        "supporting_evidence": [s.content_hash for s in _evidence_steps(record)[-2:]],
        "confidence": "medium" if ok else "low",
        "validation": reason,
    }


def enrich_dataset(
    db: Database,
    dataset_id: str,
    *,
    mode: str = "deterministic",
    llm=None,
    limit: int = 200,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
) -> EnrichmentResult:
    dataset = get_dataset(db, dataset_id)
    if not dataset:
        raise KeyError(dataset_id)
    if dataset["status"] == "frozen":
        raise ValueError("dataset is frozen; create a new version to add synthetic rows")

    rows = db.query(
        "SELECT id, payload FROM dataset_records WHERE dataset_id=? AND synthetic=0 AND excluded=0 ORDER BY rowid LIMIT ?",
        (dataset_id, limit),
    )
    created = rejected = skipped = 0
    teacher = "deterministic-template" if mode == "deterministic" else getattr(llm, "teacher_name", "local-model")

    for index, row in enumerate(rows):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("enrichment cancelled")
        if progress is not None:
            progress(index / max(1, len(rows)) * 100, f"Annotating {index + 1}/{len(rows)}")
        original = CanonicalRecord.from_dict(json.loads(row["payload"]))
        if mode in {"model", "gateway"} and llm is not None:
            annotation = model_annotation(original, llm)
        else:
            annotation = deterministic_annotation(original)
            annotation["validation"] = "template"
        if not annotation.get("decision_summary"):
            skipped += 1
            continue
        ok, reason = validate_annotation(original, annotation.get("decision_summary"))
        if mode in {"model", "gateway"} and not ok:
            rejected += 1
            annotation["decision_summary"] = None
            annotation["confidence"] = "low"
            annotation["validation"] = reason
            skipped += 1
            continue

        synthetic = CanonicalRecord.from_dict(original.to_dict())
        synthetic.record_id = new_id("syn")
        synthetic.task["task_id"] = synthetic.record_id
        synthetic.metadata["synthetic"] = True
        synthetic.metadata["source_record_ids"] = [original.record_id]
        synthetic.metadata["source_record_hashes"] = [original.canonical_hash()]
        synthetic.metadata["teacher"] = teacher
        synthetic.metadata["template_version"] = TEMPLATE_VERSION
        synthetic.metadata["created_at"] = utc_now()
        synthetic.metadata["validation"] = annotation.get("validation")
        synthetic.teaching.update(
            {
                "decision_summary": annotation.get("decision_summary"),
                "method_summary": annotation.get("method_summary"),
                "negative_example": annotation.get("negative_example"),
                "supporting_evidence": annotation.get("supporting_evidence") or [],
                "annotation_origin": "synthetic",
                "confidence": annotation.get("confidence"),
            }
        )
        synthetic.split = original.split  # keep lineage in the same partition

        db.insert(
            "dataset_records",
            {
                "id": new_id("row"),
                "dataset_id": dataset_id,
                "record_key": synthetic.record_id,
                "split": synthetic.split,
                "synthetic": 1,
                "excluded": 0,
                "rights_status": original.source.get("rights_status", "needs_review"),
                "content_hash": synthetic.canonical_hash(),
                "simhash": 0,
                "payload": synthetic.to_json(),
            },
        )
        created += 1

    total = db.query_one("SELECT COUNT(*) AS c FROM dataset_records WHERE dataset_id=?", (dataset_id,))["c"]
    db.execute("UPDATE datasets SET record_count=? WHERE id=?", (total, dataset_id))
    db.activity(
        "dataset.enrich",
        f"Created {created} synthetic teaching rows",
        {"dataset_id": dataset_id, "mode": mode, "rejected": rejected},
    )
    return EnrichmentResult(created=created, rejected=rejected, skipped=skipped, teacher=teacher, mode=mode)


def annotation_report(db: Database, dataset_id: str) -> dict:
    rows = db.query("SELECT payload FROM dataset_records WHERE dataset_id=?", (dataset_id,))
    synthetic = 0
    grounded = 0
    ungrounded = 0
    for row in rows:
        payload = json.loads(row["payload"])
        if payload.get("metadata", {}).get("synthetic"):
            synthetic += 1
            if payload.get("teaching", {}).get("decision_summary"):
                grounded += 1
            else:
                ungrounded += 1
    return {
        "dataset_id": dataset_id,
        "synthetic_rows": synthetic,
        "grounded": grounded,
        "ungrounded": ungrounded,
        "template_version": TEMPLATE_VERSION,
    }
