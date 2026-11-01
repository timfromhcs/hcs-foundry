"""Canonical, versioned event schema shared by every provider adapter.

Shape follows AGENTS.md §7.5. Downstream stages (redaction, dedupe, split, export,
training) only ever see this representation, which keeps provider quirks out of the
rest of the system.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"

PROVIDERS = {"claude_code", "codex", "opencode", "antigravity", "generic"}

TRACE_KINDS = {
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "test_result",
    "artifact",
    "system_message",
}

OUTCOME_STATUS = {"success", "partial", "failure", "unknown"}
TASK_TYPES = {"coding", "debugging", "research", "visual", "agentic", "chat", "other"}
RIGHTS_STATUS = {"confirmed", "needs_review", "restricted", "excluded"}
ANNOTATION_ORIGIN = {"source", "synthetic", "human_reviewed"}
SPLITS = {"train", "validation", "test", "unassigned"}


def content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_record_id(provider: str, session_id: str | None, index: int, seed: str) -> str:
    raw = f"{provider}|{session_id or ''}|{index}|{seed[:2000]}"
    return "rec_" + hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:24]


@dataclass
class TraceStep:
    kind: str
    content: str
    timestamp: str | None = None
    tool_name: str | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in TRACE_KINDS:
            raise ValueError(f"unknown trace kind: {self.kind}")
        if not self.content_hash:
            self.content_hash = content_hash(self.content)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CanonicalRecord:
    record_id: str
    source: dict = field(default_factory=dict)
    task: dict = field(default_factory=dict)
    trace: list[TraceStep] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)
    teaching: dict = field(default_factory=dict)
    privacy: dict = field(default_factory=dict)
    split: str = "unassigned"
    metadata: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.source.setdefault("provider", "generic")
        self.source.setdefault("adapter_version", "unknown")
        self.source.setdefault("rights_status", "needs_review")
        self.task.setdefault("task_id", self.record_id)
        self.task.setdefault("task_type", "other")
        self.task.setdefault("user_request", "")
        self.task.setdefault("context_summary", None)
        self.outcome.setdefault("status", "unknown")
        self.outcome.setdefault("evidence", [])
        self.outcome.setdefault("tests", [])
        self.outcome.setdefault("user_feedback", None)
        self.teaching.setdefault("decision_summary", None)
        self.teaching.setdefault("method_summary", None)
        self.teaching.setdefault("supporting_evidence", [])
        self.teaching.setdefault("negative_example", None)
        self.teaching.setdefault("annotation_origin", "source")
        self.teaching.setdefault("confidence", None)
        self.privacy.setdefault("redactions", [])
        self.privacy.setdefault("contains_sensitive_data", False)
        self.privacy.setdefault("approved_for_training", False)
        if self.split not in SPLITS:
            raise ValueError(f"invalid split: {self.split}")

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        data = asdict(self)
        data["trace"] = [step.to_dict() if isinstance(step, TraceStep) else step for step in self.trace]
        data["schema_version"] = SCHEMA_VERSION
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalRecord":
        trace = [TraceStep(**step) if isinstance(step, dict) else step for step in data.get("trace", [])]
        return cls(
            record_id=data["record_id"],
            source=dict(data.get("source") or {}),
            task=dict(data.get("task") or {}),
            trace=trace,
            outcome=dict(data.get("outcome") or {}),
            teaching=dict(data.get("teaching") or {}),
            privacy=dict(data.get("privacy") or {}),
            split=data.get("split", "unassigned"),
            metadata=dict(data.get("metadata") or {}),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    # -- derived -----------------------------------------------------------
    @property
    def is_synthetic(self) -> bool:
        return bool(self.metadata.get("synthetic")) or self.teaching.get("annotation_origin") == "synthetic"

    @property
    def text_for_embedding(self) -> str:
        parts = [self.task.get("user_request") or ""]
        parts += [step.content for step in self.trace if step.kind in {"assistant_message", "tool_result"}]
        return "\n".join(p for p in parts if p)

    def canonical_hash(self) -> str:
        return content_hash(self.to_json())

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.source.get("provider") not in PROVIDERS:
            problems.append(f"unknown provider {self.source.get('provider')!r}")
        if self.task.get("task_type") not in TASK_TYPES:
            problems.append(f"unknown task_type {self.task.get('task_type')!r}")
        if self.outcome.get("status") not in OUTCOME_STATUS:
            problems.append(f"unknown outcome status {self.outcome.get('status')!r}")
        if not any(step.kind in {"user_message", "assistant_message"} for step in self.trace):
            problems.append("trace has no user or assistant message")
        if not self.task.get("user_request"):
            problems.append("empty user_request")
        return problems


def new_record(
    *,
    provider: str,
    adapter_version: str,
    session_id: str | None,
    index: int,
    user_request: str,
    steps: list[TraceStep],
    task_type: str = "other",
    outcome: str = "unknown",
    rights_status: str = "needs_review",
    source_file_hash: str | None = None,
    metadata: dict | None = None,
) -> CanonicalRecord:
    seed = user_request + "".join(step.content for step in steps)
    rid = stable_record_id(provider, session_id, index, seed)
    return CanonicalRecord(
        record_id=rid,
        source={
            "provider": provider,
            "source_file_hash": source_file_hash,
            "source_session_id": session_id,
            "adapter_version": adapter_version,
            "rights_status": rights_status,
        },
        task={"task_id": rid, "task_type": task_type, "user_request": user_request},
        trace=steps,
        outcome={"status": outcome},
        metadata=metadata or {},
    )
