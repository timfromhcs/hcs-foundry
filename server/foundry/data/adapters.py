"""Provider adapters.

Each adapter is versioned and tolerant: unknown fields are preserved in
``metadata``, malformed lines are counted instead of aborting the import, and a
partial final line (common in crash-truncated JSONL) is handled explicitly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .schema import CanonicalRecord, TraceStep, content_hash, new_record

ADAPTER_VERSIONS = {
    "generic": "1.0.0",
    "claude_code": "1.0.0",
    "codex": "1.0.0",
    "opencode": "1.0.0",
    "antigravity": "1.0.0",
}

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass
class AdapterResult:
    records: list[CanonicalRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def merge(self, other: "AdapterResult") -> None:
        self.records.extend(other.records)
        self.errors.extend(other.errors)
        for key, value in other.stats.items():
            if isinstance(value, int):
                self.stats[key] = int(self.stats.get(key, 0)) + value
            else:
                self.stats[key] = value


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _blocks_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                # Tool payloads are captured as separate trace steps; never fold them
                # into the user/assistant message text.
                if btype in {"tool_use", "tool_call", "tool_result", "function_call", "function_call_output"}:
                    continue
                found = False
                for key in ("text", "input_text", "output_text"):
                    value = block.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        found = True
                        break
                if not found and btype in {None, "text"} and isinstance(block.get("content"), str):
                    parts.append(block["content"])
        return "\n".join(parts)
    if isinstance(content, dict):
        return _blocks_to_text([content])
    return str(content)


def _tool_steps_from_blocks(content: Any, ts: str | None) -> list[dict]:
    steps: list[dict] = []
    if not isinstance(content, list):
        return steps
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in {"tool_use", "tool_call", "function_call"}:
            name = block.get("name") or block.get("tool") or "unknown"
            payload = block.get("input") or block.get("arguments") or {}
            steps.append(
                {
                    "kind": "tool_call",
                    "tool_name": str(name),
                    "content": json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload,
                    "timestamp": ts,
                }
            )
        elif btype in {"tool_result", "function_call_output"}:
            steps.append(
                {
                    "kind": "tool_result",
                    "tool_name": block.get("tool_use_id") or block.get("name"),
                    "content": _blocks_to_text(block.get("content") or block.get("output")),
                    "timestamp": ts,
                }
            )
    return steps


def _episodes(events: list[dict]) -> list[list[dict]]:
    """Split a flat event list into user→assistant episodes.

    A new episode starts at a *non-empty* user message once the current episode
    already contains an assistant message. Tool results often arrive in a following
    user turn with no text; those must stay with the episode they belong to.
    """
    episodes: list[list[dict]] = []
    current: list[dict] = []
    for event in events:
        text = (event.get("content") or "").strip()
        if (
            event["kind"] == "user_message"
            and text
            and any(e["kind"] == "assistant_message" for e in current)
        ):
            episodes.append(current)
            current = []
        current.append(event)
    if current:
        episodes.append(current)
    return episodes


def _records_from_events(
    events: list[dict],
    *,
    provider: str,
    session_id: str | None,
    file_hash: str,
    rights_status: str,
    task_type: str,
    metadata: dict | None = None,
) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    for index, episode in enumerate(_episodes(events)):
        user_text = ""
        steps: list[TraceStep] = []
        for event in episode:
            content = event.get("content") or ""
            if event["kind"] == "user_message" and not content.strip():
                continue
            steps.append(
                TraceStep(
                    kind=event["kind"],
                    content=content,
                    timestamp=event.get("timestamp"),
                    tool_name=event.get("tool_name"),
                )
            )
            if event["kind"] == "user_message" and not user_text:
                user_text = content
        if not user_text:
            continue
        records.append(
            new_record(
                provider=provider,
                adapter_version=ADAPTER_VERSIONS[provider],
                session_id=session_id,
                index=index,
                user_request=user_text,
                steps=steps,
                task_type=task_type,
                outcome=_infer_outcome(steps),
                rights_status=rights_status,
                source_file_hash=file_hash,
                metadata=metadata or {},
            )
        )
    return records


SUCCESS_SIGNAL = re.compile(r"(\b\d+\s+passed\b|\ball tests passed\b|\bok\b|\bsuccess(ful|fully)?\b)", re.I)
FAILURE_SIGNAL = re.compile(r"(\b\d+\s+failed\b|\bfailed\b|\btraceback\b|\bexception\b|\berror\b)", re.I)


def _infer_outcome(steps: list[TraceStep]) -> str:
    """Derive an outcome only from observable tool/test output."""
    text = "\n".join(s.content for s in steps if s.kind in {"tool_result", "test_result"})
    if not text:
        return "unknown"
    if FAILURE_SIGNAL.search(text) and not SUCCESS_SIGNAL.search(text):
        return "failure"
    if SUCCESS_SIGNAL.search(text) and not FAILURE_SIGNAL.search(text):
        return "success"
    if SUCCESS_SIGNAL.search(text) and FAILURE_SIGNAL.search(text):
        return "partial"
    return "unknown"


def _guess_task_type(text: str) -> str:
    lower = text.lower()
    if any(word in lower for word in ("error", "traceback", "bug", "fix", "fail", "enoent", "stack trace")):
        return "debugging"
    if any(word in lower for word in ("implement", "refactor", "function", "typescript", "python", "compile")):
        return "coding"
    if any(word in lower for word in ("research", "explain", "compare", "summarize")):
        return "research"
    return "chat"


def _jsonl_lines(text: str, errors: list[str]) -> tuple[list[dict], int, int]:
    items: list[dict] = []
    bad = 0
    total = 0
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        total += 1
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            if len(errors) < 20:
                errors.append(f"line {number}: invalid JSON (skipped)")
            continue
        if isinstance(value, dict):
            items.append(value)
        else:
            bad += 1
            if len(errors) < 20:
                errors.append(f"line {number}: expected an object (skipped)")
    return items, bad, total


# ---------------------------------------------------------------------------
# Generic adapter
# ---------------------------------------------------------------------------

def parse_generic(path: Path, text: str) -> AdapterResult:
    result = AdapterResult()
    suffix = path.suffix.lower()
    file_hash = content_hash(text)

    if suffix in TEXT_SUFFIXES:
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.strip()), path.stem)
        result.records.append(
            new_record(
                provider="generic",
                adapter_version=ADAPTER_VERSIONS["generic"],
                session_id=path.stem,
                index=0,
                user_request=f"Review the document: {title}",
                steps=[TraceStep(kind="assistant_message", content=text)],
                task_type="research",
                source_file_hash=file_hash,
                metadata={"format": "markdown", "source_name": path.name},
            )
        )
        result.stats = {"files": 1, "records": 1, "lines": len(text.splitlines()), "skipped_lines": 0}
        return result

    if suffix == ".jsonl":
        items, bad, total = _jsonl_lines(text, result.errors)
        events = _events_from_loose_items(items)
        result.records.extend(
            _records_from_events(
                events,
                provider="generic",
                session_id=path.stem,
                file_hash=file_hash,
                rights_status="needs_review",
                task_type=_guess_task_type(events[0]["content"] if events else ""),
                metadata={"format": "jsonl", "source_name": path.name},
            )
        )
        result.stats = {"files": 1, "records": len(result.records), "lines": total, "skipped_lines": bad}
        return result

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        result.errors.append(f"{path.name}: invalid JSON ({exc.msg})")
        result.stats = {"files": 1, "records": 0, "lines": len(text.splitlines()), "skipped_lines": 1}
        return result

    sessions = _sessions_from_json(payload)
    for index, session in enumerate(sessions):
        events = _events_from_session(session)
        result.records.extend(
            _records_from_events(
                events,
                provider="generic",
                session_id=str(session.get("id") or session.get("session_id") or f"{path.stem}-{index}"),
                file_hash=file_hash,
                rights_status="needs_review",
                task_type=_guess_task_type(events[0]["content"] if events else ""),
                metadata={"format": "json", "source_name": path.name},
            )
        )
    result.stats = {"files": 1, "records": len(result.records), "lines": len(text.splitlines()), "skipped_lines": 0}
    return result


def _events_from_loose_items(items: Iterable[dict]) -> list[dict]:
    events: list[dict] = []
    for item in items:
        role = item.get("role") or (item.get("message") or {}).get("role")
        if role:
            kind = "user_message" if role in {"user", "human"} else "assistant_message"
            content = item.get("content") if "content" in item else (item.get("message") or {}).get("content")
            events.append(
                {
                    "kind": kind,
                    "content": _blocks_to_text(content),
                    "timestamp": item.get("timestamp") or item.get("created_at"),
                }
            )
            continue
        events.extend(_events_from_session(item))
    return events


def _sessions_from_json(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("sessions", "conversations", "chats", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _events_from_session(session: dict) -> list[dict]:
    events: list[dict] = []
    messages = session.get("messages") or session.get("conversation") or session.get("turns")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or (message.get("info") or {}).get("role")
            if not role:
                continue
            kind = "user_message" if role in {"user", "human"} else "assistant_message"
            content = message.get("content")
            if content is None and isinstance(message.get("parts"), list):
                texts = []
                for part in message["parts"]:
                    if isinstance(part, dict):
                        if part.get("type") in {"text", None} and isinstance(part.get("text"), str):
                            texts.append(part["text"])
                        elif part.get("type") in {"tool", "tool_call"}:
                            events.append(
                                {
                                    "kind": "tool_call",
                                    "content": json.dumps(part.get("state") or part.get("input") or {}, ensure_ascii=False),
                                    "tool_name": part.get("tool") or part.get("name"),
                                    "timestamp": message.get("timestamp"),
                                }
                            )
                content = "\n".join(texts)
            events.append(
                {
                    "kind": kind,
                    "content": _blocks_to_text(content),
                    "timestamp": message.get("timestamp") or (message.get("info") or {}).get("time", {}).get("created"),
                }
            )
    elif session.get("content"):
        events.append({"kind": "assistant_message", "content": _blocks_to_text(session.get("content"))})
    return events


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

def parse_claude_code(path: Path, text: str) -> AdapterResult:
    result = AdapterResult()
    items, bad, total = _jsonl_lines(text, result.errors)
    file_hash = content_hash(text)
    events: list[dict] = []
    session_id = None
    cwd = None
    for item in items:
        if not session_id:
            session_id = item.get("sessionId") or item.get("session_id")
        if not cwd:
            cwd = item.get("cwd")
        kind = item.get("type")
        message = item.get("message") or {}
        ts = item.get("timestamp")
        if kind == "summary":
            continue
        if kind == "user":
            content = message.get("content")
            events.append({"kind": "user_message", "content": _blocks_to_text(content), "timestamp": ts})
            events.extend(_tool_steps_from_blocks(content, ts))
        elif kind == "assistant":
            content = message.get("content")
            events.append({"kind": "assistant_message", "content": _blocks_to_text(content), "timestamp": ts})
            events.extend(_tool_steps_from_blocks(content, ts))
        elif kind in {"system", "queue-operation"}:
            continue

    events.sort(key=lambda e: e.get("timestamp") or "")
    records = _records_from_events(
        events,
        provider="claude_code",
        session_id=session_id or path.stem,
        file_hash=file_hash,
        rights_status="needs_review",
        task_type=_guess_task_type(events[0]["content"] if events else ""),
        metadata={"format": "jsonl", "source_name": path.name, "cwd": cwd},
    )
    result.records = records
    result.stats = {
        "files": 1,
        "records": len(records),
        "lines": total,
        "skipped_lines": bad,
        "session_id": session_id,
    }
    return result


# ---------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------

def parse_codex(path: Path, text: str) -> AdapterResult:
    result = AdapterResult()
    items, bad, total = _jsonl_lines(text, result.errors)
    file_hash = content_hash(text)
    events: list[dict] = []
    session_id = None

    for item in items:
        kind = item.get("type")
        payload = item.get("payload") or {}
        ts = item.get("timestamp")
        if kind == "session_meta":
            session_id = payload.get("id") or payload.get("session_id") or session_id
            continue
        if kind == "turn_context":
            continue
        if kind == "response_item":
            ptype = payload.get("type")
            if ptype == "message":
                role = payload.get("role")
                content = payload.get("content")
                events.append(
                    {
                        "kind": "user_message" if role in {"user", "human"} else "assistant_message",
                        "content": _blocks_to_text(content),
                        "timestamp": ts,
                    }
                )
                events.extend(_tool_steps_from_blocks(content, ts))
            elif ptype in {"function_call", "local_shell_call", "custom_tool_call"}:
                events.append(
                    {
                        "kind": "tool_call",
                        "tool_name": payload.get("name") or ptype,
                        "content": payload.get("arguments") or payload.get("input") or "",
                        "timestamp": ts,
                    }
                )
            elif ptype in {"function_call_output", "local_shell_call_output", "custom_tool_call_output"}:
                events.append(
                    {
                        "kind": "tool_result",
                        "tool_name": payload.get("name"),
                        "content": _blocks_to_text(payload.get("output")),
                        "timestamp": ts,
                    }
                )
        elif kind == "event_msg":
            ptype = payload.get("type")
            if ptype in {"agent_message", "assistant_message"}:
                events.append(
                    {"kind": "assistant_message", "content": _blocks_to_text(payload.get("message")), "timestamp": ts}
                )
            elif ptype in {"user_message", "user_input"}:
                events.append(
                    {"kind": "user_message", "content": _blocks_to_text(payload.get("message")), "timestamp": ts}
                )

    records = _records_from_events(
        events,
        provider="codex",
        session_id=session_id or path.stem,
        file_hash=file_hash,
        rights_status="needs_review",
        task_type=_guess_task_type(events[0]["content"] if events else ""),
        metadata={"format": "jsonl", "source_name": path.name},
    )
    result.records = records
    result.stats = {
        "files": 1,
        "records": len(records),
        "lines": total,
        "skipped_lines": bad,
        "session_id": session_id,
    }
    return result


# ---------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------

def parse_opencode(path: Path, text: str) -> AdapterResult:
    result = AdapterResult()
    file_hash = content_hash(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        result.errors.append(f"{path.name}: invalid JSON ({exc.msg})")
        result.stats = {"files": 1, "records": 0, "lines": len(text.splitlines()), "skipped_lines": 1}
        return result

    sessions = _sessions_from_json(payload)
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        sessions = [payload]
    for index, session in enumerate(sessions):
        events = _events_from_session(session)
        info = session.get("info") or {}
        session_id = info.get("id") or session.get("id") or f"{path.stem}-{index}"
        result.records.extend(
            _records_from_events(
                events,
                provider="opencode",
                session_id=str(session_id),
                file_hash=file_hash,
                rights_status="needs_review",
                task_type=_guess_task_type(events[0]["content"] if events else ""),
                metadata={"format": "json", "source_name": path.name, "title": info.get("title")},
            )
        )
    result.stats = {"files": 1, "records": len(result.records), "lines": len(text.splitlines()), "skipped_lines": 0}
    return result


# ---------------------------------------------------------------------------
# Antigravity / fallback
# ---------------------------------------------------------------------------

def parse_antigravity(path: Path, text: str) -> AdapterResult:
    """Antigravity transcripts (``step_index`` JSONL) or tolerant generic parsing.

    ``thinking`` fields are deliberately ignored: they are the provider's private
    reasoning, not observable evidence.
    """
    if path.suffix.lower() == ".jsonl" and '"step_index"' in text[:4000]:
        result = AdapterResult()
        items, bad, total = _jsonl_lines(text, result.errors)
        file_hash = content_hash(text)
        events: list[dict] = []
        for item in items:
            kind = str(item.get("type") or "")
            content = item.get("content")
            content = content if isinstance(content, str) else ""
            ts = item.get("created_at")
            if kind == "USER_INPUT":
                match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.S)
                events.append({"kind": "user_message", "content": (match.group(1) if match else content).strip(), "timestamp": ts})
            elif kind in {"PLANNER_RESPONSE", "MODEL_RESPONSE", "ASSISTANT_MESSAGE"}:
                if content.strip():
                    events.append({"kind": "assistant_message", "content": content.strip(), "timestamp": ts})
            elif "TOOL" in kind:
                step_kind = "tool_result" if "RESULT" in kind or "OUTPUT" in kind else "tool_call"
                events.append(
                    {
                        "kind": step_kind,
                        "content": content.strip(),
                        "tool_name": item.get("tool_name") or item.get("name"),
                        "timestamp": ts,
                    }
                )
        records = _records_from_events(
            events,
            provider="antigravity",
            session_id=path.parent.parent.parent.name or path.stem,
            file_hash=file_hash,
            rights_status="needs_review",
            task_type=_guess_task_type(events[0]["content"] if events else ""),
            metadata={"format": "antigravity_transcript", "source_name": path.name},
        )
        result.records = records
        result.stats = {"files": 1, "records": len(records), "lines": total, "skipped_lines": bad}
        return result

    result = parse_generic(path, text)
    for record in result.records:
        record.source["provider"] = "antigravity"
        record.source["adapter_version"] = ADAPTER_VERSIONS["antigravity"]
        record.metadata["adapter_note"] = "tolerant parser; provider schema is not public"
    result.stats["adapter"] = "antigravity"
    return result


ADAPTERS = {
    "generic": parse_generic,
    "claude_code": parse_claude_code,
    "codex": parse_codex,
    "opencode": parse_opencode,
    "antigravity": parse_antigravity,
}


def detect_provider(path: Path) -> str:
    lower = str(path).lower().replace("\\", "/")
    name = path.name.lower()
    if "/.claude/projects/" in lower or name.endswith(".jsonl") and "claude" in lower:
        return "claude_code"
    if "/.codex/sessions/" in lower or "codex" in lower:
        return "codex"
    if "opencode" in lower or name.startswith("ses_") or name.startswith("session-"):
        return "opencode"
    if "antigravity" in lower:
        return "antigravity"
    return "generic"


def parse_file(path: Path, provider: str = "auto") -> AdapterResult:
    path = Path(path)
    if not path.is_file():
        return AdapterResult(errors=[f"not a file: {path}"], stats={"files": 0})
    if provider == "auto":
        provider = detect_provider(path)
    adapter = ADAPTERS.get(provider)
    if adapter is None:
        return AdapterResult(errors=[f"unknown provider: {provider}"], stats={"files": 0})
    text = read_text(path)
    try:
        return adapter(path, text)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop an import
        return AdapterResult(errors=[f"{path.name}: adapter error: {exc}"], stats={"files": 1, "records": 0})
