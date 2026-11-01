from __future__ import annotations

from pathlib import Path

from foundry.data.adapters import detect_provider, parse_file


def test_claude_code_adapter(fixtures_dir: Path):
    result = parse_file(fixtures_dir / "claude_session.jsonl", "claude_code")
    assert result.errors == []
    assert len(result.records) == 1
    record = result.records[0]
    assert record.source["provider"] == "claude_code"
    assert record.source["source_session_id"] == "sess-claude-1"
    kinds = [step.kind for step in record.trace]
    assert kinds[0] == "user_message"
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert record.outcome["status"] == "success"
    assert "sk-ant-" in record.task["user_request"]  # raw parse, redaction happens later


def test_codex_adapter(fixtures_dir: Path):
    result = parse_file(fixtures_dir / "codex_session.jsonl", "codex")
    assert len(result.records) == 1
    record = result.records[0]
    assert record.source["provider"] == "codex"
    assert record.source["source_session_id"] == "sess-codex-1"
    assert any(step.kind == "tool_call" and step.tool_name == "shell" for step in record.trace)
    assert record.outcome["status"] == "partial"  # 3 passed, 1 failed


def test_opencode_adapter(fixtures_dir: Path):
    result = parse_file(fixtures_dir / "opencode_export.json", "opencode")
    assert len(result.records) == 1
    record = result.records[0]
    assert record.source["provider"] == "opencode"
    assert record.task["task_type"] == "debugging"


def test_generic_markdown(fixtures_dir: Path):
    result = parse_file(fixtures_dir / "notes.md", "generic")
    assert len(result.records) == 1
    assert result.records[0].task["task_type"] == "research"
    assert result.records[0].trace[0].kind == "assistant_message"


def test_malformed_jsonl_is_counted_not_fatal(fixtures_dir: Path):
    result = parse_file(fixtures_dir / "malformed.jsonl", "generic")
    assert result.stats["skipped_lines"] >= 1
    assert len(result.records) >= 1
    assert result.errors


def test_provider_detection(fixtures_dir: Path):
    assert detect_provider(fixtures_dir / "claude_session.jsonl") in {"generic", "claude_code"}
    assert detect_provider(Path("C:/Users/me/.codex/sessions/x.jsonl")) == "codex"


def test_unknown_provider_returns_error(fixtures_dir: Path):
    result = parse_file(fixtures_dir / "notes.md", "does-not-exist")
    assert result.records == []
    assert result.errors


def test_episode_splitting_does_not_mix_tasks(fixtures_dir: Path):
    result = parse_file(fixtures_dir / "generic_messages.jsonl", "generic")
    # two user requests in one file become two episodes
    assert len(result.records) == 2
    assert result.records[0].task["user_request"].startswith("Summarize")
    assert result.records[1].task["user_request"].startswith("Now answer")
