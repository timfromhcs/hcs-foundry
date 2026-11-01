from __future__ import annotations

from pathlib import Path

import pytest

from foundry.agent import TOOLS, ToolContext, execute_task, plan


class FakeJobCtx:
    def __init__(self):
        self.cancelled = False
        self.logs: list[str] = []

    def progress(self, pct, message=None):
        self.logs.append(f"{pct}:{message}")

    def log(self, message, level="info", data=None):
        self.logs.append(message)

    def emit(self, kind, message, data=None, level="info"):
        self.logs.append(f"{kind}:{message}")

    def check_cancel(self):
        pass


def test_plan_maps_goals_to_readonly_tools():
    result = plan("Inspect the machine and list local models", "supervised")
    tools = [step["tool"] for step in result["steps"]]
    assert "machine_scan" in tools
    assert "scan_models" in tools
    assert result["needs_approval"] is False


def test_mutating_tool_requires_approval():
    result = plan("run a benchmark for the loaded model", "supervised")
    assert result["needs_approval"] is True
    assert any(step["requires_approval"] for step in result["steps"])


def test_trusted_autonomy_does_not_require_approval():
    result = plan("run a benchmark", "trusted")
    assert result["needs_approval"] is False


def test_tool_catalog_has_no_shell():
    assert "shell" not in TOOLS
    assert "run_command" not in TOOLS
    assert all(tool.name for tool in TOOLS.values())


def test_read_file_cannot_escape_workspace(tmp_path: Path, settings):
    context = ToolContext(db=None, settings=settings, workspace=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TOOLS["read_file"].run(context, {"path": "../../etc/passwd"})


def test_read_file_inside_workspace(tmp_path: Path, settings):
    target = tmp_path / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    context = ToolContext(db=None, settings=settings, workspace=tmp_path)  # type: ignore[arg-type]
    result = TOOLS["read_file"].run(context, {"path": "notes.txt"})
    assert result["content"] == "hello"


def test_execute_task_runs_allowlisted_steps(tmp_path: Path, settings, db):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    task = {
        "goal": "list files",
        "plan": [
            {"tool": "list_files", "args": {"limit": 10}, "mutating": False},
        ],
    }
    context = ToolContext(db=db, settings=settings, workspace=tmp_path)
    result = execute_task(context, task, FakeJobCtx())
    assert result["steps"][0]["status"] == "ok"
    assert "Workspace files listed" in result["summary"]


def test_unknown_tool_fails_loudly(tmp_path: Path, settings, db):
    task = {"goal": "x", "plan": [{"tool": "rm_rf", "args": {}}]}
    context = ToolContext(db=db, settings=settings, workspace=tmp_path)
    with pytest.raises(ValueError):
        execute_task(context, task, FakeJobCtx())
