"""Agent workspace: visible plans, typed allowlisted tools, approval queue.

There is no generic shell. Every tool is a named function with typed arguments and
an explicit mutating flag. The planner is deterministic so the user can see exactly
what will happen before it happens.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Settings
from .db import Database, new_id, utc_now

MAX_ATTEMPTS = 2


@dataclass
class ToolContext:
    db: Database
    settings: Settings
    server: object | None = None
    profile: dict | None = None
    workspace: Path | None = None
    runner: object | None = None


@dataclass
class Tool:
    name: str
    description: str
    mutating: bool
    run: Callable[[ToolContext, dict], dict]
    args_schema: dict = field(default_factory=dict)


def _tool_machine_scan(ctx: ToolContext, args: dict) -> dict:
    from .machine import scan_machine

    profile = scan_machine(ctx.settings)
    ctx.db.insert(
        "machine_profiles",
        {"id": new_id("mach"), "created_at": utc_now(), "payload": json.dumps(profile)},
    )
    return {"tier": profile["tier"], "warnings": profile["warnings"]}


def _tool_scan_models(ctx: ToolContext, args: dict) -> dict:
    from .library import scan_models

    result = scan_models(ctx.db, ctx.settings)
    return {k: result[k] for k in ("discovered", "added", "updated", "invalid")}


def _tool_list_datasets(ctx: ToolContext, args: dict) -> dict:
    rows = ctx.db.query("SELECT id, name, status, record_count, rights_status FROM datasets ORDER BY created_at DESC")
    return {"datasets": [dict(row) for row in rows]}


def _tool_dataset_report(ctx: ToolContext, args: dict) -> dict:
    from .data.pipeline import get_dataset

    dataset = get_dataset(ctx.db, args["dataset_id"])
    if not dataset:
        raise KeyError(args["dataset_id"])
    return {
        "id": dataset["id"],
        "name": dataset["name"],
        "status": dataset["status"],
        "stats": dataset["stats"],
        "manifest": dataset["manifest"],
    }


def _tool_list_runs(ctx: ToolContext, args: dict) -> dict:
    rows = ctx.db.query(
        "SELECT id, suite_id, model_id, status, created_at FROM benchmark_runs ORDER BY created_at DESC LIMIT 20"
    )
    return {"runs": [dict(row) for row in rows]}


def _tool_analyze_failures(ctx: ToolContext, args: dict) -> dict:
    from .upgrade import analyze_run

    return analyze_run(ctx.db, args["run_id"])


def _tool_run_benchmark(ctx: ToolContext, args: dict) -> dict:
    from .benchmarks import run_suite

    if ctx.server is None:
        raise RuntimeError("no model is loaded")
    result = run_suite(ctx.db, ctx.server, args["suite_id"], profile=ctx.profile)
    return {"run_id": result["run_id"], "score": result["results"]["aggregate"]["score"]}


def _tool_list_files(ctx: ToolContext, args: dict) -> dict:
    root = (ctx.workspace or Path.cwd()).resolve()
    limit = int(args.get("limit", 200))
    entries = []
    for path in sorted(root.rglob("*")):
        if len(entries) >= limit:
            break
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if any(part in {"node_modules", "__pycache__", "dist", ".venv"} for part in rel.parts):
            continue
        if path.is_file():
            entries.append({"path": str(rel).replace("\\", "/"), "size": path.stat().st_size})
    return {"root": str(root), "files": entries}


def _tool_read_file(ctx: ToolContext, args: dict) -> dict:
    root = (ctx.workspace or Path.cwd()).resolve()
    target = (root / args["path"]).resolve()
    if root not in target.parents and target != root:
        raise ValueError("path escapes the workspace")
    if not target.is_file():
        raise FileNotFoundError(args["path"])
    if target.stat().st_size > 400_000:
        raise ValueError("file too large for the agent to read")
    return {"path": args["path"], "content": target.read_text("utf-8", errors="replace")[:20000]}


def _git(ctx: ToolContext, *args: str) -> dict:
    root = ctx.workspace or Path.cwd()
    proc = subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=0x08000000 if __import__("sys").platform == "win32" else 0,
    )
    return {"ok": proc.returncode == 0, "output": (proc.stdout or proc.stderr)[:8000]}


def _tool_git_status(ctx: ToolContext, args: dict) -> dict:
    return _git(ctx, "status", "--short")


def _tool_git_diff(ctx: ToolContext, args: dict) -> dict:
    return _git(ctx, "diff", "--stat")


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("machine_scan", "Re-scan the host hardware and runtimes.", False, _tool_machine_scan),
        Tool("scan_models", "Scan configured model roots for GGUF files.", False, _tool_scan_models),
        Tool("list_datasets", "List datasets with status and record counts.", False, _tool_list_datasets),
        Tool("dataset_report", "Summarise one dataset.", False, _tool_dataset_report, {"dataset_id": "str"}),
        Tool("list_benchmark_runs", "List recent benchmark runs.", False, _tool_list_runs),
        Tool("analyze_failures", "Classify failures of a benchmark run.", False, _tool_analyze_failures, {"run_id": "str"}),
        Tool(
            "run_benchmark",
            "Run a benchmark suite against the loaded model.",
            True,
            _tool_run_benchmark,
            {"suite_id": "str"},
        ),
        Tool("list_files", "List workspace files (read-only).", False, _tool_list_files),
        Tool("read_file", "Read a workspace file (read-only).", False, _tool_read_file, {"path": "str"}),
        Tool("git_status", "Show git status of the workspace.", False, _tool_git_status),
        Tool("git_diff", "Show a git diff summary.", False, _tool_git_diff),
    ]
}

PLAN_RULES = [
    (
        ("run benchmark", "run a benchmark", "execute benchmark", "benchmark the model", "benchmark this"),
        [{"tool": "run_benchmark", "args": {"suite_id": "instruction_adherence"}}],
    ),
    (("hardware", "machine", "gpu", "scan"), [{"tool": "machine_scan", "args": {}}]),
    (("model", "gguf", "library"), [{"tool": "scan_models", "args": {}}]),
    (("dataset", "data", "session", "import"), [{"tool": "list_datasets", "args": {}}]),
    (("benchmark", "score", "compare"), [{"tool": "list_benchmark_runs", "args": {}}]),
    (("failure", "improve", "upgrade", "regression"), [{"tool": "list_benchmark_runs", "args": {}}]),
    (("git", "diff", "repository"), [{"tool": "git_status", "args": {}}, {"tool": "git_diff", "args": {}}]),
]


def plan(goal: str, autonomy: str = "supervised") -> dict:
    lower = goal.lower()
    steps: list[dict] = []
    for keywords, actions in PLAN_RULES:
        if any(keyword in lower for keyword in keywords):
            for action in actions:
                tool = TOOLS[action["tool"]]
                steps.append(
                    {
                        "tool": tool.name,
                        "args": action["args"],
                        "mutating": tool.mutating,
                        "description": tool.description,
                        "requires_approval": tool.mutating and autonomy != "trusted",
                    }
                )
    if not steps:
        steps = [
            {
                "tool": "list_files",
                "args": {"limit": 100},
                "mutating": False,
                "description": TOOLS["list_files"].description,
                "requires_approval": False,
            },
            {
                "tool": "machine_scan",
                "args": {},
                "mutating": False,
                "description": TOOLS["machine_scan"].description,
                "requires_approval": False,
            },
        ]
    needs_approval = any(step["requires_approval"] for step in steps)
    return {
        "goal": goal,
        "autonomy": autonomy,
        "steps": steps,
        "needs_approval": needs_approval,
        "tools_available": sorted(TOOLS),
        "risks": [
            "Read-only inspection of the workspace and local configuration.",
            *(["A mutating step (benchmark execution) is included and needs approval."] if needs_approval else []),
        ],
    }


def create_task(db: Database, goal: str, autonomy: str = "supervised") -> dict:
    if autonomy not in {"suggest", "supervised", "trusted"}:
        raise ValueError("invalid autonomy level")
    task_plan = plan(goal, autonomy)
    status = "suggested" if autonomy == "suggest" else ("awaiting_approval" if task_plan["needs_approval"] else "queued")
    task_id = new_id("agent")
    db.insert(
        "agent_tasks",
        {
            "id": task_id,
            "goal": goal,
            "autonomy": autonomy,
            "status": status,
            "plan": json.dumps(task_plan["steps"]),
            "result": json.dumps({"plan": task_plan}),
            "created_at": utc_now(),
        },
    )
    db.activity("agent.create", f"Agent task created: {goal[:80]}", {"task_id": task_id, "status": status})
    return get_task(db, task_id) or {}


def get_task(db: Database, task_id: str) -> dict | None:
    row = db.query_one("SELECT * FROM agent_tasks WHERE id=?", (task_id,))
    if not row:
        return None
    data = dict(row)
    data["plan"] = json.loads(data.get("plan") or "[]")
    data["result"] = json.loads(data.get("result") or "{}")
    return data


def list_tasks(db: Database, limit: int = 50) -> list[dict]:
    rows = db.query("SELECT * FROM agent_tasks ORDER BY created_at DESC LIMIT ?", (limit,))
    out = []
    for row in rows:
        data = dict(row)
        data["plan"] = json.loads(data.get("plan") or "[]")
        data["result"] = json.loads(data.get("result") or "{}")
        out.append(data)
    return out


def execute_task(ctx: ToolContext, task: dict, job_ctx) -> dict:
    """Run the planned steps with a bounded repair loop."""
    steps = task["plan"]
    results: list[dict] = []
    job_ctx.log(f"Executing {len(steps)} step(s) for: {task['goal'][:120]}")
    for index, step in enumerate(steps):
        if job_ctx.cancelled:
            raise InterruptedError("agent task cancelled")
        tool = TOOLS.get(step["tool"])
        if tool is None:
            raise ValueError(f"unknown tool {step['tool']}")
        job_ctx.progress(index / max(1, len(steps)) * 100, f"{tool.name} ({index + 1}/{len(steps)})")
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                output = tool.run(ctx, step.get("args") or {})
                results.append(
                    {"tool": tool.name, "attempt": attempt, "status": "ok", "output": output}
                )
                job_ctx.emit("tool_result", f"{tool.name} ok", {"tool": tool.name, "attempt": attempt})
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - agent step boundary
                last_error = str(exc)
                job_ctx.emit(
                    "tool_error",
                    f"{tool.name} failed (attempt {attempt}): {exc}",
                    {"tool": tool.name, "attempt": attempt},
                    level="error",
                )
        if last_error is not None:
            raise RuntimeError(f"step {tool.name} failed after {MAX_ATTEMPTS} attempts: {last_error}")

    summary = _summarise(task["goal"], results)
    job_ctx.log(summary)
    return {"steps": results, "summary": summary, "completed_at": utc_now()}


def _summarise(goal: str, results: list[dict]) -> str:
    parts = [f"Goal: {goal}"]
    for item in results:
        output = item.get("output") or {}
        if item["tool"] == "machine_scan":
            parts.append(f"Machine tier: {output.get('tier', {}).get('label')} ({output.get('tier', {}).get('confidence')})")
        elif item["tool"] == "scan_models":
            parts.append(f"Models: {output.get('discovered')} discovered, {output.get('added')} new")
        elif item["tool"] == "list_datasets":
            parts.append(f"Datasets: {len(output.get('datasets', []))}")
        elif item["tool"] == "list_benchmark_runs":
            parts.append(f"Benchmark runs: {len(output.get('runs', []))}")
        elif item["tool"] == "list_files":
            parts.append(f"Workspace files listed: {len(output.get('files', []))}")
        elif item["tool"] == "git_status":
            parts.append("Git status captured")
        elif item["tool"] == "git_diff":
            parts.append("Git diff summary captured")
        else:
            parts.append(f"{item['tool']}: ok")
    return " · ".join(parts)
