from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .. import agent as agent_module
from ..agent import TOOLS, ToolContext, execute_task, get_task, list_tasks
from ..benchmarks import (
    compare_runs,
    export_run,
    get_run as get_benchmark_run,
    list_runs as list_benchmark_runs,
    run_suite,
    suite_catalog,
)
from ..config import repo_root
from ..db import utc_now
from ..training import (
    create_run,
    detect_environment,
    get_run as get_training_run,
    list_runs as list_training_runs,
    preflight,
    start_smoke_run,
)
from ..upgrade import (
    FAILURE_TAXONOMY,
    analyze_run,
    create_candidate,
    evaluate_candidate,
    get_candidate,
    lineage,
    list_candidates,
    promote,
    rollback,
)

router = APIRouter()


def _state(request: Request):
    return request.app.state.foundry


def _require_model(state):
    if state.server.status().get("status") != "ready":
        raise HTTPException(status_code=409, detail={"code": "no_model", "message": "Load a model first."})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@router.get("/training/environment")
def training_environment(request: Request):
    state = _state(request)
    return detect_environment(state.settings, state.latest_profile())


class PreflightRequest(BaseModel):
    dataset_id: str
    base_model: str | None = None
    method: str = Field(default="lora_sft", pattern="^(lora_sft|dpo)$")


@router.post("/training/preflight")
def training_preflight(request: Request, body: PreflightRequest):
    state = _state(request)
    plan = preflight(
        state.db,
        state.settings,
        dataset_id=body.dataset_id,
        base_model=body.base_model,
        method=body.method,
        profile=state.latest_profile(),
    )
    return plan


@router.post("/training/smoke")
def training_smoke(request: Request):
    state = _state(request)

    def job(ctx):
        return start_smoke_run(
            state.db,
            state.settings,
            job_id=ctx.job_id,
            progress=lambda pct, msg: ctx.progress(pct, msg),
            cancel=ctx.cancel_event,
        )

    job_id = state.runner.submit("training.smoke", job, heavy=True)
    return {"job_id": job_id}


@router.get("/training/recipes")
def training_recipes(request: Request):
    from ..training import recipe_catalog

    return {"recipes": recipe_catalog()}


class SFTRequest(BaseModel):
    dataset_id: str
    model_path: str
    recipe_id: str | None = None
    max_steps: int = Field(default=20, ge=1, le=20000)
    max_seq_len: int = Field(default=256, ge=32, le=4096)
    micro_batch: int = Field(default=1, ge=1, le=16)
    grad_accum: int = Field(default=4, ge=1, le=128)
    lora_r: int = Field(default=8, ge=1, le=128)
    lora_alpha: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=2e-4, gt=0, le=1)
    seed: int = Field(default=42, ge=0)


@router.post("/training/project")
def training_project(request: Request, body: SFTRequest):
    from ..training import SFTConfig, estimate_resources

    config = SFTConfig(
        model_path=body.model_path,
        dataset_id=body.dataset_id,
        max_steps=body.max_steps,
        max_seq_len=body.max_seq_len,
        micro_batch=body.micro_batch,
        grad_accum=body.grad_accum,
        lora_r=body.lora_r,
        lora_alpha=body.lora_alpha,
        learning_rate=body.learning_rate,
        seed=body.seed,
    )
    return {"config": config.to_public(), "resources": estimate_resources(body.model_path, config)}


@router.post("/training/sft")
def training_sft(request: Request, body: SFTRequest):
    state = _state(request)
    from ..training import SFTConfig, create_sft_run, estimate_resources

    config = SFTConfig(
        model_path=body.model_path,
        dataset_id=body.dataset_id,
        max_steps=body.max_steps,
        max_seq_len=body.max_seq_len,
        micro_batch=body.micro_batch,
        grad_accum=body.grad_accum,
        lora_r=body.lora_r,
        lora_alpha=body.lora_alpha,
        learning_rate=body.learning_rate,
        seed=body.seed,
    )
    resources = estimate_resources(body.model_path, config)
    if not resources["fits"]:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "does_not_fit",
                "message": "The estimated memory exceeds available RAM. Lower sequence length or batch size.",
                "resources": resources,
            },
        )
    output_dir = state.settings.training_dir / f"run_{int(time.time())}"

    def job(ctx):
        result = create_sft_run(
            state.db,
            config,
            job_id=ctx.job_id,
            output_dir=output_dir,
            progress=lambda pct, msg: ctx.progress(pct, msg),
            log=lambda message: ctx.log(message),
            cancel=ctx.cancel_event,
        )
        ctx.artifact("adapter", result["manifest"]["adapter_dir"])
        return result

    job_id = state.runner.submit("training.sft", job, input_data={"dataset_id": body.dataset_id, "model": body.model_path}, heavy=True)
    return {"job_id": job_id, "output_dir": str(output_dir), "resources": resources}


@router.get("/training/runs")
def training_runs(request: Request):
    return {"runs": list_training_runs(_state(request).db)}


@router.get("/training/runs/{run_id}")
def training_run_detail(request: Request, run_id: str):
    run = get_training_run(_state(request).db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Training run not found."})
    return {"run": run}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@router.get("/benchmarks/suites")
def benchmark_suites(request: Request):
    return {"suites": suite_catalog()}


class BenchmarkRequest(BaseModel):
    suite_id: str
    judge: bool = False
    model_id: str | None = None


@router.post("/benchmarks/runs")
def benchmark_start(request: Request, body: BenchmarkRequest):
    state = _state(request)
    _require_model(state)
    if body.suite_id == "ui_regression":
        raise HTTPException(
            status_code=400,
            detail={
                "code": "external_suite",
                "message": "The UI regression suite runs from web/ with Playwright, not through this endpoint.",
            },
        )

    def job(ctx):
        model = None
        if body.model_id:
            from ..library import get_model

            model = get_model(state.db, body.model_id)
        else:
            status = state.server.status()
            if status.get("model_id"):
                from ..library import get_model

                model = get_model(state.db, status["model_id"])
        result = run_suite(
            state.db,
            state.server,
            body.suite_id,
            model=model,
            profile=state.latest_profile(),
            progress=lambda pct, msg: ctx.progress(pct, msg),
            cancel=ctx.cancel_event,
            judge=body.judge,
        )
        return {"run_id": result["run_id"], "score": result["results"]["aggregate"]["score"]}

    job_id = state.runner.submit("benchmarks.run", job, input_data={"suite_id": body.suite_id}, heavy=True)
    return {"job_id": job_id}


@router.get("/benchmarks/runs")
def benchmark_runs(request: Request, limit: int = Query(default=30, ge=1, le=200)):
    return {"runs": list_benchmark_runs(_state(request).db, limit)}


@router.get("/benchmarks/runs/{run_id}")
def benchmark_run_detail(request: Request, run_id: str):
    run = get_benchmark_run(_state(request).db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Benchmark run not found."})
    return {"run": run}


@router.get("/benchmarks/runs/{run_id}/export")
def benchmark_export(request: Request, run_id: str, format: str = Query(default="json", pattern="^(json|csv|md)$")):
    try:
        text, media = export_run(_state(request).db, run_id, format)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Benchmark run not found."})
    return PlainTextResponse(
        text,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{run_id}.{format}"'},
    )


class CompareRequest(BaseModel):
    a: str
    b: str


@router.post("/benchmarks/compare")
def benchmark_compare(request: Request, body: CompareRequest):
    try:
        return compare_runs(_state(request).db, body.a, body.b)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Benchmark run not found."})


# ---------------------------------------------------------------------------
# Upgrade lab
# ---------------------------------------------------------------------------

@router.get("/upgrades/taxonomy")
def upgrade_taxonomy(request: Request):
    return {"taxonomy": FAILURE_TAXONOMY, "active": _state(request).db.get_setting("active_upgrade")}


@router.post("/upgrades/analyze")
def upgrade_analyze(request: Request, run_id: str = Query(...)):
    try:
        return analyze_run(_state(request).db, run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Benchmark run not found."})


class CandidateRequest(BaseModel):
    parent_run_id: str
    dataset_id: str
    kind: str = Field(default="prompt", pattern="^(prompt|runtime|dataset)$")
    note: str | None = Field(default=None, max_length=400)


@router.post("/upgrades/candidates")
def upgrade_create(request: Request, body: CandidateRequest):
    state = _state(request)
    try:
        candidate = create_candidate(
            state.db,
            parent_run_id=body.parent_run_id,
            dataset_id=body.dataset_id,
            kind=body.kind,
            note=body.note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": f"Not found: {exc}"})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "cannot_create", "message": str(exc)})
    return {"candidate": candidate}


@router.get("/upgrades/candidates")
def upgrade_list(request: Request):
    return {"candidates": list_candidates(_state(request).db)}


@router.get("/upgrades/candidates/{candidate_id}")
def upgrade_detail(request: Request, candidate_id: str):
    candidate = get_candidate(_state(request).db, candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Candidate not found."})
    return {"candidate": candidate}


@router.post("/upgrades/candidates/{candidate_id}/evaluate")
def upgrade_evaluate(request: Request, candidate_id: str):
    state = _state(request)
    _require_model(state)
    if not get_candidate(state.db, candidate_id):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Candidate not found."})

    def job(ctx):
        result = evaluate_candidate(
            state.db,
            candidate_id,
            state.server,
            profile=state.latest_profile(),
            progress=lambda pct, msg: ctx.progress(pct, msg),
            cancel=ctx.cancel_event,
        )
        return {"candidate_id": candidate_id, "metrics": result["metrics"], "run_id": result["run_id"]}

    job_id = state.runner.submit("upgrades.evaluate", job, input_data={"candidate_id": candidate_id}, heavy=True)
    return {"job_id": job_id}


class PromoteRequest(BaseModel):
    approver: str = Field(min_length=2, max_length=80)
    threshold: float = Field(default=0.05, ge=0.0, le=1.0)


@router.post("/upgrades/candidates/{candidate_id}/promote")
def upgrade_promote(request: Request, candidate_id: str, body: PromoteRequest):
    state = _state(request)
    try:
        result = promote(state.db, candidate_id, approver=body.approver, threshold=body.threshold)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Candidate not found."})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "cannot_promote", "message": str(exc)})
    return result


@router.post("/upgrades/candidates/{candidate_id}/rollback")
def upgrade_rollback(request: Request, candidate_id: str):
    state = _state(request)
    try:
        return rollback(state.db, candidate_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Candidate not found."})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "cannot_rollback", "message": str(exc)})


@router.get("/upgrades/lineage")
def upgrade_lineage(request: Request):
    return {"lineage": lineage(_state(request).db)}


# ---------------------------------------------------------------------------
# Agent workspace
# ---------------------------------------------------------------------------

@router.get("/agent/tools")
def agent_tools(request: Request):
    return {
        "tools": [
            {"name": tool.name, "description": tool.description, "mutating": tool.mutating, "args": tool.args_schema}
            for tool in TOOLS.values()
        ]
    }


class AgentTaskRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=500)
    autonomy: str = Field(default="supervised", pattern="^(suggest|supervised|trusted)$")


@router.post("/agent/tasks")
def agent_create(request: Request, body: AgentTaskRequest):
    state = _state(request)
    task = agent_module.create_task(state.db, body.goal, body.autonomy)
    if task["status"] == "queued":
        _launch_agent_job(state, task)
    return {"task": get_task(state.db, task["id"])}


def _launch_agent_job(state, task: dict) -> str:
    def job(ctx):
        state.db.execute("UPDATE agent_tasks SET status='running' WHERE id=?", (task["id"],))
        ctx.log(f"Agent started: {task['goal'][:120]}")
        tool_ctx = ToolContext(
            db=state.db,
            settings=state.settings,
            server=state.server,
            profile=state.latest_profile(),
            workspace=repo_root(),
            runner=state.runner,
        )
        try:
            result = execute_task(tool_ctx, task, ctx)
        except Exception as exc:
            state.db.execute(
                "UPDATE agent_tasks SET status='failed', result=?, finished_at=? WHERE id=?",
                (json.dumps({"error": str(exc)}), utc_now(), task["id"]),
            )
            raise
        state.db.execute(
            "UPDATE agent_tasks SET status='completed', result=?, finished_at=? WHERE id=?",
            (json.dumps(result), utc_now(), task["id"]),
        )
        return result

    return state.runner.submit(
        "agent.task",
        job,
        input_data={"goal": task["goal"], "task_id": task["id"]},
        owner="agent",
    )


@router.get("/agent/tasks")
def agent_list(request: Request):
    return {"tasks": list_tasks(_state(request).db)}


@router.get("/agent/tasks/{task_id}")
def agent_detail(request: Request, task_id: str):
    state = _state(request)
    task = get_task(state.db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Agent task not found."})
    job_id = None
    row = state.db.query_one("SELECT id FROM jobs WHERE type='agent.task' AND input LIKE ? ORDER BY created_at DESC LIMIT 1", (f"%{task_id}%",))
    if row:
        job_id = row["id"]
    events = state.runner.events(job_id) if job_id else []
    return {"task": task, "job_id": job_id, "events": events}


@router.post("/agent/tasks/{task_id}/approve")
def agent_approve(request: Request, task_id: str):
    state = _state(request)
    task = get_task(state.db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Agent task not found."})
    if task["status"] not in {"awaiting_approval", "suggested"}:
        raise HTTPException(status_code=409, detail={"code": "not_waiting", "message": "This task is not waiting for approval."})
    state.db.execute("UPDATE agent_tasks SET status='queued' WHERE id=?", (task_id,))
    state.db.insert(
        "approvals",
        {
            "id": f"appr_{task_id[-12:]}",
            "subject": task_id,
            "action": "agent.execute",
            "payload": json.dumps({"goal": task["goal"]}),
            "granted": 1,
            "created_at": utc_now(),
            "decided_at": utc_now(),
        },
    )
    _launch_agent_job(state, task)
    state.activity("agent.approve", "Agent task approved and started", {"task_id": task_id})
    return {"task": get_task(state.db, task_id)}


@router.post("/agent/tasks/{task_id}/cancel")
def agent_cancel(request: Request, task_id: str):
    state = _state(request)
    task = get_task(state.db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Agent task not found."})
    row = state.db.query_one("SELECT id FROM jobs WHERE type='agent.task' AND input LIKE ? ORDER BY created_at DESC LIMIT 1", (f"%{task_id}%",))
    if row:
        state.runner.cancel(row["id"])
    state.db.execute("UPDATE agent_tasks SET status='cancelled', finished_at=? WHERE id=?", (utc_now(), task_id))
    return {"task": get_task(state.db, task_id)}
