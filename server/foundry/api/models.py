from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from ..db import utc_now
from ..library import (
    get_model,
    list_models,
    model_fit,
    remove_model,
    scan_models,
    update_model,
    verify_model,
)
from ..machine import compute_budget
from ..runtime.llama_server import LoadConfig, RuntimeFailure

router = APIRouter()


def _state(request: Request):
    return request.app.state.foundry


def _pick_binary(state, requested: str | None) -> str:
    if requested:
        path = Path(requested).expanduser()
        if not path.is_file():
            raise HTTPException(status_code=400, detail={"code": "binary_missing", "message": f"Runtime binary not found: {requested}"})
        return str(path)
    profile = state.latest_profile() or {}
    candidates = ((profile.get("runtimes") or {}).get("llama_server") or {}).get("candidates") or []
    verified = [c for c in candidates if c.get("status") == "verified"]
    if not verified:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "no_runtime",
                "message": "No verified llama-server binary. Scan for runtimes on the Machine screen first.",
            },
        )
    accelerators = profile.get("accelerators") or {}
    preferred = None
    for accel in ("cuda", "vulkan", "rocm", "metal"):
        if (accelerators.get(accel) or {}).get("status") == "detected":
            match = [c for c in verified if c.get("accelerator") == accel]
            if match:
                preferred = match[0]
                break
    chosen = preferred or verified[0]
    return chosen["path"]


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

@router.get("/models")
def models(request: Request):
    state = _state(request)
    items = list_models(state.db)
    budget = compute_budget(state.latest_profile() or {}, state.settings)
    for item in items:
        default_ctx = min(int(item.get("context_max") or 4096), 4096)
        item["default_fit"] = model_fit(item, default_ctx, budget)
    return {"models": items, "roots": [str(p) for p in state.settings.model_roots]}


class ScanRequest(BaseModel):
    roots: list[str] | None = None


@router.post("/models/scan")
def scan(request: Request, body: ScanRequest):
    state = _state(request)

    def job(ctx):
        roots = [Path(r).expanduser() for r in body.roots] if body.roots else state.settings.model_roots
        ctx.progress(2, "Walking model roots")
        result = scan_models(
            state.db,
            state.settings,
            roots,
            progress=lambda pct, msg: ctx.progress(pct * 0.95, msg),
            cancel=ctx.cancel_event,
        )
        ctx.progress(100, f"{result['discovered']} GGUF files found")
        return result

    job_id = state.runner.submit("models.scan", job, input_data={"roots": body.roots}, heavy=True)
    return {"job_id": job_id}


@router.get("/models/{model_id}")
def model_detail(request: Request, model_id: str):
    state = _state(request)
    model = get_model(state.db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Model not found."})
    budget = compute_budget(state.latest_profile() or {}, state.settings)
    fits = {str(ctx): model_fit(model, ctx, budget) for ctx in (2048, 4096, 8192, 16384)}
    return {"model": model, "fits": fits, "budget": budget.to_public()}


class ModelPatch(BaseModel):
    name: str | None = None
    license: str | None = None
    source_url: str | None = None
    task: str | None = None


@router.patch("/models/{model_id}")
def patch_model(request: Request, model_id: str, body: ModelPatch):
    state = _state(request)
    try:
        return {"model": update_model(state.db, model_id, **body.model_dump())}
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Model not found."})


@router.delete("/models/{model_id}")
def delete_model(request: Request, model_id: str):
    state = _state(request)
    if state.server.status().get("model_id") == model_id:
        raise HTTPException(status_code=409, detail={"code": "model_loaded", "message": "Unload the model before removing it."})
    remove_model(state.db, model_id)
    return {"removed": model_id, "file_deleted": False}


@router.post("/models/{model_id}/verify")
def verify(request: Request, model_id: str):
    state = _state(request)
    if not get_model(state.db, model_id):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Model not found."})

    def job(ctx):
        ctx.log("Hashing the model file (this can take a while)")
        result = verify_model(
            state.db,
            model_id,
            progress=lambda pct: ctx.progress(pct, "Hashing…"),
            cancel=ctx.cancel_event,
        )
        return result

    job_id = state.runner.submit("models.verify", job, input_data={"model_id": model_id}, heavy=True)
    return {"job_id": job_id}


@router.get("/models/{model_id}/fit")
def fit(request: Request, model_id: str, context: int = Query(default=4096, ge=256, le=1048576)):
    state = _state(request)
    model = get_model(state.db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Model not found."})
    budget = compute_budget(state.latest_profile() or {}, state.settings)
    return {"fit": model_fit(model, context, budget), "budget": budget.to_public()}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    model_id: str
    context: int = Field(default=4096, ge=256, le=1048576)
    gpu_layers: int = Field(default=99, ge=0, le=1024)
    threads: int = Field(default=0, ge=0, le=128)
    batch: int = Field(default=2048, ge=32, le=8192)
    ubatch: int = Field(default=512, ge=16, le=4096)
    flash_attn: str = Field(default="auto", pattern="^(auto|on|off)$")
    kv_type_k: str = Field(default="f16")
    kv_type_v: str = Field(default="f16")
    mlock: bool = False
    binary_path: str | None = None
    force: bool = False


@router.post("/inference/load")
def load(request: Request, body: LoadRequest):
    state = _state(request)
    if state.server.status().get("status") in {"ready", "starting"}:
        raise HTTPException(status_code=409, detail={"code": "already_loaded", "message": "A model is already loaded."})

    model = get_model(state.db, body.model_id)
    if not model:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Model not found."})
    if not model["exists"]:
        raise HTTPException(status_code=410, detail={"code": "file_missing", "message": "The model file no longer exists on disk."})

    profile = state.latest_profile() or {}
    budget = compute_budget(profile, state.settings)
    fit = model_fit(model, body.context, budget)
    if fit["status"] == "too_large" and not body.force:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "does_not_fit",
                "message": "This configuration exceeds the safe memory budget on this machine.",
                "fit": fit,
            },
        )

    binary = _pick_binary(state, body.binary_path)
    config = LoadConfig(
        context=body.context,
        gpu_layers=body.gpu_layers,
        threads=body.threads,
        batch=body.batch,
        ubatch=body.ubatch,
        flash_attn=body.flash_attn,
        kv_type_k=body.kv_type_k,
        kv_type_v=body.kv_type_v,
        mlock=body.mlock,
    )

    def log_sink(line: str) -> None:
        state.bus.publish({"type": "inference_log", "line": line})

    try:
        status = state.server.load(
            model_id=model["id"],
            model_path=model["path"],
            binary=binary,
            config=config,
            alias=model["name"],
            on_log=log_sink,
        )
    except RuntimeFailure as exc:
        raise HTTPException(status_code=500, detail={"code": "load_failed", "message": str(exc), "log_tail": state.server.logs(60)})

    profile_id = f"rt_{model['id'][4:16]}_{body.context}"
    db = state.db
    db.execute(
        "INSERT INTO runtime_profiles(id, model_id, name, config, created_at, last_used_at) "
        "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET config=excluded.config, last_used_at=excluded.last_used_at",
        (
            profile_id,
            model["id"],
            f"{model['name']} @ {body.context}",
            json.dumps(config.to_public()),
            utc_now(),
            utc_now(),
        ),
    )
    state.activity(
        "inference.load",
        f"Loaded {model['name']} (ctx {body.context}, ngl {body.gpu_layers})",
        {"model_id": model["id"], "binary": binary},
    )
    return {"status": status, "fit": fit}


@router.post("/inference/unload")
def unload(request: Request):
    state = _state(request)
    status = state.server.unload()
    state.activity("inference.unload", "Model unloaded")
    return {"status": status}


@router.get("/inference/status")
def status(request: Request):
    state = _state(request)
    data = state.server.status()
    if data.get("status") == "ready":
        data["metrics"] = state.server.metrics()
        try:
            data["props"] = state.server.props()
        except Exception:  # noqa: BLE001 - props are informational
            data["props"] = None
        model = get_model(state.db, data.get("model_id") or "")
        data["model"] = {
            "id": model["id"],
            "name": model["name"],
            "quantization": model["quantization"],
            "context_max": model["context_max"],
            "size_bytes": model["size_bytes"],
        } if model else None
    return data


@router.get("/inference/logs")
def logs(request: Request, limit: int = Query(default=200, ge=1, le=2000)):
    return {"lines": _state(request).server.logs(limit)}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)


@router.post("/inference/chat")
async def chat(request: Request, body: ChatRequest):
    state = _state(request)
    if state.server.status().get("status") != "ready":
        raise HTTPException(status_code=409, detail={"code": "no_model", "message": "Load a model first."})

    messages = [m.model_dump() for m in body.messages]
    generator = state.server.chat(
        messages,
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
    )

    async def stream():
        async for event in iterate_in_threadpool(generator):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/inference/cancel")
def cancel(request: Request):
    _state(request).server.cancel()
    return {"cancelled": True}


@router.get("/inference/metrics")
def metrics(request: Request):
    state = _state(request)
    return {"metrics": state.server.metrics(), "status": state.server.status().get("status")}
