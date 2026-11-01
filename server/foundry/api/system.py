from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import VERSION
from ..db import SCHEMA_VERSION, utc_now
from ..machine import compute_budget, scan_machine
from ..machine.detect import discover_llama_servers, describe_llama_server
from ..security import resolve_within

router = APIRouter()


def _state(request: Request):
    return request.app.state.foundry


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@router.get("/health")
def health(request: Request):
    state = _state(request)
    return {
        "status": "ok",
        "app": "hcs-foundry",
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "time": utc_now(),
        "model_loaded": state.server.status().get("status") == "ready",
    }


class PairRequest(BaseModel):
    label: str = Field(default="local ui", max_length=64)
    code: str | None = None


@router.post("/system/pair")
def pair(request: Request, body: PairRequest):
    state = _state(request)
    if state.pairing.require_code:
        if not body.code or not state.pairing.verify_code(body.code):
            raise HTTPException(status_code=403, detail={"code": "invalid_pairing_code", "message": "Pairing code is wrong."})
    token = state.pairing.issue(body.label)
    state.activity("system.pair", "Issued a local pairing token", {"label": body.label})
    return {
        "token": token,
        "label": body.label,
        "created_at": utc_now(),
        "note": "Store this in memory only. It is never embedded in the frontend bundle.",
    }


@router.post("/system/revoke")
def revoke(request: Request):
    state = _state(request)
    count = state.pairing.revoke_all()
    state.activity("system.revoke", f"Revoked {count} pairing token(s)")
    return {"revoked": count}


@router.get("/system/tokens")
def tokens(request: Request):
    return {"tokens": _state(request).pairing.public_list()}


@router.get("/system/info")
def info(request: Request):
    state = _state(request)
    profile = state.latest_profile()
    return {
        "app": "HCS Foundry",
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "data_dir": str(state.settings.data_dir),
        "telemetry": state.settings.telemetry,
        "local_only": True,
        "paired_tokens": len([t for t in state.pairing.public_list() if not t["revoked"]]),
        "has_profile": profile is not None,
        "model_loaded": state.server.status().get("status") == "ready",
        "time": utc_now(),
    }


@router.get("/settings")
def get_settings(request: Request):
    state = _state(request)
    overrides = state.db.all_settings()
    return {"effective": state.settings.to_public(), "overrides": overrides}


class SettingsPatch(BaseModel):
    model_roots: list[str] | None = None
    import_roots: list[str] | None = None
    ui_origins: list[str] | None = None
    os_reserve_ratio: float | None = Field(default=None, ge=0.05, le=0.5)
    gpu_headroom_bytes: int | None = Field(default=None, ge=0)
    allow_concurrent_heavy_jobs: bool | None = None
    telemetry: bool | None = None
    llama_server_candidates: list[str] | None = None


@router.put("/settings")
def update_settings(request: Request, body: SettingsPatch):
    state = _state(request)
    if body.telemetry is True:
        raise HTTPException(
            status_code=400,
            detail={"code": "telemetry_not_implemented", "message": "HCS Foundry does not implement telemetry; local-only is the only mode."},
        )
    updates = body.model_dump(exclude_none=True)
    from ..config import apply_overrides

    apply_overrides(state.settings, updates)
    state.settings.ensure_dirs()
    merged = dict(state.db.get_setting("settings") or {})
    merged.update(updates)
    state.db.set_setting("settings", merged)
    state.activity("settings.update", "Settings updated", updates)
    return get_settings(request)


@router.get("/activity")
def activity(request: Request, limit: int = Query(default=80, ge=1, le=500)):
    rows = _state(request).db.query("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,))
    return {
        "items": [
            {"id": r["id"], "ts": r["ts"], "kind": r["kind"], "message": r["message"], "data": json.loads(r["data"] or "{}")}
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------

@router.post("/machine/scan")
def machine_scan(request: Request):
    state = _state(request)
    with state.scan_lock:
        profile = scan_machine(state.settings)
        state.store_profile(profile)
    state.activity("machine.scan", f"Machine scanned: {profile['tier']['label']}")
    return {"profile": profile}


@router.get("/machine/profile")
def machine_profile(request: Request):
    state = _state(request)
    profile = state.latest_profile()
    if profile is None:
        raise HTTPException(status_code=404, detail={"code": "no_profile", "message": "Run a machine scan first."})
    return {"profile": profile, "budget": compute_budget(profile, state.settings).to_public()}


@router.get("/machine/budget")
def machine_budget(request: Request):
    state = _state(request)
    profile = state.latest_profile() or {}
    return compute_budget(profile, state.settings).to_public()


@router.get("/machine/runtimes")
def machine_runtimes(request: Request):
    state = _state(request)
    profile = state.latest_profile() or {}
    return {"runtimes": (profile.get("runtimes") or {}), "binary_in_use": state.server.status().get("binary")}


class RuntimeScanRequest(BaseModel):
    roots: list[str] = Field(min_length=1, max_length=8)


@router.post("/machine/runtimes/scan")
def scan_runtimes(request: Request, body: RuntimeScanRequest):
    state = _state(request)
    roots = [Path(r).expanduser() for r in body.roots]
    found = discover_llama_servers(roots, max_depth=5, time_budget=20)
    state.activity("runtime.scan", f"Found {len(found)} llama-server binaries", {"roots": body.roots})
    return {"candidates": found, "roots": body.roots}


class RuntimeAddRequest(BaseModel):
    path: str


@router.post("/machine/runtimes/add")
def add_runtime(request: Request, body: RuntimeAddRequest):
    state = _state(request)
    path = Path(body.path).expanduser()
    if not path.is_file():
        raise HTTPException(status_code=400, detail={"code": "not_a_file", "message": f"Not a file: {body.path}"})
    info = describe_llama_server(str(path))
    state.settings.llama_server_candidates = [
        p for p in state.settings.llama_server_candidates if str(p) != str(path)
    ] + [path]
    state.db.set_setting("llama_server_candidates", [str(p) for p in state.settings.llama_server_candidates])
    profile = state.latest_profile()
    if profile:
        runtimes = profile.setdefault("runtimes", {}).setdefault("llama_server", {})
        candidates = runtimes.setdefault("candidates", [])
        if not any(c.get("path") == str(path) for c in candidates):
            candidates.append(info)
        state.store_profile(profile)
    state.activity("runtime.add", f"Registered runtime {path.name}", {"path": str(path), "status": info["status"]})
    return {"candidate": info}


# ---------------------------------------------------------------------------
# Event stream
# ---------------------------------------------------------------------------

@router.get("/events")
async def events(request: Request):
    state = _state(request)
    bus = state.bus

    async def stream():
        sub_id, queue = bus.subscribe()
        try:
            yield "retry: 3000\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            bus.unsubscribe(sub_id)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Runs (unified job history)
# ---------------------------------------------------------------------------

@router.get("/runs")
def runs(request: Request, limit: int = Query(default=50, ge=1, le=300), type: str | None = None):
    return {"runs": _state(request).runner.list(limit=limit, type_prefix=type)}


@router.get("/runs/{job_id}")
def run_detail(request: Request, job_id: str):
    state = _state(request)
    job = state.runner.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Run not found."})
    return {"run": job, "events": state.runner.events(job_id, limit=300)}


@router.post("/runs/{job_id}/cancel")
def cancel_run(request: Request, job_id: str):
    state = _state(request)
    if not state.runner.get(job_id):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Run not found."})
    ok = state.runner.cancel(job_id)
    return {"cancelled": ok}
