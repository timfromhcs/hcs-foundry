from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .. import gateway
from ..security import resolve_within

router = APIRouter()


def _state(request: Request):
    return request.app.state.foundry


def _roots(state) -> list[Path]:
    return list(state.settings.model_roots) + [state.settings.data_dir]


def _resolve(state, raw: str) -> Path:
    try:
        return resolve_within(raw, _roots(state))
    except ValueError:
        raise HTTPException(status_code=400, detail={"code": "path_not_allowed", "message": "Path is outside the allowed roots."})


def _model_path(state, model_id: str | None, path: str | None) -> Path:
    from ..library import get_model

    if model_id:
        model = get_model(state.db, model_id)
        if not model:
            raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Model not found."})
        return Path(model["path"]).expanduser()
    if path:
        return _resolve(state, path)
    raise HTTPException(status_code=400, detail={"code": "missing_path", "message": "Provide model_id or path."})


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------

@router.get("/gguf/tools")
def gguf_tools(request: Request):
    from ..gguf_tools import discover

    state = _state(request)
    toolchain = discover(state.settings)
    return {"toolchain": toolchain.to_public(), "ready": bool(toolchain.converter and toolchain.quantize)}


class ExportRequest(BaseModel):
    model_id: str | None = None
    path: str | None = None
    dtype: str = Field(default="f16", pattern="^(f16|bf16|f32)$")


@router.post("/gguf/export")
def gguf_export(request: Request, body: ExportRequest):
    from ..gguf_tools import discover, export_gguf

    state = _state(request)
    source = _model_path(state, body.model_id, body.path)
    toolchain = discover(state.settings)
    out = state.settings.data_dir / "gguf" / f"{source.name}-{body.dtype}.gguf"

    def job(ctx):
        ctx.log(f"exporting {source} → {out}")
        result = export_gguf(source, out, toolchain=toolchain, dtype=body.dtype, log=ctx.log, cancel=ctx.cancel_event)
        ctx.artifact("gguf", result["output"], size=result["size_bytes"])
        return result

    return {"job_id": state.runner.submit("gguf.export", job, input_data={"path": str(source)}, heavy=True), "output": str(out)}


class QuantizeRequest(BaseModel):
    source: str
    method: str = "Q4_K_M"


@router.post("/gguf/quantize")
def gguf_quantize(request: Request, body: QuantizeRequest):
    from ..gguf_tools import QUANT_METHODS, discover, quantize_gguf

    state = _state(request)
    source = _resolve(state, body.source)
    if not source.is_file():
        raise HTTPException(status_code=400, detail={"code": "not_a_file", "message": "source must be a GGUF file"})
    method = body.method.upper()
    if method not in QUANT_METHODS:
        raise HTTPException(status_code=400, detail={"code": "bad_method", "message": f"Use one of {QUANT_METHODS}"})
    toolchain = discover(state.settings)
    out = source.with_name(f"{source.stem}-{method}.gguf")

    def job(ctx):
        result = quantize_gguf(source, out, method, toolchain=toolchain, log=ctx.log, cancel=ctx.cancel_event)
        ctx.artifact("gguf", result["output"], size=result["size_bytes"])
        return result

    return {"job_id": state.runner.submit("gguf.quantize", job, input_data={"source": str(source), "method": method}, heavy=True), "output": str(out)}


class DequantizeRequest(BaseModel):
    source: str
    method: str = Field(default="F16", pattern="^(F16|BF16)$")


@router.post("/gguf/dequantize")
def gguf_dequantize(request: Request, body: DequantizeRequest):
    from ..gguf_tools import discover, requantize

    state = _state(request)
    source = _resolve(state, body.source)
    toolchain = discover(state.settings)
    out = source.with_name(f"{source.stem}-{body.method}.gguf")

    def job(ctx):
        result = requantize(source, out, toolchain=toolchain, method=body.method, log=ctx.log, cancel=ctx.cancel_event)
        return result

    return {"job_id": state.runner.submit("gguf.dequantize", job, input_data={"source": str(source)}, heavy=True), "output": str(out)}


@router.get("/gguf/inspect")
def gguf_inspect(request: Request, file: str = Query(...)):
    from ..runtime.gguf import read_gguf

    state = _state(request)
    path = _resolve(state, file)
    try:
        return read_gguf(path).to_public()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail={"code": "gguf_error", "message": str(exc)})


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class MoERequest(BaseModel):
    model_id: str | None = None
    path: str | None = None
    experts: int = Field(default=4, ge=2, le=16)
    top_k: int = Field(default=2, ge=1, le=16)
    noise: float = Field(default=0.01, ge=0, le=0.5)
    force: bool = False


@router.post("/moe/build")
def moe_build(request: Request, body: MoERequest):
    from ..moe import MoEConfig, moeify

    state = _state(request)
    source = _model_path(state, body.model_id, body.path)
    config = MoEConfig(experts=body.experts, top_k=body.top_k, noise=body.noise)
    out = source.parent / f"{source.name}-moe"

    def job(ctx):
        result = moeify(source, out, config=config, progress=lambda pct, msg: ctx.progress(pct, msg), log=ctx.log, force=body.force)
        ctx.artifact("moe", result["output"], experts=body.experts)
        return result

    return {"job_id": state.runner.submit("moe.build", job, input_data={"path": str(source), "experts": body.experts}, heavy=True), "output": str(out)}


# ---------------------------------------------------------------------------
# Vulkan tuning
# ---------------------------------------------------------------------------

class TuneRequest(BaseModel):
    model_id: str
    ngl: list[int] = Field(default_factory=lambda: [0, 16, 99])
    context: int = Field(default=2048, ge=256, le=8192)


@router.post("/tune/vulkan")
def tune_vulkan(request: Request, body: TuneRequest):
    from ..library import get_model
    from ..tuning import sweep

    state = _state(request)
    model = get_model(state.db, body.model_id)
    if not model:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Model not found."})
    profile = state.latest_profile() or {}
    candidates = ((profile.get("runtimes") or {}).get("llama_server") or {}).get("candidates") or []
    verified = [c for c in candidates if c.get("status") == "verified"]
    if not verified:
        raise HTTPException(status_code=409, detail={"code": "no_runtime", "message": "No verified llama-server binary."})
    binary = verified[0]["path"]

    def job(ctx):
        result = sweep(
            state.db, state.server, model=model, binary=binary, context=body.context,
            gpu_layers=body.ngl, progress=lambda pct, msg: ctx.progress(pct, msg), cancel=ctx.cancel_event,
        )
        return result

    return {"job_id": state.runner.submit("tune.vulkan", job, input_data={"model_id": body.model_id}, heavy=True)}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@router.get("/sources")
def sources(request: Request):
    from ..data.sources import scan

    return {"sources": scan(_state(request).settings)}


class SourcesImportRequest(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["claude_code", "codex", "antigravity", "opencode"])
    rights_status: str = Field(default="needs_review", pattern="^(needs_review|confirmed|restricted|excluded)$")


@router.post("/sources/import")
def sources_import(request: Request, body: SourcesImportRequest):
    from ..data.sources import import_all

    state = _state(request)

    def job(ctx):
        result = import_all(
            state.db, state.settings, body.sources, rights_status=body.rights_status,
            progress=lambda pct, msg: ctx.progress(pct, msg), cancel=ctx.cancel_event,
        )
        return result

    return {"job_id": state.runner.submit("sources.import", job, input_data={"sources": body.sources}, heavy=True)}


# ---------------------------------------------------------------------------
# Training algorithms / playbook
# ---------------------------------------------------------------------------

class AutoRequest(BaseModel):
    dataset_id: str
    model_path: str
    max_steps: int = Field(default=20, ge=1, le=5000)
    max_seq_len: int = Field(default=256, ge=32, le=4096)
    micro_batch: int = Field(default=1, ge=1, le=16)
    grad_accum: int = Field(default=4, ge=1, le=128)
    lora_r: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=2e-4, gt=0, le=1)
    eval_every: int = Field(default=5, ge=0, le=500)
    patience: int = Field(default=3, ge=1, le=50)
    algorithm: str = Field(default="auto", pattern="^(auto|dpo|distill|int4)$")
    teacher: str = Field(default="local", pattern="^(local|gateway)$")


@router.post("/training/auto")
def training_auto(request: Request, body: AutoRequest):
    from ..training import SFTConfig, run_auto, run_distill, run_dpo

    state = _state(request)
    config = SFTConfig(
        model_path=body.model_path, dataset_id=body.dataset_id, max_steps=body.max_steps,
        max_seq_len=body.max_seq_len, micro_batch=body.micro_batch, grad_accum=body.grad_accum,
        lora_r=body.lora_r, learning_rate=body.learning_rate, auto=True,
        eval_every=body.eval_every, patience=body.patience,
    )
    output = state.settings.training_dir / f"api_{int(time.time())}"

    def job(ctx):
        if body.algorithm == "int4":
            from ..training import create_int4_run

            return create_int4_run(
                state.db, config, job_id=ctx.job_id, output_dir=output,
                progress=lambda pct, msg: ctx.progress(pct, msg), log=ctx.log, cancel=ctx.cancel_event,
            )
        if body.algorithm == "dpo":
            return run_dpo(state.db, config, output_dir=output, progress=lambda pct, msg: ctx.progress(pct, msg), log=ctx.log, cancel=ctx.cancel_event)
        if body.algorithm == "distill":
            prompts = _prompts(state, body.dataset_id, 8)
            if body.teacher == "gateway":
                cfg = gateway.get_config(state.settings.state_dir)
                if not (cfg["configured"] and cfg["enabled"]):
                    raise RuntimeError("gateway teacher is not configured/enabled")
                teacher = gateway.GatewayTeacher(cfg["key"], cfg["model"])
            else:
                if state.server.status().get("status") != "ready":
                    raise RuntimeError("load a local model or use the gateway teacher")
                teacher = state.server

            def complete(messages):
                result = teacher.complete(messages, temperature=0.3, max_tokens=256)
                return result.get("text") or result.get("reasoning") or ""

            return run_distill(
                state.db, config, teacher_complete=complete, prompts=prompts,
                job_id=ctx.job_id, output_dir=output, progress=lambda pct, msg: ctx.progress(pct, msg), log=ctx.log, cancel=ctx.cancel_event,
            )
        return run_auto(
            state.db, config, job_id=ctx.job_id, output_dir=output,
            progress=lambda pct, msg: ctx.progress(pct, msg), log=ctx.log, cancel=ctx.cancel_event,
        )

    return {"job_id": state.runner.submit("training.auto", job, input_data={"dataset_id": body.dataset_id, "algorithm": body.algorithm}, heavy=True)}


def _prompts(state, dataset_id: str, limit: int) -> list[str]:
    rows = state.db.query(
        "SELECT payload FROM dataset_records WHERE dataset_id=? AND split='train' AND excluded=0 LIMIT ?",
        (dataset_id, limit),
    )
    import json

    prompts = []
    for row in rows:
        payload = json.loads(row["payload"])
        request = ((payload.get("task") or {}).get("user_request") or "").strip()
        if request:
            prompts.append(request)
    return prompts


@router.get("/training/playbook")
def training_playbook(request: Request):
    from ..training import playbook

    state = _state(request)
    return {"entries": playbook.entries(state.db)}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    stages: list[str] = Field(min_length=1, max_length=12)
    options: dict = Field(default_factory=dict)


@router.post("/pipeline/run")
def pipeline_run(request: Request, body: PipelineRequest):
    from ..pipeline import run_pipeline

    state = _state(request)
    allowed = {"pull", "dataset", "train", "merge", "export", "quantize", "moe", "report"}
    unknown = [stage for stage in body.stages if stage not in allowed]
    if unknown:
        raise HTTPException(status_code=400, detail={"code": "unknown_stage", "message": f"Unknown stages: {unknown}"})

    def job(ctx):
        return run_pipeline(
            state.db, state.settings, stages=body.stages, options=dict(body.options),
            progress=lambda pct, msg: ctx.progress(pct, msg), log=ctx.log, cancel=ctx.cancel_event,
        )

    return {"job_id": state.runner.submit("pipeline.run", job, input_data={"stages": body.stages}, heavy=True)}
