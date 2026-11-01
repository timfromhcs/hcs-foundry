from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..data.enrich import annotation_report, enrich_dataset
from ..data.pipeline import (
    ImportOptions,
    delete_dataset,
    export_jsonl,
    freeze_dataset,
    get_dataset,
    import_dataset,
    list_datasets,
    list_records,
    preview,
    set_dataset_rights,
    set_record_excluded,
)
from ..identity import (
    approve_profile,
    create_dataset as create_identity_dataset,
    create_profile,
    evaluate_identity,
    get_profile,
    list_profiles,
    update_card,
)

router = APIRouter()


def _state(request: Request):
    return request.app.state.foundry


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=200)
    provider: str = "auto"


@router.post("/datasets/preview")
def preview_import(request: Request, body: PreviewRequest):
    state = _state(request)
    try:
        return preview(body.paths, state.settings.import_roots, body.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "invalid_path", "message": str(exc)})


class ImportRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    paths: list[str] = Field(min_length=1, max_length=200)
    provider: str = "auto"
    rights_status: str = "needs_review"
    redact: bool = True
    dedupe: bool = True
    task_type_override: str | None = None


@router.post("/datasets/import")
def start_import(request: Request, body: ImportRequest):
    state = _state(request)
    if body.rights_status not in {"needs_review", "confirmed", "restricted", "excluded"}:
        raise HTTPException(status_code=400, detail={"code": "invalid_rights", "message": "Unknown rights status."})

    options = ImportOptions(
        name=body.name,
        provider=body.provider,
        rights_status=body.rights_status,
        do_redact=body.redact,
        do_dedupe=body.dedupe,
        task_type_override=body.task_type_override,
    )

    def job(ctx):
        return import_dataset(
            state.db,
            paths=body.paths,
            import_roots=state.settings.import_roots,
            options=options,
            progress=lambda pct, msg: ctx.progress(pct, msg),
            cancel=ctx.cancel_event,
        )

    job_id = state.runner.submit("datasets.import", job, input_data={"name": body.name, "files": len(body.paths)}, heavy=True)
    return {"job_id": job_id}


@router.get("/datasets")
def datasets(request: Request):
    return {"datasets": list_datasets(_state(request).db)}


@router.get("/datasets/{dataset_id}")
def dataset_detail(request: Request, dataset_id: str):
    dataset = get_dataset(_state(request).db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Dataset not found."})
    return {"dataset": dataset}


@router.get("/datasets/{dataset_id}/records")
def dataset_records(
    request: Request,
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=200),
    split: str | None = None,
    synthetic: bool | None = None,
    include_excluded: bool = True,
):
    if not get_dataset(_state(request).db, dataset_id):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Dataset not found."})
    return list_records(
        _state(request).db,
        dataset_id,
        offset=offset,
        limit=limit,
        split=split,
        synthetic=synthetic,
        include_excluded=include_excluded,
    )


class ExcludeRequest(BaseModel):
    excluded: bool = True
    reason: str | None = Field(default=None, max_length=200)


@router.post("/datasets/{dataset_id}/records/{record_id}/exclude")
def exclude_record(request: Request, dataset_id: str, record_id: str, body: ExcludeRequest):
    try:
        return set_record_excluded(_state(request).db, dataset_id, record_id, body.excluded, body.reason)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Record not found."})


class RightsRequest(BaseModel):
    rights_status: str


@router.post("/datasets/{dataset_id}/rights")
def set_rights(request: Request, dataset_id: str, body: RightsRequest):
    state = _state(request)
    try:
        result = set_dataset_rights(state.db, dataset_id, body.rights_status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "invalid_rights", "message": str(exc)})
    state.activity("dataset.rights", f"Rights set to {body.rights_status}", {"dataset_id": dataset_id})
    return result


class EnrichRequest(BaseModel):
    mode: str = Field(default="deterministic", pattern="^(deterministic|model)$")
    limit: int = Field(default=100, ge=1, le=2000)


@router.post("/datasets/{dataset_id}/enrich")
def enrich(request: Request, dataset_id: str, body: EnrichRequest):
    state = _state(request)
    if not get_dataset(state.db, dataset_id):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Dataset not found."})
    if body.mode == "model" and state.server.status().get("status") != "ready":
        raise HTTPException(status_code=409, detail={"code": "no_model", "message": "Load a model to use model-based enrichment."})

    def job(ctx):
        llm = state.server if body.mode == "model" else None
        result = enrich_dataset(
            state.db,
            dataset_id,
            mode=body.mode,
            llm=llm,
            limit=body.limit,
            progress=lambda pct, msg: ctx.progress(pct, msg),
            cancel=ctx.cancel_event,
        )
        return {
            "created": result.created,
            "rejected": result.rejected,
            "skipped": result.skipped,
            "teacher": result.teacher,
            "mode": result.mode,
        }

    job_id = state.runner.submit("datasets.enrich", job, input_data={"dataset_id": dataset_id, "mode": body.mode})
    return {"job_id": job_id}


@router.post("/datasets/{dataset_id}/freeze")
def freeze(request: Request, dataset_id: str):
    state = _state(request)
    try:
        manifest = freeze_dataset(state.db, dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Dataset not found."})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "cannot_freeze", "message": str(exc)})
    state.activity("dataset.freeze", "Dataset frozen", {"dataset_id": dataset_id})
    return {"manifest": manifest}


@router.get("/datasets/{dataset_id}/annotations")
def annotations(request: Request, dataset_id: str):
    if not get_dataset(_state(request).db, dataset_id):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Dataset not found."})
    return annotation_report(_state(request).db, dataset_id)


@router.get("/datasets/{dataset_id}/export")
def export(request: Request, dataset_id: str, include_synthetic: bool = True):
    state = _state(request)
    try:
        text = export_jsonl(state.db, dataset_id, include_synthetic=include_synthetic)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Dataset not found."})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "cannot_export", "message": str(exc)})
    state.activity("dataset.export", "Dataset exported as JSONL", {"dataset_id": dataset_id})
    return PlainTextResponse(
        text,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{dataset_id}.jsonl"'},
    )


@router.delete("/datasets/{dataset_id}")
def remove_dataset(request: Request, dataset_id: str):
    state = _state(request)
    if not get_dataset(state.db, dataset_id):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Dataset not found."})
    delete_dataset(state.db, dataset_id)
    return {"deleted": dataset_id}


# ---------------------------------------------------------------------------
# Identity Maker
# ---------------------------------------------------------------------------

class IdentityDraftRequest(BaseModel):
    prompt: str = Field(min_length=12, max_length=8000)
    name: str | None = Field(default=None, max_length=60)
    languages: list[str] | None = None


@router.post("/identity/drafts")
def identity_draft(request: Request, body: IdentityDraftRequest):
    state = _state(request)
    try:
        result = create_profile(state.db, body.prompt, body.name, body.languages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "invalid_prompt", "message": str(exc)})
    return result


@router.get("/identity")
def identity_list(request: Request):
    return {"profiles": list_profiles(_state(request).db)}


@router.get("/identity/{profile_id}")
def identity_detail(request: Request, profile_id: str):
    profile = get_profile(_state(request).db, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Identity profile not found."})
    return {"profile": profile}


class CardPatch(BaseModel):
    name: str | None = None
    purpose: str | None = None
    explicit_facts: list[str] | None = None
    boundaries: list[str] | None = None
    values: list[str] | None = None
    voice: dict | None = None
    languages: list[str] | None = None


@router.patch("/identity/{profile_id}")
def identity_update(request: Request, profile_id: str, body: CardPatch):
    state = _state(request)
    try:
        profile = update_card(state.db, profile_id, body.model_dump(exclude_none=True))
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Identity profile not found."})
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": "immutable", "message": str(exc)})
    return {"profile": profile}


@router.post("/identity/{profile_id}/approve")
def identity_approve(request: Request, profile_id: str):
    state = _state(request)
    try:
        profile = approve_profile(state.db, profile_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Identity profile not found."})
    state.activity("identity.approve", f"Approved identity {profile['name']}", {"profile_id": profile_id})
    return {"profile": profile}


@router.post("/identity/{profile_id}/dataset")
def identity_dataset(request: Request, profile_id: str):
    state = _state(request)
    try:
        result = create_identity_dataset(state.db, profile_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Identity profile not found."})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "cannot_create", "message": str(exc)})
    state.activity("identity.dataset", "Identity dataset created", {"dataset_id": result["dataset_id"]})
    return result


@router.post("/identity/{profile_id}/evaluate")
def identity_evaluate(request: Request, profile_id: str):
    state = _state(request)
    profile = get_profile(state.db, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Identity profile not found."})
    if state.server.status().get("status") != "ready":
        raise HTTPException(status_code=409, detail={"code": "no_model", "message": "Load a model to evaluate an identity."})

    def complete(messages):
        return state.server.complete(messages, temperature=0.2, max_tokens=120).get("text", "")

    result = evaluate_identity(profile_id, profile["card"], complete)
    state.activity("identity.evaluate", f"Identity score {result['score']:.0%}", {"profile_id": profile_id})
    return result
