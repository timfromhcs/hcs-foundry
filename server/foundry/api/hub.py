from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .. import hf
from ..library import scan_models
from ..security import resolve_within

router = APIRouter()


def _state(request: Request):
    return request.app.state.foundry


def _token(state) -> str | None:
    return hf.get_token(state.settings.state_dir)


def _hub_error(exc: hf.HFError) -> HTTPException:
    status = 401 if exc.code in {"auth_required", "invalid_token"} else 404 if exc.code == "not_found" else 429 if exc.code == "rate_limited" else 400
    return HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)})


@router.get("/hub/search")
def hub_search(
    request: Request,
    q: str = Query(default="", max_length=200),
    task: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="downloads", pattern="^(downloads|likes|lastModified|trendingScore)$"),
):
    state = _state(request)
    try:
        results = hf.search_models(q, task=task, limit=limit, token=_token(state), sort=sort)
    except hf.HFError as exc:
        raise _hub_error(exc)
    return {"query": q, "results": results, "anonymous": _token(state) is None}


@router.get("/hub/models/{repo_id:path}")
def hub_model(request: Request, repo_id: str):
    state = _state(request)
    try:
        info = hf.model_info(repo_id, token=_token(state))
        files = hf.list_files(repo_id, token=_token(state))
    except hf.HFError as exc:
        raise _hub_error(exc)
    return {
        "model": info,
        "files": files,
        "suggested": hf.pick_default_files(files),
        "total_bytes": sum(f["size"] for f in files),
    }


@router.get("/hub/token")
def hub_token(request: Request):
    return hf.token_status(_state(request).settings.state_dir)


class TokenRequest(BaseModel):
    token: str = Field(min_length=4, max_length=400)


@router.put("/hub/token")
def set_hub_token(request: Request, body: TokenRequest):
    state = _state(request)
    token = body.token.strip()
    try:
        identity = hf.whoami(token)
    except hf.HFError as exc:
        raise _hub_error(exc)
    hf.set_token(state.settings.state_dir, token)
    state.activity("hub.token", "Hugging Face token configured", {"user": identity.get("name")})
    return {"status": hf.token_status(state.settings.state_dir), "identity": identity}


@router.delete("/hub/token")
def clear_hub_token(request: Request):
    state = _state(request)
    hf.set_token(state.settings.state_dir, None)
    state.activity("hub.token", "Hugging Face token removed")
    return hf.token_status(state.settings.state_dir)


@router.post("/hub/token/test")
def test_hub_token(request: Request):
    state = _state(request)
    token = _token(state)
    if not token:
        return {"ok": True, "mode": "anonymous", "note": "Public repositories work without a token."}
    try:
        identity = hf.whoami(token)
    except hf.HFError as exc:
        raise _hub_error(exc)
    return {"ok": True, "mode": "token", "identity": identity}


class DownloadRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=200)
    files: list[str] = Field(default_factory=list, max_length=64)
    dest_root: str | None = None


@router.post("/hub/download")
def hub_download(request: Request, body: DownloadRequest):
    state = _state(request)
    token = _token(state)

    roots = [Path(p).expanduser() for p in state.settings.model_roots]
    default_root = state.settings.data_dir / "models"
    default_root.mkdir(parents=True, exist_ok=True)
    if default_root not in roots:
        roots.append(default_root)
    if body.dest_root:
        try:
            dest_root = resolve_within(body.dest_root, roots)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={"code": "dest_not_allowed", "message": "Destination must be inside a configured model root."},
            )
    else:
        dest_root = default_root

    try:
        files = hf.list_files(body.repo_id, token=token)
    except hf.HFError as exc:
        raise _hub_error(exc)
    by_path = {f["path"]: f for f in files}
    selected = [name for name in body.files if name in by_path] or hf.pick_default_files(files)
    if not selected:
        raise HTTPException(status_code=400, detail={"code": "no_files", "message": "This repository has no downloadable files."})

    dest = dest_root / hf.sanitize_repo_dir(body.repo_id)

    def job(ctx):
        ctx.log(f"Pulling {len(selected)} file(s) from {body.repo_id}")
        result = hf.download_many(
            body.repo_id,
            selected,
            dest,
            token=token,
            files_meta=by_path,
            progress=lambda pct, msg: ctx.progress(pct, msg),
            cancel=ctx.cancel_event,
        )
        ctx.progress(96, "Indexing downloaded files")
        scan = scan_models(state.db, state.settings, roots=[dest])
        ctx.artifact("model-dir", str(dest), repo_id=body.repo_id)
        return {
            "repo_id": body.repo_id,
            "dest": str(dest),
            "files": result["files"],
            "indexed": scan["added"] + scan["updated"],
            "anonymous": token is None,
        }

    job_id = state.runner.submit("hub.download", job, input_data={"repo_id": body.repo_id, "files": selected}, heavy=True)
    return {"job_id": job_id, "files": selected, "dest": str(dest), "anonymous": token is None}
