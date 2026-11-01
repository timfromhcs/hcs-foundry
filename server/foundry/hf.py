"""Hugging Face Hub client.

Anonymous access works for public repositories. A token is only attached when one is
configured, which adds access to gated and private repositories. Downloads are
resumable, cancellable, hash-verified when the Hub publishes an LFS digest, and
written atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import httpx

HF_WEB = "https://huggingface.co"
HF_API = f"{HF_WEB}/api"
CHUNK = 2 * 1024 * 1024
MAX_FILES = 200

TASK_FILTERS = {
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "automatic-speech-recognition",
    "sentence-similarity",
    "feature-extraction",
    "text-classification",
    "token-classification",
    "translation",
    "summarization",
}


class HFError(RuntimeError):
    def __init__(self, code: str, message: str, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class HFModel:
    id: str
    author: str | None = None
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    gated: bool | str = False
    private: bool = False
    last_modified: str | None = None
    library_name: str | None = None

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "author": self.author,
            "downloads": self.downloads,
            "likes": self.likes,
            "tags": self.tags,
            "pipeline_tag": self.pipeline_tag,
            "gated": self.gated,
            "private": self.private,
            "last_modified": self.last_modified,
            "library_name": self.library_name,
        }


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------

def _token_path(state_dir: Path) -> Path:
    return Path(state_dir) / "hf-token.json"


def get_token(state_dir: Path) -> str | None:
    path = _token_path(state_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    token = data.get("token")
    return token if isinstance(token, str) and token else None


def set_token(state_dir: Path, token: str | None) -> None:
    path = _token_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not token:
        path.unlink(missing_ok=True)
        return
    payload = json.dumps({"token": token.strip()}, indent=2)
    path.write_text(payload, "utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def mask_token(token: str | None) -> str | None:
    if not token:
        return None
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}…{token[-4:]}"


def token_status(state_dir: Path) -> dict:
    token = get_token(state_dir)
    return {
        "configured": bool(token),
        "masked": mask_token(token),
        "note": "A token is only needed for gated or private repositories.",
    }


def whoami(token: str | None) -> dict:
    if not token:
        raise HFError("auth_required", "No token configured.")
    with _client() as client:
        response = client.get(f"{HF_API}/whoami-v2", headers=_headers(token))
        if response.status_code in (401, 403):
            raise HFError("invalid_token", "The Hugging Face token was rejected.", response.status_code)
        _raise_for(response)
        data = response.json()
    return {
        "name": data.get("name"),
        "type": data.get("type"),
        "can_pay": data.get("canPay"),
        "orgs": [org.get("name") for org in data.get("orgs") or [] if isinstance(org, dict)],
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "HCS-Foundry/2.0 (+local)"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _raise_for(response: httpx.Response, repo_id: str | None = None) -> None:
    if response.status_code < 400:
        return
    target = repo_id or "the Hub"
    if response.status_code in (401, 403):
        hint = ""
        try:
            hint = response.json().get("error", "")
        except (ValueError, AttributeError):
            hint = response.headers.get("x-error-message", "")
        raise HFError(
            "auth_required",
            f"{target} is gated or private. Add a Hugging Face token with access"
            + (f" ({hint})" if hint else "."),
            response.status_code,
        )
    if response.status_code == 404:
        raise HFError("not_found", f"{target} was not found on the Hub.", 404)
    if response.status_code == 429:
        raise HFError("rate_limited", "The Hub rate limit was hit. Try again later or add a token.", 429)
    raise HFError("http_error", f"Hub request failed with HTTP {response.status_code}.", response.status_code)


def _client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(connect=15.0, read=timeout, write=30.0, pool=15.0), follow_redirects=True)


# ---------------------------------------------------------------------------
# Search / inspect
# ---------------------------------------------------------------------------

def _to_model(item: dict) -> HFModel:
    return HFModel(
        id=str(item.get("id") or item.get("modelId") or ""),
        author=item.get("author"),
        downloads=int(item.get("downloads") or 0),
        likes=int(item.get("likes") or 0),
        tags=list(item.get("tags") or []),
        pipeline_tag=item.get("pipeline_tag"),
        gated=item.get("gated", False),
        private=bool(item.get("private", False)),
        last_modified=item.get("lastModified"),
        library_name=item.get("library_name"),
    )


def search_models(
    query: str,
    *,
    task: str | None = None,
    limit: int = 20,
    token: str | None = None,
    sort: str = "downloads",
) -> list[dict]:
    params: dict[str, str | int] = {"limit": max(1, min(limit, 100)), "sort": sort, "direction": "-1"}
    if query.strip():
        params["search"] = query.strip()
    if task:
        params["pipeline_tag"] = task
    with _client() as client:
        response = client.get(f"{HF_API}/models", params=params, headers=_headers(token))
        _raise_for(response)
        items = response.json()
    return [_to_model(item).to_public() for item in items if isinstance(item, dict)]


def model_info(repo_id: str, *, token: str | None = None) -> dict:
    with _client() as client:
        response = client.get(f"{HF_API}/models/{repo_id}", headers=_headers(token))
        _raise_for(response, repo_id)
        data = response.json()
    model = _to_model(data).to_public()
    siblings = data.get("siblings") or []
    model["file_count"] = len(siblings)
    model["files"] = [s.get("rfilename") for s in siblings if s.get("rfilename")][:MAX_FILES]
    return model


def list_files(repo_id: str, *, token: str | None = None) -> list[dict]:
    with _client() as client:
        response = client.get(
            f"{HF_API}/models/{repo_id}/tree/main",
            params={"recursive": "true"},
            headers=_headers(token),
        )
        _raise_for(response, repo_id)
        items = response.json()
    files: list[dict] = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        lfs = item.get("lfs") or {}
        files.append(
            {
                "path": item.get("path"),
                "size": int(item.get("size") or lfs.get("size") or 0),
                "sha256": lfs.get("oid") if isinstance(lfs.get("oid"), str) and len(lfs.get("oid")) == 64 else None,
                "is_lfs": bool(lfs),
            }
        )
    files.sort(key=lambda f: f["size"])
    return files[:MAX_FILES]


def pick_default_files(files: Iterable[dict]) -> list[str]:
    """Prefer one quantized GGUF plus its projector, else one weights format."""
    entries = list(files)
    gguf = [f for f in entries if str(f["path"]).lower().endswith(".gguf") and "mmproj" not in str(f["path"]).lower()]
    projectors = [f for f in entries if "mmproj" in str(f["path"]).lower() and str(f["path"]).lower().endswith(".gguf")]
    if gguf:
        chosen = [min(gguf, key=lambda f: f["size"])]
        if projectors:
            chosen.append(min(projectors, key=lambda f: f["size"]))
        return [f["path"] for f in chosen]
    for suffix in (".safetensors", ".bin", ".pt"):
        weights = [f for f in entries if str(f["path"]).lower().endswith(suffix)]
        if weights:
            return [f["path"] for f in sorted(weights, key=lambda f: f["size"])[:4]]
    return [f["path"] for f in entries[:1]]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    *,
    token: str | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    timeout: float = 120.0,
) -> dict:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(filename).name
    part = target.with_suffix(target.suffix + ".part")

    if target.exists() and expected_size and target.stat().st_size == expected_size:
        if not expected_sha256 or _sha256_file(target) == expected_sha256:
            return {"path": str(target), "size": target.stat().st_size, "skipped": True, "sha256": expected_sha256}

    if expected_size:
        free = shutil.disk_usage(dest_dir).free
        if free < expected_size * 1.1:
            raise HFError("no_space", f"Not enough disk space: need {expected_size} bytes, {free} free.")

    existing = part.stat().st_size if part.exists() else 0
    headers = _headers(token)
    if existing:
        headers["Range"] = f"bytes={existing}-"

    url = f"{HF_WEB}/{repo_id}/resolve/main/{filename}?download=true"
    written = existing
    digest = hashlib.sha256()
    if existing and expected_sha256 is None:
        existing = 0  # cannot trust an unverifiable partial; restart
        headers.pop("Range", None)
        written = 0
    if written:
        with part.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)

    mode = "ab" if written else "wb"
    with _client(timeout=timeout) as client:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416 and written:
                total = written
            elif response.status_code not in (200, 206):
                _raise_for(response, f"{repo_id}/{filename}")
            else:
                if response.status_code == 200 and written:
                    # The server ignored the Range request; restart cleanly.
                    written = 0
                    digest = hashlib.sha256()
                    mode = "wb"
                content_length = int(response.headers.get("content-length") or 0)
                total = written + content_length
                if expected_size and total and total != expected_size:
                    total = expected_size
                with part.open(mode) as handle:
                    for block in response.iter_bytes(CHUNK):
                        if cancel is not None and cancel.is_set():
                            raise InterruptedError("download cancelled")
                        handle.write(block)
                        digest.update(block)
                        written += len(block)
                        if progress is not None:
                            progress(written, total or expected_size or written)

    if expected_size and written != expected_size:
        raise HFError("size_mismatch", f"Downloaded {written} bytes, expected {expected_size}.")
    actual = digest.hexdigest()
    if expected_sha256 and actual != expected_sha256:
        part.unlink(missing_ok=True)
        raise HFError("hash_mismatch", "SHA-256 does not match the Hub digest; file discarded.")

    part.replace(target)
    return {"path": str(target), "size": written, "skipped": False, "sha256": actual}


def download_many(
    repo_id: str,
    filenames: list[str],
    dest_dir: Path,
    *,
    token: str | None = None,
    files_meta: dict[str, dict] | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
) -> dict:
    meta = files_meta or {}
    results = []
    total = len(filenames)
    for index, filename in enumerate(filenames):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("download cancelled")
        info = meta.get(filename, {})

        def on_progress(done: int, size: int, index=index, name=filename) -> None:
            if progress is None:
                return
            fraction = (done / size) if size else 0.0
            progress(((index + fraction) / max(1, total)) * 100, f"{name}: {done}/{size or '?'} bytes")

        if progress is not None:
            progress((index / max(1, total)) * 100, f"Downloading {filename}")
        result = download_file(
            repo_id,
            filename,
            dest_dir,
            token=token,
            expected_sha256=info.get("sha256"),
            expected_size=info.get("size") or None,
            progress=on_progress,
            cancel=cancel,
        )
        results.append(result)
    if progress is not None:
        progress(100, f"{len(results)} file(s) ready")
    return {"repo_id": repo_id, "dest": str(dest_dir), "files": results}


def sanitize_repo_dir(repo_id: str) -> Path:
    """Map ``org/name`` to a safe relative directory (no traversal)."""
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "_", part) for part in repo_id.split("/") if part not in {"", ".", ".."}]
    if not parts:
        raise HFError("invalid_repo", "Repository id is not valid.")
    return Path(*parts[-2:])
