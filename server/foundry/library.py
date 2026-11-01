"""Local model library: discovery, GGUF metadata, integrity and fit estimates."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Callable, Iterable

from .config import Settings
from .db import Database, utc_now
from .machine.governor import Budget, estimate_fit
from .runtime.gguf import GGUFError, read_gguf

CHUNK = 4 * 1024 * 1024

TASK_HINTS = [
    ("coding", ("coder", "code", "starcoder", "devstral", "qwen2.5-coder", "deepseek-coder")),
    ("embedding", ("embed", "bge", "e5-", "gte-", "nomic")),
    ("reranking", ("rerank", "reranker")),
    ("vision", ("vl", "vision", "llava", "mmproj", "minicpm-v", "gemma3-vision")),
    ("agentic", ("agent", "tool", "function-calling", "hermes")),
    ("reasoning", ("reason", "think", "r1", "qwq")),
]


def model_id_for(path: str | Path) -> str:
    digest = hashlib.sha1(str(Path(path)).encode("utf-8")).hexdigest()
    return f"mdl_{digest[:20]}"


def classify_task(name: str) -> str:
    lower = name.lower()
    for task, hints in TASK_HINTS:
        if any(hint in lower for hint in hints):
            return task
    return "chat"


def hash_file(path: Path, progress: Callable[[float], None] | None = None, cancel: threading.Event | None = None) -> str:
    total = max(1, path.stat().st_size)
    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as handle:
        while True:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("hashing cancelled")
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress is not None:
                progress(min(99.0, done / total * 100))
    return digest.hexdigest()


def _sibling_assets(path: Path) -> dict:
    folder = path.parent
    try:
        files = [p for p in folder.iterdir() if p.is_file()]
    except OSError:
        files = []
    projectors = [str(p) for p in files if "mmproj" in p.name.lower() and p.suffix.lower() == ".gguf"]
    total_bytes = sum(p.stat().st_size for p in files if p.suffix.lower() == ".gguf")
    return {"projectors": projectors, "folder_gguf_bytes": total_bytes, "sibling_count": len(files)}


def scan_models(
    db: Database,
    settings: Settings,
    roots: Iterable[Path] | None = None,
    *,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
) -> dict:
    roots = [Path(r) for r in (roots or settings.model_roots)]
    found: list[Path] = []
    errors: list[dict] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            errors.append({"root": str(root), "error": "root does not exist"})
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("scan cancelled")
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in {"node_modules", "__pycache__"}]
            for filename in filenames:
                if filename.lower().endswith(".gguf"):
                    found.append(Path(dirpath) / filename)

    added, updated, invalid = 0, 0, 0
    for index, path in enumerate(found):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("scan cancelled")
        if progress is not None:
            progress(index / max(1, len(found)) * 100, f"Inspecting {path.name}")
        try:
            info = read_gguf(path)
        except (GGUFError, OSError) as exc:
            invalid += 1
            errors.append({"path": str(path), "error": str(exc)})
            continue

        assets = _sibling_assets(path)
        mid = model_id_for(path)
        existing = db.query_one("SELECT id, sha256, added_at FROM models WHERE path=?", (str(path),))
        meta = {
            "gguf": info.to_public(),
            "assets": assets,
            "has_projector": bool(assets["projectors"]),
            "size_bytes": path.stat().st_size,
        }
        name = info.name or path.stem
        values = {
            "id": mid,
            "path": str(path),
            "name": name,
            "size_bytes": path.stat().st_size,
            "architecture": info.architecture,
            "quantization": info.quantization,
            "parameter_count": info.parameter_count,
            "context_max": info.context_length,
            "license": info.license,
            "task": classify_task(name),
            "meta": _json(meta),
        }
        if existing:
            db.execute(
                "UPDATE models SET name=?, size_bytes=?, architecture=?, quantization=?, parameter_count=?, "
                "context_max=?, license=COALESCE(?, license), meta=? WHERE id=?",
                (
                    values["name"], values["size_bytes"], values["architecture"], values["quantization"],
                    values["parameter_count"], values["context_max"], values["license"], values["meta"], mid,
                ),
            )
            updated += 1
        else:
            values["added_at"] = utc_now()
            db.insert("models", values)
            added += 1

    if progress is not None:
        progress(100, f"{len(found)} GGUF files inspected")
    db.activity(
        "models.scan",
        f"Scanned {len(roots)} roots: {added} added, {updated} updated, {invalid} invalid",
        {"roots": [str(r) for r in roots]},
    )
    return {
        "roots": [str(r) for r in roots],
        "discovered": len(found),
        "added": added,
        "updated": updated,
        "invalid": invalid,
        "errors": errors[:50],
    }


def _json(value) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def list_models(db: Database) -> list[dict]:
    rows = db.query("SELECT * FROM models ORDER BY added_at DESC")
    return [_row(r) for r in rows]


def get_model(db: Database, model_id: str) -> dict | None:
    row = db.query_one("SELECT * FROM models WHERE id=?", (model_id,))
    return _row(row) if row else None


def _row(row) -> dict:
    import json

    data = dict(row)
    data["meta"] = json.loads(data.get("meta") or "{}")
    data["exists"] = Path(data["path"]).is_file()
    data["sha256_verified"] = bool(data.get("sha256"))
    return data


def verify_model(db: Database, model_id: str, progress=None, cancel=None) -> dict:
    model = get_model(db, model_id)
    if not model:
        raise KeyError(model_id)
    path = Path(model["path"])
    if not path.is_file():
        raise FileNotFoundError(str(path))
    digest = hash_file(path, progress=progress, cancel=cancel)
    db.execute("UPDATE models SET sha256=?, verified_at=? WHERE id=?", (digest, utc_now(), model_id))
    db.activity("models.verify", f"Verified {model['name']}", {"sha256": digest})
    return {"model_id": model_id, "sha256": digest, "size_bytes": path.stat().st_size}


def update_model(db: Database, model_id: str, **fields) -> dict:
    allowed = {"name", "license", "source_url", "task"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if updates:
        sets = ", ".join(f"{k}=?" for k in updates)
        db.execute(f"UPDATE models SET {sets} WHERE id=?", (*updates.values(), model_id))
    model = get_model(db, model_id)
    if model is None:
        raise KeyError(model_id)
    return model


def remove_model(db: Database, model_id: str) -> None:
    """Remove from the library only. The file on disk is never deleted."""
    db.execute("DELETE FROM models WHERE id=?", (model_id,))
    db.activity("models.remove", "Removed model from library (file untouched)", {"model_id": model_id})


def model_fit(model: dict, context: int, budget: Budget) -> dict:
    gguf = (model.get("meta") or {}).get("gguf") or {}
    kv = gguf.get("kv_bytes_per_token")
    fit = estimate_fit(
        model_size_bytes=int(model["size_bytes"]),
        kv_bytes_per_token=kv,
        context=context,
        budget=budget,
    )
    fit["model_id"] = model["id"]
    fit["model_name"] = model["name"]
    fit["kv_bytes_per_token"] = kv
    return fit
