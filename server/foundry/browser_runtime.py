"""Browser runtime support: serve ONNX models from the local data directory.

The browser runtime (WebGPU/WASM via Transformers.js) must not depend on a CDN, so the SPA loads
model files from this service on the loopback interface. A model is a directory under
``data/browser-models/<model_id>/`` containing an ONNX file plus tokenizer assets.

Nothing here downloads anything: if the directory is empty, the UI says so and shows the exact
command to fill it.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from .security import resolve_within

#: Files that make a directory a loadable browser model.
REQUIRED_ANY = (".onnx",)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "config.json")


def models_root(data_dir: Path) -> Path:
    return Path(data_dir) / "browser-models"


def inventory(root: Path) -> list[dict]:
    """Describe every browser-model directory without reading model contents."""
    if not root.is_dir():
        return []
    found: list[dict] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = [path for path in sorted(directory.rglob("*")) if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        names = {path.name for path in files}
        has_model = any(path.name.endswith(REQUIRED_ANY) for path in files)
        has_tokenizer = "tokenizer.json" in names
        missing = []
        if not has_model:
            missing.append("*.onnx")
        if not has_tokenizer:
            missing.append("tokenizer.json")
        found.append(
            {
                "id": directory.name,
                "path": str(directory),
                "size_bytes": total,
                "file_count": len(files),
                "files": [{"name": str(path.relative_to(directory)), "size": path.stat().st_size} for path in files],
                "ready": has_model and has_tokenizer,
                "missing": missing,
            }
        )
    return found


def model_dir(root: Path, model_id: str) -> Path:
    """Resolve a model id inside the root and refuse anything that escapes it."""
    if not model_id or "/" in model_id or "\\" in model_id or model_id.startswith("."):
        raise ValueError("invalid model id")
    base = resolve_within(root, [root])
    candidate = resolve_within(base / model_id, [root])
    if not candidate.is_dir():
        raise FileNotFoundError(model_id)
    return candidate


def resolve_file(root: Path, model_id: str, relative: str) -> Path:
    directory = model_dir(root, model_id)
    target = resolve_within(directory / relative, [root])
    if not target.is_file():
        raise FileNotFoundError(relative)
    return target


def content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    if path.suffix == ".onnx":
        return "application/octet-stream"
    if path.suffix == ".json":
        return "application/json"
    return "application/octet-stream"


def summary(root: Path) -> dict:
    models = inventory(root)
    ready = [model for model in models if model["ready"]]
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "models": models,
        "total": len(models),
        "ready": len(ready),
        "hint": (
            "put an ONNX export in data/browser-models/<id>/ (e.g. "
            "`optimum-cli export onnx --model HuggingFaceTB/SmolLM2-135M-Instruct <dir>`)"
            if not ready
            else "ready"
        ),
    }
