from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.browser_runtime import inventory, resolve_file, summary


def _make_model(root: Path, name: str = "smollm-135m", *, onnx: bool = True, tokenizer: bool = True) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    if onnx:
        (directory / "onnx").mkdir(exist_ok=True)
        (directory / "onnx" / "model_q4.onnx").write_bytes(b"ONNX" + b"\0" * 64)
        (directory / "onnx" / "model_q4.onnx_data").write_bytes(b"\0" * 128)
    if tokenizer:
        (directory / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}), "utf-8")
        (directory / "config.json").write_text(json.dumps({"model_type": "llama"}), "utf-8")
    return directory


def test_inventory_reports_ready_and_missing(tmp_path: Path):
    root = tmp_path / "browser-models"
    _make_model(root)
    _make_model(root, "broken", onnx=False)
    models = {item["id"]: item for item in inventory(root)}
    assert models["smollm-135m"]["ready"] is True
    assert models["smollm-135m"]["file_count"] == 4
    assert models["smollm-135m"]["size_bytes"] > 0
    assert models["broken"]["ready"] is False
    assert "*.onnx" in models["broken"]["missing"]


def test_summary_without_a_directory_is_honest(tmp_path: Path):
    payload = summary(tmp_path / "does-not-exist")
    assert payload["exists"] is False
    assert payload["ready"] == 0
    assert "browser-models" in payload["hint"]


def test_nested_files_are_served_and_escapes_are_refused(tmp_path: Path):
    root = tmp_path / "browser-models"
    _make_model(root)
    target = resolve_file(root, "smollm-135m", "onnx/model_q4.onnx")
    assert target.name == "model_q4.onnx"

    with pytest.raises(ValueError):
        resolve_file(root, "../etc", "passwd")
    with pytest.raises(ValueError):
        resolve_file(root, "smollm-135m", "../../../secrets.txt")
    with pytest.raises(FileNotFoundError):
        resolve_file(root, "smollm-135m", "nope.onnx")


def test_api_lists_and_serves_browser_models(api, settings):
    root = settings.data_dir / "browser-models"
    _make_model(root)
    listing = api.get("/api/v1/browser/runtime")
    assert listing.status_code == 200
    payload = listing.json()
    assert payload["ready"] == 1
    assert payload["models"][0]["id"] == "smollm-135m"

    served = api.get("/api/v1/browser/models/smollm-135m/tokenizer.json")
    assert served.status_code == 200
    assert served.headers["x-content-type-options"] == "nosniff"
    assert served.json()["model"]["type"] == "BPE"

    binary = api.get("/api/v1/browser/models/smollm-135m/onnx/model_q4.onnx")
    assert binary.status_code == 200
    assert binary.content.startswith(b"ONNX")

    assert api.get("/api/v1/browser/models/smollm-135m/../../etc/passwd").status_code in (400, 404)
    assert api.get("/api/v1/browser/models/..%2F..%2Fetc/passwd").status_code in (400, 404)
