from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from foundry import hf


def _client_with(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_headers_omit_authorization_when_anonymous():
    assert "Authorization" not in hf._headers(None)
    assert hf._headers("hf_secret")["Authorization"] == "Bearer hf_secret"


def test_search_uses_anonymous_and_token_requests(monkeypatch):
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json=[{"id": "org/model", "downloads": 5, "likes": 1, "tags": ["gguf"]}])

    monkeypatch.setattr(hf, "_client", lambda timeout=30.0: _client_with(handler))
    anonymous = hf.search_models("qwen", limit=1)
    authenticated = hf.search_models("qwen", limit=1, token="hf_abc")
    assert anonymous[0]["id"] == "org/model"
    assert seen[0] is None
    assert seen[1] == "Bearer hf_abc"
    assert authenticated[0]["downloads"] == 5


def test_gated_repository_maps_to_auth_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Repository is gated"})

    monkeypatch.setattr(hf, "_client", lambda timeout=30.0: _client_with(handler))
    with pytest.raises(hf.HFError) as exc:
        hf.model_info("org/gated")
    assert exc.value.code == "auth_required"
    assert "token" in str(exc.value).lower()


def test_not_found_maps_to_not_found(monkeypatch):
    monkeypatch.setattr(hf, "_client", lambda timeout=30.0: _client_with(lambda request: httpx.Response(404)))
    with pytest.raises(hf.HFError) as exc:
        hf.list_files("org/missing")
    assert exc.value.code == "not_found"


def test_token_store_roundtrip_and_mask(tmp_path: Path):
    assert hf.get_token(tmp_path) is None
    hf.set_token(tmp_path, "hf_1234567890abcdef")
    assert hf.get_token(tmp_path) == "hf_1234567890abcdef"
    status = hf.token_status(tmp_path)
    assert status["configured"] is True
    assert status["masked"].startswith("hf_1")
    assert "1234567890" not in status["masked"]
    hf.set_token(tmp_path, None)
    assert hf.get_token(tmp_path) is None


def test_repo_directory_is_traversal_safe():
    assert hf.sanitize_repo_dir("org/name") == Path("org") / "name"
    assert hf.sanitize_repo_dir("../../etc/passwd") == Path("etc") / "passwd"
    assert hf.sanitize_repo_dir("org/../../x") == Path("org") / "x"


def test_pick_default_files_prefers_smallest_gguf_and_projector():
    files = [
        {"path": "big.gguf", "size": 900},
        {"path": "small.gguf", "size": 100},
        {"path": "mmproj-model.gguf", "size": 50},
        {"path": "README.md", "size": 10},
    ]
    assert hf.pick_default_files(files) == ["small.gguf", "mmproj-model.gguf"]


def test_pick_default_files_prefers_safetensors_over_bin():
    files = [
        {"path": "model.safetensors", "size": 200},
        {"path": "pytorch_model.bin", "size": 200},
    ]
    assert hf.pick_default_files(files) == ["model.safetensors"]


def test_download_resumes_from_part_file(monkeypatch, tmp_path: Path):
    payload = b"0123456789abcdef"
    (tmp_path / "weights.bin.part").write_bytes(payload[:6])
    digest = hashlib.sha256(payload).hexdigest()
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["range"] = request.headers.get("range")
        return httpx.Response(206, content=payload[6:], headers={"content-length": str(len(payload) - 6)})

    monkeypatch.setattr(hf, "_client", lambda timeout=30.0: _client_with(handler))
    result = hf.download_file(
        "org/repo",
        "weights.bin",
        tmp_path,
        expected_sha256=digest,
        expected_size=len(payload),
    )
    assert seen["range"] == "bytes=6-"
    assert (tmp_path / "weights.bin").read_bytes() == payload
    assert result["sha256"] == digest
    assert not (tmp_path / "weights.bin.part").exists()


def test_download_restarts_when_server_ignores_range(monkeypatch, tmp_path: Path):
    payload = b"abcdefghij"
    (tmp_path / "w.bin.part").write_bytes(b"old")
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    monkeypatch.setattr(hf, "_client", lambda timeout=30.0: _client_with(handler))
    hf.download_file("org/repo", "w.bin", tmp_path, expected_sha256=digest, expected_size=len(payload))
    assert (tmp_path / "w.bin").read_bytes() == payload


def test_download_rejects_hash_mismatch(monkeypatch, tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"wrong")

    monkeypatch.setattr(hf, "_client", lambda timeout=30.0: _client_with(handler))
    with pytest.raises(hf.HFError) as exc:
        hf.download_file("org/repo", "w.bin", tmp_path, expected_sha256="0" * 64, expected_size=5)
    assert exc.value.code == "hash_mismatch"
    assert not (tmp_path / "w.bin").exists()


def test_download_cancellation(monkeypatch, tmp_path: Path):
    import threading

    cancel = threading.Event()
    cancel.set()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 1024)

    monkeypatch.setattr(hf, "_client", lambda timeout=30.0: _client_with(handler))
    with pytest.raises(InterruptedError):
        hf.download_file("org/repo", "w.bin", tmp_path, cancel=cancel)
