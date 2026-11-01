from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from foundry import gateway


def _client_with(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_config_roundtrip_and_mask(tmp_path: Path):
    assert gateway.get_config(tmp_path)["configured"] is False
    config = gateway.set_config(tmp_path, key="vck_1234567890abcdef", model="alibaba/qwen-3-14b")
    assert config["configured"] is True
    status = gateway.public_status(tmp_path)
    assert status["masked"].startswith("vck_")
    assert "1234567890" not in status["masked"]
    assert status["enabled"] is False
    gateway.set_config(tmp_path, key=config["key"], enabled=True)
    assert gateway.public_status(tmp_path)["enabled"] is True
    gateway.set_config(tmp_path, key=None)
    assert gateway.get_config(tmp_path)["configured"] is False


def test_list_models(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer vck_x"
        return httpx.Response(200, json={"data": [{"id": "a/b", "name": "B", "owned_by": "a"}]})

    monkeypatch.setattr(gateway, "_client", lambda timeout=30.0: _client_with(handler))
    models = gateway.list_models("vck_x")
    assert models[0]["id"] == "a/b"


def test_billing_error_is_explicit(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "AI Gateway requires a valid credit card on file"}})

    monkeypatch.setattr(gateway, "_client", lambda timeout=30.0: _client_with(handler))
    with pytest.raises(gateway.GatewayError) as exc:
        gateway.list_models("vck_x")
    assert exc.value.code == "billing_required"
    assert "credit card" in str(exc.value)


def test_invalid_key_mapping(monkeypatch):
    monkeypatch.setattr(
        gateway, "_client", lambda timeout=30.0: _client_with(lambda request: httpx.Response(401))
    )
    with pytest.raises(gateway.GatewayError) as exc:
        gateway.list_models("vck_x")
    assert exc.value.code == "invalid_key"


def test_complete_returns_text(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "a/b",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"total_tokens": 3},
            },
        )

    monkeypatch.setattr(gateway, "_client", lambda timeout=30.0: _client_with(handler))
    result = gateway.complete("vck_x", [{"role": "user", "content": "hi"}])
    assert result["text"] == "hello"
    assert result["model"] == "a/b"


def test_teacher_adapter_name():
    teacher = gateway.GatewayTeacher("vck_x", "a/b")
    assert teacher.teacher_name == "vercel-ai-gateway:a/b"
