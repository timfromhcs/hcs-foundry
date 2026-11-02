from __future__ import annotations

import os

import pytest

from foundry.config import DEFAULT_UI_ORIGINS, _split_origins, load_settings
from foundry.security import SecurityPolicy


def test_split_origins_keeps_urls_intact():
    raw = "https://hcs-foundry-site.vercel.app, http://127.0.0.1:5173 ; http://localhost:4173/"
    assert _split_origins(raw) == [
        "https://hcs-foundry-site.vercel.app",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ]


def test_split_origins_refuses_a_wildcard():
    assert _split_origins("*") == []
    assert _split_origins("https://a.example, *") == ["https://a.example"]


def test_env_origins_survive_the_scheme(monkeypatch):
    monkeypatch.setenv("FOUNDRY_UI_ORIGINS", "https://demo.example,http://127.0.0.1:5173")
    settings = load_settings()
    assert "https://demo.example" in settings.ui_origins
    assert not any(":/demo" in origin for origin in settings.ui_origins)


def test_policy_accepts_the_deployed_site_and_nothing_else():
    policy = SecurityPolicy(origins=list(DEFAULT_UI_ORIGINS), allow_lan=False)
    assert policy.origin_allowed("https://hcs-foundry-site.vercel.app")
    assert policy.origin_allowed("https://hcs-foundry-site.vercel.app/")
    assert policy.origin_allowed("http://127.0.0.1:8770")
    assert not policy.origin_allowed("https://evil.example")
    assert not policy.origin_allowed("*")


def test_cors_echoes_the_exact_origin(settings):
    from fastapi.testclient import TestClient

    from foundry.main import create_app

    settings.ui_origins = ["https://demo.example"]
    app = create_app(settings, serve_web=False)
    with TestClient(app, base_url="http://127.0.0.1:8770") as client:
        allowed = client.options(
            "/api/v1/system/pair",
            headers={
                "Origin": "https://demo.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-foundry-client",
            },
        )
        assert allowed.status_code in (200, 204)
        assert allowed.headers.get("access-control-allow-origin") == "https://demo.example"
        assert allowed.headers.get("access-control-allow-origin") != "*"

        foreign = client.options(
            "/api/v1/system/pair",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert foreign.status_code in (400, 403)
        assert foreign.headers.get("access-control-allow-origin") in (None, "https://demo.example")


def test_pairing_code_is_enforced_when_enabled(settings):
    from fastapi.testclient import TestClient

    from foundry.main import create_app

    app = create_app(settings, serve_web=False)
    with TestClient(app, base_url="http://127.0.0.1:8770") as client:
        state = client.app.state.foundry
        state.pairing.require_code = True
        code = state.pairing.pairing_code()
        denied = client.post(
            "/api/v1/system/pair",
            json={"label": "web demo"},
            headers={"X-Foundry-Client": "1", "Origin": "http://127.0.0.1:8770"},
        )
        assert denied.status_code == 403
        accepted = client.post(
            "/api/v1/system/pair",
            json={"label": "web demo", "code": code},
            headers={"X-Foundry-Client": "1", "Origin": "http://127.0.0.1:8770"},
        )
        assert accepted.status_code == 200
        assert accepted.json()["token"].startswith("fnd_")
