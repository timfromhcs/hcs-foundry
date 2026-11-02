"""The update check is opt-in, ID-less and cannot download anything."""

from __future__ import annotations

import pytest

from foundry import update as upd


def test_version_ordering_handles_prefixes_and_lengths():
    assert upd.parse_version("v7.3.0") == (7, 3, 0)
    assert upd.is_newer("7.4.0", "7.3.0") is True
    assert upd.is_newer("7.3", "7.3.0") is False
    assert upd.is_newer("7.10.0", "7.9.9") is True
    assert upd.parse_version("7.3.0-rc1") == (7, 3, 0)


def test_a_newer_release_is_reported_without_downloading_anything():
    payload = upd.check_for_update(
        "7.3.0",
        fetch=lambda: {"tag_name": "v7.4.0", "html_url": "https://example.invalid/r"},
    )
    assert payload["status"] == "ok"
    assert payload["available"] is True
    assert payload["latest"] == "v7.4.0"
    assert payload["url"] == "https://example.invalid/r"


def test_the_current_version_is_not_reported_as_an_update():
    payload = upd.check_for_update("7.3.0", fetch=lambda: {"tag_name": "v7.3.0"})
    assert payload["status"] == "ok"
    assert payload["available"] is False


def test_a_network_failure_is_a_status_not_an_exception():
    def boom():
        raise OSError("no route to host")

    payload = upd.check_for_update("7.3.0", fetch=boom)
    assert payload["status"] == "unreachable"
    assert payload["available"] is False
    assert "no route to host" in payload["message"]


def test_a_response_without_a_tag_is_rejected():
    payload = upd.check_for_update("7.3.0", fetch=lambda: {"name": ""})
    assert payload["status"] == "invalid_response"
    assert payload["available"] is False


def test_disabled_by_default_and_never_calls_the_network(api, monkeypatch):
    from foundry import update as module

    calls = []

    def must_not_run(*args, **kwargs):
        calls.append("called")
        raise AssertionError("the update check must not run while it is disabled")

    monkeypatch.setattr(module, "check_for_update", must_not_run)
    response = api.get("/api/v1/update/check")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["status"] == "disabled"
    assert calls == []


def test_enabled_check_reports_the_available_version(api, monkeypatch):
    from foundry import update as module

    monkeypatch.setattr(
        module,
        "check_for_update",
        lambda current, **kwargs: {
            "status": "ok",
            "current": current,
            "latest": "v9.9.9",
            "available": True,
            "url": "https://example.invalid/latest",
            "message": "a newer release exists",
        },
    )
    patched = api.put("/api/v1/settings", json={"update_check": True})
    assert patched.status_code == 200
    assert patched.json()["effective"]["update_check"] is True

    body = api.get("/api/v1/update/check").json()
    assert body["enabled"] is True
    assert body["available"] is True
    assert body["latest"] == "v9.9.9"
    assert body["url"].startswith("https://")


def test_telemetry_still_cannot_be_enabled(api):
    response = api.put("/api/v1/settings", json={"telemetry": True})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "telemetry_not_implemented"
