from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from foundry.security import PairingStore, PathNotAllowed, resolve_within


def test_path_allowlist_rejects_traversal(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    with pytest.raises(PathNotAllowed):
        resolve_within(outside, [root])
    with pytest.raises(PathNotAllowed):
        resolve_within(root / ".." / "secret.txt", [root])
    assert resolve_within(root / "inside.txt", [root]) == (root / "inside.txt").resolve()


def test_pairing_tokens_are_hashed_and_revocable(tmp_path: Path):
    store = PairingStore(tmp_path / "state")
    token = store.issue("test")
    assert token.startswith("fnd_")
    assert store.validate(token)
    assert token not in (tmp_path / "state" / "pairing.json").read_text("utf-8")
    assert store.revoke_all() == 1
    assert not store.validate(token)


def test_pairing_code_flow(tmp_path: Path):
    store = PairingStore(tmp_path / "state", require_code=True)
    code = store.pairing_code()
    assert code and len(code) == 6
    assert store.verify_code(code)
    assert not store.verify_code("000000" if code != "000000" else "111111")


def test_missing_token_is_rejected(api):
    raw = TestClient(api.app, base_url="http://127.0.0.1:8770")
    response = raw.get("/api/v1/settings")
    assert response.status_code == 401


def test_foreign_origin_is_rejected(api):
    response = api.get("/api/v1/settings", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "origin_not_allowed"


def test_foreign_host_is_rejected(api):
    raw = TestClient(api.app, base_url="http://evil.example")
    raw.headers.update(dict(api.headers))
    response = raw.get("/api/v1/settings")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "host_not_allowed"


def test_mutating_request_requires_client_header(api):
    response = api.post("/api/v1/system/revoke", json={}, headers={"X-Foundry-Client": ""})
    assert response.status_code == 403


def test_mutating_request_requires_json(api):
    response = api.post(
        "/api/v1/system/revoke",
        content=b"x=1",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 415


def test_health_is_public(api):
    raw = TestClient(api.app, base_url="http://127.0.0.1:8770")
    assert raw.get("/api/v1/health").status_code == 200
