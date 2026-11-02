from __future__ import annotations


def test_health(api):
    response = api.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_system_info_and_settings(api):
    info = api.get("/api/v1/system/info").json()
    assert info["local_only"] is True
    assert info["telemetry"] is False
    settings = api.get("/api/v1/settings").json()
    assert "effective" in settings
    assert isinstance(settings["effective"]["model_roots"], list)


def test_telemetry_cannot_be_enabled(api):
    response = api.put("/api/v1/settings", json={"telemetry": True})
    assert response.status_code == 400


def test_machine_budget_without_profile(api):
    response = api.get("/api/v1/machine/budget")
    assert response.status_code == 200
    body = response.json()
    assert body["safe_budget_bytes"] >= 0
    assert "notes" in body


def test_machine_profile_without_scan_is_an_empty_state(api):
    # "not scanned yet" must not be a 404: the browser logs a 404 resource load as a console
    # error on every fresh install, and the UI already renders a scan prompt for a null profile.
    response = api.get("/api/v1/machine/profile")
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] is None
    assert body["budget"] is None
    assert body["scanned"] is False


def test_models_list_empty(api):
    body = api.get("/api/v1/models").json()
    assert body["models"] == []


def test_datasets_list_empty(api):
    assert api.get("/api/v1/datasets").json()["datasets"] == []


def test_identity_flow(api):
    draft = api.post(
        "/api/v1/identity/drafts",
        json={"prompt": "You are Nova, a precise assistant. Always answer briefly. Never invent facts.", "name": "Nova"},
    )
    assert draft.status_code == 200
    profile_id = draft.json()["id"]
    approve = api.post(f"/api/v1/identity/{profile_id}/approve", json={})
    assert approve.status_code == 200
    dataset = api.post(f"/api/v1/identity/{profile_id}/dataset", json={})
    assert dataset.status_code == 200
    assert dataset.json()["record_count"] >= 7


def test_training_environment(api):
    body = api.get("/api/v1/training/environment").json()
    assert body["state"] in {"unsupported", "cpu-experimental", "gpu"}
    assert body["can_finetune_verified"] is False


def test_benchmark_requires_model(api):
    response = api.post("/api/v1/benchmarks/runs", json={"suite_id": "instruction_adherence"})
    assert response.status_code == 409


def test_agent_suggest_mode_does_not_run(api):
    task = api.post("/api/v1/agent/tasks", json={"goal": "inspect the machine", "autonomy": "suggest"}).json()["task"]
    assert task["status"] == "suggested"
    assert task["plan"]


def test_unknown_route_returns_404(api):
    assert api.get("/api/v1/does-not-exist").status_code == 404


def test_settings_survive_a_restart(settings):
    from fastapi.testclient import TestClient

    from foundry.config import Settings
    from foundry.main import create_app

    def client_for(config: Settings):
        app = create_app(config, serve_web=False)
        client = TestClient(app, base_url="http://127.0.0.1:8770")
        token = client.post(
            "/api/v1/system/pair", json={"label": "t"}, headers={"X-Foundry-Client": "1"}
        ).json()["token"]
        client.headers.update({"Authorization": f"Bearer {token}", "X-Foundry-Client": "1"})
        return client

    first = client_for(settings)
    response = first.put(
        "/api/v1/settings",
        json={"model_roots": ["/tmp/one", "/tmp/two"], "os_reserve_ratio": 0.2},
    )
    assert response.status_code == 200
    first.close()

    restarted = Settings(data_dir=settings.data_dir, import_roots=settings.import_roots)
    restarted.ensure_dirs()
    second = client_for(restarted)
    effective = second.get("/api/v1/settings").json()["effective"]
    from pathlib import Path

    assert [Path(p) for p in effective["model_roots"]] == [Path("/tmp/one"), Path("/tmp/two")]
    assert abs(float(effective["os_reserve_ratio"]) - 0.2) < 1e-9
    second.close()
