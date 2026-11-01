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
