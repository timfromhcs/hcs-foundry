"""Spec-driven training over HTTP: plan is free, run is queued, blockers are 409."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from foundry.train.spec import Backend, Method, Profile, spec_for_profile

pytest.importorskip("transformers", reason="the plan endpoint runs the real preflight")


def _spec(tmp_path: Path, model_dir: Path, dataset_id: str, *, steps: int = 1) -> dict:
    return spec_for_profile(
        method=Method.SFT,
        backend=Backend.FP32_LORA,
        model_path=str(model_dir),
        dataset_id=dataset_id,
        profile=Profile.CI_TINY,
        hyper={"seq_len": 24, "max_steps": steps, "lora_r": 2, "lora_alpha": 4, "lr": 1e-3},
        method_args={"eval_every": 0},
    ).model_dump(mode="json")


def test_plan_reports_the_trainer_and_preflight(api, db, tmp_path: Path):
    from tests.test_sft import make_tiny_model, seed_dataset

    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    response = api.post("/api/v1/training/plan", json={"spec": _spec(tmp_path, model_dir, dataset_id)})
    assert response.status_code == 200
    body = response.json()
    assert body["trainer"] == "sft/fp32_lora"
    assert body["preflight"]["ok"] is True
    assert body["spec_hash"].startswith("sha256:")
    assert len(body["spec_hash"]) == 71
    assert body["spec"]["method"] == "sft"


def test_plan_is_honest_about_a_missing_model(api, db, tmp_path: Path):
    from tests.test_sft import seed_dataset

    dataset_id = seed_dataset(db)
    response = api.post("/api/v1/training/plan", json={"spec": _spec(tmp_path, tmp_path / "nope", dataset_id)})
    assert response.status_code == 200
    body = response.json()
    assert body["preflight"]["ok"] is False
    assert any("not found" in blocker for blocker in body["preflight"]["blockers"])


def test_plan_rejects_a_spec_that_violates_the_schema(api):
    response = api.post("/api/v1/training/plan", json={"spec": {"method": "sft"}})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_spec"


def test_run_refuses_to_queue_a_blocked_spec(api, db, tmp_path: Path):
    from tests.test_sft import seed_dataset

    dataset_id = seed_dataset(db)
    response = api.post("/api/v1/training/runs", json={"spec": _spec(tmp_path, tmp_path / "nope", dataset_id)})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "preflight_blocked"
    assert detail["blockers"]


def test_run_queues_and_completes_a_tiny_spec(api, db, tmp_path: Path):
    from tests.test_sft import make_tiny_model, seed_dataset

    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    response = api.post("/api/v1/training/runs", json={"spec": _spec(tmp_path, model_dir, dataset_id)})
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"]
    assert body["trainer"] == "sft/fp32_lora"

    job_id = body["job_id"]
    deadline = time.time() + 120
    status = None
    while time.time() < deadline:
        job = api.get(f"/api/v1/runs/{job_id}").json()["run"]
        status = job.get("status")
        if status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.5)
    detail = api.get(f"/api/v1/runs/{job_id}").json()
    assert status == "succeeded", detail
    assert detail["run"]["output"]["manifest"]["status"] == "completed"
    assert detail["run"]["output"]["manifest"]["steps"] >= 1
    assert detail["run"]["output"]["manifest"]["adapter_dir"]
