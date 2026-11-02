"""The spec-driven MCP surfaces: train_plan and eval_compare.

Both are typed, path-allowlisted and never run a shell. eval_compare refuses an invalid pairing
instead of reporting a number that cannot be trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.mcp_server import McpServer


@pytest.fixture()
def server(db, settings):
    from foundry.events import EventBus
    from foundry.jobs import JobRunner
    from foundry.runtime.llama_server import LlamaServer
    from foundry.security import PairingStore
    from foundry.state import AppState

    state = AppState(
        settings=settings,
        db=db,
        bus=EventBus(),
        runner=JobRunner(db, EventBus()),
        server=LlamaServer(settings.state_dir),
        pairing=PairingStore(settings.state_dir),
    )
    return McpServer(state, log=lambda message: None)


def _call(server: McpServer, name: str, arguments: dict):
    return server.handle(
        {"jsonrpc": "2.0", "id": 99, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    )


def test_train_plan_returns_a_preflight(server, db, tmp_path: Path):
    pytest.importorskip("transformers")
    from foundry.train.spec import Backend, Method, Profile, spec_for_profile
    from tests.test_sft import make_tiny_model, seed_dataset

    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    spec = spec_for_profile(
        method=Method.SFT,
        backend=Backend.FP32_LORA,
        model_path=str(model_dir),
        dataset_id=dataset_id,
        profile=Profile.CI_TINY,
        hyper={"seq_len": 24, "max_steps": 1, "lora_r": 2, "lora_alpha": 4, "lr": 1e-3},
        method_args={"eval_every": 0},
    ).model_dump(mode="json")

    response = _call(server, "train_plan", {"spec": spec})
    assert "result" in response, response
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["trainer"] == "sft/fp32_lora"
    assert payload["preflight"]["ok"] is True
    assert payload["spec_hash"].startswith("sha256:")


def test_train_plan_reports_a_schema_error_as_a_typed_error(server):
    response = _call(server, "train_plan", {"spec": {"method": "sft"}})
    assert response["error"]["code"] == -32602
    assert "invalid spec" in response["error"]["message"]


def _seed_benchmark_run(db, run_id: str, suite_id: str, outcomes: dict[str, str], version: str = "1.0") -> None:
    db.insert(
        "benchmark_runs",
        {
            "id": run_id,
            "suite_id": suite_id,
            "suite_version": version,
            "model_id": "m",
            "model_hash": "sha256:x",
            "status": "completed",
            "comparability": "project-specific smoke",
            "config": "{}",
            "results": "{}",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    for index, (case_id, status) in enumerate(outcomes.items()):
        db.insert(
            "benchmark_cases",
            {
                "id": f"{run_id}_{index}",
                "run_id": run_id,
                "case_id": case_id,
                "status": status,
                "score": 1.0 if status == "pass" else 0.0,
                "detail": "{}",
            },
        )


def test_eval_compare_pairs_cases_and_lists_regressions(server, db):
    base = {f"c{i}": ("pass" if i < 20 else "fail") for i in range(40)}
    cand = dict(base)
    for i in range(8):
        cand[f"c{i}"] = "fail"
    for i in range(20, 30):
        cand[f"c{i}"] = "pass"
    _seed_benchmark_run(db, "br_base", "instruction", base)
    _seed_benchmark_run(db, "br_cand", "instruction", cand)

    response = _call(server, "eval_compare", {"baseline_run_id": "br_base", "candidate_run_id": "br_cand"})
    assert "result" in response, response
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["n"] == 40
    assert payload["improvements"] == 10
    assert payload["regressions"] == 8
    assert payload["paired"]["n"] == 40
    assert len(payload["regression_cases"]) == 8
    assert payload["suite_id"] == "instruction"


def test_eval_compare_refuses_mismatched_suites_and_unknown_runs(server, db):
    _seed_benchmark_run(db, "br_a", "instruction", {"c1": "pass"})
    _seed_benchmark_run(db, "br_b", "agentic", {"c1": "pass"})

    mismatch = _call(server, "eval_compare", {"baseline_run_id": "br_a", "candidate_run_id": "br_b"})
    assert mismatch["error"]["code"] == -32602
    assert "different suites" in mismatch["error"]["message"]

    missing = _call(server, "eval_compare", {"baseline_run_id": "br_a", "candidate_run_id": "br_nope"})
    assert missing["error"]["code"] == -32602
    assert "not found" in missing["error"]["message"]
