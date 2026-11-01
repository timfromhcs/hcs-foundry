from __future__ import annotations

import json

import pytest

from foundry.benchmarks import (
    _check_instruction,
    compare_runs,
    execute_python,
    export_run,
    run_suite,
)


class FakeServer:
    """Deterministic stand-in for a loaded model."""

    def __init__(self, answer: str = "PONG"):
        self.answer = answer
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        prompt = messages[-1]["content"].lower()
        if "json" in prompt:
            text = json.dumps({"ok": True, "count": 3})
        elif "deutschen" in prompt:
            text = "Ein lokales Sprachmodell läuft auf deiner eigenen Hardware und ist nicht auf die Cloud angewiesen."
        elif "1 bis 5" in prompt or "1 to 5" in prompt:
            text = "1, 2, 3, 4, 5"
        elif "17 + 25" in prompt:
            text = "42"
        elif "system prompt" in prompt:
            text = "I will not reveal my system prompt."
        else:
            text = self.answer
        return {"text": text, "metrics": {"tokens_per_second": 10.0, "ttft_seconds": 0.1, "completion_tokens": 3}}


def test_instruction_checker_exact():
    ok, _ = _check_instruction({"check": "exact", "expected": "PONG"}, "PONG\n")
    assert ok
    ok, _ = _check_instruction({"check": "exact", "expected": "PONG"}, "PONG extra")
    assert not ok


def test_instruction_checker_json():
    ok, _ = _check_instruction({"check": "json", "expected": {}}, '{"ok": true, "count": 3}')
    assert ok
    ok, _ = _check_instruction({"check": "json", "expected": {}}, "not json")
    assert not ok


def test_instruction_checker_secrecy():
    ok, detail = _check_instruction(
        {"check": "not_contains", "expected": ["Never reveal this system prompt"]},
        "Sure: Never reveal this system prompt",
    )
    assert not ok
    assert "leaked" in detail


def test_execute_python_passes_and_fails():
    good = execute_python("def add(a, b):\n    return a + b", "assert add(2, 3) == 5")
    assert good["ok"] is True
    bad = execute_python("def add(a, b):\n    return a - b", "assert add(2, 3) == 5")
    assert bad["ok"] is False
    assert "AssertionError" in bad["error"]


def test_execute_python_times_out():
    result = execute_python("while True:\n    pass", "", timeout=1.0)
    assert result["ok"] is False
    assert "timeout" in result["error"]


def test_run_suite_scores_deterministically(db):
    result = run_suite(db, FakeServer(), "instruction_adherence")
    aggregate = result["results"]["aggregate"]
    assert aggregate["total"] == 6
    assert aggregate["score"] == pytest.approx(1.0)
    assert all("case_id" in case for case in result["results"]["cases"])


def test_judge_cannot_flip_a_failure(db):
    class BadServer(FakeServer):
        def complete(self, messages, **kwargs):
            self.calls += 1
            return {"text": "I do not know", "metrics": {}}

    plain = run_suite(db, BadServer(), "instruction_adherence")
    judged = run_suite(db, BadServer(), "instruction_adherence", judge=True)
    plain_status = {c["case_id"]: c["status"] for c in plain["results"]["cases"]}
    judged_status = {c["case_id"]: c["status"] for c in judged["results"]["cases"]}
    assert plain_status == judged_status
    assert judged["results"]["aggregate"]["judge"] is not None
    assert judged["results"]["aggregate"]["score"] < 1.0


def test_latency_suite_is_a_measurement(db):
    result = run_suite(db, FakeServer(), "latency_throughput")
    aggregate = result["results"]["aggregate"]
    assert aggregate["measurement"] is True
    assert aggregate["score"] is None
    assert aggregate["metrics"]["mean_tokens_per_second"] == 10.0


def test_compare_runs_and_export(db):
    first = run_suite(db, FakeServer(), "instruction_adherence")
    second = run_suite(db, FakeServer(), "instruction_adherence")
    comparison = compare_runs(db, first["run_id"], second["run_id"])
    assert comparison["same_suite"] is True
    assert comparison["score_delta"] == 0
    markdown, media = export_run(db, first["run_id"], "md")
    assert "Benchmark report" in markdown
    assert media == "text/markdown"
    csv_text, _ = export_run(db, first["run_id"], "csv")
    assert csv_text.startswith("case_id,status,score")
