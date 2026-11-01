from __future__ import annotations

import json
from pathlib import Path

from foundry.pipeline import run_pipeline


def test_report_stage_writes_json_and_markdown(db, settings):
    result = run_pipeline(db, settings, stages=["report"], options={})
    report = Path(result["artifacts"]["report"])
    assert report.exists()
    assert report.with_suffix(".md").exists()
    payload = json.loads(report.read_text("utf-8"))
    assert payload["run_id"] == result["run_id"]


def test_failed_stage_is_recorded_and_stops(db, settings):
    result = run_pipeline(db, settings, stages=["quantize", "report"], options={})
    stages = {stage["stage"]: stage for stage in result["stages"]}
    assert stages["quantize"]["status"] == "failed"
    assert "gguf_f16" in stages["quantize"]["error"]
    assert "report" not in stages


def test_continue_on_error_runs_next_stage(db, settings):
    result = run_pipeline(db, settings, stages=["quantize", "report"], options={"continue_on_error": True})
    stages = {stage["stage"]: stage for stage in result["stages"]}
    assert stages["quantize"]["status"] == "failed"
    assert stages["report"]["status"] == "ok"


def test_unknown_stage_is_reported(db, settings):
    result = run_pipeline(db, settings, stages=["nonsense"], options={})
    assert result["stages"][0]["status"] == "failed"
    assert "unknown stage" in result["stages"][0]["error"]
