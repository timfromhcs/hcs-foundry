from __future__ import annotations

from foundry.training import detect_environment, preflight, run_smoke


def test_detect_environment_is_truthful(settings):
    env = detect_environment(settings)
    assert env["state"] in {"unsupported", "cpu-experimental", "gpu"}
    assert isinstance(env["packages"], dict)
    assert env["can_finetune_verified"] is False
    if env["state"] != "gpu":
        assert env["device"] == "cpu"


def test_smoke_requires_a_real_parameter_update():
    result = run_smoke()
    if not result.ok:
        assert "not installed" in result.reason or "no parameter" in result.reason
        return
    assert result.metrics["max_parameter_delta"] > 0
    assert result.metrics["loss_last"] < result.metrics["loss_first"]
    assert result.metrics["steps"] == 40


def test_preflight_blocks_unsupported_machine(db, settings):
    plan = preflight(db, settings, dataset_id="missing", base_model=None)
    assert plan["can_start"] is False
    assert plan["blockers"]
    assert any("dataset not found" in blocker for blocker in plan["blockers"])


def test_preflight_reports_missing_finetune_stack(db, settings, fixtures_dir):
    from foundry.data.pipeline import ImportOptions, freeze_dataset, import_dataset

    result = import_dataset(
        db,
        paths=[str(fixtures_dir / "generic_messages.jsonl")],
        import_roots=[fixtures_dir],
        options=ImportOptions(name="Train", provider="auto", rights_status="confirmed"),
    )
    freeze_dataset(db, result["dataset_id"])
    plan = preflight(db, settings, dataset_id=result["dataset_id"], base_model="C:/models/base")
    env = plan["environment"]
    if not env["can_finetune"]:
        assert plan["can_start"] is False
        assert any("fine-tuning" in blocker for blocker in plan["blockers"])
    assert plan["preset"]["micro_batch_size"] >= 1
