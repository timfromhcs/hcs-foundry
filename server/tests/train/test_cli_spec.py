from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="the spec CLI trains a real (tiny) model")

from foundry.cli import main  # noqa: E402
from foundry.train.spec import Backend, Method, Profile, spec_for_profile  # noqa: E402


def _write_spec(tmp_path: Path, model_dir: Path, dataset_id: str, method: Method, backend: Backend) -> Path:
    spec = spec_for_profile(
        method=method,
        backend=backend,
        model_path=str(model_dir),
        dataset_id=dataset_id,
        profile=Profile.CI_TINY,
        hyper={"seq_len": 24, "max_steps": 1, "lora_r": 2, "lora_alpha": 4, "lr": 1e-3},
        method_args={"eval_every": 0},
    )
    path = tmp_path / "spec.yaml"
    path.write_text(spec.to_yaml(), "utf-8")
    return path


def test_train_plan_prints_the_spec_hash(settings, db, tmp_path: Path, monkeypatch, capsys):
    from tests.test_sft import make_tiny_model, seed_dataset

    monkeypatch.setenv("FOUNDRY_DATA_DIR", str(settings.data_dir))
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    spec_path = _write_spec(tmp_path, model_dir, dataset_id, Method.SFT, Backend.FP32_LORA)
    exit_code = main(["train", "plan", "--spec", str(spec_path)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "spec      sha256:" in output
    assert "preflight ok" in output


def test_train_plan_rejects_an_invalid_spec(settings, monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("FOUNDRY_DATA_DIR", str(settings.data_dir))
    bad = tmp_path / "bad.yaml"
    bad.write_text("method: sft\nbackend: fp32_lora\nhypers:\n  lr: 0.1\n", "utf-8")
    assert main(["train", "plan", "--spec", str(bad)]) != 0
    assert "invalid spec" in capsys.readouterr().err


def test_train_run_writes_a_v2_manifest(settings, db, tmp_path: Path, monkeypatch, capsys):
    from tests.test_sft import make_tiny_model, seed_dataset

    monkeypatch.setenv("FOUNDRY_DATA_DIR", str(settings.data_dir))
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    spec_path = _write_spec(tmp_path, model_dir, dataset_id, Method.INT4, Backend.INT4_STREAM)
    out_dir = tmp_path / "run"
    exit_code = main(["train", "run", "--spec", str(spec_path), "--out", str(out_dir), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    manifest = payload["manifest"]
    assert manifest["manifest_version"] == 2
    assert manifest["spec_hash"]
    assert manifest["param_delta_max"] > 0
    assert manifest["grads_finite"] is True
