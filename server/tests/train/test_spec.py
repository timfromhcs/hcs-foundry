from __future__ import annotations

import pytest
from pydantic import ValidationError

from foundry.train.spec import (
    Backend,
    Method,
    Profile,
    TrainSpec,
    spec_for_profile,
)


def _minimal(**overrides):
    payload = {
        "method": "sft",
        "backend": "fp32_lora",
        "model": {"path": "/models/qwen"},
        "data": {"dataset_id": "ds_1"},
        "hyper": {"lr": 2e-4, "lora_r": 4, "lora_alpha": 16},
    }
    payload.update(overrides)
    return TrainSpec.model_validate(payload)


def test_valid_spec_roundtrips_through_json_and_yaml():
    spec = _minimal()
    again = TrainSpec.from_dict(spec.model_dump(mode="json"))
    assert again.spec_hash == spec.spec_hash
    yaml_text = spec.to_yaml()
    assert "method: sft" in yaml_text
    assert "dataset_id: ds_1" in yaml_text


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError) as error:
        _minimal(hypers={"lr": 1e-3})  # typo: should be 'hyper'
    assert "hypers" in str(error.value)


def test_unknown_hyper_key_is_rejected():
    with pytest.raises(ValidationError):
        _minimal(hyper={"lr": 2e-4, "lora_r": 4, "lora_alpha": 16, "learningrate": 1e-3})


def test_spec_hash_is_stable_under_key_order():
    a = _minimal()
    payload = {"hyper": {"lora_alpha": 16, "lora_r": 4, "lr": 2e-4},
               "data": {"dataset_id": "ds_1"},
               "model": {"path": "/models/qwen"},
               "backend": "fp32_lora", "method": "sft"}
    b = TrainSpec.model_validate(payload)
    assert a.spec_hash == b.spec_hash


def test_int4_requires_an_int4_backend():
    with pytest.raises(ValidationError):
        _minimal(method="int4", backend="fp32_lora")


def test_remote_backend_requires_the_remote_profile():
    with pytest.raises(ValidationError):
        _minimal(backend="remote", profile="local-smoke")


def test_profile_defaults_produce_a_valid_spec():
    spec = spec_for_profile(
        method=Method.INT4,
        backend=Backend.INT4_STORE,
        model_path="/models/qwen",
        dataset_id="ds_1",
        profile=Profile.CI_TINY,
    )
    assert spec.hyper.max_steps == 2
    assert spec.hyper.seq_len == 24
    assert spec.budget.max_wall_s == 120


def test_hyper_bounds_are_enforced():
    with pytest.raises(ValidationError):
        _minimal(hyper={"lr": 0, "lora_r": 4, "lora_alpha": 16})
    with pytest.raises(ValidationError):
        _minimal(hyper={"lr": 2e-4, "lora_r": 4, "lora_alpha": 16, "max_steps": -1})
