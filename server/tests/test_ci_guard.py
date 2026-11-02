from __future__ import annotations

import pytest

from tests.conftest import TRAINING_MODULES, pytest_collection_modifyitems


class _FakeModule:
    def __init__(self, name: str) -> None:
        self.__name__ = f"tests.{name}"


class _FakeItem:
    def __init__(self, module: str) -> None:
        self.module = _FakeModule(module)


def test_guard_requires_every_training_module(monkeypatch):
    monkeypatch.setenv("FOUNDRY_REQUIRE_TRAIN", "1")
    items = [_FakeItem(name) for name in sorted(TRAINING_MODULES)]
    pytest_collection_modifyitems(None, items)  # all present → no raise


def test_guard_fires_when_a_training_module_is_missing(monkeypatch):
    monkeypatch.setenv("FOUNDRY_REQUIRE_TRAIN", "1")
    items = [_FakeItem(name) for name in sorted(TRAINING_MODULES)[1:]]
    with pytest.raises(pytest.UsageError) as excinfo:
        pytest_collection_modifyitems(None, items)
    assert "not collected" in str(excinfo.value)


def test_guard_is_silent_without_the_flag(monkeypatch):
    monkeypatch.delenv("FOUNDRY_REQUIRE_TRAIN", raising=False)
    pytest_collection_modifyitems(None, [])  # local runs are allowed to skip


def test_training_modules_are_the_ones_that_need_the_ml_stack():
    assert "test_int4" in TRAINING_MODULES
    assert "test_sft" in TRAINING_MODULES
    assert "test_api" not in TRAINING_MODULES
