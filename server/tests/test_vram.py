from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="training extras are not installed")

from foundry.training.vram import (  # noqa: E402
    BlockCache,
    BlockKey,
    budget_from_available,
    clear_arena,
    resolve_budget,
)


def _dequant_factory(rows: int, cols: int, calls: dict):
    import torch

    def dequantize(module: str, start: int, stop: int):
        calls["count"] = calls.get("count", 0) + 1
        # deterministic values so equality can be asserted
        base = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
        return base[start:stop].clone()

    return dequantize


def test_cache_returns_bit_identical_blocks_and_hits(tmp_path: Path):
    calls: dict = {}
    cache = BlockCache(
        dequantize=_dequant_factory(16, 8, calls),
        budget_bytes=1024**2,
        arena_dir=tmp_path / "arena",
        arena_bytes=1024**2,
        block_rows=4,
        prefetch=False,
    )
    first = cache.get("m", 0, 4)
    second = cache.get("m", 0, 4)
    assert calls["count"] == 1, "second lookup must not dequantize again"
    assert first.data_ptr() == second.data_ptr() or bool((first == second).all())

    fresh = _dequant_factory(16, 8, {})("m", 0, 4)
    assert bool((first == fresh).all())
    cache.close()


def test_hot_budget_is_enforced_and_evictions_go_to_the_arena(tmp_path: Path):
    import torch

    cache = BlockCache(
        dequantize=_dequant_factory(64, 32, {}),
        budget_bytes=8 * 4 * 32 * 4,  # room for two 4x32 fp32 blocks
        arena_dir=tmp_path / "arena",
        arena_bytes=1024**2,
        block_rows=4,
        prefetch=False,
    )
    for start in range(0, 64, 4):
        cache.get("m", start, start + 4)
    hot_bytes = sum(t.numel() * 4 for t in cache._hot.values())
    assert hot_bytes <= cache.budget_bytes
    stats = cache.stats_public()
    assert stats["evictions"] >= 1
    assert stats["arena_writes"] >= 1

    # an evicted block comes back from the arena, bit-identical
    again = cache.get("m", 0, 4)
    fresh = _dequant_factory(64, 32, {})("m", 0, 4)
    assert bool((again == fresh).all())
    assert cache.stats_public()["arena_hits"] >= 1
    cache.close()


def test_arena_reset_keeps_serving(tmp_path: Path):
    cache = BlockCache(
        dequantize=_dequant_factory(256, 64, {}),
        budget_bytes=4 * 64 * 4,  # one block in RAM
        arena_dir=tmp_path / "arena",
        arena_bytes=8 * 64 * 4 * 2,  # room for a couple of blocks only
        block_rows=4,
        prefetch=False,
    )
    for start in range(0, 256, 4):
        value = cache.get("m", start, start + 4)
        assert value.shape == (4, 64)
    stats = cache.stats_public()
    assert stats["arena_resets"] >= 1, "the arena must wrap instead of growing"
    cache.close()


def test_prefetch_warms_the_cache(tmp_path: Path):
    import time

    calls: dict = {}
    cache = BlockCache(
        dequantize=_dequant_factory(32, 16, calls),
        budget_bytes=1024**2,
        arena_dir=tmp_path / "arena",
        arena_bytes=1024**2,
        block_rows=4,
        prefetch=True,
    )
    cache.prefetch(cache.keys_for("m", 32))
    deadline = time.time() + 3
    while time.time() < deadline and cache.stats.prefetch_calls and calls.get("count", 0) < 8:
        time.sleep(0.02)
    before = calls.get("count", 0)
    for key in cache.keys_for("m", 32):
        cache.get(key.module, key.start, key.stop)
    assert calls.get("count", 0) == before, "prefetched blocks must not dequantize again"
    assert cache.stats_public()["ram_hits"] >= 8
    cache.close()


def test_closed_or_disabled_cache_still_serves(tmp_path: Path):
    calls: dict = {}
    cache = BlockCache(
        dequantize=_dequant_factory(8, 4, calls),
        budget_bytes=0,
        arena_dir=tmp_path / "arena",
        arena_bytes=0,
        block_rows=4,
        prefetch=False,
    )
    assert cache.get("m", 0, 4).shape == (4, 4)
    assert cache.get("m", 4, 8).shape == (4, 4)
    assert calls["count"] == 2, "without a budget every lookup dequantizes"
    cache.close()


def test_arena_index_is_written(tmp_path: Path):
    cache = BlockCache(
        dequantize=_dequant_factory(8, 4, {}),
        budget_bytes=1024,
        arena_dir=tmp_path / "arena",
        arena_bytes=1024**2,
        block_rows=4,
        prefetch=False,
    )
    cache.get("m", 0, 4)
    cache.close()
    payload = json.loads((tmp_path / "arena" / "arena.json").read_text("utf-8"))
    assert payload["arena_bytes"] == 1024**2
    assert "hit_rate" in payload["stats"]

    cleared = clear_arena(tmp_path / "arena")
    assert "arena.bin" in cleared["removed"]
    assert not (tmp_path / "arena" / "arena.bin").exists()


def test_budget_helpers():
    assert resolve_budget(None, "off") == 0
    assert resolve_budget(0, "auto") == 0
    assert resolve_budget(2, "auto") == 2 * 1024**3
    auto = budget_from_available()
    assert 64 * 1024**2 <= auto <= 3072 * 1024**2


def test_block_key_roundtrip():
    key = BlockKey("model.layers.0.mlp.down_proj", 0, 256)
    assert BlockKey.from_list(key.to_list()) == key
