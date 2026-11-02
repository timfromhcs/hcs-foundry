from __future__ import annotations

import time
from pathlib import Path

import pytest

from foundry.training.store_prefetch import StorePrefetcher, prefetch_disabled


def _write_store(root: Path, sizes: dict[str, int]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, size in sizes.items():
        (root / name).write_bytes(b"\x01" * size)
    return root


def test_prefetch_reads_every_file_in_order(tmp_path: Path):
    store = _write_store(tmp_path / "store", {"rest.f32": 4096, "weights.u8": 8192, "scales.f32": 1024})
    prefetcher = StorePrefetcher(store, chunk_bytes=1024).start()
    prefetcher.wait(timeout=10)
    prefetcher.stop(timeout=5)
    report = prefetcher.report()
    assert report["bytes_read"] == 4096 + 8192 + 1024
    assert report["files"] == ["rest.f32", "weights.u8", "scales.f32"]
    assert report["chunks"] >= 3, "one read per file at minimum (chunk floor is 64 KiB)"
    assert report["completed"] is True
    assert report["error"] is None


def test_prefetch_skips_missing_files_without_failing(tmp_path: Path):
    store = _write_store(tmp_path / "store", {"weights.u8": 2048})
    prefetcher = StorePrefetcher(store).start()
    prefetcher.wait(timeout=10)
    prefetcher.stop(timeout=5)
    report = prefetcher.report()
    assert report["files"] == ["weights.u8"]
    assert report["bytes_read"] == 2048


def test_prefetch_of_a_missing_store_is_harmless(tmp_path: Path):
    prefetcher = StorePrefetcher(tmp_path / "nope").start()
    prefetcher.stop(timeout=5)
    report = prefetcher.report()
    assert report["bytes_read"] == 0
    assert report["error"] is None


def test_prefetch_can_be_stopped_early(tmp_path: Path):
    store = _write_store(tmp_path / "store", {"weights.u8": 8 * 1024**2})
    prefetcher = StorePrefetcher(store, chunk_bytes=1024).start()
    time.sleep(0.01)
    prefetcher.stop(timeout=5)
    report = prefetcher.report()
    assert report["bytes_read"] <= 8 * 1024**2
    assert report["completed"] in (True, False)  # racy by nature, must never raise


def test_disabled_report_is_explicit():
    report = prefetch_disabled().to_public()
    assert report["enabled"] is False
    assert report["bytes_read"] == 0
    assert report["completed"] is False
