"""Three-tier weight cache: hot RAM, warm disk arena, cold INT4 store.

The INT4 path dequantizes a block, uses it, and throws it away; the next step does the same work
again. This module keeps those fp32 blocks around in a bounded way:

* **hot** — anonymous RAM, LRU, hard byte budget (the part that costs RAM),
* **warm** — a sparse memory-mapped arena file on disk, written through on eviction,
* **cold** — the INT4 store itself, dequantized on demand.

A hit in the warm tier is a memcpy instead of dequantization, and a hit in hot is free. Nothing
here is a source of truth: the arena may be cleared at any time and the cache falls back to the
store. The budget is enforced, so a run cannot silently fill the disk.
"""

from __future__ import annotations

import json
import mmap
import os
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ARENA_NAME = "arena.bin"
INDEX_NAME = "arena.json"
GROWTH_CHUNK = 64 * 1024**2
DEFAULT_ARENA_GB = 6.0
DEFAULT_HOT_FRACTION = 0.25
DEFAULT_HOT_MIN_MB = 128
DEFAULT_HOT_MAX_MB = 3072


def _torch():
    import torch

    return torch


@dataclass(frozen=True)
class BlockKey:
    module: str
    start: int
    stop: int

    def to_list(self) -> list:
        return [self.module, self.start, self.stop]

    @staticmethod
    def from_list(value) -> "BlockKey":
        return BlockKey(str(value[0]), int(value[1]), int(value[2]))


@dataclass
class CacheStats:
    ram_hits: int = 0
    arena_hits: int = 0
    misses: int = 0
    evictions: int = 0
    arena_writes: int = 0
    arena_bytes_written: int = 0
    arena_bytes_read: int = 0
    dequant_calls: int = 0
    dequant_seconds: float = 0.0
    prefetch_calls: int = 0
    prefetch_used: int = 0
    arena_resets: int = 0

    def to_public(self, *, budget_bytes: int, arena_bytes: int) -> dict:
        lookups = self.ram_hits + self.arena_hits + self.misses
        hit_rate = (self.ram_hits + self.arena_hits) / lookups if lookups else 0.0
        return {
            "budget_bytes": budget_bytes,
            "arena_bytes": arena_bytes,
            "lookups": lookups,
            "ram_hits": self.ram_hits,
            "arena_hits": self.arena_hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self.evictions,
            "arena_writes": self.arena_writes,
            "arena_mb_written": round(self.arena_bytes_written / 1024**2, 1),
            "arena_mb_read": round(self.arena_bytes_read / 1024**2, 1),
            "dequant_calls": self.dequant_calls,
            "dequant_seconds": round(self.dequant_seconds, 3),
            "prefetch_calls": self.prefetch_calls,
            "prefetch_used": self.prefetch_used,
            "arena_resets": self.arena_resets,
        }


class BlockCache:
    """Bounded fp32 block cache with a disk-backed warm tier.

    ``dequantize(module_path, start, stop)`` must return an fp32 tensor of ``stop - start`` rows.
    """

    def __init__(
        self,
        *,
        dequantize: Callable[[str, int, int], "object"],
        budget_bytes: int,
        arena_dir: str | Path,
        arena_bytes: int,
        block_rows: int = 256,
        prefetch: bool = True,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.dequantize = dequantize
        self.budget_bytes = max(0, int(budget_bytes))
        self.arena_limit = max(0, int(arena_bytes))
        self.block_rows = max(1, int(block_rows))
        self.log = log
        self.enabled_prefetch = bool(prefetch)
        self._lock = threading.RLock()
        self._hot: "OrderedDict[BlockKey, object]" = OrderedDict()
        self._hot_bytes = 0
        self._arena_offsets: dict[BlockKey, list] = {}
        self._arena_epoch = 0
        self._arena_cursor = 0
        self._closed = False
        self.stats = CacheStats()
        self._pending: set[BlockKey] = set()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_queue: list[BlockKey] = []
        self._prefetch_stop = threading.Event()

        self.arena_dir = Path(arena_dir).expanduser()
        self.arena_path = self.arena_dir / ARENA_NAME
        self._file = None
        self._map = None
        self._map_size = 0
        if self.arena_limit > 0:
            self._open_arena()
        if self.enabled_prefetch:
            self._prefetch_thread = threading.Thread(target=self._prefetch_loop, name="vram-prefetch", daemon=True)
            self._prefetch_thread.start()

    # ------------------------------------------------------------------ arena
    def _open_arena(self) -> None:
        try:
            self.arena_dir.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.arena_dir).free
            if free < self.arena_limit + 1024**3:
                size = max(0, free - 1024**3)
                if self.log:
                    self.log(f"vram: only {free / 1024**3:.1f} GiB free, arena limited to {size / 1024**3:.1f} GiB")
                self.arena_limit = size
            if self.arena_limit <= 0:
                return
            self._file = open(self.arena_path, "w+b")
            # Grow on demand: pre-truncating a multi-GB file costs tens of seconds on NTFS and
            # would be paid by every run even when the cache barely fills.
            self._map_size = 0
            self._ensure_capacity(min(self.arena_limit, GROWTH_CHUNK))
        except OSError as exc:  # noqa: BLE001 - the cache must never break a run
            if self.log:
                self.log(f"vram: arena disabled ({exc})")
            self.arena_limit = 0
            self._map = None

    def _ensure_capacity(self, needed: int) -> bool:
        """Grow the arena file/mapping so ``needed`` bytes fit, or return False."""
        if self._file is None:
            return False
        needed = min(needed, self.arena_limit)
        if needed <= self._map_size and self._map is not None:
            return True
        target = max(needed, min(self.arena_limit, self._map_size + GROWTH_CHUNK))
        target = min(target, self.arena_limit)
        if target < needed:
            return False
        try:
            if self._map is not None:
                self._map.close()
                self._map = None
            self._file.truncate(target)
            self._map = mmap.mmap(self._file.fileno(), target)
            self._map_size = target
            return True
        except (OSError, ValueError) as exc:  # noqa: BLE001
            if self.log:
                self.log(f"vram: arena grow failed ({exc})")
            self._map = None
            self._map_size = 0
            return False

    def _arena_alloc(self, size: int) -> int | None:
        if self._file is None or size <= 0 or size > self.arena_limit:
            return None
        if self._arena_cursor + size > self.arena_limit:
            self.stats.arena_resets += 1
            self._arena_epoch += 1
            self._arena_cursor = 0
            self._arena_offsets.clear()
        offset = self._arena_cursor
        if not self._ensure_capacity(offset + size):
            return None
        self._arena_cursor += size
        return offset

    def _arena_put(self, key: BlockKey, tensor) -> None:
        if self._map is None:
            return
        payload = tensor.detach().contiguous()
        size = payload.numel() * payload.element_size()
        offset = self._arena_alloc(size)
        if offset is None:
            return
        try:
            self._map[offset : offset + size] = payload.numpy().tobytes()
        except (ValueError, BufferError) as exc:  # noqa: BLE001
            if self.log:
                self.log(f"vram: arena write skipped ({exc})")
            return
        self._arena_offsets[key] = [offset, list(payload.shape), str(payload.dtype), self._arena_epoch]
        self.stats.arena_writes += 1
        self.stats.arena_bytes_written += size

    def _arena_get(self, key: BlockKey):
        entry = self._arena_offsets.get(key)
        if not entry or self._map is None:
            return None
        offset, shape, dtype_name, epoch = entry
        if epoch != self._arena_epoch:
            self._arena_offsets.pop(key, None)
            return None
        torch = _torch()
        dtype = getattr(torch, dtype_name.split(".")[-1])
        count = 1
        for dim in shape:
            count *= dim
        size = count * 4
        try:
            buffer = torch.frombuffer(self._map, dtype=torch.uint8, count=size, offset=offset)
        except (TypeError, ValueError):
            return None
        # copy out: a live tensor must never alias an arena region that can be reset
        tensor = buffer.view(dtype).view(*shape).clone()
        self.stats.arena_bytes_read += size
        return tensor

    # ------------------------------------------------------------------ cache
    def _store_hot(self, key: BlockKey, tensor) -> None:
        size = tensor.numel() * tensor.element_size()
        if self.budget_bytes and self._hot_bytes + size > self.budget_bytes:
            while self._hot and self._hot_bytes + size > self.budget_bytes:
                old_key, old_tensor = self._hot.popitem(last=False)
                self._hot_bytes -= old_tensor.numel() * old_tensor.element_size()
                self.stats.evictions += 1
                self._arena_put(old_key, old_tensor)
                del old_tensor
        if self.budget_bytes and size > self.budget_bytes:
            # a single block larger than the budget: use it, do not cache it
            self._arena_put(key, tensor)
            return
        self._hot[key] = tensor
        self._hot_bytes += size

    def get(self, module_path: str, start: int, stop: int):
        key = BlockKey(module_path, int(start), int(stop))
        with self._lock:
            tensor = self._hot.get(key)
            if tensor is not None:
                self._hot.move_to_end(key)
                self.stats.ram_hits += 1
                if key in self._pending:
                    self._pending.discard(key)
                    self.stats.prefetch_used += 1
                return tensor
            tensor = self._arena_get(key)
            if tensor is not None:
                self.stats.arena_hits += 1
                self._store_hot(key, tensor)
                return tensor
            self.stats.misses += 1
            started = time.time()
            tensor = self.dequantize(module_path, key.start, key.stop)
            self.stats.dequant_calls += 1
            self.stats.dequant_seconds += time.time() - started
            self._store_hot(key, tensor)
            return tensor

    def keys_for(self, module_path: str, out_features: int, block: int | None = None) -> list[BlockKey]:
        step = block or self.block_rows
        return [BlockKey(module_path, start, min(start + step, out_features)) for start in range(0, out_features, step)]

    def prefetch(self, keys: list[BlockKey]) -> None:
        if not self.enabled_prefetch or self._closed:
            return
        with self._lock:
            for key in keys:
                if key in self._hot or key in self._arena_offsets or key in self._pending:
                    continue
                self._pending.add(key)
                self._prefetch_queue.append(key)
                self.stats.prefetch_calls += 1

    def _prefetch_loop(self) -> None:
        while not self._prefetch_stop.is_set():
            with self._lock:
                key = self._prefetch_queue.pop(0) if self._prefetch_queue else None
            if key is None:
                time.sleep(0.002)
                continue
            try:
                with self._lock:
                    known = key in self._hot or key in self._arena_offsets
                    self._pending.discard(key)
                if known:
                    continue
                started = time.time()
                tensor = self.dequantize(key.module, key.start, key.stop)
                with self._lock:
                    self.stats.dequant_calls += 1
                    self.stats.dequant_seconds += time.time() - started
                    self._store_hot(key, tensor)
            except Exception:  # noqa: BLE001 - prefetch must never kill a run
                with self._lock:
                    self._pending.discard(key)

    # ---------------------------------------------------------------- control
    def stats_public(self) -> dict:
        with self._lock:
            payload = self.stats.to_public(budget_bytes=self.budget_bytes, arena_bytes=self.arena_limit)
            payload["arena_mapped_bytes"] = self._map_size
            return payload

    def write_index(self) -> None:
        """Record what the arena holds, for inspection (not needed to run)."""
        with self._lock:
            payload = {
                "epoch": self._arena_epoch,
                "cursor": self._arena_cursor,
                "arena_bytes": self.arena_limit,
                "entries": len(self._arena_offsets),
                "stats": self.stats_public(),
            }
        try:
            self.arena_dir.mkdir(parents=True, exist_ok=True)
            (self.arena_dir / INDEX_NAME).write_text(json.dumps(payload, indent=2), "utf-8")
        except OSError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._prefetch_stop.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=2.0)
            self._prefetch_thread = None
        self.write_index()
        with self._lock:
            self._hot.clear()
            self._hot_bytes = 0
            self._arena_offsets.clear()
        if self._map is not None:
            try:
                self._map.close()
            except (BufferError, ValueError):
                pass
            self._map = None
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def __enter__(self) -> "BlockCache":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def budget_from_available(psutil_module=None) -> int:
    """Hot-tier budget from *live* available RAM (never more than a quarter of it)."""
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return DEFAULT_HOT_MIN_MB * 1024**2
    available = psutil.virtual_memory().available
    budget = int(available * DEFAULT_HOT_FRACTION)
    lower = DEFAULT_HOT_MIN_MB * 1024**2
    upper = DEFAULT_HOT_MAX_MB * 1024**2
    return max(lower, min(upper, budget))


def resolve_budget(vram_gb: float | None, mode: str = "auto") -> int:
    if mode == "off" or (vram_gb is not None and vram_gb <= 0):
        return 0
    if vram_gb:
        return int(float(vram_gb) * 1024**3)
    return budget_from_available()


def clear_arena(arena_dir: str | Path) -> dict:
    base = Path(arena_dir).expanduser()
    removed = []
    for name in (ARENA_NAME, INDEX_NAME):
        path = base / name
        if path.exists():
            try:
                os.remove(path)
                removed.append(name)
            except OSError:
                pass
    return {"arena_dir": str(base), "removed": removed}
