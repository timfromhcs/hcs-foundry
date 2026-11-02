"""Warm the OS page cache for the INT4 store before the model faults it in.

The store is memory-mapped, so the first pass over 4.6 GB arrives as page faults — one small
read at a time, and on a slow external disk that dominates a training step (observed: 76 s/step
cold vs 25 s/step warm on the 4B). Page faults cannot be batched, but the *cache* can be filled
ahead of time by reading the files sequentially in large chunks from a background thread.

The prefetcher is a cache warmer, not a data source: it reads into a scratch buffer and drops it.
If it is slow, wrong or unavailable, training is unaffected — only the warm-up is.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

STORE_FILES = ("rest.f32", "weights.u8", "scales.f32")
CHUNK_BYTES = 4 * 1024**2


@dataclass
class PrefetchStats:
    bytes_read: int = 0
    chunks: int = 0
    duration_seconds: float = 0.0
    files: list[str] = field(default_factory=list)
    completed: bool = False
    error: str | None = None
    enabled: bool = True

    def to_public(self) -> dict:
        megabytes = self.bytes_read / 1024**2
        rate = megabytes / self.duration_seconds if self.duration_seconds else 0.0
        return {
            "enabled": self.enabled,
            "bytes_read": self.bytes_read,
            "mb_read": round(megabytes, 1),
            "chunks": self.chunks,
            "duration_seconds": round(self.duration_seconds, 2),
            "mb_per_second": round(rate, 1),
            "files": list(self.files),
            "completed": self.completed,
            "error": self.error,
        }


class StorePrefetcher:
    """Sequentially read the store files into the page cache, in a background thread.

    ``order`` decides which file is warmed first; the default follows the forward pass
    (embeddings from ``rest.f32`` before the quantized blocks).
    """

    def __init__(
        self,
        store_dir: str | Path,
        *,
        chunk_bytes: int = CHUNK_BYTES,
        order: tuple[str, ...] = STORE_FILES,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.chunk_bytes = max(64 * 1024, int(chunk_bytes))
        self.order = order
        self.log = log
        self.stats = PrefetchStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._buffer = bytearray(self.chunk_bytes)

    def start(self) -> "StorePrefetcher":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="store-prefetch", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        started = time.time()
        try:
            for name in self.order:
                if self._stop.is_set():
                    break
                path = self.store_dir / name
                if not path.exists():
                    continue
                self.stats.files.append(name)
                with open(path, "rb", buffering=0) as handle:
                    while not self._stop.is_set():
                        read = handle.readinto(self._buffer)
                        if not read:
                            break
                        self.stats.bytes_read += read
                        self.stats.chunks += 1
            self.stats.completed = not self._stop.is_set()
        except Exception as exc:  # noqa: BLE001 - a cache warmer must never break training
            self.stats.error = str(exc)[:200]
            if self.log:
                self.log(f"prefetch: stopped ({exc})")
        finally:
            self.stats.duration_seconds = time.time() - started
            if self.log:
                public = self.stats.to_public()
                self.log(
                    f"prefetch: {public['mb_read']} MB in {public['duration_seconds']} s "
                    f"({public['mb_per_second']} MB/s){'' if public['completed'] else ' — stopped early'}"
                )

    def stop(self, timeout: float = 2.0) -> "StorePrefetcher":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        return self

    def wait(self, timeout: float = 30.0) -> "StorePrefetcher":
        """Block until the warm-up finishes (or the timeout expires)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        return self

    def report(self) -> dict:
        return self.stats.to_public()

    def __enter__(self) -> "StorePrefetcher":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


def prefetch_disabled() -> PrefetchStats:
    """A report for runs that deliberately do not warm the cache."""
    stats = PrefetchStats(enabled=False)
    return stats
