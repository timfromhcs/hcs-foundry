"""Measured memory for training runs.

Estimates are not evidence. Every training manifest carries the *measured* peak RSS of
the worker process and the lowest available system memory observed during the run, so a
later reader can compare the prediction in ``resources`` with what actually happened.
"""

from __future__ import annotations

import threading
import time


class PeakRSS:
    """Sample this process' RSS and the system's available memory in a background thread.

    Usage::

        with PeakRSS() as memory:
            ...work...
        memory.report()  # {"peak_rss_mb": ..., "min_available_mb": ..., ...}

    The sampler never raises: a failing probe degrades to the values sampled so far.
    """

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = max(0.01, float(interval))
        self.start_rss_mb = 0.0
        self.peak_rss_mb = 0.0
        self.peak_private_mb: float | None = None
        self.min_available_mb: float | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil = None
        self._process = None

    def _snapshot(self) -> None:
        info = self._process.memory_info()
        rss = info.rss / 1024**2
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        # ``private`` is Windows' anonymous commit; file-backed mmap pages are not part of it.
        private = getattr(info, "private", None)
        if private is not None:
            private_mb = private / 1024**2
            self.peak_private_mb = private_mb if self.peak_private_mb is None else max(self.peak_private_mb, private_mb)
        available = self._psutil.virtual_memory().available / 1024**2
        self.min_available_mb = available if self.min_available_mb is None else min(self.min_available_mb, available)
        self.samples += 1

    def start(self) -> "PeakRSS":
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process()
            self._snapshot()
            self.start_rss_mb = self.peak_rss_mb
        except Exception:  # noqa: BLE001 - measurement must never break training
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="peak-rss", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._snapshot()
            except Exception:  # noqa: BLE001
                return

    def stop(self) -> "PeakRSS":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self

    def __enter__(self) -> "PeakRSS":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    def report(self) -> dict:
        return {
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "start_rss_mb": round(self.start_rss_mb, 1),
            "peak_private_mb": None if self.peak_private_mb is None else round(self.peak_private_mb, 1),
            "min_available_mb": None if self.min_available_mb is None else round(self.min_available_mb, 1),
            "samples": self.samples,
            "interval_s": self.interval,
            "source": (
                "psutil: RSS (includes file-backed mmap pages) and private/anonymous bytes, "
                "sampled in a background thread"
            ),
        }


def measure(func, *args, **kwargs):
    """Run ``func`` under a sampler and return ``(result, memory_report)``."""
    sampler = PeakRSS().start()
    try:
        result = func(*args, **kwargs)
        return result, sampler.stop().report()
    except Exception:
        sampler.stop()
        raise


def elapsed(started: float) -> float:
    return round(time.time() - started, 2)
