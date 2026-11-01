"""Vulkan inference auto-tuning.

Sweeps GPU layer offload and batch size, measures real throughput, and stores the
winning profile. This tunes **inference**; PyTorch has no Vulkan backend, so training
is unaffected and stays CPU-only on this machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .db import Database
from .runtime.llama_server import LlamaServer, LoadConfig, RuntimeFailure

PROMPT = "Explain in one short paragraph why memory bandwidth matters for local inference."


@dataclass
class TunePoint:
    gpu_layers: int
    batch: int
    ok: bool
    tokens_per_second: float | None = None
    ttft_seconds: float | None = None
    seconds: float | None = None
    error: str | None = None

    def to_public(self) -> dict:
        return {
            "gpu_layers": self.gpu_layers,
            "batch": self.batch,
            "ok": self.ok,
            "tokens_per_second": self.tokens_per_second,
            "ttft_seconds": self.ttft_seconds,
            "seconds": self.seconds,
            "error": self.error,
        }


def profile_for(db: Database, model_id: str) -> dict | None:
    profiles = db.get_setting("vulkan_profiles", {}) or {}
    return profiles.get(model_id)


def apply_profile(db: Database, model_id: str, *, context: int | None = None) -> dict | None:
    """Return the stored winning profile, optionally constrained by context."""
    profile = profile_for(db, model_id)
    if not profile:
        return None
    if context is not None and context < int(profile.get("context") or 0):
        return {**profile, "context": context, "note": "context reduced below the tuned value"}
    return profile


def sweep(
    db: Database,
    server: LlamaServer,
    *,
    model: dict,
    binary: str,
    context: int = 2048,
    gpu_layers: list[int] | None = None,
    batches: list[int] | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel=None,
) -> dict:
    gpu_layers = gpu_layers or [0, 8, 16, 99]
    batches = batches or [512]
    points: list[TunePoint] = []
    total = len(gpu_layers) * len(batches)
    index = 0

    for ngl in gpu_layers:
        for batch in batches:
            index += 1
            if cancel is not None and cancel.is_set():
                raise InterruptedError("tuning cancelled")
            if progress:
                progress((index - 1) / total * 100, f"ngl={ngl} batch={batch}")
            started = time.time()
            try:
                server.load(
                    model_id=model["id"],
                    model_path=model["path"],
                    binary=binary,
                    config=LoadConfig(context=context, gpu_layers=ngl, batch=batch),
                    alias=model["name"],
                )
                result = server.complete(
                    [{"role": "user", "content": PROMPT}], max_tokens=96, temperature=0.2
                )
                metrics = result.get("metrics") or {}
                points.append(
                    TunePoint(
                        gpu_layers=ngl,
                        batch=batch,
                        ok=True,
                        tokens_per_second=metrics.get("tokens_per_second"),
                        ttft_seconds=metrics.get("ttft_seconds"),
                        seconds=round(time.time() - started, 2),
                    )
                )
            except (RuntimeFailure, Exception) as exc:  # noqa: BLE001 - a failed point is data
                points.append(
                    TunePoint(gpu_layers=ngl, batch=batch, ok=False, seconds=round(time.time() - started, 2), error=str(exc)[:300])
                )
            finally:
                try:
                    server.unload()
                except Exception:  # noqa: BLE001
                    pass

    usable = [p for p in points if p.ok and p.tokens_per_second]
    usable.sort(key=lambda p: p.tokens_per_second or 0, reverse=True)
    best = usable[0] if usable else None
    cpu = next((p for p in points if p.gpu_layers == 0 and p.ok), None)

    result = {
        "model_id": model["id"],
        "model_name": model["name"],
        "context": context,
        "points": [p.to_public() for p in points],
        "best": best.to_public() if best else None,
        "cpu_baseline": cpu.to_public() if cpu else None,
        "speedup_vs_cpu": (
            round((best.tokens_per_second or 0) / cpu.tokens_per_second, 2)
            if best and cpu and cpu.tokens_per_second
            else None
        ),
        "note": "Inference only. Training has no Vulkan backend.",
    }
    if best:
        profiles = db.get_setting("vulkan_profiles", {}) or {}
        profiles[model["id"]] = {
            "gpu_layers": best.gpu_layers,
            "batch": best.batch,
            "context": context,
            "tokens_per_second": best.tokens_per_second,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        db.set_setting("vulkan_profiles", profiles)
    if progress:
        progress(100, "sweep finished")
    return result
