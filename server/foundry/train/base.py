"""Trainer protocol, context and preflight — the seam every method implements.

A trainer is boring on purpose: it validates, it trains, it returns a v2 manifest. Progress,
logging, cancellation and the output directory arrive through ``TrainContext`` so no trainer
invents its own callback shape again.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .manifest import RunManifest
from .spec import Backend, Method, TrainSpec


@dataclass
class Preflight:
    """What a trainer can say *before* it starts. Never a precise fake ETA."""

    ok: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    est_ram_bytes: int = 0
    est_disk_bytes: int = 0
    est_step_seconds: tuple[float, float] = (0.0, 0.0)  # (low, high); (0, 0) means unknown

    def to_public(self) -> dict:
        return {
            "ok": self.ok,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "est_ram_bytes": self.est_ram_bytes,
            "est_disk_bytes": self.est_disk_bytes,
            "est_step_seconds": list(self.est_step_seconds),
        }


@dataclass
class TrainContext:
    out_dir: Path
    progress: Callable[[float, str], None] = lambda pct, message: None
    log: Callable[[str], None] = lambda message: None
    cancel: threading.Event = field(default_factory=threading.Event)
    db: object | None = None
    settings: object | None = None
    clock: Callable[[], float] = time.time


@dataclass
class RunResult:
    run_id: str
    manifest: RunManifest
    adapter_dir: Path | None = None
    checkpoints: list[Path] = field(default_factory=list)


class Trainer(Protocol):
    method: Method
    backend: Backend

    def preflight(self, spec: TrainSpec, ctx: TrainContext) -> Preflight: ...

    def run(self, spec: TrainSpec, ctx: TrainContext, *, resume_from: Path | None = None) -> RunResult: ...


class TrainingError(RuntimeError):
    """A trainer refused to run; ``str(exc)`` is user-facing."""


class NoUpdate(TrainingError):
    """The run finished without changing a single parameter — never report that as success."""
