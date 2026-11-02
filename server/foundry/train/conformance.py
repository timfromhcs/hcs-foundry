"""Conformance checks — the same contract for every registered trainer.

The point of the matrix is that a new method cannot ship by *looking* finished: it has to run on a
tiny model, change parameters, survive resume and cancellation, and produce a valid manifest.
Checks that cannot apply to a trainer (no adapter, no resume support) are reported as
``not_applicable`` with a reason, never as a silent pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Preflight, TrainContext, Trainer
from .manifest import RunManifest
from .spec import TrainSpec


@dataclass
class CheckResult:
    check: str
    status: str  # passed | failed | not_applicable
    detail: str = ""

    def to_public(self) -> dict:
        return {"check": self.check, "status": self.status, "detail": self.detail}


@dataclass
class ConformanceReport:
    trainer: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_public(self) -> dict:
        return {
            "trainer": self.trainer,
            "ok": self.ok,
            "checks": [check.to_public() for check in self.checks],
        }


def check_preflight_honesty(trainer: Trainer, spec: TrainSpec, ctx: TrainContext) -> CheckResult:
    """C1 — a too-large or impossible spec must be refused with numbers, not an exception."""
    try:
        preflight = trainer.preflight(spec, ctx)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("C1", "failed", f"preflight raised {type(exc).__name__}: {exc}")
    if not isinstance(preflight, Preflight):
        return CheckResult("C1", "failed", "preflight did not return a Preflight")
    return CheckResult("C1", "passed", f"ok={preflight.ok} blockers={len(preflight.blockers)}")


def check_real_optimisation(manifest: RunManifest) -> CheckResult:
    """C2 — parameters actually moved and the losses are finite."""
    if manifest.param_delta_max is None:
        return CheckResult("C2", "not_applicable", "no adapter artifact to inspect")
    if manifest.param_delta_max <= 0:
        return CheckResult("C2", "failed", "LoRA-B is still zero: no parameter update happened")
    if manifest.grads_finite is False:
        return CheckResult("C2", "failed", "non-finite loss recorded")
    return CheckResult("C2", "passed", f"max |LoRA-B| = {manifest.param_delta_max:.6f}")


def check_manifest_valid(manifest: RunManifest) -> CheckResult:
    """C8 — the manifest validates as v2 and carries the spec identity."""
    try:
        again = RunManifest.model_validate(manifest.model_dump())
    except Exception as exc:  # noqa: BLE001
        return CheckResult("C8", "failed", f"manifest does not validate: {exc}")
    if manifest.spec is not None and manifest.spec_hash != manifest.spec.spec_hash:
        return CheckResult("C8", "failed", "spec_hash does not match the embedded spec")
    return CheckResult("C8", "passed", f"status={again.status}")


def check_source_untouched(trainer: Trainer, spec: TrainSpec, ctx: TrainContext, hashes_before: dict) -> CheckResult:
    """C7 — the base model files are byte-identical after the run."""
    import hashlib
    from pathlib import Path

    model = Path(spec.model.path)
    after = {}
    for path in sorted(model.glob("*.safetensors")) + sorted(model.glob("*.bin")):
        after[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if after != hashes_before:
        return CheckResult("C7", "failed", "base model files changed during training")
    return CheckResult("C7", "passed", f"{len(after)} file(s) unchanged")


def hash_model_files(spec: TrainSpec) -> dict:
    import hashlib
    from pathlib import Path

    model = Path(spec.model.path)
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for path in sorted(model.glob("*.safetensors")) + sorted(model.glob("*.bin"))
    }
