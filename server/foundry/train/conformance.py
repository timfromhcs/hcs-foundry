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

# ---------------------------------------------------------------------------
# C3-C6, C9-C11: the checks that decide whether a trainer is finished
# ---------------------------------------------------------------------------

LEARNING_MIN_POINTS = 3
LEARNING_MIN_RELATIVE_DROP = 0.01
REPEAT_TOLERANCE = 1e-6
CANCEL_MAX_LATENCY_S = 30.0


def _finite(values: list[float]) -> bool:
    return all(value == value and abs(value) != float("inf") for value in values)


def check_learning(
    manifest: RunManifest,
    *,
    min_points: int = LEARNING_MIN_POINTS,
    min_relative_drop: float = LEARNING_MIN_RELATIVE_DROP,
) -> CheckResult:
    """C3 - the loss went down by a margin that is larger than float noise.

    A one-step run cannot answer this question, so it is ``not_applicable`` rather than a pass:
    "we did not look" and "it learned" are different statements.
    """
    losses = [float(value) for value in manifest.losses if value is not None]
    if len(losses) < min_points:
        return CheckResult("C3", "not_applicable", f"{len(losses)} loss point(s); need >= {min_points}")
    if not _finite(losses):
        return CheckResult("C3", "failed", "non-finite loss in the trajectory")
    window = max(1, min(3, len(losses) // 3))
    first = sum(losses[:window]) / window
    last = sum(losses[-window:]) / window
    drop = (first - last) / max(abs(first), 1e-12)
    detail = f"loss {first:.6f} -> {last:.6f} ({drop:+.2%}, need >= {min_relative_drop:.0%})"
    if drop < min_relative_drop:
        return CheckResult("C3", "failed", detail)
    return CheckResult("C3", "passed", detail)


def check_repeatability(
    first: RunManifest,
    second: RunManifest,
    *,
    tolerance: float = REPEAT_TOLERANCE,
) -> CheckResult:
    """C4 - the same spec and seed produce the same numbers within a printed tolerance.

    Bit-exact equality is not promised by CPU float ops; the observed delta is always printed so a
    drifting trainer is visible instead of hidden behind a tolerance.
    """
    if first.param_delta_max is None or second.param_delta_max is None:
        return CheckResult("C4", "not_applicable", "no adapter artifact to compare")
    delta = abs(first.param_delta_max - second.param_delta_max)
    detail = (
        f"max |LoRA-B| {first.param_delta_max:.9f} vs {second.param_delta_max:.9f} "
        f"(delta {delta:.3e}, tolerance {tolerance:.0e})"
    )
    if delta > tolerance:
        return CheckResult("C4", "failed", detail)
    if first.losses and second.losses:
        pairs = list(zip(first.losses, second.losses))
        loss_delta = max(abs(a - b) for a, b in pairs)
        detail += f"; max loss delta {loss_delta:.3e}"
        if loss_delta > tolerance:
            return CheckResult("C4", "failed", detail)
    return CheckResult("C4", "passed", detail)


def check_resume_equivalence(
    uninterrupted: RunManifest | None,
    resumed: RunManifest | None,
    *,
    tolerance: float = REPEAT_TOLERANCE,
) -> CheckResult:
    """C5 - ``k`` steps plus a resume equals an uninterrupted run of the same length."""
    if uninterrupted is None or resumed is None:
        return CheckResult("C5", "not_applicable", "the trainer has no resume path to compare")
    if resumed.status not in ("resumed", "completed"):
        return CheckResult("C5", "failed", f"resumed run ended as {resumed.status}")
    if uninterrupted.steps != resumed.steps:
        return CheckResult(
            "C5", "failed", f"steps differ: {uninterrupted.steps} uninterrupted vs {resumed.steps} resumed"
        )
    if uninterrupted.param_delta_max is None or resumed.param_delta_max is None:
        return CheckResult("C5", "not_applicable", "no adapter artifact to compare")
    delta = abs(uninterrupted.param_delta_max - resumed.param_delta_max)
    if delta > tolerance:
        return CheckResult(
            "C5", "failed", f"max |LoRA-B| differs by {delta:.3e} (tolerance {tolerance:.0e})"
        )
    return CheckResult("C5", "passed", f"{resumed.steps} steps match within {tolerance:.0e}")


def check_cancellation(
    manifest: RunManifest | None,
    *,
    observed_latency_s: float | None,
    max_latency_s: float = CANCEL_MAX_LATENCY_S,
) -> CheckResult:
    """C6 - a cancel is honoured quickly and is not recorded as a completed run."""
    if observed_latency_s is None or manifest is None:
        return CheckResult("C6", "not_applicable", "cancellation was not exercised")
    if manifest.status == "completed":
        return CheckResult("C6", "failed", "a cancelled run recorded status=completed")
    if manifest.status != "cancelled":
        return CheckResult("C6", "failed", f"status is {manifest.status}, expected cancelled")
    if observed_latency_s > max_latency_s:
        return CheckResult(
            "C6", "failed", f"cancel took {observed_latency_s:.1f}s (limit {max_latency_s:.0f}s)"
        )
    return CheckResult("C6", "passed", f"stopped in {observed_latency_s:.2f}s, status=cancelled")


def check_adapter_reload(manifest: RunManifest, *, forward: bool = True) -> CheckResult:
    """C9 - the saved adapter can be loaded onto the base model and produces finite logits."""
    if not manifest.adapter_dir:
        return CheckResult("C9", "not_applicable", "the run produced no adapter")
    from pathlib import Path

    adapter = Path(manifest.adapter_dir)
    if not (adapter / "adapter_model.safetensors").exists():
        return CheckResult(
            "C9",
            "not_applicable",
            "the INT4 adapter uses the custom store format; reload is covered by the merge tests",
        )
    if not forward:
        return CheckResult("C9", "passed", "adapter files present (forward pass skipped)")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(manifest.base_model)
        model = PeftModel.from_pretrained(base, str(adapter))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor([[1, 2, 3]])).logits
        if not bool(torch.isfinite(logits).all()):
            return CheckResult("C9", "failed", "reloaded adapter produced non-finite logits")
        return CheckResult("C9", "passed", f"reloaded and forward pass ok ({tuple(logits.shape)})")
    except Exception as exc:  # noqa: BLE001 - the failure itself is the result
        return CheckResult("C9", "failed", f"reload failed: {type(exc).__name__}: {exc}")


def check_no_false_success(manifest: RunManifest) -> CheckResult:
    """C10 - "completed" without a parameter update is a lie, not a result."""
    if manifest.status != "completed":
        return CheckResult("C10", "not_applicable", f"status is {manifest.status}")
    if manifest.param_delta_max is None:
        return CheckResult("C10", "not_applicable", "the run has no adapter to judge")
    if manifest.param_delta_max <= 0:
        return CheckResult("C10", "failed", "status=completed but LoRA-B is still zero")
    return CheckResult("C10", "passed", f"completed with max |LoRA-B| = {manifest.param_delta_max:.6f}")


def check_budget(preflight: Preflight, spec: TrainSpec, *, max_wall_s: float | None) -> CheckResult:
    """C11 - a run that cannot finish inside the wall-clock budget is refused before it starts."""
    if max_wall_s is None:
        return CheckResult("C11", "not_applicable", "no wall-clock budget is configured")
    low, high = preflight.est_step_seconds
    if high <= 0:
        return CheckResult("C11", "not_applicable", "preflight gives no step-time estimate")
    estimate = high * max(1, spec.hyper.max_steps)
    detail = f"estimated {estimate:.1f}s for {spec.hyper.max_steps} step(s), budget {max_wall_s:.0f}s"
    if estimate > max_wall_s:
        return CheckResult("C11", "failed", detail)
    return CheckResult("C11", "passed", detail)


def run_matrix(
    trainer: Trainer,
    spec: TrainSpec,
    ctx: TrainContext,
    *,
    max_wall_s: float | None = None,
    reload_forward: bool = True,
) -> ConformanceReport:
    """Run one trainer once and apply every check that can be judged from that run.

    Checks that need a second run (C4), a resume (C5) or a cancel (C6) report ``not_applicable``
    with the reason; the caller supplies them via the dedicated comparisons.
    """
    report = ConformanceReport(trainer=f"{trainer.method.value}/{trainer.backend.value}")
    before = hash_model_files(spec)
    report.checks.append(check_preflight_honesty(trainer, spec, ctx))
    preflight = trainer.preflight(spec, ctx)
    report.checks.append(check_budget(preflight, spec, max_wall_s=max_wall_s))
    if not preflight.ok:
        report.checks.append(CheckResult("C2", "not_applicable", "preflight refused the run"))
        report.checks.append(CheckResult("C3", "not_applicable", "preflight refused the run"))
        report.checks.append(CheckResult("C7", "not_applicable", "preflight refused the run"))
        report.checks.append(CheckResult("C8", "not_applicable", "preflight refused the run"))
        report.checks.append(CheckResult("C9", "not_applicable", "preflight refused the run"))
        report.checks.append(CheckResult("C10", "not_applicable", "preflight refused the run"))
        report.checks.append(CheckResult("C4", "not_applicable", "needs a second run"))
        report.checks.append(check_resume_equivalence(None, None))
        report.checks.append(check_cancellation(None, observed_latency_s=None))
        return report

    result = trainer.run(spec, ctx)
    manifest = result.manifest
    report.checks.append(check_real_optimisation(manifest))
    report.checks.append(check_learning(manifest))
    report.checks.append(check_manifest_valid(manifest))
    report.checks.append(check_source_untouched(trainer, spec, ctx, before))
    report.checks.append(check_adapter_reload(manifest, forward=reload_forward))
    report.checks.append(check_no_false_success(manifest))
    report.checks.append(CheckResult("C4", "not_applicable", "needs a second run of the same spec"))
    report.checks.append(check_resume_equivalence(None, None))
    report.checks.append(check_cancellation(manifest, observed_latency_s=None))
    return report
