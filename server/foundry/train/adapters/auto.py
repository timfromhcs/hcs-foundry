"""The unified chain (decide → train → merge → export → quantize) as a callable, not a trainer.

It is a *pipeline*, not a method/backend pair, so it is deliberately not in the registry: the
conformance matrix tests trainers, and a pipeline that can skip stages cannot pass C2 (a real
parameter update).
"""

from __future__ import annotations

from ..base import RunResult, TrainContext
from ..manifest import from_legacy
from ..spec import TrainSpec


def run_unified_spec(spec: TrainSpec, ctx: TrainContext) -> RunResult:
    from ...training import run_unified

    report = run_unified(
        ctx.db,
        ctx.settings,
        model_path=spec.model.path,
        dataset_id=spec.data.dataset_id,
        options={
            "max_steps": spec.hyper.max_steps,
            "max_seq_len": spec.hyper.seq_len,
            "vram_gb": spec.method_args.get("vram_gb"),
            "vram_mode": str(spec.method_args.get("vram_mode") or "auto"),
            "quant": str(spec.method_args.get("quant") or "Q4_K_M"),
        },
        progress=ctx.progress,
        log=ctx.log,
        cancel=ctx.cancel,
    )
    stages = report.get("stages") or []
    artifacts = report.get("artifacts") or {}
    adapter = artifacts.get("adapter_dir")
    legacy = {
        "kind": "unified",
        "base_model": spec.model.path,
        "dataset": {"dataset_id": spec.data.dataset_id},
        "steps": 0,
        "adapter_dir": adapter,
        "note": ", ".join(f"{stage['stage']}={stage['status']}" for stage in stages),
    }
    return RunResult(
        run_id=str(report.get("run_id") or "unified"),
        manifest=from_legacy(legacy, run_id=str(report.get("run_id") or "unified"), spec=spec),
        adapter_dir=None,
    )
