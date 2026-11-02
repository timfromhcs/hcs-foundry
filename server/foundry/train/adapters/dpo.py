"""DPO trainer under the v8 contract."""

from __future__ import annotations

from ..base import Preflight, RunResult, TrainContext
from ..registry import register
from ..spec import Backend, Method, TrainSpec
from ._common import common_preflight, finish, legacy_config


@register
class DpoTrainer:
    method = Method.DPO
    backend = Backend.FP32_LORA

    def preflight(self, spec: TrainSpec, ctx: TrainContext) -> Preflight:
        return common_preflight(spec, ctx)

    def run(self, spec: TrainSpec, ctx: TrainContext, *, resume_from=None) -> RunResult:
        from ...training import run_dpo

        started = ctx.clock()
        result = run_dpo(
            ctx.db,
            legacy_config(spec),
            beta=float(spec.method_args.get("beta", 0.1)),
            job_id=None,
            output_dir=ctx.out_dir,
            progress=ctx.progress,
            log=ctx.log,
            cancel=ctx.cancel,
        )
        return finish(spec, ctx, result=result, started=started, clock=ctx.clock)
