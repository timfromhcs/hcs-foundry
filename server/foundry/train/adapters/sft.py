"""SFT (fp32 LoRA) trainer under the v8 contract."""

from __future__ import annotations

import time

from ..base import Preflight, RunResult, TrainContext
from ..registry import register
from ..spec import Backend, Method, TrainSpec
from ._common import common_preflight, finish, legacy_config, _dataset_id


@register
class SftTrainer:
    method = Method.SFT
    backend = Backend.FP32_LORA

    def preflight(self, spec: TrainSpec, ctx: TrainContext) -> Preflight:
        return common_preflight(spec, ctx)

    def run(self, spec: TrainSpec, ctx: TrainContext, *, resume_from=None) -> RunResult:
        from ...training import create_sft_run

        started = ctx.clock()
        result = create_sft_run(
            ctx.db,
            legacy_config(spec),
            job_id=None,
            output_dir=ctx.out_dir,
            progress=ctx.progress,
            log=ctx.log,
            cancel=ctx.cancel,
        )
        return finish(spec, ctx, result=result, started=started, clock=ctx.clock)


@register
class AutoSftTrainer:
    """SFT with the auto decision (LR halving, early stop, best checkpoint, playbook)."""

    method = Method.SFT
    backend = Backend.INT4_STREAM  # the auto path can pick a backend; this key keeps it distinct

    def preflight(self, spec: TrainSpec, ctx: TrainContext) -> Preflight:
        return common_preflight(spec, ctx)

    def run(self, spec: TrainSpec, ctx: TrainContext, *, resume_from=None) -> RunResult:
        from ...training import run_auto

        started = ctx.clock()
        result = run_auto(
            ctx.db,
            legacy_config(spec),
            job_id=None,
            output_dir=ctx.out_dir,
            progress=ctx.progress,
            log=ctx.log,
            cancel=ctx.cancel,
        )
        return finish(spec, ctx, result=result, started=started, clock=ctx.clock)
