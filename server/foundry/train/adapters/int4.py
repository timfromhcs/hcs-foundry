"""INT4 (QLoRA-style) trainer under the v8 contract, for both backends."""

from __future__ import annotations

from ..base import Preflight, RunResult, TrainContext
from ..registry import register
from ..spec import Backend, Method, TrainSpec
from ._common import common_preflight, finish, legacy_config


class _Int4Base:
    method = Method.INT4
    streaming = True
    offload = True

    def preflight(self, spec: TrainSpec, ctx: TrainContext) -> Preflight:
        from ...training.int4 import estimate_int4_resources

        base = common_preflight(spec, ctx)
        try:
            estimate = estimate_int4_resources(spec.model.path, legacy_config(spec))
        except Exception as exc:  # noqa: BLE001 - a broken estimate is not a blocker
            base.warnings.append(f"memory estimate unavailable: {exc}")
            return base
        base.est_ram_bytes = int(estimate.get("estimated_high_bytes") or 0)
        if not estimate.get("fits", True) and self.offload:
            base.warnings.append(
                "the fp32 estimate does not fit; the disk-backed store will be used and step time "
                "depends on disk throughput"
            )
        return base

    def run(self, spec: TrainSpec, ctx: TrainContext, *, resume_from=None) -> RunResult:
        from ...training import create_int4_run

        started = ctx.clock()
        result = create_int4_run(
            ctx.db,
            legacy_config(spec),
            job_id=None,
            output_dir=ctx.out_dir,
            progress=ctx.progress,
            log=ctx.log,
            cancel=ctx.cancel,
            streaming=self.streaming,
            offload=self.offload,
            vram_gb=spec.method_args.get("vram_gb"),
            vram_mode=str(spec.method_args.get("vram_mode") or "auto"),
        )
        return finish(spec, ctx, result=result, started=started, clock=ctx.clock)


@register
class Int4StreamTrainer(_Int4Base):
    backend = Backend.INT4_STREAM
    streaming = True
    offload = False


@register
class Int4StoreTrainer(_Int4Base):
    backend = Backend.INT4_STORE
    streaming = True
    offload = True
