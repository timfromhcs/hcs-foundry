"""Distillation trainer under the v8 contract (teacher = the loaded local model or the gateway)."""

from __future__ import annotations

from ..base import Preflight, RunResult, TrainContext, TrainingError
from ..registry import register
from ..spec import Backend, Method, TrainSpec
from ._common import common_preflight, finish, legacy_config


@register
class DistillTrainer:
    method = Method.DISTILL
    backend = Backend.FP32_LORA

    def preflight(self, spec: TrainSpec, ctx: TrainContext) -> Preflight:
        base = common_preflight(spec, ctx)
        teacher = str(spec.method_args.get("teacher") or "local")
        if teacher not in ("local", "gateway"):
            base.blockers.append(f"unknown teacher: {teacher}")
        if teacher == "local" and ctx.settings is not None:
            server = getattr(ctx.settings, "_foundry_server", None)
            if server is None:
                base.warnings.append("teacher 'local' needs a loaded model at run time")
        return base

    def run(self, spec: TrainSpec, ctx: TrainContext, *, resume_from=None) -> RunResult:
        from ...training import run_distill

        teacher_mode = str(spec.method_args.get("teacher") or "local")
        prompts = list(spec.method_args.get("prompts") or [])
        if not prompts:
            raise TrainingError("distillation needs method_args.prompts (or use the API, which samples them)")
        if teacher_mode == "gateway":
            raise TrainingError("the gateway teacher is configured through the API, not the CLI spec")
        raise TrainingError(
            "distillation needs a live teacher: run it through POST /api/v1/training/auto "
            "with algorithm='distill' (the spec contract covers the config, the API owns the teacher)"
        )
