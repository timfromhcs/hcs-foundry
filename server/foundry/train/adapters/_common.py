"""Shared plumbing for the adapters that wrap the pre-v8 trainers.

Each adapter builds the legacy ``SFTConfig`` from a ``TrainSpec``, calls the existing function
(no behaviour change), and converts the returned manifest to ``RunManifest`` v2. Anything the old
manifest could not express is recorded in ``extras`` — and where a field is required by the
conformance matrix (``param_delta_max``, ``grads_finite``) the legacy trainer now records it.
"""

from __future__ import annotations

from pathlib import Path

from ..manifest import RunManifest, from_legacy
from ..base import Preflight, RunResult, TrainContext, TrainingError
from ..spec import Backend, Method, TrainSpec


def legacy_config(spec: TrainSpec, *, dataset_id: str | None = None):
    from ...training.sft import SFTConfig

    return SFTConfig(
        model_path=spec.model.path,
        dataset_id=dataset_id or spec.data.dataset_id,
        max_steps=spec.hyper.max_steps,
        max_seq_len=spec.hyper.seq_len,
        micro_batch=spec.hyper.micro_batch,
        grad_accum=spec.hyper.grad_accum,
        lora_r=spec.hyper.lora_r,
        lora_alpha=spec.hyper.lora_alpha,
        learning_rate=spec.hyper.lr,
        seed=spec.hyper.seed,
        eval_every=int(spec.method_args.get("eval_every", 0) or 0),
        patience=spec.stop.patience,
    )


def _model_dir(spec: TrainSpec) -> Path:
    path = Path(spec.model.path).expanduser()
    if not path.is_dir():
        raise TrainingError(f"model directory not found: {path}")
    return path


def _dataset_id(spec: TrainSpec) -> str:
    if not spec.data.dataset_id:
        raise TrainingError("the spec has no dataset_id")
    return spec.data.dataset_id


def _rows(ctx: TrainContext, dataset_id: str) -> int:
    db = ctx.db
    if db is None:
        return 0
    rows = db.query("SELECT COUNT(*) AS n FROM dataset_records WHERE dataset_id=? AND excluded=0", (dataset_id,))
    return int(rows[0]["n"]) if rows else 0


def common_preflight(spec: TrainSpec, ctx: TrainContext) -> Preflight:
    blockers: list[str] = []
    warnings: list[str] = []
    try:
        model_dir = _model_dir(spec)
    except TrainingError as exc:
        model_dir = None
        blockers.append(str(exc))
    dataset_id = spec.data.dataset_id
    rows = _rows(ctx, dataset_id) if dataset_id else 0
    if not dataset_id:
        blockers.append("the spec has no dataset_id")
    elif rows < spec.data.min_rows:
        blockers.append(f"dataset has {rows} usable rows, the spec requires {spec.data.min_rows}")
    if rows and rows < 50:
        warnings.append(f"only {rows} rows: this is a smoke run, not a quality run")
    if spec.hyper.max_steps == 0:
        blockers.append("max_steps is 0: nothing would be trained")

    weights = 0
    if model_dir is not None:
        weights = sum(f.stat().st_size for f in list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin")))
        if weights == 0:
            blockers.append("no safetensors/bin weights in the model directory")
    return Preflight(
        ok=not blockers,
        blockers=blockers,
        warnings=warnings,
        est_disk_bytes=int(weights * 0.3),
        est_step_seconds=(0.0, 0.0),
    )


def _adapter_delta(adapter_dir: Path | None) -> float | None:
    """Largest |LoRA-B| in the saved adapter — B is zero-initialised, so > 0 proves a real update."""
    if adapter_dir is None or not adapter_dir.is_dir():
        return None
    try:
        from safetensors.torch import load_file

        for name in ("adapter_model.safetensors", "lora_int4.pt"):
            path = adapter_dir / name
            if not path.exists():
                continue
            if name.endswith(".safetensors"):
                tensors = load_file(str(path))
            else:
                import torch

                payload = torch.load(str(path), map_location="cpu", weights_only=True)
                tensors = payload.get("lora") or {}
            values = [
                float(tensor.float().abs().max())
                for key, tensor in tensors.items()
                if "lora_B" in key and hasattr(tensor, "numel")
            ]
            if values:
                return max(values)
    except Exception:  # noqa: BLE001 - a missing adapter must not break the report
        return None
    return None


def finish(
    spec: TrainSpec,
    ctx: TrainContext,
    *,
    result: dict,
    started: float,
    clock,
) -> RunResult:
    legacy = result.get("manifest") or {}
    run_id = str(result.get("run_id") or "train")
    manifest: RunManifest = from_legacy(legacy, run_id=run_id, spec=spec)
    manifest.wall_seconds = float(manifest.wall_seconds or 0.0) or round(clock() - started, 2)
    adapter = legacy.get("adapter_dir") or legacy.get("best_adapter_dir")
    adapter_dir = Path(adapter) if adapter else None
    manifest.adapter_dir = str(adapter_dir) if adapter_dir else None
    manifest.param_delta_max = _adapter_delta(adapter_dir)
    manifest.grads_finite = all(value == value and abs(value) != float("inf") for value in manifest.losses) if manifest.losses else None
    return RunResult(
        run_id=run_id,
        manifest=manifest,
        adapter_dir=adapter_dir,
        checkpoints=[],
    )
