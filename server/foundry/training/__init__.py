"""Training capability detection, a real smoke optimisation and the LoRA/SFT backend.

Honesty rules implemented here:

* A training state is only ``gpu`` when an accelerated torch device is actually
  available, and ``cpu-experimental`` when only CPU torch exists.
* A job may only be created after a smoke step proves that a parameter actually
  changed and the loss is finite.
* No path in this module reports "trained" for anything short of a completed
  optimisation run with checkpoints.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psutil

from ..config import Settings
from ..db import Database, new_id, utc_now
from ..data.pipeline import get_dataset

TRAINING_PACKAGES = ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes")


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _importable(pkg: str) -> tuple[bool, str | None]:
    """Metadata presence is not proof that a package can be imported."""
    try:
        importlib.import_module(pkg)
        return True, None
    except Exception as exc:  # noqa: BLE001 - any import failure disqualifies the package
        return False, f"{type(exc).__name__}: {exc}"[:240]


def detect_environment(settings: Settings, profile: dict | None = None) -> dict:
    packages = {pkg: _version(pkg) for pkg in TRAINING_PACKAGES}
    import_errors: dict[str, str] = {}
    importable: dict[str, bool] = {}
    for pkg in ("torch", "transformers", "peft", "datasets", "trl"):
        ok, error = _importable(pkg)
        importable[pkg] = ok
        if error:
            import_errors[pkg] = error

    torch_info: dict = {"installed": packages["torch"] is not None and importable["torch"]}
    device = "cpu"
    cuda = False
    gpu_name = None
    if torch_info["installed"]:
        try:
            import torch

            torch_info["version"] = torch.__version__
            torch_info["cuda_build"] = bool(torch.version.cuda)
            torch_info["hip_build"] = bool(getattr(torch.version, "hip", None))
            cuda = bool(torch.cuda.is_available())
            torch_info["cuda_available"] = cuda
            torch_info["device_count"] = int(torch.cuda.device_count()) if cuda else 0
            if cuda:
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
            torch_info["threads"] = torch.get_num_threads()
        except Exception as exc:  # noqa: BLE001
            torch_info["error"] = str(exc)

    libs_ok = all(importable.get(pkg, False) for pkg in ("transformers", "peft", "datasets"))
    missing = [pkg for pkg, version in packages.items() if version is None]
    broken = [pkg for pkg, ok in importable.items() if not ok and packages.get(pkg)]

    if not torch_info["installed"]:
        state = "unsupported"
        reason = "PyTorch is not installed or cannot be imported, so no optimisation step can run."
    elif cuda:
        state = "gpu"
        reason = f"Accelerated training available on {gpu_name}."
    elif libs_ok:
        state = "cpu-experimental"
        reason = "CPU-only PyTorch with the fine-tuning stack importable; small experiments only."
    else:
        state = "cpu-experimental"
        reason = (
            "CPU-only PyTorch. The fine-tuning stack is not usable: "
            + ", ".join(sorted(set(missing + broken)))
            + ". Only the smoke step can run."
        )

    disk = None
    try:
        usage = psutil.disk_usage(str(settings.training_dir))
        disk = {"total_bytes": usage.total, "free_bytes": usage.free}
    except OSError:
        pass

    tier = (profile or {}).get("tier", {}) if profile else {}
    return {
        "state": state,
        "reason": reason,
        "device": device,
        "gpu_name": gpu_name,
        "packages": packages,
        "importable": importable,
        "import_errors": import_errors,
        "missing_packages": sorted(set(missing + broken)),
        "torch": torch_info,
        "disk": disk,
        "machine_tier": tier,
        "can_smoke": state != "unsupported",
        "can_finetune": state == "gpu" or (state == "cpu-experimental" and libs_ok),
        "can_finetune_verified": False,  # set true only after a real fine-tune on this machine
        "install_hint": (
            None
            if libs_ok
            else "pip install \"transformers>=4.40\" \"peft>=0.10\" trl datasets accelerate  (only needed for real fine-tuning)"
        ),
    }


# ---------------------------------------------------------------------------
# Real smoke optimisation
# ---------------------------------------------------------------------------

@dataclass
class SmokeResult:
    ok: bool
    metrics: dict
    reason: str


def run_smoke(progress: Callable[[float, str], None] | None = None) -> SmokeResult:
    """A tiny but genuine forward/backward/update cycle."""
    try:
        import torch
    except ImportError:
        return SmokeResult(ok=False, metrics={}, reason="PyTorch is not installed")

    if progress:
        progress(10, "Building a tiny model")
    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 1),
    )
    inputs = torch.randn(128, 8)
    targets = (inputs.sum(dim=1, keepdim=True) * 0.5) + 0.1

    before = [param.detach().clone() for param in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    losses: list[float] = []
    steps = 40
    for step in range(steps):
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if progress and step % 5 == 0:
            progress(10 + step / steps * 80, f"Optimisation step {step + 1}/{steps}")
        if not math.isfinite(losses[-1]):
            return SmokeResult(ok=False, metrics={"losses": losses}, reason="loss became non-finite")

    max_delta = 0.0
    for param, previous in zip(model.parameters(), before):
        delta = float((param.detach() - previous).abs().max())
        max_delta = max(max_delta, delta)

    final_loss = losses[-1]
    improved = final_loss < losses[0]
    ok = math.isfinite(final_loss) and max_delta > 0 and improved
    reason = (
        "parameter update verified"
        if ok
        else "no parameter change or no loss improvement — refusing to call this training"
    )
    if progress:
        progress(100, reason)
    return SmokeResult(
        ok=ok,
        metrics={
            "steps": steps,
            "loss_first": round(losses[0], 6),
            "loss_last": round(final_loss, 6),
            "max_parameter_delta": max_delta,
            "device": "cpu" if not torch.cuda.is_available() else "cuda",
            "duration_seconds": None,
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Preflight for a real fine-tune
# ---------------------------------------------------------------------------

def preflight(
    db: Database,
    settings: Settings,
    *,
    dataset_id: str,
    base_model: str | None,
    method: str = "lora_sft",
    profile: dict | None = None,
) -> dict:
    env = detect_environment(settings, profile)
    blockers: list[str] = []
    warnings: list[str] = []

    dataset = get_dataset(db, dataset_id)
    if not dataset:
        blockers.append("dataset not found")
    else:
        if dataset["rights_status"] != "confirmed":
            blockers.append("dataset rights are not confirmed")
        stats = dataset.get("stats") or {}
        splits = stats.get("by_split") or {}
        if not splits.get("train"):
            blockers.append("dataset has no training split")
        if stats.get("record_count", 0) < 8:
            warnings.append("fewer than 8 training records: expect overfitting, not learning")
        if stats.get("synthetic_ratio", 0) > 0.5:
            warnings.append("more than half of the dataset is synthetic")

    if env["state"] == "unsupported":
        blockers.append(env["reason"])
    if method in {"lora_sft", "dpo"} and not env["can_finetune"]:
        blockers.append("fine-tuning libraries are missing (transformers, peft, datasets)")
    if method == "dpo":
        blockers.append("DPO requires labelled preference pairs; none are produced by this pipeline yet")
    if not base_model:
        blockers.append("no training-native base model selected (GGUF is inference-only)")

    required_disk = 8 * 1024**3
    if env.get("disk") and env["disk"]["free_bytes"] < required_disk:
        warnings.append("less than 8 GiB free disk space for checkpoints")

    plan = {
        "method": method,
        "dataset_id": dataset_id,
        "base_model": base_model,
        "preset": _preset(env, dataset),
        "blockers": blockers,
        "warnings": warnings,
        "can_start": not blockers,
        "environment": env,
        "checked_at": utc_now(),
    }
    return plan


def _preset(env: dict, dataset: dict | None) -> dict:
    records = ((dataset or {}).get("stats") or {}).get("record_count", 0)
    if env["state"] == "gpu":
        return {
            "micro_batch_size": 2,
            "gradient_accumulation": 8,
            "max_sequence_length": 1024,
            "lora_rank": 16,
            "learning_rate": 2e-4,
            "epochs": 1,
            "checkpoint_every_steps": 50,
        }
    return {
        "micro_batch_size": 1,
        "gradient_accumulation": 16,
        "max_sequence_length": 256,
        "lora_rank": 4,
        "learning_rate": 1e-4,
        "epochs": 1,
        "checkpoint_every_steps": 10,
        "note": f"CPU experimental preset for {records} records; expect very slow steps.",
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def create_run(db: Database, *, job_id: str | None, dataset_id: str, base_model: str | None, backend: str, config: dict) -> str:
    run_id = new_id("trn")
    db.insert(
        "training_runs",
        {
            "id": run_id,
            "job_id": job_id,
            "dataset_id": dataset_id,
            "base_model": base_model,
            "backend": backend,
            "status": "configured",
            "config": json.dumps(config),
            "metrics": "{}",
            "created_at": utc_now(),
        },
    )
    return run_id


def list_runs(db: Database) -> list[dict]:
    rows = db.query("SELECT * FROM training_runs ORDER BY created_at DESC")
    return [_run_row(row) for row in rows]


def get_run(db: Database, run_id: str) -> dict | None:
    row = db.query_one("SELECT * FROM training_runs WHERE id=?", (run_id,))
    return _run_row(row) if row else None


def _run_row(row) -> dict:
    data = dict(row)
    data["config"] = json.loads(data.get("config") or "{}")
    data["metrics"] = json.loads(data.get("metrics") or "{}")
    return data


def start_smoke_run(db: Database, settings: Settings, job_id: str | None, progress=None, cancel=None) -> dict:
    """Recorded smoke run. This is *not* a fine-tune and is labelled as such."""
    started = time.time()
    result = run_smoke(progress)
    run_id = create_run(
        db,
        job_id=job_id,
        dataset_id="",
        base_model="synthetic-tiny-mlp",
        backend="torch-smoke",
        config={"kind": "smoke", "purpose": "verify that optimisation actually updates parameters"},
    )
    db.execute(
        "UPDATE training_runs SET status=?, metrics=?, finished_at=? WHERE id=?",
        (
            "smoke_passed" if result.ok else "smoke_failed",
            json.dumps({**result.metrics, "reason": result.reason, "duration_seconds": round(time.time() - started, 2)}),
            utc_now(),
            run_id,
        ),
    )
    db.activity("training.smoke", result.reason, {"run_id": run_id, "ok": result.ok})
    return {"run_id": run_id, "ok": result.ok, "metrics": result.metrics, "reason": result.reason}

from .recipes import project, recipe_catalog, recipe_config
from .sft import SFTConfig, TrainingUnavailable, create_sft_run, estimate_resources, train_sft

__all__ = [
    "detect_environment",
    "preflight",
    "run_smoke",
    "create_run",
    "list_runs",
    "get_run",
    "start_smoke_run",
    "SFTConfig",
    "TrainingUnavailable",
    "create_sft_run",
    "estimate_resources",
    "train_sft",
    "recipe_catalog",
    "recipe_config",
    "project",
]
