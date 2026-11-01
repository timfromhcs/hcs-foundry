"""Unified auto trainer: one entry point that decides and runs the whole local chain.

Decision inputs: model size, free RAM, dataset size, and the measured playbook.
Chain: decide → train (SFT or streaming INT4) → merge → export GGUF → quantize → report.

Every stage records its numbers. A stage that cannot run is skipped with a reason.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..db import Database, new_id, utc_now
from . import playbook
from .int4 import estimate_int4_resources
from .algorithms import run_auto
from .sft import SFTConfig, estimate_resources

INT4_MIN_BYTES = 900 * 1024**2  # below this, plain SFT is simpler and faster


def decide(db: Database, settings: Settings, *, model_path: str, dataset_id: str, dataset_stats: dict) -> dict:
    model = Path(model_path).expanduser()
    if not model.is_dir():
        return {"ok": False, "reason": f"model directory not found: {model}"}
    weights = sum(f.stat().st_size for f in list(model.glob("*.safetensors")) + list(model.glob("*.bin")))
    if weights == 0:
        return {"ok": False, "reason": "no safetensors/bin weights found"}
    records = int((dataset_stats or {}).get("record_count") or 0)
    if records < 2:
        return {"ok": False, "reason": "dataset needs at least two records"}

    base = SFTConfig(model_path=str(model), dataset_id=dataset_id, max_steps=20, max_seq_len=256)
    fp32 = estimate_resources(model, base)
    int4 = estimate_int4_resources(model, base)

    signature = playbook.signature(str(model), (dataset_stats or {}).get("dataset_hash"), dataset_id)
    suggestion = playbook.suggest(db, signature)

    if fp32["fits"]:
        algorithm = "sft"
        reason = "fp32 load fits the safe envelope"
        estimate = fp32
    elif int4["fits"] and weights >= INT4_MIN_BYTES:
        algorithm = "int4"
        reason = "fp32 does not fit; streaming INT4 does"
        estimate = int4
    else:
        return {
            "ok": False,
            "reason": "neither fp32 nor streaming INT4 fits the safe memory envelope",
            "fp32": fp32,
            "int4": int4,
        }

    config = {
        "max_steps": 20,
        "max_seq_len": 256 if algorithm == "sft" else 128,
        "lora_r": 8 if algorithm == "sft" else 4,
        "learning_rate": 2e-4,
        "eval_every": 5,
        "patience": 3,
    }
    if suggestion:
        config.update({k: v for k, v in (suggestion.get("config") or {}).items() if v is not None})

    return {
        "ok": True,
        "algorithm": algorithm,
        "reason": reason,
        "config": config,
        "signature": signature,
        "playbook_seeded": bool(suggestion),
        "weights_bytes": weights,
        "estimate": estimate,
    }


def run_unified(
    db: Database,
    settings: Settings,
    *,
    model_path: str,
    dataset_id: str,
    options: dict | None = None,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancel=None,
) -> dict:
    from ..data.pipeline import get_dataset
    from ..gguf_tools import discover, export_gguf, quantize_gguf
    from ..pipeline import merge_adapter

    options = options or {}
    run_id = new_id("uni")
    started = utc_now()
    stages: list[dict] = []
    artifacts: dict = {}

    def record(stage: str, status: str, detail: dict) -> None:
        stages.append({"stage": stage, "status": status, "at": utc_now(), **detail})
        if log:
            log(f"[{stage}] {status}: {json.dumps(detail)[:200]}")

    dataset = get_dataset(db, dataset_id)
    if not dataset:
        raise ValueError("dataset not found")
    decision = decide(db, settings, model_path=model_path, dataset_id=dataset_id, dataset_stats=dataset.get("stats") or {})
    record("decide", "ok" if decision["ok"] else "failed", decision)
    if not decision["ok"]:
        return {"run_id": run_id, "started_at": started, "decision": decision, "stages": stages, "artifacts": artifacts}

    algorithm = decision["algorithm"]
    cfg = decision["config"]
    config = SFTConfig(
        model_path=model_path,
        dataset_id=dataset_id,
        max_steps=int(options.get("max_steps", cfg["max_steps"])),
        max_seq_len=int(options.get("max_seq_len", cfg["max_seq_len"])),
        lora_r=int(cfg.get("lora_r", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        auto=True,
        eval_every=int(cfg.get("eval_every", 5)),
        patience=int(cfg.get("patience", 3)),
    )
    output = settings.training_dir / f"unified_{int(time.time())}"

    if progress:
        progress(5, f"training ({algorithm})")
    try:
        if algorithm == "int4":
            from .int4 import create_int4_run

            train_result = create_int4_run(
                db, config, job_id=None, output_dir=output,
                progress=lambda pct, msg: progress(5 + pct * 0.45, msg) if progress else None,
                log=log, cancel=cancel, streaming=True,
            )
        else:
            train_result = run_auto(
                db, config, job_id=None, output_dir=output,
                progress=lambda pct, msg: progress(5 + pct * 0.45, msg) if progress else None,
                log=log, cancel=cancel,
            )
        manifest = train_result["manifest"]
        artifacts["train_run_id"] = train_result["run_id"]
        artifacts["adapter_dir"] = manifest.get("adapter_dir")
        record("train", "ok", {
            "run_id": train_result["run_id"],
            "algorithm": algorithm,
            "steps": manifest.get("steps"),
            "loss_first": manifest.get("loss_first"),
            "loss_last": manifest.get("loss_last"),
            "best_eval_loss": manifest.get("best_eval_loss"),
        })
    except Exception as exc:  # noqa: BLE001
        record("train", "failed", {"error": str(exc)[:400]})
        return _finish(db, settings, run_id, started, decision, stages, artifacts)

    if algorithm == "int4":
        record("merge", "skipped", {"reason": "INT4 adapters are a custom artifact and cannot be merged into fp32 weights"})
        record("export", "skipped", {"reason": "no merged fp32 directory"})
        record("quantize", "skipped", {"reason": "no GGUF produced"})
        return _finish(db, settings, run_id, started, decision, stages, artifacts)

    if progress:
        progress(55, "merging adapter")
    try:
        merged = merge_adapter(model_path, manifest["adapter_dir"], output / "merged", log=log)
        artifacts["merged_dir"] = merged["output"]
        record("merge", "ok", merged)
    except Exception as exc:  # noqa: BLE001
        record("merge", "failed", {"error": str(exc)[:400]})
        return _finish(db, settings, run_id, started, decision, stages, artifacts)

    if progress:
        progress(68, "exporting GGUF")
    toolchain = discover(settings)
    try:
        exported = export_gguf(artifacts["merged_dir"], output / "model-f16.gguf", toolchain=toolchain, log=log, cancel=cancel)
        artifacts["gguf_f16"] = exported["output"]
        record("export", "ok", {"size_bytes": exported["size_bytes"], "seconds": exported["duration_seconds"]})
    except Exception as exc:  # noqa: BLE001
        record("export", "failed", {"error": str(exc)[:400]})
        return _finish(db, settings, run_id, started, decision, stages, artifacts)

    if progress:
        progress(85, "quantizing")
    method = str(options.get("quant", "Q4_K_M"))
    try:
        quantized = quantize_gguf(artifacts["gguf_f16"], output / f"model-{method}.gguf", method, toolchain=toolchain, log=log, cancel=cancel)
        artifacts["gguf_quant"] = quantized["output"]
        record("quantize", "ok", {"method": method, "size_bytes": quantized["size_bytes"], "ratio": quantized["ratio"]})
    except Exception as exc:  # noqa: BLE001
        record("quantize", "failed", {"error": str(exc)[:400]})

    return _finish(db, settings, run_id, started, decision, stages, artifacts)


def _finish(db: Database, settings: Settings, run_id: str, started: str, decision: dict, stages: list[dict], artifacts: dict) -> dict:
    finished = utc_now()
    report = {
        "run_id": run_id,
        "started_at": started,
        "finished_at": finished,
        "decision": decision,
        "stages": stages,
        "artifacts": artifacts,
    }
    target = settings.data_dir / "reports" / f"{run_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), "utf-8")
    report["report"] = str(target)
    db.activity("training.unified", f"Unified run {run_id}: " + ", ".join(f"{s['stage']}={s['status']}" for s in stages))
    return report
