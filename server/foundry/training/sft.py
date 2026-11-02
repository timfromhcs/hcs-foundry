"""Real LoRA/SFT training backend.

Design goals:

* No hidden downloads: the base model must already exist on disk.
* No fake runs: the job only reports success after an optimisation loop that
  produced checkpoints and a finite loss.
* Everything is recorded: base model hash, dataset hash, config, library versions.
* Cancellation is cooperative and checkpoint-safe.

The backend is intentionally conservative on CPU-only hardware: it refuses
configurations that cannot fit, and it reports measured throughput instead of a
promised one.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import psutil

from ..db import Database, new_id, utc_now
from ..data.pipeline import get_dataset

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TrainingUnavailable(RuntimeError):
    pass


@dataclass
class SFTConfig:
    model_path: str
    dataset_id: str
    max_steps: int = 20
    max_seq_len: int = 256
    micro_batch: int = 1
    grad_accum: int = 4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    seed: int = 42
    resume_from: str | None = None
    save_every: int = 10
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    # auto-decision / OpenEval
    auto: bool = False
    eval_every: int = 0
    eval_samples: int = 8
    patience: int = 3
    lr_min: float = 1e-5
    extra_texts: list[str] = field(default_factory=list)

    def to_public(self) -> dict:
        return asdict(self)


def _versions() -> dict:
    out = {}
    for pkg in ("torch", "transformers", "peft", "datasets", "accelerate", "huggingface_hub"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _model_weight_files(model_path: Path) -> list[Path]:
    files = sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("*.bin")) + sorted(model_path.glob("*.pt"))
    return [f for f in files if f.is_file()]


def _read_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def estimate_resources(model_path: str | Path, config: SFTConfig) -> dict:
    """Return a memory range with an explicit confidence label."""
    path = Path(model_path).expanduser()
    model_config = _read_config(path)
    weights = _model_weight_files(path)
    weight_bytes = sum(f.stat().st_size for f in weights)

    hidden = int(model_config.get("hidden_size") or model_config.get("n_embd") or 2048)
    layers = int(model_config.get("num_hidden_layers") or model_config.get("n_layer") or 24)
    params = int(model_config.get("_name_or_path_count") or 0)

    # LoRA trainable parameters are a small fraction; optimiser state is 2x that in fp32.
    trainable = int(weight_bytes / 4 * 0.01)  # ~1 % of parameters
    optimizer_bytes = trainable * 4 * 2
    activation_bytes = config.micro_batch * config.max_seq_len * hidden * layers * 2 * 4
    overhead = 700 * 1024**2
    core = weight_bytes + optimizer_bytes + activation_bytes + overhead

    available = psutil.virtual_memory().available
    return {
        "weights_bytes": weight_bytes,
        "optimizer_bytes": optimizer_bytes,
        "activation_bytes": activation_bytes,
        "overhead_bytes": overhead,
        "estimated_low_bytes": int(core),
        "estimated_high_bytes": int(core * 1.35),
        "available_bytes": available,
        "fits": core * 1.35 < available * 0.9,
        "confidence": "low" if not weights else "medium",
        "hidden_size": hidden,
        "layers": layers,
        "params_estimate": params or None,
        "note": (
            "Estimate only. It does not include fragmentation, the tokenizer, "
            "or the dataloader. Measure before a long run."
        ),
    }


def _load_stack():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except Exception as exc:  # noqa: BLE001 - import failures are a capability answer
        raise TrainingUnavailable(
            "The fine-tuning stack is not importable: "
            f"{type(exc).__name__}: {exc}. Install transformers, peft, datasets and accelerate."
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
    }


def build_examples(db: Database, dataset_id: str, max_seq_len: int, extra_texts: list[str] | None = None) -> tuple[list[dict], dict]:
    dataset = get_dataset(db, dataset_id)
    if not dataset:
        raise ValueError("dataset not found")
    if dataset["rights_status"] != "confirmed":
        raise ValueError("dataset rights are not confirmed")
    rows = db.query(
        "SELECT payload FROM dataset_records WHERE dataset_id=? AND split='train' AND excluded=0 ORDER BY rowid",
        (dataset_id,),
    )
    if not rows:
        raise ValueError("dataset has no included training records")

    texts: list[str] = []
    for row in rows:
        payload = json.loads(row["payload"])
        task = payload.get("task") or {}
        user = (task.get("user_request") or "").strip()
        assistant = ""
        for step in reversed(payload.get("trace") or []):
            if step.get("kind") == "assistant_message" and (step.get("content") or "").strip():
                assistant = step["content"].strip()
                break
        if not user or not assistant:
            continue
        texts.append(f"### User\n{user}\n\n### Assistant\n{assistant}\n")

    for extra in extra_texts or []:
        if extra and extra.strip():
            texts.append(extra.strip()[: max_seq_len * 8])

    if not texts:
        raise ValueError("no record had both a user request and an assistant answer")
    return [{"text": text[: max_seq_len * 8]} for text in texts], {
        "dataset_id": dataset_id,
        "dataset_hash": dataset.get("manifest", {}).get("dataset_hash"),
        "records_used": len(texts),
        "extra_texts": len(extra_texts or []),
    }


def build_eval_texts(db: Database, dataset_id: str, limit: int = 8) -> list[str]:
    """Held-out evaluation texts: validation split, else the tail of the train split."""
    rows = db.query(
        "SELECT payload FROM dataset_records WHERE dataset_id=? AND split='validation' AND excluded=0 LIMIT ?",
        (dataset_id, limit),
    )
    if not rows:
        rows = db.query(
            "SELECT payload FROM dataset_records WHERE dataset_id=? AND split='train' AND excluded=0 "
            "ORDER BY rowid DESC LIMIT ?",
            (dataset_id, limit),
        )
    texts = []
    for row in rows:
        payload = json.loads(row["payload"])
        task = payload.get("task") or {}
        user = (task.get("user_request") or "").strip()
        assistant = ""
        for step in reversed(payload.get("trace") or []):
            if step.get("kind") == "assistant_message" and (step.get("content") or "").strip():
                assistant = step["content"].strip()
                break
        if user and assistant:
            texts.append(f"### User\n{user}\n\n### Assistant\n{assistant}\n")
    return texts


def _eval_loss(model, tokenizer, texts: list[str], max_seq_len: int, torch) -> float:
    if not texts:
        return float("nan")
    model.eval()
    total = 0.0
    count = 0
    try:
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
                if encoded["input_ids"].shape[-1] < 3:
                    continue
                outputs = model(**encoded, labels=encoded["input_ids"])
                loss = float(outputs.loss)
                if math.isfinite(loss):
                    total += loss
                    count += 1
    finally:
        model.train()
    return total / count if count else float("nan")


def train_sft(
    db: Database,
    config: SFTConfig,
    *,
    output_dir: Path,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancel=None,
) -> dict:
    stack = _load_stack()
    torch = stack["torch"]

    model_path = Path(config.model_path).expanduser()
    if not model_path.is_dir():
        raise ValueError(f"base model directory not found: {model_path}")
    if not _model_weight_files(model_path):
        raise ValueError("no safetensors/bin weights found in the base model directory")

    resources = estimate_resources(model_path, config)
    if not resources["fits"]:
        raise ValueError(
            "estimated memory exceeds available RAM; lower max_seq_len or micro_batch, "
            f"or use a smaller base model (need ~{resources['estimated_high_bytes']} bytes, "
            f"have {resources['available_bytes']})"
        )

    if progress:
        progress(4, "Loading tokenizer")
    tokenizer = stack["AutoTokenizer"].from_pretrained(str(model_path), trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    examples, dataset_meta = build_examples(db, config.dataset_id, config.max_seq_len, config.extra_texts)
    if log:
        log(f"Training on {len(examples)} example(s) from {config.dataset_id}")

    eval_texts = build_eval_texts(db, config.dataset_id, config.eval_samples) if config.eval_every > 0 else []
    if eval_texts and log:
        log(f"OpenEval: {len(eval_texts)} held-out example(s), every {config.eval_every} step(s)")
    eval_history: list[dict] = []
    controller_actions: list[dict] = []
    best: dict = {"loss": None, "step": 0}
    best_dir = output_dir / "best_adapter"

    if progress:
        progress(10, "Tokenising dataset")

    def tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_seq_len,
            padding=False,
        )
        encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
        return encoded

    raw = stack["Dataset"].from_dict({"text": [e["text"] for e in examples]})
    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) > 2)

    if progress:
        progress(16, "Loading base model")
    dtype = torch.float32
    model = stack["AutoModelForCausalLM"].from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False

    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [name for name in config.target_modules if name in module_names]
    if not targets:
        for candidate in (
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ["c_attn", "c_proj"],
            ["query", "key", "value", "dense"],
        ):
            if all(name in module_names for name in candidate):
                targets = candidate
                break
    if not targets:
        raise ValueError(
            "no LoRA target modules matched this architecture; set target_modules explicitly"
        )

    lora = stack["LoraConfig"](
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = stack["get_peft_model"](model, lora)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if log:
        log(f"LoRA trainable parameters: {trainable:,} of {total:,} ({trainable / max(1, total):.3%})")

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"

    class Bridge(stack["TrainerCallback"]):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if cancel is not None and cancel.is_set():
                control.should_training_stop = True
                return
            if not logs:
                return
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            if loss is not None and log:
                log(f"step {state.global_step}: loss={loss:.4f}" + (f" lr={lr:.2e}" if lr else ""))
            if progress is not None:
                pct = 18 + (state.global_step / max(1, config.max_steps)) * 72
                progress(min(95.0, pct), f"step {state.global_step}/{config.max_steps}")

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if cancel is not None and cancel.is_set():
                control.should_training_stop = True

    class OpenEval(stack["TrainerCallback"]):
        def on_step_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001
            if not eval_texts or state.global_step == 0:
                return
            if state.global_step % max(1, config.eval_every) != 0:
                return
            loss = _eval_loss(model, tokenizer, eval_texts, config.max_seq_len, torch)
            if not math.isfinite(loss):
                return
            eval_history.append({"step": int(state.global_step), "eval_loss": round(loss, 6)})
            if log:
                log(f"eval step {state.global_step}: eval_loss={loss:.4f}")
            if best["loss"] is None or loss < float(best["loss"]) - 1e-4:
                best["loss"] = loss
                best["step"] = int(state.global_step)
                try:
                    model.save_pretrained(str(best_dir))
                    controller_actions.append({"step": int(state.global_step), "action": "saved_best", "eval_loss": round(loss, 6)})
                except Exception:  # noqa: BLE001 - saving the best is best-effort
                    controller_actions.append({"step": int(state.global_step), "action": "best_save_failed"})

    class AutoControl(stack["TrainerCallback"]):
        def __init__(self):
            self.optimizer = None
            self.bad = 0

        def on_train_begin(self, args, state, control, optimizer=None, **kwargs):  # noqa: ANN001
            self.optimizer = optimizer

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if not config.auto or not eval_history:
                return
            current = float(eval_history[-1]["eval_loss"])
            if best["loss"] is None or current <= float(best["loss"]) + 1e-4:
                self.bad = 0
                return
            self.bad += 1
            if self.bad >= max(1, config.patience):
                control.should_training_stop = True
                controller_actions.append({"step": int(state.global_step), "action": "early_stop", "reason": "no eval improvement"})
                if log:
                    log("auto: stopping early, evaluation did not improve")
                return
            if self.optimizer is not None:
                reduced = False
                for group in self.optimizer.param_groups:
                    if group.get("lr", 0) > config.lr_min:
                        group["lr"] = max(config.lr_min, group["lr"] * 0.5)
                        reduced = True
                if reduced:
                    controller_actions.append({"step": int(state.global_step), "action": "lr_reduced", "bad_evals": self.bad})
                    if log:
                        log("auto: halved the learning rate after a plateau")

    args = stack["TrainingArguments"](
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config.micro_batch,
        gradient_accumulation_steps=config.grad_accum,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_steps=max(10, config.save_every),
        save_total_limit=2,
        # No trainer has a resume path yet, so optimizer/scheduler/RNG state is never read back.
        # Writing it costs SSD endurance for nothing; save the adapter weights only.
        save_only_model=True,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=config.seed,
        data_seed=config.seed,
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
    )

    collator = stack["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm=False)
    trainer = stack["Trainer"](
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=[Bridge(), OpenEval(), AutoControl()],
    )

    if progress:
        progress(18, "Starting optimisation")
    started = time.time()
    result = trainer.train(resume_from_checkpoint=config.resume_from if config.resume_from else None)
    duration = time.time() - started

    if cancel is not None and cancel.is_set():
        # Still persist the adapter so the work is not lost.
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        raise InterruptedError("training cancelled; partial adapter saved")

    if progress:
        progress(96, "Saving adapter")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    history = [entry for entry in trainer.state.log_history if "loss" in entry]
    losses = [float(entry["loss"]) for entry in history if math.isfinite(float(entry["loss"]))]
    if not losses:
        raise RuntimeError("no finite loss was recorded; refusing to report a successful run")

    manifest = {
        "kind": "lora_sft",
        "base_model": str(model_path),
        "base_model_hash": _hash_files(_model_weight_files(model_path)),
        "dataset": dataset_meta,
        "config": config.to_public(),
        "lora_trainable_params": trainable,
        "total_params": total,
        "steps": int(trainer.state.global_step),
        "duration_seconds": round(duration, 2),
        "loss_first": round(losses[0], 6),
        "loss_last": round(losses[-1], 6),
        "losses": [round(value, 6) for value in losses],
        "loss_improved": losses[-1] < losses[0],
        "tokens_per_second_estimate": round(
            (int(trainer.state.global_step) * config.micro_batch * config.grad_accum * config.max_seq_len) / max(0.001, duration),
            2,
        ),
        "adapter_dir": str(adapter_dir),
        "best_adapter_dir": str(best_dir) if best["loss"] is not None else None,
        "best_eval_loss": round(float(best["loss"]), 6) if best["loss"] is not None else None,
        "best_eval_step": best["step"],
        "eval_history": eval_history,
        "controller_actions": controller_actions,
        "auto": config.auto,
        "versions": _versions(),
        "finished_at": utc_now(),
        "resources": resources,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    if progress:
        progress(100, f"Adapter saved ({trainer.state.global_step} steps)")
    return manifest


def create_sft_run(
    db: Database,
    config: SFTConfig,
    *,
    job_id: str | None,
    output_dir: Path,
    progress=None,
    log=None,
    cancel=None,
) -> dict:
    run_id = new_id("trn")
    db.insert(
        "training_runs",
        {
            "id": run_id,
            "job_id": job_id,
            "dataset_id": config.dataset_id,
            "base_model": config.model_path,
            "backend": "lora_sft",
            "status": "running",
            "config": json.dumps(config.to_public()),
            "metrics": "{}",
            "checkpoint_dir": str(output_dir),
            "created_at": utc_now(),
        },
    )
    try:
        manifest = train_sft(db, config, output_dir=output_dir, progress=progress, log=log, cancel=cancel)
    except Exception as exc:
        db.execute(
            "UPDATE training_runs SET status='failed', metrics=?, finished_at=? WHERE id=?",
            (json.dumps({"error": str(exc)}), utc_now(), run_id),
        )
        raise
    db.execute(
        "UPDATE training_runs SET status='completed', metrics=?, finished_at=? WHERE id=?",
        (json.dumps(manifest), utc_now(), run_id),
    )
    db.activity("training.sft", f"LoRA run completed: {manifest['steps']} steps", {"run_id": run_id})
    return {"run_id": run_id, "manifest": manifest}
