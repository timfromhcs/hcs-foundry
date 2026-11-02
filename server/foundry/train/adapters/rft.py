"""Rejection-sampling fine-tuning (RFT) over verified task generators.

The loop is deliberately boring: sample the model on **train-split** generated tasks, keep only the
answers a deterministic verifier accepts, and train on those pairs. Nothing here asks a model to
grade itself, and nothing here can quietly train on a held-out template.

Truthfulness rules:
- zero accepted candidates is a **failure**, never a "completed" run;
- a template that appears in more than one split is refused before sampling;
- every count, seed and version is written into the manifest.

The candidate budget and split checks happen in preflight, so a plan can be inspected before any
model is loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from ...tasks import generators as gen
from ..base import Preflight, RunResult, TrainContext, TrainingError
from ..registry import register
from ..spec import Backend, Method, TrainSpec
from ._common import _model_dir, finish, legacy_config

VERIFIER_VERSION = "1.0"
GENERATOR_VERSION = "1.0"
DEFAULT_SAMPLES_PER_TASK = 1
DEFAULT_MAX_CANDIDATES = 64
DEFAULT_MIN_ACCEPTED = 4

#: ``(prompt, n) -> [answer, ...]``. Replaced in tests; the default samples the base model.
Sampler = Callable[[str, int], list[str]]


def _method_args(spec: TrainSpec) -> dict:
    return dict(spec.method_args or {})


def tasks_path(spec: TrainSpec) -> Path:
    raw = _method_args(spec).get("tasks_path")
    if not raw:
        raise TrainingError("rft needs method_args.tasks_path pointing at a generated JSONL file")
    return Path(str(raw)).expanduser()


def load_tasks(path: Path) -> list[gen.TaskItem]:
    if not path.is_file():
        raise TrainingError(f"task file not found: {path}")
    items: list[gen.TaskItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        items.append(gen.TaskItem(**payload))
    if not items:
        raise TrainingError(f"task file has no items: {path}")
    return items


def make_sampler(spec: TrainSpec) -> Sampler:
    """Default sampler: greedy generation from the base model. Tests replace this."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(_model_dir(spec))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    max_new_tokens = int(_method_args(spec).get("max_new_tokens", 48))

    def sample(prompt: str, n: int) -> list[str]:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=max(1, n),
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
            for row in output
        ]

    return sample


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_failed_manifest(out_dir: Path, spec: TrainSpec, reason: str, extras: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "status": "failed",
                "failure": reason,
                "spec_hash": spec.spec_hash,
                "extras": extras,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _register_dataset(db, dataset_id: str, pairs: list[tuple[gen.TaskItem, str]], *, dataset_hash: str) -> None:
    """Store accepted pairs as a frozen, synthetic dataset the SFT path can read."""
    from ...db import utc_now

    now = utc_now()
    db.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    db.insert(
        "datasets",
        {
            "id": dataset_id,
            "name": f"RFT accepted ({len(pairs)} pairs)",
            "provider": "synthetic",
            "status": "frozen",
            "rights_status": "confirmed",
            "record_count": len(pairs),
            "manifest": json.dumps({"dataset_hash": dataset_hash, "synthetic": True}),
            "stats": json.dumps({"by_split": {"train": len(pairs)}, "synthetic": True}),
            "holdout_hash": None,
            "created_at": now,
            "frozen_at": now,
            "notes": "rejection-sampling output; verified by deterministic task verifiers",
        },
    )
    for index, (task, answer) in enumerate(pairs):
        payload = {
            "schema_version": "1.0",
            "record_id": f"{task.task_id}#{index}",
            "source": {
                "provider": "synthetic",
                "adapter_version": f"rft/{GENERATOR_VERSION}",
                "rights_status": "confirmed",
            },
            "task": {
                "task_id": task.task_id,
                "task_type": "other",
                "user_request": task.prompt,
                "context_summary": f"family={task.family} template={task.template_id}",
            },
            "trace": [
                {"kind": "assistant_message", "content": answer, "timestamp": None, "tool_name": None}
            ],
            "outcome": {"status": "success", "evidence": [f"verifier:{VERIFIER_VERSION}"]},
            "privacy": {"contains_sensitive_data": False, "approved_for_training": True},
            "split": "train",
        }
        encoded = json.dumps(payload, sort_keys=True)
        db.insert(
            "dataset_records",
            {
                "id": f"{dataset_id}_{index}",
                "dataset_id": dataset_id,
                "record_key": f"{task.task_id}#{index}",
                "split": "train",
                "synthetic": 1,
                "excluded": 0,
                "exclusion_reason": None,
                "rights_status": "confirmed",
                "content_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                "simhash": None,
                "payload": encoded,
            },
        )


@register
class RftTrainer:
    method = Method.RFT
    backend = Backend.FP32_LORA

    # -- preflight ---------------------------------------------------------
    def preflight(self, spec: TrainSpec, ctx: TrainContext) -> Preflight:
        blockers: list[str] = []
        warnings: list[str] = []
        try:
            _model_dir(spec)
        except TrainingError as exc:
            blockers.append(str(exc))

        args = _method_args(spec)
        try:
            items = load_tasks(tasks_path(spec))
        except (TrainingError, json.JSONDecodeError) as exc:
            items = []
            blockers.append(str(exc))

        samples = int(args.get("samples_per_task", DEFAULT_SAMPLES_PER_TASK))
        budget = int(args.get("max_candidates", DEFAULT_MAX_CANDIDATES))
        if samples < 1:
            blockers.append("samples_per_task must be >= 1")
        if items:
            train_items = [item for item in items if item.split == "train"]
            if not train_items:
                blockers.append("the task file has no train-split items")
            leaks = gen.check_template_isolation(items)
            if leaks:
                blockers.append(
                    "templates appear in more than one split: " + ", ".join(leaks)
                )
            wanted = len(train_items) * max(1, samples)
            if wanted > budget:
                blockers.append(
                    f"candidate budget {budget} is smaller than the planned {wanted} "
                    f"({len(train_items)} train items x {samples})"
                )
            if wanted < int(args.get("min_accepted", DEFAULT_MIN_ACCEPTED)):
                warnings.append(
                    f"only {wanted} candidate(s) planned but {args.get('min_accepted', DEFAULT_MIN_ACCEPTED)} "
                    "acceptances are required"
                )

        return Preflight(
            ok=not blockers,
            blockers=blockers,
            warnings=warnings,
            est_step_seconds=(0.0, 0.0),
        )

    # -- run ---------------------------------------------------------------
    def run(self, spec: TrainSpec, ctx: TrainContext, *, resume_from=None) -> RunResult:
        args = _method_args(spec)
        items = load_tasks(tasks_path(spec))
        leaks = gen.check_template_isolation(items)
        if leaks:
            raise TrainingError("refusing to sample: templates leak across splits: " + ", ".join(leaks))

        train_items = [item for item in items if item.split == "train"]
        test_items = [item for item in items if item.split == "test"]
        if not train_items:
            raise TrainingError("the task file has no train-split items")

        samples_per_task = int(args.get("samples_per_task", DEFAULT_SAMPLES_PER_TASK))
        max_candidates = int(args.get("max_candidates", DEFAULT_MAX_CANDIDATES))
        min_accepted = int(args.get("min_accepted", DEFAULT_MIN_ACCEPTED))
        seed = int(args.get("seed", 0))

        planned = len(train_items) * max(1, samples_per_task)
        if planned > max_candidates:
            raise TrainingError(
                f"candidate budget {max_candidates} is smaller than the planned {planned} "
                f"({len(train_items)} train items x {samples_per_task}); lower samples_per_task "
                "or raise max_candidates"
            )

        sampler = make_sampler(spec)
        accepted: list[tuple[gen.TaskItem, str]] = []
        rejected: dict[str, int] = {}
        candidates = 0
        for item in train_items:
            if candidates >= max_candidates:
                break
            want = min(samples_per_task, max_candidates - candidates)
            ctx.progress(2, f"sampling {item.template_id}")
            try:
                answers = sampler(item.prompt, want)
            except Exception as exc:  # noqa: BLE001 - a broken sampler is a result
                rejected["sampler_error"] = rejected.get("sampler_error", 0) + 1
                ctx.log(f"sampler error on {item.template_id}: {type(exc).__name__}: {exc}")
                continue
            for answer in answers:
                candidates += 1
                if not str(answer or "").strip():
                    rejected["empty_response"] = rejected.get("empty_response", 0) + 1
                elif gen.verify(item, answer):
                    accepted.append((item, str(answer).strip()))
                else:
                    rejected["verifier_rejected"] = rejected.get("verifier_rejected", 0) + 1

        counts = {
            "accepted_count": len(accepted),
            "rejected_count": sum(rejected.values()),
            "candidates": candidates,
            "rejection_reasons": dict(rejected),
            "used_test_items": len(test_items),
            "seed": seed,
            "generator_version": GENERATOR_VERSION,
            "verifier_version": VERIFIER_VERSION,
        }

        if len(accepted) < min_accepted:
            reason = (
                f"only {len(accepted)} of {candidates} candidate(s) passed the verifier; "
                f"{min_accepted} required"
            )
            _write_failed_manifest(ctx.out_dir, spec, reason, counts)
            raise TrainingError(reason)

        accepted_text = "\n".join(
            json.dumps({"task_id": task.task_id, "prompt": task.prompt, "answer": answer}, sort_keys=True)
            for task, answer in accepted
        )
        dataset_hash = _sha256_text(accepted_text)
        dataset_id = f"rft_{dataset_hash[7:19]}"
        accepted_path = ctx.out_dir / "accepted.jsonl"
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        accepted_path.write_text(accepted_text + "\n", encoding="utf-8")
        _register_dataset(ctx.db, dataset_id, accepted, dataset_hash=dataset_hash)

        ctx.log(
            f"RFT: {len(accepted)} accepted / {candidates} candidates "
            f"({sum(rejected.values())} rejected) -> dataset {dataset_id}"
        )
        ctx.progress(30, f"training on {len(accepted)} accepted pair(s)")

        from ...training import create_sft_run

        result = create_sft_run(
            ctx.db,
            legacy_config(spec, dataset_id=dataset_id),
            job_id=None,
            output_dir=ctx.out_dir / "sft",
            progress=lambda pct, message: ctx.progress(30 + pct * 0.7, message),
            log=ctx.log,
            cancel=ctx.cancel,
        )
        run = finish(spec, ctx, result=result, started=ctx.clock(), clock=ctx.clock)
        run.manifest.extras.update(counts)
        run.manifest.extras["accepted_dataset_id"] = dataset_id
        run.manifest.extras["accepted_dataset_hash"] = dataset_hash
        run.manifest.extras["accepted_file"] = str(accepted_path)
        run.manifest.dataset_id = dataset_id
        run.manifest.dataset_manifest_hash = dataset_hash
        return run
