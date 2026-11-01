"""Training algorithms: SFT, distillation and DPO.

All three operate on a frozen dataset with confirmed rights. SFT and distillation share
the LoRA trainer; DPO uses a reference-free PEFT trick (adapter disabled for the
reference pass) so it does not need a second model in memory.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

from ..db import Database, new_id, utc_now
from ..data.pipeline import get_dataset
from . import playbook
from .sft import SFTConfig, build_eval_texts, build_examples, create_sft_run, train_sft


def _config_dict(config: SFTConfig) -> dict:
    return config.to_public()


def run_auto(
    db: Database,
    config: SFTConfig,
    *,
    job_id: str | None,
    output_dir: Path,
    progress=None,
    log=None,
    cancel=None,
) -> dict:
    """SFT with OpenEval + auto-decision, seeded by the playbook."""
    dataset = get_dataset(db, config.dataset_id)
    if not dataset:
        raise ValueError("dataset not found")
    signature_key = playbook.signature(config.model_path, (dataset.get("manifest") or {}).get("dataset_hash"), config.dataset_id)
    suggestion = playbook.suggest(db, signature_key) if config.auto else None
    if suggestion and log:
        log(f"playbook: starting from best known config (eval {suggestion.get('best_eval_loss')})")
    if suggestion:
        merged = playbook.merge_suggestion(_config_dict(config), suggestion)
        config = SFTConfig(**merged)

    config.auto = True
    if config.eval_every <= 0:
        config.eval_every = max(1, config.max_steps // 4)

    result = create_sft_run(
        db,
        config,
        job_id=job_id,
        output_dir=output_dir,
        progress=progress,
        log=log,
        cancel=cancel,
    )
    manifest = result["manifest"]
    best = manifest.get("best_eval_loss")
    history = manifest.get("eval_history") or []
    eval_improved = bool(len(history) >= 2 and history[-1]["eval_loss"] < history[0]["eval_loss"])
    improved = best is not None
    playbook.record(
        db,
        signature_key,
        config=_config_dict(config),
        best_eval_loss=best,
        steps=int(manifest.get("steps") or 0),
        run_id=result["run_id"],
        improved=improved,
        note="auto run",
    )
    manifest["playbook_signature"] = signature_key
    manifest["playbook_seeded"] = bool(suggestion)
    manifest["eval_improved"] = eval_improved
    try:
        (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    except OSError:
        pass
    return {"run_id": result["run_id"], "manifest": manifest, "signature": signature_key}


def run_distill(
    db: Database,
    config: SFTConfig,
    *,
    teacher_complete: Callable[[list[dict]], str],
    prompts: list[str],
    job_id: str | None,
    output_dir: Path,
    progress=None,
    log=None,
    cancel=None,
) -> dict:
    """Distil a teacher into the student: the teacher writes answers, SFT trains them."""
    if not prompts:
        raise ValueError("no prompts for distillation")
    extra: list[str] = []
    total = max(1, len(prompts))
    for index, prompt in enumerate(prompts):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("distillation cancelled")
        try:
            answer = teacher_complete([{"role": "user", "content": prompt}])
        except Exception as exc:  # noqa: BLE001 - one failed prompt must not kill the run
            if log:
                log(f"teacher failed for prompt {index}: {exc}")
            continue
        if answer and answer.strip():
            extra.append(f"### User\n{prompt.strip()}\n\n### Assistant\n{answer.strip()}\n")
        if progress:
            progress(4 + (index / total) * 12, f"teacher {index + 1}/{total}")
    if not extra:
        raise ValueError("the teacher produced no usable answers")

    config.extra_texts = extra
    result = create_sft_run(
        db,
        config,
        job_id=job_id,
        output_dir=output_dir,
        progress=progress,
        log=log,
        cancel=cancel,
    )
    manifest = result["manifest"]
    manifest["kind"] = "distillation"
    manifest["teacher_samples"] = len(extra)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    return {"run_id": result["run_id"], "manifest": manifest}


def _corrupt(answer: str) -> str:
    words = answer.split()
    if len(words) <= 3:
        return "I do not know."
    return " ".join(words[: max(1, len(words) // 3)]) + " I am not sure."


def run_dpo(
    db: Database,
    config: SFTConfig,
    *,
    beta: float = 0.1,
    job_id: str | None = None,
    output_dir: Path | None = None,
    progress=None,
    log=None,
    cancel=None,
) -> dict:
    """Direct preference optimisation with synthetic pairs.

    The rejected answer is a truncated/hedged version of the chosen answer. The
    reference is the same model with the LoRA adapter disabled, which avoids a second
    model in memory.
    """
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"training stack not importable: {exc}") from exc

    dataset = get_dataset(db, config.dataset_id)
    if not dataset:
        raise ValueError("dataset not found")
    if dataset["rights_status"] != "confirmed":
        raise ValueError("dataset rights are not confirmed")
    rows = db.query(
        "SELECT payload FROM dataset_records WHERE dataset_id=? AND split='train' AND excluded=0 LIMIT 200",
        (config.dataset_id,),
    )
    pairs = []
    for row in rows:
        payload = json.loads(row["payload"])
        prompt = ((payload.get("task") or {}).get("user_request") or "").strip()
        chosen = ""
        for step in reversed(payload.get("trace") or []):
            if step.get("kind") == "assistant_message" and (step.get("content") or "").strip():
                chosen = step["content"].strip()
                break
        if prompt and chosen:
            pairs.append((prompt, chosen, _corrupt(chosen)))
    if len(pairs) < 2:
        raise ValueError("DPO needs at least two records with a prompt and an answer")

    model_path = Path(config.model_path).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache = False

    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [name for name in config.target_modules if name in module_names]
    if not targets:
        for candidate in (["q_proj", "k_proj", "v_proj", "o_proj"], ["c_attn", "c_proj"]):
            if all(name in module_names for name in candidate):
                targets = candidate
                break
    if not targets:
        raise ValueError("no LoRA target modules matched this architecture")
    model = get_peft_model(
        model,
        LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=targets),
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
    model.train()
    losses: list[float] = []
    output_dir = output_dir or Path("dpo-run")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def encode(prompt: str, answer: str):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=config.max_seq_len)["input_ids"]
        ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
        return ids, len(prompt_ids)

    def logprob(ids, prompt_len, reference: bool):
        context = model.disable_adapter() if reference else _nullcontext()
        with context:
            logits = model(input_ids=ids.unsqueeze(0)).logits[:, :-1]
            logprobs = torch.log_softmax(logits, dim=-1)
            targets = ids[1:].unsqueeze(0)
            token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)
            return token_lp[prompt_len - 1 :].sum()

    steps = 0
    for step in range(config.max_steps):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("DPO cancelled")
        prompt, chosen, rejected = pairs[step % len(pairs)]
        ids_c, len_c = encode(prompt, chosen)
        ids_r, len_r = encode(prompt, rejected)
        if len_c < 2 or len_r < 2:
            continue
        lp_c = logprob(ids_c, len_c, False)
        lp_r = logprob(ids_r, len_r, False)
        with torch.no_grad():
            ref_c = logprob(ids_c, len_c, True)
            ref_r = logprob(ids_r, len_r, True)
        margin = (lp_c - lp_r) - (ref_c - ref_r)
        loss = -torch.nn.functional.logsigmoid(beta * margin)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        steps += 1
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError("DPO loss became non-finite")
        losses.append(value)
        if log:
            log(f"dpo step {steps}: loss={value:.4f}")
        if progress:
            progress(5 + (steps / max(1, config.max_steps)) * 85, f"dpo step {steps}/{config.max_steps}")

    adapter = output_dir / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    manifest = {
        "kind": "dpo",
        "base_model": str(model_path),
        "dataset_id": config.dataset_id,
        "pairs": len(pairs),
        "steps": steps,
        "beta": beta,
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "loss_improved": bool(losses and losses[-1] < losses[0]),
        "duration_seconds": round(time.time() - started, 2),
        "adapter_dir": str(adapter),
        "finished_at": utc_now(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    return {"run_id": new_id("trn"), "manifest": manifest}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
