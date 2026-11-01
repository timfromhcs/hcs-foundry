"""Training playbook: remembers which configurations actually improved evaluation.

This is deliberately simple and auditable: a JSON map in the settings table keyed by a
signature of (base model, dataset hash). It is not a neural meta-learner; it is a
record of measured outcomes that biases the next automatic run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..db import Database, utc_now

KEY = "training_playbook"
MAX_ENTRIES = 200


def signature(model_path: str, dataset_hash: str | None, dataset_id: str) -> str:
    raw = f"{Path(model_path).name}|{dataset_hash or dataset_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load(db: Database) -> dict:
    value = db.get_setting(KEY, {})
    return value if isinstance(value, dict) else {}


def entries(db: Database, signature_key: str | None = None) -> list[dict]:
    data = load(db)
    if signature_key:
        return list(data.get(signature_key, []))
    flat: list[dict] = []
    for key, items in data.items():
        for item in items:
            flat.append({**item, "signature": key})
    flat.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return flat[:MAX_ENTRIES]


def suggest(db: Database, signature_key: str) -> dict | None:
    """Best known configuration for this signature, if it improved evaluation."""
    items = [item for item in entries(db, signature_key) if item.get("improved")]
    if not items:
        return None
    items.sort(key=lambda item: item.get("best_eval_loss") or 1e9)
    return items[0]


def record(
    db: Database,
    signature_key: str,
    *,
    config: dict,
    best_eval_loss: float | None,
    steps: int,
    run_id: str,
    improved: bool,
    note: str | None = None,
) -> dict:
    data = load(db)
    items = list(data.get(signature_key, []))
    entry = {
        "run_id": run_id,
        "config": {
            "learning_rate": config.get("learning_rate"),
            "lora_r": config.get("lora_r"),
            "lora_alpha": config.get("lora_alpha"),
            "micro_batch": config.get("micro_batch"),
            "grad_accum": config.get("grad_accum"),
            "max_seq_len": config.get("max_seq_len"),
            "max_steps": config.get("max_steps"),
        },
        "best_eval_loss": best_eval_loss,
        "steps": steps,
        "improved": improved,
        "note": note,
        "created_at": utc_now(),
    }
    items.append(entry)
    data[signature_key] = items[-MAX_ENTRIES:]
    db.set_setting(KEY, data)
    return entry


def merge_suggestion(base: dict, suggestion: dict | None) -> dict:
    """Start from the best known config, keeping the caller's model/dataset."""
    if not suggestion:
        return base
    merged = dict(base)
    for key in ("learning_rate", "lora_r", "lora_alpha", "micro_batch", "grad_accum"):
        value = (suggestion.get("config") or {}).get(key)
        if value is not None and key not in {"micro_batch", "grad_accum"}:
            merged[key] = value
    return merged
