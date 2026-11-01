from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="training extras are not installed")
pytest.importorskip("peft", reason="training extras are not installed")

from foundry.training.sft import SFTConfig, estimate_resources, train_sft  # noqa: E402

WORDS = [
    "<pad>",
    "<eos>",
    "<unk>",
    "###",
    "User",
    "Assistant",
    "fix",
    "the",
    "failing",
    "test",
    "and",
    "keep",
    "diff",
    "small",
    "the",
    "null",
    "branch",
    "returns",
    "none",
    "patch",
    "handles",
    "it",
    "tests",
    "pass",
]


def make_tiny_model(path: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {}
    for word in WORDS:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    path.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(path))

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(str(path))
    return path


def seed_dataset(db, dataset_id: str = "ds_train") -> str:
    db.insert(
        "datasets",
        {
            "id": dataset_id,
            "name": "Tiny train",
            "provider": "generic",
            "status": "frozen",
            "rights_status": "confirmed",
            "record_count": 2,
            "manifest": json.dumps({"dataset_hash": "abc"}),
            "stats": json.dumps({"by_split": {"train": 2}}),
            "holdout_hash": None,
            "created_at": "2026-01-01T00:00:00Z",
            "frozen_at": "2026-01-01T00:00:00Z",
        },
    )
    for index in range(2):
        payload = {
            "record_id": f"rec_{index}",
            "source": {"provider": "generic", "adapter_version": "test", "rights_status": "confirmed"},
            "task": {"user_request": "fix the failing test and keep the diff small"},
            "trace": [
                {"kind": "user_message", "content": "fix the failing test"},
                {"kind": "assistant_message", "content": "the null branch returns none and the patch handles it tests pass"},
            ],
            "outcome": {"status": "success"},
            "teaching": {},
            "privacy": {},
            "split": "train",
            "metadata": {},
            "schema_version": "1.0",
        }
        db.insert(
            "dataset_records",
            {
                "id": f"row_{index}",
                "dataset_id": dataset_id,
                "record_key": f"rec_{index}",
                "split": "train",
                "synthetic": 0,
                "excluded": 0,
                "rights_status": "confirmed",
                "content_hash": f"sha256:{index}",
                "simhash": 0,
                "payload": json.dumps(payload),
            },
        )
    return dataset_id


def test_estimate_resources_returns_a_range(tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    config = SFTConfig(model_path=str(model_dir), dataset_id="x", max_steps=2, max_seq_len=32)
    estimate = estimate_resources(model_dir, config)
    assert estimate["estimated_high_bytes"] > estimate["estimated_low_bytes"] > 0
    assert estimate["confidence"] in {"low", "medium"}
    assert "note" in estimate


def test_sft_trains_and_saves_an_adapter(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    output = tmp_path / "run"
    config = SFTConfig(
        model_path=str(model_dir),
        dataset_id=dataset_id,
        max_steps=2,
        max_seq_len=32,
        micro_batch=1,
        grad_accum=1,
        lora_r=2,
        lora_alpha=4,
        learning_rate=1e-3,
        save_every=2,
    )
    logs: list[str] = []
    manifest = train_sft(db, config, output_dir=output, log=logs.append)

    assert manifest["steps"] == 2
    assert manifest["loss_first"] is not None
    assert manifest["loss_last"] is not None
    assert manifest["dataset"]["records_used"] == 2
    assert manifest["lora_trainable_params"] > 0
    adapter = Path(manifest["adapter_dir"])
    assert (adapter / "adapter_config.json").exists()
    assert (output / "manifest.json").exists()
    assert any("loss=" in line for line in logs)
    saved = json.loads((output / "manifest.json").read_text("utf-8"))
    assert saved["kind"] == "lora_sft"


def test_sft_rejects_missing_model(db, tmp_path: Path):
    dataset_id = seed_dataset(db)
    config = SFTConfig(model_path=str(tmp_path / "nope"), dataset_id=dataset_id, max_steps=1)
    with pytest.raises(ValueError):
        train_sft(db, config, output_dir=tmp_path / "run")


def test_sft_rejects_dataset_without_rights(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db, "ds_bad")
    db.execute("UPDATE datasets SET rights_status='needs_review' WHERE id=?", (dataset_id,))
    config = SFTConfig(model_path=str(model_dir), dataset_id=dataset_id, max_steps=1, max_seq_len=32)
    with pytest.raises(ValueError):
        train_sft(db, config, output_dir=tmp_path / "run")
