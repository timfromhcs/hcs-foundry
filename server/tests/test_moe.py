from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="training extras are not installed")

from foundry.moe import MoEConfig, MoEError, estimate, moeify  # noqa: E402


def make_tiny_model(path: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "hello": 3, "world": 4, "fix": 5, "test": 6}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    path.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(path))
    config = GPT2Config(vocab_size=len(vocab), n_positions=32, n_embd=32, n_layer=2, n_head=2)
    GPT2LMHeadModel(config).save_pretrained(str(path))
    return path


def test_estimate_reports_projection(tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    result = estimate(model_dir, MoEConfig(experts=4, top_k=2))
    assert result["projected_bytes"] > 0
    assert result["layers"] == 2
    assert result["fits_ram"] is True


def test_moeify_converts_blocks_and_saves_manifest(tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    out = tmp_path / "moe"
    result = moeify(model_dir, out, config=MoEConfig(experts=3, top_k=2, noise=0.0))
    manifest = result["manifest"]
    assert manifest["kind"] == "moe_dense_conversion"
    assert manifest["experts"] == 3
    assert manifest["blocks_converted"] == 2
    assert manifest["trained"] is False
    assert (out / "moe_manifest.json").exists()
    saved = json.loads((out / "moe_manifest.json").read_text("utf-8"))
    assert saved["top_k"] == 2
    assert saved["output_sha256"]


def test_moe_rejects_invalid_top_k(tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    with pytest.raises(MoEError):
        moeify(model_dir, tmp_path / "out", config=MoEConfig(experts=2, top_k=3))


def test_moe_rejects_missing_source(tmp_path: Path):
    with pytest.raises(MoEError):
        moeify(tmp_path / "missing", tmp_path / "out", config=MoEConfig(experts=2, top_k=1))
