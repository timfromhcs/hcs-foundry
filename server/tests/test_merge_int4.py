from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="training extras are not installed")

from foundry.training.int4 import dequantize_int4, train_int4  # noqa: E402
from foundry.training.merge_int4 import merge_int4_adapter  # noqa: E402
from foundry.training.offload import Int4Store, build_store  # noqa: E402
from foundry.training.sft import SFTConfig  # noqa: E402

from .test_sft import make_tiny_model, seed_dataset  # noqa: E402


def _train(db, tmp_path: Path):
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    config = SFTConfig(
        model_path=str(model_dir),
        dataset_id=dataset_id,
        max_steps=1,
        max_seq_len=24,
        grad_accum=1,
        lora_r=2,
        lora_alpha=4,
        learning_rate=1e-3,
        target_modules=["c_attn", "c_proj"],
    )
    manifest = train_int4(db, config, output_dir=tmp_path / "run", streaming=True, offload=True)
    return model_dir, manifest


def test_merge_rebuilds_dequantized_weights_plus_delta(db, tmp_path: Path):
    import torch
    from safetensors.torch import load_file

    model_dir, manifest = _train(db, tmp_path)
    store_dir = Path(manifest["offload"]["store_dir"])
    merged = merge_int4_adapter(store_dir, manifest["adapter_dir"], tmp_path / "merged")

    out = Path(merged["output"])
    index = json.loads((out / "model.safetensors.index.json").read_text("utf-8"))
    assert index["weight_map"], "index must map every tensor to a shard"
    assert len(merged["shards"]) == len(set(merged["shards"]))
    assert merged["quantized_weights_merged"] == manifest["converted_layers"]

    tensors = {}
    for shard in merged["shards"]:
        tensors.update(load_file(str(out / shard)))

    # every tensor named in the index must exist in exactly one shard
    assert set(tensors) == set(index["weight_map"])

    store = Int4Store(store_dir)
    adapter = torch.load(str(Path(manifest["adapter_dir"]) / "lora_int4.pt"), map_location="cpu", weights_only=True)
    module = next(iter(store.meta["tensors"]))
    info = store.meta["tensors"][module]
    shape = (info["out_features"], info["in_features"], info["padded"])
    base = dequantize_int4(store.packed(module), store.scales(module), shape, torch.float32)
    delta = (
        adapter["lora"][f"{module}.lora_B"].float() @ adapter["lora"][f"{module}.lora_A"].float()
    ) * (adapter["lora_alpha"] / adapter["lora_r"])
    expected = (base + delta).to(torch.float16)

    written = tensors[f"{module}.weight"]
    assert written.dtype == torch.float16
    assert written.shape == expected.shape
    assert torch.allclose(written.float(), expected.float(), atol=2e-2)

    config = json.loads((out / "config.json").read_text("utf-8"))
    assert config["dtype"] == "float16"
    assert (out / "tokenizer.json").exists()
    store.close()


def test_merge_shards_when_the_budget_is_small(db, tmp_path: Path):
    from safetensors.torch import load_file

    _, manifest = _train(db, tmp_path)
    merged = merge_int4_adapter(
        Path(manifest["offload"]["store_dir"]),
        manifest["adapter_dir"],
        tmp_path / "merged",
        shard_bytes=4096,
    )
    out = Path(merged["output"])
    assert len(merged["shards"]) > 1
    index = json.loads((out / "model.safetensors.index.json").read_text("utf-8"))
    seen = {}
    for shard in merged["shards"]:
        for key in load_file(str(out / shard)):
            assert key not in seen, f"{key} appears in two shards"
            seen[key] = shard
    assert seen == index["weight_map"]


def test_merge_refuses_a_store_that_does_not_match_the_adapter(db, tmp_path: Path):
    import torch

    model_dir = make_tiny_model(tmp_path / "tiny")
    store = build_store(model_dir, tmp_path / "store", targets=["c_attn"])
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    torch.save(
        {
            "lora": {"transformer.h.0.attn.c_attn.lora_A": torch.zeros(2, 32)},
            "lora_r": 2,
            "lora_alpha": 4,
            "converted_layers": 5,
            "target_modules": ["c_attn"],
        },
        adapter_dir / "lora_int4.pt",
    )
    with pytest.raises(ValueError, match="trained on 5"):
        merge_int4_adapter(tmp_path / "store", adapter_dir, tmp_path / "merged")
    assert store["counts"]["quantized"] >= 1


def test_merge_refuses_mismatched_target_sets(db, tmp_path: Path):
    import torch

    model_dir = make_tiny_model(tmp_path / "tiny")
    build_store(model_dir, tmp_path / "store", targets=["c_attn"])
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    torch.save(
        {
            "lora": {"transformer.h.0.attn.c_attn.lora_A": torch.zeros(2, 32)},
            "lora_r": 2,
            "lora_alpha": 4,
            "converted_layers": 1,
            "target_modules": ["c_attn", "c_proj"],
        },
        adapter_dir / "lora_int4.pt",
    )
    with pytest.raises(ValueError, match="without these adapter targets"):
        merge_int4_adapter(tmp_path / "store", adapter_dir, tmp_path / "merged")


def test_merge_requires_an_adapter_file(tmp_path: Path):
    with pytest.raises(ValueError, match="no lora_int4.pt"):
        merge_int4_adapter(tmp_path / "store", tmp_path / "missing", tmp_path / "merged")
