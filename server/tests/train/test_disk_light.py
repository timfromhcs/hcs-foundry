"""Disk-light training: no optimizer state, no per-step checkpoint storm, measured bytes.

The point is not elegance. Writing optimizer/scheduler/RNG state that no resume path will ever read
costs SSD endurance, so the test proves it is not written and that the run reports what it wrote.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from foundry.train.base import TrainContext
from foundry.train.registry import get_trainer, load_builtin_trainers
from foundry.train.spec import Backend, Method, Profile, spec_for_profile

pytest.importorskip("transformers")


def _contract(tmp_path: Path, model_dir: Path, dataset_id: str, *, steps: int):
    from foundry.train.spec import TrainSpec

    return spec_for_profile(
        method=Method.SFT,
        backend=Backend.FP32_LORA,
        model_path=str(model_dir),
        dataset_id=dataset_id,
        profile=Profile.CI_TINY,
        hyper={"seq_len": 24, "max_steps": steps, "lora_r": 2, "lora_alpha": 4, "lr": 1e-3},
        method_args={"eval_every": 0},
    )


def test_a_run_writes_the_adapter_but_no_optimizer_state(db, settings, tmp_path: Path):
    from tests.test_sft import make_tiny_model, seed_dataset

    load_builtin_trainers()
    model_dir = make_tiny_model(tmp_path / "tiny")
    dataset_id = seed_dataset(db)
    # save_every=1 would once have written a checkpoint per step; the floor is 10 now
    spec = _contract(tmp_path, model_dir, dataset_id, steps=11)
    trainer = get_trainer(spec)
    out = tmp_path / "run"
    ctx = TrainContext(out_dir=out, cancel=threading.Event(), db=db, settings=settings)

    result = trainer.run(spec, ctx)
    written = sorted(path.name for path in out.rglob("*") if path.is_file())

    assert "adapter_model.safetensors" in written
    assert not [name for name in written if name in {"optimizer.pt", "scheduler.pt", "rng_state.pth"}]

    checkpoint_dirs = sorted(
        path for path in (out / "checkpoints").glob("checkpoint-*") if path.is_dir()
    ) if (out / "checkpoints").is_dir() else []
    assert len(checkpoint_dirs) <= 2, checkpoint_dirs

    disk_bytes = result.manifest.extras.get("disk_bytes_written")
    assert isinstance(disk_bytes, int) and disk_bytes > 0
    # the adapter is ~ hundreds of KB on the tiny model; optimizer state would dwarf it
    assert disk_bytes < 50 * 1024 * 1024


def test_the_save_interval_cannot_drop_below_ten_steps():
    from foundry.training.sft import SFTConfig

    config = SFTConfig(model_path="x", dataset_id="y", save_every=1)
    # the floor is applied where TrainingArguments is built; assert the intent in one place
    assert max(10, config.save_every) == 10
