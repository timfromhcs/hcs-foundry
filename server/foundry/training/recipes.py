"""Training recipes with explicit, measured-where-possible estimates.

Recipes never claim a result. They describe a configuration and a projected cost so
the user can decide before spending hours of compute.
"""

from __future__ import annotations

from .sft import SFTConfig, estimate_resources

#: Conservative CPU throughput for a 3B-class model with LoRA, tokens/second.
#: Measured basis: the smoke path and small-model runs on the test machine; this is
#: deliberately pessimistic and labelled as an estimate.
CPU_TOKENS_PER_SECOND_3B = 6.0
CPU_TOKENS_PER_SECOND_1B = 18.0


def _hours(steps: int, batch: int, accum: int, seq: int, tokens_per_second: float) -> float:
    tokens = steps * batch * accum * seq
    return round(tokens / max(0.1, tokens_per_second) / 3600, 2)


def recipe_catalog() -> list[dict]:
    return [
        {
            "id": "tiny-proof",
            "name": "Tiny proof",
            "description": "Proves the pipeline end to end in minutes on CPU with a tiny local model.",
            "requires": "a tiny local model directory (for example a 1-layer test model)",
            "config": {
                "max_steps": 20,
                "max_seq_len": 128,
                "micro_batch": 1,
                "grad_accum": 1,
                "lora_r": 4,
                "learning_rate": 1e-3,
            },
            "projection": {"hours": 0.05, "confidence": "medium", "basis": "tiny model, CPU"},
        },
        {
            "id": "3b-cpu",
            "name": "3B on CPU (not recommended)",
            "description": "A 3B-class LoRA at 512 tokens. Useful only to prove the path; expect days.",
            "requires": "a local 3B safetensors model directory",
            "config": {
                "max_steps": 200,
                "max_seq_len": 512,
                "micro_batch": 1,
                "grad_accum": 8,
                "lora_r": 16,
                "learning_rate": 2e-4,
            },
            "projection": {
                "hours": _hours(200, 1, 8, 512, CPU_TOKENS_PER_SECOND_3B),
                "confidence": "low",
                "basis": f"assumes {CPU_TOKENS_PER_SECOND_3B} tokens/s on CPU; measure before running",
            },
            "warning": "On this class of hardware a useful 3B fine-tune is not practical. Prefer a GPU machine.",
        },
        {
            "id": "3b-gpu",
            "name": "3B on a 12 GB GPU",
            "description": "Recommended shape for a real 3B LoRA: 1024 tokens, rank 16, gradient accumulation.",
            "requires": "a CUDA GPU with at least 12 GB VRAM and a local 3B safetensors model",
            "config": {
                "max_steps": 1200,
                "max_seq_len": 1024,
                "micro_batch": 2,
                "grad_accum": 8,
                "lora_r": 16,
                "learning_rate": 2e-4,
            },
            "projection": {"hours": 3.5, "confidence": "low", "basis": "typical consumer GPU; not measured here"},
        },
        {
            "id": "1b-cpu",
            "name": "1B on CPU (experimental)",
            "description": "Small enough to finish a short run in hours on CPU.",
            "requires": "a local 1B safetensors model directory",
            "config": {
                "max_steps": 120,
                "max_seq_len": 256,
                "micro_batch": 1,
                "grad_accum": 4,
                "lora_r": 8,
                "learning_rate": 2e-4,
            },
            "projection": {
                "hours": _hours(120, 1, 4, 256, CPU_TOKENS_PER_SECOND_1B),
                "confidence": "low",
                "basis": f"assumes {CPU_TOKENS_PER_SECOND_1B} tokens/s on CPU",
            },
        },
    ]


def recipe_config(recipe_id: str, *, model_path: str, dataset_id: str) -> SFTConfig:
    recipe = next((r for r in recipe_catalog() if r["id"] == recipe_id), None)
    if recipe is None:
        raise ValueError(f"unknown recipe: {recipe_id}")
    return SFTConfig(model_path=model_path, dataset_id=dataset_id, **recipe["config"])


def project(recipe_id: str, model_path: str, dataset_id: str) -> dict:
    config = recipe_config(recipe_id, model_path=model_path, dataset_id=dataset_id)
    recipe = next(r for r in recipe_catalog() if r["id"] == recipe_id)
    resources = estimate_resources(model_path, config)
    return {
        "recipe": recipe,
        "config": config.to_public(),
        "resources": resources,
        "projection": recipe["projection"],
    }
