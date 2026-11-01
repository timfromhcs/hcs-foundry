# Training

## The honest state machine

| State | Meaning | What you can do |
|---|---|---|
| `unsupported` | PyTorch missing or not importable | Nothing; the UI explains why |
| `cpu-experimental` | CPU torch works, fine-tuning stack missing or broken | Run the smoke optimisation |
| `gpu` | Accelerated torch device available **and** the stack importable | Preflight for a real fine-tune |
| verified fine-tune | Set only after a completed run with checkpoints on this machine | — |

The environment report lists every package with its version, whether it is **importable**, and the import error if it is not. Metadata alone is not treated as proof: a package with a broken dependency is reported as missing.

## Smoke optimisation

`POST /api/v1/training/smoke` runs a genuine 40-step optimisation on a tiny MLP:

- records the loss before and after,
- asserts the loss is finite,
- asserts the maximum parameter delta is greater than zero.

If any assertion fails, the run is `smoke_failed` and the reason is stored. This is the only thing on the tested machine that can honestly be called "training".

Measured result on the tested machine: loss `2.1940 → 0.0303`, parameter update verified, CPU only.

## Preflight

`POST /api/v1/training/preflight` checks:

- dataset exists, rights confirmed, has a training split,
- record count and synthetic ratio (warnings, not blockers),
- PyTorch and fine-tuning stack importability,
- a training-native base model was supplied,
- free disk space.

It returns `blockers`, `warnings`, `can_start`, and a memory-aware preset (micro-batch, gradient accumulation, sequence length, LoRA rank, learning rate, checkpoint interval).

## Why GGUF is not trained directly

GGUF is an inference/distribution format. A fine-tune needs a training-native checkpoint (safetensors + tokenizer). The correct flow is:

```text
base checkpoint → SFT/LoRA → adapter → validation → optional merge → convert → quantize → GGUF
```

0.1.0 implements the detection and the smoke step. It does **not** claim to run a real fine-tune on hardware where the stack is unavailable, and it does not implement GGUF export.

## Presets

| Preset | When |
|---|---|
| tiny smoke | always, if torch imports |
| CPU experimental | CPU-only; batch 1, grad-accum 16, seq 256, rank 4 |
| GPU balanced | only when a CUDA device is verified; batch 2, grad-accum 8, seq 1024, rank 16 |

## Jobs and checkpoints

Training runs are jobs: they have progress, events, cancellation and an output payload. Checkpoint retention is part of the run config; the last three and the best checkpoint are kept. A source model is never overwritten.

## What is deliberately absent

- DPO/GRPO: no preference-pair pipeline is shipped yet, so the method is blocked with a precise reason.
- Full fine-tuning: not offered on shared-memory hardware.
- Distributed training.
- Automatic promotion of a trained adapter: it must pass the same benchmark and upgrade gates as any other candidate.
