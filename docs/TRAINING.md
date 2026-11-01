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

## LoRA/SFT backend (v2)

`foundry/training/sft.py` implements a real LoRA fine-tune:

1. Load a **local** safetensors base model (no hidden downloads).
2. Build examples from the frozen dataset's train split (`### User` / `### Assistant`).
3. Tokenise with the model's own tokenizer and truncate to the configured sequence length.
4. Attach LoRA adapters with architecture-aware target modules.
5. Train with `transformers.Trainer`, live loss events, checkpoints and cooperative cancellation.
6. Save the adapter and a manifest with base-model hash, dataset hash, library versions and resources.

It refuses to run when the estimated memory exceeds available RAM, when rights are not confirmed, or when no example has both a request and an answer.

### Measured proof on the test machine

Base: `Qwen/Qwen3-0.6B` (pulled anonymously from the Hub, 1.5 GB).
Dataset: 16 train records. LoRA r=4, sequence 128, batch 1.

```
steps            2
duration         4.16 s
loss             6.043644 → 5.014467
trainable params 2,523,136 of 598,573,056 (0.42 %)
throughput       61.6 tokens/s (estimated)
adapter          data/training-runs/proof-qwen06b/adapter
```

The adapter directory contains `adapter_config.json` and the LoRA weights. That is a real optimisation, not a simulation.

### Why there is no "good 3B model" from this machine

A 3B model is roughly five times larger. At the measured rate the same setup lands around 10–15 tokens/s on this CPU, so a 200-step run at 512 tokens with gradient accumulation 8 needs roughly **15–38 hours**, and a useful SFT run needs several times that. There is no CUDA or ROCm device here.

The honest deliverable is the verified trainer plus a recipe and a projection. The `3b-cpu` recipe prints that warning; `3b-gpu` is the shape to use on a 12 GB GPU.

### Recipes

| id | Purpose |
|---|---|
| `tiny-proof` | end-to-end proof in minutes |
| `1b-cpu` | short experimental run on CPU |
| `3b-cpu` | warns that this is not practical |
| `3b-gpu` | recommended shape for a real 3B LoRA |

### Using the adapter

The adapter is a PEFT artifact. To use it for inference you need the base model and `peft`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("E:/models/Qwen3-0.6B")
model = PeftModel.from_pretrained(base, "data/training-runs/proof-qwen06b/adapter")
```

Converting a merged model to GGUF is **not** implemented in v2. That remains an explicit, documented gap rather than a fake button.

## INT4 (QLoRA-style) training

`foundry/training/int4.py` implements 4-bit training without `bitsandbytes`:

- group-wise symmetric absmax quantization (group 64), two nibbles per byte,
- frozen `Int4Linear` base layers, dequantized per layer in the forward pass,
- LoRA adapters in fp32, `lora_B` initialised to zero so the delta starts at exactly zero,
- Conv1D and Linear architectures both supported,
- custom adapter artifact (`lora_int4.pt`) plus a manifest.

Measured on Qwen3-0.6B with the user's cleaned logs: 196 layers quantized, 2.52 M LoRA
parameters, loss 5.7211 → 4.8595, eval 4.2595 → 4.1792, 13 s for two steps, peak RSS
4.86 GB (the fp32 load peak).

```bash
foundry train int4 --dataset ds_xxx --model E:/models/Qwen3-0.6B --steps 10 --seq 128
```

**Limit:** loading still materialises the model in fp32, so a 4B model needs ~17 GB at
load time before the INT4 saving applies to training. A streaming INT4 loader is the
documented next step.
