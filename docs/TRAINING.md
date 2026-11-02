# Training

## The unified training contract (v8)

One declarative spec drives every method, and one manifest shape comes back.

```yaml
# spec.yaml
spec_version: 1
method: int4            # sft | int4 | dpo | distill | rft | onpolicy_distill | grpo
backend: int4_store     # fp32_lora | int4_stream | int4_store | remote
profile: local-smoke    # ci-tiny | local-smoke | local-real | remote
model:
  path: "E:/models/Qwen3-0.6B"
data:
  dataset_id: ds_xxx
  min_rows: 2
hyper:                  # every value explicit, no hidden global
  lr: 0.0002
  lora_r: 4
  lora_alpha: 16
  micro_batch: 1
  grad_accum: 1
  seq_len: 128
  max_steps: 2
  seed: 42
budget:
  max_wall_s: 1800
```

```bash
foundry train plan   --spec spec.yaml                  # validate + preflight, no training
foundry train run    --spec spec.yaml                  # execute, writes a RunManifest v2
foundry train doctor --model <dir> --dataset <id>      # run the conformance matrix
```

Why this exists:

- **Typos fail loudly.** `extra="forbid"` on the spec and every sub-model; a misspelled
  `learningrate` raises `ValidationError` instead of silently using a default.
- **One manifest.** `RunManifest` v2 carries the spec hash, environment, losses, memory, tuning
  and `extras`; `from_legacy()` maps the pre-v8 shapes (sft/int4/dpo/distill) so old run
  directories stay readable and nothing is dropped.
- **One conformance matrix.** Every registered trainer is checked the same way: honest preflight,
  a real parameter update (`max |LoRA-B| > 0`), a valid manifest, untouched base weights. A check
  that cannot apply reports `not_applicable` with a reason instead of pretending.
- **Profiles, not copies.** `ci-tiny` (random tiny model, 24 tokens, 1–2 steps) proves mechanics in
  CI; `local-smoke` proves integration with a real tokenizer; `local-real` is the quality run.

The legacy commands (`train sft|int4|dpo|distill|auto|unified`) still work unchanged — they are
the implementation the adapters call.

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

Checkpoints are **weights-only** (`save_only_model`): no optimizer, scheduler or RNG state is written, because no trainer has a resume path that would read it. The save interval is floored at 10 steps, and every run records `extras.disk_bytes_written`. See `docs/DISK_LIGHT.md`.

## RFT — rejection sampling over verified tasks (v7.5)

`method: rft` samples the base model on **train-split** generated tasks, keeps only answers that a
deterministic verifier accepts, and trains SFT on the accepted pairs:

```yaml
method: rft
backend: fp32_lora
method_args:
  tasks_path: data/exports/tasks.jsonl   # foundry tasks generate ...
  samples_per_task: 1
  min_accepted: 4
  max_candidates: 128
  seed: 0
```

What makes it trustworthy:

- **No test template is ever sampled.** The task file must have isolated splits; a template that
  appears in two splits refuses the run before a model is loaded (`TrainingError`).
- **Zero accepted candidates is a failure**, not an empty success: the run writes
  `manifest.json` with `status: failed` and raises.
- The accepted pairs become a frozen, synthetic dataset (`provider: synthetic`, rights confirmed)
  that the existing SFT path trains on — the counts, seed, generator/verifier versions, dataset hash
  and rejection reasons are all in the manifest extras.
- The candidate budget is a preflight blocker, so `plan` shows the work before any sampling.

Tested in `server/tests/train/test_rft.py`: the matrix passes with accepted samples, a zero-accept
run fails, a plausible wrong answer is rejected, leaked templates are refused, and the budget is
enforced.

## What is deliberately absent

- DPO/GRPO: no preference-pair pipeline is shipped yet, so the method is blocked with a precise reason.
- Full fine-tuning: not offered on shared-memory hardware.
- Distributed training.
- Automatic promotion of a trained adapter: it must pass the same benchmark and upgrade gates as any other candidate.
- **No measured quality claim for RFT.** The adapter machinery is tested; whether it improves a model
  is exactly what V8-S5 has to measure.

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

## Streaming INT4 and the disk-backed store (v4 / v4.1)

Three loading strategies, chosen so that a run is only started when it can finish:

| Mode | Anonymous RAM | Speed | When |
|---|---|---|---|
| `fp32` | full fp32 model (18 GiB for 4B) | fastest to load | small models |
| `streaming` | INT4 weights + fp32 leftovers | medium | fits the envelope |
| `offload` | LoRA + one dequantized layer + activations | slowest first pass | does not fit |

`streaming` builds the model on the `meta` device and quantizes each target linear while
reading the shard, so the full fp32 checkpoint is never resident.

`offload` writes the quantized model to disk once and memory-maps it:

```
int4-store/
  weights.u8    packed 4-bit nibbles for every target linear
  scales.f32    one fp32 scale per 64-value group
  rest.f32      embeddings, norms and the remaining frozen tensors in fp32
  meta.json     byte offsets, shapes, checkpoint identity
```

The mapped pages are file-backed, so Windows can drop them instead of swapping, and the
page cache works as the RAM buffer: hot layers stay resident, cold ones are read again.
`train int4` accepts `--offload` / `--no-offload`; without a flag the trainer decides from
its own estimate, and `train unified` picks it when fp32 and in-RAM INT4 both do not fit.

Every manifest records the **measured** peak RSS (`memory.peak_rss_mb`) next to the
estimate, plus `streaming`, `offload` and the skipped checkpoint subtrees.

```bash
# automatic: uses the store when the RAM estimate does not fit
foundry train int4 --dataset ds_xxx --model <path> --steps 2 --seq 128 --streaming

# forced
foundry train int4 --dataset ds_xxx --model <path> --steps 2 --seq 128 --streaming --offload
```

The measured 4B run is in [`docs/evidence/v41_int4_4b.md`](evidence/v41_int4_4b.md).

### Tiered cache: "virtual RAM on disk" (v6, opt-in)

`foundry/training/vram.py` can keep dequantized blocks in a bounded RAM tier (LRU) and spill them
to a memory-mapped arena file (`data/cache/vram/arena.bin`, grown lazily in 64 MB steps). It is
**off by default**, because it was measured to be slower here:

| Qwen3-0.6B, 2 steps, seq 128 | Duration | Peak RSS |
|---|---|---|
| cache off | **20.7 s** | 2937 MB |
| `--vram-gb 3` (RAM tier) | 117.1 s | 4536 MB |
| arena + prefetch | 63.5 s | 4318 MB |

Keeping fp32 blocks resident pushes the mapped INT4 store out of the page cache, so a cache hit
costs a store re-read. Use it only if a block miss is much more expensive than a memcpy on your
machine:

```bash
foundry train int4 … --vram-gb 4          # explicit opt-in
foundry train int4 … --vram-mode off      # explicit off (default)
foundry vram status                        # arena size + last run's cache stats
foundry vram clear                         # it is only a cache
```

Every manifest records `vram.stats` when the cache ran, and `null` when it did not.


## Turning an INT4 adapter into a runnable model (v5)

An INT4 adapter is not a PEFT directory, so it cannot be handed to `llama.cpp` directly.
`foundry train merge-int4` rebuilds the weights the forward pass actually used:

```text
W_merged = dequantize_int4(packed, scales) + (lora_B @ lora_A) * (alpha / r)
```

- one layer at a time, streamed into fp16 shards (≤ 1.5 GB) plus `model.safetensors.index.json`
- non-target tensors (vision tower, MTP head, embeddings) are copied from the original
  checkpoint, so the output is a complete HF directory
- `config.json` is written with `dtype: float16`, tokenizer files are copied
- the store fingerprint and target set must match the adapter, otherwise it refuses

```bash
foundry train int4 --dataset ds_xxx --model <hf dir> --steps 2 --streaming --offload
foundry train merge-int4 --adapter data/training-runs/<run> --out data/training-runs/<run>/merged
foundry gguf export data/training-runs/<run>/merged --out data/gguf/model-f16.gguf
foundry gguf quantize data/gguf/model-f16.gguf --method Q4_K_M --out data/gguf/model-Q4_K_M.gguf
```

`train unified` runs that chain itself for INT4 runs; the Training Lab shows a
"Merge → GGUF" action per completed INT4 run and the runs table lists the load mode and the
measured peak memory. The 0.6B proof is in [`docs/evidence/v5_loop.md`](evidence/v5_loop.md).

