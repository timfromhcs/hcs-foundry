# v4.1 evidence — 4B streaming INT4 training through the disk-backed store

The v4.0 release left one item open: "4B streaming INT4 run — the download did not finish".
The checkpoint is now complete on disk, so the run was executed for real. It needed one
additional mechanism: **the disk-backed INT4 store**, because a 4B model does not fit in RAM
in any resident form on this machine.

## Model and dataset

| Item | Value |
|---|---|
| Checkpoint | `empero-ai/Qwen3.8-4B-Distill` |
| Weights | `model.safetensors`, 9,319,821,272 bytes (8.68 GiB, bf16) |
| fp32 footprint | 18.0 GiB (measured from the instantiated architecture) |
| Architecture | `Qwen3_5ForConditionalGeneration` → text-only `Qwen3_5ForCausalLM`: 32 layers, 24 linear-attention + 8 full-attention, hidden 2560, vocab 248,320, tied embeddings |
| Checkpoint layout | `model.language_model.*`, `model.visual.*` (297 tensors), `mtp.*` (15 tensors) |
| Dataset | `ds_e639a804ca5a414d9b12`, 53 training records, hash `4c28a267674e0531…` |
| Host | Ryzen 7 7735HS, Radeon 680M (unified), 19.8 GiB RAM visible, ~11.3 GiB available at start |

## Step 1 — build the store (once per checkpoint)

```
foundry train int4 --streaming --offload …      # builds int4-store/ on first use
```

| Measurement | Result |
|---|---|
| Build time | 700 s |
| Quantized tensors | 200 (q/k/v/o, gate/up/down, in_proj_qkv, in_proj_z, out_proj) |
| fp32 tensors kept | 226 (embeddings, norms, small projections) |
| Skipped subtrees | `model.visual` 297, `mtp.*` 15 — recorded in `meta.json`, not silently dropped |
| `weights.u8` | 1.78 GB (packed 4-bit nibbles) |
| `scales.f32` | 222 MB (one fp32 scale per 64-value group) |
| `rest.f32` | 2.56 GB (fp32 leftovers) |
| Store total on disk | 4.57 GB (vs 8.68 GiB bf16 checkpoint, 18.0 GiB fp32) |
| Peak RSS while building | 8,876 MB — the build is I/O + one tensor at a time, but a 2.4 GiB fp32 embedding still passes through |

## Step 2 — train through the mapped store

```
foundry train int4 --dataset ds_e639a804ca5a414d9b12 \
  --model data/models/empero-ai/Qwen3.8-4B-Distill \
  --steps 2 --seq 128 --accum 1 --rank 4 --lr 1e-3 --streaming --offload
```

| Measurement | Result |
|---|---|
| Mode | `offload(disk)` — store reused (`reused: true`, fingerprint `d6f954fda9ba462f`) |
| Quantized layers | 200 |
| LoRA parameters | 7,618,560 across the 10 target suffixes |
| Steps | 2, sequence 128, micro-batch 1, rank 4, lr 1e-3 |
| Loss | **4.453481 → 3.545403** |
| Duration | 51.1 s warm page cache (151.4 s on the first, cold pass) |
| Peak RSS | 9,450.7 MB — **includes 4.57 GB of file-backed mapped pages** |
| Peak private (anonymous) | 6,343.5 MB |
| Lowest available system RAM | 1,410 MB |
| Adapter | `data/training-runs/v41-int4-4b/adapter/lora_int4.pt` |
| Manifest | `data/training-runs/v41-int4-4b/manifest.json` |

Phase-level measurement (`seq 16`, one forward/backward, warm cache):

```
start                     rss=  209 MB  private=  671 MB
meta model                rss=  476 MB  private= 1588 MB
store mapped              rss=  476 MB  private= 1597 MB    <- mapping costs nothing
attached (226 tensors)    rss=  510 MB  private= 1600 MB    <- zero-copy, verified by data_ptr
forward 9s                rss= 6186 MB  private= 2880 MB    <- file pages paged in
backward 9s               rss= 5573 MB  private= 2224 MB
```

## What this proves, and what it does not

**Proven:** the pipeline runs end to end at 4B scale on this machine — checkpoint with a
vision-language layout → key normalisation → INT4 store → memory-mapped load → LoRA
optimisation with a decreasing loss → adapter artifact. No fp32 model was ever resident, and
the process never needed the 18.0 GiB that a plain load would require.

**Not proven:** model quality. Two steps on 53 records move the loss; they do not improve the
model. A useful 4B run needs many hours of CPU here (projected 30–60 s per step warm, more
with cold cache, and tens of thousands of steps for a real SFT).

**Honest caveat on memory:** the mapped store counts toward the working set, and Windows
counts the read-only section in the process working set while it is resident. The saving is
therefore not "18 GiB → 1 GiB"; it is "18 GiB resident + fp32 temporaries → 4.6 GiB
reclaimable file pages + ~6.3 GiB private peak". The machine survived at 1.4 GiB free; a
smaller machine should lower `--seq` or close other applications first.

## Why the training loader needed a recompute in backward

Dequantising inside a normal forward pass retains every fp32 weight matrix until backward
finishes — 14 GiB for 3.56 B quantized parameters, which is what the first attempt did
(observed: an abandoned run climbing past 10 GiB private). `_Int4Matmul` saves only
`packed`/`scales` and dequantises again during backward, so peak memory stays at roughly one
layer. `server/tests/test_offload.py::test_forward_and_backward_recompute_instead_of_retaining_fp32_weights`
asserts that the dequantisation runs twice per layer.

## Reproduce

```bash
# 1. pull the checkpoint (public, no token needed)
foundry hf pull empero-ai/Qwen3.8-4B-Distill --file model.safetensors --file tokenizer.json \
  --file tokenizer_config.json --file vocab.json --file merges.txt --file config.json

# 2. build the store (once) and train
foundry train int4 --dataset <frozen-dataset> --model <path> \
  --steps 2 --seq 128 --rank 4 --lr 1e-3 --streaming --offload
```
