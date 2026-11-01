# v3 evidence — training algorithms, OpenEval, playbook, logs

## Real auto run on the user's cleaned logs

Dataset `ds_d642c29dc9bf42a984ce` — imported from the machine's own agent sessions:

| Source | Files | Imported |
|---|---|---|
| Claude Code | 6 | records |
| Codex CLI | 6 (zstd) | records |
| Antigravity | 2 transcripts | records |
| OpenCode | 1 (active, skipped) | 0 |
| **Total** | 15 | **72 records** (49 train / 19 validation / 4 test) |

Originals were read only. Codex `.zst` files were decompressed into the import area;
nothing was deleted or modified.

Model: `Qwen3-0.6B`, LoRA r=8, sequence 128, 6 steps, OpenEval every 2 steps.

```
train loss     4.142238 → 4.162251   (noisy, expected with 6 steps)
eval loss      4.344190 → 4.261183 → 4.233414   (monotonic improvement)
best eval      4.233414 at step 6
checkpoints    best_adapter saved at each improvement
playbook       entry recorded (signature 10c3ca9e8d12be3c, improved=true)
```

## Algorithms

| Algorithm | Test | Proof |
|---|---|---|
| SFT | `test_sft.py` | 2 steps, adapter saved |
| Auto SFT (OpenEval + controller + playbook) | `test_algorithms.py` | eval history + playbook entry |
| Distillation | `test_algorithms.py` | 2 teacher answers added, SFT runs |
| DPO (synthetic pairs, adapter-disabled reference) | `test_algorithms.py` | 2 steps, finite loss, adapter saved |

## Why the 4B run is refused

`empero-ai/Qwen3.8-4B-Distill` ships 8.68 GB of BF16 weights. The trainer loads fp32,
which is ≈ 17.4 GB before optimizer state and activations. Visible RAM is 19.8 GB with
~5.6 GB free. `estimate_resources` returns `fits: false` and the API/CLI refuse the job
with those numbers instead of starting something that would swap or crash.

A GPU recipe (`3b-gpu`) and the exact CLI command are documented in `docs/TRAINING.md`.
