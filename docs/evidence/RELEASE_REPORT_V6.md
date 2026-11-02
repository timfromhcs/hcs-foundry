# Release report — HCS Foundry v6.0.0

**Date:** 2026-11-02
**Branch:** `main`
**Tag:** `v6.0.0`
**Baseline:** `b59427c` (v5.0.0)
**Plan:** `docs/plan/V6_PLAN.md` (written before implementation)

## Theme

"2–4B locally, more virtual RAM on HDD with smart use." Both were addressed, and the second one
produced a negative result that is now shipped as a measured default instead of a marketing claim.

## Delivered

| # | Ask | Result |
|---|---|---|
| 1 | 2–4B trainable locally | **1.7B full chain** (train → merge → GGUF → answer) and **4B adapter → GGUF → answer**; see below |
| 2 | More virtual RAM on HDD, smart use | tiered cache (`hot RAM` → `mmap arena` → `INT4 store`) with budgets, stats and CLI/API; **measured slower here**, therefore off by default |
| 3 | Then 4B | 8.70 GiB fp16 merge, 8264 MiB F16 GGUF, 2655 MiB Q4_K_M, answers "Hallo Welt" at 8.75 tok/s |
| 4 | Push loop | five pushes during the work, CI green after each |

## Proofs

```
1.7B (Qwen3-1.7B) — full chain
  train int4 --streaming --offload   196 layers · loss 5.023045 → 4.080684 · peak RSS 4496 MB
  train merge-int4                   3 shards · 3.80 GiB fp16 · 179 s
  gguf export                        3881 MiB F16
  gguf quantize Q4_K_M               1223 MiB · ratio 0.315 · 60.3 s
  models run                         "Hallo Welt" · 51.4 tok/s · TTFT 5.06 s

4B (Qwen3.8-4B-Distill, from the v4.1 adapter)
  train merge-int4                   200 layers + LoRA · 6 shards · 8.70 GiB · 514 s
  gguf export                        8264 MiB F16 · ~8 min
  gguf quantize Q4_K_M               2655 MiB · ratio 0.321 · 219 s
  models run                         "Hallo Welt" · 8.75 tok/s · TTFT 4.33 s

tiered cache (Qwen3-0.6B, 2 steps, same store)
  cache off                          20.72 s · peak RSS 2937 MB
  --vram-gb 3 (RAM tier)            117.08 s · peak RSS 4536 MB · 9344 hits / 1344 misses
  arena + prefetch                   63.50 s · peak RSS 4318 MB · 87 % hit rate
  → keeping fp32 blocks resident pushes the mapped store out of the page cache;
    a hit saves ~1 ms of dequantization and costs a store re-read. Default: off.

tests        204 backend · 7 frontend · 9 Playwright E2E
```

## Honest gaps

- **4B training was not re-run today.** Two attempts stalled in page faults at ~3.6 GB free /
  89 % commit (other agent processes) and no step completed; the last complete 4B training run
  remains the v4.1 one (loss 4.453 → 3.545). Nothing is claimed for the stalled runs.
- **Activation checkpointing was planned and cancelled** with a measurement: 4B activations at
  seq 128 are ~84 MB, not the bottleneck.
- The tiered cache is a **capability, not a default**; the numbers above are why.
- Quality is still not the deliverable: all runs are two steps on 53 records.

## Reproduce

```bash
cd server && python -m pytest -q                     # 204 passed
cd web && npx vitest run                             # 7 passed
cd web && npm run build && npx playwright test       # 9 passed (service on :8770)
```

## Publication

| Item | Value |
|---|---|
| Tag | `v6.0.0` → commit `e574895` |
| CI | run `41424116570` — green (backend, frontend, Playwright) |
| GitHub Release | <https://github.com/timfromhcs/hcs-foundry/releases/tag/v6.0.0> (Latest) |
| Windows EXE | `HCSFoundry.exe`, 384,063,077 bytes, built from this commit |
| Landing page | <https://hcs-foundry-site.vercel.app> |
