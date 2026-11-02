# Release report — HCS Foundry v5.0.0

**Date:** 2026-11-02
**Branch:** `main`
**Tag:** `v5.0.0`
**Baseline:** `71f9f65` (v4.1.0)
**Plan:** `docs/plan/V5_PLAN.md` (written before implementation)

## Theme

v4 proved two mechanisms that ended in dead ends: a 4B model could be trained on disk but
never left the machine as a runnable artifact, and the deployed page was decoration. v5.0.0
closes both: **train → merge → GGUF → quantize → answer** now works end to end, and the
deployed page can pair with the local service.

## Delivered

| # | Request | Result |
|---|---|---|
| P1 | Memory-bounded INT4 matmul | `blocked_linear`/`blocked_grad_input`, block 256; transient on the 4B MLP shape **279 MB → 0 MB**, same speed |
| P2 | INT4 adapter → real model | `training/merge_int4.py`; CLI `train merge-int4`, API `POST /training/merge-int4`, pipeline stage `merge_int4`, unified trainer finishes the chain |
| P3 | Paired web demo | `/#pair` on the live site; `FOUNDRY_UI_ORIGINS` URL parsing fixed; CORS echoes the exact origin; pairing code enforced |
| P4 | UI surface | Training Lab: load mode, measured peak, "Merge → GGUF" (de/en/fr) |
| P5 | Release | version 5.0.0, docs, EXE, tag, GitHub release |

## Proofs

```
0.6B end-to-end
  train int4 --streaming --offload   196 layers, loss 5.414693 → 4.308523, peak RSS 2942 MB
  train merge-int4                   196 layers + LoRA → 1 shard, 1.41 GiB fp16, 29.8 s
  gguf export                        1439 MiB F16
  gguf quantize Q4_K_M               462 MiB, ratio 0.321, 13.8 s
  models run                         "Hallo! Sollte dir helfen, gerne." · 117.2 tok/s · TTFT 0.998 s

blocked dequant (4B MLP shape 9216x2560, forward+backward)
  full matrix   +279 MB transient   0.16 s
  block 1024     +28 MB transient   0.13 s
  block 256       +0 MB transient   0.12 s

tests        193 backend · 7 frontend · 9 Playwright E2E (all green locally)
site         https://hcs-foundry-site.vercel.app (200) · /#pair present · pairing section
             shows "No local service answered" instead of a fake success
```

## What is deliberately not claimed

- **No 4B export.** Merging + exporting the 4B vision-language checkpoint writes ~17 GB and the
  installed `convert_hf_to_gguf.py` support for Qwen3.5-VL is unverified. 0.6B is the proof.
- **No 4B memory re-measurement.** Three attempts stalled on host memory pressure (free RAM
  1.4–3.5 GB, commit 87 %, unrelated processes); no step completed. The isolated per-layer
  measurement carries the claim; the last complete 4B run is the v4.1 one.
- **Model quality** is unchanged and unclaimed: two optimisation steps on 53 records produce a
  valid artifact, not a better model.

## Reproduce

```bash
cd server && python -m pytest -q          # 193 passed
cd web && npx vitest run                  # 7 passed
cd web && npm run build && npx playwright test   # 9 passed (service on :8770)
```

## Publication

| Item | Value |
|---|---|
| Commit | see `git log -1` on `main` |
| Tag | `v5.0.0` |
| CI | green (backend, frontend, Playwright) |
| GitHub Release | <https://github.com/timfromhcs/hcs-foundry/releases/tag/v5.0.0> |
| Landing page | <https://hcs-foundry-site.vercel.app> · pair section <https://hcs-foundry-site.vercel.app/#pair> |
