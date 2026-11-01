# Release report — HCS Foundry v4.1.0

**Date:** 2026-11-01
**Branch:** `main`
**Tag:** `v4.1.0`
**Baseline:** `b7230d9` (v4.0.0 + landing site)

This release adds no product surface. It closes the five gaps v4.0 left open, so that
"v4 is done" is a measured statement.

## Delivered

| Gap from v4.0 | Result in v4.1 |
|---|---|
| 4B streaming INT4 run never completed | **Completed** through a new disk-backed INT4 store: 200 layers, loss 4.453 → 3.545, 51 s warm, adapter written (`docs/evidence/v41_int4_4b.md`) |
| No measured memory in the manifest | `memory.peak_rss_mb`, `peak_private_mb`, `min_available_mb`, `samples` in every INT4 manifest (`foundry/training/memory.py`) |
| Streaming manifests claimed an fp32 load peak | Note is now conditional; unit test asserts the streaming wording |
| `KNOWN_LIMITATIONS.md` stuck at 0.1.0 | Rewritten for 4.1 (28 entries, current behaviour) |
| Landing site had no stable URL | Vercel project `hcs-foundry-site`, production <https://hcs-foundry-site.vercel.app>, `.vercelignore` limited to `site/` |

## New mechanism: disk-backed INT4 store

```
int4-store/  weights.u8 (1.78 GB) · scales.f32 (222 MB) · rest.f32 (2.56 GB) · meta.json
```

- built once per checkpoint, reused by identity + requested targets,
- loaded with a **read-only** `mmap` (`ACCESS_READ`, so Windows reserves no copy-on-write
  commit for 4.5 GB) and turned into tensors zero-copy — verified by `data_ptr` equality,
- vision tower (297 tensors) and MTP head (15 tensors) skipped and reported,
- `model.language_model.*` normalised onto the text-only module tree,
- a hard meta-device check still fails the load if a needed weight is missing,
- `_Int4Matmul` dequantizes again in backward, so fp32 weights are never retained.

Automatic selection: `train int4` without flags uses RAM when the estimate fits, the store
when it does not; `train unified` reports the decision in its first stage.

## Proofs

```
tests        178 backend, 7 frontend, 7 Playwright E2E (all green locally)
4B store     build 700 s, 4.57 GB, 200 quantized / 226 fp32 / 312 skipped tensors
4B training  2 steps, seq 128, rank 4 → loss 4.453481 → 3.545403 in 51.1 s
             peak RSS 9450.7 MB (4.57 GB file-backed), peak private 6343.5 MB,
             lowest available RAM 1410 MB, adapter 7,618,560 LoRA params
site         HTTP 200 · assets 200 · <title>HCS Foundry — local AI model lab</title>
             Permissions-Policy / Referrer-Policy present · static only, no functions
```

## Honest gaps that remain

- **4B/8B quality training** is still impossible here (18.0 GiB fp32 vs ~11 GiB available);
  the 4B run proves the mechanism, not the model.
- **Official SWE-bench** needs the dataset and repository checkouts; without them the suite
  reports `not_run` with the reason.
- **Vulkan training** remains impossible (no PyTorch Vulkan backend).
- **Memory accounting nuance:** the mapped store counts toward the process working set while
  pages are resident, so the saving is "no 18 GiB resident fp32 model", not "1 GiB total".
- **Windows only** for the store: `mmap.ACCESS_READ` + `torch.frombuffer` is exercised there;
  Linux/macOS should work but was not tested in this release.

## Reproduce the gates

```bash
cd server && python -m pytest -q            # 178 passed
cd web && npx vitest run                    # 7 passed
cd web && npm run build                     # tsc + vite
cd web && npx playwright test               # 7 passed (service on :8770)
```
