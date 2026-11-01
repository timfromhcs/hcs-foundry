# HCS Foundry v4.1 — Plan (closing the v4 gaps)

**Written before implementation.** Baseline: `b7230d9` (v4.0.0 + landing site), CI green,
165 backend / 7 frontend tests green on the author's machine.

This release adds **no new product surface**. It closes the five open items the v4 release
left behind, so that "v4 is done" is a verified statement instead of an intention.

---

## 1. Where the previous session stopped

Recorded in `docs/evidence/RELEASE_REPORT_V4.md` and `CHANGELOG.md`:

| # | Open item | Evidence of the gap |
|---|---|---|
| 1 | 4B streaming INT4 run never completed | `CHANGELOG` "Blocked / pending": download did not finish |
| 2 | No measured peak memory in the training manifest | plan W1 checkbox "peak RSS + manifest field" is unchecked; `grep peak_rss` finds nothing |
| 3 | Manifest note lies for streaming runs | `data/training-runs/v4-int4-stream/manifest.json` says "Loading still needs the fp32 peak" although `streaming: true` |
| 4 | `KNOWN_LIMITATIONS.md` is stale | still titled `0.1.0`; claims "adapter → GGUF export is not implemented", which v3 shipped |
| 5 | Landing site has no stable URL | README links `https://agile-jasper-pcdb33c.tmp.now`; no Vercel project exists for `site/` |

Everything else from v4 (unified trainer, log cleanup, repo-fix / agentic / SWE-bench
adapters, Vulkan profile reuse) is implemented and verified; this plan does not touch it.

## 2. Requested outcome

> "Work until the next logical stable horizon."

The next stable horizon is **v4.1.0 = v4 with a real 4B proof, truthful artifacts, a stable
site URL, and all gates re-run.** A new feature version (v5) would be scope creep while the
last release still carries an unproven claim.

## 3. Work packages

### T1 — Measured memory in the training manifest

- [ ] `foundry/training/memory.py`: `PeakRSS` sampler (psutil RSS + minimum available RAM),
      thread-based, exception-safe
- [ ] `train_int4` wraps load + train + save with the sampler and writes
      `manifest["memory"] = {peak_rss_mb, start_rss_mb, min_available_mb, samples}` and
      `manifest["streaming"]`
- [ ] Streaming note corrected: streaming runs must not claim an fp32 load peak
- [ ] Tests: sampler records a rising peak; streaming manifest carries memory + `streaming`

### T2 — Real 4B streaming INT4 run

- [ ] Model `empero-ai/Qwen3.8-4B-Distill` (8.7 GB safetensors, already on disk)
- [ ] `train int4 --streaming --steps 2 --seq 128` on the frozen agent-log dataset
- [ ] Record peak RSS, min available RAM, loss, duration, adapter path
- [ ] If the run cannot fit or the architecture is unsupported: record the exact error and
      keep the "no 4B claim" wording. **No run is invented.**
- [ ] `docs/evidence/v41_int4_4b.md`

### T2b — Disk-backed INT4 store ("smart HDD offload / RAM buffer")

Requested after the probe showed the 4B checkpoint is a **vision-language** layout
(`model.language_model.*`, `model.visual.*`, `mtp.*`) whose fp32 form is 18.0 GiB on a
machine with ~6 GiB available. RAM alone cannot carry it; refusing would end the v4 claim.

- [ ] `foundry/training/offload.py`: build a store once per model
      (`weights.u8` packed nibbles, `scales.f32`, `rest.f32` for embeddings/norms,
      `meta.json` with byte offsets, shapes, model file sizes, sha256)
- [ ] Load the store with `mmap(ACCESS_COPY)` + `torch.frombuffer`, so packed weights are
      file-backed pages the OS can evict instead of anonymous RAM
- [ ] Dequantize one layer at a time in the forward pass (fp32 transient ≈ one matrix)
- [ ] Checkpoint key normalisation: `model.language_model.*` → `model.*`; unresolved keys
      (vision tower, MTP head) are skipped **and reported**; the meta-device check still
      fails the load if a needed weight is missing
- [ ] `train_int4(..., offload=None|True|False)`: `None` = automatic when the RAM estimate
      does not fit; manifest records `offload` + measured `memory`
- [ ] CLI `train int4 --offload/--no-offload`, API option, tests (store round-trip equals
      in-RAM INT4 within tolerance, 1-step training, reuse, honest manifest)
- [ ] Real 4B run through the store; evidence records disk bytes, page-cache behaviour and
      the fact that the vision tower was not used for a text-only run

### T3 — Truthful docs

- [ ] `KNOWN_LIMITATIONS.md` rewritten for 4.1 (hardware facts unchanged; remove the
      contradicted training/GGUF/benchmark items; add the real ones)
- [ ] `docs/plan/V4_PLAN.md` progress table updated with the final state
- [ ] `CHANGELOG.md` 4.1.0, `README.md` (version badge, site URL, benchmark table),
      `docs/evidence/RELEASE_REPORT_V4_1.md`
- [ ] `docs/TRAINING.md`: streaming path + measured memory numbers

### T4 — Stable site URL

- [ ] `vercel link` → project `hcs-foundry-site` in the existing scope
- [ ] `vercel deploy --prod` from `site/`
- [ ] Verify: HTTP 200, asset 200s, security headers, `cleanUrls`
- [ ] README website link points at the production URL
- [ ] `docs/evidence/vercel_v41.md`

### T5 — Gates

- [ ] `pytest` (server), `vitest` (web), `npm run build` (web), Playwright E2E
- [ ] Secret scan of the diff, no model weights or session data staged
- [ ] Commit, tag `v4.1.0`, push, CI green, tag visible on the remote

## 4. Loop protocol

```text
inspect → implement → unit test → real proof → docs → gates → commit → tag → push
```

Bounded repair: 3 attempts per failing test, then document the blocker and continue with the
independent items. Nothing is marked done without a command output that supports it.

## 5. Result

| WP | Status | Evidence |
|---|---|---|
| T1 Measured memory | **done** | `training/memory.py`; manifests carry `peak_rss_mb` / `peak_private_mb`; 5 new tests |
| T2 4B run | **done** | `docs/evidence/v41_int4_4b.md` — loss 4.453 → 3.545, 200 layers, 51 s |
| T2b Disk-backed store | **done** | `training/offload.py`; 4.57 GB store, zero-copy mmap verified by `data_ptr`, recompute-in-backward |
| T3 Docs | **done** | KNOWN_LIMITATIONS, TRAINING, README, CHANGELOG, V4_PLAN, release report |
| T4 Stable site | **done** | `docs/evidence/vercel_v41.md` — <https://hcs-foundry-site.vercel.app> |
| T5 Gates | **done** | 178 backend / 7 frontend / 7 E2E green; tag `v4.1.0`; CI green |

Two bounded repair loops happened for real reasons, both kept in the record:

1. the first 4B attempt copied every tensor to bf16 because the checkpoint declares bf16
   (`peak_private` 10.8 GB) → fp32 forced on the config, mapping became zero-copy;
2. the first successful 4B run retained dequantized fp32 weights in the autograd graph →
   replaced by a recompute-in-backward autograd function; `peak_private` fell from 10.8 GB
   (bf16 copy path) to 6.3 GB.

