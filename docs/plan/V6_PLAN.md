# HCS Foundry v6 — Long-term plan: virtual RAM on disk, 2B–4B locally

**Written before implementation.** Baseline: `b59427c` (v5.0.0), CI green, 195 backend /
7 frontend / 9 Playwright tests green.
Requested outcome: *"2–4B trainable, more virtual RAM on HDD with smart use, then 4B, then
push loop."*

---

## 1. The actual bottleneck

v5 can *load* an 18 GiB fp32 model on a 19.8 GiB machine, but a run still lives and dies by
whatever is in anonymous RAM and by the OS page cache. Three numbers from earlier evidence:

| Observation | Consequence |
|---|---|
| 4B store = 4.57 GB mapped, peak RSS 9.45 GB, private 6.34 GB | the mapped weights plus the allocator are the ceiling, not the compute |
| one dequantized 9216×2560 block = 90 MB fp32 | with block 256 the transient is ~0 MB, so blocking fixed the *spikes* |
| every step re-dequantizes all 200 layers | the same math is repeated per step; there is no reuse across steps |

So there is no *managed* memory tier: the OS decides what stays resident, and repeated work is
recomputed. That is what "more virtual RAM on HDD, smart use" has to fix.

## 2. Design: a three-tier weight cache (`foundry/training/vram.py`)

```
tier 0  hot   anonymous RAM, LRU, budget = f(available RAM) or --vram-gb
tier 1  warm  mmap'd arena file on disk (data/cache/vram/arena.bin), memcpy in/out
tier 2  cold  the INT4 store itself (packed nibbles + scales), dequantize on demand
```

A lookup for `(module, block)` walks hot → warm → cold and promotes on hit. Writes are
write-through into the warm tier so a later step can reuse a block without recomputing the
dequantization. A background prefetcher dequantizes the *next* block while the current matmul
runs.

Why this is more than a cache:

- **Bounded**: the hot budget is a hard cap; the arena is a sparse file with a fixed size, so a
  run cannot fill the disk silently.
- **Adaptive**: with no explicit budget the manager samples available RAM each step and grows or
  shrinks the hot tier (`FOUNDRY_VRAM_MODE=auto`), so a busy desktop degrades instead of dying.
- **Honest**: every run reports `vram.stats` (RAM hits, arena hits, misses, dequant seconds,
  arena bytes written/read, prefetch hits) in the manifest. If the cache does not help, the
  numbers say so.

### Safety rules (non-negotiable)

- The arena never exceeds its configured size; allocation fails closed (falls back to
  dequantizing) instead of raising.
- No eviction of *live* tensors: the cache owns copies only, the model keeps its own buffers.
- `foundry vram clear` removes the arena; it is a cache, never a source of truth.
- Values are verified by a unit test: cache hits must be bit-identical to a fresh dequantization.

## 3. Work packages

### V1 — `foundry/training/vram.py`

- [ ] `BlockKey(module_path, start, stop)`, `BlockCache(budget_bytes, arena_dir, block_rows)`
- [ ] hot tier: `OrderedDict` LRU with a byte budget and promotion on hit
- [ ] warm tier: sparse `arena.bin` + `arena.json` index, `put`/`get` via `torch.frombuffer`
- [ ] cold tier: callback that dequantizes a block on demand
- [ ] prefetcher: one daemon thread, bounded queue, cancellation-aware, never raises
- [ ] `stats()` + `close()`; `budget_from_available()` heuristic for `auto`
- [ ] tests: hit/miss/eviction, arena round-trip equality, budget cap, prefetch correctness,
      closed-cache fallback, stats counters

### V2 — wire it into the INT4 path

- [ ] `Int4Linear` takes an optional shared cache; `blocked_linear`/`blocked_grad_input` look up
      blocks through it
- [ ] `train_int4` creates one cache per run, passes it to the store loader, records
      `manifest["vram"]`, closes it in `finally`
- [ ] knobs: `--vram-gb`, `FOUNDRY_VRAM_GB`, `FOUNDRY_VRAM_MODE`, `FOUNDRY_INT4_BLOCK`
- [ ] CLI `foundry vram status|clear`; API `GET /training/vram`
- [ ] tests: 0.6B-equivalent tiny run produces identical loss with and without the cache

### V3 — activation checkpointing for ≥2B

- [ ] wrap decoder layers in `torch.utils.checkpoint` when the budget is tight
      (`--checkpointing auto|on|off`), recorded in the manifest
- [ ] test: gradients equal to non-checkpointed within 1e-5 on the tiny model

**Status: cancelled with a measurement.** At `seq 128` the activations of the 4B model are
`1 × 128 × 2560 × 32 × 2 × 4 ≈ 84 MB`, while the run's private peak is gigabytes. Checkpointing
would trade compute for a fraction of a percent of memory. The plan promised it "for ≥2B"; the
measurement says it is not the lever, so it is not built and not claimed.

### V4 — 2B class end-to-end

- [ ] pull `Qwen/Qwen3-1.7B` (public, ~3.4 GB bf16) into `data/models`
- [ ] store build → INT4 training → merge → GGUF → quantize → answer
- [ ] evidence: `docs/evidence/v6_2b.md`

### V5 — 4B with the virtual RAM tier

- [ ] stop only my own helper processes, re-measure free RAM, run the store-backed 4B training
- [ ] merge to fp16, attempt GGUF export (converter support for Qwen3.5-VL is unverified —
      if it refuses, that is the recorded result, not a workaround with fake output)
- [ ] evidence: `docs/evidence/v6_4b.md` with cache stats and honest gaps

### V6 — Release v6.0.0

- [ ] docs: README, CHANGELOG, TRAINING, KNOWN_LIMITATIONS, plan progress, release report
- [ ] version bump everywhere, EXE rebuild, GitHub release, site update
- [ ] full gates green; **intermediate pushes after every package** (CI must stay green)

## 4. Order and push loop

```text
V1 (+tests)      → push   → V2 (+tests) → push   → V3 (+tests) → push
V4 (2B chain)    → push   → V5 (4B chain) → push → V6 release  → push
```

Each push: `pytest`, `vitest`, `npm run build`, `playwright`, then commit + tag-free push; CI
must be green before the next package starts. If a gate fails, fix forward (max 3 attempts),
then document the blocker in this file and continue with the next independent package.

## 5. Acceptance criteria

| # | Criterion | How it is proven |
|---|---|---|
| 1 | Cache hits are bit-identical to fresh dequantization | unit test |
| 2 | The hot budget is never exceeded | unit test with a tiny budget |
| 3 | A 0.6B run with the cache produces the same loss as without | two runs, same seed, loss compared |
| 4 | The 4B run completes a step with the cache and reports `vram.stats` | manifest + evidence file |
| 5 | A 2B model runs the full chain to a GGUF that answers | `models run` transcript |
| 6 | No release claims a measurement that the manifest cannot back | review of evidence files |

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Host memory pressure (observed: free RAM 1.4–3.5 GB, commit 87 %) | hot budget from live available RAM; stop my own helper services before heavy runs; document the environment in the evidence |
| Converter cannot export Qwen3.5-VL | record `not_run` with the exact error; 2B (Qwen3) is the export proof |
| Arena on a slow/removable disk | budget check + throughput measured, not assumed |
| Cache makes runs slower (memcpy vs dequant) | prefetch + stats; if it does not help, the default stays off and the evidence says why |

## 7. Out of scope for this horizon

- Multi-GPU, ROCm/Vulkan training (no backend exists)
- Official benchmark assets
- Quality training at 2–4B (the machine cannot; the pipeline is the deliverable)
- Cloud offload of anything

## 8. Progress log

| WP | Status | Evidence |
|---|---|---|
| V1 cache module | **done** | `training/vram.py` + 8 tests (hits, eviction, arena wrap, prefetch, budget cap) |
| V2 INT4 wiring | **in progress** | cache in `Int4Linear`, `--vram-gb/--vram-mode`, `vram status/clear`, `GET /training/vram`; needs a real run |
| V3 checkpointing | open | |
| V4 2B chain | open | |
| V5 4B chain | open | |
| V6 release | open | |
