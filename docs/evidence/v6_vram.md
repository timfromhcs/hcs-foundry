# v6 evidence — the tiered "virtual RAM" cache, measured

Goal: *more virtual RAM on HDD, used smartly* for 2–4B training. This file records what was
built and what the numbers said. The short version: **the mechanism works, the measurement says
"do not use it by default on this machine".**

## What was built

`foundry/training/vram.py` — three tiers behind one `get(module, start, stop)`:

| Tier | Storage | Cost |
|---|---|---|
| hot | anonymous RAM, LRU with a hard byte budget | free on hit |
| warm | memory-mapped arena file (`data/cache/vram/arena.bin`), grown in 64 MB steps | one memcpy in/out |
| cold | the INT4 store, dequantized on demand | read 0.25× + dequant math |

Wired into `Int4Linear` (forward and backward go through the same cache), `train int4
--vram-gb/--vram-mode`, `foundry vram status|clear`, `GET /training/vram`.
Every manifest records `vram.stats`.

## Unit level (all green)

```text
server/tests/test_vram.py  8 passed
  hits are bit-identical to a fresh dequantization
  the hot budget is never exceeded (evictions go to the arena)
  the arena wraps instead of growing past its limit
  prefetch warms blocks and they are not dequantized twice
  budget 0 disables the cache (falls back to dequantizing)
  the arena index is written and `clear` removes it
```

## Measurement 1 — per-block cost

`python scripts/vram_bench.py` style micro benchmark, 256-row blocks, fp32:

| shape | dequantize (miss) | RAM hit | empty-arena read |
|---|---|---|---|
| 1024×1024 | 0.80 ms | 5.7 µs | — |
| 1024×2560 | 1.05 ms | 4.7 µs | — |
| 9216×2560 | 1.12 ms | 3.6 µs | — |

Dequantizing one block is ~1 ms. That is cheap, which is exactly why caching has little to win.

## Measurement 2 — the same run with and without the cache

Qwen3-0.6B, converged store on disk, 2 steps, seq 128, dataset `ds_e639a804…`:

| Mode | Duration (training only) | Peak RSS | Cache stats |
|---|---|---|---|
| `--vram-mode off` | **20.72 s** | 2937 MB | — |
| `--vram-gb 3` (RAM tier only) | 117.08 s | 4536 MB | 9344 RAM hits, 1344 misses, 0 evictions, 9.45 s dequant |
| default (`auto`, arena + prefetch) | 63.50 s | 4318 MB | 87 % hit rate, 4.1 GB arena writes, 3.6 GB reads |

Cause: keeping fp32 blocks resident **inflates the working set**, which pushes the mapped INT4
store out of the page cache — so the "hit" saves 1 ms of dequantization but costs a re-read of
the store from disk. Prefetching the immediately-next block doubles the work instead of hiding
it (the main thread needs the block in the same loop iteration).

## Consequence in the product

- `resolve_budget()` returns **0 by default**: the cache runs only when a size is requested
  explicitly (`--vram-gb N`), and `--vram-mode off` is the documented default.
- Prefetching is opt-in (`FOUNDRY_VRAM_PREFETCH=1`).
- The manifest still records `vram` when it is enabled, so an experiment is reproducible.
- The real memory levers remain: the mmap'd INT4 store, block-wise dequantization (transient
  279 MB → 0 MB, `docs/evidence/v5_loop.md`) and the file-backed fp32 leftovers.

## When this would change

The cache pays off only if a block miss is much more expensive than a memcpy — for example a
disk-backed store on a slow medium with a fast CPU, or a machine whose page cache cannot hold
the store. The knob exists for that; it is not a default.
