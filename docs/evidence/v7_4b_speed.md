# v7 evidence — why the 4B is slow here, and what actually helps

**Research first, then code.** All numbers were produced on this machine with
`scripts/store_io_bench.py` and real `train int4` runs against the v4.1 4B store
(`E:\AILOCALTRAIN\data\training-runs\v41-int4-4b\int4-store`, 4.57 GB, external 466 GB disk).

## 1. Raw read throughput of the store

| Access pattern | Throughput |
|---|---|
| sequential 4 MiB reads (the prefetcher) | **105 MB/s** |
| 8 MiB sequential reads, warm file | 2 660 MB/s |
| 4 KiB page-fault reads of `rest.f32`, cold | **66.6 MB/s** |
| 4 KiB page-fault reads of `weights.u8`, warm | 2 035 MB/s |

The disk itself delivers ~105 MB/s sequentially. Everything slower is a *cache miss*: once a
file is resident it is memory speed; once it is evicted, a page fault costs what the disk costs.

## 2. What one 4B step costs

| Run | Steps / seq | Duration | Notes |
|---|---|---|---|
| v4.1 (2026-11-01, machine idle, ~11 GB free) | 2 / 128 | **51 s** | store + leftovers fit in the page cache |
| v7 probe (store on the external disk, ~3.9 GB free) | 1 / 64 | **740 s** | `faulthandler` shows the time inside `dequantize_int4` — page faults, not compute |
| v7 with prefetch (4.36 GB read in 41.5 s) | 2 / 128 | **756 s** | prefetch completes long before the step ends and is evicted again |

So the same code and the same store differ by **15×** purely by how much of the working set is
resident. Peak RSS was 9.08 GB against ~4 GB free: the store (4.57 GB) plus `rest.f32` (2.39 GB)
cannot be held, and each pass re-reads several GB at 66–105 MB/s.

## 3. What ships because of this

- **Prefetch guard**: the background warm-up only runs when the store is smaller than 60 % of
  available RAM. Warming 4.36 GB that is immediately evicted is wasted disk time; the guard
  records `prefetch: {"enabled": false, "reason": ...}` in the manifest instead.
- **Adaptive block size** (`choose_block_rows`): ~64 MB of fp32 per block, clamped at 256 rows.
  It bounds the transient (see v5 evidence: 279 MB → 0 MB) but cannot change I/O volume.
- **`manifest["tuning"]`**: block rows, hidden size, available bytes and torch threads per run, so
  a slow run can be explained instead of guessed.
- **`foundry run doctor`** warns when the INT4 stores outgrow the RAM and points at
  `--seq 64 --steps 1` or a smaller model.

## 4. What the research says would help next (not shipped)

1. **Sequential read path**: 2 660 MB/s warm vs 66 MB/s faulted on the same file — a batched
   reader per layer would remove the fault cost. Windows has no `os.pread`, so it needs
   `seek`+`read` on an unbuffered handle; the code path exists in `store_prefetch.py` and can be
   reused. Deferred to keep this release honest about what was measured on a *cold* cache.
2. **bf16 leftovers**: `rest.f32` is 2.39 GiB and is touched by every forward pass. Storing it as
   bf16 halves that to 1.20 GiB — but the embedding must then stay bf16 inside the model, or the
   attach step copies it into RAM and defeats the purpose. That is a model-loading change, not a
   store-format change.
3. **A store that fits**: with ~11 GB free the 4B runs at 25 s/step. This is the cheapest "fix"
   and it is a scheduling decision, not code: close other applications, or run the 4B when the
   machine is idle.

## 5. Honest conclusion

On this hardware the 4B is **I/O-bound**. No amount of prefetching or block tuning changes the
fact that a 4.57 GB store plus 2.39 GB of leftovers must be read from a 105 MB/s disk when they
do not fit in RAM. The release therefore ships the guard, the report and the guidance — and
claims no speedup it did not measure.
