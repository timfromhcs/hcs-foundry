"""Efficiency research: how do we actually read the INT4 store fastest on this machine?

Three access patterns on the same 4.57 GB store that lives on E: (external 466 GB disk):

1. mmap page faults — what ``Int4Store.packed()`` slicing does today (4 KiB granularity).
2. unbuffered ``read`` in 8 MiB chunks — sequential, few syscalls.
3. unbuffered ``read`` in 21 MiB chunks — one call per layer, the granularity of a step.

Windows has no ``os.pread``, so the sequential path is ``seek`` + ``read`` on an unbuffered file.

Run:  python scripts/store_io_bench.py [store_dir]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STORE = Path(sys.argv[1] if len(sys.argv) > 1 else r"E:\AILOCALTRAIN\data\training-runs\v41-int4-4b\int4-store")
SAMPLE_BYTES = 512 * 1024**2
CHUNK = 8 * 1024**2


def rate(megabytes: float, seconds: float) -> float:
    return megabytes / seconds if seconds else 0.0


def bench_mmap(path: Path, sample: int) -> float:
    import mmap

    handle = open(path, "rb")
    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    started = time.time()
    reached = 0
    block = 0
    while reached < sample and block < len(mapped):
        chunk = mapped[block : block + 4096]
        reached += len(chunk)
        block += 4096
    seconds = time.time() - started
    del mapped
    handle.close()
    return rate(reached / 1024**2, seconds)


def bench_read(path: Path, sample: int, chunk: int) -> float:
    handle = open(path, "rb", buffering=0)
    started = time.time()
    reached = 0
    try:
        handle.seek(0)
        while reached < sample:
            data = handle.read(chunk)
            if not data:
                break
            reached += len(data)
    finally:
        handle.close()
    return rate(reached / 1024**2, time.time() - started)


def main() -> int:
    store = STORE
    weights = store / "weights.u8"
    rest = store / "rest.f32"
    if not weights.exists():
        print(json.dumps({"error": f"no store at {store}"}))
        return 1

    results = {
        "store": str(store),
        "sample_mib": round(SAMPLE_BYTES / 1024**2),
        "weights_gib": round(weights.stat().st_size / 1024**3, 2),
        "mmap_4k_pagefault_mb_s": round(bench_mmap(weights, SAMPLE_BYTES), 1),
        "read_8mib_mb_s": round(bench_read(weights, SAMPLE_BYTES, CHUNK), 1),
        "read_21mib_layer_mb_s": round(bench_read(weights, SAMPLE_BYTES, 21 * 1024**2), 1),
    }
    if rest.exists():
        results["rest_f32_size_gib"] = round(rest.stat().st_size / 1024**3, 2)
        results["rest_mmap_4k_pagefault_mb_s"] = round(bench_mmap(rest, SAMPLE_BYTES // 2), 1)
        results["rest_read_8mib_mb_s"] = round(bench_read(rest, SAMPLE_BYTES // 2, CHUNK), 1)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
