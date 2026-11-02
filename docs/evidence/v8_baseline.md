# v8 baseline — what the repository actually does today

**Date:** 2026-11-02
**Branch:** `v8/work`, based on `05182d3` (v7.0.0 publication facts)
**Machine:** Ryzen 7 7735HS, Radeon 680M (unified), 19.8 GiB RAM — free at start **7.5 GiB**,
commit 31.0 / 38.6 GiB
**Interpreter:** `C:\Program Files\Python314\python.exe` 3.14.6 (the dev loop's interpreter;
CI runs 3.12 — see WI-0.3)

## Backend suite (with the training stack installed)

```
cd server && python -m pytest -q -rs --durations=8
```

```
217 passed, 6 warnings in 65.19s (0:01:05)
```

**0 skipped locally** — the 8 training files run here because `torch`, `transformers`, `peft`
and `accelerate` are installed. This is the number CI does *not* reproduce yet (G1): CI installs
`pip install -e .` without `[train]`, so the same 53 tests are silently skipped there.

Slowest tests (local, warm):

| Test | Seconds |
|---|---|
| `test_benchmarks_v4::test_swebench_instance_runs_in_venv` | 9.85 |
| `test_unified::test_unified_int4_merges_before_export_and_reports_every_stage` | 9.07 |
| `test_benchmarks_v4::test_repo_fix_case_passes_with_correct_code` | 8.96 |
| `test_benchmarks_v4::test_repo_fix_case_fails_with_wrong_code` | 8.95 |
| `test_mcp::test_machine_scan_returns_a_profile` | 7.28 |

Frontend and E2E baselines (measured in the v7 release run, unchanged since):

| Suite | Result |
|---|---|
| `npx vitest run` | 7 passed |
| `npm run build` | tsc + vite, 476 KB JS / 23.8 KB CSS |
| `npx playwright test` | 9 passed |

## Library versions (the environment a number is tied to)

| Package | Version |
|---|---|
| Python | 3.14.6 |
| torch | 2.12.1 (+cpu) |
| transformers | 5.12.1 |
| peft | 0.19.1 |
| accelerate | 1.14.0 |
| datasets | 5.0.0 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| numpy | 2.4.6 |
| fastapi | 0.136.3 |
| pydantic | 2.13.5 |
| uvicorn | 0.51.0 |
| psutil | 7.2.2 |
| pytest | 9.1.1 |

CI installs Python 3.12 from `pyproject.toml` (`requires-python >= 3.11`) and pins nothing else —
so CI and local differ in interpreter **and** in resolved versions. WI-0.3 addresses this.

## What this baseline does not prove

- That any training run improves a model. Every run so far is a 2-step smoke run on 53 records
  (`CHANGELOG.md` 7.0.0, "Not claimed").
- That CI would catch a regression in the trainers — it would not (G1).
- That the 4B step time is predictable — it is measured as I/O-bound (`v7_4b_speed.md`), and
  Phase 7 turns that into a predictor.
