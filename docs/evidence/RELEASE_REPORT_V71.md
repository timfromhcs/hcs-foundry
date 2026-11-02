# Release report — HCS Foundry v7.1.0 (trust release)

**Date:** 2026-11-02
**Tag:** `v7.1.0` → commit `2a1a82f`
**Plan:** `PLAN.md` (Phases 0–3), blockers in `docs/plan/V8_BLOCKERS.md`

## Why this release exists

Two things were untrue about v7.0.0, and both are fixed here:

1. **"CI green" said nothing about the trainers.** CI installed `pip install -e .` without the
   training extras, so 53 of 217 tests called `pytest.importorskip("torch")` and vanished. The
   backend job now installs a compiled, constrained lock and runs with `FOUNDRY_REQUIRE_TRAIN=1`;
   a missing training module is a **failure**, not a skip.
2. **The promotion gate could only react to noise.** `delta >= 0.05` on a 6-case suite: one flipped
   case is 16.7 pp. The gate now needs a paired result with `n ≥ 30`, a 95 % CI lower bound > 0,
   a practical minimum effect and no critical regression.

## Delivered

| Phase | What |
|---|---|
| P0 | baseline evidence (`docs/evidence/v8_baseline.md`), `requirements-lock.txt` + `requirements-train-lock.txt` constrained to the measured versions, `web/dist` untracked |
| P1 | CI runs the training suite: **264 passed, 0 skipped on the GitHub runner** (run `41436603665`); `tests/conftest.py` guard + `tests/test_ci_guard.py` |
| P2 | unified contract: `TrainSpec` (strict schema, profile defaults, stable hash, JSON/YAML), `RunManifest` v2 + `from_legacy`, trainer registry, adapters for sft/int4×2/dpo/distill, conformance checks C1/C2/C7/C8 on the tiny model, pure loss functions with oracle tests, CLI `train plan|run|doctor` |
| P3 | `foundry/train/stats.py`: Wilson interval, paired McNemar / clustered bootstrap, `required_n`, `pass_at_k`; the Upgrade Lab gate rewritten; the six-case flip is refused with the word *underpowered* |

## Proof

```
CI (github, linux, py3.12, torch cpu)   264 passed, 0 skipped, 2m18s   run 41436603665
local (windows, py3.14)                 264 passed, 0 skipped
frontend                                7 vitest, build 476 KB JS, 9 Playwright E2E
stats known answers                     Wilson(8/10) = (0.490, 0.943)
                                        McNemar(b=10,c=2) p = 0.0386
                                        required_n(0.10, 0.20) = 157 · (0.05, 0.20) = 628
conformance matrix                      sft, int4_stream, int4_store, dpo: C1/C2/C7/C8 pass
```

## Not claimed

- **No quality improvement.** Phase 4 (task generators) and Phase 5 (flywheel + measured result)
  were not started; the release makes the *measurement* honest, not the model better.
- Conformance checks C3–C6, C9–C11 (learning, determinism, resume, cancel, adapter reload,
  no-false-success, budget) are planned and not implemented — listed in `V8_BLOCKERS.md`.
- The API (`POST /training/runs`) and the spec-driven UI form are not wired yet; the CLI is.
- The lock was compiled for python 3.12/linux; the dev machine runs 3.14. Both are recorded.

## Publication

| Item | Value |
|---|---|
| Tag | `v7.1.0` |
| CI | run `41436603665` — green (backend with training extras) |
| Landing page | <https://hcs-foundry-site.vercel.app> |
| EXE | rebuilt from the tagged commit |
