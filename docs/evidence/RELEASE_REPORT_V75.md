# Release report — HCS Foundry v7.5.0 (RFT adapter)

**Date:** 2026-11-02
**Charter:** `HCS_FOUNDRY_LONG_HORIZON_AGENT.md` §5, phase **V8-S4**
**Feature commit:** `f9ba8c6` (CI run `41488672639`, all three jobs green)
**Tag:** `v7.5.0` (on the documentation commit that follows)

## Why this release exists

The charter says: finish V8 before expanding scope, and the next unfinished gate after v7.3/v7.4 was
**V8-S4 — RFT**. This release implements it as a first-class trainer, not a script:

```
train-split generated tasks
   → refuse if any template appears in two splits
   → sample the base model (bounded candidate budget)
   → deterministic verifier: accepted / rejected (+ reason counts)
   → < min_accepted → manifest.json status=failed + TrainingError
   → frozen synthetic dataset (provider=synthetic, rights=confirmed)
   → existing LoRA SFT path
   → RunManifest v2 + extras (counts, reasons, seed, versions, hashes)
```

## Verified

| Requirement | Evidence |
|---|---|
| uses the existing contract | `rft/fp32_lora` registered; `train plan`, `POST /training/plan` accept the spec |
| only train prompts are sampled | run filters `split == "train"`; `used_test_items` recorded, never sampled |
| deterministic verification | `foundry.tasks.generators.verify`; plausible wrong answers rejected |
| counts/seed/versions/hashes | asserted in the RFT manifest test |
| zero accepted ⇒ failed | `manifest.json` `status: failed`, `TrainingError`, reason counts present |
| leakage refused | preflight blocker **and** run-time refusal |
| bounded work | candidate budget enforced in preflight and in `run()` |
| conformance matrix | C1/C2/C3/C7/C8/C9/C10 passed; C4/C5/C6 `not_applicable` with reasons |

```
backend    325 passed (6 new)
frontend    17 passed
E2E         11 passed
build       green
```

## Not claimed

- **No quality improvement.** The positive path uses an injected oracle sampler; the default sampler
  has not produced a real accepted sample on a real model yet.
- **The default sampler is not governed** — it loads a second copy of the base model.
- **V8-S5 (measured flywheel) has not run.** `docs/progress/NEXT_ACTION.md` records the exact next
  command and the missing loader that turns generated tasks into benchmark cases.
- C5/C6 remain `not_applicable` until the trainers expose resume and a cancel-aware manifest status.

## Publication

| Item | Value |
|---|---|
| Tag | `v7.5.0` |
| Feature CI | `41488672639` — green |
| Docs CI | recorded in the tag commit's check |
| EXE | rebuilt from the tagged commit |
| Site | <https://hcs-foundry-site.vercel.app> |
