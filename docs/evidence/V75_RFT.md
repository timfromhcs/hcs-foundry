# Evidence — v7.5.0 (RFT trainer adapter, charter V8-S4)

**Date:** 2026-11-02
**Charter:** `HCS_FOUNDRY_LONG_HORIZON_AGENT.md` §5 V8-S4
**Code:** `server/foundry/train/adapters/rft.py`, registered as `rft/fp32_lora`

## What the adapter does

```
tasks.jsonl (train/dev/test, template-ranked)
        │  refuse if a template appears in two splits
        ▼
sample the base model on TRAIN items only (bounded candidate budget)
        │
        ▼
deterministic verifier (foundry.tasks.generators) ──► accepted / rejected (+ reason counts)
        │  fewer than min_accepted → manifest.json status=failed + TrainingError
        ▼
frozen synthetic dataset (provider=synthetic, rights=confirmed) + accepted.jsonl + sha256
        │
        ▼
existing SFT path (LoRA) → adapter, loss trajectory, param delta
        │
        ▼
RunManifest v2 + extras: counts, reasons, seed, generator/verifier versions, dataset hash
```

## Verified behaviour

| Requirement (charter) | Evidence |
|---|---|
| uses TrainSpec/registry/preflight/manifest | `rft/fp32_lora` in `list_trainers()`, `foundry train plan` works |
| candidates only from allowed training prompts | run selects `split == "train"`; `used_test_items` recorded but never sampled |
| deterministic verifiers | `foundry.tasks.generators.verify`; a plausible wrong answer is rejected |
| counts, versions, seed, hashes recorded | asserted in `test_rft_manifest_records_counts_versions_and_seed` |
| zero accepted ⇒ not completed | `test_rft_fails_when_no_candidate_passes`: `manifest.json` `status: failed` + `TrainingError` |
| no test/holdout leakage | `test_leaked_templates_are_refused_before_sampling` (preflight blocker **and** run-time refusal) |
| bounded candidate budget | preflight blocker + the same check in `run()`; `test_the_candidate_budget_is_enforced_in_preflight` |
| conformance matrix | `run_matrix`: C1/C2/C3/C7/C8/C9/C10 **passed**, C4/C5/C6 `not_applicable` with reasons |

## Numbers

```
RFT tests                       6 passed
backend suite                   325 passed (319 + 6)
matrix on the tiny fixture      accepted >= 6, rejected 0, param delta > 0, adapter reloaded with a finite forward pass
zero-accept run                 rejected_count > 0, rejection_reasons.verifier_rejected > 0, status=failed
```

## Not claimed

- **No quality improvement.** The default sampler on the tiny random model produces useless answers
  by design; the positive path is proven with an injected oracle sampler.
- **The default sampler is not governed.** It loads a second copy of the base model; offering RFT on
  4B needs the runtime/memory governor first.
- **No flywheel run.** S5 remains open (`docs/progress/NEXT_ACTION.md`).
