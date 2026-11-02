# Evidence — v7.3.0 hardening (S1–S3 of `docs/plan/V8_STABLE_PLAN.md`)

**Date:** 2026-11-02
**Commit:** `49d5f54` (branch `main`)
**Scope:** conformance matrix, spec surfaces, verifiable task generators
**Not in scope:** any quality claim (S5), engine/quant gates (S6)

## S1 — conformance C3–C11

`foundry/train/conformance.py` now has C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11 and
`run_matrix()`. Every check has a test that drives it into the failing branch.

```
C3   learning            needs >= 3 loss points, last window < first by >= 1 %   (real run: passed)
C4   repeatability       prints |delta| vs tolerance, fails above it              (logic tests)
C5   resume equivalence  not_applicable: no trainer exposes a resume path
C6   cancellation        not_applicable: legacy manifests record no cancel status
C9   adapter reload      PeftModel.from_pretrained + finite forward pass          (real run: passed)
C10  no false success    completed with LoRA-B == 0 is a failure
C11  budget              refuses an estimate above max_wall_s; not_applicable without an ETA
```

Observed on the tiny model (`sft/fp32_lora`, 6 steps, lr 5e-2): C3 `loss 3.026 -> 2.912`,
C2 `max |LoRA-B| = 0.174`, C9 `forward ok`, C10 `passed`, C4/C5/C6 `not_applicable` with reasons.

The matrix runs in `foundry train doctor --model … --dataset …` (three steps so C3 can judge).

## S2 — surfaces

| Surface | Behaviour | Tests |
|---|---|---|
| `POST /training/plan` | validates the spec, returns trainer + preflight + `sha256:` spec hash; no training | 3 |
| `POST /training/runs` | queues the spec; a blocked preflight returns 409 with blockers | 2 |
| MCP `train_plan` | same plan, typed error on a schema violation | 2 |
| MCP `eval_compare` | pairs two stored benchmark runs by case id, refuses different suites/versions, returns the paired CI + regression cases | 2 |

MCP tool count: 11 (`TOOL_NAMES` asserted by the existing test).

## S3 — generators

```
families    arith, fmt, extract, toolcall (10 templates)
verifiers   exact numeric, JSON deep equality, CSV line equality, {tool, arguments} equality
splits      template-ranked 60/20/20 → a test template cannot appear in training
gate        train >= 60, dev >= 20, test >= 20, no template leak
```

`foundry tasks generate --family all --seed 1 --per-template 10` → 100 items, splits 60/20/20,
readiness ok; `foundry tasks check` on the written JSONL reproduces the same counts.

Adversarial tests: a wrong number fails, trailing prose fails, a missing or extra tool argument
fails, a numeric field sent as a string fails, a hand-made template leak is detected, and a
too-small split is refused.

## Gates

```
backend (windows, py3.14)   309 passed (40 new)
frontend (vitest)            17 passed
E2E (playwright)             10 passed
build                        green
CI                           see the run id in the release report of the tag
```

## Open, and why

| Item | Blocks on |
|---|---|
| S4 `rft` adapter | code (next session) |
| S5 measurement (A4) | hours of sampling + training + paired eval |
| S6 engine/quant gates (A5/A6) | hours of machine-exclusive runs |
| C5/C6 actually passing | trainers need resume + a cancel-aware manifest status |
| two generator families | the subprocess test runner has to be wired into the generator API |
