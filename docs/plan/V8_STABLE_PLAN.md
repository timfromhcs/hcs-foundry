# Plan to the next stable — v8.0.0

**Written:** 2026-11-02, after `v7.2.0` (browser runtime)
**Supersedes:** the v7.2.0 rows in `PLAN.md` §4.1 where they conflict
**Rule:** no claim without a gate. Every phase below names the test that can fail.

---

## 0. Verified starting point

| Fact | Value | Where it was measured |
|---|---|---|
| Branch / commit | `main` @ `94a6257` (`v7.2.0` tag at `20bcb6e`) | `git log` |
| Backend tests | 269 passed, 0 skipped with the ML stack installed | `python -m pytest -q` |
| Frontend / E2E | 17 vitest, 10 Playwright | `npm run build && npx playwright test` |
| CI | 3 jobs green on `4be25fd`, `20bcb6e`, `94a6257` | runs `41447889286`, `41448332378`, `41448871518` |
| Local GGUF store | `qwen3-0.6b` (F16 + Q4_K_M), `v6-2b` (F16 + Q4_K_M), `v6-4b` (F16 + Q4_K_M), `v5-int4-merged` | `data/gguf` |
| Hardware | Ryzen 7 7735HS, 16 threads, Radeon 680M (Vulkan 1.3), ~10 GB usable RAM | `foundry run doctor` |
| Training | LoRA / INT4 stream / INT4 store / DPO / distill run on the tiny fixture and 0.6B | `tests/train` |
| Browser runtime | detection + limits measured; **token throughput not measured** (no ONNX export) | `docs/BROWSER_RUNTIME.md` |

## 1. What "stable v8.0.0" means (unchanged from `PLAN.md` §5)

| # | Criterion | Status now | Blocks on |
|---|---|---|---|
| **A1** | CI runs the training matrix with 0 skips | ✅ done | — |
| **A2** | one spec drives ≥ 5 methods incl. `rft`; manifest v2 for each | ⚠️ 5 configs, no `rft` | S4 |
| **A3** | promotion uses paired statistics | ✅ done | — |
| **A4** | held-out improvement with paired CI **or** a truthful negative report | ❌ not attempted | S3 → S5 |
| **A5** | preflight predicts step time from measured bandwidth ±50 % | ❌ | S6 |
| **A6** | quantize reports KL + flip rate; KV sweep from the installed build | ❌ | S6 |
| **A7** | MCP exposes `train_plan, train_run, models_run, bench_run, eval_compare` | ⚠️ 2 of 5 | S2 |
| **A8** | docs honest and current | ✅ through 7.2.0, must follow this work | S7 |
| **A9** | release completed through CI, site, EXE | ✅ machinery proven twice | S7 |

A4 is the one criterion that cannot be met by code alone: it needs hours of sampling, training and a
paired evaluation, and it is allowed to end in a **negative report**. Nothing below pretends
otherwise.

## 2. Phases

**Progress (2026-11-02, after v7.5.0):** S1–S4 are implemented and green (325 backend / 17
frontend / 11 E2E). S5, S6 and S7 remain open; A4/A5/A6 still need measured runs.


### S1 — Close the conformance matrix (M)

C1, C2, C7, C8 exist. Add the rest as pure checks over the manifest and artifacts, so a trainer
cannot ship by looking finished:

| Check | Meaning | Fails when |
|---|---|---|
| **C3** | it learns | `losses` has ≥ 3 points and the last window is not below the first by a real margin |
| **C4** | repeatable | same spec + seed produces a param delta that differs beyond tolerance; the report prints the observed delta |
| **C5** | resume equivalence | resuming from step *k* lands in the same state as an uninterrupted run (steps + loss window) |
| **C6** | cancellation is timely | a cancel during a run is honoured within `cancel_latency_s` and the manifest says `cancelled`, not `completed` |
| **C9** | adapter reloads | the saved adapter loads and produces a finite forward pass |
| **C10** | no false success | a run with zero parameter movement cannot be `completed` |
| **C11** | budget respected | a run whose estimate exceeds `Budget.max_wall_s` is refused in preflight |

*Acceptance:* `run_matrix(spec, trainer, ctx)` returns `ok=True` for sft on the tiny fixture; every
check has a negative test (a constructed manifest or a deliberately broken trainer) proving it can
fail. `foundry train doctor` prints the matrix.

*Loop:* run pytest (train subset) → read failure → fix the check or the trainer → rerun; max 5
iterations, never weaken an assertion to pass.

### S2 — The missing surfaces (M)

- `POST /api/v1/training/plan` — validate a `TrainSpec`, return preflight + normalized spec hash,
  **no training**. `POST /api/v1/training/runs` — enqueue a spec-driven job (same runner as the CLI).
- MCP: `train_plan`, `models_run` (start/stop the supervised llama-server), `eval_compare`
  (paired statistics over two stored benchmark runs), and `train_run` / `bench_run` **with an
  explicit budget refusal** rather than a silent long run.
- The UI gets the plan form generated from the schema fields the endpoint returns.

*Acceptance:* API tests cover plan (valid, invalid, missing model, unknown dataset) and runs
(enqueued, refused when over budget); MCP tests call every new tool through `_call` and check the
typed error codes; no tool can execute a shell command.

### S3 — Verifiable task generators (L)

Pure Python, deterministic given a seed, each family = generator + verifier + metadata:

| Family | Verifier |
|---|---|
| `fmt` | reformat (CSV→JSON, names→sorted list) exact-match |
| `arith` | integer/float arithmetic with a tolerance of 0 |
| `extract` | field extraction from a synthetic record with a schema |
| `code_synth` | executed tests in a subprocess (already exists in `benchmarks.execute_python`) |
| `code_fix` | failing test → patched function, executed |
| `toolcall` | JSON tool call matches the allowed schema |

Plus: split by **template**, not by row (anti-leakage by construction); decontamination against a
pinned eval set at freeze time; a readiness gate that refuses a dataset with < N verified rows per
skill.

*Acceptance:* every family has an oracle test (generator output passes its own verifier) and an
adversarial test (a plausible wrong answer fails). Splits are disjoint by template; a row whose
template appears in the eval set is refused.

### S4 — `rft` adapter (M)

Rejection-sampling fine-tuning over the generators from S3, recording accepted/rejected counts in
the manifest extras. Completes A2.

*Acceptance:* conformance matrix runs `rft` on the tiny fixture; a run with zero accepted samples is
`failed`, never `completed`.

### S5 — The measurement (L, hours)

Run the flywheel on Qwen3-0.6B: base vs candidate on ≥ 200 held-out generated tasks per skill,
paired comparison, 95 % CI, guard suite. Outcome is either a promoted candidate or a written
negative report (`docs/evidence/v8_flywheel.md`). This is the only phase that can fail on
mathematics rather than code, and it is allowed to.

### S6 — Engine and quantization gates (L, hours, machine-exclusive)

E1–E5 from `PLAN.md` §9 on the INT4 store (micro-batch curve, layer-major scheduling, bf16
leftovers, sequential reads, ETA prediction), then the quant/KV sweep with KL + flip rate from the
installed `llama.cpp` build. Losing experiments are documented as losses.

### S7 — Docs, site, release v8.0.0 (M)

README, `KNOWN_LIMITATIONS`, `TRAINING`, `BENCHMARKS`, `UI_GUIDE`, `TUTORIAL`, `CHANGELOG`, release
report, site changelog, EXE, tag — only when S1–S6 have evidence. If A4 ends negative, the release
text says so in the title.

## 3. Self-healing loop protocol

Every work item runs the same bounded loop. No exceptions, no unbounded retries.

```
1. reproduce      run the narrowest failing test; keep the error
2. diagnose       read the traceback and the code path; state the hypothesis in one sentence
3. patch          smallest change that addresses the cause (no rewrites)
4. narrow test    run only that test
5. impacted tests run the whole module, then the suite
6. review         read the diff; check nothing unrelated moved
7. repeat         max 5 iterations per item, then stop and write the blocker
```

Hard rules:

- Never weaken an assertion, skip a test, or change scoring to make a run green.
- Never delete a regression test that now fails.
- A fix that needs a new dependency stops and is recorded, not installed blindly.
- If a failure is environmental (missing model, no GPU, no network), the item is marked `blocked`
  with the exact command that would unblock it — not "done with a caveat".
- Each item ends with: tests green, evidence line in the phase file, commit `type(scope): summary`.

## 4. Order and gates

| Gate | Requires | Ships |
|---|---|---|
| **SG-1** | S1 + S2 + S3 green locally, CI green | `v7.3.0` hardening release (no quality claim) |
| **SG-2** | S4 + S5 evidence (positive or negative) | decides whether `v8.0.0` is a win or an honest negative |
| **SG-3** | S6 evidence | `v8.0.0` |

This session executes **S1 → S2 → S3** and stops at SG-1 if the loops are green. S4–S6 need
multi-hour CPU runs and are scheduled, not faked.

## 5. Risks carried into this session

1. **Bit-determinism (C4) is not guaranteed by CPU float ops.** The check reports the observed
   delta and fails only above a stated tolerance; the tolerance is printed in the detail, never
   hidden.
2. **Tiny-model learning (C3) can be noisy at 1–2 steps.** The check requires a minimum step count
   and refuses to judge below it (`not_applicable`, not `passed`).
3. **Generators can leak through shared templates.** Splits are by template and the readiness gate
   refuses overlaps; the test proves the refusal.
4. **The 4B store experiments need the page cache cold.** If the machine cannot be flushed safely,
   E1–E5 record `warm cache` and the numbers are labelled accordingly — never presented as cold.
