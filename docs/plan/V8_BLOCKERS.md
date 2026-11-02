# v8 blockers and scope report

**Date:** 2026-11-02
**Branch:** `v8/work` (based on `05182d3`, v7.0.0)
**Plan:** `PLAN.md` (12 phases, 3 release gates)

This file is the honest stop point of this session. It records what the plan asked for, what was
delivered and verified, and what was not attempted — with the reason, so the next run does not
have to rediscover it.

---

## Delivered and verified

| Phase | Status | Evidence |
|---|---|---|
| **Phase 0** baseline, locks, hygiene | **done** | `docs/evidence/v8_baseline.md`; `server/requirements-lock.txt`, `requirements-train-lock.txt`, `constraints.txt`; `web/dist` untracked (build verified) |
| **Phase 1** CI tests the trainers | **done** | CI installs the train lock, runs with `FOUNDRY_REQUIRE_TRAIN=1` and `--strict-markers`; `tests/conftest.py` guard; `tests/test_ci_guard.py` (4 tests) |
| **Phase 2** unified contract | **core done, surfaces partial** | `foundry/train/{spec,manifest,base,registry,losses,conformance}.py`, adapters for sft/int4(×2)/dpo/distill; `train plan|run|doctor`; 31 tests incl. the conformance matrix on the tiny model |
| **Phase 3** statistics gate | **done** | `foundry/train/stats.py` (11 known-answer tests); `upgrade.promote()` rewritten; the six-case flip is refused (`test_gate_refuses_a_single_flipped_case`) |

Local gate at this point: **264 backend tests pass** (0 skips with
`FOUNDRY_REQUIRE_TRAIN=1`), frontend 7 + 9 E2E unchanged.

## Delivered in part

| Plan item | What exists | What is missing |
|---|---|---|
| WI-2.5 conformance C1–C12 | C1 (preflight), C2 (real update via `max|LoRA-B|`), C7 (source untouched), C8 (manifest v2) run for every registered trainer | C3 (learns something), C4 (bit-determinism), C5 (resume equivalence), C6 (cancel latency), C9 (adapter reload), C10 (no-false-success), C11 (budget), C12 is covered by `test_spec.py` |
| WI-2.7 surfaces | CLI `train plan|run|doctor` | API `POST /training/runs`, UI form generated from the schema, legacy-command aliases with deprecation notes |
| WI-2.8 playbook LR split | not started | the playbook still stores hyper-parameters as one blob |

## Not started (with the reason)

| Phase | Why not |
|---|---|
| **P4** verifiable task generators | needs the generator/verifier families (fmt, arith, extract, code_synth, code_fix, toolcall) plus split-by-template and decontamination. Each family needs its own adversarial test set; this is a multi-day item, not a plausible end-of-session task |
| **P5** flywheel + first measured result | depends on P4 and on hours of sampling + training + paired eval. It is the release's actual goal (A4) and was not attempted — no quality claim is made |
| **P6** on-policy distillation | needs `n_probs` support in the llama-server client (WI-6.1), which was not started |
| **P7** engine experiments (E1–E5) | machine-exclusive runs; the host was already at 3.6–7.5 GB free. Not attempted rather than measured badly |
| **P8** quant/KV gates | needs logprobs or `llama-perplexity --kl-divergence` verified against the installed build |
| **P9** MCP/UI additions | `train_plan`, `train_run`, `models_run`, `bench_run`, `eval_compare` not exposed yet |
| **P10/P11** docs/remote | remote backend is design-only by the plan; the docs are current for what shipped |

## Known risks carried forward

1. **The CI install path is unverified on GitHub until the first push** of this branch. The lock
   was compiled for `python 3.12 / linux` with the CPU torch index; the install command uses
   `--extra-index-url` and `pip` resolves `torch==2.12.1+cpu` from it. If CI fails here, the fix is
   in `.github/workflows/ci.yml`, not in the tests.
2. **A local clean-venv install of the lock on Windows 3.13 failed** while building a dependency
   wheel; the same lock is expected to install fine on the CI image. This needs the CI result to
   be either confirmed or corrected.
3. **`train doctor --model --dataset` runs real trainers**, so it takes minutes on a real 0.6B
   model. With the tiny fixture it is fast; the command does not create that fixture itself.
4. **The gate rewrite is a behaviour break** for anyone who relied on `threshold=`; it is now the
   practical minimum, and the significance decision needs `metrics["paired"]`.

## Next actions, in order

1. Push `v8/work` (PG-1) and watch CI; fix the install path if it fails.
2. Finish WI-2.5 (C3, C5, C6, C10) — these are the checks most likely to find real trainer bugs.
3. P4 generators (`fmt`, `arith`, `extract` first) with verifier tests.
4. P5 wireframe: RFT sampling with the GGUF engine, verification, SFT, merge, paired eval.
