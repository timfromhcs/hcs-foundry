# Agent state

```yaml
phase: V8-S4
status: complete
branch: main
base_commit: 14723ea          # v7.4.0
head_commit: pending           # filled by the release commit
verified_facts:
  - "RFT is registered as rft/fp32_lora and reachable through TrainSpec, the API plan/run endpoints and the CLI"
  - "server/tests/train/test_rft.py: 6 tests, all passing"
  - "full backend suite: 325 passed"
  - "run_matrix on RFT reports C1/C2/C3/C7/C8/C9/C10 passed and C4/C5/C6 not_applicable with reasons"
  - "zero accepted candidates writes manifest.json status=failed and raises TrainingError"
  - "a leaked template is refused in preflight and again in run()"
  - "the candidate budget is a preflight blocker and run() refuses the same mismatch"
assumptions:
  - "the default sampler (greedy generation from the base model) was exercised only with the tiny fixture, where it produces useless answers on purpose"
  - "no quality claim is made: S5 has not run"
commands_run:
  - "python -m pytest tests/train/test_rft.py -q"
  - "python -m pytest -q"
tests:
  - "tests/train/test_rft.py::test_rft_reaches_the_matrix_with_accepted_samples"
  - "tests/train/test_rft.py::test_rft_manifest_records_counts_versions_and_seed"
  - "tests/train/test_rft.py::test_rft_fails_when_no_candidate_passes"
  - "tests/train/test_rft.py::test_a_plausible_but_wrong_answer_is_rejected"
  - "tests/train/test_rft.py::test_leaked_templates_are_refused_before_sampling"
  - "tests/train/test_rft.py::test_the_candidate_budget_is_enforced_in_preflight"
artifacts:
  - "server/foundry/train/adapters/rft.py"
  - "docs/evidence/V75_RFT.md"
risks:
  - "the default sampler loads a second copy of the base model in memory; for 4B that must go through the governed runtime before it is offered broadly"
  - "RFT trains SFT on accepted pairs only; the derived dataset is written to the same database and needs a retention policy for repeated runs"
```

## Notes

The charter's next gate is **V8-S5** (measured flywheel). It needs hours of sampling, training and a
paired evaluation on held-out tasks; nothing in this session pretends to have run it.
