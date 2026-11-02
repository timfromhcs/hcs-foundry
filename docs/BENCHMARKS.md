# Benchmarks — how to read a number, and when a difference is real

All suites are **project-specific**: they measure this machine, this model file and this runtime.
A score here is never a leaderboard claim.

## What is measured

| Suite | Oracle | Cases (current) |
|---|---|---|
| `instruction_adherence` | deterministic assertions (format, language, secrecy, arithmetic) | 6 |
| `codegen_exec` | executed unit tests in a sandbox | 4 |
| `repo_fix` | a real patch applied to a fixture repo, tests run in a venv | 3 |
| `agentic` | multi-step file tasks with deterministic checks | 3 |
| `latency_throughput` | measured tokens/s, TTFT | — |

## Why a difference needs an interval

The Upgrade Lab used to promote on `delta ≥ 0.05`. With six cases one flipped case is **16.7 pp**,
and the smallest effect that six cases could ever detect at 80 % power is ≈ 63 pp. That gate could
only react to noise or to enormous effects.

`foundry/train/stats.py` replaces it:

```python
from foundry.train.stats import paired_compare, required_n, wilson_ci

baseline  = [1, 0, 1, 1, 0, 1]      # per-item outcome, 1 = pass
candidate = [1, 1, 1, 1, 0, 1]
result = paired_compare(candidate, baseline, target_delta=0.05)
result.to_public()
# {'delta': 0.1667, 'ci_low': -0.09, 'ci_high': 0.42, 'p_value': 1.0, 'n': 6,
#  'method': 'mcnemar_exact', 'underpowered': True, 'required_n': 628}
```

- **Paired** differences cancel the shared difficulty of an item; comparing aggregate means throws
  that information away.
- **Exact McNemar** for binary outcomes, a clustered bootstrap for continuous scores. Pass
  `clusters=<template_id list>` when items come from generated templates — otherwise the interval
  pretends the items are independent.
- **`underpowered`** is set when the observed discordance implies a required n above the sample
  size. A promotion is refused with that word in the reason, not silently allowed.
- **`required_n(delta, discordance)`** answers "how many items do I need?": 157 for a 10 pp effect
  at 20 % discordance, 628 for 5 pp.

## The gate (Upgrade Lab)

`promote()` refuses unless all of these hold:

1. a paired result is recorded (`metrics["paired"]`),
2. `n ≥ 30` and the 95 % CI lower bound is **> 0**,
3. the effect is at least `min_effect` (default 3 pp, the practical minimum),
4. no critical regression (`system_prompt_secrecy`),
5. the holdout hash is unchanged and the dataset is still frozen,
6. a named human approver.

The old `threshold=` argument still works as a deprecated alias for `min_effect`.

## Reproducing a comparison

```bash
foundry bench instruction_adherence --limit 200     # store per-item outcomes
curl -s "$BASE/api/v1/benchmarks/runs" | jq '.runs[0].results.cases'
```

Each run stores a stable `case_id` and a per-item status, which is what makes the paired test
possible. A comparison between runs whose case ids do not overlap is rejected as unfair rather
than guessed.
