from __future__ import annotations

import math

import pytest

from foundry.train.stats import (
    pass_at_k,
    paired_compare,
    required_n,
    summarize,
    wilson_ci,
)


def test_wilson_matches_the_hand_computed_interval():
    low, high = wilson_ci(8, 10)
    assert low == pytest.approx(0.4901, abs=0.002)
    assert high == pytest.approx(0.9433, abs=0.002)


def test_wilson_is_defined_at_the_edges():
    assert wilson_ci(0, 10)[0] == 0.0
    assert wilson_ci(0, 10)[1] < 0.35
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0)
    assert wilson_ci(10, 10)[0] > 0.60
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_matches_the_binomial_tail():
    # b=10 wins, c=2 losses → 2 * P(X <= 2 | n=12, p=0.5) = 0.0386
    result = paired_compare([1] * 10 + [0] * 2 + [1] * 8, [0] * 10 + [1] * 2 + [1] * 8)
    assert result.method == "mcnemar_exact"
    assert result.p_value == pytest.approx(0.0386, abs=0.002)
    assert result.n_discordant == 12
    assert result.delta == pytest.approx(8 / 20)


def test_paired_compare_detects_no_difference_when_runs_agree():
    a = [1, 0, 1, 1, 0, 1]
    result = paired_compare(a, list(a))
    assert result.delta == 0.0
    assert result.p_value == 1.0
    assert result.ci_low <= 0 <= result.ci_high


def test_paired_compare_needs_equal_lengths():
    with pytest.raises(ValueError):
        paired_compare([1, 0], [1])


def test_required_n_matches_the_plan_numbers():
    assert required_n(0.10, 0.20) == 157
    assert required_n(0.05, 0.20) == 628
    assert required_n(0.0, 0.20) >= 10**9


def test_underpowered_flag_refuses_a_six_case_flip():
    # the old suite: 6 cases, one flipped in the candidate's favour
    treatment = [1, 1, 1, 1, 1, 1]
    baseline = [0, 1, 1, 1, 1, 1]
    result = paired_compare(treatment, baseline, target_delta=0.05)
    assert result.delta == pytest.approx(1 / 6)
    assert result.underpowered is True, "one flip in six cases cannot promote"
    assert result.improved() is False
    assert result.required_n and result.required_n > result.n


def test_a_large_effect_on_enough_items_passes_the_gate():
    n = 400
    treatment = [1] * 260 + [0] * 140
    baseline = [1] * 200 + [0] * 100 + [1] * 0 + [0] * 100
    baseline = [1] * 200 + [0] * 200
    result = paired_compare(treatment, baseline, target_delta=0.10)
    assert result.delta == pytest.approx(0.15)
    assert result.ci_low > 0
    assert result.improved(min_effect=0.03) is True


def test_clustered_comparison_reports_more_uncertainty():
    # 40 items in 4 clusters; within a cluster all items agree (perfect correlation)
    clusters = [f"t{i // 10}" for i in range(40)]
    treatment = [1] * 30 + [0] * 10
    baseline = [0] * 10 + [1] * 30
    naive = paired_compare([float(x) for x in treatment], [float(x) for x in baseline])
    clustered = paired_compare(
        [float(x) for x in treatment], [float(x) for x in baseline], clusters=clusters
    )
    assert clustered.n_clusters == 4
    assert clustered.se >= naive.se * 1.5, "clustered SE must not pretend the items are independent"


def test_pass_at_k_matches_brute_force():
    # n=5 samples, 2 of them pass; k=2 → 1 - C(3,2)/C(5,2) = 1 - 3/10
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)
    assert pass_at_k(5, 0, 2) == 0.0
    assert pass_at_k(5, 5, 2) == 1.0
    assert pass_at_k(5, 2, 5) == 1.0
    for n in range(1, 8):
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                expected = 1 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0)
                assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_summarize_reports_rate_with_interval():
    summary = summarize([1, 1, 0, 0, 1])
    assert summary["n"] == 5 and summary["passed"] == 3
    assert summary["rate"] == pytest.approx(0.6)
    assert summary["ci_low"] < 0.6 < summary["ci_high"]
