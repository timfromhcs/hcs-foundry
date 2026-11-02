from __future__ import annotations

import math

import pytest

pytest.importorskip("torch", reason="loss oracles need torch")

import torch  # noqa: E402

from foundry.train.losses import (  # noqa: E402
    clip_ratios,
    dapo_filter_groups,
    dapo_token_loss,
    dpo_loss,
    grpo_advantages,
    reverse_kl_loss,
    soft_overlong_penalty,
)


def test_dpo_loss_matches_the_formula():
    beta = 0.1
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -2.5])
    ref_chosen = torch.tensor([-1.5, -2.2])
    ref_rejected = torch.tensor([-2.5, -2.4])
    loss, margin = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)
    expected_margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    assert torch.allclose(margin, expected_margin)
    assert torch.allclose(loss, -torch.nn.functional.logsigmoid(expected_margin))


def test_dpo_loss_at_zero_margin_is_log_two():
    zero = torch.zeros(1)
    loss, margin = dpo_loss(zero, zero, zero, zero, beta=0.1)
    assert margin.item() == 0.0
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)


def test_dpo_loss_is_monotone_in_the_margin():
    ref = torch.zeros(1)
    small, _ = dpo_loss(torch.tensor([0.1]), torch.zeros(1), ref, ref, beta=1.0)
    large, _ = dpo_loss(torch.tensor([2.0]), torch.zeros(1), ref, ref, beta=1.0)
    assert large.item() < small.item()


def test_reverse_kl_is_zero_when_student_matches_teacher():
    same = torch.log_softmax(torch.randn(4, 16), dim=-1)
    assert reverse_kl_loss(same, same).item() == pytest.approx(0.0, abs=1e-6)


def test_reverse_kl_is_positive_for_a_worse_student():
    torch.manual_seed(0)
    teacher = torch.log_softmax(torch.randn(64, 32), dim=-1)
    student = torch.log_softmax(torch.randn(64, 32), dim=-1)
    assert reverse_kl_loss(student, teacher).item() > 0


def test_grpo_advantages_are_standardised_per_group():
    advantages = grpo_advantages([1.0, 0.0, 0.5, 0.5])
    assert sum(advantages) / len(advantages) == pytest.approx(0.0, abs=1e-9)
    std = math.sqrt(sum(value * value for value in advantages) / len(advantages))
    assert std == pytest.approx(1.0, abs=1e-9)


def test_dr_grpo_drops_the_std_normalisation():
    rewards = [1.0, 0.0, 0.5, 0.5]
    plain = grpo_advantages(rewards)
    dr = grpo_advantages(rewards, mode="dr_grpo")
    assert sum(dr) == pytest.approx(0.0, abs=1e-9)
    assert max(abs(value) for value in dr) < 1.0
    assert any(abs(a - b) > 1e-9 for a, b in zip(plain, dr))


def test_grpo_advantages_of_equal_rewards_are_zero():
    assert grpo_advantages([0.3, 0.3, 0.3]) == [0.0, 0.0, 0.0]


def test_dapo_filters_groups_without_signal():
    groups = [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5]]
    kept, report = dapo_filter_groups(groups)
    assert kept == [1]
    assert report["dropped_all_pass"] == 1
    assert report["dropped_all_fail"] == 2  # [0,0] and [0.5,0.5] carry no signal


def test_dapo_token_loss_weights_by_tokens_not_sequences():
    # two sequences: short one is bad, long one is good; per-sequence mean would say 0.5
    losses = [[1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert dapo_token_loss(losses, [2, 6]) == pytest.approx(2 / 8)


def test_soft_overlong_penalty_is_bounded():
    assert soft_overlong_penalty(100, soft_limit=200, max_length=400) == 0.0
    assert soft_overlong_penalty(300, soft_limit=200, max_length=400) == pytest.approx(-0.5)
    assert soft_overlong_penalty(400, soft_limit=200, max_length=400) == pytest.approx(-1.0)


def test_clip_higher_allows_an_asymmetric_bound():
    ratios = torch.tensor([0.5, 1.0, 1.5])
    clipped = clip_ratios(ratios, low=0.2, high=0.4)
    assert clipped[0].item() == pytest.approx(0.8)
    assert clipped[2].item() == pytest.approx(1.4)
