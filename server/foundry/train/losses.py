"""Pure loss functions — the math behind each method, testable without a model.

Keeping these out of the trainers has two effects: the methods can be checked against known
values (and against TRL when it is importable), and any backend — CPU today, remote later — can
reuse them unchanged.
"""

from __future__ import annotations

import math
from typing import Sequence


def _torch():
    import torch

    return torch


def sft_loss(logits, labels, *, ignore_index: int = -100):
    torch = _torch()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index
    )


def dpo_loss(
    policy_chosen_logp,
    policy_rejected_logp,
    reference_chosen_logp,
    reference_rejected_logp,
    *,
    beta: float = 0.1,
):
    """Direct preference optimisation: ``-log sigmoid(β[(π_w−ref_w) − (π_l−ref_l)])``."""
    torch = _torch()
    chosen = policy_chosen_logp - reference_chosen_logp
    rejected = policy_rejected_logp - reference_rejected_logp
    margin = beta * (chosen - rejected)
    loss = -torch.nn.functional.logsigmoid(margin)
    return loss, margin.detach()


def reverse_kl_loss(student_logprobs, teacher_logprobs):
    """Per-token reverse KL used by on-policy distillation.

    ``KL(π_student ‖ π_teacher) ≈ mean(log π_s − log π_t)`` over the student's own samples;
    exactly zero when the distributions agree on those tokens.
    """
    torch = _torch()
    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError("student and teacher log-probs must have the same shape")
    return (student_logprobs - teacher_logprobs).mean()


def grpo_advantages(rewards: Sequence[float], *, mode: str = "grpo") -> list[float]:
    """Group-relative advantages.

    ``grpo``: ``(r − mean) / std`` per group. ``dr_grpo``: no std normalisation (Dr. GRPO), which
    removes the length bias the std term introduces.
    """
    values = [float(value) for value in rewards]
    if not values:
        return []
    n = len(values)
    mean = sum(values) / n
    centered = [value - mean for value in values]
    if mode == "dr_grpo":
        return centered
    if mode != "grpo":
        raise ValueError(f"unknown advantage mode: {mode}")
    variance = sum(value * value for value in centered) / n
    std = math.sqrt(variance)
    if std < 1e-6:
        return [0.0 for _ in values]
    return [value / std for value in centered]


def dapo_filter_groups(groups: Sequence[Sequence[float]]) -> tuple[list[int], dict]:
    """Drop groups whose rewards are all equal — they carry no gradient.

    Returns ``(kept_indices, report)``. If every group is dropped the caller must report
    "no signal" instead of training on zeros.
    """
    kept: list[int] = []
    all_pass = all_fail = 0
    for index, group in enumerate(groups):
        values = [float(value) for value in group]
        if not values:
            continue
        low, high = min(values), max(values)
        if high - low < 1e-9:
            if high >= 1.0:
                all_pass += 1
            else:
                all_fail += 1
            continue
        kept.append(index)
    report = {
        "groups": len(groups),
        "kept": len(kept),
        "dropped_all_pass": all_pass,
        "dropped_all_fail": all_fail,
    }
    return kept, report


def dapo_token_loss(token_losses: Sequence[Sequence[float]], lengths: Sequence[int]) -> float:
    """Sum over tokens divided by the total token count (not a per-sequence mean)."""
    total = 0.0
    tokens = 0
    for losses, length in zip(token_losses, lengths):
        total += sum(float(value) for value in losses)
        tokens += max(0, int(length))
    return total / tokens if tokens else 0.0


def soft_overlong_penalty(length: int, *, soft_limit: int, max_length: int) -> float:
    """0 below the soft limit, linear down to −1 at the hard maximum."""
    if max_length <= soft_limit:
        return 0.0
    if length <= soft_limit:
        return 0.0
    fraction = (min(length, max_length) - soft_limit) / (max_length - soft_limit)
    return -1.0 * fraction


def clip_ratios(ratios, *, low: float, high: float):
    """DAPO clip-higher: the upper bound is allowed to exceed the lower one."""
    torch = _torch()
    if low <= 0 or high <= 0:
        raise ValueError("clip bounds must be positive")
    return torch.clamp(ratios, 1.0 - low, 1.0 + high)
