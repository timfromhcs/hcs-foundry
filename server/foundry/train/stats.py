"""Statistics for evaluation gates — pure Python, no torch.

The product's central promise is "tested to improve", and until v7 the only gate was
``delta >= 0.05`` on suites with 3–6 cases. At n = 6 a single flipped case is 16.7 pp, so that
gate could only ever react to noise. This module is the replacement:

* ``wilson_ci`` — interval for a pass rate that behaves at small n and at 0/1,
* ``paired_compare`` — paired difference between two runs on the *same* items, with an exact
  McNemar test for binary outcomes and a clustered bootstrap otherwise,
* ``required_n`` — how many items a target effect actually needs (power planning),
* ``pass_at_k`` — the unbiased estimator for k-sample evaluation.

Design notes
------------
Paired beats unpaired because the shared difficulty of an item cancels. Clustering matters:
items generated from the same template are not independent, and ignoring that inflates the
apparent sample size (Miller, "Adding Error Bars to Evals", arXiv 2411.00640). Hence ``clusters``
defaults to per-item identity, but callers pass ``template_id`` when they have it.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence


def _z(alpha: float) -> float:
    """Two-sided normal quantile for ``alpha`` without scipy (Acklam's approximation)."""
    p = 1.0 - alpha / 2.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    lower, upper = 0.02425, 1 - 0.02425
    if p < lower:
        q = math.sqrt(-2 * math.log(p))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        return numerator / denominator
    if p > upper:
        q = math.sqrt(-2 * math.log(1 - p))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        return -(numerator / denominator)
    q = p - 0.5
    r = q * q
    numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    denominator = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return numerator / denominator


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes out of ``n`` (never returns NaN at 0 or n)."""
    if n <= 0:
        return (0.0, 1.0)
    k = max(0, min(int(k), int(n)))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class PairedResult:
    delta: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    n_clusters: int
    n_discordant: int
    method: str
    underpowered: bool
    required_n: int | None

    def improved(self, *, min_effect: float = 0.0) -> bool:
        return self.ci_low > 0 and self.delta >= min_effect

    def to_public(self) -> dict:
        return {
            "delta": round(self.delta, 4),
            "se": round(self.se, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "p_value": round(self.p_value, 5),
            "n": self.n,
            "n_clusters": self.n_clusters,
            "n_discordant": self.n_discordant,
            "method": self.method,
            "underpowered": self.underpowered,
            "required_n": self.required_n,
        }


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for ``b`` wins and ``c`` losses."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def required_n(delta: float, discordance: float, *, alpha: float = 0.05, power: float = 0.80) -> int:
    """Items needed to detect ``delta`` at ``alpha``/``power`` given a discordance rate.

    ``n ≈ (z_{α/2} + z_{β})² · d / Δ²`` — the standard paired-binary approximation.
    ``required_n(0.10, 0.20) == 157`` and ``required_n(0.05, 0.20) == 628``.
    """
    if delta <= 0:
        return 10**9
    d = max(1e-6, min(1.0, float(discordance)))
    z_alpha = abs(_z(alpha))
    z_beta = abs(_z(2 * power - 1)) if power < 1 else 1.0
    # z_{1-power} for the usual one-sided power term
    z_beta = abs(_z(2 * (1 - power))) if 0 < power < 1 else 0.8416212335729143
    value = (z_alpha + z_beta) ** 2 * d / (delta**2)
    return int(math.ceil(value - 1e-9))


def _cluster_bootstrap(
    diffs: Sequence[float],
    clusters: Sequence[Hashable],
    *,
    alpha: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Cluster bootstrap of the mean difference → (se, ci_low, ci_high)."""
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for value, key in zip(diffs, clusters):
        grouped[key].append(value)
    keys = list(grouped)
    means = [sum(grouped[key]) / len(grouped[key]) for key in keys]
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(max(100, n_boot)):
        sample = [means[rng.randrange(len(means))] for _ in means]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    overall = sum(diffs) / len(diffs)
    se = (sum((value - overall) ** 2 for value in diffs) / max(1, len(diffs) - 1)) ** 0.5 / math.sqrt(len(diffs))
    if len(keys) > 1:
        cluster_means = means
        cm = sum(cluster_means) / len(cluster_means)
        cluster_var = sum((value - cm) ** 2 for value in cluster_means) / (len(cluster_means) - 1)
        se = math.sqrt(cluster_var / len(cluster_means))
    lo_idx = int((alpha / 2) * len(boot))
    hi_idx = min(len(boot) - 1, int((1 - alpha / 2) * len(boot)))
    return se, boot[lo_idx], boot[hi_idx]


def paired_compare(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    clusters: Sequence[Hashable] | None = None,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
    target_delta: float | None = None,
) -> PairedResult:
    """Compare two runs item by item; positive ``delta`` means better than baseline.

    Binary per-item scores use an exact McNemar test on the discordant pairs (the default in
    eval practice); continuous scores use a paired bootstrap, clustered when ``clusters`` is
    given. ``underpowered`` is set when the observed discordance implies a required n above the
    sample size — that flag is what stops the gate from promoting noise.
    """
    if len(treatment) != len(baseline):
        raise ValueError("paired comparison needs the same number of items in both runs")
    n = len(treatment)
    if n == 0:
        raise ValueError("paired comparison needs at least one item")
    diffs = [float(a) - float(b) for a, b in zip(treatment, baseline)]
    delta = sum(diffs) / n
    keys = list(clusters) if clusters is not None else list(range(n))
    binary = all(value in (0.0, 1.0) for value in list(treatment) + list(baseline))

    if clusters is not None:
        discordant = sum(1 for value in diffs if abs(value) > 1e-12)
        method = "paired_bootstrap_clustered"
        se, ci_low, ci_high = _cluster_bootstrap(diffs, keys, alpha=alpha, n_boot=n_boot, seed=seed)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(delta) / (se * math.sqrt(2))))) if se > 0 else 1.0
    elif binary:
        b = sum(1 for a, base in zip(treatment, baseline) if a > base)
        c = sum(1 for a, base in zip(treatment, baseline) if a < base)
        discordant = b + c
        method = "mcnemar_exact"
        p_value = _mcnemar_exact(b, c)
        rate = discordant / n
        se = math.sqrt(max(1e-12, rate) / n)
        z = abs(_z(alpha))
        ci_low, ci_high = delta - z * se, delta + z * se
    else:
        discordant = sum(1 for value in diffs if abs(value) > 1e-12)
        method = "paired_bootstrap"
        se, ci_low, ci_high = _cluster_bootstrap(diffs, keys, alpha=alpha, n_boot=n_boot, seed=seed)
        if se > 0:
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(delta) / (se * math.sqrt(2)))))
        else:
            p_value = 0.0 if abs(delta) > 0 else 1.0

    discordance = discordant / n
    target = target_delta if target_delta is not None else max(abs(delta), 1e-9)
    need = required_n(target, discordance, alpha=alpha) if target > 0 else None
    underpowered = bool(need is not None and n < need and ci_low <= 0)
    return PairedResult(
        delta=delta,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n=n,
        n_clusters=len(set(keys)),
        n_discordant=discordant,
        method=method,
        underpowered=underpowered,
        required_n=need,
    )


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: probability that at least one of ``k`` samples passes."""
    if n <= 0 or k <= 0 or c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if k >= n:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(items: Sequence[float], *, alpha: float = 0.05) -> dict:
    """Pass rate with a Wilson interval — the number a report should quote."""
    n = len(items)
    k = sum(1 for value in items if value >= 0.5)
    low, high = wilson_ci(k, n)
    return {
        "n": n,
        "passed": k,
        "rate": round(k / n, 4) if n else 0.0,
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "alpha": alpha,
    }
