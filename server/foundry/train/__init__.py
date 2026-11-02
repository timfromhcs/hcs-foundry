"""The unified training contract (v8).

Import surface:

* ``spec`` — the declarative TrainSpec every trainer accepts,
* ``manifest`` — RunManifest v2 and ``from_legacy`` for the pre-v8 shapes,
* ``stats`` — the paired statistics used by every gate.

The trainers themselves stay in ``foundry.training`` and are wrapped, not rewritten; see
``PLAN.md`` §6.
"""

from .stats import PairedResult, paired_compare, pass_at_k, required_n, summarize, wilson_ci

__all__ = [
    "PairedResult",
    "paired_compare",
    "pass_at_k",
    "required_n",
    "summarize",
    "wilson_ci",
]
