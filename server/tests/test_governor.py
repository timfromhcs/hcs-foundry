from __future__ import annotations

from foundry.config import Settings
from foundry.machine.governor import Budget, compute_budget, downshift_plan, estimate_fit


def _profile(unified: bool = True, shared: int = 4 * 1024**3) -> dict:
    return {
        "gpu": {
            "unified_memory": unified,
            "shared_gpu_memory_bytes": shared,
            "devices": [],
        },
        "accelerators": {"vulkan": {"status": "detected", "max_heap_bytes": None}},
        "tier": {"tier": 1, "label": "Tier 1"},
    }


def test_shared_memory_is_never_dedicated(settings: Settings):
    budget = compute_budget(_profile(), settings)
    assert budget.unified_memory is True
    assert budget.gpu_headroom_bytes > 0
    # The safe budget must be bounded by the OS reserve, not by the reported "VRAM".
    assert budget.safe_budget_bytes <= budget.available_ram_bytes
    assert budget.safe_budget_bytes < budget.total_ram_bytes


def test_os_reserve_is_applied(settings: Settings):
    settings.os_reserve_min_bytes = 3 * 1024**3
    budget = compute_budget(_profile(), settings)
    assert budget.os_reserve_bytes >= 3 * 1024**3
    assert budget.usable_ram_bytes <= budget.available_ram_bytes - 3 * 1024**3 + 1


def test_fit_statuses_are_monotonic():
    budget = Budget(
        measured_at="now",
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        os_reserve_bytes=2 * 1024**3,
        usable_ram_bytes=6 * 1024**3,
        unified_memory=True,
        gpu_shared_bytes=8 * 1024**3,
        gpu_headroom_bytes=1 * 1024**3,
        usable_gpu_bytes=7 * 1024**3,
        safe_budget_bytes=6 * 1024**3,
        tier=1,
        tier_label="Tier 1",
    )
    small = estimate_fit(model_size_bytes=1 * 1024**3, kv_bytes_per_token=100_000, context=2048, budget=budget)
    large = estimate_fit(model_size_bytes=8 * 1024**3, kv_bytes_per_token=100_000, context=8192, budget=budget)
    assert small["status"] == "comfortable"
    assert large["status"] == "too_large"
    assert small["estimated_bytes_high"] > small["estimated_bytes_low"]


def test_downshift_plan_offers_context_reductions():
    budget = Budget(
        measured_at="now",
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        os_reserve_bytes=2 * 1024**3,
        usable_ram_bytes=6 * 1024**3,
        unified_memory=True,
        gpu_shared_bytes=8 * 1024**3,
        gpu_headroom_bytes=1 * 1024**3,
        usable_gpu_bytes=7 * 1024**3,
        safe_budget_bytes=2 * 1024**3,
        tier=1,
        tier_label="Tier 1",
    )
    plan = downshift_plan(model_size_bytes=4 * 1024**3, kv_bytes_per_token=100_000, context=8192, budget=budget)
    actions = [step["action"] for step in plan]
    assert "choose_smaller_model" in actions
    assert "reduce_context" in actions
