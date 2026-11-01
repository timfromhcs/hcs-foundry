"""Resource governor.

Single authoritative source of truth for "how much can we safely use right now".
It is deliberately conservative: an unstable desktop is a worse outcome than a
refused configuration.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import psutil

from ..config import Settings


@dataclass
class Budget:
    measured_at: str
    total_ram_bytes: int
    available_ram_bytes: int
    os_reserve_bytes: int
    usable_ram_bytes: int
    unified_memory: bool
    gpu_shared_bytes: int
    gpu_headroom_bytes: int
    usable_gpu_bytes: int | None
    safe_budget_bytes: int
    tier: int
    tier_label: str
    notes: list[str] = field(default_factory=list)

    def to_public(self) -> dict:
        return asdict(self)


def compute_budget(profile: dict, settings: Settings) -> Budget:
    vm = psutil.virtual_memory()
    total = int(vm.total)
    available = int(vm.available)
    reserve = max(settings.os_reserve_min_bytes, int(total * settings.os_reserve_ratio))
    usable_ram = max(0, available - reserve)

    gpu = profile.get("gpu", {}) if profile else {}
    unified = bool(gpu.get("unified_memory"))
    shared = int(gpu.get("shared_gpu_memory_bytes") or 0)
    vulkan_heap = int(
        (profile.get("accelerators", {}) or {}).get("vulkan", {}).get("max_heap_bytes") or 0
    ) if profile else 0
    gpu_cap = max(shared, vulkan_heap)

    notes: list[str] = []
    if unified:
        # An iGPU has no dedicated VRAM; its allocations come from system RAM. When
        # the driver reports no heap size, 50 % of physical RAM is the conventional
        # driver default on Windows and is used as an *estimated* cap only. The real
        # binding constraint remains the OS-safe free RAM envelope.
        estimated_cap = total // 2
        if gpu_cap == 0:
            gpu_cap = estimated_cap
            notes.append("Unified memory: GPU heap cap estimated as 50 % of physical RAM.")
        elif gpu_cap < total * 0.4:
            gpu_cap = max(gpu_cap, estimated_cap)
            notes.append(
                "Unified memory: driver-reported shared memory is smaller than half of RAM; "
                "using the larger value as an estimated GPU heap cap."
            )
        usable_gpu = max(0, gpu_cap - settings.gpu_headroom_bytes) if gpu_cap else None
        if usable_gpu is None:
            notes.append("Unified memory with no reported GPU heap: budgeting against system RAM only.")
            safe = usable_ram
        else:
            safe = min(usable_ram, usable_gpu)
            notes.append(
                "Unified memory: GPU allocations come from system RAM; the budget is the smaller of the "
                "OS-safe RAM envelope and the GPU heap cap."
            )
    else:
        usable_gpu = gpu_cap
        # For a discrete GPU the weights live in VRAM and the rest in RAM, so the
        # combined envelope is additive. Never count shared memory as dedicated.
        safe = usable_ram + gpu_cap if gpu_cap else usable_ram
        if gpu_cap:
            notes.append("Discrete GPU: VRAM and system RAM budgets are additive.")

    tier = int((profile.get("tier") or {}).get("tier", 0)) if profile else 0
    tier_label = str((profile.get("tier") or {}).get("label", "unknown")) if profile else "unknown"

    if safe < 1 * 1024**3:
        notes.append("Less than 1 GiB of safe headroom: inference will likely fail. Close applications first.")

    return Budget(
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_ram_bytes=total,
        available_ram_bytes=available,
        os_reserve_bytes=reserve,
        usable_ram_bytes=usable_ram,
        unified_memory=unified,
        gpu_shared_bytes=gpu_cap,
        gpu_headroom_bytes=settings.gpu_headroom_bytes,
        usable_gpu_bytes=usable_gpu,
        safe_budget_bytes=int(max(0, safe)),
        tier=tier,
        tier_label=tier_label,
        notes=notes,
    )


OVERHEAD_BYTES = 600 * 1024**2  # runtime, context bookkeeping, tokenizer, CUDA/Vulkan buffers


def estimate_fit(
    *,
    model_size_bytes: int,
    kv_bytes_per_token: int | None,
    context: int,
    budget: Budget,
    extra_bytes: int = 0,
) -> dict:
    """Return a *range* with a confidence label, never a single fake-precise number."""
    kv = int((kv_bytes_per_token or 0) * context)
    core = int(model_size_bytes) + kv + OVERHEAD_BYTES + int(extra_bytes)
    low = int(core * 1.0)
    high = int(core * 1.25)  # allocator slack, fragmentation, driver overhead

    safe = budget.safe_budget_bytes
    if high <= safe * 0.8:
        status, confidence = "comfortable", "high"
    elif high <= safe:
        status, confidence = "tight", "medium"
    elif low <= safe:
        status, confidence = "risky", "low"
    else:
        status, confidence = "too_large", "high"

    return {
        "status": status,
        "confidence": confidence,
        "estimated_bytes_low": low,
        "estimated_bytes_high": high,
        "weights_bytes": int(model_size_bytes),
        "kv_cache_bytes": kv,
        "overhead_bytes": OVERHEAD_BYTES,
        "context": context,
        "safe_budget_bytes": safe,
        "headroom_bytes": safe - high,
        "notes": list(budget.notes),
    }


def downshift_plan(
    *,
    model_size_bytes: int,
    kv_bytes_per_token: int | None,
    context: int,
    budget: Budget,
) -> list[dict]:
    """Ordered, monotonic suggestions when a configuration does not fit."""
    plan: list[dict] = []
    safe = max(1, budget.safe_budget_bytes)
    ctx = context
    for target in (context // 2, context // 4, 2048, 1024):
        if target < 512 or target >= ctx:
            continue
        ctx = target
        fit = estimate_fit(
            model_size_bytes=model_size_bytes,
            kv_bytes_per_token=kv_bytes_per_token,
            context=ctx,
            budget=budget,
        )
        plan.append({"action": "reduce_context", "context": ctx, "fit": fit["status"]})
        if fit["status"] in {"comfortable", "tight"}:
            break
    plan.append({"action": "reduce_gpu_layers", "note": "Offload fewer layers; the rest runs on CPU."})
    plan.append({"action": "use_smaller_quantization", "note": "Prefer Q4_K_M over Q5/Q6/Q8 when memory-bound."})
    plan.append({"action": "close_applications", "note": "Free RAM before loading."})
    if model_size_bytes > safe:
        plan.insert(0, {"action": "choose_smaller_model", "note": "Weights alone exceed the safe budget."})
    return plan
