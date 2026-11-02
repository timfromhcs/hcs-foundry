"""RunManifest v2 — one shape for every training run, with a bridge from the old ones.

The three legacy trainers wrote three different manifests (``lora_sft``, ``int4_lora_sft``,
``dpo``/``distill``). Downstream code (UI, MCP, gates) had to know all of them. v2 keeps the
superset and puts method-specific data in ``extras``; ``from_legacy`` maps what exists today so
old run directories stay readable.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .spec import TrainSpec

MANIFEST_VERSION = 2


class EnvInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python: str
    platform: str
    torch: str | None = None
    transformers: str | None = None
    peft: str | None = None
    threads: int | None = None
    device: str = "cpu"
    git_sha: str | None = None

    @classmethod
    def collect(cls, *, threads: int | None = None, device: str = "cpu") -> "EnvInfo":
        def version(name: str) -> str | None:
            try:
                return md.version(name)
            except md.PackageNotFoundError:
                return None

        sha = None
        try:
            head = Path(__file__).resolve().parents[3] / ".git" / "HEAD"
            if head.exists():
                content = head.read_text("utf-8").strip()
                if content.startswith("ref: "):
                    ref = Path(__file__).resolve().parents[3] / ".git" / content[5:]
                    sha = ref.read_text("utf-8").strip()[:12] if ref.exists() else None
                else:
                    sha = content[:12]
        except OSError:
            sha = None
        return cls(
            python=sys.version.split()[0],
            platform=f"{platform.system()} {platform.release()}",
            torch=version("torch"),
            transformers=version("transformers"),
            peft=version("peft"),
            threads=threads,
            device=device,
            git_sha=sha,
        )


class MemInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    peak_rss_mb: float | None = None
    start_rss_mb: float | None = None
    private_mb: float | None = None
    min_available_mb: float | None = None


class IoInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passes: int | None = None
    bytes_read_store: int | None = None
    seq_read_mbps: float | None = None
    micro_batch: int | None = None


class EvalPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    loss: float


class RunManifest(BaseModel):
    """The superset of everything the trainers record, plus a stable identity."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal[2] = MANIFEST_VERSION
    run_id: str
    status: Literal["completed", "cancelled", "failed", "resumed"] = "completed"
    failure: str | None = None

    spec: TrainSpec | None = None
    spec_hash: str | None = None
    legacy_kind: str | None = None

    env: EnvInfo
    base_model: str
    base_model_sha256: str | None = None
    dataset_id: str | None = None
    dataset_manifest_hash: str | None = None
    adapter_dir: str | None = None
    adapter_sha256: str | None = None

    steps: int = 0
    tokens_seen: int = 0
    wall_seconds: float = 0.0
    losses: list[float] = Field(default_factory=list)
    eval_history: list[EvalPoint] = Field(default_factory=list)
    param_delta_max: float | None = None
    grads_finite: bool | None = None

    memory: MemInfo = Field(default_factory=MemInfo)
    io: IoInfo | None = None
    tuning: dict[str, Any] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(default_factory=dict)


def _memory_from(legacy: dict) -> MemInfo:
    memory = legacy.get("memory") or {}
    resources = legacy.get("resources") or {}
    return MemInfo(
        peak_rss_mb=memory.get("peak_rss_mb"),
        start_rss_mb=memory.get("start_rss_mb"),
        private_mb=memory.get("peak_private_mb"),
        min_available_mb=memory.get("min_available_mb"),
    )


def from_legacy(manifest: dict, *, run_id: str | None = None, spec: TrainSpec | None = None) -> RunManifest:
    """Map a pre-v8 manifest (``lora_sft`` / ``int4_lora_sft`` / ``dpo`` / ``distill``) to v2.

    Unknown keys are preserved in ``extras`` instead of being dropped — a reader can always see
    what the old run recorded.
    """
    kind = str(manifest.get("kind") or "unknown")
    known = {
        "kind",
        "base_model",
        "base_model_hash",
        "dataset",
        "config",
        "steps",
        "duration_seconds",
        "loss_first",
        "loss_last",
        "losses",
        "eval_history",
        "best_eval_loss",
        "best_eval_step",
        "adapter_dir",
        "best_adapter_dir",
        "lora_parameters",
        "lora_trainable_params",
        "converted_layers",
        "target_modules",
        "quantization",
        "streaming",
        "offload",
        "vram",
        "prefetch",
        "tuning",
        "resources",
        "memory",
        "note",
        "finished_at",
        "teacher",
        "pairs",
        "param_delta_max",
        "grads_finite",
        "tokens_seen",
    }
    extras = {key: value for key, value in manifest.items() if key not in known}
    dataset = manifest.get("dataset") or {}
    losses: list[float] = [float(value) for value in (manifest.get("losses") or []) if value is not None]
    if not losses:
        for key in ("loss_first", "loss_last"):
            if manifest.get(key) is not None:
                losses.append(float(manifest[key]))
    eval_history = [
        EvalPoint(step=int(point.get("step") or 0), loss=float(point.get("eval_loss") or 0.0))
        for point in (manifest.get("eval_history") or [])
        if isinstance(point, dict)
    ]
    config = manifest.get("config") or {}
    return RunManifest(
        run_id=run_id or str(manifest.get("run_id") or "legacy"),
        status="completed",
        spec=spec,
        spec_hash=spec.spec_hash if spec else None,
        legacy_kind=kind,
        env=EnvInfo.collect(threads=(manifest.get("tuning") or {}).get("torch_threads_used")),
        base_model=str(manifest.get("base_model") or config.get("model_path") or ""),
        base_model_sha256=manifest.get("base_model_hash"),
        dataset_id=dataset.get("dataset_id") or config.get("dataset_id"),
        dataset_manifest_hash=dataset.get("dataset_hash"),
        adapter_dir=manifest.get("adapter_dir"),
        steps=int(manifest.get("steps") or 0),
        tokens_seen=int(manifest.get("tokens_seen") or 0),
        wall_seconds=float(manifest.get("duration_seconds") or 0.0),
        losses=losses,
        eval_history=eval_history,
        param_delta_max=manifest.get("param_delta_max"),
        grads_finite=manifest.get("grads_finite"),
        memory=_memory_from(manifest),
        tuning=dict(manifest.get("tuning") or {}),
        extras={
            **extras,
            "converted_layers": manifest.get("converted_layers"),
            "lora_parameters": manifest.get("lora_parameters") or manifest.get("lora_trainable_params"),
            "quantization": manifest.get("quantization"),
            "offload": manifest.get("offload"),
            "vram": manifest.get("vram"),
            "prefetch": manifest.get("prefetch"),
            "note": manifest.get("note"),
            "best_eval_loss": manifest.get("best_eval_loss"),
            "best_adapter_dir": manifest.get("best_adapter_dir"),
        },
    )
