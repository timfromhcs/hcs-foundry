"""All-in-one pipeline: pull → dataset → train → merge → export → quantize → MoE → report.

Stages are explicit, ordered and resumable. A stage that cannot run is skipped with a
recorded reason; it is never simulated. Every stage writes its output into a JSON report
that can be exported as Markdown.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Settings
from .db import Database, new_id, utc_now


@dataclass
class PipelineState:
    run_id: str
    started_at: str
    stages: list[dict] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_public(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.artifacts.get("finished_at"),
            "stages": self.stages,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }


def merge_adapter(base_dir: str | Path, adapter_dir: str | Path, out_dir: str | Path, *, log=None) -> dict:
    """Merge a PEFT adapter into the base weights and save a plain HF directory."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"training stack not importable: {exc}") from exc

    base = Path(base_dir).expanduser()
    adapter = Path(adapter_dir).expanduser()
    target = Path(out_dir).expanduser()
    if not (adapter / "adapter_config.json").exists():
        raise ValueError(f"not a PEFT adapter directory: {adapter}")
    if log:
        log(f"merging {adapter.name} into {base.name}")
    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    merged = model.merge_and_unload()
    target.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(target), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(target))
    size = sum(p.stat().st_size for p in target.glob("*") if p.is_file())
    return {"output": str(target), "size_bytes": size}


def run_pipeline(
    db: Database,
    settings: Settings,
    *,
    stages: list[str],
    options: dict,
    progress: Callable[[float, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancel=None,
) -> dict:
    state = PipelineState(run_id=new_id("pipe"), started_at=utc_now())
    total = max(1, len(stages))

    def record(name: str, status: str, detail: dict) -> None:
        state.stages.append({"stage": name, "status": status, "at": utc_now(), **detail})
        if log:
            log(f"[{name}] {status}")

    for index, stage in enumerate(stages):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("pipeline cancelled")
        if progress:
            progress(index / total * 100, f"stage {stage}")
        try:
            if stage == "pull":
                result = _stage_pull(db, settings, options, log=log)
            elif stage == "dataset":
                result = _stage_dataset(db, settings, options, progress=progress, cancel=cancel, log=log)
            elif stage == "train":
                result = _stage_train(db, settings, options, progress=progress, cancel=cancel, log=log)
            elif stage == "merge":
                result = _stage_merge(options, log=log)
            elif stage == "export":
                result = _stage_export(options, log=log, cancel=cancel)
            elif stage == "quantize":
                result = _stage_quantize(options, log=log, cancel=cancel)
            elif stage == "moe":
                result = _stage_moe(options, progress=progress, log=log)
            elif stage == "report":
                result = {"report": str(_write_report(state, options, settings))}
            else:
                raise ValueError(f"unknown stage: {stage}")
            state.artifacts[stage] = result
            record(stage, "ok", {"result": result})
        except Exception as exc:  # noqa: BLE001 - each stage is isolated
            record(stage, "failed", {"error": str(exc)[:500]})
            if not options.get("continue_on_error"):
                break

    state.artifacts["finished_at"] = utc_now()
    report = _write_report(state, options, settings)
    state.artifacts["report"] = str(report)
    if progress:
        progress(100, "pipeline finished")
    return state.to_public()


def _stage_pull(db: Database, settings: Settings, options: dict, *, log=None) -> dict:
    from . import hf

    repo_id = options.get("repo_id")
    if not repo_id:
        raise ValueError("pull needs repo_id")
    token = hf.get_token(settings.state_dir)
    files = hf.list_files(repo_id, token=token)
    meta = {f["path"]: f for f in files}
    selected = options.get("files") or hf.pick_default_files(files)
    dest = settings.data_dir / "models" / hf.sanitize_repo_dir(repo_id)
    result = hf.download_many(repo_id, selected, dest, token=token, files_meta=meta, progress=lambda p, m: log(m) if log else None)
    return {"dest": str(dest), "files": [f["path"] for f in result["files"]]}


def _stage_dataset(db: Database, settings: Settings, options: dict, *, progress=None, cancel=None, log=None) -> dict:
    from .data.sources import import_source

    providers = options.get("sources") or []
    if not providers:
        raise ValueError("dataset needs at least one source provider")
    results = []
    for provider in providers:
        results.append(import_source(db, settings, provider, rights_status=options.get("rights_status", "needs_review"), progress=progress, cancel=cancel))
    return {"datasets": [{"dataset_id": r["dataset_id"], "records": r["record_count"], "source": r["source"]["provider"]} for r in results]}


def _stage_train(db: Database, settings: Settings, options: dict, *, progress=None, cancel=None, log=None) -> dict:
    from .training import SFTConfig, run_auto

    dataset_id = options.get("dataset_id")
    model_path = options.get("model_path")
    if not dataset_id or not model_path:
        raise ValueError("train needs dataset_id and model_path")
    config = SFTConfig(
        model_path=model_path,
        dataset_id=dataset_id,
        max_steps=int(options.get("max_steps", 20)),
        max_seq_len=int(options.get("max_seq_len", 256)),
        auto=True,
        eval_every=int(options.get("eval_every", 5)),
        patience=int(options.get("patience", 3)),
    )
    output = settings.training_dir / f"pipe_{int(time.time())}"
    result = run_auto(db, config, job_id=None, output_dir=output, progress=progress, log=log, cancel=cancel)
    options["adapter_dir"] = result["manifest"]["adapter_dir"]
    options["train_run_id"] = result["run_id"]
    return {"run_id": result["run_id"], "adapter": result["manifest"]["adapter_dir"], "best_eval_loss": result["manifest"].get("best_eval_loss")}


def _stage_merge(options: dict, *, log=None) -> dict:
    if not options.get("adapter_dir") or not options.get("model_path"):
        raise ValueError("merge needs model_path and adapter_dir (from train)")
    out = Path(options.get("merged_dir") or (Path(options["model_path"]).parent / "merged"))
    result = merge_adapter(options["model_path"], options["adapter_dir"], out, log=log)
    options["merged_dir"] = result["output"]
    return result


def _stage_export(options: dict, *, log=None, cancel=None) -> dict:
    from .config import load_settings
    from .gguf_tools import discover, export_gguf

    source = options.get("merged_dir") or options.get("model_path")
    if not source:
        raise ValueError("export needs a merged or base model directory")
    settings = load_settings()
    toolchain = discover(settings)
    out = Path(options.get("gguf_out") or (settings.data_dir / "gguf" / f"{Path(source).name}-f16.gguf"))
    result = export_gguf(source, out, toolchain=toolchain, dtype=options.get("dtype", "f16"), log=log, cancel=cancel)
    options["gguf_f16"] = result["output"]
    return result


def _stage_quantize(options: dict, *, log=None, cancel=None) -> dict:
    from .config import load_settings
    from .gguf_tools import discover, quantize_gguf

    source = options.get("gguf_f16")
    if not source:
        raise ValueError("quantize needs gguf_f16 (from export)")
    settings = load_settings()
    toolchain = discover(settings)
    method = options.get("quant", "Q4_K_M")
    out = Path(options.get("quant_out") or (Path(source).parent / f"{Path(source).stem}-{method}.gguf"))
    result = quantize_gguf(source, out, method, toolchain=toolchain, log=log, cancel=cancel)
    options["gguf_quant"] = result["output"]
    return result


def _stage_moe(options: dict, *, progress=None, log=None) -> dict:
    from .moe import MoEConfig, moeify

    source = options.get("model_path")
    if not source:
        raise ValueError("moe needs model_path")
    config = MoEConfig(experts=int(options.get("experts", 4)), top_k=int(options.get("top_k", 2)))
    out = Path(options.get("moe_dir") or (Path(source).parent / f"{Path(source).name}-moe"))
    result = moeify(source, out, config=config, progress=progress, log=log, force=bool(options.get("force")))
    options["moe_dir"] = result["output"]
    return {"output": result["output"], "experts": config.experts, "size_bytes": result["size_bytes"]}


def _write_report(state: PipelineState, options: dict, settings: Settings) -> Path:
    target = Path(options.get("report_path") or (settings.data_dir / "reports" / f"{state.run_id}.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_public()
    payload["options"] = {k: v for k, v in options.items() if isinstance(v, (str, int, float, bool, list))}
    target.write_text(json.dumps(payload, indent=2), "utf-8")
    markdown = target.with_suffix(".md")
    lines = [f"# Pipeline {state.run_id}", "", f"Started: {state.started_at}", ""]
    lines.append("| Stage | Status | Detail |")
    lines.append("| --- | --- | --- |")
    for stage in state.stages:
        detail = stage.get("error") or stage.get("result") or ""
        lines.append(f"| {stage['stage']} | {stage['status']} | {str(detail)[:160].replace('|', '/')} |")
    lines.append("")
    lines.append("_Local pipeline report. Failed stages are listed with their reason; nothing is simulated._")
    markdown.write_text("\n".join(lines), "utf-8")
    return target
