"""Command line interface for HCS Foundry.

Every command works without starting the HTTP service. ``--json`` prints a
machine-readable result; failures exit non-zero with a precise message.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import APP_NAME, VERSION, apply_overrides, load_settings, repo_root
from .db import init_db
from .events import EventBus
from .jobs import JobRunner
from .runtime.llama_server import LoadConfig, LlamaServer, RuntimeFailure
from .security import PairingStore
from .state import AppState


def _fail(message: str, code: int = 1) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def build_state() -> AppState:
    settings = load_settings()
    db = init_db(settings.db_path)
    apply_overrides(settings, db.get_setting("settings"))
    settings.ensure_dirs()
    bus = EventBus()
    runner = JobRunner(db, bus, allow_concurrent_heavy=settings.allow_concurrent_heavy_jobs)
    server = LlamaServer(settings.state_dir)
    pairing = PairingStore(settings.state_dir)
    return AppState(settings=settings, db=db, bus=bus, runner=runner, server=server, pairing=pairing)


def _print(data, as_json: bool, human: str | None = None) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(human if human is not None else json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# serve / scan / info / token
# ---------------------------------------------------------------------------

def cmd_serve(args) -> int:
    import uvicorn

    from .main import create_app

    settings = load_settings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if not settings.allow_lan and settings.host not in {"127.0.0.1", "::1", "localhost"}:
        return _fail("refusing to bind to a non-loopback host without FOUNDRY_ALLOW_LAN=1", 2)
    app = create_app(settings)
    print(f"{APP_NAME} {VERSION}")
    print(f"  API:  http://{settings.host}:{settings.port}/api/v1")
    print(f"  UI:   http://{settings.host}:{settings.port}/" if (repo_root() / "web" / "dist").exists() else "  UI:   npm run dev in web/")
    print(f"  Data: {settings.data_dir}")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())
    return 0


def cmd_scan(args) -> int:
    from .machine import compute_budget, scan_machine

    settings = load_settings()
    profile = scan_machine(settings)
    budget = compute_budget(profile, settings)
    if args.write:
        out = Path(args.output) if args.output else repo_root() / "docs" / "audit" / "MACHINE_PROFILE.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(profile, indent=2), "utf-8")
        if not args.json:
            print(f"wrote {out}")
    if args.json:
        _print({"profile": profile, "budget": budget.to_public()}, True)
        return 0
    print(f"OS:    {profile['os'].get('name')} {profile['os'].get('version')}")
    print(f"CPU:   {profile['cpu'].get('model')} ({profile['cpu'].get('physical_cores')}c/{profile['cpu'].get('logical_cores')}t)")
    ram = profile["memory"]
    print(f"RAM:   {ram['total_bytes'] / 1024**3:.1f} GiB total, {ram['available_bytes'] / 1024**3:.1f} GiB free")
    for device in profile["gpu"]["devices"]:
        print(
            f"GPU:   {device['name']} — dedicated VRAM {int(device.get('dedicated_vram_bytes') or 0)} B, "
            f"shared {int(device.get('shared_gpu_memory_bytes') or 0) / 1024**3:.1f} GiB, unified={device.get('unified_memory')}"
        )
    for name, accel in profile["accelerators"].items():
        print(f"ACCEL: {name}: {accel.get('status')}")
    for server in profile["runtimes"]["llama_server"]["candidates"]:
        print(f"RUNTIME: {server['path']} ({server.get('version')}, {server.get('accelerator')}, {server.get('status')})")
    print(f"TIER:  {profile['tier']['label']} ({profile['tier']['confidence']})")
    print(f"SAFE BUDGET: {budget.safe_budget_bytes / 1024**3:.2f} GiB")
    for warning in profile["warnings"]:
        print(f"WARN:  {warning}")
    return 0


def cmd_info(args) -> int:
    settings = load_settings()
    data = {
        "app": APP_NAME,
        "version": VERSION,
        "data_dir": str(settings.data_dir),
        "db": str(settings.db_path),
        "model_roots": [str(p) for p in settings.model_roots],
        "import_roots": [str(p) for p in settings.import_roots],
        "ui_origins": settings.ui_origins,
    }
    _print(data, args.json, "\n".join(f"{k}: {v}" for k, v in data.items()))
    return 0


def cmd_token(args) -> int:
    settings = load_settings()
    store = PairingStore(settings.state_dir)
    token = store.issue(args.label)
    _print({"token": token, "label": args.label}, args.json, token)
    return 0


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

def _hf_token(state: AppState) -> str | None:
    from . import hf

    return hf.get_token(state.settings.state_dir)


def cmd_hf_search(args) -> int:
    from . import hf

    state = build_state()
    try:
        results = hf.search_models(args.query, task=args.task, limit=args.limit, token=_hf_token(state), sort=args.sort)
    except hf.HFError as exc:
        return _fail(str(exc))
    if args.json:
        _print(results, True)
        return 0
    for item in results:
        print(f"{item['id']:<60} ↓{item['downloads']:<10} ♥{item['likes']:<6} {item['pipeline_tag'] or ''}")
    return 0


def cmd_hf_files(args) -> int:
    from . import hf

    state = build_state()
    try:
        files = hf.list_files(args.repo, token=_hf_token(state))
    except hf.HFError as exc:
        return _fail(str(exc))
    if args.json:
        _print(files, True)
        return 0
    for item in files:
        print(f"{item['size']:>14,}  {item['path']}")
    print(f"suggested: {', '.join(hf.pick_default_files(files))}")
    return 0


def cmd_hf_pull(args) -> int:
    from . import hf

    state = build_state()
    token = _hf_token(state)
    try:
        files = hf.list_files(args.repo, token=token)
    except hf.HFError as exc:
        return _fail(str(exc))
    meta = {f["path"]: f for f in files}
    selected = args.file or hf.pick_default_files(files)
    missing = [name for name in selected if name not in meta]
    if missing:
        return _fail(f"files not found in repository: {', '.join(missing)}")
    dest_root = Path(args.dest).expanduser() if args.dest else state.settings.data_dir / "models"
    dest = dest_root / hf.sanitize_repo_dir(args.repo)

    def progress(pct: float, message: str) -> None:
        if not args.json:
            print(f"\r{pct:5.1f}% {message[:100]:<100}", end="", flush=True)

    try:
        result = hf.download_many(args.repo, selected, dest, token=token, files_meta=meta, progress=progress)
    except (hf.HFError, InterruptedError) as exc:
        return _fail(str(exc))
    if not args.json:
        print()
    from .library import scan_models

    scan = scan_models(state.db, state.settings, roots=[dest])
    payload = {**result, "indexed": scan["added"] + scan["updated"], "anonymous": token is None}
    _print(payload, args.json, f"pulled {len(result['files'])} file(s) into {dest} (indexed: {scan['added'] + scan['updated']})")
    return 0


def cmd_hf_token(args) -> int:
    from . import hf

    state = build_state()
    if args.action == "set":
        if not args.token:
            return _fail("--token is required", 2)
        try:
            identity = hf.whoami(args.token)
        except hf.HFError as exc:
            return _fail(str(exc))
        hf.set_token(state.settings.state_dir, args.token)
        _print({"status": hf.token_status(state.settings.state_dir), "identity": identity}, args.json, f"saved token for {identity.get('name')}")
        return 0
    if args.action == "clear":
        hf.set_token(state.settings.state_dir, None)
        _print(hf.token_status(state.settings.state_dir), args.json, "token removed")
        return 0
    status = hf.token_status(state.settings.state_dir)
    _print(status, args.json, f"configured: {status['configured']} ({status['masked'] or 'anonymous'})")
    return 0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def cmd_models_list(args) -> int:
    from .library import list_models

    state = build_state()
    models = list_models(state.db)
    if args.json:
        _print(models, True)
        return 0
    for model in models:
        size = model["size_bytes"] / 1024**3
        print(f"{model['id']}  {model['name'][:44]:<44} {model['quantization'] or '—':<8} {size:6.2f} GB  {model['task']}")
    return 0


def cmd_models_scan(args) -> int:
    from .library import scan_models

    state = build_state()
    roots = [Path(r).expanduser() for r in args.root] if args.root else state.settings.model_roots
    result = scan_models(state.db, state.settings, roots=roots)
    _print(result, args.json, f"{result['discovered']} found · {result['added']} added · {result['updated']} updated · {result['invalid']} invalid")
    return 0


def cmd_models_verify(args) -> int:
    from .library import verify_model

    state = build_state()
    try:
        result = verify_model(state.db, args.model_id)
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    _print(result, args.json, f"{result['sha256']}  {result['size_bytes']} bytes")
    return 0


def _pick_binary(state: AppState) -> str | None:
    profile = state.latest_profile() or {}
    candidates = ((profile.get("runtimes") or {}).get("llama_server") or {}).get("candidates") or []
    verified = [c for c in candidates if c.get("status") == "verified"]
    if not verified:
        return None
    accelerators = profile.get("accelerators") or {}
    for accel in ("cuda", "vulkan", "rocm", "metal"):
        if (accelerators.get(accel) or {}).get("status") == "detected":
            match = [c for c in verified if c.get("accelerator") == accel]
            if match:
                return match[0]["path"]
    return verified[0]["path"]


def cmd_models_load(args) -> int:
    from .library import get_model, model_fit
    from .machine import compute_budget

    state = build_state()
    model = get_model(state.db, args.model_id)
    if not model:
        return _fail(f"model not found: {args.model_id}")
    binary = _pick_binary(state)
    if not binary:
        return _fail("no verified llama-server binary; register one in the UI or add it to settings")
    profile = state.latest_profile() or {}
    fit = model_fit(model, args.context, compute_budget(profile, state.settings))
    if fit["status"] == "too_large" and not args.force:
        return _fail(f"configuration does not fit ({fit['status']}); use --force to override")
    config = LoadConfig(context=args.context, gpu_layers=args.ngl)
    try:
        status = state.server.load(
            model_id=model["id"], model_path=model["path"], binary=binary, config=config, alias=model["name"]
        )
    except RuntimeFailure as exc:
        return _fail(str(exc))
    _print(status, args.json, f"loaded {model['name']} on port {status.get('port')}")
    return 0


def cmd_models_unload(args) -> int:
    state = build_state()
    status = state.server.unload()
    _print(status, args.json, "unloaded")
    return 0


def cmd_models_run(args) -> int:
    from .library import get_model
    from .machine import compute_budget, estimate_fit

    state = build_state()
    model = get_model(state.db, args.model_id)
    if not model:
        return _fail(f"model not found: {args.model_id}")
    binary = _pick_binary(state)
    if not binary:
        return _fail("no verified llama-server binary")
    profile = state.latest_profile() or {}
    budget = compute_budget(profile, state.settings)
    gguf = (model.get("meta") or {}).get("gguf") or {}
    fit = estimate_fit(
        model_size_bytes=int(model["size_bytes"]),
        kv_bytes_per_token=gguf.get("kv_bytes_per_token"),
        context=args.context,
        budget=budget,
    )
    if fit["status"] == "too_large" and not args.force:
        return _fail("configuration does not fit; use --force")
    config = LoadConfig(context=args.context, gpu_layers=args.ngl)
    try:
        state.server.load(model_id=model["id"], model_path=model["path"], binary=binary, config=config, alias=model["name"])
        result = state.server.complete(
            [{"role": "user", "content": args.prompt}], max_tokens=args.max_tokens, temperature=args.temperature
        )
    except RuntimeFailure as exc:
        return _fail(str(exc))
    finally:
        state.server.unload()
    if args.json:
        _print(result, True)
    else:
        answer = result.get("text") or ""
        reasoning = result.get("reasoning") or ""
        if answer.strip():
            print(answer)
        elif reasoning.strip():
            print(reasoning)
            print("\n[output was reasoning-only; increase --max-tokens for a final answer]", file=sys.stderr)
        else:
            print("[no output]", file=sys.stderr)
        metrics = result.get("metrics") or {}
        if metrics:
            print(
                f"\n[{metrics.get('completion_tokens')} tokens · {metrics.get('tokens_per_second')} tok/s · ttft {metrics.get('ttft_seconds')}s]",
                file=sys.stderr,
            )
    return 0


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def cmd_bench(args) -> int:
    from .benchmarks import run_suite

    state = build_state()
    if state.server.status().get("status") != "ready":
        return _fail("no model loaded; run `foundry models load <id>` first")
    result = run_suite(state.db, state.server, args.suite, profile=state.latest_profile())
    aggregate = result["results"]["aggregate"]
    if args.json:
        _print(result, True)
    else:
        print(f"{args.suite}: {aggregate['passed']}/{aggregate['total']} ({aggregate['score']})")
        for case in result["results"]["cases"]:
            print(f"  {case['status']:<8} {case.get('case_id') or case.get('id')}")
    return 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cmd_train_smoke(args) -> int:
    from .training import run_smoke

    result = run_smoke()
    _print({"ok": result.ok, "metrics": result.metrics, "reason": result.reason}, args.json, f"{'passed' if result.ok else 'failed'}: {result.reason}")
    return 0 if result.ok else 1


def cmd_train_preflight(args) -> int:
    from .training import preflight

    state = build_state()
    plan = preflight(state.db, state.settings, dataset_id=args.dataset, base_model=args.model, profile=state.latest_profile())
    if args.json:
        _print(plan, True)
    else:
        print(f"can_start: {plan['can_start']}")
        for blocker in plan["blockers"]:
            print(f"  blocker: {blocker}")
        for warning in plan["warnings"]:
            print(f"  warning: {warning}")
        print(f"  preset: {json.dumps(plan['preset'])}")
    return 0 if plan["can_start"] else 1


def cmd_train_sft(args) -> int:
    from .training import SFTConfig, create_sft_run, estimate_resources

    state = build_state()
    config = SFTConfig(
        model_path=args.model,
        dataset_id=args.dataset,
        max_steps=args.steps,
        max_seq_len=args.seq,
        micro_batch=args.batch,
        grad_accum=args.accum,
        lora_r=args.rank,
        learning_rate=args.lr,
    )
    resources = estimate_resources(args.model, config)
    if not resources["fits"] and not args.force:
        return _fail(f"estimated memory exceeds available RAM: {resources['estimated_high_bytes']} > {resources['available_bytes']}")
    output = Path(args.output).expanduser() if args.output else state.settings.training_dir / f"cli_{int(time.time())}"

    def progress(pct: float, message: str) -> None:
        if not args.json:
            print(f"\r{pct:5.1f}% {message[:90]:<90}", end="", flush=True)

    try:
        result = create_sft_run(state.db, config, job_id=None, output_dir=output, progress=progress, log=None)
    except (ValueError, InterruptedError, RuntimeError) as exc:
        return _fail(str(exc))
    if not args.json:
        print()
    manifest = result["manifest"]
    _print(
        result,
        args.json,
        f"run {result['run_id']}: {manifest['steps']} steps, loss {manifest['loss_first']} → {manifest['loss_last']}, adapter {manifest['adapter_dir']}",
    )
    return 0


def cmd_datasets_list(args) -> int:
    from .data.pipeline import list_datasets

    state = build_state()
    datasets = list_datasets(state.db)
    if args.json:
        _print(datasets, True)
        return 0
    for item in datasets:
        print(f"{item['id']}  {item['name'][:40]:<40} {item['status']:<8} {item['record_count']:>5} records  {item['rights_status']}")
    return 0


# ---------------------------------------------------------------------------
# GGUF / MoE / tuning / logs / pipeline
# ---------------------------------------------------------------------------

def _resolve_model_dir(state: AppState, value: str) -> Path:
    from .library import get_model

    model = get_model(state.db, value)
    if model:
        return Path(model["path"]).expanduser()
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"error: model id or directory not found: {value}")
    return path


def cmd_gguf_export(args) -> int:
    from .gguf_tools import discover, export_gguf

    state = build_state()
    source = _resolve_model_dir(state, args.model)
    toolchain = discover(state.settings)
    out = Path(args.out).expanduser() if args.out else state.settings.data_dir / "gguf" / f"{source.name}-{args.dtype}.gguf"
    try:
        result = export_gguf(
            source, out, toolchain=toolchain, dtype=args.dtype,
            log=(None if args.json else lambda line: print(line[:160])),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    _print(result, args.json, f"exported {result['output']} ({result['size_bytes'] / 1024**2:.0f} MiB)")
    return 0


def cmd_gguf_quantize(args) -> int:
    from .gguf_tools import discover, quantize_gguf

    state = build_state()
    toolchain = discover(state.settings)
    out = Path(args.out).expanduser() if args.out else Path(args.source).with_name(f"{Path(args.source).stem}-{args.method}.gguf")
    try:
        result = quantize_gguf(
            args.source, out, args.method, toolchain=toolchain,
            log=(None if args.json else lambda line: print(line[:160])),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    _print(result, args.json, f"{result['method']}: {result['size_bytes'] / 1024**2:.0f} MiB (ratio {result['ratio']})")
    return 0


def cmd_gguf_dequantize(args) -> int:
    from .gguf_tools import discover, requantize

    state = build_state()
    toolchain = discover(state.settings)
    out = Path(args.out).expanduser() if args.out else Path(args.source).with_name(f"{Path(args.source).stem}-{args.method}.gguf")
    try:
        result = requantize(args.source, out, toolchain=toolchain, method=args.method)
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    _print(
        result,
        args.json,
        f"upcast to {result['method']}: {result['output']} ({result['size_bytes'] / 1024**2:.0f} MiB). "
        "This is a GGUF requantization, not a conversion to HF safetensors.",
    )
    return 0


def cmd_gguf_inspect(args) -> int:
    from .runtime.gguf import read_gguf

    try:
        info = read_gguf(Path(args.file))
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    _print(info.to_public(), args.json, json.dumps(info.to_public(), indent=2))
    return 0


def cmd_moe_build(args) -> int:
    from .moe import MoEConfig, estimate, moeify

    state = build_state()
    source = _resolve_model_dir(state, args.model)
    config = MoEConfig(experts=args.experts, top_k=args.top_k, noise=args.noise)
    resources = estimate(source, config)
    if not resources["fits_ram"] and not args.force:
        return _fail(
            f"projected {resources['projected_bytes'] / 1024**3:.1f} GiB exceeds available "
            f"{resources['available_bytes'] / 1024**3:.1f} GiB; use --force or fewer experts"
        )
    out = Path(args.out).expanduser() if args.out else source.parent / f"{source.name}-moe"
    try:
        result = moeify(source, out, config=config, log=(None if args.json else print), force=args.force)
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    _print(result, args.json, f"MoE saved to {result['output']} ({result['size_bytes'] / 1024**2:.0f} MiB, {config.experts} experts)")
    return 0


def cmd_tune_apply(args) -> int:
    from .tuning import apply_profile

    state = build_state()
    profile = apply_profile(state.db, args.model_id)
    if not profile:
        return _fail("no Vulkan profile for this model; run `foundry tune vulkan <model>` first")
    _print(profile, args.json, f"ngl={profile.get('gpu_layers')} batch={profile.get('batch')} ({profile.get('tokens_per_second')} tok/s)")


def cmd_logs_clean(args) -> int:
    from .data.cleanup import cleanup_all

    state = build_state()
    providers = [item.strip() for item in (args.providers or "").split(",") if item.strip()] or None
    try:
        result = cleanup_all(
            state.db, state.settings, providers=providers, rights_status=args.rights,
            quality_filter=not args.no_quality, quarantine=not args.no_quarantine,
            progress=(None if args.json else lambda pct, msg: print(f"\r{pct:5.1f}% {msg[:70]:<70}", end="", flush=True)),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    if not args.json:
        print()
    _print(result, args.json, f"cleaned {result['records_included']} records -> {result['dataset_id']} (excluded {result['records_excluded']})")
    return 0


def cmd_train_unified(args) -> int:
    from .training.unified import run_unified

    state = build_state()
    try:
        result = run_unified(
            state.db, state.settings, model_path=args.model, dataset_id=args.dataset,
            options={"max_steps": args.steps, "quant": args.quant},
            progress=(None if args.json else lambda pct, msg: print(f"\r{pct:5.1f}% {msg[:70]:<70}", end="", flush=True)),
            log=(None if args.json else print),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    if not args.json:
        print()
    _print(result, args.json, f"unified {result['run_id']}: " + ", ".join(f"{s['stage']}={s['status']}" for s in result["stages"]))
    return 0


def cmd_tune_vulkan(args) -> int:
    from .library import get_model
    from .tuning import sweep

    state = build_state()
    model = get_model(state.db, args.model_id)
    if not model:
        return _fail(f"model not found: {args.model_id}")
    binary = _pick_binary(state)
    if not binary:
        return _fail("no verified llama-server binary")
    ngl = [int(value) for value in args.ngl.split(",") if value.strip()]
    result = sweep(
        state.db, state.server, model=model, binary=binary,
        context=args.context, gpu_layers=ngl,
        progress=(None if args.json else lambda pct, msg: print(f"\r{pct:5.1f}% {msg:<50}", end="", flush=True)),
    )
    if not args.json:
        print()
    best = result.get("best") or {}
    _print(result, args.json, f"best: ngl={best.get('gpu_layers')} {best.get('tokens_per_second')} tok/s (speedup {result.get('speedup_vs_cpu')}x)")
    return 0


def cmd_logs_scan(args) -> int:
    from .data.sources import scan

    state = build_state()
    result = scan(state.settings)
    if args.json:
        _print(result, True)
        return 0
    for source in result:
        print(f"{source['provider']:<12} exists={source['exists']!s:<5} files={source['files']:<5} {source['bytes'] / 1024**2:8.1f} MiB  {source['label']}")
    return 0


def cmd_logs_import(args) -> int:
    from .data.sources import import_all, import_source

    state = build_state()
    try:
        if args.source == "all":
            result = import_all(
                state.db, state.settings, ["claude_code", "codex", "antigravity", "opencode"],
                rights_status=args.rights,
                progress=(None if args.json else lambda pct, msg: print(f"\r{pct:5.1f}% {msg[:70]:<70}", end="", flush=True)),
            )
            label = "all sources"
        else:
            result = import_source(
                state.db, state.settings, args.source, rights_status=args.rights,
                progress=(None if args.json else lambda pct, msg: print(f"\r{pct:5.1f}% {msg[:70]:<70}", end="", flush=True)),
            )
            label = args.source
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    if not args.json:
        print()
    _print(result, args.json, f"{label}: {result['record_count']} records → {result['dataset_id']}")
    return 0


def _dataset_prompts(db, dataset_id: str, limit: int) -> list[str]:
    rows = db.query(
        "SELECT payload FROM dataset_records WHERE dataset_id=? AND split='train' AND excluded=0 LIMIT ?",
        (dataset_id, limit),
    )
    prompts = []
    for row in rows:
        payload = json.loads(row["payload"])
        request = ((payload.get("task") or {}).get("user_request") or "").strip()
        if request:
            prompts.append(request)
    return prompts


def cmd_train_auto(args) -> int:
    from .training import SFTConfig, run_auto

    state = build_state()
    config = SFTConfig(
        model_path=args.model, dataset_id=args.dataset, max_steps=args.steps,
        max_seq_len=args.seq, micro_batch=args.batch, grad_accum=args.accum,
        lora_r=args.rank, learning_rate=args.lr, auto=True,
        eval_every=args.eval_every, patience=args.patience,
    )
    output = Path(args.output).expanduser() if args.output else state.settings.training_dir / f"auto_{int(time.time())}"
    try:
        result = run_auto(
            state.db, config, job_id=None, output_dir=output,
            progress=(None if args.json else lambda pct, msg: print(f"\r{pct:5.1f}% {msg[:70]:<70}", end="", flush=True)),
            log=(None if args.json else print),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    if not args.json:
        print()
    manifest = result["manifest"]
    _print(result, args.json, f"auto run {result['run_id']}: best eval {manifest.get('best_eval_loss')}, steps {manifest.get('steps')}")
    return 0


def cmd_train_dpo(args) -> int:
    from .training import SFTConfig, run_dpo

    state = build_state()
    config = SFTConfig(model_path=args.model, dataset_id=args.dataset, max_steps=args.steps, max_seq_len=args.seq, lora_r=args.rank, learning_rate=args.lr)
    output = Path(args.output).expanduser() if args.output else state.settings.training_dir / f"dpo_{int(time.time())}"
    try:
        result = run_dpo(state.db, config, beta=args.beta, output_dir=output, log=(None if args.json else print))
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    _print(result, args.json, f"dpo: loss {result['manifest']['loss_first']} → {result['manifest']['loss_last']}")
    return 0


def cmd_train_distill(args) -> int:
    from . import gateway
    from .training import SFTConfig, run_distill

    state = build_state()
    prompts = _dataset_prompts(state.db, args.dataset, args.prompts)
    if not prompts:
        return _fail("dataset has no training prompts")
    if args.teacher == "gateway":
        config = gateway.get_config(state.settings.state_dir)
        if not (config["configured"] and config["enabled"]):
            return _fail("gateway teacher is not configured/enabled")
        teacher = gateway.GatewayTeacher(config["key"], config["model"])
    else:
        if state.server.status().get("status") != "ready":
            return _fail("no local model loaded; load one or use --teacher gateway")
        teacher = state.server

    def complete(messages):
        result = teacher.complete(messages, temperature=0.3, max_tokens=args.max_tokens)
        return result.get("text") or result.get("reasoning") or ""

    config = SFTConfig(model_path=args.model, dataset_id=args.dataset, max_steps=args.steps, max_seq_len=args.seq, lora_r=args.rank, learning_rate=args.lr)
    output = Path(args.output).expanduser() if args.output else state.settings.training_dir / f"distill_{int(time.time())}"
    try:
        result = run_distill(state.db, config, teacher_complete=complete, prompts=prompts, job_id=None, output_dir=output, log=(None if args.json else print))
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    _print(result, args.json, f"distilled {result['manifest']['teacher_samples']} answers → {result['manifest']['adapter_dir']}")
    return 0


def cmd_train_int4(args) -> int:
    from .training import SFTConfig, create_int4_run

    state = build_state()
    config = SFTConfig(
        model_path=args.model, dataset_id=args.dataset, max_steps=args.steps,
        max_seq_len=args.seq, micro_batch=args.batch, grad_accum=args.accum,
        lora_r=args.rank, learning_rate=args.lr, eval_every=args.eval_every,
        eval_samples=args.eval_samples,
    )
    output = Path(args.output).expanduser() if args.output else state.settings.training_dir / f"int4_{int(time.time())}"
    offload = True if getattr(args, "offload", False) else (False if getattr(args, "no_offload", False) else None)
    try:
        result = create_int4_run(
            state.db, config, job_id=None, output_dir=output,
            progress=(None if args.json else lambda pct, msg: print(f"\r{pct:5.1f}% {msg[:70]:<70}", end="", flush=True)),
            log=(None if args.json else print),
            streaming=args.streaming,
            offload=offload,
            vram_gb=getattr(args, "vram_gb", None),
            vram_mode=getattr(args, "vram_mode", "auto"),
            vram_dir=state.settings.data_dir / "cache" / "vram",
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    if not args.json:
        print()
    manifest = result["manifest"]
    mode = "offload(disk)" if manifest.get("offload") else ("streaming" if manifest.get("streaming") else "fp32-peak")
    peak = (manifest.get("memory") or {}).get("peak_rss_mb")
    _print(
        result,
        args.json,
        f"int4 run {result['run_id']}: {manifest['steps']} steps, {mode}, "
        f"loss {manifest['loss_first']} → {manifest['loss_last']}"
        + (f", peak RSS {peak:.0f} MB" if peak else "")
        + f", adapter {manifest['adapter_dir']}",
    )
    return 0


def cmd_vram(args) -> int:
    from .training.vram import budget_from_available, clear_arena

    state = build_state()
    arena = state.settings.data_dir / "cache" / "vram"
    if args.vram_command == "clear":
        result = clear_arena(arena)
        _print(result, args.json, f"cleared {len(result['removed'])} file(s) in {result['arena_dir']}")
        return 0
    index_path = arena / "arena.json"
    payload = {
        "arena_dir": str(arena),
        "arena_bytes_on_disk": (arena / "arena.bin").stat().st_size if (arena / "arena.bin").exists() else 0,
        "auto_budget_bytes": budget_from_available(),
        "index": json.loads(index_path.read_text("utf-8")) if index_path.exists() else None,
    }
    _print(
        payload,
        args.json,
        f"arena {payload['arena_bytes_on_disk'] / 1024**2:.0f} MB on disk at {arena}\n"
        f"auto budget {payload['auto_budget_bytes'] / 1024**2:.0f} MB"
        + (f"\nlast run: {json.dumps(payload['index'].get('stats'))}" if payload["index"] else "\nno index yet"),
    )
    return 0


def cmd_train_merge_int4(args) -> int:
    from .training.merge_int4 import merge_int4_adapter

    state = build_state()
    adapter = Path(args.adapter).expanduser()
    adapter_dir = next(
        (candidate for candidate in (adapter, adapter / "adapter") if (candidate / "lora_int4.pt").exists()),
        None,
    )
    if adapter_dir is None:
        return _fail(f"no lora_int4.pt found under {adapter}")
    store = Path(args.store).expanduser() if args.store else adapter_dir.parent / "int4-store"
    out = Path(args.out).expanduser() if args.out else adapter_dir.parent / "merged"
    try:
        result = merge_int4_adapter(
            store,
            adapter_dir,
            out,
            base_model_dir=args.base,
            dtype=args.dtype,
            shard_bytes=int(args.shard_gb * 1024**3),
            log=(None if args.json else print),
            progress=(None if args.json else lambda pct, msg: print(f"\r{pct:5.1f}% {msg[:70]:<70}", end="", flush=True)),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    if not args.json:
        print()
    _print(
        result,
        args.json,
        f"merged {result['quantized_weights_merged']} INT4 layers + LoRA into {result['output']} "
        f"({len(result['shards'])} shards, {result['size_bytes'] / 1024**3:.2f} GiB, {result['duration_seconds']} s)",
    )
    return 0


def cmd_pipeline(args) -> int:
    from .pipeline import run_pipeline

    state = build_state()
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    options = {
        "repo_id": args.repo,
        "files": args.file or None,
        "sources": [item.strip() for item in (args.sources or "").split(",") if item.strip()],
        "rights_status": args.rights,
        "dataset_id": args.dataset,
        "model_path": args.model,
        "max_steps": args.steps,
        "max_seq_len": args.seq,
        "quant": args.quant,
        "experts": args.experts,
        "top_k": args.top_k,
        "continue_on_error": args.continue_on_error,
    }
    try:
        result = run_pipeline(
            state.db, state.settings, stages=stages, options=options,
            progress=(None if args.json else lambda pct, msg: print(f"\r{pct:5.1f}% {msg[:70]:<70}", end="", flush=True)),
            log=(None if args.json else print),
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc))
    if not args.json:
        print()
    _print(result, args.json, f"pipeline {result['run_id']}: " + ", ".join(f"{s['stage']}={s['status']}" for s in result["stages"]))
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foundry", description=f"{APP_NAME} — local-first AI model lab")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local API service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=cmd_serve)

    scan = sub.add_parser("scan", help="scan this machine and print a capability report")
    scan.add_argument("--json", action="store_true")
    scan.add_argument("--write", action="store_true")
    scan.add_argument("--output", default=None)
    scan.set_defaults(func=cmd_scan)

    info = sub.add_parser("info", help="show resolved paths and roots")
    info.add_argument("--json", action="store_true")
    info.set_defaults(func=cmd_info)

    token = sub.add_parser("token", help="issue a pairing token for the local API")
    token.add_argument("--label", default="cli")
    token.add_argument("--json", action="store_true")
    token.set_defaults(func=cmd_token)

    hf = sub.add_parser("hf", help="Hugging Face Hub operations")
    hf_sub = hf.add_subparsers(dest="hf_command", required=True)
    hf_search = hf_sub.add_parser("search", help="search models")
    hf_search.add_argument("query")
    hf_search.add_argument("--task", default=None)
    hf_search.add_argument("--limit", type=int, default=15)
    hf_search.add_argument("--sort", default="downloads")
    hf_search.add_argument("--json", action="store_true")
    hf_search.set_defaults(func=cmd_hf_search)
    hf_files = hf_sub.add_parser("files", help="list files in a repository")
    hf_files.add_argument("repo")
    hf_files.add_argument("--json", action="store_true")
    hf_files.set_defaults(func=cmd_hf_files)
    hf_pull = hf_sub.add_parser("pull", help="download files from a repository")
    hf_pull.add_argument("repo")
    hf_pull.add_argument("--file", action="append", default=None, help="file path inside the repo (repeatable)")
    hf_pull.add_argument("--dest", default=None)
    hf_pull.add_argument("--json", action="store_true")
    hf_pull.set_defaults(func=cmd_hf_pull)
    hf_token = hf_sub.add_parser("token", help="manage the Hugging Face token")
    hf_token.add_argument("action", choices=["status", "set", "clear"])
    hf_token.add_argument("--token", default=None)
    hf_token.add_argument("--json", action="store_true")
    hf_token.set_defaults(func=cmd_hf_token)

    models = sub.add_parser("models", help="model library and inference")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    m_list = models_sub.add_parser("list")
    m_list.add_argument("--json", action="store_true")
    m_list.set_defaults(func=cmd_models_list)
    m_scan = models_sub.add_parser("scan")
    m_scan.add_argument("--root", action="append", default=None)
    m_scan.add_argument("--json", action="store_true")
    m_scan.set_defaults(func=cmd_models_scan)
    m_verify = models_sub.add_parser("verify")
    m_verify.add_argument("model_id")
    m_verify.add_argument("--json", action="store_true")
    m_verify.set_defaults(func=cmd_models_verify)
    m_load = models_sub.add_parser("load")
    m_load.add_argument("model_id")
    m_load.add_argument("--context", type=int, default=4096)
    m_load.add_argument("--ngl", type=int, default=99)
    m_load.add_argument("--force", action="store_true")
    m_load.add_argument("--json", action="store_true")
    m_load.set_defaults(func=cmd_models_load)
    m_unload = models_sub.add_parser("unload")
    m_unload.add_argument("--json", action="store_true")
    m_unload.set_defaults(func=cmd_models_unload)
    m_run = models_sub.add_parser("run", help="one-shot prompt against a model")
    m_run.add_argument("model_id")
    m_run.add_argument("--prompt", required=True)
    m_run.add_argument("--context", type=int, default=4096)
    m_run.add_argument("--ngl", type=int, default=99)
    m_run.add_argument("--max-tokens", type=int, default=256)
    m_run.add_argument("--temperature", type=float, default=0.2)
    m_run.add_argument("--force", action="store_true")
    m_run.add_argument("--json", action="store_true")
    m_run.set_defaults(func=cmd_models_run)

    bench = sub.add_parser("bench", help="run a benchmark suite against the loaded model")
    bench.add_argument("suite", choices=["instruction_adherence", "codegen_exec", "latency_throughput", "repo_fix", "agentic"])
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(func=cmd_bench)

    train = sub.add_parser("train", help="training utilities")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    t_smoke = train_sub.add_parser("smoke")
    t_smoke.add_argument("--json", action="store_true")
    t_smoke.set_defaults(func=cmd_train_smoke)
    t_pre = train_sub.add_parser("preflight")
    t_pre.add_argument("--dataset", required=True)
    t_pre.add_argument("--model", default=None)
    t_pre.add_argument("--json", action="store_true")
    t_pre.set_defaults(func=cmd_train_preflight)
    t_sft = train_sub.add_parser("sft", help="run a real LoRA/SFT job")
    t_sft.add_argument("--dataset", required=True)
    t_sft.add_argument("--model", required=True)
    t_sft.add_argument("--steps", type=int, default=20)
    t_sft.add_argument("--seq", type=int, default=256)
    t_sft.add_argument("--batch", type=int, default=1)
    t_sft.add_argument("--accum", type=int, default=4)
    t_sft.add_argument("--rank", type=int, default=8)
    t_sft.add_argument("--lr", type=float, default=2e-4)
    t_sft.add_argument("--output", default=None)
    t_sft.add_argument("--force", action="store_true")
    t_sft.add_argument("--json", action="store_true")
    t_sft.set_defaults(func=cmd_train_sft)

    datasets = sub.add_parser("datasets", help="dataset utilities")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)
    d_list = datasets_sub.add_parser("list")
    d_list.add_argument("--json", action="store_true")
    d_list.set_defaults(func=cmd_datasets_list)

    gguf = sub.add_parser("gguf", help="GGUF export, quantization and inspection")
    gguf_sub = gguf.add_subparsers(dest="gguf_command", required=True)
    g_export = gguf_sub.add_parser("export", help="convert a HF safetensors directory to GGUF")
    g_export.add_argument("model", help="model id or HF directory")
    g_export.add_argument("--out", default=None)
    g_export.add_argument("--dtype", default="f16", choices=["f16", "bf16", "f32"])
    g_export.add_argument("--json", action="store_true")
    g_export.set_defaults(func=cmd_gguf_export)
    g_quant = gguf_sub.add_parser("quantize", help="quantize a GGUF file")
    g_quant.add_argument("source")
    g_quant.add_argument("--method", default="Q4_K_M")
    g_quant.add_argument("--out", default=None)
    g_quant.add_argument("--json", action="store_true")
    g_quant.set_defaults(func=cmd_gguf_quantize)
    g_dequant = gguf_sub.add_parser("dequantize", help="upcast a GGUF to F16/BF16")
    g_dequant.add_argument("source")
    g_dequant.add_argument("--method", default="F16", choices=["F16", "BF16"])
    g_dequant.add_argument("--out", default=None)
    g_dequant.add_argument("--json", action="store_true")
    g_dequant.set_defaults(func=cmd_gguf_dequantize)
    g_inspect = gguf_sub.add_parser("inspect", help="read GGUF metadata")
    g_inspect.add_argument("file")
    g_inspect.add_argument("--json", action="store_true")
    g_inspect.set_defaults(func=cmd_gguf_inspect)

    moe = sub.add_parser("moe", help="dense → mixture-of-experts conversion")
    moe_sub = moe.add_subparsers(dest="moe_command", required=True)
    m_build = moe_sub.add_parser("build")
    m_build.add_argument("model", help="model id or directory")
    m_build.add_argument("--experts", type=int, default=4)
    m_build.add_argument("--top-k", type=int, default=2)
    m_build.add_argument("--noise", type=float, default=0.01)
    m_build.add_argument("--out", default=None)
    m_build.add_argument("--force", action="store_true")
    m_build.add_argument("--json", action="store_true")
    m_build.set_defaults(func=cmd_moe_build)

    tune = sub.add_parser("tune", help="runtime auto-tuning")
    tune_sub = tune.add_subparsers(dest="tune_command", required=True)
    t_vulkan = tune_sub.add_parser("vulkan", help="sweep GPU offload and batch size")
    t_vulkan.add_argument("model_id")
    t_vulkan.add_argument("--ngl", default="0,8,16,99")
    t_vulkan.add_argument("--context", type=int, default=2048)
    t_vulkan.add_argument("--json", action="store_true")
    t_vulkan.set_defaults(func=cmd_tune_vulkan)
    t_apply = tune_sub.add_parser("apply", help="show the stored winning profile")
    t_apply.add_argument("model_id")
    t_apply.add_argument("--json", action="store_true")
    t_apply.set_defaults(func=cmd_tune_apply)

    logs = sub.add_parser("logs", help="discover and import local agent sessions")
    logs_sub = logs.add_subparsers(dest="logs_command", required=True)
    l_scan = logs_sub.add_parser("scan")
    l_scan.add_argument("--json", action="store_true")
    l_scan.set_defaults(func=cmd_logs_scan)
    l_import = logs_sub.add_parser("import")
    l_import.add_argument("source", choices=["claude_code", "codex", "antigravity", "opencode", "all"])
    l_import.add_argument("--rights", default="needs_review", choices=["needs_review", "confirmed", "restricted", "excluded"])
    l_import.add_argument("--json", action="store_true")
    l_import.set_defaults(func=cmd_logs_import)
    l_clean = logs_sub.add_parser("clean", help="import, redact, filter, freeze and quarantine all sources")
    l_clean.add_argument("--rights", default="confirmed", choices=["needs_review", "confirmed", "restricted", "excluded"])
    l_clean.add_argument("--providers", default=None)
    l_clean.add_argument("--no-quality", action="store_true")
    l_clean.add_argument("--no-quarantine", action="store_true")
    l_clean.add_argument("--json", action="store_true")
    l_clean.set_defaults(func=cmd_logs_clean)

    t_auto = train_sub.add_parser("auto", help="SFT with OpenEval, auto-decision and playbook")
    t_auto.add_argument("--dataset", required=True)
    t_auto.add_argument("--model", required=True)
    t_auto.add_argument("--steps", type=int, default=20)
    t_auto.add_argument("--seq", type=int, default=256)
    t_auto.add_argument("--batch", type=int, default=1)
    t_auto.add_argument("--accum", type=int, default=4)
    t_auto.add_argument("--rank", type=int, default=8)
    t_auto.add_argument("--lr", type=float, default=2e-4)
    t_auto.add_argument("--eval-every", type=int, default=5)
    t_auto.add_argument("--patience", type=int, default=3)
    t_auto.add_argument("--output", default=None)
    t_auto.add_argument("--json", action="store_true")
    t_auto.set_defaults(func=cmd_train_auto)
    t_dpo = train_sub.add_parser("dpo", help="direct preference optimisation with synthetic pairs")
    t_dpo.add_argument("--dataset", required=True)
    t_dpo.add_argument("--model", required=True)
    t_dpo.add_argument("--steps", type=int, default=10)
    t_dpo.add_argument("--seq", type=int, default=128)
    t_dpo.add_argument("--rank", type=int, default=4)
    t_dpo.add_argument("--lr", type=float, default=5e-5)
    t_dpo.add_argument("--beta", type=float, default=0.1)
    t_dpo.add_argument("--output", default=None)
    t_dpo.add_argument("--json", action="store_true")
    t_dpo.set_defaults(func=cmd_train_dpo)
    t_distill = train_sub.add_parser("distill", help="teacher-generated answers, then SFT")
    t_distill.add_argument("--dataset", required=True)
    t_distill.add_argument("--model", required=True)
    t_distill.add_argument("--teacher", default="local", choices=["local", "gateway"])
    t_distill.add_argument("--prompts", type=int, default=8)
    t_distill.add_argument("--steps", type=int, default=10)
    t_distill.add_argument("--seq", type=int, default=256)
    t_distill.add_argument("--rank", type=int, default=4)
    t_distill.add_argument("--lr", type=float, default=2e-4)
    t_distill.add_argument("--max-tokens", type=int, default=256)
    t_distill.add_argument("--output", default=None)
    t_distill.add_argument("--json", action="store_true")
    t_distill.set_defaults(func=cmd_train_distill)
    t_int4 = train_sub.add_parser("int4", help="INT4 (QLoRA-style) training without bitsandbytes")
    t_int4.add_argument("--dataset", required=True)
    t_int4.add_argument("--model", required=True)
    t_int4.add_argument("--steps", type=int, default=10)
    t_int4.add_argument("--seq", type=int, default=128)
    t_int4.add_argument("--batch", type=int, default=1)
    t_int4.add_argument("--accum", type=int, default=2)
    t_int4.add_argument("--rank", type=int, default=8)
    t_int4.add_argument("--lr", type=float, default=2e-4)
    t_int4.add_argument("--eval-every", type=int, default=0)
    t_int4.add_argument("--eval-samples", type=int, default=4)
    t_int4.add_argument("--streaming", action="store_true", help="quantize while loading (avoids the fp32 peak)")
    t_int4.add_argument("--offload", action="store_true", help="force the disk-backed INT4 store (memory-mapped)")
    t_int4.add_argument("--no-offload", action="store_true", help="keep the INT4 weights in RAM even if the estimate does not fit")
    t_int4.add_argument("--vram-gb", type=float, default=None, help="hot-tier budget for the tiered weight cache (default: auto from free RAM)")
    t_int4.add_argument("--vram-mode", default="auto", choices=["auto", "off"], help="auto budgets from live RAM; off disables the cache")
    t_int4.add_argument("--output", default=None)
    t_int4.add_argument("--json", action="store_true")
    t_int4.set_defaults(func=cmd_train_int4)
    t_unified = train_sub.add_parser("unified", help="decide and run train → merge → export → quantize")
    t_unified.add_argument("--dataset", required=True)
    t_unified.add_argument("--model", required=True)
    t_unified.add_argument("--steps", type=int, default=20)
    t_unified.add_argument("--quant", default="Q4_K_M")
    t_unified.add_argument("--json", action="store_true")
    t_unified.set_defaults(func=cmd_train_unified)
    t_merge = train_sub.add_parser("merge-int4", help="merge an INT4 LoRA adapter into a plain fp16 checkpoint")
    t_merge.add_argument("--adapter", required=True, help="run directory or adapter directory with lora_int4.pt")
    t_merge.add_argument("--store", default=None, help="int4-store directory (default: next to the adapter)")
    t_merge.add_argument("--base", default=None, help="base HF directory (default: recorded in the store)")
    t_merge.add_argument("--out", default=None)
    t_merge.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    t_merge.add_argument("--shard-gb", type=float, default=1.5)
    t_merge.add_argument("--json", action="store_true")
    t_merge.set_defaults(func=cmd_train_merge_int4)

    vram = sub.add_parser("vram", help="inspect or clear the virtual-RAM arena on disk")
    vram_sub = vram.add_subparsers(dest="vram_command", required=True)
    for name, help_text in (("status", "show the arena, the last run's cache stats and the auto budget"), ("clear", "delete the arena file (it is only a cache)")):
        v = vram_sub.add_parser(name, help=help_text)
        v.add_argument("--json", action="store_true")
        v.set_defaults(func=cmd_vram)

    pipeline = sub.add_parser("pipeline", help="all-in-one orchestration")
    pipeline_sub = pipeline.add_subparsers(dest="pipeline_command", required=True)
    p_run = pipeline_sub.add_parser("run")
    p_run.add_argument("--stages", required=True, help="comma separated: pull,dataset,train,merge,export,quantize,moe,report")
    p_run.add_argument("--repo", default=None)
    p_run.add_argument("--file", action="append", default=None)
    p_run.add_argument("--sources", default=None)
    p_run.add_argument("--dataset", default=None)
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--steps", type=int, default=20)
    p_run.add_argument("--seq", type=int, default=256)
    p_run.add_argument("--quant", default="Q4_K_M")
    p_run.add_argument("--experts", type=int, default=4)
    p_run.add_argument("--top-k", type=int, default=2)
    p_run.add_argument("--rights", default="needs_review")
    p_run.add_argument("--continue-on-error", action="store_true")
    p_run.add_argument("--json", action="store_true")
    p_run.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; keep unicode output safe.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return _fail("interrupted", 130)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
