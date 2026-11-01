"""Command line entry point: ``foundry <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import APP_NAME, VERSION, load_settings, repo_root


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .main import create_app

    settings = load_settings()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if not settings.allow_lan and settings.host not in {"127.0.0.1", "::1", "localhost"}:
        print("Refusing to bind to a non-loopback host without FOUNDRY_ALLOW_LAN=1", file=sys.stderr)
        return 2
    app = create_app(settings)
    print(f"{APP_NAME} {VERSION}")
    print(f"  API:  http://{settings.host}:{settings.port}/api/v1")
    print(f"  UI:   http://{settings.host}:{settings.port}/" if (repo_root() / "web" / "dist").exists() else "  UI:   run `npm run dev` in web/")
    print(f"  Data: {settings.data_dir}")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from .machine import compute_budget, scan_machine

    settings = load_settings()
    profile = scan_machine(settings)
    budget = compute_budget(profile, settings)
    if args.write:
        out = Path(args.output) if args.output else repo_root() / "docs" / "audit" / "MACHINE_PROFILE.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(profile, indent=2), "utf-8")
        print(f"wrote {out}")
    if args.json:
        print(json.dumps({"profile": profile, "budget": budget.to_public()}, indent=2))
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


def cmd_token(args: argparse.Namespace) -> int:
    from .security import PairingStore

    settings = load_settings()
    store = PairingStore(settings.state_dir)
    token = store.issue(args.label)
    print(token)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    settings = load_settings()
    print(f"{APP_NAME} {VERSION}")
    print(f"data dir:   {settings.data_dir}")
    print(f"database:   {settings.db_path}")
    print(f"model roots: {', '.join(str(p) for p in settings.model_roots) or '(none)'}")
    print(f"import roots: {', '.join(str(p) for p in settings.import_roots) or '(none)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foundry", description=f"{APP_NAME} — local-first AI model lab")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local API service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=cmd_serve)

    scan = sub.add_parser("scan", help="scan this machine and print a capability report")
    scan.add_argument("--json", action="store_true", help="print the full profile as JSON")
    scan.add_argument("--write", action="store_true", help="write docs/audit/MACHINE_PROFILE.json")
    scan.add_argument("--output", default=None)
    scan.set_defaults(func=cmd_scan)

    token = sub.add_parser("token", help="issue a pairing token for the local API")
    token.add_argument("--label", default="cli")
    token.set_defaults(func=cmd_token)

    info = sub.add_parser("info", help="show resolved paths and roots")
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
