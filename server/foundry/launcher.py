"""Entry point for the standalone executable.

The bundled build contains the API, the built SPA and the Python runtime. User data
lives in ``%LOCALAPPDATA%\\HCS Foundry`` (Windows) or ``~/.local/share/hcs-foundry``
so the executable can sit anywhere, including a read-only folder.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _default_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "HCS Foundry"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "HCS Foundry"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "hcs-foundry"


def _free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def _open_browser(url: str, delay: float = 1.2) -> None:
    def worker() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - opening a browser is best effort
            pass

    threading.Thread(target=worker, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="HCSFoundry", description="HCS Foundry — local AI model lab")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--mcp", action="store_true", help="serve the Model Context Protocol on stdin/stdout instead of the web UI")
    parser.add_argument("--doctor", action="store_true", help="print a local health report and exit")
    args = parser.parse_args(argv)

    os.environ.setdefault("FOUNDRY_DATA_DIR", str(Path(args.data_dir).expanduser() if args.data_dir else _default_data_dir()))
    os.environ.setdefault("FOUNDRY_HOST", "127.0.0.1")

    if args.mcp:
        from foundry.cli import build_state

        from foundry.mcp_server import run_mcp

        return run_mcp(build_state())

    if args.doctor:
        from foundry.cli import main as cli_main

        return cli_main(["run", "doctor"])

    port = _free_port(args.port)
    os.environ["FOUNDRY_PORT"] = str(port)

    import uvicorn

    from foundry.config import APP_NAME, VERSION, load_settings
    from foundry.main import create_app

    settings = load_settings()
    settings.host = "127.0.0.1"
    settings.port = port
    app = create_app(settings)
    url = f"http://127.0.0.1:{port}/"

    print(f"{APP_NAME} {VERSION}")
    print(f"  URL:  {url}")
    print(f"  Data: {settings.data_dir}")
    print("  Press Ctrl+C to stop.")
    if not args.no_browser:
        _open_browser(url)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
