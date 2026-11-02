"""ASGI application factory."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import api_router
from .config import APP_NAME, VERSION, Settings, apply_overrides, load_settings, repo_root
from .db import init_db
from .events import EventBus
from .jobs import JobRunner
from .runtime.llama_server import LlamaServer
from .security import PairingStore, PrivateNetworkAccessMiddleware, SecurityMiddleware, SecurityPolicy
from .state import AppState

log = logging.getLogger("foundry")


def _setup_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(handler)
    logfile = settings.logs_dir / "foundry.log"
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(file_handler)


def create_app(settings: Settings | None = None, *, serve_web: bool = True) -> FastAPI:
    settings = settings or load_settings()
    settings.ensure_dirs()
    _setup_logging(settings)

    db = init_db(settings.db_path)
    # Persisted overrides win over defaults, so a restart keeps the user's paths.
    apply_overrides(settings, db.get_setting("settings"))
    settings.ensure_dirs()
    bus = EventBus()
    runner = JobRunner(db, bus, allow_concurrent_heavy=settings.allow_concurrent_heavy_jobs)
    server = LlamaServer(settings.state_dir)
    pairing = PairingStore(settings.state_dir)
    state = AppState(
        settings=settings,
        db=db,
        bus=bus,
        runner=runner,
        server=server,
        pairing=pairing,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncio

        bus.bind_loop(asyncio.get_running_loop())
        recovered = runner.recover_interrupted()
        if recovered:
            log.warning("marked %s interrupted job(s) after restart", recovered)
        reaped = server.reap_orphan()
        if reaped:
            log.warning(reaped)
        db.activity("service.start", f"{APP_NAME} {VERSION} started")
        log.info("%s %s listening on %s:%s", APP_NAME, VERSION, settings.host, settings.port)
        try:
            yield
        finally:
            server.shutdown()
            db.activity("service.stop", "Service stopped")
            log.info("service stopped")

    app = FastAPI(
        title=APP_NAME,
        version=VERSION,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.foundry = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ui_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Foundry-Client", "X-Foundry-Token"],
        expose_headers=["Content-Disposition"],
        max_age=600,
    )
    app.add_middleware(
        SecurityMiddleware,
        store=pairing,
        policy=SecurityPolicy(origins=settings.ui_origins, allow_lan=settings.allow_lan),
    )
    # outermost: adds the Private Network Access header to the CORS preflight response
    app.add_middleware(PrivateNetworkAccessMiddleware)

    app.include_router(api_router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # pragma: no cover - safety net
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Unexpected server error. See the local log."}},
        )

    # --- static frontend -------------------------------------------------
    web_dist = repo_root() / "web" / "dist"
    if serve_web and web_dist.exists():
        assets = web_dist / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(web_dist / "index.html")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            candidate = (web_dist / path).resolve()
            if web_dist.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(web_dist / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def dev_root():
            return JSONResponse(
                {
                    "app": APP_NAME,
                    "version": VERSION,
                    "message": "API is running. Start the web UI with `npm run dev` in web/ (http://127.0.0.1:5173).",
                }
            )

    return app


def main() -> None:  # pragma: no cover - console entry point
    import uvicorn

    settings = load_settings()
    if not settings.allow_lan and settings.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("Refusing to bind to a non-loopback host without FOUNDRY_ALLOW_LAN=1")
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":  # pragma: no cover
    main()
