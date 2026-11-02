"""Configuration and filesystem layout for the local Foundry service.

Nothing here is machine-specific by hard-coding: defaults are *proposals* that the
scanner validates, and every path can be overridden through the settings API or
environment variables.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "HCS Foundry"
from . import __version__ as _package_version

#: Single source of truth: ``foundry.__version__``. A second literal here drifted before.
VERSION = _package_version
SCHEMA_VERSION = 1
API_PREFIX = "/api/v1"

#: Origin allowlist used when the SPA is served by Vite in dev, by `vite preview`,
#: or by the Foundry service itself. Extra origins can be added in settings.
DEFAULT_UI_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
    "http://127.0.0.1:8770",
    "http://localhost:8770",
    # the deployed landing page may open the pairing page against this service
    "https://hcs-foundry-site.vercel.app",
]


def repo_root() -> Path:
    """Repository root, or the PyInstaller bundle root when frozen."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle)
    return Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_paths(raw: str | None) -> list[Path]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(";", os.pathsep).split(os.pathsep)]
    return [Path(p).expanduser() for p in parts if p]


def _split_origins(raw: str | None) -> list[str]:
    """Split an origin list without touching the value.

    Origins are URLs, so they must not go through ``Path`` (``Path("https://a").as_posix()``
    collapses the double slash) and ``os.pathsep`` splitting is wrong on Windows for URLs.
    Commas and semicolons are the separators.
    """
    if not raw:
        return []
    parts = [p.strip().rstrip("/") for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p and p != "*"]


def default_model_roots() -> list[Path]:
    """Proposed model roots. Only existing directories are returned."""
    home = Path.home()
    candidates = [
        repo_root() / "data" / "models",
        Path("D:/aimodels"),
        home / ".lmstudio" / "models",
        home / ".cache" / "huggingface" / "hub",
        home / "models",
        Path("C:/models"),
    ]
    if sys.platform == "darwin":
        candidates.append(home / "Library" / "Application Support" / "LM Studio" / "models")
    return [p for p in candidates if p.exists()]


def default_import_roots() -> list[Path]:
    home = Path.home()
    candidates = [
        repo_root() / "data" / "imports",
        home / ".claude" / "projects",
        home / ".codex" / "sessions",
        home / ".local" / "share" / "opencode",
    ]
    return [p for p in candidates if p.exists()]


@dataclass
class Settings:
    """Runtime configuration. User-editable values live in the settings table."""

    data_dir: Path
    model_roots: list[Path] = field(default_factory=list)
    import_roots: list[Path] = field(default_factory=list)
    ui_origins: list[str] = field(default_factory=lambda: list(DEFAULT_UI_ORIGINS))
    host: str = "127.0.0.1"
    port: int = 8770
    allow_lan: bool = False
    telemetry: bool = False
    # Opt-in only: the app is local-first and makes no outbound call by default.
    update_check: bool = False
    log_level: str = "INFO"

    # Resource governor knobs (see machine/governor.py)
    os_reserve_ratio: float = 0.15
    os_reserve_min_bytes: int = 2 * 1024**3
    gpu_headroom_bytes: int = 1 * 1024**3
    allow_concurrent_heavy_jobs: bool = False

    # Runtime discovery
    llama_server_candidates: list[Path] = field(default_factory=list)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "foundry.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def training_dir(self) -> Path:
        return self.data_dir / "training-runs"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.artifacts_dir,
            self.datasets_dir,
            self.training_dir,
            self.logs_dir,
            self.state_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_public(self) -> dict:
        return {
            "data_dir": str(self.data_dir),
            "model_roots": [str(p) for p in self.model_roots],
            "import_roots": [str(p) for p in self.import_roots],
            "ui_origins": list(self.ui_origins),
            "host": self.host,
            "port": self.port,
            "allow_lan": self.allow_lan,
            "telemetry": self.telemetry,
            "update_check": self.update_check,
            "os_reserve_ratio": self.os_reserve_ratio,
            "os_reserve_min_bytes": self.os_reserve_min_bytes,
            "gpu_headroom_bytes": self.gpu_headroom_bytes,
            "allow_concurrent_heavy_jobs": self.allow_concurrent_heavy_jobs,
            "llama_server_candidates": [str(p) for p in self.llama_server_candidates],
        }


def load_settings() -> Settings:
    data_dir = Path(_env("FOUNDRY_DATA_DIR", str(repo_root() / "data"))).expanduser()
    model_roots = _split_paths(_env("FOUNDRY_MODEL_ROOTS")) or default_model_roots()
    import_roots = _split_paths(_env("FOUNDRY_IMPORT_ROOTS")) or default_import_roots()
    ui_origins = _split_origins(_env("FOUNDRY_UI_ORIGINS"))
    origins = ui_origins or list(DEFAULT_UI_ORIGINS)

    settings = Settings(
        data_dir=data_dir,
        model_roots=model_roots,
        import_roots=import_roots,
        ui_origins=origins,
        host=_env("FOUNDRY_HOST", "127.0.0.1") or "127.0.0.1",
        port=int(_env("FOUNDRY_PORT", "8770") or 8770),
        allow_lan=_env_bool("FOUNDRY_ALLOW_LAN", False),
        telemetry=_env_bool("FOUNDRY_TELEMETRY", False),
        log_level=_env("FOUNDRY_LOG_LEVEL", "INFO") or "INFO",
    )
    settings.ensure_dirs()
    return settings


def apply_overrides(settings: Settings, overrides: dict | None) -> Settings:
    """Apply persisted settings from the database over the defaults."""
    if not overrides:
        return settings
    if isinstance(overrides.get("model_roots"), list):
        roots = [Path(p).expanduser() for p in overrides["model_roots"] if isinstance(p, str) and p.strip()]
        if roots:
            settings.model_roots = roots
    if isinstance(overrides.get("import_roots"), list):
        roots = [Path(p).expanduser() for p in overrides["import_roots"] if isinstance(p, str) and p.strip()]
        if roots:
            settings.import_roots = roots
    if isinstance(overrides.get("ui_origins"), list):
        origins = [str(o) for o in overrides["ui_origins"] if isinstance(o, str) and o.strip()]
        if origins:
            settings.ui_origins = origins
    if isinstance(overrides.get("os_reserve_ratio"), (int, float)):
        settings.os_reserve_ratio = max(0.05, min(0.5, float(overrides["os_reserve_ratio"])))
    if isinstance(overrides.get("gpu_headroom_bytes"), (int, float)):
        settings.gpu_headroom_bytes = max(0, int(overrides["gpu_headroom_bytes"]))
    if isinstance(overrides.get("allow_concurrent_heavy_jobs"), bool):
        settings.allow_concurrent_heavy_jobs = overrides["allow_concurrent_heavy_jobs"]
    if isinstance(overrides.get("update_check"), bool):
        settings.update_check = overrides["update_check"]
    if isinstance(overrides.get("llama_server_candidates"), list):
        settings.llama_server_candidates = [
            Path(p).expanduser() for p in overrides["llama_server_candidates"] if isinstance(p, str) and p.strip()
        ]
    return settings
