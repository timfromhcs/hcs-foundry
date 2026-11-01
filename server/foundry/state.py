"""Shared application state for the API layer."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

from .config import Settings
from .db import Database, new_id, utc_now
from .events import EventBus
from .jobs import JobRunner
from .runtime.llama_server import LlamaServer
from .security import PairingStore


@dataclass
class AppState:
    settings: Settings
    db: Database
    bus: EventBus
    runner: JobRunner
    server: LlamaServer
    pairing: PairingStore
    profile: dict | None = None
    scan_lock: threading.Lock = field(default_factory=threading.Lock)

    # -- machine profile ---------------------------------------------------
    def latest_profile(self) -> dict | None:
        row = self.db.query_one("SELECT payload FROM machine_profiles ORDER BY created_at DESC LIMIT 1")
        if row:
            return json.loads(row["payload"])
        return self.profile

    def store_profile(self, profile: dict) -> None:
        self.profile = profile
        self.db.insert(
            "machine_profiles",
            {"id": new_id("mach"), "created_at": utc_now(), "payload": json.dumps(profile)},
        )

    def activity(self, kind: str, message: str, data: dict | None = None) -> None:
        self.db.activity(kind, message, data)
