"""Durable job runner.

Every heavy operation is a job row with a lifecycle, events, cancellation and an
output payload. Jobs execute in worker threads; they never touch the asyncio loop
directly, they publish through the event bus.
"""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .db import Database, new_id, utc_now
from .events import EventBus


class JobCancelled(Exception):
    """Raised inside a job when the user requested cancellation."""


TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted"}


@dataclass
class JobRecord:
    id: str
    type: str
    status: str
    progress: float
    message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    input: dict
    output: dict


class JobContext:
    """Handed to job functions. All methods are thread-safe."""

    def __init__(self, job_id: str, db: Database, bus: EventBus, cancel: threading.Event):
        self.job_id = job_id
        self.db = db
        self.bus = bus
        self._cancel = cancel
        self._seq = 0
        self._seq_lock = threading.Lock()
        self.artifacts: list[dict] = []

    # -- cancellation ------------------------------------------------------
    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled("job cancelled by user")

    # -- progress ----------------------------------------------------------
    def progress(self, percent: float, message: str | None = None) -> None:
        pct = max(0.0, min(100.0, float(percent)))
        self.db.execute(
            "UPDATE jobs SET progress=?, message=COALESCE(?, message) WHERE id=?",
            (pct, message, self.job_id),
        )
        self.emit("progress", message or f"{pct:.0f}%", {"progress": pct})

    def log(self, message: str, level: str = "info", data: dict | None = None) -> None:
        self.emit("log", message, data or {}, level=level)

    def emit(self, kind: str, message: str, data: dict | None = None, level: str = "info") -> None:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        ts = utc_now()
        payload = data or {}
        self.db.execute(
            "INSERT INTO job_events(job_id, seq, ts, level, kind, message, data) VALUES(?,?,?,?,?,?,?)",
            (self.job_id, seq, ts, level, kind, message, json.dumps(payload)),
        )
        self.bus.publish(
            {
                "type": "job_event",
                "job_id": self.job_id,
                "seq": seq,
                "ts": ts,
                "level": level,
                "kind": kind,
                "message": message,
                "data": payload,
            }
        )

    def artifact(self, kind: str, path: str, **meta: Any) -> None:
        entry = {"kind": kind, "path": path, **meta}
        self.artifacts.append(entry)
        self.emit("artifact", f"{kind}: {path}", entry)

    def child_output(self, line: str) -> None:
        """Forward a line of subprocess output as a log event."""
        text = line.rstrip()
        if text:
            self.emit("log", text, {})


class JobRunner:
    def __init__(self, db: Database, bus: EventBus, *, allow_concurrent_heavy: bool = False):
        self.db = db
        self.bus = bus
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._heavy = threading.Semaphore(2 if allow_concurrent_heavy else 1)
        self._heavy_enabled = True

    # -- lifecycle ---------------------------------------------------------
    def recover_interrupted(self) -> int:
        rows = self.db.query("SELECT id FROM jobs WHERE status IN ('running','queued')")
        for row in rows:
            self.db.execute(
                "UPDATE jobs SET status='interrupted', finished_at=?, "
                "error='service stopped while the job was running' WHERE id=?",
                (utc_now(), row["id"]),
            )
            self._append_event(row["id"], "error", "interrupted", "Service restarted mid-job.")
        return len(rows)

    def _append_event(self, job_id: str, level: str, kind: str, message: str) -> None:
        row = self.db.query_one("SELECT COALESCE(MAX(seq),0) AS s FROM job_events WHERE job_id=?", (job_id,))
        seq = int(row["s"]) + 1 if row else 1
        self.db.execute(
            "INSERT INTO job_events(job_id, seq, ts, level, kind, message, data) VALUES(?,?,?,?,?,?,?)",
            (job_id, seq, utc_now(), level, kind, message, "{}"),
        )

    def submit(
        self,
        job_type: str,
        fn: Callable[[JobContext], dict | None],
        *,
        input_data: dict | None = None,
        owner: str = "user",
        heavy: bool = False,
        job_id: str | None = None,
    ) -> str:
        jid = job_id or new_id("job")
        self.db.insert(
            "jobs",
            {
                "id": jid,
                "type": job_type,
                "status": "queued",
                "progress": 0,
                "message": "queued",
                "owner": owner,
                "input": json.dumps(input_data or {}),
                "output": "{}",
                "created_at": utc_now(),
            },
        )
        cancel = threading.Event()
        with self._lock:
            self._cancel_events[jid] = cancel
        self.bus.publish({"type": "job_created", "job_id": jid, "job_type": job_type})

        def runner() -> None:
            acquired = False
            try:
                if heavy:
                    self.bus.publish(
                        {"type": "job_event", "job_id": jid, "kind": "waiting", "message": "Waiting for a resource slot…"}
                    )
                    acquired = self._heavy.acquire(timeout=3600)
                    if not acquired:
                        raise RuntimeError("timed out waiting for a resource slot")
                self.db.execute(
                    "UPDATE jobs SET status='running', started_at=?, message='running' WHERE id=?",
                    (utc_now(), jid),
                )
                self.bus.publish({"type": "job_started", "job_id": jid})
                ctx = JobContext(jid, self.db, self.bus, cancel)
                result = fn(ctx) or {}
                if cancel.is_set():
                    raise JobCancelled("job cancelled by user")
                self.db.execute(
                    "UPDATE jobs SET status='succeeded', progress=100, message='done', output=?, "
                    "finished_at=? WHERE id=?",
                    (json.dumps(result), utc_now(), jid),
                )
                self.bus.publish({"type": "job_finished", "job_id": jid, "status": "succeeded"})
            except JobCancelled as exc:
                self.db.execute(
                    "UPDATE jobs SET status='cancelled', message=?, finished_at=?, error=? WHERE id=?",
                    ("cancelled", utc_now(), str(exc), jid),
                )
                self.bus.publish({"type": "job_finished", "job_id": jid, "status": "cancelled"})
            except Exception as exc:  # noqa: BLE001 - job boundary must not leak
                detail = traceback.format_exc(limit=6)
                self.db.execute(
                    "UPDATE jobs SET status='failed', message=?, finished_at=?, error=? WHERE id=?",
                    (str(exc)[:500], utc_now(), detail[-4000:], jid),
                )
                self._append_event(jid, "error", "failed", str(exc))
                self.bus.publish(
                    {"type": "job_finished", "job_id": jid, "status": "failed", "error": str(exc)}
                )
            finally:
                if acquired:
                    self._heavy.release()
                with self._lock:
                    self._cancel_events.pop(jid, None)

        thread = threading.Thread(target=runner, name=f"job-{jid}", daemon=True)
        with self._lock:
            self._threads[jid] = thread
        thread.start()
        return jid

    def cancel(self, job_id: str) -> bool:
        event = self._cancel_events.get(job_id)
        if event is None:
            row = self.db.query_one("SELECT status FROM jobs WHERE id=?", (job_id,))
            return bool(row and row["status"] in TERMINAL_STATES)
        event.set()
        self.db.execute("UPDATE jobs SET cancel_requested=1, message='cancelling…' WHERE id=?", (job_id,))
        self.bus.publish({"type": "job_event", "job_id": job_id, "kind": "cancel", "message": "Cancellation requested"})
        return True

    def wait(self, job_id: str, timeout: float = 30.0) -> None:
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)

    # -- queries -----------------------------------------------------------
    def get(self, job_id: str) -> dict | None:
        row = self.db.query_one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._row_to_dict(row) if row else None

    def list(self, limit: int = 50, type_prefix: str | None = None) -> list[dict]:
        if type_prefix:
            rows = self.db.query(
                "SELECT * FROM jobs WHERE type LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"{type_prefix}%", limit),
            )
        else:
            rows = self.db.query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._row_to_dict(r) for r in rows]

    def events(self, job_id: str, after_seq: int = 0, limit: int = 500) -> list[dict]:
        rows = self.db.query(
            "SELECT * FROM job_events WHERE job_id=? AND seq>? ORDER BY seq ASC LIMIT ?",
            (job_id, after_seq, limit),
        )
        return [
            {
                "seq": r["seq"],
                "ts": r["ts"],
                "level": r["level"],
                "kind": r["kind"],
                "message": r["message"],
                "data": json.loads(r["data"]),
            }
            for r in rows
        ]

    @staticmethod
    def _row_to_dict(row) -> dict:
        data = dict(row)
        data["input"] = json.loads(data.get("input") or "{}")
        data["output"] = json.loads(data.get("output") or "{}")
        data["cancel_requested"] = bool(data.get("cancel_requested"))
        return data
