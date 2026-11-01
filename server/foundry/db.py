"""SQLite persistence: schema, migrations and small query helpers.

Design notes
------------
* WAL mode so readers (the API) are never blocked by a writer (a job).
* One connection per thread: the API is async but jobs run in worker threads.
* No ORM. Hand-written SQL keeps the schema auditable, and every migration is
  additive so an interrupted upgrade cannot destroy user data.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = 1


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


MIGRATIONS: list[tuple[int, list[str]]] = [
    (
        1,
        [
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS machine_profiles (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT,
                architecture TEXT,
                quantization TEXT,
                parameter_count INTEGER,
                context_max INTEGER,
                license TEXT,
                source_url TEXT,
                task TEXT,
                verified_at TEXT,
                added_at TEXT NOT NULL,
                meta TEXT NOT NULL DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS runtime_profiles (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                name TEXT NOT NULL,
                config TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                provider TEXT NOT NULL,
                status TEXT NOT NULL,
                rights_status TEXT NOT NULL DEFAULT 'needs_review',
                record_count INTEGER NOT NULL DEFAULT 0,
                manifest TEXT NOT NULL DEFAULT '{}',
                stats TEXT NOT NULL DEFAULT '{}',
                holdout_hash TEXT,
                created_at TEXT NOT NULL,
                frozen_at TEXT,
                notes TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dataset_records (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                record_key TEXT NOT NULL,
                split TEXT NOT NULL DEFAULT 'unassigned',
                synthetic INTEGER NOT NULL DEFAULT 0,
                excluded INTEGER NOT NULL DEFAULT 0,
                exclusion_reason TEXT,
                rights_status TEXT NOT NULL DEFAULT 'needs_review',
                content_hash TEXT NOT NULL,
                simhash INTEGER,
                payload TEXT NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_records_dataset ON dataset_records(dataset_id, split, excluded)",
            "CREATE INDEX IF NOT EXISTS idx_records_hash ON dataset_records(dataset_id, content_hash)",
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                message TEXT,
                owner TEXT NOT NULL DEFAULT 'user',
                approval_id TEXT,
                input TEXT NOT NULL DEFAULT '{}',
                output TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC)",
            """
            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                kind TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_job_events ON job_events(job_id, seq)",
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id TEXT PRIMARY KEY,
                suite_id TEXT NOT NULL,
                suite_version TEXT NOT NULL,
                model_id TEXT,
                model_hash TEXT,
                status TEXT NOT NULL,
                comparability TEXT NOT NULL DEFAULT 'project-specific smoke',
                config TEXT NOT NULL DEFAULT '{}',
                results TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                finished_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS benchmark_cases (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                detail TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(run_id) REFERENCES benchmark_runs(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_bench_cases ON benchmark_cases(run_id)",
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY,
                job_id TEXT,
                dataset_id TEXT,
                base_model TEXT,
                backend TEXT,
                status TEXT NOT NULL,
                config TEXT NOT NULL DEFAULT '{}',
                metrics TEXT NOT NULL DEFAULT '{}',
                checkpoint_dir TEXT,
                created_at TEXT NOT NULL,
                finished_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS upgrade_candidates (
                id TEXT PRIMARY KEY,
                parent_run_id TEXT,
                dataset_id TEXT,
                status TEXT NOT NULL,
                config TEXT NOT NULL DEFAULT '{}',
                metrics TEXT NOT NULL DEFAULT '{}',
                decision TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                decided_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lineage (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                ref TEXT NOT NULL,
                parent_ref TEXT,
                detail TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                autonomy TEXT NOT NULL,
                status TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT '[]',
                result TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                finished_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                action TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                granted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                decided_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS identity_profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                card TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                dataset_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}'
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_activity_ts ON activity(ts DESC)",
        ],
    ),
]


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._local = threading.local()

    # -- connections -------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- schema ------------------------------------------------------------
    def migrate(self) -> int:
        conn = self.connect()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
        current = int(row["value"]) if row else 0
        for version, statements in MIGRATIONS:
            if version <= current:
                continue
            conn.execute("BEGIN")
            try:
                for statement in statements:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(version),),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            current = version
        return current

    # -- helpers -----------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        conn = self.connect()
        if conn.in_transaction:
            conn.executemany(sql, rows)
            return
        conn.execute("BEGIN")
        try:
            conn.executemany(sql, rows)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.connect().execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()

    def insert(self, table: str, values: dict[str, Any]) -> None:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        self.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values()))

    # -- settings ----------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM settings WHERE key=?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO settings(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value), utc_now()),
        )

    def all_settings(self) -> dict[str, Any]:
        return {row["key"]: json.loads(row["value"]) for row in self.query("SELECT * FROM settings")}

    def activity(self, kind: str, message: str, data: dict | None = None) -> None:
        self.execute(
            "INSERT INTO activity(ts, kind, message, data) VALUES(?,?,?,?)",
            (utc_now(), kind, message, json.dumps(data or {})),
        )


def init_db(path: Path | str) -> Database:
    db = Database(path)
    db.migrate()
    return db
