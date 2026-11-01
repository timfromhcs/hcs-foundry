from __future__ import annotations

import sqlite3

from foundry.db import Database, init_db


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "db.sqlite"
    db = init_db(path)
    version = db.migrate()
    assert version >= 1
    assert db.migrate() == version
    db.close()


def test_wal_and_foreign_keys(tmp_path):
    db = init_db(tmp_path / "db.sqlite")
    conn = db.connect()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_settings_roundtrip(tmp_path):
    db = init_db(tmp_path / "db.sqlite")
    db.set_setting("model_roots", ["/tmp/a", "/tmp/b"])
    assert db.get_setting("model_roots") == ["/tmp/a", "/tmp/b"]
    db.set_setting("model_roots", ["/tmp/c"])
    assert db.get_setting("model_roots") == ["/tmp/c"]


def test_delete_dataset_cascades(tmp_path):
    db = init_db(tmp_path / "db.sqlite")
    db.insert(
        "datasets",
        {
            "id": "ds_1",
            "name": "x",
            "provider": "generic",
            "status": "review",
            "rights_status": "confirmed",
            "record_count": 1,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    db.insert(
        "dataset_records",
        {
            "id": "row_1",
            "dataset_id": "ds_1",
            "record_key": "rec_1",
            "split": "train",
            "synthetic": 0,
            "excluded": 0,
            "rights_status": "confirmed",
            "content_hash": "sha256:x",
            "simhash": 0,
            "payload": "{}",
        },
    )
    db.execute("DELETE FROM datasets WHERE id=?", ("ds_1",))
    assert db.query_one("SELECT COUNT(*) AS c FROM dataset_records")["c"] == 0


def test_interrupted_job_recovery(tmp_path):
    from foundry.events import EventBus
    from foundry.jobs import JobRunner

    db = init_db(tmp_path / "db.sqlite")
    runner = JobRunner(db, EventBus())
    db.insert(
        "jobs",
        {
            "id": "job_1",
            "type": "x",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
            "input": "{}",
            "output": "{}",
        },
    )
    assert runner.recover_interrupted() == 1
    assert db.query_one("SELECT status FROM jobs WHERE id='job_1'")["status"] == "interrupted"
