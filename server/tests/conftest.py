"""Shared pytest fixtures."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundry.config import Settings  # noqa: E402
from foundry.db import init_db  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    config = Settings(
        data_dir=data,
        model_roots=[tmp_path / "models"],
        import_roots=[FIXTURES, tmp_path / "imports"],
    )
    config.ensure_dirs()
    return config


@pytest.fixture()
def db(settings: Settings):
    return init_db(settings.db_path)


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES


def write_minimal_gguf(path: Path, *, architecture: str = "llama", context: int = 4096, layers: int = 4) -> Path:
    """Write a valid GGUF header (metadata + tensor info, no tensor data)."""
    meta = {
        "general.architecture": architecture,
        "general.name": "Tiny Test Model",
        "general.file_type": 15,  # Q4_K_M
        "general.license": "mit",
        f"{architecture}.context_length": context,
        f"{architecture}.block_count": layers,
        f"{architecture}.embedding_length": 256,
        f"{architecture}.attention.head_count": 8,
        f"{architecture}.attention.head_count_kv": 4,
        f"{architecture}.attention.key_length": 32,
    }

    def s(value: str) -> bytes:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", 1)  # tensor count
    out += struct.pack("<Q", len(meta))

    for key, value in meta.items():
        out += s(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + s(value)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", int(value))

    # one tensor: token_embd.weight, 2 dims, type 0, offset 0
    out += s("token_embd.weight")
    out += struct.pack("<I", 2)
    out += struct.pack("<Q", 256)
    out += struct.pack("<Q", 64)
    out += struct.pack("<I", 0)
    out += struct.pack("<Q", 0)

    path.write_bytes(bytes(out))
    return path


@pytest.fixture()
def gguf_file(tmp_path: Path) -> Path:
    return write_minimal_gguf(tmp_path / "tiny.gguf")


@pytest.fixture()
def api(settings: Settings):
    from fastapi.testclient import TestClient

    from foundry.main import create_app

    app = create_app(settings, serve_web=False)
    with TestClient(app, base_url="http://127.0.0.1:8770") as client:
        response = client.post(
            "/api/v1/system/pair",
            json={"label": "test"},
            headers={"X-Foundry-Client": "1"},
        )
        token = response.json()["token"]
        client.headers.update({"Authorization": f"Bearer {token}", "X-Foundry-Client": "1"})
        yield client


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
