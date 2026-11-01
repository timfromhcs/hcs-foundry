from __future__ import annotations

from pathlib import Path

from foundry.data.sources import Source, _files, _prepare_files, scan


def test_scan_reports_known_providers(settings):
    result = scan(settings)
    providers = {item["provider"] for item in result}
    assert {"claude_code", "codex", "antigravity", "opencode"} <= providers
    for item in result:
        assert "exists" in item and "files" in item and "bytes" in item


def test_files_skips_transcript_chunks(tmp_path: Path):
    root = tmp_path / "brain"
    chunks = root / "session" / "chunks"
    chunks.mkdir(parents=True)
    (root / "session" / "transcript.jsonl").write_text("{}")
    (chunks / "transcript-1.jsonl").write_text("{}")
    source = Source(provider="antigravity", label="x", kind="files", roots=[root], patterns=["transcript*.jsonl"])
    files = _files(source)
    assert len(files) == 1
    assert "chunks" not in files[0].parts


def test_compressed_sessions_are_decompressed(settings, tmp_path: Path):
    import zstandard

    root = tmp_path / "sessions"
    root.mkdir()
    payload = b'{"type":"user"}\n'
    compressed = root / "rollout.jsonl.zst"
    compressed.write_bytes(zstandard.ZstdCompressor().compress(payload))
    source = Source(provider="codex", label="x", kind="files", roots=[root], patterns=["*.jsonl.zst"])
    prepared = _prepare_files(source, settings)
    assert len(prepared) == 1
    assert prepared[0].read_bytes() == payload
    assert compressed.exists(), "the original must never be modified"
