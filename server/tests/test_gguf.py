from __future__ import annotations

import pytest

from foundry.runtime.gguf import GGUFError, read_gguf

from .conftest import write_minimal_gguf


def test_reads_metadata_and_parameters(gguf_file):
    info = read_gguf(gguf_file)
    assert info.architecture == "llama"
    assert info.quantization == "Q4_K_M"
    assert info.context_length == 4096
    assert info.block_count == 4
    assert info.head_count_kv == 4
    assert info.parameter_count == 256 * 64
    assert info.kv_bytes_per_token() == 2 * 4 * 4 * 32 * 2


def test_rejects_non_gguf(tmp_path):
    path = tmp_path / "nope.gguf"
    path.write_bytes(b"NOTGGUF" + b"\x00" * 32)
    with pytest.raises(GGUFError):
        read_gguf(path)


def test_rejects_truncated_header(tmp_path):
    path = write_minimal_gguf(tmp_path / "tiny.gguf")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(GGUFError):
        read_gguf(path)


def test_handles_unknown_architecture(tmp_path):
    path = write_minimal_gguf(tmp_path / "weird.gguf", architecture="customarch", context=2048, layers=2)
    info = read_gguf(path)
    assert info.architecture == "customarch"
    assert info.context_length == 2048
