from __future__ import annotations

from pathlib import Path

import pytest

from foundry import gguf_tools


def test_quantize_rejects_unknown_method(tmp_path: Path, settings):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"x" * 16)
    toolchain = gguf_tools.Toolchain(converter=None, quantize="llama-quantize", split=None, source_root=None)
    with pytest.raises(gguf_tools.GGUFError) as exc:
        gguf_tools.quantize_gguf(source, tmp_path / "out.gguf", "Q9_Z", toolchain=toolchain)
    assert exc.value.code == "bad_method"


def test_quantize_requires_a_quantizer(tmp_path: Path, settings):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"x" * 16)
    toolchain = gguf_tools.Toolchain(converter=None, quantize=None, split=None, source_root=None)
    with pytest.raises(gguf_tools.GGUFError) as exc:
        gguf_tools.quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M", toolchain=toolchain)
    assert exc.value.code == "no_quantizer"


def test_export_requires_converter(tmp_path: Path, settings):
    source = tmp_path / "hf"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"x" * 32)
    toolchain = gguf_tools.Toolchain(converter=None, quantize=None, split=None, source_root=None)
    with pytest.raises(gguf_tools.GGUFError) as exc:
        gguf_tools.export_gguf(source, tmp_path / "out.gguf", toolchain=toolchain)
    assert exc.value.code == "no_converter"


def test_export_requires_weights(tmp_path: Path, settings):
    source = tmp_path / "hf"
    source.mkdir()
    (source / "config.json").write_text("{}")
    toolchain = gguf_tools.Toolchain(converter="convert.py", quantize=None, split=None, source_root=None)
    with pytest.raises(gguf_tools.GGUFError) as exc:
        gguf_tools.export_gguf(source, tmp_path / "out.gguf", toolchain=toolchain)
    assert exc.value.code == "no_weights"


def test_quantize_records_result(monkeypatch, tmp_path: Path):
    source = tmp_path / "model.gguf"
    source.write_bytes(b"x" * 1000)
    toolchain = gguf_tools.Toolchain(converter=None, quantize="llama-quantize", split=None, source_root=None)

    def fake_run(cmd, **kwargs):
        Path(cmd[2]).write_bytes(b"y" * 250)
        return 0

    monkeypatch.setattr(gguf_tools, "_run_streaming", fake_run)
    result = gguf_tools.quantize_gguf(source, tmp_path / "out.gguf", "Q4_K_M", toolchain=toolchain)
    assert result["size_bytes"] == 250
    assert result["ratio"] == 0.25
    assert result["upcast"] is False


def test_discover_returns_public_shape(settings):
    toolchain = gguf_tools.discover(settings)
    public = toolchain.to_public()
    assert set(public) == {"converter", "quantize", "split", "source_root"}
