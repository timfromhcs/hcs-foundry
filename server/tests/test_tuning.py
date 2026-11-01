from __future__ import annotations

from pathlib import Path

from foundry.tuning import sweep


class FakeServer:
    def __init__(self):
        self.ngl = 0
        self.loaded = False

    def load(self, *, model_id, model_path, binary, config, alias=None):
        self.ngl = config.gpu_layers
        self.loaded = True
        return {"status": "ready"}

    def unload(self):
        self.loaded = False
        return {"status": "stopped"}

    def complete(self, messages, **kwargs):
        speed = 20.0 if self.ngl == 0 else 60.0
        return {"text": "ok", "metrics": {"tokens_per_second": speed, "ttft_seconds": 0.2}}


def test_sweep_picks_fastest_and_stores_profile(db, tmp_path: Path):
    model = {"id": "mdl_x", "name": "Test", "path": str(tmp_path / "m.gguf")}
    result = sweep(
        db,
        FakeServer(),
        model=model,
        binary=str(tmp_path / "llama-server.exe"),
        gpu_layers=[0, 99],
        batches=[512],
    )
    assert result["best"]["gpu_layers"] == 99
    assert result["best"]["tokens_per_second"] == 60.0
    assert result["cpu_baseline"]["tokens_per_second"] == 20.0
    assert result["speedup_vs_cpu"] == 3.0
    profiles = db.get_setting("vulkan_profiles")
    assert profiles["mdl_x"]["gpu_layers"] == 99


def test_sweep_records_failures(db, tmp_path: Path):
    class Broken(FakeServer):
        def load(self, **kwargs):
            raise RuntimeError("no vulkan device")

    result = sweep(
        db,
        Broken(),
        model={"id": "mdl_y", "name": "Broken", "path": str(tmp_path / "m.gguf")},
        binary="x",
        gpu_layers=[99],
    )
    assert result["best"] is None
    assert result["points"][0]["ok"] is False
    assert "no vulkan device" in result["points"][0]["error"]
