# Install on macOS

The macOS provider uses `system_profiler` for the GPU and the same `llama-server` discovery as the other platforms. Apple Silicon uses unified memory, so the governor treats reported GPU memory as shared, never as dedicated VRAM.

## 1. Prerequisites

```bash
brew install python node git cmake
python3 --version   # 3.11+
```

## 2. Backend

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m foundry.cli serve --port 8770
```

## 3. Frontend

```bash
cd web
npm install
npm run build
```

## 4. Runtime

Build `llama-server` with Metal:

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build -j
```

Register `build/bin/llama-server` in **Machine → Inference runtimes**.

## 5. Training

`torch` on Apple Silicon uses MPS. The Training Lab only reports `gpu` when `torch.cuda.is_available()` is true; MPS is currently reported as `cpu-experimental` and is **not** claimed as a verified training backend. This is deliberate: an untested backend must not be presented as working.

## Notes

- Codesigning/notarisation is not handled by this repository; run from source.
- Unified memory means the governor budgets against free system RAM. Close memory-hungry apps before loading large models.
