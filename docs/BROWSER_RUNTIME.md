# Browser runtime — WebGPU / WASM, honestly

The app can run a small model **inside the tab**. That is genuinely useful (a quick check without
starting the native service, a demo on a machine where nothing is installed), and it is **not
parity** with the local runtime. This page says exactly where the line is.

## How it works

```
data/browser-models/<id>/          ← you put an ONNX export here (one folder per model)
        onnx/model_q4.onnx         ← weights (Transformers.js dtype mapping)
        tokenizer.json, config.json
                 │
                 ▼
GET /api/v1/browser/runtime            list + readiness (missing files are named)
GET /api/v1/browser/models/<id>/<file> serve one file, path-checked, no-store
                 │
                 ▼
Playground → "Browser runtime" card → detection → safe limits → load → generate
```

No CDN, no telemetry, no upload: the SPA fetches the model from `127.0.0.1`, and Transformers.js
is configured with `allowRemoteModels = false`.

## Install a model

```bash
# any ONNX export works; this one is 135M parameters and runs in a browser tab
python -m pip install optimum[onnxruntime]
optimum-cli export onnx --model HuggingFaceTB/SmolLM2-135M-Instruct \
  data/browser-models/smollm2-135m
```

The folder must contain `tokenizer.json` and at least one `*.onnx`; the endpoint reports what is
missing instead of pretending the model is ready.

## What is detected (live, in the browser)

| Signal | Why it matters |
|---|---|
| `navigator.gpu.requestAdapter()` + adapter limits (`maxBufferSize`, `maxStorageBufferBindingSize`) | WebGPU is the only path with usable speed; tiny buffers make a transformer layer impossible |
| WebAssembly SIMD (validated module probe) | without SIMD expect 3–5× slower |
| WASM threads (`SharedArrayBuffer` + validated probe + `crossOriginIsolated`) | multi-threaded matmuls; needs COOP/COEP |
| `navigator.deviceMemory`, `hardwareConcurrency` | coarse memory bucket and worker count |
| `navigator.storage.estimate()` | whether a model can even be cached |
| secure context | WebGPU/WASM features require https or localhost |

## Safe limits and why they are what they are

The panel prints every number with its reason. Defaults:

| Backend | dtype | max model | context | threads |
|---|---|---|---|---|
| WebGPU | q4 | min(3 GB, ½ deviceMemory) | 2048 | — (GPU) |
| WASM | q8 (fp32 without SIMD) | **1.25 GB** | 512 | min(4, cores−1) if cross-origin isolated, else 1 |

The 1.25 GB WASM ceiling is not a guess: a bounded allocation probe in this Chromium granted
**1536 MiB** to the tab (`web/scripts/browser_stress.mjs`), so anything close to that would fail
mid-load. A model above the limit is refused with numbers, not attempted.

## Measured on this machine (2026-11-02)

```
backend      WASM (headless Chromium: "no adapter was granted")
SIMD         yes · threads no (not cross-origin isolated) · 16 cores · 16 GB device memory
limits       model up to 1.25 GB · context 512 · dtype q8 · 1 thread
allocation   tab granted 1536 MiB
inference    skipped — no ONNX model is installed under data/browser-models/
console      no errors
```

Reproduce: `cd web && node scripts/browser_stress.mjs`.

## What this is not

- **Not parity with llama.cpp.** No Vulkan, no GGUF, no KV-cache quantization beyond what ONNX
  Runtime offers, no 2–4B. The browser is for small models and quick checks.
- **Not a download funnel.** Nothing is fetched from the internet at runtime; if the folder is
  empty the UI says so and shows the export command.
- **Not measured end to end yet.** The engine path (Transformers.js with `q4`/`q8`) is wired and
  the limits are real, but no ONNX model was installed in this session, so token throughput in the
  browser is **not measured**. The stress test prints `skipped` with the reason rather than a
  fabricated number.

## Tests

```
server/tests/test_browser_runtime.py     4 passed   inventory, readiness, path traversal, serve
web/src/lib/browserRuntime.test.ts      10 passed   limits, ceilings, refusals, formatBytes
web/tests/e2e/browser-runtime.spec.ts    1 passed   panel renders, refuses without a model
web/scripts/browser_stress.mjs                     capability + allocation probe, exit code
```
