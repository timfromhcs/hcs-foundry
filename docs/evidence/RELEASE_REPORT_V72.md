# Release report — HCS Foundry v7.2.0 (browser runtime)

**Date:** 2026-11-02
**Tag:** `v7.2.0` → commit `e0faf3d`
**Plan deviation:** this release ships the browser runtime instead of phases 4–6 of `PLAN.md`
(`PG-2`), on explicit request. Phases 4–6 stay open.

## Why this release exists

The request was: *"make WebAssembly/WebGPU in the browser as far as it goes — not just a download
page and ads — native local stays better, with automatic hardware detection and safe limits."*

So the browser path is a **real runtime**, not a landing page:

1. It runs small ONNX models **in the tab** through Transformers.js, with WebGPU when the browser
   grants an adapter and WASM otherwise.
2. Models are served by the local service (`data/browser-models/<id>/`), never by a CDN:
   `env.allowRemoteModels = false`.
3. Limits are computed from live capabilities and every number is printed with its reason.
4. The native runtime stays the better path, and the UI says so instead of implying parity.

## Delivered

| Area | What |
|---|---|
| Detection | WebGPU adapter + `maxBufferSize` / `maxStorageBufferBindingSize`; validated WebAssembly probes for SIMD and threads; `SharedArrayBuffer`; `crossOriginIsolated`; `deviceMemory`; `hardwareConcurrency`; `navigator.storage.estimate()`; secure context |
| Limits | WebGPU → `q4`, context 2048, model ≤ 3 GB. WASM → `q8` (`fp32` without SIMD), context 512, model ≤ **1.25 GB**. Threads only when the page is cross-origin isolated. Blockers refuse the run with numbers |
| Serving | `GET /api/v1/browser/runtime` (inventory, readiness, missing files) and `GET /api/v1/browser/models/{id}/{path}` (path-traversal checked, `no-store`, `nosniff`) |
| UI | `BrowserRuntimePanel` in *Playground*, rendered with and without a loaded GGUF model: chips, limits with reasons, model picker, load progress, streaming generation, tok/s, cancel |
| Stress test | `web/scripts/browser_stress.mjs`: pairs, reads the panel, probes what the tab can really allocate, optionally generates, exits non-zero on fake support or console errors |
| Docs | `docs/BROWSER_RUNTIME.md`; README row + limitation; `KNOWN_LIMITATIONS.md` 32–34; `UI_GUIDE.md` Playground section |

## Fixed on the way

- **i18n interpolation never worked.** Dictionaries use single braces; i18next defaulted to double.
  Six keys — including the pre-existing `overview.jobs_running` — rendered `{count}` literally.
  The prefix/suffix is now explicit and `locales.test.ts` guards placeholder parity.
- **WASM SIMD was always reported as missing.** The probe module was invalid
  (`WebAssembly.validate` → false); it is now a valid minimal module, verified in Node and Chromium.
  The unused relaxed-SIMD probe was removed.
- **The panel was unreachable.** `Playground` returned early without a loaded model, hiding a
  fallback UI behind the very runtime it is meant to replace.
- **`GET /machine/profile` answered 404 on a fresh install.** The UI already treated it as "not
  scanned yet", but Chromium logs a 404 resource load as a console error, so a clean install left a
  red line in the console. The endpoint now returns
  `200 {"profile": null, "budget": null, "scanned": false}` — the same tolerance `/machine/budget`
  already had. The new E2E test, which asserts zero console errors, is what caught it on CI.

## Proof

```
backend (windows, py3.14)        269 passed (5 new: inventory, empty state, traversal, serve,
                                 machine profile without a scan)
frontend (vitest)                 17 passed (10 new: limits, ceilings, refusals, formatBytes)
E2E (playwright, :8770)           10 passed (1 new: live chips + refusal without a model)
E2E on a fresh data dir (:8771)   browser_stress: no console errors (reproduces the CI condition)
build                             index 480 KB JS, transformers chunk lazy (566 KB), ort wasm 26 MB
stress (headless chromium)        WASM · SIMD yes · threads no · 16 cores · 16 GB device memory
                                  limits 1.25 GB / 512 / q8 / 1 thread
                                  allocation probe granted 1536 MiB to the tab
                                  console errors: none
```

The 1.25 GB WASM ceiling is derived from that 1536 MiB measurement, not guessed.

## Not claimed

- **Browser token throughput is not measured.** No ONNX export was installed in this session
  (`Xenova/tiny-random-GPT2` and `onnx-community/SmolLM2-135M-Instruct` both answered as gated or
  private). The stress test prints `skipped` with the reason; `docs/BROWSER_RUNTIME.md` says the
  same. The load/generate path is wired and type-checked, but a numbers-free claim would be
  dishonest.
- **No parity with llama.cpp.** No Vulkan, no GGUF, no KV-cache quantization beyond ONNX Runtime,
  no 2–4B models in the tab.
- **Phases 4–6 of the v8 plan were not started** (generators, flywheel, on-policy distillation).
  `PLAN.md` records the deviation.
- CI on GitHub is reported by its run id below; local numbers are from this Windows machine.

## Publication

| Item | Value |
|---|---|
| Tag | `v7.2.0` |
| Commit | `e0faf3d` |
| CI | run `41446876386` |
| Landing page | <https://hcs-foundry-site.vercel.app> |
| EXE | rebuilt from the tagged commit |
