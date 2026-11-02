# Known limitations - HCS Foundry 7.0.0

Honest list as of 2026-11-01. Anything not listed as working should be assumed
unimplemented. Every statement here is either measured on this machine or explicitly
marked as untested.

## Hardware-derived

1. **No dedicated VRAM.** The host has an integrated Radeon 680M with unified memory. The
   governor never reports the ~4.3 GB driver value as VRAM. Large models are slow and
   memory-tight.
2. **No CUDA/ROCm.** PyTorch has no Vulkan backend, so training is CPU-only. Vulkan is an
   inference and tuning path, never a training backend.
3. **Thermal/power data** is only available where the OS exposes it; GPU power is treated
   as informational, not authoritative.
4. **A quality 4B fine-tune is out of reach.** The pipeline runs end to end at 4B scale
   (see `docs/evidence/v41_int4_4b.md`) for a few optimisation steps on this CPU. That
   proves the mechanism, not the model quality.

## Runtime

5. **One inference server at a time.** The supervisor enforces a single loaded model;
   parallel models are not supported.
6. **KV-cache quantization** is limited to what the discovered `llama-server` build exposes
   (`--cache-type-k/v`). If the flag is rejected the UI falls back to FP16 and says so.
7. **TurboQuant** is a capability probe only. No discovered build exposes it, so it reports
   `not available on this build`.
8. **Vision** works only for models with a matching `mmproj` file; the UI refuses to claim
   vision support otherwise.

## Data

9. **Adapters cover observed formats.** Claude Code, Codex CLI, OpenCode and Antigravity
   on-disk schemas are provider-internal and may change; adapters are versioned and tolerant
   of unknown fields, but a provider change can still produce "unknown record" counts.
10. **Near-duplicate detection** is SimHash-based with a fixed threshold; it is a heuristic,
    not a proof.
11. **PII/secret detection** is regex+entropy based. It reduces risk, it does not guarantee
    complete redaction. Review is mandatory before training export.
12. **Reasoning annotations** are evidence-grounded summaries, never a model's hidden
    reasoning. Rows that cannot be grounded are labelled `unknown`.

## Training

13. **GGUF is not a training format.** GGUF stays an inference/distribution format. Training
    runs on safetensors checkpoints; `train merge-int4` rebuilds fp16 safetensors from an INT4
    store and the GGUF export writes from that directory.
14. **Merging an INT4 adapter needs the disk-backed store.** An INT4 run started without
    `--offload` has no `int4-store/`, so the adapter cannot be merged; the trainer says so
    instead of pretending. Merging also needs the original base directory for the tensors the
    store skipped (vision tower, MTP head).
15. **0.6B, 1.7B and the 4B merge chain are proven end-to-end.** 0.6B and 1.7B go through
    merge → GGUF → quantize → answer; the 4B adapter merges to 8.70 GiB fp16 and exports to an
    8264 MiB F16 GGUF plus a 2655 MiB Q4_K_M that answers at 8.75 tok/s. What is **not** proven
    for the 4B is a *training* step under memory pressure: it degenerates to disk speed
    (378 s/step vs ~25 s in `docs/evidence/v7_4b_speed.md`).
16. **The disk-backed INT4 store trades throughput for memory.** Weights are memory-mapped
    from `weights.u8` / `scales.f32` / `rest.f32`, so they occupy file-backed pages instead of
    anonymous RAM. The first pass reads from disk and is slower than the in-RAM path; the
    OS page cache is the RAM buffer, not a guarantee of residency.
17. **Checkpoint key wrappers.** Multimodal checkpoints (`model.language_model.*`,
    `model.visual.*`, `mtp.*`) are normalised by dropping the wrapper segment and skipping
    unresolved subtrees. The skip is reported in the manifest, and a hard meta-device check
    fails the load if a *needed* weight was missing — but a text-only run genuinely does not
    use the vision tower.
18. **Checkpoint resume** works for the local SFT/INT4 trainers; interoperability with
    external trainers is not guaranteed.
19. **The preflight is an estimate.** `estimate_int4_resources` predicts the envelope; the
    manifest's `memory.peak_rss_mb` is the measured truth and can differ.

## Benchmarks

20. **Local suites only.** Scores are labelled `project-specific smoke` and are **not**
    comparable to leaderboards.
21. **No official benchmark assets** (SWE-bench, Terminal-Bench, OSWorld, WebArena, GAIA,
    MMMU) are bundled or downloaded. The UI shows them as "not run" with a link to the
    official instructions.
22. **Generated-code execution** is sandboxed by a subprocess with a timeout, a temp working
    directory and an optional per-case venv — not by a VM or container. Do not treat it as a
    hardened sandbox.

## Agent

23. **Allowlisted tools only.** No arbitrary shell. The agent cannot install packages, push,
    or touch paths outside the workspace.
24. **MCP exposes nine tools, not the whole product.** `machine_scan`, `machine_budget`,
    `models_list`, `datasets_list`, `datasets_export`, `vram_status`, `train_int4`,
    `train_merge_int4`, `gguf_quantize`. There is deliberately no shell tool; running a prompt
    (`models_run`) and running a benchmark (`bench_run`) are not exposed yet (v8 plan, Phase 9).

## Deployment

25. **Vercel hosts the landing page only** — a static page, no functions, no inference, no
    training. The local service binds to `127.0.0.1`; a deployed page reaches it only through
    the explicit pairing flow.
26. **A public page reaching loopback depends on the browser.** Chromium applies its Local
    Network Access permission and may refuse the request outright ("Permission was denied for
    this request to access the `loopback` address space") even though the service answers the
    Private Network Access preflight. The page reports that instead of pretending. The local UI
    on `http://127.0.0.1:8770` is not affected.
27. **The landing page is one static file.** It has no CMS and no build step; edits are
    committed and redeployed.
28. **Deployment credentials** are environment-specific; if absent, the release stops before
    deployment and says so.

## Packaging

29. **The executable excludes the ML stack.** Fine-tuning requires the source install; the
    EXE reports `requires_stack` instead of failing silently.
30. **The standalone build is Windows-only** in this release. The same PyInstaller spec can
    be adapted for Linux/macOS but was not tested.

## Language

31. **de/en/fr are complete for all shipped UI strings.** Additional locales are
    architecture-ready but not translated.
