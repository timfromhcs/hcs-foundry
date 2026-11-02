# Changelog

All notable changes to HCS Foundry.

## [7.2.0] — 2026-11-02

Browser runtime release: a small model can now run **inside the tab** (WebGPU/WASM via
Transformers.js), fed from the local service instead of a CDN. Detection and limits are measured,
not advertised.

### Added

- **Browser runtime panel** in *Playground* (`web/src/components/BrowserRuntimePanel.tsx`). It shows
  live capability chips (WebGPU adapter, WASM SIMD, threads, cores, device memory, secure context),
  the computed safe limits with a reason for every number, the browser-model inventory, and a
  load/generate path with streaming output and tok/s. It is rendered with and without a loaded GGUF
  model.
- **Capability detection** (`web/src/lib/browserRuntime.ts`): `navigator.gpu.requestAdapter()` plus
  `maxBufferSize`/`maxStorageBufferBindingSize`, validated WebAssembly module probes for SIMD and
  threads, `SharedArrayBuffer`, `crossOriginIsolated`, `deviceMemory`, `hardwareConcurrency`,
  `navigator.storage.estimate()`.
- **Safe limits as rules, not magic values**: WebGPU → `q4`, context 2048, model ≤ 3 GB;
  WASM → `q8` (fp32 without SIMD), context 512, model ≤ **1.25 GB**; threads only when the page is
  cross-origin isolated. Models above the limit are refused with numbers instead of attempting a
  load that would kill the tab.
- **Local model serving** (`server/foundry/browser_runtime.py`): `GET /api/v1/browser/runtime`
  (inventory + readiness, naming missing files) and
  `GET /api/v1/browser/models/{id}/{path}` (one file, path-traversal checked, `no-store` +
  `nosniff`). Transformers.js runs with `allowRemoteModels = false`, so nothing is fetched from the
  internet at runtime.
- **Stress test** `web/scripts/browser_stress.mjs`: pairs with the service, reads the panel, probes
  how much memory the tab is actually granted, optionally runs a real generation, and exits non-zero
  when the panel errors or claims a backend it cannot deliver.
- `docs/BROWSER_RUNTIME.md` with the measured table and an explicit "not parity / not measured"
  section.

### Fixed

- **i18n placeholders never interpolated.** All three dictionaries use single braces (`{count}`),
  but i18next was configured with its default double-brace prefix, so six keys — including the
  pre-existing `overview.jobs_running` — rendered `{...}` literally. The interpolation prefix/suffix
  is now explicit and `locales.test.ts` keeps the placeholder sets aligned.
- **WASM SIMD was reported as unavailable in every browser.** The probe module was not valid
  WebAssembly (`WebAssembly.validate` returned false), so the panel always claimed "no SIMD". The
  probe is now a valid minimal module, verified in Node and Chromium; the unused relaxed-SIMD probe
  was removed.
- **The browser panel was unreachable.** `Playground` returned early when no model was loaded, so
  the card only existed after starting the native runtime — exactly backwards for a fallback UI.

### Measured (headless Chromium, this machine, 2026-11-02)

```
backend      WASM (no adapter granted to the headless browser)
SIMD         yes · threads no (page is not cross-origin isolated) · 16 cores · 16 GB device memory
limits       model ≤ 1.25 GB · context 512 · dtype q8 · 1 thread
allocation   tab granted 1536 MiB (bounded probe, released immediately)
inference    skipped — no ONNX model installed under data/browser-models/
console      no errors
```

Browser token throughput is **not measured**: no ONNX export was installed in this session, and the
stress test prints that instead of a number.

### Tests

- Backend 268 passed (4 new: inventory, honest empty state, path traversal, serve).
- Frontend 17 passed (10 new: limits, ceilings, refusals, `formatBytes`).
- E2E 10 passed (1 new: panel renders live capabilities and refuses without a model).

## [7.1.0] — 2026-11-02

Trust release: CI now tests the trainers, and the promotion gate uses statistics instead of a
fixed delta.

### Added

- **`foundry/train/stats.py`** — `wilson_ci`, `paired_compare` (exact McNemar for binary
  outcomes, clustered bootstrap otherwise), `required_n` (power planning), `pass_at_k`,
  `summarize`. 11 known-answer tests (`tests/test_stats.py`).
- **CI honesty guard** — `tests/conftest.py` fails when `FOUNDRY_REQUIRE_TRAIN=1` and a training
  test module was collected zero times, so a missing ML stack can no longer look like "green".
- **`server/requirements-lock.txt` / `requirements-train-lock.txt`** compiled for the CI
  interpreter, constrained by `server/constraints.txt` to the versions the evidence was produced
  with. CI installs from the lock.

### Changed (breaking for the Upgrade Lab)

- **`upgrade.promote()` no longer promotes on `delta ≥ threshold`.** It requires a paired result
  with `n ≥ 30`, a 95 % CI lower bound > 0, a practical minimum effect (3 pp default), no critical
  regression, an unchanged holdout and a human approver. `threshold=` is kept as a deprecated
  alias for `min_effect`. A six-case suite is now refused as *underpowered* — `test_upgrade.py`
  proves both the refusal and the promoted path.
- `web/dist/**` and `web/tsconfig.tsbuildinfo` are no longer tracked (build output);
  `scripts/build_exe.ps1` builds the SPA first and `web/vercel.json` builds from source.

### Fixed

- `KNOWN_LIMITATIONS.md` header and items 15/24 corrected (it still said 4.1.0 and "only 0.6B");
  `V7_PLAN.md` progress log set to the real status.

### Verified

- 233 backend tests locally, 0 skips with `FOUNDRY_REQUIRE_TRAIN=1`.
- Baseline evidence: `docs/evidence/v8_baseline.md` (versions, free RAM, durations).

## [7.0.0] — 2026-11-02

This release answers four asks: **why is the 4B slow**, **run it locally with commands and MCP**,
**a UI that shows what happens**, **a site and docs that are a product surface** — plus branding.

### Added

- **MCP server** (`foundry/mcp_server.py`, `foundry mcp serve`, `HCSFoundry.exe --mcp`): nine
  typed tools over JSON-RPC/stdio — `machine_scan`, `machine_budget`, `models_list`,
  `datasets_list`, `datasets_export`, `vram_status`, `train_int4`, `train_merge_int4`,
  `gguf_quantize`. Paths are validated against the configured roots; there is no shell tool.
  `docs/MCP.md` documents client config, examples and the safety model.
- **`foundry run doctor`**: one screen with machine, budget, runtime, stores and data, plus the
  single next action (and a warning when the INT4 stores outgrow the RAM).
- **Store prefetch guard** (`training/store_prefetch.py`): the page cache is warmed in a daemon
  thread, but only when the store is smaller than 60 % of available RAM — warming 4.36 GB that
  is immediately evicted is wasted disk time. The reason is recorded in the manifest.
- **Adaptive block size** (`choose_block_rows`): ~64 MB fp32 per block, clamped to 256 rows, and
  recorded per run in `manifest["tuning"]` (block, hidden size, available bytes, torch threads).
- **Data Studio distribution panel**: task types, languages and split sizes as bars, plus
  `median`/`max` record lengths.
- **Export formats**: `?format=jsonl|json|csv` and `?limit=N` for samples, wired to four buttons
  in the dataset pane (JSONL, JSON, CSV, 20 samples).
- **Tutorial checklist** on Overview (scan → model → load → data → train → playground) with a
  link into `docs/TUTORIAL.md`.
- **Brand assets**: `logo.svg`, `logo-mark.svg`, favicons (32/192/apple-touch) and a 1200×630
  `og.png`, generated by `scripts/make_brand_assets.py`; the app and the site use the same mark.
- **Site**: docs section (tutorial, MCP, UI guide, CLI, architecture, changelog), a generated
  changelog page (`scripts/make_site_changelog.py`), canonical/OG/Twitter tags, JSON-LD,
  `robots.txt` and `sitemap.xml`.
- **Docs**: `docs/TUTORIAL.md` (five walkthroughs), `docs/UI_GUIDE.md`, `docs/BRAND.md`.

### Research that changed the product

`docs/evidence/v7_4b_speed.md` + `scripts/store_io_bench.py`:

| Fact | Number |
|---|---|
| sequential store read (prefetch) | 105 MB/s |
| 4 KiB page-fault read, cold | 66.6 MB/s |
| same bytes once resident | 2 660 MB/s |
| one 4B step when the working set fits | ~25 s |
| one 4B step when it does not (seq 128) | **378 s** |

The 4B is I/O-bound on this machine, not compute-bound. The prefetch guard, the tuning report and
the doctor hint are the honest consequences; the byte-reduction work (bf16 leftovers, batched
per-layer reads) is documented as the next experiment instead of being claimed.

### Fixed

- `foundry vram status` and the new doctor report used field names that did not exist
  (`safe_bytes`, `runtimes.llama_server.path`), so they printed zeros/NO.
- The changelog renderer split wrapped list items into separate bullets and the sitemap had a
  wrong namespace.

### Not claimed

- No 4B speedup: the guard avoids wasted reads, it does not make the disk faster.
- No new training-quality claim; all runs remain two-step smoke runs on 53 records.

## [6.0.0] — 2026-11-02

Two asks: make the 2–4B class usable locally, and add "virtual RAM on HDD with smart use".

### Added

- **Tiered weight cache** (`foundry/training/vram.py`): hot RAM LRU with a hard byte budget,
  a memory-mapped arena file on disk (`data/cache/vram/arena.bin`, grown lazily in 64 MB steps),
  and the INT4 store as the cold tier. `Int4Linear` and its backward pass go through it, so a
  block can be reused across passes. Knobs: `--vram-gb`, `--vram-mode auto|off`,
  `FOUNDRY_VRAM_PREFETCH`, `FOUNDRY_INT4_BLOCK`. CLI `foundry vram status|clear`,
  API `GET /training/vram` + `POST /training/vram/clear`, `vram.stats` in every manifest.
- **`train int4 --store <dir>`** reuses an existing store instead of rebuilding it.
- **2B class proven end to end**: Qwen3-1.7B → INT4 training (loss 5.023 → 4.081, peak RSS
  4496 MB) → merge (3 shards, 3.80 GiB) → GGUF F16 (3881 MiB) → Q4_K_M (1223 MiB) → answers
  "Hallo Welt" at 51.4 tok/s (`docs/evidence/v6_2b.md`).
- **4B artifact proven**: the v4.1 4B adapter (200 layers) merges into 8.70 GiB of fp16 shards
  in 514 s (`docs/evidence/v6_4b.md`).

### Changed

- The cache is **off by default**. Measured on Qwen3-0.6B, same store, 2 steps: 20.7 s without,
  117.1 s with a 3 GB RAM tier, 63.5 s with arena + prefetch. Keeping fp32 blocks resident
  pushes the mapped INT4 store out of the page cache, so a hit costs a store re-read. Numbers in
  `docs/evidence/v6_vram.md`; the knob stays for machines where a miss is more expensive than a
  memcpy.
- Prefetching is opt-in (`FOUNDRY_VRAM_PREFETCH=1`); prefetching the immediately-next block
  duplicates work.

### Fixed

- A 6 GB `truncate` for the arena cost 47 s per run on NTFS; the file now grows on demand
  (0.10 s).
- `budget_bytes=0` meant "unlimited" instead of "cache disabled".
- The prefetch hit counter never incremented.

### Not claimed

- **4B training was not re-run today.** Two attempts stalled in page faults while the host sat at
  ~3.6 GB free / 89 % commit (other agent processes); no step completed, so no number is
  reported. The last complete 4B training run remains the v4.1 one.
- **4B GGUF export** depends on `convert_hf_to_gguf.py` support for the Qwen3.5-VL
  architecture; the result is recorded in `docs/evidence/v6_4b.md` either way.
- Activation checkpointing was planned and cancelled with a measurement (4B activations at
  seq 128 are ~84 MB against a multi-GB private peak).

## [5.0.0] — 2026-11-02

Turns the proven mechanisms into one usable path: a model that was trained with INT4 now
leaves the machine as a normal GGUF, and the deployed page can talk to the local service.

### Added

- **INT4 adapter → merged checkpoint** (`foundry/training/merge_int4.py`):
  `W = dequantize_int4(packed, scales) + (B @ A) * (alpha/r)` streamed into ≤1.5 GB fp16
  shards with an index, plus config/tokenizer. Non-target tensors (vision tower, MTP head)
  are taken from the original checkpoint, so the result is a complete HF directory.
  CLI `train merge-int4`, API `POST /training/merge-int4`, pipeline stage `merge_int4`.
- **The unified trainer now finishes the chain for INT4**: train → merge → export GGUF →
  quantize, instead of skipping merge with "cannot be merged".
- **Memory-bounded INT4 matmul**: `blocked_linear` / `blocked_grad_input` dequantize one
  row block at a time (default 256 rows, `FOUNDRY_INT4_BLOCK`) and write into the output
  buffer. Isolated measurement on the 4B MLP shape: transient 279 MB → 0 MB, no slowdown.
- **Pairing section on the landing page** (`/#pair`): probes the local service, pairs with a
  token kept in memory, explains the boundary, and reports "no local service answered"
  instead of pretending.
- **`FOUNDRY_UI_ORIGINS`** now parses URLs correctly (it went through `Path`, which collapsed
  `https://`); the deployed site origin is in the default allowlist.
- **Training Lab**: runs table shows the load mode (`fp32` / `streaming` / `disk offload`),
  the measured peak memory, and a "Merge → GGUF" action for completed INT4 runs (de/en/fr).

### Fixed

- **`VERSION` was hardcoded to 4.0.0** in `config.py`, so the API and UI reported the wrong
  version; it now derives from `foundry.__version__` (with a test that fails on drift).
- **Cross-origin preflights were rejected with 401** for authenticated paths, so a paired
  foreign page could never call the API; `OPTIONS` now reaches CORS after the origin check.
- **Private Network Access**: the service answers the browser's preflight with
  `Access-Control-Allow-Private-Network: true`. Chromium still applies its own Local Network
  Access permission for public → loopback, and the page now says exactly that.
- The deployment uploaded a stale `site/.vercel/output`; `.vercelignore` excludes it now.
- A `dl[hidden]` element was still visible because an inline `display: grid` overrode it.
- `vercel.json` no longer relies on `cleanUrls` for the second page (the rewrite 404'd).

### Verified

- 0.6B end-to-end: INT4 offload training (loss 5.4147 → 4.3085) → merge (196 layers + LoRA,
  1.41 GiB fp16, 29.8 s) → GGUF F16 (1439 MiB) → Q4_K_M (462 MiB, 13.8 s) → **loads and
  answers at 117 tok/s** (`docs/evidence/v5_loop.md`).
- 195 backend tests, 7 frontend tests, 9 Playwright E2E tests, CI green.
- Pairing verified in real Chromium from a loopback origin: "Paired. Token received for this
  tab only.", version 5.0.0, no console errors. From the public origin Chromium blocks the
  loopback request (Local Network Access); the page states that instead of faking success.
- Landing page: HTTP 200, `/#pair` present, offline state asserted in Playwright.

### Still open

- **4B end-to-end export** was not completed: the 4B merge/GGUF chain needs ~17 GB of disk
  writes and the converter's Qwen3.5 support is unverified; only 0.6B is proven.
- Official SWE-bench assets, Vulkan training, GGUF→HF conversion remain open.

## [4.1.0] — 2026-11-01

Closes the gaps the v4.0 release left open. No new product surface.

### Added

- **Disk-backed INT4 store** (`foundry/training/offload.py`): the quantized model is written
  once to `weights.u8` + `scales.f32` + `rest.f32` and loaded with `mmap`, so the weights are
  file-backed pages instead of anonymous RAM and the OS page cache becomes the RAM buffer.
  `train int4 --offload/--no-offload`, `POST /training/auto {offload}`, and `train unified`
  picks it automatically when fp32 and in-RAM INT4 both do not fit.
- **Measured memory in every INT4 manifest**: `memory.peak_rss_mb`, `start_rss_mb`,
  `min_available_mb` (psutil sampler in `foundry/training/memory.py`) next to the estimate.
- **Checkpoint key normalisation**: `model.language_model.*` is mapped onto the text-only
  module tree, unresolved subtrees (vision tower, MTP head) are skipped *and reported*, and
  the meta-device check still fails the load when a needed weight is missing.
- **Hybrid-architecture LoRA targets**: `in_proj_qkv`, `in_proj_z`, `out_proj` are part of the
  default INT4 targets, so Qwen3-Next/Qwen3.5-style linear-attention layers are quantized
  instead of staying fp32.
- **Stable landing site**: Vercel project `hcs-foundry-site`, production URL
  <https://hcs-foundry-site.vercel.app>, `.vercelignore` limited to `site/`.

### Fixed

- INT4 manifests claimed "loading still needs the fp32 peak" even for streaming runs.
- The INT4 memory estimate assumed 1.2 GiB of fp32 leftovers, which is wrong for models with
  a 248k vocabulary and a vision tower; it now derives the embedding size from the config.
- A store built with a different target set could be silently reused (identity now includes
  the requested targets).

### Verified

- 178 backend tests, 7 frontend unit tests, 7 Playwright E2E tests.
- 4B checkpoint (`Qwen3.8-4B-Distill`, 8.7 GB safetensors, 18.0 GiB in fp32) trained through
  the disk-backed store on a machine with ~6 GiB available RAM — numbers in
  `docs/evidence/v41_int4_4b.md`.
- Landing page: HTTP 200, assets 200, security headers, `<title>HCS Foundry — local AI model lab</title>`.

### Still open

- **Official SWE-bench**: dataset and checkouts are not bundled (`not_run` with reason).
- **Vulkan training**: still no PyTorch Vulkan backend; Vulkan stays an inference path.
- **Quality at 4B**: the pipeline is proven, the model quality is not.

## [4.0.0] — 2026-11-01

### Added

- **Streaming INT4 loader**: builds the model on `meta`, streams safetensors shards,
  quantizes target linears immediately, and refuses if any tensor is left on `meta`.
  `train int4 --streaming` and `POST /training/auto {algorithm:"int4", streaming:true}`.
- **Unified auto trainer** (`training/unified.py`): decides SFT vs streaming INT4 from
  model size, free RAM and the playbook, then runs train → merge → export GGUF →
  quantize and writes a report.
- **Full chat-log cleanup** (`data/cleanup.py`): discover → import all → redact →
  dedupe → quality filter → rights → freeze → quarantine copy → report. Originals are
  never modified; excluded records are marked, not deleted.
- **Repo-fix benchmark**: the model repairs a buggy file and the tests run in a
  per-case venv.
- **Agentic benchmark**: multi-step file tasks answered as JSON, applied in a sandbox,
  one task executed in a venv.
- **SWE-bench adapter**: official JSONL shape, per-instance repo copy + venv, command
  allowlist, and honest `not_run` when assets are missing.
- **Vulkan profile reuse**: `tune apply`, `/tune/profile/{model_id}`, and the tuned
  profile recorded in benchmark metadata.
- CLI `logs clean`, `train unified`, `tune apply`, `bench repo_fix|agentic`; API
  `/sources/clean`, `/training/unified`, `/tune/profile`; Pipeline page buttons for
  "Clean all logs" and "Full local train".

### Fixed

- Streaming INT4 computed `in_features` from the parent module instead of the linear,
  producing shape mismatches on Qwen-style models.
- CLI crash after the v4 command additions (indentation).
- Redaction now covers `vck_` gateway keys.

### Verified

- 165 backend tests, 7 frontend unit tests, 7 Playwright E2E tests.
- Cleanup: 79 records imported, 75 included, 4 excluded, frozen with hash and a
  quarantine copy; originals untouched.
- Streaming INT4 on 0.6B: 196 layers, loss 5.72 → 4.89.
- Repo-fix and agentic benchmarks pass on a deterministic fake server; SWE-bench
  instance passes in a venv.

### Blocked / pending

- **4B streaming INT4 run**: the 9.32 GB safetensors download did not finish in this
  session, so no 4B training claim is made.
- **Official SWE-bench**: dataset and checkouts are not bundled.
- **Vulkan training**: still no PyTorch Vulkan backend.

## [3.0.0] — 2026-11-01

### Added

- **GGUF export** from safetensors via the discovered `convert_hf_to_gguf.py`, with disk
  precheck, streaming log capture and cancellation.
- **Quantization** wrapper around `llama-quantize` for F16/BF16/Q8_0/Q6_K/Q5_K_M/Q4_K_M/
  Q4_K_S/Q3_K_M/Q2_K, with size/ratio recording.
- **GGUF upcast** to F16/BF16, explicitly labelled as requantization because no
  GGUF→HF converter exists in the toolchain.
- **MoEfication**: dense FFN → N experts with a top-k router, architecture-aware,
  forward-pass verified, with a memory projection that refuses impossible conversions.
- **INT4 (QLoRA-style) training** without bitsandbytes: group-wise absmax packing,
  `Int4Linear`, LoRA adapters in fp32, custom adapter artifact and manifest.
- **Training algorithms**: SFT, DPO with synthetic pairs (adapter-disabled reference),
  and distillation from a local or gateway teacher.
- **OpenEval** during training, best-checkpoint saving, **auto-decision** (LR reduction,
  early stop) and a **training playbook** that records measured recipes and seeds the
  next automatic run.
- **Vulkan auto-tuning** sweep over GPU layers and batch size, stored per model.
- **Local log discovery/import** for Claude Code, Codex (including zstd sessions),
  Antigravity transcripts and OpenCode (via `session export --sanitize`), with a
  guarantee that originals are never modified.
- **All-in-one pipeline** with ordered, resumable stages and a JSON + Markdown report.
- **Pipeline page** and `gguf`, `moe`, `tune`, `logs`, `pipeline`, `train int4/auto/dpo/distill`
  CLI commands; new `/gguf`, `/moe`, `/tune`, `/sources`, `/pipeline` API routes.
- `docs/plan/V3_PLAN.md`, `docs/evidence/v3_*.md`.

### Fixed

- Reasoning-only models returned an empty CLI answer; `complete()` now returns
  `reasoning` separately and the CLI prints it with a hint.
- `vercel_key` (`vck_…`) added to the redaction patterns.
- Codex `.jsonl.zst` sessions are decompressed into the import area instead of being
  silently skipped.
- Windows `.CMD` shims are executed through `cmd /c` for OpenCode imports.
- Antigravity transcripts get a dedicated parser that ignores private `thinking` fields.

### Verified

- 149 backend tests, 7 frontend unit tests, 7 Playwright E2E tests.
- 0.6B GGUF export + Q4_K_M quantization + inference (113 tok/s).
- Vulkan sweep: 51.4 → 107.8 tok/s (2.1×).
- Auto SFT on 72 imported records: eval 4.344 → 4.233, playbook entry recorded.
- INT4 SFT: 196 layers quantized, loss 5.72 → 4.86, adapter saved.
- 4B Q4_K_M loads and answers on Vulkan at context 512.

### Blocked (documented, not faked)

- Complete 4B/8B fine-tune on this machine (fp32 load peak ≈ 17.4 GB vs ~5.6 GB free).
- Vulkan-accelerated training (no PyTorch Vulkan backend).
- GGUF → HF safetensors conversion (no converter in the discovered toolchain).
- 4B MoE with four experts (projected memory exceeds RAM).

## [2.0.0] — 2026-11-01

### Added

- **Hugging Face Hub client**: search, file listing with sizes and LFS digests, resumable
  downloads, atomic rename, SHA-256 verification, disk precheck, cancellation, and a
  token store that is masked in every response. Public repos work anonymously.
- **Model Hub page**: live search, task/sort filters, file picker with suggested GGUF +
  projector selection, download queue and progress.
- **Standalone executable**: PyInstaller one-file build (50.8 MB) that bundles the API
  and the SPA, picks a free port, stores data in `%LOCALAPPDATA%`, and opens the browser.
- **CLI v2**: `hf search|files|pull|token`, `models list|scan|verify|load|unload|run`,
  `bench`, `train smoke|preflight|sft`, `datasets list`, all with `--json` and real exit
  codes. UTF-8 output on Windows.
- **Real LoRA/SFT backend**: transformers + PEFT, architecture-aware target modules,
  tokenisation from frozen datasets, live loss events, checkpoints, cooperative
  cancellation, manifest with model and dataset hashes. Verified on Qwen3-0.6B.
- **Training recipes** with explicit projections and a warning for 3B on CPU.
- **Command palette** (Ctrl/Cmd+K) and **mobile bottom navigation**.
- **Opt-in Vercel AI Gateway teacher** for annotations, disabled by default, with the
  billing error surfaced verbatim.
- `docs/HUB.md`, `docs/CLI.md`, `docs/STANDALONE.md`, `docs/plan/V2_PLAN.md`.

### Fixed

- Settings were lost after a restart; persisted overrides now load at startup.
- `runtime_profiles` returned 500 when the same model/context was loaded twice.
- Duplicate completion toasts.
- Benchmark cases lost their `case_id` before persistence.
- 64-bit SimHash overflowed SQLite's signed INTEGER.
- HF downloads duplicated bytes when a server ignored a `Range` request.
- Windows console crashed on unicode CLI output.
- `transformers`/`peft` were unimportable with `huggingface-hub 2.0`; pinned `<2`.
- Playwright suite detection used the wrong repository root.

### Changed

- `foundry.training` is now a package (`training/`) with `sft.py` and `recipes.py`.
- Version bumped to 2.0.0 across backend, CLI, frontend and lockfile.
- `.gitignore` root-anchors artifact directories so source packages cannot be hidden.

### Verified

- 121 backend tests, 7 frontend unit tests, 7 Playwright E2E tests.
- Real anonymous 1.5 GB Hub pull.
- Real LoRA run: loss 6.0436 → 5.0145, adapter saved.
- EXE health and UI 200 from a clean data directory.
- Upgrade candidate correctly rejected by the regression gate.

### Blocked (documented, not faked)

- `vercel ai-gateway setup`: the supplied credential is an AI Gateway key, not a platform
  token; gateway inference additionally requires a billing method.
- A good 3B fine-tune on this hardware: no CUDA/ROCm, projected 15–38 h for a short run.
- GGUF export of a trained adapter.

## [0.1.0] — 2026-11-01

First working local release.

### Added

- **Machine scan** with independent probes for OS, CPU, memory, GPU, Vulkan/CUDA/ROCm/DirectML/NPU, disks, toolchains and runtimes. UMA devices report **zero dedicated VRAM**.
- **Resource governor** with OS reserve, GPU headroom, safe budget, fit ranges and downshift suggestions.
- **GGUF library**: safe header parser, metadata, parameter count, KV bytes/token, hashing, fit estimates.
- **Inference**: supervised `llama-server`, runtime discovery, per-launch API key, health checks, streaming SSE, cancellation, metrics, orphan reaping.
- **Data Studio**: five adapters, consent gate, redacted preview, redaction, exact + near-duplicate removal, group-based splits, review, freeze with holdout hash, JSONL export.
- **Teaching enrichment**: evidence-grounded decision summaries, negative examples, synthetic provenance, validator that rejects invented quotes.
- **Identity Maker**: one prompt → editable card (facts vs proposals) → approval → synthetic dataset → deterministic identity evaluation.
- **Training Lab**: importability-aware environment detection, real smoke optimisation with parameter-delta assertion, preflight with blockers and presets.
- **Benchmarks**: instruction adherence, executed codegen, latency measurement, comparison, JSON/CSV/Markdown export.
- **Upgrade Lab**: failure taxonomy, prompt-patch candidates, re-evaluation, holdout guard, regression gate, human approval, rollback, lineage.
- **Agent workspace**: deterministic planner, allowlisted typed tools, approval queue, bounded retry, timeline.
- **Runs & Reports**: every job with events, output, errors and cancellation.
- **UI**: React 18 + TypeScript, dark/light, responsive (360 → 1920), keyboard accessible, reduced motion, complete German/English/French translations.
- **Security**: loopback binding, hashed pairing tokens, Origin/Host validation, CSRF header, JSON-only mutations, path allowlist.
- **Tests**: 98 backend tests, 7 frontend unit tests, 6 Playwright end-to-end tests.

### Verified on the test machine

- Windows 11 Pro, Ryzen 7 7735HS, Radeon 680M (unified memory), 19.8 GiB visible RAM.
- Vulkan inference through `llama-server` build 11064: 163 tok/s prompt, 46 tok/s generation on a 2B Q4_K_M model.
- CPU smoke optimisation: loss 2.1940 → 0.0303, parameters changed.
- Upgrade candidate correctly **rejected**: −33 % score with a critical secrecy regression.

### Known limitations

See [docs/architecture/KNOWN_LIMITATIONS.md](docs/architecture/KNOWN_LIMITATIONS.md). Highlights: no GPU training on this hardware, no TurboQuant, no official benchmark suites, no GGUF export, subprocess sandbox is not a VM.
