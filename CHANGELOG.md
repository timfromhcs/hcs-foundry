# Changelog

All notable changes to HCS Foundry.

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
