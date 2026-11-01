# Changelog

All notable changes to HCS Foundry.

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
