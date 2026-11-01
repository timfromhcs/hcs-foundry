# Changelog

All notable changes to HCS Foundry.

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
