# Known Limitations — HCS Foundry 0.1.0

Honest list as of 2026-11-01. Anything not listed as working should be assumed unimplemented.

## Hardware-derived

1. **No dedicated VRAM.** This machine has an integrated Radeon 680M with unified memory. The governor never reports the 4.29 GB driver value as VRAM. Large models will be slow and memory-tight.
2. **No CUDA/ROCm.** Training is CPU-only. The Training Lab therefore exposes at most a `cpu-experimental` tier and a tiny smoke optimisation; it will not claim a real fine-tune on this machine.
3. **Thermal/power data** is only available where the OS exposes it; on this machine GPU power is visible through the patched `nvidia-smi` only and is treated as informational, not authoritative.

## Runtime

4. **One inference server at a time.** The supervisor enforces a single loaded model; parallel models are not supported in 0.1.0.
5. **KV-cache quantization** is limited to what the discovered `llama-server` build exposes (`--cache-type-k/v`). If the flag is rejected, the UI falls back to FP16 and says so.
6. **TurboQuant** is a capability probe only. No discovered build exposes it, so it reports `not available on this build`.
7. **Vision** works only for models with a matching `mmproj` file; the UI refuses to claim vision support otherwise.

## Data

8. **Adapters cover observed formats.** Claude Code, Codex CLI, OpenCode and Antigravity on-disk schemas are provider-internal and may change; adapters are versioned and tolerant of unknown fields, but a future provider change can still produce "unknown record" counts.
9. **Near-duplicate detection** is SimHash-based with a fixed threshold; it is a heuristic, not a proof.
10. **PII/secret detection** is regex+entropy based. It reduces risk, it does not guarantee complete redaction. Review is mandatory before training export.
11. **Reasoning annotations** are evidence-grounded summaries, never the model's hidden reasoning. Rows that cannot be grounded are labelled `unknown`.

## Training

12. **No GGUF training.** GGUF remains an inference format; training-native checkpoints are the source of truth, and export to GGUF is not implemented in 0.1.0.
13. **Checkpoint resume** works for the local smoke/CPU trainer; interoperability with external trainers is not guaranteed.

## Benchmarks

14. **Local suites only.** Scores are labelled `project-specific smoke` and are **not** comparable to leaderboards.
15. **No official benchmark assets** (SWE-bench, Terminal-Bench, OSWorld, WebArena, GAIA, MMMU) are bundled or downloaded. The UI shows them as "not run" with a link to official instructions.
16. **Generated-code execution** is sandboxed by a subprocess with a timeout and a temp working directory, not by a VM or container. Do not treat it as a hardened sandbox.

## Agent

17. **Allowlisted tools only.** No arbitrary shell. The agent cannot install packages, push, or touch paths outside the workspace.
18. **No MCP auto-discovery** of third-party servers in 0.1.0; the settings screen records declared MCP servers but does not spawn them.

## Deployment

19. **Vercel hosts the frontend only.** Local inference/training never runs in serverless functions. A deployed frontend can only reach a local service through the pairing flow.
20. **Deployment credentials** are environment-specific; if absent, the release stops before deployment and says so.

## Language

21. **de/en/fr are complete for all shipped UI strings.** Additional locales are architecture-ready but not translated.

## v2 additions

22. **Hub downloads are plain HTTP streams.** They are resumable and hash-checked, but there is no parallel chunking or bandwidth scheduler.
23. **Gated models require a token with accepted terms.** The UI cannot accept the provider's terms for you.
24. **The executable excludes the ML stack.** Fine-tuning requires the source install; the EXE reports `requires_stack` instead of failing silently.
25. **A good 3B fine-tune is not achievable on this hardware.** No CUDA/ROCm; a 200-step 3B run at 512 tokens is projected at 15–38 hours on CPU.
26. **Adapter → GGUF export is not implemented.** Merging and quantizing remain manual.
27. **The cloud teacher is opt-in and currently blocked by account billing.** It only ever annotates; chat, training and benchmarks stay local.
28. **Vercel platform setup needs a platform token.** The supplied `vck_` key is an AI Gateway key and cannot create gateway keys or manage teams.
29. **The standalone build is Windows-only** in this release. The same PyInstaller spec can be adapted for Linux/macOS but was not tested.
