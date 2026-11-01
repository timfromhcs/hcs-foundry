# Implementation Plan — HCS Foundry

Derived from the audit (Gate A). Milestones follow AGENTS.md §16.1. Each milestone lists its acceptance evidence.

## M1 — Audited foundation

- Repository skeleton, `.gitignore`, `.env.example`, docs.
- `foundry` Python package: config, SQLite schema + migrations, job runner, SSE event bus, security middleware, structured logging.
- Machine detection providers (Windows/Linux/macOS) behind one interface, each failing independently.
- First-run scan API + Machine screen.

**Evidence:** `pytest tests/test_db.py tests/test_jobs.py tests/test_machine.py`, `GET /api/v1/health`, real scan JSON written to the (git-ignored) machine profile.

## M2 — Local model runtime

- GGUF header parser (metadata only, no full read).
- Model library scanner with hashing and compatibility estimation.
- `llama-server` supervisor: discovery, arg building, health check, streaming proxy, cancellation, orphan cleanup, logs.
- Resource governor wired into load.

**Evidence:** real end-to-end run on this machine (Qwen3-VL-2B Q4_K_M, Vulkan), recorded in `docs/evidence/`.

## M3 — Polished main UI

- Overview, Machine, Models, Playground, Runs, Settings.
- de/en/fr i18n, dark/light theme, responsive, keyboard accessible, reduced motion, job tray, toasts + activity panel.

**Evidence:** Playwright route/a11y/responsive suite, Vitest i18n parity test.

## M4 — Data Studio

- Generic JSON/JSONL/Markdown importer + Claude Code, Codex CLI, OpenCode adapters (versioned, fixture-tested).
- Consent/rights gate, inventory → preview → parse → normalize → redact → dedupe → segment → label → review → split → freeze.
- Secret/PII redaction, near-duplicate detection, conversation-level split, manifest hashing, JSONL export.

**Evidence:** `tests/test_adapters.py`, `tests/test_pipeline.py`, `tests/test_redact.py`.

## M5 — Teaching layer + Identity Maker

- Evidence-grounded decision summaries, negative examples, synthetic provenance, validation pass.
- One-prompt identity card (facts vs. generated proposals), dataset preview, approval gate, export to Training Lab.

**Evidence:** `tests/test_enrich.py`, `tests/test_identity.py`.

## M6 — Training Lab

- Environment preflight (torch/transformers/trl/peft, device, memory, disk).
- Real smoke optimisation step with parameter-delta assertion.
- Job lifecycle with checkpoints, live metrics, cancel, resume, export.

**Evidence:** `tests/test_training.py`; on this machine the expected state is `cpu-experimental` or `unsupported`, reported truthfully.

## M7 — Benchmarks

- Suite registry (codegen-exec, instruction-adherence, latency/throughput, UI regression).
- Sandboxed subprocess execution for generated code, deterministic oracles, optional secondary judge.
- Comparison matrix, failure taxonomy, JSON/CSV/Markdown export.

**Evidence:** `tests/test_benchmarks.py`, one recorded real run.

## M8 — Upgrade Lab

- Failure classification, candidate creation, frozen-holdout guard, regression gate, promotion, rollback, lineage.

**Evidence:** `tests/test_upgrade.py`.

## M9 — Agent workspace

- Plan → approval → execution with typed, allowlisted tools, timeline, artifacts, bounded repair loop.

**Evidence:** `tests/test_agent.py`.

## M10 — Release hardening

- Full local test suite, clean-install check, offline behaviour, restart recovery, secret scan, docs, release report.
- Git init + commits, GitHub push (credentials verified), Vercel last (reported honestly if unavailable).

**Evidence:** `docs/evidence/RELEASE_REPORT.md`.

## Out of scope for this release (documented, not faked)

- Real GPU training on this machine (no CUDA/ROCm).
- Official SWE-bench/Terminal-Bench/OSWorld/WebArena runs (isolation + licensing + assets not available here).
- TurboQuant KV-cache integration (no engine exposes it here; capability probe returns "not available").
- Cloud/hosted compute.
