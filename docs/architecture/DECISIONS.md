# Architecture Decision Records — HCS Foundry

Each record states context, options, decision, trade-offs, how it is tested, and when to revisit.

---

## ADR-001 — Two-process architecture: browser SPA + local companion service

**Context.** A browser page cannot inspect host hardware, spawn `llama-server`, or read arbitrary local files. AGENTS.md §3.2 requires a local companion. The repository is empty, so there is no inherited architecture.

**Options.**
1. Electron/Tauri desktop shell with embedded Python.
2. Browser SPA + separate loopback HTTP service (chosen).
3. Pure browser app with WebGPU/WASM only.

**Decision.** Option 2. A Vite/React SPA talks to a FastAPI service bound to `127.0.0.1`. The service owns all privileged work. The SPA remains deployable as static files, which keeps the later Vercel step meaningful and keeps the local compute out of serverless functions.

**Trade-offs.** Two processes to start (documented one-command dev script plus a Windows launcher). Requires a pairing flow. Gains: clear security boundary, static deployability, no native-shell toolchain lock-in.

**Tested by.** `tests/test_security.py` (origin/host/token enforcement), `tests/test_api_smoke.py` (service boots and answers over loopback), Playwright E2E against the built SPA.

**Revisit if.** The product must ship as a single double-click binary for non-technical users — then wrap the same service in Tauri as a shell, keeping the API unchanged.

---

## ADR-002 — FastAPI + Pydantic v2 + SQLite (WAL) + filesystem artifacts

**Context.** Model ecosystem work is Python-native; the machine already has FastAPI/uvicorn/Pydantic/psutil working on Python 3.14. Heavy artifacts (GGUF, datasets, checkpoints) must not live in a database.

**Decision.** FastAPI for the API, Pydantic v2 for boundary validation, SQLite in WAL mode for metadata/jobs/lineage, plain files under a configurable data root for artifacts (content-addressed by SHA-256 where practical).

**Trade-offs.** SQLite single-writer limits concurrency; acceptable because heavy work happens in child processes that report back through the job table. No ORM: hand-written SQL keeps the schema auditable and avoids another dependency.

**Tested by.** `tests/test_db.py` (migrations idempotent, WAL enabled, foreign keys on), `tests/test_jobs.py`.

**Revisit if.** Multi-user or remote access becomes a requirement (out of scope: product is local-first and single-user).

---

## ADR-003 — llama.cpp / `llama-server` as the primary GGUF adapter

**Context.** GGUF is the dominant local format and `llama-server` binaries already exist on this machine; a Vulkan build was measured working at 46 tok/s generation on a 2B Q4 model.

**Options.** `llama-cpp-python` bindings, `llama-server` subprocess, Ollama daemon, LM Studio API.

**Decision.** Supervise `llama-server` as a child process and talk to its OpenAI-compatible HTTP API. Discover binaries by probing a configurable list of candidates and by verifying `--version`; never hard-code one path. Keep a CPU fallback candidate. Ollama/LM Studio are not required and not silently started.

**Trade-offs.** Process management complexity (ports, health, orphan cleanup, logs) versus crash isolation and zero Python-binding build pain. Chosen for isolation: a model crash cannot take down the API.

**Tested by.** `tests/test_runtime.py` (binary probe, GGUF header parsing, arg construction, port allocation), plus a real end-to-end smoke run recorded in `docs/evidence/`.

**Revisit if.** A backend exposes capabilities the HTTP API cannot (e.g. fine-grained KV-cache control) — then add a second adapter behind the same interface.

---

## ADR-004 — Pairing token + strict Origin/Host checks, loopback only

**Context.** A local privileged HTTP service is reachable by any web page the user visits unless it validates origin. DNS-rebinding and CSRF are the realistic threats.

**Decision.** Bind to `127.0.0.1`. Require a bearer token minted by an explicit `POST /api/v1/system/pair` call that the user triggers from the UI (or from the CLI on first start, printed once). Reject requests whose `Origin` is not in the allowlist (the dev origins plus explicitly paired deployment origins) and whose `Host` is not loopback. Mutating requests additionally require `X-Foundry-Client: 1` (a header a cross-site form cannot set without CORS preflight) and a JSON content type.

**Trade-offs.** Slightly more setup friction; the UI stores the token in `sessionStorage`, never in the bundle.

**Tested by.** `tests/test_security.py`: missing token → 401, wrong origin → 403, foreign Host → 403, `*` CORS never returned, token rotation invalidates old tokens.

**Revisit if.** A native shell replaces the browser (then pairing can be implicit via IPC).

---

## ADR-005 — Path allowlist and no generic shell endpoint

**Context.** The service reads model files and imported sessions. An LLM-driven agent also proposes actions.

**Decision.** Every filesystem operation resolves the path and checks it against configured roots (`model_roots`, `import_roots`, `data_root`). Agent shell actions run only allowlisted read-only commands (`git status`, `git diff`, `node --version`, …) inside the workspace, through an explicit approval record. There is no endpoint that accepts an arbitrary command string.

**Tested by.** `tests/test_security.py::test_path_traversal_rejected`, `test_agent_shell_allowlist`.

**Revisit if.** Benchmarks need real sandboxed execution — then a container-based executor replaces the allowlist for that specific job class.

---

## ADR-006 — Resource governor as a single authoritative budget

**Context.** 19.8 GB visible RAM with a 4.2 GB UMA carve-out and a browser open. Loading a 9B model at 8k context can push the machine into swap.

**Decision.** One governor module computes a safe budget from live measurements, reserves `max(2 GiB, 15 %)` for OS/desktop, caps GPU-side allocation at the reported shared memory minus 1 GiB, and enforces "one heavy job at a time" unless explicitly overridden. Model-fit estimation is a range with a confidence label, never a single number. Before a load, available memory is re-measured.

**Trade-offs.** Conservative defaults may refuse configurations that would technically work. That is preferred over an unstable desktop.

**Tested by.** `tests/test_governor.py` (UMA not counted as dedicated, reserve applied, concurrent heavy job refused, downshift proposals monotonic).

**Revisit if.** Real measurements show the reserve is too pessimistic on this class of APU.

---

## ADR-007 — Canonical, versioned event schema for all imported sessions

**Context.** Claude Code, Codex CLI, OpenCode and Antigravity differ in on-disk formats and change without notice.

**Decision.** Every adapter maps into `foundry.data.schema.CanonicalRecord` (`schema_version: "1.0"`), with provenance, task, trace, outcome, teaching, privacy and split fields exactly as specified in AGENTS.md §7.5. Adapters are versioned (`adapter_version`), tolerant of unknown fields, and covered by fixture tests. Unparseable lines are counted and reported, never silently dropped.

**Trade-offs.** Some provider-specific richness is flattened into `metadata`. Accepted to keep downstream stages (redaction, splits, training export) provider-agnostic.

**Tested by.** `tests/test_adapters.py` with fixtures for each provider including malformed/partial JSONL.

**Revisit if.** A provider publishes a stable schema with fields we cannot represent.

---

## ADR-008 — Evidence-based teaching annotations, never reconstructed chain-of-thought

**Context.** The user wants "why is this correct" connective text plus negative examples. AGENTS.md §7.6 forbids fabricating hidden reasoning.

**Decision.** Annotations are generated from observable evidence only (user request, visible output, tool calls, tool results, test output, diffs). Each synthetic row carries `synthetic: true`, teacher model, template version, source record hashes, and a validation status. A deterministic validator rejects annotations that cite evidence not present in the trace. No field is ever named `chain_of_thought` or presented as the provider's internal reasoning.

**Tested by.** `tests/test_enrich.py` (unsupported citations rejected, synthetic flags present, originals never mutated).

**Revisit if.** Providers officially expose reasoning traces and the user opts in — even then they are labelled as provider-supplied, not reconstructed.

---

## ADR-009 — Training: detect honestly, smoke-test for real, never fake a run

**Context.** `torch` is CPU-only here; `transformers`/`trl`/`peft` are absent. AGENTS.md §6 demands a real forward/backward/update before enabling long jobs.

**Decision.** The Training Lab has three truthful states: `unsupported` (no backend), `cpu-experimental` (torch CPU present), `gpu` (accelerated backend verified). A preflight runs a tiny real optimisation step on a synthetic tensor, asserts finite loss and a parameter delta, and only then unlocks job creation. If the environment cannot train, the UI says so with the precise missing piece and offers data preparation instead.

**Trade-offs.** On this machine the Training Lab will mostly show `cpu-experimental`, which is the truth.

**Tested by.** `tests/test_training.py::test_smoke_requires_parameter_update`, `test_preflight_reports_missing_backend`.

**Revisit if.** ROCm or a working DirectML training path is installed.

---

## ADR-010 — Benchmarks: deterministic oracles first, LLM judge never decides alone

**Context.** Benchmark scores are easy to inflate.

**Decision.** Suites are local and small but real: code-generation tasks executed against unit tests in a sandboxed subprocess, instruction-adherence checks with deterministic assertions, latency/throughput measurement, and a UI accessibility/route regression suite driven by Playwright. An optional LLM judge is recorded as a *secondary* signal and can never flip a failing deterministic check to pass. Every run stores model hash, runtime version, sampler settings, hardware profile, and suite version.

**Trade-offs.** Local suites are not leaderboard-comparable and are labelled `project-specific smoke`, never "official".

**Tested by.** `tests/test_benchmarks.py` (judge cannot promote, metadata completeness, failed case stays failed).

**Revisit if.** Official SWE-bench/Terminal-Bench adapters are added with proper isolation and licensing.

---

## ADR-011 — Upgrade loop with frozen holdout and human promotion gate

**Context.** Recursive self-improvement is the riskiest feature in the product.

**Decision.** Failures can become *candidates* only after review. A frozen holdout partition is hashed at dataset freeze time and the upgrade loop refuses to train on any record whose hash appears in it. Promotion requires: reproducibility metadata, passing tests, target-metric improvement above a configured threshold, no critical regression, and an explicit user approval event. Rollback restores the previous known-good profile and keeps the rejected candidate.

**Tested by.** `tests/test_upgrade.py` (holdout contamination refused, regression blocks promotion, rollback restores pointer).

**Revisit if.** Automated promotion is ever requested — it would still require a deterministic gate.

---

## ADR-012 — Frontend: React + TypeScript + Vite, hand-rolled design system, i18next

**Context.** Empty repository; the UI must be premium, dense, responsive, accessible and trilingual.

**Decision.** Vite + React 18 + TypeScript. No heavyweight component library: a small set of accessible primitives (Button, Field, Select, Tabs, Dialog, Toast, Table) styled with CSS variables and a token file, so dark/light themes and reduced motion are first-class. Charts are hand-written SVG (no chart dependency) so bundle size and accessibility stay controlled. i18next with complete `de`/`en`/`fr` resources and a key-parity test.

**Trade-offs.** More UI code to write than adopting a kit; total control over look, a11y and bundle size.

**Tested by.** Vitest for i18n key parity and formatting helpers; Playwright for route coverage, keyboard navigation, responsive breakpoints and reduced motion.

**Revisit if.** A11y audits demand primitives we cannot reasonably hand-roll (e.g. full combobox behaviour) — adopt a headless library then.
