# Architecture

This document describes what is actually implemented, not an aspiration.

## 1. Processes

| Process | Responsibility | Binds |
|---|---|---|
| Browser SPA | Rendering, i18n, charts, SSE consumption, streaming chat | — |
| Foundry service | API, security, jobs, governor, library, data pipeline, benchmarks, agent | `127.0.0.1:8770` |
| `llama-server` child | GGUF inference with a random per-launch API key | `127.0.0.1:<ephemeral>` |

The service supervises at most one `llama-server`. On shutdown it terminates the child; on startup it reaps an orphan recorded in `data/state/runtime.json`.

## 2. Backend modules

```text
server/foundry/
  config.py            paths, env overrides, UI origin allowlist
  db.py                SQLite schema + additive migrations (WAL, FK on)
  security.py          pairing store, origin/host/token middleware, path allowlist
  jobs.py              durable job runner (thread per job, events, cancellation)
  events.py            in-process pub/sub bridged to the asyncio loop for SSE
  state.py             AppState shared by routers
  library.py           GGUF discovery, metadata, hashing, fit estimates
  machine/
    detect.py          per-platform probes, each isolated and failure-tolerant
    governor.py        memory budget and fit estimation
  runtime/
    gguf.py            safe GGUF header parser (no tensor data)
    llama_server.py    process supervisor + streaming proxy
  data/
    schema.py          canonical record v1.0
    adapters.py        generic / claude_code / codex / opencode / antigravity
    redact.py          secrets + PII patterns, entropy flagging
    pipeline.py        inventory→preview→parse→redact→dedupe→split→freeze
    enrich.py          evidence-grounded teaching annotations
  identity.py          one-prompt identity card + synthetic dataset
  training.py          environment detection, real smoke step, preflight
  benchmarks.py        deterministic suites + sandboxed code execution
  upgrade.py           failure taxonomy, candidates, gates, rollback
  agent.py             planner, allowlisted tools, bounded retry
  api/                 system, models, data, lab routers
  main.py              app factory, middleware, static SPA
  cli.py               serve / scan / token / info
```

## 3. Request lifecycle

1. Browser sends `Authorization: Bearer <token>` and `X-Foundry-Client: 1`.
2. Middleware validates `Host` (loopback), `Origin` (allowlist) and token.
3. Mutating requests must be JSON; this blocks cross-site form posts.
4. Routers validate bodies with Pydantic and return typed errors.
5. Heavy work is submitted to the job runner and returns `{job_id}` immediately.
6. Job events are persisted in `job_events` and streamed over `/api/v1/events`.
7. The SPA polls `/runs/{id}` as a fallback and updates the job tray.

## 4. Persistence

SQLite in WAL mode. Large artifacts never enter the database.

Tables: `machine_profiles`, `models`, `runtime_profiles`, `datasets`, `dataset_records`, `jobs`, `job_events`, `benchmark_runs`, `benchmark_cases`, `training_runs`, `upgrade_candidates`, `lineage`, `agent_tasks`, `approvals`, `identity_profiles`, `settings`, `activity`.

Migrations are additive and idempotent; `schema_meta.version` tracks the applied level.

## 5. Resource governor

```text
os_reserve   = max(2 GiB, 15 % of total RAM)
usable_ram   = available_ram - os_reserve
gpu_cap      = reported shared memory (or 50 % of RAM when a unified device reports none)
usable_gpu   = gpu_cap - 1 GiB headroom
safe_budget  = unified ? min(usable_ram, usable_gpu) : usable_ram + gpu_cap
```

Fit estimation returns a range (`low … high`) plus a status:

- `comfortable` — high estimate ≤ 80 % of the safe budget
- `tight` — high estimate ≤ safe budget
- `risky` — low estimate ≤ safe budget
- `too_large` — low estimate exceeds the safe budget

The API refuses `too_large` loads unless the caller explicitly sets `force: true`.

## 6. Data pipeline

```text
inventory → preview (redacted) → parse → normalise → redact → dedupe
          → segment → label → split → review → freeze → export
```

- Splits are assigned by hashing the session/task group, so records from the same conversation never cross partitions.
- Dedupe uses exact canonical hashes plus a 64-bit SimHash with a configurable Hamming threshold.
- Freezing records a dataset hash and a holdout hash; without a test split it records `warning: no_test_split` and the upgrade lab refuses to use it.

## 7. Benchmark execution

- Instruction adherence: deterministic assertions on format, language, secrecy and arithmetic.
- Code generation: generated Python runs in a subprocess with a timeout and a temp working directory; assertions decide pass/fail.
- Latency: measured TTFT and tokens/sec, reported as a measurement (no pass/fail score).
- UI regression: Playwright from `web/`, not from the service.

The optional LLM judge is stored under `aggregate.judge` and never changes a case status.

## 8. Upgrade loop

```text
benchmark run → failure taxonomy → candidate (prompt patch) → re-run same suite
→ delta + regression check → holdout hash check → human approval → promote/rollback
```

Promotion is refused when the improvement is below the threshold, a critical case regresses, the holdout changed, or no approver is supplied.

## 9. Frontend

React 18 + TypeScript, hand-rolled design system, CSS variables for dark/light, i18next with complete `en`/`de`/`fr` resources (key parity enforced by a unit test). Charts are SVG. The only runtime state store is `AppContext` (connection, profile, jobs, toasts, theme, language).

## 10. Failure behaviour

- A probe that fails degrades one machine section to `unknown`; the scan still completes.
- A job that fails records the traceback tail and stays visible in Runs.
- A model load that fails captures the child's log tail.
- A restart marks running jobs `interrupted` instead of pretending they finished.
