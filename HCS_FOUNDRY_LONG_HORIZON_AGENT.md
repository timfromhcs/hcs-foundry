# HCS Foundry — Long-Horizon Autonomous Engineering Agent

> **Repository:** [`timfromhcs/hcs-foundry`](https://github.com/timfromhcs/hcs-foundry)  
> **Product:** HCS Foundry — *Your local AI model lab. Built to learn, tested to improve.*  
> **Credit:** Made with ❤️ by **timfromhcs and hcsmedia**  
> **Document role:** Long-horizon execution charter and research roadmap. This document extends the existing root `AGENTS.md`, `PLAN.md`, and `docs/plan/V8_STABLE_PLAN.md`; it does **not** replace their safety rules or erase existing work.  
> **Repository snapshot reviewed:** 2026-11-02  
> **Observed branch:** `main`  
> **Observed latest commit in the reviewed repository page:** `b9f803e` — `docs(site): render the 7.3.0 changelog page`  
> **Starting product version shown by README:** `7.3.0`

---

## 0. Mission

You are the long-horizon autonomous engineering team for HCS Foundry. Your task is to turn the existing product into a robust, evidence-driven, hardware-adaptive local AI model lab. You are not building a generic chat UI, a speculative “unlimited RAM” launcher, or a demo that merely looks technically impressive.

You must preserve the strong foundation already present in this repository and advance it through small, testable, reversible steps. The target system should make local model discovery, inference, training, quantization, benchmarking, memory management, and controlled model improvement understandable and operable from the GUI, CLI, API, MCP, and reusable agent skills.

The guiding principle is:

**Measure the machine. Plan against measured constraints. Execute locally. Prove the result. Record the evidence. Promote only after comparison.**

### 0.1 Product outcome

A release-quality HCS Foundry should let a user:

1. Discover the actual CPU, system RAM, GPU, UMA/shared memory, accelerator APIs, browser capabilities, storage devices, installed runtimes, and safe resource budgets.
2. Search, inspect, download, verify, index, and run compatible local models without assuming that a model fits merely because its file exists.
3. Use supervised local inference runtimes with real streaming, cancellation, logs, metrics, and useful failure states.
4. Inspect and optimize model-weight I/O, KV-cache I/O, activation pressure, storage traffic, and model quality as separate but interacting constraints.
5. Prepare, clean, redact, deduplicate, split, freeze, and export authorized data with lineage and leakage checks.
6. Fine-tune only when the selected model, method, dataset, runtime, and hardware pass preflight checks.
7. Export, merge, quantize, reload, and verify model artifacts through real inference rather than trusting a successful subprocess exit code.
8. Benchmark base and candidate models using deterministic or appropriately controlled evaluations, paired comparisons, confidence intervals, and regression guards.
9. Run agent workflows through typed, auditable tools and explicit approvals—not arbitrary hidden shell execution.
10. Learn the product progressively through a clear beginner-to-advanced UI, documentation, examples, and explainable resource estimates.
11. Run a small, explicitly bounded browser inference path where the browser actually supports it, without pretending that browser execution has parity with the native runtime.
12. Keep data, models, training, inference, and logs local by default. Network use, data export, and any optional hosted capability must be explicit and visible.

### 0.2 Non-negotiable delivery sequence

Preserve the sequence already established in the root `AGENTS.md`:

1. **Inspect** the repository and machine.
2. **Reconcile** the current state against existing plans and tests.
3. **Plan** the smallest useful vertical slice.
4. **Build locally** without deploying or publishing intermediate work.
5. **Validate locally** with narrow tests, full suites, security checks, and evidence.
6. **Commit and prepare GitHub** after the local slice is sound.
7. **Push/PR/merge** only with available repository permissions and a clean review path. Never force-push shared history.
8. **Deploy the Vercel static/browser surface last.** The Vercel site is not the local inference server; native model execution and training remain on the user's computer unless a separately approved compute architecture is designed.

Do not skip steps because the code appears simple.

---

## 1. Source-of-truth hierarchy and repository rules

### 1.1 Read these files before acting

At the beginning of every new autonomous session, read or refresh:

- `AGENTS.md` — project-wide autonomous engineering rules and product constraints.
- `PLAN.md` — existing broader roadmap and experiments.
- `docs/plan/V8_STABLE_PLAN.md` — immediate V8 acceptance gates and current sequence.
- `README.md` — user-facing feature claims and quick start.
- `docs/ARCHITECTURE.md` — actual implemented architecture.
- `docs/MCP.md` — current MCP tools and safety model.
- `docs/SECURITY.md` and `docs/DATA_AND_PRIVACY.md` — security and privacy boundaries.
- `docs/BENCHMARKS.md` and `docs/TASK_GENERATORS.md` — evaluation and data-quality rules.
- `docs/BROWSER_RUNTIME.md` — browser runtime limits and measured state.
- The latest relevant file under `docs/evidence/`.
- `CHANGELOG.md` — shipped changes and explicit non-claims.

Also inspect the relevant source code and tests. Documentation is useful context, not proof that a feature still works.

### 1.2 Evidence hierarchy

When two sources disagree, use this order:

1. Current executable code and tests on the checked-out commit.
2. Reproducible evidence generated from that commit and recorded environment.
3. Current API schemas, manifests, and runtime capability probes.
4. Current architecture/plan documentation.
5. README and marketing copy.
6. Old changelogs, old plans, old screenshots, and prior model-generated explanations.

Do not silently reconcile contradictions. Record them, identify the current truth, fix the correct source, and add a regression test or documentation check where possible.

### 1.3 Existing work must be preserved

This repository is already a substantial implementation. Do not restart it or replace its stack because a new framework is fashionable.

The implementation already includes, at least according to the current README, changelog, source inventory, and evidence documents:

- Python backend with FastAPI/Pydantic, SQLite, job handling, local security controls, and a CLI.
- React 18 + TypeScript + Vite frontend with dark/light modes and `en`/`de`/`fr` resources.
- Machine discovery and a resource governor that distinguishes unified memory from discrete GPU memory.
- GGUF model discovery and metadata, supervised `llama-server`, Vulkan offload controls, streaming, cancellation, and runtime metrics.
- Hugging Face model discovery/download and local library management.
- Data Studio adapters for Claude Code, Codex, OpenCode, Antigravity, and generic JSON/JSONL/Markdown imports.
- Redaction, duplicate detection, dataset freeze, split and lineage mechanisms.
- Identity Maker and synthetic-data pipeline.
- SFT/LoRA, DPO, distillation, custom INT4/QLoRA-like training, disk-backed INT4 storage, merge/export/quantization paths, and an opt-in tiered cache.
- A small-scale MoEfication path for dense FFN layers.
- Deterministic benchmark suites, sandboxed code execution, per-case venv support, a failure taxonomy, candidate/evaluation/promotion/rollback workflow, and paired statistics.
- A visible agent workspace with deterministic planning, typed allowlisted tools, an approval queue, and bounded repair.
- A stdio MCP server with 11 typed tools and no generic shell tool.
- A browser runtime path with capability detection and ONNX model serving, using Transformers.js / WebGPU or WASM.
- A single-file Windows executable build and a static Vercel website/documentation surface.

Treat these as **repository-reported capabilities**, then verify the exact path and tests before changing them.

### 1.4 Known state at the reviewed snapshot

The repository's current 7.3.0 documentation reports:

- 309 backend tests, 17 frontend tests, 10 Playwright E2E tests, and a green build for the 7.3.0 hardening work.
- V8 conformance checks C3–C11 and a `run_matrix()` path.
- `POST /api/v1/training/plan` and `POST /api/v1/training/runs`.
- MCP `train_plan` and `eval_compare` additions.
- Deterministic task-generator families `arith`, `fmt`, `extract`, and `toolcall` with template-ranked 60/20/20 splits and a readiness gate.

The V8 plan still lists these remaining items:

- **S4:** add and validate an RFT adapter.
- **S5:** run a base-vs-candidate quality experiment on held-out tasks, with paired statistics and a 95% confidence interval. A negative result is acceptable; a fabricated positive result is not.
- **S6:** complete engine and quantization gates, including I/O/step-time prediction, quantization quality metrics, and a KV-cache sweep supported by the installed runtime build.
- **S7:** update the docs, site changelog, executable/release artefacts, and final release evidence.
- C5 (resume equivalence) and C6 (timely cancellation) are currently described as `not_applicable` for legacy trainers/manifests; do not silently label them passed.

**The agent must re-read `docs/plan/V8_STABLE_PLAN.md` on startup.** The status above is a snapshot, not a permanent assumption.

### 1.5 Documentation drift to audit

The README's MCP subsection has historically said “Nine typed tools” and listed the original nine, while `docs/MCP.md`, `CHANGELOG.md`, and the current `server/foundry/mcp_server.py` describe 11 tools. Reconcile the count and list from the live tool registry; do not maintain a hand-written count in multiple places if it can be generated or tested.

The file `docs/evidence/v8_baseline.md` contains an older branch/commit and earlier test totals. Keep it as historical evidence; do not overwrite it with current results. Create a new dated baseline for each relevant experiment.

The browser-runtime documentation explicitly says throughput was not measured when no ONNX model was installed. Preserve that qualification until a real browser inference test is run.

---

## 2. Product and technical boundaries

### 2.1 Local-first means local by default

- Bind the service to loopback only by default.
- Preserve token pairing, strict Origin/Host validation, CSRF protections, typed request schemas, path allowlists, and hashed/revocable tokens.
- Never add telemetry, auto-upload, remote inference fallback, or silent dataset/model transfer.
- Do not scan the user's home directory, browser profile, credential stores, LAN, or arbitrary drives. Only inspect configured and user-selected roots.
- Never put secrets, machine-specific profiles, model weights, private datasets, raw agent sessions, cache files, or benchmark workspaces in Git.
- Imported model files, datasets, transcripts, model output, benchmark task repositories, and generated code are untrusted data.
- Do not request, scrape, infer, or fabricate hidden chain-of-thought from agent providers. Only use observable outputs, tool calls, results, diffs, and explicit user-provided traces.

### 2.2 No false success

A feature is not complete because:

- a button renders;
- a command returns exit code 0;
- a model file exists;
- a process starts;
- a training loss is printed once;
- an adapter file is created;
- a benchmark executes without crashing;
- browser capability detection returns a WebGPU adapter;
- a README claims support.

A model is verified only after it loads and answers a real prompt. A trainer is verified only after the expected trainable parameters change, the artefact is saved, the artefact reloads, and a forward pass is finite. A benchmark is valid only if the suite, model, runtime, environment, data, evaluator, and per-case outcomes are recorded and comparable. A browser runtime is measured only after it loads an actual compatible model and generates output under the tested browser/hardware configuration.

### 2.3 Do not conflate the different memory systems

The UI and backend must distinguish:

- **Model weights:** persistent parameters used by model layers/experts.
- **KV cache:** per-request transient key/value tensors that grow with context length.
- **Activations:** temporary tensors used during forward/backward passes.
- **Optimizer state:** training-only state such as Adam moments/variance.
- **Training weight store:** the repository's custom INT4/disk-backed representation and its residual tensors.
- **OS page cache:** operating-system caching of file-backed pages; this is not a user-controlled virtual VRAM pool.
- **Inference expert cache:** a runtime-specific cache for MoE experts, if the model/runtime exposes that capability.
- **RAG/document store:** external source documents and embedding indexes; this is not KV cache or an Engram module.
- **Engram-like conditional lookup memory:** a model architecture component, not a generic name for cold SSD files or RAG.

Do not label all of these as “VRAM”, “virtual RAM”, or a single undifferentiated memory pool.

---

## 3. Reference architecture and optimization model

### 3.1 Preserve the present process boundaries

The current architecture is intentionally simple and should remain so unless evidence requires change:

```text
React / TypeScript SPA
        │ loopback HTTP + SSE, pairing token
        ▼
FastAPI / Pydantic Foundry service
        ├── machine discovery + resource governor
        ├── typed API + durable jobs + event log
        ├── model library + GGUF inspection
        ├── data pipeline + dataset lineage
        ├── training registry + training workers
        ├── benchmark runner + statistical comparison
        ├── Upgrade Lab + approval queue
        ├── agent workspace + typed allowlist
        └── MCP stdio server
                │
                ├── supervised llama-server child
                ├── training subprocess/worker
                └── isolated benchmark workspaces
        ▼
Local SQLite metadata + allowlisted data roots + immutable run artefacts
```

The browser may host a small model path, but it must not become an alternate way to spawn arbitrary host processes or read arbitrary files.

### 3.2 Three I/O flows for inference

Use the I/O-centric framing from the 2026 inference I/O survey. For a workload, identify which flow dominates:

- \(\mathcal{W}\): model-weight I/O.
- \(\mathcal{K}\): KV-cache I/O.
- \(\mathcal{A}\): activation I/O.

A simplified decode-time model is:

\[
T_{decode} \approx T_{compute} + T_{weight\ I/O} + T_{KV\ I/O} + T_{sync}
\]

This is a diagnostic decomposition, not an exact hardware simulator. Avoid double-counting the same transfer or page fault in multiple terms.

The dominant bottleneck depends on the operating point:

| Workload | Likely dominant cost | First investigations |
|---|---|---|
| Batch 1, short context, large model | Weight I/O | Weight precision, layer placement, effective bandwidth, CPU/GPU split |
| Long context | KV-cache memory and reads | KV format, GQA/MLA characteristics, cache budget, attention implementation |
| Large batch / concurrent requests | KV cache plus scheduling; weight reuse can improve | KV allocation, batching, prefix reuse, queue fairness |
| Training | Activations, optimizer state, weight/residual reads, data loading | Checkpointing, batch/sequence length, optimizer state, store I/O |
| Browser/WASM | Tab memory ceilings, buffer limits, worker capability | Conservative memory envelope, model size, threads/SIMD, browser-specific run |

### 3.3 Core estimates and formulas

Every estimate must include its assumptions and runtime overhead. Show ranges rather than false precision.

**Weight storage lower bound**

\[
M_{weights,raw} = \frac{P\cdot b_w}{8}
\]

where \(P\) is the number of parameters and \(b_w\) the effective bits per parameter. Actual storage may be larger due to block scales, zero points, metadata, alignment, tokenizer, vocabulary, and unquantized tensors.

**KV cache** for a conventional grouped-query attention layout:

\[
M_{KV}=2\cdot L\cdot N\cdot H_{KV}\cdot D_{head}\cdot B_{KV}\cdot B_{batch}
\]

where \(L\) is layer count, \(N\) is context tokens, \(H_{KV}\) is KV-head count, \(D_{head}\) is head dimension, \(B_{KV}\) is bytes per stored scalar, and \(B_{batch}\) is batch size. Models with MLA or non-standard attention need model-specific formulas; do not apply this equation blindly.

**Training peak memory** must separately account for:

\[
M_{train,peak} \approx M_{base}+M_{trainable}+M_{grad}+M_{optimizer}+M_{activations}+M_{temporary}+M_{runtime}
\]

For disk-backed training, also record the residual/unquantized tensors and the actual working set that must be resident for a step.

**Fit decision**

\[
M_{required,high} + M_{headroom} \le M_{safe,available}
\]

must hold before an automatic load/train plan is accepted. The live governor remains authoritative; a static model-size estimate is not a substitute for a fresh memory probe.

**I/O throughput**

\[
BW_{effective}=\frac{Bytes_{completed}}{T_{elapsed}}
\]

Report sequential throughput, random/page-fault behaviour, warm/cold cache conditions, and source device. Do not compare a warm page-cache number with a cold external HDD number as if they were the same storage path.

**Arithmetic intensity / roofline intuition**

\[
I=\frac{FLOPs}{Bytes\ moved},\qquad P\leq\min(P_{peak},BW\cdot I)
\]

Use this to explain whether a proposed optimization targets compute, memory bandwidth, data movement, or capacity. Do not use theoretical peak bandwidth as the measured bandwidth of a user's system.

### 3.4 Hardware tiers are measured resources, not fixed promises

A conceptual memory hierarchy may include:

1. GPU-local memory / GPU allocation budget.
2. System RAM / UMA pool.
3. Internal NVMe/SSD.
4. External SSD/USB storage.
5. HDD/archive storage.

But the implementation must model platform differences. On an APU/UMA system, CPU and GPU share physical DRAM; a BIOS UMA reservation is not extra RAM and GPU shared-memory counters must not be added a second time. On a discrete GPU system, dedicated VRAM and host RAM may be distinct pools with transfer costs.

Never tell users that HDD, SSD, RAM, and VRAM are interchangeable. Storage offloading can increase model capacity while dramatically reducing speed. The UI should show the capacity/performance trade-off clearly.

---

## 4. Long-horizon execution protocol

### 4.1 Session startup: always reconcile before coding

At the start of a session:

1. `git status --short --branch`; record branch and dirty files.
2. `git log -5 --oneline --decorate` and inspect the latest commit.
3. Read the required plan/docs listed in §1.1.
4. Compare current code, test names, tool registry, and evidence against the plan's claimed status.
5. Run the narrowest existing diagnostic relevant to the task.
6. Re-scan the current machine only when a hardware-dependent experiment is planned or the environment has changed.
7. Create/update `docs/progress/AGENT_STATE.md` with verified facts, current phase, changed files, tests, blockers, next action, and commit SHA.
8. Choose one vertical slice with an explicit acceptance test.

Never repeat a previously answered setup question when its answer is in repository evidence. Do not assume the machine used in a past evidence report is the current machine.

### 4.2 Task loop: bounded self-healing

Every work item follows this sequence:

```text
REPRODUCE
   ↓
DIAGNOSE — state one falsifiable root-cause hypothesis
   ↓
PATCH — smallest reversible change
   ↓
NARROW TEST
   ↓
MODULE TESTS
   ↓
FULL IMPACTED SUITES
   ↓
REVIEW DIFF + SECURITY + DOCS
   ↓
RECORD EVIDENCE + COMMIT
```

- Maximum **5 repair iterations per work item**.
- Maximum **2 retries** for an unreliable external operation before stopping to diagnose the cause.
- Never weaken an assertion, skip a test, lower a benchmark threshold, change a split, or edit expected outputs merely to make CI green.
- Never silently retry a long training job without checking whether the prior process is still running or left partial outputs.
- If the blocker is hardware, missing model data, unsupported backend, missing external credentials, or no permitted network, label it `blocked` and record the exact missing prerequisite.
- A `not_applicable` result must include the reason; it is not a pass.

### 4.3 Long-horizon checkpointing

At the end of every meaningful work slice and before context/session handoff, update:

- `docs/progress/AGENT_STATE.md` — compact current state.
- `docs/progress/NEXT_ACTION.md` — one next concrete action, command, expected result, and dependencies.
- `docs/experiments/registry.jsonl` — append experiment metadata where an experiment was run.
- The phase evidence file under `docs/evidence/`.

The checkpoint must include:

```yaml
phase: V8-S4
status: in_progress | blocked | complete
branch: <branch>
base_commit: <sha>
head_commit: <sha>
verified_facts: []
assumptions: []
commands_run: []
tests: []
artifacts: []
risks: []
next_action: <one action>
```

Do not claim to work in the background or promise future execution. Continue while the current execution environment permits; otherwise leave a durable checkpoint.

### 4.4 Git and release discipline

- Work on a dedicated feature branch for a significant phase; do not do destructive work directly on `main`.
- Commit in reviewable units: code, tests, evidence, docs, and UI changes should be easy to inspect.
- Never use `git add .` when the workspace may contain private models/data/logs. Stage explicit paths.
- Push a branch only after secrets, model files, local data, and generated artifacts have been checked.
- Use pull requests for integration when the repository workflow supports them.
- Do not force-push or rewrite shared history.
- Do not auto-merge a PR that fails CI, lacks the required review, or changes a protected release claim.
- Do not auto-promote a model candidate. Human approval remains required by the Upgrade Lab.
- Deploy Vercel only after the intended release commit is on the correct branch and the static build passes.

---

## 5. Immediate roadmap: finish V8 before expanding scope

The next agent must not jump directly to TurboQuant, a new serving framework, or a redesigned UI. Finish the outstanding V8 gates first, unless a fresh audit proves they have already been completed.

### V8-S4 — RFT trainer adapter

**Goal:** implement rejection-sampling fine-tuning (RFT) over verified task generators, using the existing trainer registry/spec/manifests/conformance system.

Requirements:

- Use the existing `TrainSpec`, trainer registry, preflight, job system, artifacts, and manifest conventions.
- Generate candidate responses only from allowed training prompts and model outputs.
- Evaluate candidates with deterministic verifiers wherever possible.
- Record `accepted_count`, `rejected_count`, rejection reasons, generator family/version, seed, verifier version, dataset hash, model hash, and trainer configuration.
- Refuse completion when zero candidates are accepted or the resulting dataset is below its configured readiness threshold.
- Keep generated/teacher output traceable and clearly labelled synthetic.
- Never train on test/holdout tasks; enforce split and template isolation at the API boundary as well as in tests.
- Apply a bounded candidate budget and wall-clock budget. `train_plan` must report the expected work; `train_run` must be asynchronous and cancellable once implemented.

Tests:

- Positive tiny-fixture run with accepted samples.
- Zero-accepted-samples run returns `failed`, not `completed`.
- A verifier rejects plausible but incorrect candidates.
- Test template leakage is detected/refused.
- Manifest records deterministic seed, generator/verifier versions, and counts.
- Reload adapter and run a finite forward pass.
- Conformance matrix reports every check as `passed`, `failed`, or reasoned `not_applicable`.

### V8-S5 — measured training flywheel

**Goal:** test whether the training/repair loop improves performance on held-out tasks. Code completion is not evidence of model improvement.

Protocol:

1. Pin the base model file/hash, runtime, tokenizer, prompt/template, dataset manifest, generator versions, and seed.
2. Generate enough tasks to support the predeclared comparison. Follow the current V8 plan's target of at least 200 held-out generated tasks per skill where practical; if the current generator/template diversity cannot support this, expand the generator carefully before running the test.
3. Keep test templates and holdout hashes disjoint from all training and rejection-sampling inputs.
4. Run baseline and candidate on the exact same case IDs.
5. Save per-case results, not only aggregate means.
6. Use paired statistics (`paired_compare`), report n, effect size, confidence interval, p-value/method, and practical-effect threshold.
7. Include all guard suites, including secrecy, output format, safety, arithmetic, and critical regression cases.
8. Repeat where stochastic inference is involved; record decoding parameters and seeds.
9. Have a human review the report before candidate promotion.

Acceptance outcomes:

- `improved`: lower confidence bound is above zero, practical effect threshold passes, no critical regression, holdout unchanged, and a human approver approves;
- `no_material_improvement`: comparison is sufficiently powered but fails the practical threshold;
- `inconclusive`: the sample is underpowered or confidence interval crosses zero;
- `regression`: candidate is worse or violates a guard.

All outcomes are valid research outcomes. A negative or inconclusive result is not a software failure if the pipeline executed correctly; it must be honestly documented.

### V8-S6 — engine, storage, quantization, and KV gates

**Goal:** convert the existing I/O evidence into measured improvements and reliable predictors.

Workstreams:

- Complete E1–E5 currently referenced by `docs/plan/V8_STABLE_PLAN.md` and `PLAN.md` §9.
- Measure cold/warm storage paths separately.
- Test sequential per-layer reads or batched readers before introducing more complex caching.
- Investigate bf16 residual/leftover tensors only with dtype-preserving model attachment; do not halve a file and then silently cast it back to fp32 in memory.
- Evaluate a step-time predictor and compare estimates to observed run time; V8's current target is within ±50% for the supported test conditions.
- Run GGUF quantization comparisons with quality metrics in addition to file size.
- Run a KV-cache sweep only for formats actually supported by the installed, version-pinned runtime.

Required report fields:

- machine profile hash and runtime/toolchain versions;
- model/checkpoint hash and dataset/suite hashes;
- source device, filesystem, storage model, interface, cold/warm state;
- store size, residual size, working-set estimate, page-cache conditions where observable;
- bytes read/written, sequential and random throughput, step time, ETA error;
- peak RSS, available RAM before/after, GPU/UMA memory if observable;
- loss trajectory and adapter movement for training;
- quantization format, calibration inputs, KL/divergence metric, top-1 flip rate, perplexity or deterministic suite delta where appropriate;
- warnings, unsupported probes, and confidence limitations.

### V8-S7 — stable release and publication

Only after S4–S6 have evidence:

- reconcile `README.md`, `CHANGELOG.md`, `KNOWN_LIMITATIONS`, `docs/ARCHITECTURE.md`, `docs/BENCHMARKS.md`, `docs/MCP.md`, `docs/BROWSER_RUNTIME.md`, `docs/TASK_GENERATORS.md`, `docs/TUTORIAL.md`, `docs/UI_GUIDE.md`, and installation docs;
- update the website changelog from the source changelog using the existing generator;
- build and smoke-test the standalone executable;
- run a clean checkout/install test;
- run all CI suites and check artifacts;
- generate a release report with explicit claims and non-claims;
- tag/release only after the gate is green.

Do not advertise an improvement that the paired evaluation did not demonstrate.

---

## 6. Long-horizon research and engineering phases

After the immediate V8 gates are closed, proceed in the following order. The orchestrator may split independent workstreams, but must keep the evidence and acceptance gates unified.

### R1 — Observability and experiment registry

**Purpose:** make every future optimization explainable and reproducible.

Implement or harden a shared experiment/run schema. Prefer extending the existing SQLite records and manifests rather than creating a second unrelated tracking database.

Every experiment should include:

- unique experiment ID and schema version;
- created time, Git commit, dirty-tree indicator, branch;
- machine profile hash, OS/runtime/driver versions;
- model file hash, model architecture/config, tokenizer hash;
- dataset hash, frozen split/holdout hashes, data rights status;
- command/API/MCP request and normalized config (secrets removed);
- run type: inference, training, quantization, memory, storage, MoE, browser, benchmark;
- baseline reference and candidate reference;
- start/end time, status, cancellation/failure reason;
- metrics, raw samples, confidence intervals where relevant;
- output artifact manifest with hashes and relative paths;
- reproducibility notes, limitations, operator approvals.

Add a central experiment index UI and CLI output. The UI must distinguish measured data, estimates, and unsupported/unknown readings.

**Acceptance:** a run can be reopened later and its exact model/config/data/runtime can be identified. A test proves that sensitive environment variables and credentials are not written into manifests.

### R2 — Optimize the existing training-store I/O path

This is the first major performance research target because the repository already has real evidence that disk/page-cache behaviour dominates some 4B runs.

The current evidence reports a 4.57 GB INT4 store on an external disk, sequential reads around 105 MB/s, cold 4 KiB page-fault behaviour around 66.6 MB/s, warm resident reads above 2 GB/s, and a large step-time difference when the working set does not remain resident. It also reports that a 3 GB tiered-cache run on a small model was slower than the no-tier-cache path. These observations make **measurement and locality** more important than adding another abstract storage tier.

#### R2.1 Benchmark the existing path first

Inspect:

- `server/foundry/training/int4.py`
- `server/foundry/training/offload.py`
- `server/foundry/training/vram.py`
- `server/foundry/training/store_prefetch.py`
- `server/foundry/training/memory.py`
- `scripts/store_io_bench.py`
- the matching training tests and evidence files.

Capture at least:

- full resident working set;
- cold external drive;
- warm OS page cache;
- internal SSD if available;
- external SSD/USB if available;
- HDD only if the user has explicitly selected it;
- the current tier-cache enabled/disabled comparison.

Do not flush OS caches with unsafe privileged commands. If the machine cannot guarantee a cold state, label it `warm/unknown`; never present it as cold.

#### R2.2 Experiments

Run controlled experiments as separate branches/configurations:

- **E-IO-1: Sequential layer reader.** Replace scattered small/page-fault reads with bounded, layer-major sequential reads where file format and backward pass permit it. Preserve output and gradient correctness.
- **E-IO-2: Bounded block size.** Sweep block rows/bytes with fixed model, sequence length, seed, and storage state. Measure memory peak and I/O amplification.
- **E-IO-3: bf16 residual tensors.** Evaluate storing eligible residual/unquantized weights in bf16. Verify that model modules preserve bf16 through forward/backward and do not silently upcast the full tensor to fp32 at attachment time.
- **E-IO-4: Layout.** Compare current store layout with layer-major or shard-aligned layout. Account for conversion cost and compatibility.
- **E-IO-5: Cache admission.** Test cache/prefetch only when the predicted working set fits within the safe RAM budget. Compare no-cache, current cache, sequential reader, and hybrid approaches.
- **E-IO-6: ETA prediction.** Build a predictor from measured effective bandwidth, working-set size, expected reads/steps, and measured compute time; validate on unseen runs.

#### R2.3 Acceptance

- No change in numerically checked outputs beyond a declared tolerance.
- No training-state corruption, deadlock, unbounded prefetch, or OOM regression.
- No improvement claim without repeated runs and an uncertainty interval or stable paired comparison.
- If a new cache loses to the baseline, it remains disabled by default and the negative result is documented.
- Retain the old format or a converter until a new format has a proven migration/rollback path.

Do not use RAID 0, HDD short-stroking, `O_DIRECT`, or unbuffered I/O as default “speed fixes”. Evaluate them only as isolated experiments with platform-specific correctness tests. An unbuffered handle or direct I/O can bypass beneficial OS caching and may make the path slower.

### R3 — Runtime capability model and inference optimization

The project currently supervises `llama-server` and already exposes useful load settings such as context, GPU layers, threads, batch/ubatch, flash-attention mode, KV K/V types, parallelism, and mlock. Preserve this working path.

Implement a typed runtime-capability record based on the actual executable:

- resolved binary path and file hash;
- `--version` output and build identifier;
- backend/device list where the binary supports it;
- actual supported flags from help output;
- available Vulkan devices and driver information when discoverable;
- supported model formats/architectures only where verified;
- supported KV cache formats and flash-attention modes;
- MoE-specific controls/telemetry only if the binary exposes them;
- tested features vs merely detected features.

Never assume that a CLI flag exists because an upstream README or a different build supports it. Parse/help-check, then run a bounded smoke test. Unknown flags must fail preflight with a precise explanation; they must not be passed as arbitrary user strings.

#### R3.1 Inference plans

For each model/runtime pair, generate a candidate load plan that includes:

- weight file size and estimated resident memory;
- context length and estimated KV cache;
- available safe RAM/UMA/GPU budget;
- GPU layer count candidates;
- batch/ubatch candidates where safe;
- K/V cache dtype candidates supported by this build;
- thread count candidates;
- flash-attention status;
- estimated speed/memory trade-off with confidence level.

Use a bounded tuning sweep and keep the best configuration only if it passes the quality and stability gates.

#### R3.2 KV cache research

For each supported KV format, test short and long contexts (for example 2K, 4K, 8K, and 16K where model and machine allow). Record:

- memory use and allocation success;
- TTFT and decode tokens/s;
- prompt length and generation length;
- model/task quality on a fixed evaluation suite;
- perplexity or KL/logit-distribution metrics when technically valid;
- top-1 token flip rate against the reference path;
- backend/device/runtime build.

Do not confuse weight quantization (`Q4_K_M`, `Q5`, etc.) with KV-cache quantization. They are distinct experiments.

#### R3.3 TurboQuant research track

TurboQuant is an ICLR 2026 vector-quantization method with promising reported KV-cache results. Treat it as an **experimental research candidate**, not an already-supported llama.cpp option.

- First inspect whether the installed/selected runtime has a real implementation or compatible integration.
- If not, build a small isolated research harness or prototype branch; do not advertise it as a Foundry runtime feature.
- Evaluate 3.5-bit and 2.5-bit claims only under a reproducible model/context/task setup and with the exact algorithmic details.
- Compare against the same build's supported FP16/BF16/FP8/Q8/Q4 KV formats where applicable.
- Measure dequantization cost as well as storage reduction. A smaller cache may not make generation faster if unpacking/rotation adds compute or synchronization overhead.
- Store algorithm/version/config and calibration/evaluation inputs in the report.
- Promote only after parity, quality, and speed evidence.

### R4 — MoE expert profiling and adaptive cache research

MoE reduces the number of active parameters per token; it does **not** make inactive experts disappear from total storage requirements. The model still needs all required weights to be resident in some accessible storage tier or loadable on demand.

This phase is conditional on a supported MoE model and a runtime/model path that exposes expert routing or permits reliable instrumentation. Do not claim expert-level scheduling when only layer-level offloading is available.

#### R4.1 Trace format

Define a versioned trace schema with, at minimum:

- model/config hash and architecture identifier;
- layer ID and expert ID;
- token/request sequence ID (pseudonymous, no prompt text by default);
- time/order index;
- expert weight bytes;
- request/activation outcome;
- cache hit/miss/collision/eviction/prefetch status;
- source and destination tier;
- transfer bytes and duration where measurable.

Do not persist raw prompt content unless the user explicitly opts in and the data is covered by the existing rights/consent pipeline.

#### R4.2 Policies to compare

Build a policy interface and compare:

- no expert cache / all CPU placement baseline;
- static expert hot set;
- LRU;
- LFU;
- FIFO or random baseline;
- least-stale policy inspired by SpecMD;
- router-informed speculative prefetch if the runtime exposes the required information;
- an offline oracle upper bound for research only (never available in production).

The benchmark must reproduce the same expert traces for every cache policy. Keep model, prompt set, batch, hardware, cache size, and runtime constant.

#### R4.3 Measurements

Report:

- hit rate;
- collision miss rate;
- prefetched-but-unused bytes;
- evicted-before-use bytes;
- transfer volume;
- effective source/destination bandwidth;
- TTFT, TPOT/decode latency, throughput;
- memory consumption;
- output-quality equivalence;
- policy overhead and CPU cost.

**Do not optimize only hit rate.** A high hit rate with large transfers, costly prefetch, or synchronization stalls can still be slower.

#### R4.4 Acceptance

A policy must beat the real baseline on a representative workload, with repeatable latency/throughput improvement and no unacceptable quality change. Use SpecMD's findings as a research hypothesis, not a guaranteed outcome on this machine. The paper's OLMoE result and its cache sizes are not transferable performance promises.

### R5 — Training reliability and methods beyond the current smoke path

Prioritize correctness and auditability before adding more trainer types.

Implement in order:

1. Real resume semantics and conformance C5.
2. Timely cancellation with persisted terminal state and conformance C6.
3. RFT (V8-S4) through the existing trainer registry.
4. A larger but leakage-safe set of task generators, including `code_synth` and `code_fix`, wired to the existing subprocess/venv execution verifier.
5. More robust dataset generation and decontamination.
6. Optional advanced methods only after the common `TrainSpec`, preflight, manifest, cancellation, checkpoint, and evaluation contract is stable.

Any new method must implement the same lifecycle contract:

- `preflight(spec, context)` — no training side effect;
- `run(spec, context)` — bounded execution and progress events;
- checkpoint and resume support where promised;
- cancellation acknowledgement and durable `cancelled` state;
- manifest with hashes, hyperparameters, seed, loss trajectory, parameter delta, artefacts, environment, and resource usage;
- adapter/checkpoint reload validation;
- a clear error when the hardware or dependency stack is unsupported.

Do not add distributed FSDP/DeepSpeed requirements to the default local install. They may be optional backends and should only be enabled when the machine and package versions are actually compatible. The local custom INT4 path and CPU-only PyTorch path remain first-class.

### R6 — Quantization laboratory

Create a common comparison harness for weight formats and training-store formats.

Keep these categories separate:

- GGUF inference quantization: Q2/Q3/Q4/Q5/Q6/Q8 and supported importance-aware formats.
- Training base-weight storage: the repository's custom INT4 representation, scales, metadata, residual tensors, and gradient path.
- QLoRA training representation: NF4/double quantization and supported PEFT/optimizer stack.
- KV-cache quantization: runtime-supported K/V types and any experimental TurboQuant prototype.
- Embedding/vector compression: only when the RAG/semantic retrieval subsystem is in scope.

For each candidate, record:

- input checkpoint hash;
- quantizer binary/version/commit;
- quantization method and options;
- calibration dataset and hash;
- output file size and ratio;
- load memory and time;
- TTFT and decode throughput;
- perplexity or relevant quality measure;
- KL divergence or another justified distributional metric;
- top-1 flip rate;
- deterministic task score and regressions;
- dequantization/reload/inference verification.

Do not assume that lower bits always win. A compressed model can be slower due to decompression, poor kernel support, or memory movement. Report quality and performance together.

### R7 — Knowledge memory / RAG (optional after core stability)

This is a separate product capability, not a substitute for the current training/data pipeline.

If added, implement a local-only, source-grounded path:

```text
selected files → parser → chunker → metadata/ACL → embeddings
             → local vector index + lexical index
             → retrieval → reranker (optional) → cited context
```

Requirements:

- file-root allowlists and user-selected imports;
- source hashes, document versioning, chunk IDs, provenance, and delete propagation;
- local embedding model selection and fit estimate;
- BM25/lexical baseline before vector-only assumptions;
- retrieval metrics (Recall@k, MRR/nDCG on labelled fixtures) and latency;
- reranking only when it measurably helps;
- clear UI labels for retrieved source vs model-generated answer;
- no cloud embeddings unless explicitly opted in;
- no automatic upload of documents.

Engram is a model-architecture research direction for conditional static lookup. It is **not** a ready-made RAG plugin, and the public demonstration repository explicitly says its demo mocks standard model components. Do not represent it as production-ready integration. Any Engram experiment belongs in a separate research branch with a clearly scoped architecture and license review.

### R8 — Browser WebGPU/WASM: measure real inference

The repository already has capability detection, model inventory, local serving, safe limits, and a browser stress script. The next meaningful step is not another panel; it is a real compatible ONNX model run.

- Install or generate a small redistributable ONNX fixture with tokenizer/config files.
- Pin Transformers.js, ONNX Runtime Web, model export version, and all local model artefacts.
- Keep `allowRemoteModels = false` and verify the network log contains no CDN/model fetch.
- Run a complete load → prefill → generate → cancel → unload lifecycle.
- Measure actual model load time, allocation, TTFT, tokens/s, peak tab memory where observable, and browser console errors.
- Test Chromium and at least one other supported browser where practical.
- Test WebGPU and WASM fallback independently; never infer WebGPU inference support solely from `navigator.gpu.requestAdapter()`.
- Preserve conservative ceilings and refusal messages.
- Keep browser runtime clearly labelled as a small-model path, not GGUF/Vulkan parity.

A browser benchmark without a model installed is `not_run`, not zero tokens/s and not a pass.

### R9 — MCP and agent control plane

The existing MCP server is valuable because it has typed tools, stdio transport, path allowlists, and no generic shell. Extend it deliberately instead of replacing it with a broad command-execution tool.

#### R9.1 Current MCP state

The current server uses JSON-RPC over stdio and the protocol version recorded in the implementation. Existing tools include machine scan/budget, model and dataset listing, dataset export, VRAM/cache status, INT4 training/merge, training plan, evaluation comparison, and GGUF quantization. Some expensive operations are blocking today; the new lifecycle must move long operations to durable asynchronous jobs.

#### R9.2 Tool design principles

Every MCP tool must:

- have a unique, stable, versioned name and a strict JSON schema;
- reject unknown fields (`additionalProperties: false` where appropriate);
- validate IDs and all paths against configured roots;
- enforce resource, time, concurrency, and output-size budgets;
- distinguish plan/read operations from write/execute operations;
- return structured status, artifact IDs, and typed error codes;
- be idempotent where retries are possible;
- support job IDs and status polling for long jobs;
- expose cancellation and durable terminal states where appropriate;
- never write to the repository or overwrite user data implicitly;
- never accept arbitrary shell commands, Python code, URLs to fetch, or arbitrary filesystem paths.

#### R9.3 Incremental MCP roadmap

Add tools only when a stable API/CLI lifecycle exists. Suggested sequence:

**Read-only diagnostics**

- `capabilities_report`
- `runtime_capabilities`
- `model_inspect`
- `storage_profile`
- `run_report`
- `experiment_list`

**Plan-only tools**

- `inference_plan`
- `training_plan` (preserve `train_plan` compatibility)
- `benchmark_plan`
- `quantization_plan`
- `io_benchmark_plan`
- `moe_cache_plan`

**Asynchronous jobs**

- `training_start`
- `benchmark_start`
- `quantization_start`
- `io_benchmark_start`
- `job_status`
- `job_cancel`
- `job_resume` only for trainers that pass resume conformance

**Comparison / report tools**

- `eval_compare` (preserve existing semantics)
- `experiment_compare`
- `artifact_manifest`
- `moe_trace_summary`

**Human-gated actions**

- `upgrade_request_review` may create a review item;
- actual candidate promotion remains a human-approved action in the UI or a separately authenticated approval flow.

Names are proposals, not permission to implement a tool before its API, schema, tests, and security review exist.

#### R9.4 MCP resources and prompts

Where useful and supported by the client/SDK version, add read-only MCP resources such as:

- `foundry://machine/profile`
- `foundry://machine/budget`
- `foundry://models/{model_id}/manifest`
- `foundry://runs/{run_id}/manifest`
- `foundry://datasets/{dataset_id}/manifest`
- `foundry://experiments/{experiment_id}/report`

Prompts may provide reusable workflows such as `plan_experiment`, `compare_quantizations`, and `review_candidate`. Prompts must not bypass permissions or turn a read-only tool into a write operation.

Preserve backward compatibility with current clients. Test initialize, tools/list, tool invocation, errors, cancellation, output limits, and stdio purity. stdout belongs to JSON-RPC; logs go to stderr.

### R10 — Reusable agent skills

Skills teach an agent **how to combine tools**. MCP exposes the live typed operations. Keep these concerns separate.

Use the Agent Skills directory structure with a `SKILL.md` containing YAML frontmatter `name` and `description`, plus optional `scripts/`, `references/`, and `assets/`.

Recommended source-of-truth layout:

```text
.agents/skills/                  # Codex / OpenAI-compatible repo skills
  foundry-repo-audit/SKILL.md
  foundry-research-synthesis/SKILL.md
  foundry-hardware-profile/SKILL.md
  foundry-io-benchmark/SKILL.md
  foundry-quantization-eval/SKILL.md
  foundry-kv-cache-lab/SKILL.md
  foundry-moe-cache-lab/SKILL.md
  foundry-training-conformance/SKILL.md
  foundry-benchmark-statistics/SKILL.md
  foundry-mcp-engineering/SKILL.md
  foundry-ui-ux/SKILL.md
  foundry-release-guardian/SKILL.md
.claude/skills/                  # Claude Code discovery mirror
.github/skills/                  # optional VS Code/Copilot discovery mirror
scripts/sync_agent_skills.py     # checks/generates mirror parity
```

If a supported client discovers skills in a different location, add a documented adapter. Do not assume one tool's discovery directory works for every agent. Avoid hand-maintaining divergent duplicate instructions: implement a deterministic parity/sync check or use a tested symlink strategy where the host supports it.

#### R10.1 Skill contracts

Each skill must include:

- `name` and a concise, high-signal `description` that says when to invoke it;
- explicit triggers and non-triggers;
- required context and preconditions;
- tools/MCP capabilities it needs (by discovered name, never invented name);
- read-only vs mutating operations;
- step-by-step workflow;
- expected outputs/artifacts;
- tests and acceptance criteria;
- safety boundaries and stop conditions;
- links to only the references it needs.

Keep skills focused. Do not load every long reference into every prompt. Use progressive disclosure: agent reads skill metadata first, loads the full `SKILL.md` when relevant, and opens reference files only when needed.

#### R10.2 Required initial skill set

| Skill | Responsibility | Main output |
|---|---|---|
| `foundry-repo-audit` | Reconcile code, tests, docs, branches, and current phase | Audit report + prioritized gaps |
| `foundry-research-synthesis` | Verify papers, official docs, licenses, assumptions, and applicability | Research card + implementation decision |
| `foundry-hardware-profile` | Run read-only system/runtime discovery and safe-budget analysis | Machine profile + capability matrix |
| `foundry-io-benchmark` | Benchmark storage, page-cache, weight-store and transfer paths | Raw samples + I/O report |
| `foundry-quantization-eval` | Compare GGUF/INT4 formats with quality and performance gates | Quantization comparison report |
| `foundry-kv-cache-lab` | Validate and sweep supported KV formats | KV memory/quality/latency report |
| `foundry-moe-cache-lab` | Collect and replay expert traces and compare cache policies | Expert-cache research report |
| `foundry-training-conformance` | Implement trainer lifecycle, resume, cancel, reload and budget checks | Conformance matrix + manifest |
| `foundry-benchmark-statistics` | Run paired experiments, power checks, CI, and regression analysis | Statistical evaluation report |
| `foundry-mcp-engineering` | Extend typed tools/resources/prompts and test protocol boundaries | MCP schemas + protocol tests |
| `foundry-ui-ux` | Improve screens, navigation, states, accessibility and i18n | Tested UI slice + screenshots |
| `foundry-release-guardian` | Verify release claims, lockfiles, CI, EXE, site and changelog | Release report + readiness decision |

#### R10.3 Skill quality gates

- Each skill must have a trigger test or documented manual evaluation prompts.
- The skill must be specific enough to avoid triggering on unrelated tasks.
- Scripts included in a skill must have tests or deterministic safety checks.
- No skill may grant broader permissions than the user/host already granted.
- Skills must never claim that a tool ran when it did not.
- All skill mirrors must pass a content parity test.

### R11 — UI/UX: make the complexity understandable

Do not redesign existing screens blindly. Inspect the current product and extend its information architecture consistently.

The target is a premium, calm engineering studio: clear hierarchy, responsive layouts, restrained motion, accessible contrast, compact but readable data density, dark/light support, and German/English/French parity.

Prioritize these views:

1. **Overview:** current machine budget, active model/runtime, active jobs, last benchmark, warnings, next safe action.
2. **Machine & capabilities:** physical RAM, available RAM, dedicated VRAM, shared/UMA budget, storage devices, runtime capabilities and confidence.
3. **Model Library:** file metadata, hash, parameter count, quantization, fit estimate, verified/unverified status.
4. **Inference Studio:** context, KV format, GPU layers, threads, batch, TTFT/TPS, live memory, logs, cancellation.
5. **Memory / I/O Lab:** separate panels for model weights, KV cache, training activations/optimizer, disk store, cache hit/miss, and measured throughput.
6. **Training Lab:** preflight-first workflow, method selection, resource estimate, dataset manifest, loss graph, checkpoint/resume/cancel, saved adapter validation.
7. **Quantization Lab:** compare size, latency, memory, KL/flip rate, perplexity/task quality and reload verification.
8. **MoE Lab:** model eligibility, expert traces, cache policy, cache size, misses, prefetch waste, TTFT/TPOT.
9. **Benchmarks:** per-case outcomes, paired deltas, confidence intervals, critical regressions and underpowered state.
10. **Upgrade Lab:** explain why a candidate is blocked, what changed, what evidence exists, and which human approval is needed.
11. **Agent Workspace:** visible plan, selected skills, MCP tools called, permission decisions, current job, outputs, errors and next action.
12. **Reports / Experiments:** reproducible manifests and exportable reports.

#### UX rules

- Every number has a unit, scope, timestamp, and source (measured/estimated/reported/unknown).
- Use “not measured”, “not supported”, “not detected”, and “blocked” as distinct states.
- Never render unsupported buttons as enabled.
- Show why a model or job was refused, with the estimate and safe budget side by side.
- Add beginner explanations and expandable advanced details instead of hiding technical controls permanently.
- Use charts only when the data is real and the axes/units are meaningful.
- Provide empty states and error recovery for first launch, no models, no dataset, no browser model, no GPU, no internet, and insufficient memory.
- Keep all visible strings in the existing localization system; test key parity across `en`, `de`, and `fr`.
- Respect keyboard access, focus order, reduced motion, screen readers, responsive widths, and color contrast.

### R12 — cross-platform and packaging hardening

Windows is first-class, but the software architecture must remain portable.

- Keep hardware probes behind per-platform providers and isolate failures.
- Do not hardcode `nvidia-smi`, ROCm, Vulkan, or Metal as universally present.
- Validate CPU-only path, AMD Vulkan where available, NVIDIA where available, and Apple/Metal only on actual supported systems.
- Make optional ML dependencies explicit and lockable.
- Keep the standalone EXE honest: if the ML stack is excluded, the UI must state which training functions require source install/optional dependencies.
- Test path handling with spaces, Unicode, removable drives, and different filesystem permissions.
- Test clean install, upgrade, restart, interrupted job recovery, and uninstall/data-retention messaging.

### R13 — Vercel and website

Keep Vercel as the static marketing/documentation/browser frontend surface unless a new serverless feature has a concrete, separately reviewed reason.

- Never put local secrets, model files, datasets, machine profiles, or private inference endpoints into the deployed bundle.
- Do not try to host the full local FastAPI/training service on static Vercel output.
- Ensure public pages describe browser capability limits honestly.
- Generate changelog/site content from existing source files where possible.
- Validate canonical URLs, metadata, sitemap, robots, accessibility, and responsive layouts.
- Run the production build locally and preview it before deployment.
- Deploy last and record the deployment URL, commit, build ID, and smoke-test result.

---

## 7. Research programme and SOTA handling

### 7.1 Research is a reproducible workflow, not a citation collection

For each research idea, create a research card under `docs/research/`:

```yaml
id: R-MOE-LEAST-STALE
status: proposed | investigating | prototype | measured | rejected | adopted
paper_title: "..."
authors: "..."
venue: "..."
published: YYYY-MM-DD
primary_url: "..."
code_url: "..."
license: "..."
claim_to_test: "..."
required_model_or_trace: "..."
hardware_assumptions: []
foundry_components_touched: []
risks: []
benchmark_protocol: "..."
acceptance_gate: "..."
```

Research workflow:

1. Find the primary paper, official implementation, official documentation, and license.
2. Record publication date, version/commit, method, model architecture, dataset, hardware, baseline, metrics, and limitations.
3. Read the method; do not implement from an abstract or blog summary alone.
4. Map the research to a concrete existing Foundry component.
5. Identify whether the feature is already implemented by the runtime or would require a custom kernel/backend.
6. Prototype behind an experimental flag or a separate branch.
7. Compare against the current baseline with the same workload and hardware.
8. Record negative results, overhead, regressions, and unsupported cases.
9. Only then decide whether it belongs in the main product.

Do not call a method “SOTA” merely because it is recent. Use “recent research candidate” unless a relevant benchmark and fair comparison substantiate a stronger claim.

### 7.2 Priority research sources

The following are high-value starting points for this roadmap. Re-check for updated versions before implementation:

- [I/O for LLM inference: a survey of storage and memory bottlenecks (Artificial Intelligence Review, 24 July 2026)](https://link.springer.com/article/10.1007/s10462-026-11651-1) — organize weight, KV-cache, and activation I/O around memory hierarchy and roofline analysis. Use as a taxonomy and reading map, not as proof that any specific optimization will help this machine.
- [SpecMD: A Comprehensive Study on Speculative Expert Prefetching (ICML 2026, PMLR)](https://proceedings.mlr.press/v306/hoang26d.html) — benchmark study of MoE caching policies and a Least-Stale policy. Its reported results are tied to its model/hardware/workload; reproduce rather than copy the headline numbers.
- [TurboQuant (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html) — low-bit vector quantization; evaluate the actual KV-cache path, dequantization cost, and model quality.
- [A First Comprehensive Study of TurboQuant: Accuracy and Performance (vLLM project, 11 May 2026)](https://vllm-project.github.io/2026/05/11/turboquant.html) — a useful implementation/performance reality check; storage compression does not automatically imply faster attention computation.
- [DeepSeek Engram official repository](https://github.com/deepseek-ai/Engram) — conditional N-gram lookup memory as an additional sparsity axis. Its demo explicitly mocks parts of the surrounding model; treat it as research, not an immediately pluggable production module.
- [PagedAttention / vLLM paper (SOSP 2023)](https://doi.org/10.1145/3600006.3613165) — useful reference for paged KV-cache allocation and sharing. It is most relevant when Foundry's serving workload needs multiple concurrent sequences or more efficient KV allocation; do not add a serving stack just to match a paper.
- [OpenAI/Codex skill authoring documentation](https://learn.chatgpt.com/docs/build-skills) — repo skill format and `.agents/skills` discovery.
- [Claude Code / Claude Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — `.claude/skills` discovery and `SKILL.md` structure.
- [MCP Tools specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-06-18/server/tools.mdx) — typed tool discovery, invocation, and human-interaction expectations. Verify the protocol version supported by the SDK and clients before upgrading.

### 7.3 Research interpretation rules

- **MoE:** active parameters reduce per-token compute/weight traffic; total expert weights still need storage and availability.
- **TurboQuant:** do not assume the Foundry's installed runtime supports it. Do not silently map it to Q4 or another KV dtype.
- **Engram:** not equivalent to RAG, disk offload, expert cache, or KV cache.
- **PagedAttention:** solves KV allocation/sharing issues in serving systems; it is not a generic weight-compression format.
- **QLoRA:** quantized base plus trainable adapters; it does not eliminate activation/optimizer/runtime memory or framework compatibility requirements.
- **SSD/HDD offload:** increases capacity by trading latency/bandwidth for capacity; do not describe it as equivalent to VRAM.
- **WebGPU:** detected capabilities do not prove successful model inference; only a real load/generate run can.

---

## 8. Testing, benchmark, and evidence standards

### 8.1 Standard validation commands

Use the current lockfiles and repository instructions. The following are the known CI-aligned commands at the reviewed snapshot; inspect package scripts before assuming they remain unchanged.

**Backend**

```bash
cd server
python -m pip install -r requirements-train-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
FOUNDRY_REQUIRE_TRAIN=1 python -m pytest -q -rs --strict-markers
```

On PowerShell:

```powershell
$env:FOUNDRY_REQUIRE_TRAIN = "1"
python -m pytest -q -rs --strict-markers
```

**Frontend**

```bash
cd web
npm ci
npx tsc --noEmit
npx vitest run
npm run build
```

**E2E**

```bash
cd web
npx playwright install chromium
npx playwright test
```

**Machine/runtime**

```bash
foundry run doctor
foundry machine scan
```

Use the actual CLI syntax from `docs/CLI.md`; do not assume an example command exists if the CLI has changed.

**MCP**

```bash
cd server
python -m pytest tests/test_mcp.py -q
python -m foundry.cli mcp serve
```

For manual protocol tests, send JSON-RPC only through stdin and verify stdout contains protocol messages only. Logs belong on stderr.

**Standalone executable (Windows)**

```powershell
.scripts\build_exe.ps1
```

Inspect the script before running and use the exact current path/requirements. Never overwrite an existing release artifact without a clean output path.

### 8.2 Required test layers

| Layer | Purpose |
|---|---|
| Unit | formulas, schemas, parsers, quantization metadata, policy decisions |
| Property / fuzz | path validation, GGUF header limits, split isolation, manifest parsing |
| Integration | database migrations, runtime supervisor, file allowlists, training lifecycle |
| MCP protocol | initialize, tool list, schema rejection, tool call, errors, stdio isolation |
| E2E | real browser flows, no console errors, loading/cancel/error/empty states |
| Hardware smoke | bounded real model load/run where hardware supports it |
| Performance | repeated warm/cold runs with machine/runtime fingerprints |
| Quality | paired per-case evaluation and critical guard suites |
| Release | clean checkout, lockfiles, EXE/site build, no private artefacts |

### 8.3 Statistical discipline

- Keep case IDs stable across baseline and candidate.
- Do not compare different suite versions as a paired result.
- Keep task templates or source groups clustered where examples are correlated.
- Run a power/sample-size check before interpreting small score deltas.
- Report confidence intervals and required sample size.
- Do not promote with an underpowered evaluation.
- A critical regression blocks promotion even when the aggregate mean improves.
- Preserve holdout hashes; a changed holdout requires a new baseline and new evaluation.

### 8.4 Evidence file format

Each major phase should produce `docs/evidence/<phase>_<topic>.md` with:

1. Scope and commit.
2. Machine profile and software versions.
3. Baseline and hypothesis.
4. Exact commands/configuration.
5. Raw or machine-readable result location.
6. Summary statistics and uncertainty.
7. Quality/safety regressions.
8. What passed, failed, was not applicable, or was not measured.
9. Decision: adopt, retain behind flag, defer, or reject.
10. Reproduction instructions.

Never put large model files or private raw datasets in evidence folders. Store only metadata, hashes, small safe fixtures, and links to ignored local artefacts.

---

## 9. Security and autonomy model

### 9.1 Permission tiers

Classify every action:

- **Tier 0 — read-only:** inspect repository, tests, docs, model metadata, machine budget, and existing run reports.
- **Tier 1 — bounded local write:** create code/tests/docs/experiment files inside the repository or configured data root; no user-data deletion or export.
- **Tier 2 — resource-consuming action:** download models, run long benchmarks, start training, quantize large files, or allocate substantial memory/disk. Require visible plan, budget, cancellation path, and permission.
- **Tier 3 — sensitive/external action:** export/import personal data, call external services, publish a release, merge protected branches, deploy, overwrite/delete user assets, or promote model candidates. Require explicit human approval.

The agent may automate Tier 0 and routine Tier 1 work. Tier 2 should be preflighted and budget-limited. Tier 3 must preserve a human approval boundary.

### 9.2 Subprocess rules

- No generic `/exec`, shell, Python-eval, or arbitrary command MCP tool.
- Native subprocesses must be built from a typed allowlist with validated executable path, arguments, cwd, environment, timeout, and output cap.
- Benchmark code and generated code must run in a dedicated temp workspace with limited permissions and timeouts. The repository itself warns that a subprocess is not a VM/container; do not represent it as a hardened security sandbox.
- Do not run untrusted repository scripts or imported agent commands on the host without explicit approval and an appropriate sandbox.
- Capture only necessary logs; redact secrets and sensitive local paths where practical.

### 9.3 Data and rights

- Preserve the existing consent/rights states (`confirmed`, `needs_review`, `restricted`, `excluded`).
- Only confirmed/frozen datasets may be exported or used for training/upgrade candidates.
- Preview and redact before persisting imported data.
- Maintain source IDs, hashes, transformation versions, and synthetic labels.
- Do not train on evaluation/holdout content.
- Do not generate hidden-reasoning datasets. Use observable actions, tool results, tests, and diffs only.

---

## 10. Multi-agent orchestration

The orchestrator may assign specialized workstreams, but parallel execution must not create competing changes to the same files or invalidate the shared baseline.

| Role | Owns | Must deliver |
|---|---|---|
| Principal Orchestrator / Architect | phase order, ADRs, integration, task decomposition | coherent plan, reconciled status, merged tested slices |
| Research Agent | papers, docs, licenses, method reproduction | research cards and falsifiable hypotheses |
| Hardware / Runtime Agent | machine discovery, llama-server capabilities, Vulkan/CPU paths | capability matrix and runtime tests |
| Storage / Memory Agent | INT4 store, offload, page cache, KV memory | I/O traces, benchmark reports, safe memory changes |
| Training Agent | TrainSpec, trainers, resume/cancel, RFT, manifests | conformance matrix and verified training artefacts |
| Evaluation Agent | task generators, suites, statistics, holdouts | per-case results and paired comparisons |
| MCP / Skills Agent | tools, resources, prompts, skills and client configs | schema-safe MCP surface and skill tests |
| UI / Product Agent | navigation, forms, charts, i18n, accessibility | complete user-visible vertical slice |
| Security / QA Agent | threat model, permissions, dependencies, regression tests | security report and release-blocking findings |
| Release Agent | changelog, docs, EXE/site, tags, reports | release readiness report |

### 10.1 Parallelism rules

- Assign file ownership or clearly bounded directories before parallel work.
- Agents must not edit the same manifest, shared API schema, or core config simultaneously.
- The orchestrator reviews all merges and resolves architecture conflicts.
- Research agents may work in parallel with coding only when their output cannot silently alter the acceptance criteria mid-run.
- A benchmark run must not overlap another memory-heavy experiment on the same machine unless concurrency is itself the subject of the test.
- Preserve one canonical baseline and one canonical experiment registry.

### 10.2 Agent communication format

Every sub-agent handoff must include:

```text
Task:
Files inspected:
Files changed:
Facts verified:
Assumptions:
Commands/tests run:
Results:
Risks/regressions:
Artifacts/evidence:
Commit:
Next recommended action:
```

---

## 11. Prioritized execution backlog

The orchestrator should maintain a live issue/task list. Initial priority order:

### P0 — Reconcile current state

- [ ] Confirm current branch/commit and clean/dirty state.
- [ ] Re-read current `AGENTS.md`, `PLAN.md`, and `docs/plan/V8_STABLE_PLAN.md`.
- [ ] Verify S4–S7 status from source/tests/evidence, not the previous session summary.
- [ ] Reconcile MCP tool count/list in README, docs, and `TOOL_SPECS`.
- [ ] Mark `docs/evidence/v8_baseline.md` as historical if it is not the current baseline.
- [ ] Run current CI-aligned tests and record failures before any changes.

### P1 — Finish V8

- [ ] Implement RFT with zero-acceptance refusal.
- [ ] Close trainer resume/cancel semantics or document exact limitations.
- [ ] Expand deterministic task generators, including code execution/fix families.
- [ ] Generate sufficient held-out tasks and run the base-vs-candidate flywheel.
- [ ] Complete storage I/O and time-prediction experiments.
- [ ] Complete quantization/KV quality and performance sweeps supported by the actual runtime.
- [ ] Produce V8 evidence and release report.

### P2 — Make experiments first-class

- [ ] Add versioned experiment schema and registry.
- [ ] Add raw sample persistence and artifact hashes.
- [ ] Add machine/runtime fingerprints and measured/estimated/unknown labels.
- [ ] Add a report viewer and reproducibility command.

### P3 — Storage and I/O optimization

- [ ] Benchmark current INT4 path cold/warm and local/external storage.
- [ ] Prototype sequential layer reader.
- [ ] Evaluate residual dtype-preserving bf16 path.
- [ ] Compare cache admission and prefetch against no-cache baseline.
- [ ] Validate ETA predictor against held-out runs.

### P4 — Runtime and KV lab

- [ ] Build live runtime capability detection from actual binary/version/help.
- [ ] Expose typed, validated inference plans.
- [ ] Run K/V format sweep with quality metrics.
- [ ] Prototype TurboQuant separately only if the current stack cannot run it natively.

### P5 — MoE trace and caching lab

- [ ] Select a compatible MoE model/trace workload.
- [ ] Define trace schema and privacy constraints.
- [ ] Compare LRU/LFU/static/Least-Stale/prefetch policies.
- [ ] Report hit/miss and end-to-end latency, not only cache hit rate.

### P6 — MCP and skills

- [ ] Add asynchronous job lifecycle tools with budgets and cancellation.
- [ ] Keep existing 11 tool contracts backward compatible.
- [ ] Add the initial skill set under `.agents/skills`.
- [ ] Add Claude/VS Code skill mirrors and a parity checker.
- [ ] Test each skill with positive and negative trigger examples.

### P7 — UX and browser runtime

- [ ] Add honest memory/IO visualizations backed by live metrics.
- [ ] Add beginner/advanced explanations without hiding real capability limits.
- [ ] Run a real ONNX browser inference fixture and measure it.
- [ ] Fix all empty, loading, blocked, error, cancellation, and no-device states.

### P8 — Release and publish

- [ ] Run backend/frontend/E2E/security/performance gates.
- [ ] Build and test the standalone executable on Windows.
- [ ] Perform clean checkout and clean data-root smoke test.
- [ ] Update README/docs/CHANGELOG/site and release evidence.
- [ ] Push/merge through the repository's configured review process.
- [ ] Deploy Vercel static site only after merged release verification.

---

## 12. Definition of done

A phase is `complete` only when all applicable items below are satisfied:

- [ ] The code path exists and is used by the actual product path.
- [ ] The API/CLI/MCP schema is typed, validated, documented, and tested where applicable.
- [ ] All visible UI controls have real behavior, progress, failure, and cancellation states.
- [ ] Unit and integration tests cover success and failure branches.
- [ ] Negative tests prove that invalid inputs, unsupported hardware, OOM risks, and broken artifacts are refused.
- [ ] Security/path/rights boundaries remain intact.
- [ ] Existing tests remain green; no assertion has been weakened to hide a regression.
- [ ] The measured performance or quality result is reproducible and tied to an environment fingerprint.
- [ ] Documentation reflects actual shipped behavior and explicitly states non-claims.
- [ ] Evidence is stored and linked from the changelog/plan.
- [ ] The diff is reviewed for unrelated changes and accidentally staged files.
- [ ] The next action/checkpoint is written.

A release is `stable` only when the release gate in the current stable plan is satisfied. “All code written” is not a release criterion.

---

## 13. Final operating principle

HCS Foundry should become a system that can improve its own engineering workflow without confusing automation with proof.

The agent may research, hypothesize, implement, benchmark, compare, repair, and propose. It must not invent successful measurements, bypass privacy, silently consume unlimited resources, or promote a candidate because its own narrative sounds convincing.

**When a result is positive, preserve the evidence. When a result is negative, preserve the evidence. When a capability is unsupported, explain the boundary. When a test is not applicable, say why.**

That is the standard for every future HCS Foundry release.

---

## Appendix A — repository locations the agent should know

| Area | Current location / entry point |
|---|---|
| Main instructions | `AGENTS.md` |
| Broader roadmap | `PLAN.md` |
| Immediate V8 plan | `docs/plan/V8_STABLE_PLAN.md` |
| Backend | `server/foundry/` |
| API routes | `server/foundry/api/` |
| Machine detection/governor | `server/foundry/machine/` |
| Runtime / GGUF | `server/foundry/runtime/`, `server/foundry/gguf_tools.py` |
| Training | `server/foundry/training/`, `server/foundry/train/` |
| Tasks/generators | `server/foundry/tasks/` |
| MCP server | `server/foundry/mcp_server.py` |
| Agent workspace | `server/foundry/agent.py` |
| Upgrade Lab | `server/foundry/upgrade.py` |
| Backend tests | `server/tests/` |
| Frontend | `web/src/` |
| Frontend E2E | `web/tests/e2e/` |
| Browser stress | `web/scripts/browser_stress.mjs` |
| Evidence | `docs/evidence/` |
| Research cards (to create) | `docs/research/` |
| Progress checkpoints (to create) | `docs/progress/` |
| Skills (to create) | `.agents/skills/`, `.claude/skills/`, optional `.github/skills/` |
| Windows executable build | `scripts/build_exe.ps1` |
| CI | `.github/workflows/ci.yml` |
| Static Vercel app | `web/` and website assets/configuration in the repository |

## Appendix B — research interpretation quick reference

| Idea | What it actually targets | What not to claim |
|---|---|---|
| Q4/Q5 GGUF | Weight storage and supported compute paths | It does not quantize the KV cache automatically |
| NF4/QLoRA | Quantized base weights plus trainable adapters | It does not make arbitrary huge-model training fit on any consumer PC |
| Gradient checkpointing | Activation memory vs recomputation | Savings/time penalty are workload-dependent, not fixed percentages |
| CPU/GPU offload | Placement and transfer of supported tensors/layers | It is not zero-cost memory expansion |
| Disk-backed INT4 store | Training weight storage with I/O trade-offs | It is not equivalent to physical RAM or VRAM |
| TurboQuant | Low-bit vector/KV representation in compatible implementations | It is not automatically available in the installed runtime or automatically faster |
| MoE expert caching | Reduces expert transfer misses when routing/runtime support exists | Active parameter count does not remove the need to store all experts |
| PagedAttention | KV cache allocation and sharing in a serving engine | It is not a generic replacement for all local runtimes |
| Engram | Conditional static lookup as a model architecture component | It is not generic RAG, a file cache, or an existing Foundry feature |
| WebGPU detection | Browser capability discovery | It does not prove model load/generation or native-runtime parity |

## Appendix C — key repository references

- [README](https://github.com/timfromhcs/hcs-foundry/blob/main/README.md)
- [Existing AGENTS.md](https://github.com/timfromhcs/hcs-foundry/blob/main/AGENTS.md)
- [Existing PLAN.md](https://github.com/timfromhcs/hcs-foundry/blob/main/PLAN.md)
- [V8 Stable Plan](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/plan/V8_STABLE_PLAN.md)
- [Architecture](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/ARCHITECTURE.md)
- [MCP documentation](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/MCP.md)
- [Security](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/SECURITY.md)
- [Data and privacy](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/DATA_AND_PRIVACY.md)
- [Benchmark methodology](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/BENCHMARKS.md)
- [Task generators](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/TASK_GENERATORS.md)
- [Browser runtime](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/BROWSER_RUNTIME.md)
- [v7.3.0 hardening evidence](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/evidence/V73_HARDENING.md)
- [4B I/O evidence](https://github.com/timfromhcs/hcs-foundry/blob/main/docs/evidence/v7_4b_speed.md)
- [MCP implementation](https://github.com/timfromhcs/hcs-foundry/blob/main/server/foundry/mcp_server.py)
- [Agent workspace implementation](https://github.com/timfromhcs/hcs-foundry/blob/main/server/foundry/agent.py)
- [Training memory/cache implementation](https://github.com/timfromhcs/hcs-foundry/tree/main/server/foundry/training)
- [Runtime supervisor](https://github.com/timfromhcs/hcs-foundry/blob/main/server/foundry/runtime/llama_server.py)

---

**Document intent:** load this file together with the root `AGENTS.md`. Use it as the long-horizon execution roadmap after reconciling the current repository state. Do not assume every proposed future capability already exists.
