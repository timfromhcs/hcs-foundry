# HCS FOUNDRY — AGENTS.md

> **Project codename:** HCS Foundry  
> **Tagline:** *Your local AI model lab. Built to learn, tested to improve.*  
> **Credit:** Made with ❤️ by **timfromhcs and hcsmedia**  
> **Document type:** Autonomous engineering instructions, product specification, phased implementation plan, quality gates, and release protocol  
> **Research snapshot:** 2026-11-01  
> **Primary delivery order:** inspect machine and repository → plan → build and prove locally → push to GitHub → deploy the web frontend to Vercel last.

---

## 0. Mission and operating contract

You are the autonomous engineering team for **HCS Foundry**. Build a polished, local-first AI development and model-improvement suite that can discover the actual host machine, select the strongest reliable runtime for that machine, run local GGUF models, prepare and fine-tune eligible models, import authorized AI-agent session data, create auditable synthetic training examples, benchmark models and complete agent systems, and use verified results to propose controlled upgrades.

The product should feel like a premium, modern Silicon Valley developer tool: sleek, responsive, animated with restraint, information-dense without being confusing, and genuinely useful rather than a static mock-up. The UI must be approachable to a first-time user while exposing advanced controls to experienced users.

### 0.1 The non-negotiable delivery sequence

**Never reverse this order.**

1. **Discover:** inspect the actual repository, backend, frontend, dependencies, tests, current machine, OS, CPU, RAM, GPU, available accelerators, disk space, drivers, browser capabilities, and installed model runtimes.
2. **Plan:** write a concrete architecture and task plan based on observed facts. Separate verified facts, assumptions, and experiments. Identify the smallest useful vertical slice.
3. **Build locally:** implement and run the complete local product. No GitHub push and no Vercel deployment during this stage.
4. **Validate locally:** run unit, integration, UI, security, performance, hardware, training-pipeline, and benchmark smoke tests. Fix failures and repeat bounded self-healing loops until gates pass or a precise blocker is reported.
5. **Prepare GitHub:** only after local acceptance, clean the repository, remove all private data and large artifacts, update docs, run a clean checkout test, and prepare release commits.
6. **Push GitHub:** push only after local release gates pass and credentials/permissions are available. Do not silently publish user data, model weights, logs, datasets, or secrets.
7. **Deploy Vercel last:** deploy the browser frontend only after the GitHub release is verified. Keep local inference and training on the user's machine. Use a separate hosted compute service only as an explicit later opt-in.

### 0.2 Engineering behaviour

- Do not start coding before the repository and machine audit is complete.
- Do not assume the framework, language, architecture, hardware, or current feature state. Inspect first.
- Preserve working functionality. Prefer small reversible changes over broad rewrites.
- Do not invent successful test results, benchmark scores, device support, training outcomes, model capabilities, or deployment status.
- Do not leave fake buttons, empty screens, mock API responses, dead navigation, placeholder charts, or unimplemented controls in a release build.
- Every visible action must have a real outcome, progress state, failure state, and useful explanation.
- Never silently switch to cloud inference, upload user files, or transmit local session content. Local-first means local by default and local-only unless the user explicitly opts in.
- Network access for fetching packages, models, datasets, or benchmark assets must be explicit, visible, cancellable, and recorded.
- Do not run untrusted code, imported agent commands, benchmark repositories, or training scripts directly on the host without a sandbox and an approval boundary.
- Do not delete, overwrite, migrate, or export source data without a preview and a recoverable backup.
- Never claim that a model is “trained” when the system only created a prompt, changed a config, merged an adapter, or performed inference.
- Keep a visible activity log and machine-readable event trail for autonomous work.
- If an operation cannot be completed safely or on the available hardware, explain the precise reason and offer a smaller local alternative. Do not silently fake it.

### 0.3 Definition of success

A successful release is not a pretty dashboard. It is a verifiable application where a user can:

- Understand the machine's real capabilities and limitations.
- Find and run a compatible local model with a measured, reproducible configuration.
- Import a selected, authorized dataset or session export and inspect what will be used.
- Clean, deduplicate, redact, classify, enrich, version, and split the dataset.
- Create a user-approved identity/persona dataset from a single initial prompt plus optional examples.
- Fine-tune an actually supported model using an appropriate training backend, or receive a truthful “not supported on this machine” result.
- Compare base and candidate models with reproducible benchmarks.
- Turn verified failure cases into new candidate training examples without contaminating held-out test sets.
- Review and approve model upgrades, roll them back, and track exactly what changed.
- Operate the whole workflow through a polished German/English/French browser GUI, with an additional locale-ready architecture for more languages.
- Keep private content and heavyweight compute local until the user explicitly authorizes otherwise.

---

## 1. Product vision and experience

### 1.1 Name and identity

Use **HCS Foundry** as the working product name. The product is a model workshop / training lab, not another generic chat interface.

Visual direction:

- Modern developer-tool aesthetic: premium, calm, crisp, technical, and friendly.
- Dark-first interface with a carefully implemented light theme.
- Neutral graphite surfaces, subtle depth, restrained accent gradients, thin borders, readable contrast, and purposeful motion.
- No excessive neon, gratuitous glass blur, huge empty hero areas, or overdone gradients.
- Small animated status indicators for running jobs, model loading, benchmark progress, and agent activity.
- Responsive desktop-first layouts that still work on tablets and mobile browsers.
- Accessible focus states, keyboard navigation, reduced-motion support, and semantic labels.
- Consistent typography, spacing, iconography, chart conventions, empty states, and error presentation.

### 1.2 Main information architecture

Build a navigation structure around these areas. Adapt the exact labels to the existing app if one already exists, but retain the capabilities.

1. **Overview** — machine status, currently loaded model, active jobs, recent results, health warnings, quick actions.
2. **Machine** — hardware scan, capability matrix, resource budgets, runtime diagnostics, safe limits, performance profile.
3. **Models** — local model library, GGUF metadata, model discovery, downloads, compatibility, inference profiles, adapters, versions.
4. **Playground** — interactive local chat / completion test, prompt presets, context controls, tool calling where supported, latency and token statistics.
5. **Data Studio** — source imports, session connectors, redaction, deduplication, transformations, synthetic enrichment, data lineage, splits.
6. **Identity Maker** — one-shot persona/identity profile creation, editable identity card, style tests, dataset generation, approval, optional fine-tuning.
7. **Training Lab** — training environment checks, supported methods, dataset selection, hyperparameters, job queue, logs, checkpoints, adapter export.
8. **Benchmarks** — model and agent-system comparisons, coding, agentic, visual, multimodal, and custom benchmark suites.
9. **Upgrade Lab** — failure analysis, candidate generation, recursive improvement proposals, regression review, promotion and rollback.
10. **Agent Workspace** — agent tasks, plans, tool permissions, MCP integrations, approval queue, execution timeline, artifacts.
11. **Runs & Reports** — searchable history of inference, training, import, benchmark, and upgrade runs.
12. **Settings** — language, appearance, storage, privacy, data retention, model paths, runtime settings, network permissions, safety controls.

### 1.3 First-run experience

The first run must be useful without requiring an account, remote API key, or cloud service.

1. Explain that the app is local-first and show exactly what remains on the device.
2. Request permission for a hardware scan and explain the scan scope.
3. Run a bounded, read-only machine scan.
4. Show a capability report with confidence and “not detected” states.
5. Recommend a small, safe model/runtime test based on the scan.
6. Offer a local model folder selection or an explicit model download.
7. Run a short inference smoke test and record baseline speed/memory.
8. Offer a tour of Data Studio, Training Lab, and Benchmarks without requiring imports.

Never initiate a large model download or a training job automatically on first launch.

---

## 2. Phase 0 — repository, backend, frontend, and machine audit

This phase is mandatory and read-only except for temporary diagnostic files in a dedicated temp directory.

### 2.1 Repository reconnaissance

Inspect and record:

- Repository root and all top-level directories.
- Frontend framework, build tooling, route structure, component system, styling, state management, and localization approach.
- Backend framework, API contracts, worker model, persistence, authentication model (if any), job queue, and process lifecycle.
- Existing AI/model integrations, inference servers, training code, data import code, agent tools, MCP support, and benchmark modules.
- Package managers, lockfiles, runtime versions, build scripts, linting, type checking, formatting, tests, CI, and release automation.
- Existing README, architecture notes, contribution rules, licenses, changelog, environment templates, and deployment config.
- Existing TODOs, unfinished UI, placeholder APIs, disabled features, dead routes, and known bugs.
- Current security posture: secrets, permissive CORS, arbitrary command execution, unsafe file access, unsafe deserialization, and dependency vulnerabilities.

Use existing tools and conventions when sound. Do not replace a working stack simply because another stack is fashionable.

### 2.2 Machine discovery requirements

The first architecture decision must be based on a fresh runtime scan. Never hard-code one presumed development machine into the release.

Detect and report, where available:

- OS name, version, architecture, build, and shell.
- CPU vendor/model, physical/logical cores, architecture extensions, clock information where available, and thermal/power constraints where safely observable.
- Total and available system RAM.
- GPU vendor/model, device IDs, dedicated VRAM, shared memory, and whether memory is unified/shared.
- GPU APIs and drivers: Vulkan, CUDA, HIP/ROCm, DirectML, OpenCL, Metal, WebGPU, and any relevant runtime actually detected.
- NPU availability and supported runtime, if present.
- Disk free space and filesystem permissions for model cache, datasets, checkpoints, temporary files, and logs.
- Browser engine/version, WebGPU support, WASM SIMD/thread support, SharedArrayBuffer availability, and relevant security/isolation constraints.
- Installed Python, Node.js, Git, CMake, compilers, Rust, `uv`, `pip`, `torch`, `transformers`, `llama.cpp`, and model runtimes.
- Existing local inference endpoints only after a user-approved localhost probe. Do not scan the LAN.
- Process-level memory limits and whether the app can safely spawn child workers.

### 2.3 Windows-first discovery, without Windows-only architecture

Windows is a first-class target. Use native read-only probes where possible:

- PowerShell/CIM/WMI for OS, CPU, RAM, GPU, and disk metadata.
- `nvidia-smi` only if an NVIDIA runtime/device is detected.
- `rocminfo` / HIP tooling only if available; absence is not a crash.
- `vulkaninfo --summary` only if installed; otherwise report Vulkan as unverified.
- DirectML and Windows ML capability checks only when a tested backend is available.
- Browser WebGPU capability check from the frontend.

Provide corresponding Linux and macOS discovery providers. Each detector must be isolated behind an interface and must fail independently; one missing utility must not prevent the rest of the scan.

### 2.4 Distinguish physical VRAM from shared UMA memory

This is critical for APUs and integrated GPUs.

- Never treat reported shared GPU memory as dedicated VRAM.
- Never assume a BIOS UMA reservation is free application memory.
- Record `dedicated_vram_bytes`, `shared_gpu_memory_bytes`, `system_ram_total_bytes`, and `system_ram_available_bytes` separately.
- Use a conservative runtime budget that leaves memory for the OS, browser, display, and other processes.
- Re-measure available memory immediately before loading or training a model.
- Stop or downshift cleanly before the system becomes unstable.

### 2.5 Audit artefacts

Before implementation, create:

- `docs/audit/REPOSITORY_AUDIT.md`
- `docs/audit/MACHINE_PROFILE.json` (machine-specific; **never commit this file**)
- `docs/audit/CAPABILITY_MATRIX.md`
- `docs/architecture/DECISIONS.md`
- `docs/architecture/IMPLEMENTATION_PLAN.md`
- `docs/architecture/KNOWN_LIMITATIONS.md`

Mark every statement as one of:

- **Verified** — directly observed or tested.
- **Likely** — strongly supported but not yet exercised.
- **Unknown** — not detected or not tested.
- **Unsupported** — known not to work in this configuration.

Do not make the next phase's design decisions until the audit is sufficiently complete.

---

## 3. Phase 1 — architecture and technical design

### 3.1 Recommended system shape

Prefer a clear separation of responsibilities:

```text
┌────────────────────────────────────────────────────────────┐
│ Browser GUI — React/TypeScript (existing framework if sound)│
│ i18n · charts · visualizations · job controls · WebGPU/WASM │
└────────────────────────────┬───────────────────────────────┘
                             │ localhost HTTP + WebSocket/SSE
┌────────────────────────────▼───────────────────────────────┐
│ Local Foundry Service                                      │
│ FastAPI/Python or existing equivalent                      │
│ auth/pairing · API · job orchestration · permissions       │
└────────────┬─────────────────┬─────────────────┬───────────┘
             │                 │                 │
┌────────────▼────────┐ ┌──────▼──────────┐ ┌────▼─────────────┐
│ Inference adapters  │ │ Training worker │ │ Data/Agent worker│
│ llama.cpp / GGUF    │ │ PyTorch/TRL/PEFT│ │ import + sandbox │
│ Vulkan/CPU/etc.     │ │ eligible backend│ │ eval + transforms│
└────────────┬────────┘ └──────┬──────────┘ └────┬─────────────┘
             │                 │                 │
┌────────────▼─────────────────▼─────────────────▼───────────┐
│ Local storage                                               │
│ SQLite metadata · content-addressed artifacts · datasets   │
│ checkpoints · reports · explicit cache and retention rules │
└────────────────────────────────────────────────────────────┘
```

This is a reference shape, not permission to rewrite an existing good architecture. If the repository already has a reliable backend, adapt it. The key rule is that the browser UI must not pretend it can manage arbitrary host processes on its own.

### 3.2 Local browser connection model

A web page cannot safely or portably spawn native training processes, inspect all system hardware, or read arbitrary files on the user's computer. Implement a small local companion service / runtime bridge.

Requirements:

- Bind to loopback only by default (`127.0.0.1` / `::1`); never expose the local control plane to the LAN by default.
- Use an explicit pairing flow between browser and local service.
- Validate `Origin`, host, CORS, CSRF protections, and pairing token on every applicable request.
- Use short-lived scoped tokens, rotate them, and never store them in a public frontend bundle.
- Restrict allowed origins to the exact local UI origin and explicitly paired deployment origin.
- Require user approval for high-impact actions: shell commands, external network calls, file writes outside the project workspace, deletion, model downloads, training, and deployment.
- Provide a visible connection status and a one-click disconnect/revoke control.
- Expose only typed, documented APIs; never a generic “execute arbitrary command” endpoint.
- Use WebSocket or Server-Sent Events for progress, logs, and job events; handle reconnect, sequence IDs, cancellation, and backpressure.

### 3.3 Browser WebAssembly / WebGPU strategy

Support browser-local capabilities as an additional runtime, not as the sole backend.

Use a capability-tested adapter system. Evaluate, rather than blindly bundle, options such as:

- `llama.cpp` WebGPU/WebAssembly build for compatible models and browsers.
- WebLLM / MLC for supported model packages and WebGPU inference.
- Transformers.js with WASM CPU and WebGPU execution providers for compatible tasks.
- ONNX Runtime Web for suitable embedding, reranking, vision, or smaller model workloads.

The browser runtime is suitable for small/local previews, embeddings, lightweight inference, tokenization, and offline UI functionality when its model format and browser are compatible. It is **not** a promise that large GGUF models can run at desktop performance in every browser, nor a promise that browser WASM can practically train large models.

Main inference and training should use the local native service where possible. Browser WebGPU/WASM must have a graceful fallback to CPU or the local service. Feature-detect at runtime; do not rely on user-agent strings alone.

### 3.4 Suggested stack, only if the audit confirms it fits

- Frontend: TypeScript + React; retain an existing framework if it is already sound. Vite is a good fit for a local SPA and simple static Vercel deployment; Next.js is acceptable if the repository already uses it or server-rendering provides a concrete benefit.
- UI: accessible component primitives, design tokens, CSS variables, responsive layouts, and a restrained animation library already compatible with the project.
- State: server state separated from UI state; typed API client; schema validation at the boundary.
- Local API: Python + FastAPI + Pydantic is a practical default for model ecosystem integration; do not introduce a second backend language without a clear measured reason.
- Workers: a supervised worker process per heavy job class, with cancellation, heartbeats, process limits, structured logs, and recovery.
- Storage: SQLite for metadata/job records; filesystem or content-addressed object store for large artifacts. Never store model weights as database blobs.
- Inference: `llama.cpp` / `llama-server` as a preferred GGUF adapter, alongside other runtime adapters only when proven useful.
- Training: PyTorch + Transformers + TRL + PEFT, with Unsloth or another accelerator-specific toolkit only after its supported hardware matrix is tested on the actual machine.
- Packaging: Python virtual environment managed by the existing package manager; prefer `uv` if no project decision exists. Avoid making Docker mandatory for local Windows use.
- Tests: existing test framework first; otherwise pytest for backend, Vitest for frontend units, Playwright for browser E2E, and a small set of real local-engine smoke tests.

Pin known-good dependencies and capture exact versions in lockfiles. Do not blindly upgrade every dependency to “latest”.

### 3.5 Architecture decision records

For each material choice, create an ADR with:

- Context and observed constraints.
- Options considered.
- Decision and why it fits this machine/product.
- Trade-offs and failure modes.
- How to test the choice.
- Revisit conditions.

---

## 4. Phase 2 — machine-aware runtime and resource governor

### 4.1 Capability profiles

Create capability profiles based on measured limits, not only GPU model names. Profiles are dynamic and may change when the machine is under load.

Suggested broad tiers:

- **Tier 0 — CPU-only / constrained:** small quantized inference, low concurrency, short context, embeddings/reranking where practical, dataset preparation, tiny experiments.
- **Tier 1 — integrated GPU / shared memory:** Vulkan/WebGPU/DirectML only after a real test; conservative UMA budgets; CPU hybrid fallback; small-to-mid quantized inference depending on free RAM and thermal headroom.
- **Tier 2 — entry discrete GPU:** quantized inference and possibly small adapter training if the actual backend and memory budget support it.
- **Tier 3 — mid/high GPU memory:** larger inference contexts, broader adapter training, parallel evaluation where stable.
- **Tier 4 — workstation / multi-GPU:** larger models, more serious training jobs, multi-run comparisons, optional advanced optimization.

The tier labels are descriptive only. The actual model chooser must calculate a runtime fit estimate for each model and current machine state.

### 4.2 Resource governor

Implement a central resource governor used by inference, training, imports, evaluation, and browser workers.

Monitor:

- Available system RAM.
- Estimated and observed GPU memory usage.
- CPU utilization and worker count.
- Disk free space and temporary/cache use.
- GPU backend health and inference errors.
- Thermal / power information when the OS exposes it safely.
- User-configured limits and OS safety reserve.

The governor must:

- Establish a conservative safe budget and make the budget visible.
- Reserve memory for the OS, desktop, browser, and local API.
- Prevent two memory-heavy jobs from starting concurrently unless explicitly allowed.
- Downshift context length, batch size, micro-batch size, GPU offload, and concurrency when needed.
- Pause or cancel a job before a predicted out-of-memory condition where possible.
- Detect repeated OOMs and propose a lower profile instead of retrying the same failing configuration forever.
- Preserve checkpoints and state before stopping a training job.
- Never kill unrelated user processes.
- Provide manual override with clear risk information; do not override hard safety thresholds.

### 4.3 Runtime benchmark and selection

Do not use “largest model” as the only optimization target. Rank model/runtime configurations using a measured objective such as:

```text
utility = weighted_quality
        + weighted_agent_success
        + weighted_throughput
        - weighted_time_to_first_token
        - weighted_peak_memory
        - weighted_crash_risk
        - weighted_energy_or_thermal_cost
```

Weights must be configurable and the displayed score must explain its components. The user can choose priorities such as “fastest”, “best quality that fits”, “longest context”, “lowest memory”, or “best coding agent”.

Measure at least:

- Cold model load time.
- Time to first token (TTFT).
- Prompt processing tokens/sec.
- Generation tokens/sec.
- End-to-end latency for representative prompts.
- Peak RAM and GPU memory where observable.
- Stability over repeated runs.
- Output correctness on a small task-specific smoke set.

Record machine profile, driver/runtime versions, model hash, quantization, context, batch, offload layers, KV cache type, and exact launch configuration with each run.

---

## 5. Phase 3 — model library, GGUF inference, and optimization

### 5.1 Model library

Build a local catalog that supports:

- Adding an existing local GGUF file or model directory.
- Reading GGUF metadata safely and displaying architecture, parameter count when available, quantization, tokenizer metadata, context claims, file size, source, and hash.
- Verifying file integrity and reporting incomplete/corrupt downloads.
- Storing license, source URL, model card, and usage restrictions.
- Grouping models by coding, general chat, agentic, vision, embedding, reranking, and other supported tasks.
- Identifying whether a model is compatible with the selected runtime and whether that compatibility is tested or merely inferred.
- Managing aliases, tags, notes, versions, and known-good runtime profiles.
- Downloading only after explicit user approval, with resumable downloads, disk-space checks, checksums where available, and cancellation.

Never recommend a model solely because a leaderboard ranks it highly. Evaluate actual memory, context, runtime support, license, and the task.

### 5.2 GGUF inference

Use `llama.cpp` as a core adapter for GGUF inference where supported. Build or select a binary with the best proven backend for the machine, and keep a CPU fallback.

For AMD GPUs, evaluate Vulkan and HIP/ROCm based on the actual GPU, OS, drivers, build, and runtime. Do not assume every AMD device supports every backend equally. For an integrated GPU, verify that acceleration improves real throughput and stability compared with CPU/hybrid execution.

Requirements:

- Manage `llama-server` or equivalent as a supervised process.
- Start/stop/restart safely and capture stdout/stderr.
- Health-check the endpoint before marking a model ready.
- Support an OpenAI-compatible local API where possible.
- Support streaming generation and cancellation.
- Surface context, sampling, grammar/JSON, tool-call, and multimodal support only when the selected runtime/model actually implements them.
- Avoid claiming tool calling or vision support based only on model name.
- Use a per-model profile and persist working configuration.
- Prevent concurrent server instances from fighting over the same port/model resources.

### 5.3 Weight quantization versus KV-cache compression

Keep these concepts separate in the UI and implementation:

- **Weight quantization** reduces model-weight storage and often inference memory/bandwidth. GGUF variants such as Q4/Q5/Q6/Q8 and importance-aware formats affect quality, size, and speed differently.
- **KV-cache quantization/compression** reduces the memory used to retain attention keys and values as context grows. It is a separate runtime capability and may affect long-context quality and performance.
- **Context length** is a model/runtime capability, not a free slider. Do not advertise a large context that the model metadata, runtime, memory budget, and tests do not support.

Provide a quantization experiment matrix rather than hard-coding one winner:

- Weight formats: compare the compatible GGUF quantizations actually available for the selected model.
- KV cache modes: FP16/default, Q8, Q4, and other modes supported by the selected engine.
- Context lengths: test multiple context sizes within safe limits.
- Batch/offload: measure CPU/GPU split, prompt batch, micro-batch, and concurrency.
- Accuracy: compare answer quality, perplexity where meaningful, long-context retrieval, code tests, and tool-call validity.

### 5.4 TurboQuant research/integration track

Treat **TurboQuant** as an experimental KV-cache compression research track, not a universal switch or replacement for GGUF weight quantization.

Google Research introduced TurboQuant, Quantized Johnson–Lindenstrauss (QJL), and PolarQuant in March 2026, discussing KV-memory and vector-search compression and reporting long-context evaluations. The actual project implementation must still verify whether a production-ready implementation exists in the chosen inference engine and whether it supports the exact architecture/backend.

Implementation rules:

1. Research the paper, official implementation, licensing, and current upstream integration status.
2. Add a capability probe; never advertise TurboQuant as enabled if the runtime does not expose it.
3. Implement it behind a versioned KV-cache adapter/feature flag.
4. Compare against the same model/runtime with FP16, Q8, and Q4 KV cache modes where supported.
5. Measure KV bytes/token, peak memory, TTFT, throughput, long-context retrieval, perplexity where meaningful, and benchmark quality.
6. Run repeated trials and report variance.
7. Enable it by default only if a measured quality/performance/memory gate passes for that exact configuration.
8. Keep the baseline path available for rollback.

Do not assume TurboQuant has already been integrated into `llama.cpp`, every GGUF runtime, or the user's GPU. No backend support means “research prototype / not available here”, not “TurboQuant enabled”.

### 5.5 Speculative decoding and advanced optimizations

Evaluate advanced options only when the selected backend supports them:

- Speculative decoding with a small draft model.
- Multi-token prediction or native model-specific acceleration where available.
- Flash attention / fused attention where the backend and model architecture support it.
- CPU/GPU hybrid offload.
- Memory mapping and pinned memory where effective.
- Prompt caching and prefix reuse.
- KV-cache quantization and cache eviction policies.
- Continuous batching only when concurrency is useful and measured.

Every optimization needs a baseline, a feature flag, compatibility checks, and an accuracy/performance regression test. Do not bundle incompatible flags into one opaque “turbo” preset.

### 5.6 Model recommendation UX

For every suggested model show:

- Why it is recommended for the user's task.
- Exact model ID/revision and license.
- Quantization and file size.
- Expected memory range with uncertainty.
- Verified backend support on this machine.
- Expected context support and caveats.
- Download size and destination.
- A “benchmark on this machine” action.

Never promise a model will fit using parameter count alone.

---

## 6. Phase 4 — training environment and fine-tuning

### 6.1 Training is a separate capability from GGUF inference

The normal path should be:

```text
base model (training-compatible source/checkpoint)
        ↓
prepare dataset + tokenizer/template validation
        ↓
SFT / preference / other supported training method
        ↓
adapter checkpoint (usually LoRA/PEFT)
        ↓
validation + benchmark + regression review
        ↓
optional merge into compatible base checkpoint
        ↓
convert/export to GGUF when supported
        ↓
quantize for local inference
        ↓
run final inference and benchmark tests
```

GGUF is primarily an inference and distribution format. `llama.cpp` contains experimental/WIP training examples, but that is not a reason to make direct GGUF training the default workflow. Keep a training-native checkpoint and dataset as the source of truth; export GGUF as a deployment artifact when the conversion path is supported.

### 6.2 Training capability detection

Before showing a runnable training configuration, verify:

- Model architecture is supported by the selected training library.
- Base weights/tokenizer are available in a compatible training format.
- PyTorch build and device backend can actually execute a minimal forward/backward pass.
- Memory budget covers model, activations, optimizer states, sequence length, and dataloader overhead.
- Training method is supported on the detected accelerator/OS.
- Required libraries and versions are compatible.
- Disk budget covers base model, caches, dataset, checkpoints, and output.

Run a tiny disposable preflight training step and confirm loss changes and gradients are finite before enabling a long job.

### 6.3 Training backends

Evaluate and integrate the most suitable supported option, rather than tying the whole suite to one library:

- Hugging Face Transformers for model loading and training support.
- TRL for SFT and supported preference/post-training workflows.
- PEFT for LoRA and other parameter-efficient adapters.
- Unsloth where its current supported hardware/OS matrix matches the actual system.
- Other specialised backends only through isolated adapters and reproducible tests.

Do not assume CUDA-only tools will work on AMD Vulkan hardware. A Vulkan inference path does not imply a Vulkan training path. AMD/Intel training support is backend-, OS-, library-, and version-specific.

### 6.4 Training methods and sensible order

Implement in stages:

1. **Dataset validation and tokenization preview.** No training required.
2. **Small SFT/LoRA smoke job** on a tiny compatible model and tiny dataset.
3. **Normal SFT/LoRA** for style, instruction-following, and domain adaptation when resources allow.
4. **Preference optimization** such as DPO only when high-quality, clearly labelled preference pairs exist and a stable backend is supported.
5. **Other advanced methods** only after the simpler baseline is understood, tested, and measured.

Do not implement GRPO/RL or full fine-tuning merely because they are fashionable. The system must explain the extra cost, reward design, instability risks, and actual fit before enabling them.

### 6.5 Hardware-aware presets

Offer presets based on preflight results, not universal fixed values:

- Tiny smoke test.
- Low-memory adapter training.
- Balanced adapter training.
- High-quality / larger memory.
- CPU experimental mode.

Dynamically constrain sequence length, micro-batch size, gradient accumulation, rank, checkpoint frequency, dataloader workers, and optimizer choices based on the budget. Preserve a complete run config.

For limited memory, the product may recommend a smaller model, fewer examples, shorter sequences, or data preparation only. Never imply that a 20B+ model can realistically be fine-tuned on a small iGPU simply because its GGUF file can be loaded with quantization.

### 6.6 Training job controls

Every job must support:

- Preflight validation and a human-readable cost estimate.
- Start, pause where technically safe, resume from checkpoint, and cancel.
- Live loss/learning-rate/throughput/memory charts where available.
- Logs, exact config, library versions, random seed, model hash, dataset hash, and checkpoint list.
- Error capture and a useful recommendation.
- Automatic checkpoint retention policy with user-configured limits.
- Final evaluation on validation data and a separate held-out test set.
- Adapter export and metadata.
- Reproducibility report.

Training must never overwrite the source model or the last known-good adapter.

---

## 7. Phase 5 — agent-session ingestion and dataset studio

### 7.1 Goal

Allow a user to bring their own exported or locally stored agent conversations into a unified, inspectable dataset pipeline. Support Claude Code, Codex CLI, OpenCode, Antigravity, and generic JSON/JSONL/Markdown exports through provider adapters.

This is a user-controlled data import feature, not silent surveillance or a background scrape of the user's home directory.

### 7.2 Sources and adapter policy

Use documented export paths and command interfaces where possible. Provider schemas can change; every adapter must be versioned, tested with fixture files, and tolerant of unknown fields.

- **Claude / Claude Code:** support user-selected Claude account exports and explicitly selected Claude Code project-session JSONL files. Claude Code commonly stores project sessions under `~/.claude/projects/`; treat on-disk record formats as provider-internal and version-sensitive. Provide a manual folder picker and export-file import. Do not attempt to recover hidden internal reasoning or undocumented private fields.
- **Codex CLI:** support user-selected `$CODEX_HOME/sessions` and archived session exports (default `~/.codex/sessions` where applicable), with an adapter for observed JSONL records. Prefer documented export/session interfaces when available. The official troubleshooting documentation explicitly warns users to inspect shared logs for sensitive information.
- **OpenCode:** prefer the supported export command, including its sanitize option, or a user-selected exported JSON file. The CLI supports session export and sanitization. Do not export authentication files or credentials.
- **Antigravity:** support official user-driven export/import paths where available, plus generic JSON/JSONL/Markdown import. Do not assume a stable, public on-disk transcript schema. If a supported CLI/API is added later, isolate it in a separately versioned adapter and test against documented contracts.
- **Generic agents:** allow a mapping wizard for role, content, timestamp, tool call, tool result, attachments, task result, and metadata.

The app must show the detected source, number of files, size estimate, date range, parse errors, and exact import scope before it reads or copies content.

### 7.3 Consent and rights

Before an import can be used for training, require the user to confirm:

- They own the content or have permission and legal rights to use it for this purpose.
- The selected provider's applicable terms and organizational rules permit the intended processing and training use.
- The content does not include third-party confidential material, customer data, credentials, or personal data without a lawful basis and appropriate permission.
- They understand that imported content is stored locally and may be transformed into training examples.

The app cannot decide licensing or provider terms automatically. It must show a per-source rights/status field: `confirmed`, `needs review`, `restricted`, or `excluded`.

### 7.4 Import pipeline

Use a staged, auditable pipeline:

1. **Select:** user chooses specific files, folders, or exports.
2. **Inventory:** enumerate file names, sizes, hashes, types, and modified dates without parsing message content where possible.
3. **Preview:** show sample records and inferred schema.
4. **Parse:** provider adapter converts source records into a common event schema.
5. **Normalize:** standardize timestamps, roles, content blocks, tool calls, tool outputs, attachments, and metadata.
6. **Redact:** identify secrets, API keys, tokens, passwords, private URLs, email addresses, phone numbers, personal identifiers, and sensitive content. Default to quarantine or redaction, never silent retention.
7. **Filter:** remove empty, corrupt, duplicate, unsupported, out-of-scope, or user-excluded data.
8. **Segment:** reconstruct turns, task episodes, and tool/action sequences without inventing missing content.
9. **Label:** assign task type, language, outcome, evidence, model/provider, and data-quality flags.
10. **Enrich:** optionally create clearly labelled synthetic summaries, teaching annotations, alternatives, and negative examples.
11. **Review:** let the user inspect examples, exclusions, redactions, and synthetic additions.
12. **Split:** create leakage-resistant train/validation/test partitions by conversation, task lineage, and time where appropriate.
13. **Freeze:** hash and version the dataset manifest.

Every stage must be reproducible and reversible from the original imported copy, subject to the user's retention settings.

### 7.5 Canonical event schema

Create a versioned common representation. Example fields:

```json
{
  "schema_version": "1.0",
  "record_id": "stable-content-derived-id",
  "source": {
    "provider": "codex|claude_code|opencode|antigravity|generic",
    "source_file_hash": "sha256:...",
    "source_session_id": null,
    "adapter_version": "...",
    "rights_status": "confirmed"
  },
  "task": {
    "task_id": "...",
    "task_type": "coding|debugging|research|visual|agentic|chat|other",
    "user_request": "...",
    "context_summary": "..."
  },
  "trace": [
    {
      "kind": "assistant_message|tool_call|tool_result|user_message|test_result|artifact",
      "timestamp": null,
      "tool_name": null,
      "content": "...",
      "content_hash": "..."
    }
  ],
  "outcome": {
    "status": "success|partial|failure|unknown",
    "evidence": [],
    "tests": [],
    "user_feedback": null
  },
  "teaching": {
    "decision_summary": null,
    "supporting_evidence": [],
    "negative_example": null,
    "annotation_origin": "source|synthetic|human_reviewed",
    "confidence": null
  },
  "privacy": {
    "redactions": [],
    "contains_sensitive_data": false,
    "approved_for_training": false
  },
  "split": "unassigned"
}
```

Do not store raw provider authentication state, environment variables, API keys, browser cookies, or tool credentials in the canonical dataset.

### 7.6 Reasoning annotations: allowed and required boundaries

The user wants examples with useful connective teaching text explaining why an answer is correct, plus negative examples. Implement that as **evidence-based decision summaries**, not fabricated hidden chain-of-thought.

Rules:

- Do not request, scrape, reconstruct, or fabricate private/hidden chain-of-thought from Claude, Codex, OpenAI, Anthropic, Google, or any other model.
- Do not label synthetic explanatory text as the original model's internal reasoning.
- Do not infer that a hidden reasoning trace exists because a provider displayed an answer or a “thinking” indicator.
- Use only observable evidence: user request, visible assistant output, tool calls, tool results, code diffs, test output, and user feedback.
- Generate a concise `decision_summary` that explains the decisive evidence and why the final action/result is supported.
- Where useful, add a short `method_summary` explaining the approach at a high level, without private internal deliberation.
- Add negative examples that show a plausible mistake, the failure signal, and the corrected rule. Mark them as synthetic.
- Keep all generated annotations traceable to source record IDs and the teacher model/version/prompt used.
- Require an independent validation pass and sample review. If the trace does not justify a conclusion, label it `unknown` rather than hallucinating a reason.

Example of a valid teaching annotation:

> “The first patch was rejected because the failing unit test reproduced the reported bug. The corrected patch changes the null-handling branch and passes both the regression test and the existing suite.”

Example of a valid negative example:

> “Do not treat a successful build as proof that the bug is fixed; the regression test still fails.”

This provides useful supervision without claiming access to private model thoughts.

### 7.7 Synthetic enrichment engine

Build optional enrichment modes:

- Concise outcome summaries.
- Evidence-grounded decision summaries.
- Step-level task labels.
- Tool-use intent labels.
- Test-writing examples.
- Error → diagnosis → correction triples.
- Positive/negative contrast pairs.
- Alternative valid solutions where independently verifiable.
- User-approved style variants.
- Multilingual paraphrases that preserve exact technical meaning.

Each synthetic row must include:

- `synthetic: true`.
- Teacher model name and version.
- Prompt/template version.
- Source record IDs and hashes.
- Timestamp.
- Validation result and reviewer status.
- Confidence / uncertainty labels.

Never allow synthetic rows to silently replace original source rows. Synthetic data must be filterable, removable, and separately benchmarked against original-only training.

### 7.8 Dataset quality controls

Implement:

- Exact and near-duplicate detection.
- Conversation/session lineage grouping.
- Secret scanning and PII detection.
- Broken JSONL and partial-final-line handling.
- Oversized records and attachment handling.
- Language detection with a manual correction option.
- Role and tool-call validation.
- Prompt/answer truncation warnings.
- Template/tokenizer preview.
- Instruction/response alignment checks.
- Contradiction and low-confidence flags.
- Contamination checks against evaluation sets where legally and technically possible.
- Dataset statistics by source, language, task, outcome, length, and synthetic/original ratio.
- Exclusion reasons and an audit trail.

Never split individual messages at random if they come from the same conversation/task. Keep related messages and revisions in the same partition to avoid leakage.

### 7.9 Data export

Support safe exports such as JSONL and supported training formats. Before export, show included fields, redacted fields, synthetic ratio, licenses/rights status, and dataset hash. Keep all raw imported data local by default.

---

## 8. Phase 6 — Identity Maker (one-shot identity-to-dataset)

### 8.1 Product concept

A user can enter one detailed prompt describing the identity they want a model to embody. The Identity Maker turns that prompt into an editable, structured identity profile and a small, clearly labelled synthetic dataset. Optional user-provided examples can improve fidelity.

This feature should help create a consistent assistant persona or role profile. It must not pretend that a single prompt reveals a person's real identity, hidden biography, or facts the user did not provide.

### 8.2 One-shot workflow

1. User enters one initial identity prompt.
2. The system parses explicit identity attributes, voice, goals, values, domain scope, preferred style, boundaries, and example behaviours.
3. It distinguishes user-supplied facts from generated proposals.
4. It generates a draft identity card and a small dataset preview.
5. It creates positive examples, edge cases, refusal/boundary examples, and contrastive negative examples.
6. User reviews, edits, approves, and chooses the language mix.
7. The system validates consistency, non-contradiction, and unsupported claims.
8. The user can export the identity pack or send it to Training Lab if an eligible base model and backend are available.
9. After training, compare base and candidate model on a fixed identity evaluation set.

### 8.3 Identity profile structure

Suggested components:

- Name / label (user-provided or editable).
- Purpose and role.
- Explicitly supplied facts.
- Style and tone.
- Values and priorities.
- Areas of competence.
- Areas where it must say “I do not know”.
- Behavioural rules.
- Boundaries and privacy rules.
- Response format preferences.
- Example responses.
- Counterexamples and correction rules.
- Language preferences.
- Version and provenance.

Generated details must be marked as generated suggestions, not facts. The user must approve identity claims before they enter a training dataset.

### 8.4 Identity dataset design

Generate a balanced dataset rather than thousands of repetitive paraphrases. Include:

- Direct requests.
- Ambiguous requests and clarification behaviour.
- Short and long answers.
- Technical and non-technical explanations.
- Multilingual examples.
- Conflicting instruction tests.
- Privacy and safety boundaries.
- Negative examples showing behaviours to avoid.
- “Unknown / insufficient information” cases.

Do not use the identity dataset as a substitute for factual knowledge or model capability. Persona alignment, factual accuracy, and task performance must be tested separately.

### 8.5 Identity evaluation

Compare the base model and candidate on:

- Voice/style consistency.
- Instruction adherence.
- Factual humility / unsupported-claim rate.
- Boundary adherence.
- Language consistency.
- Task performance regression.

A persona improvement must not be accepted if it causes significant general-task, factuality, or safety regressions.

---

## 9. Phase 7 — benchmarks for models and complete agent systems

### 9.1 Benchmark principle

Benchmark the **whole system** as well as the raw model. A coding agent score can depend on the model, system prompt, tool definitions, harness, sandbox, retry policy, context budget, and environment. Never attribute an agent-system score to the model alone.

Every result must include:

- Model file/hash/revision and adapter hash.
- Runtime and version.
- Agent harness, system prompt, tools, and versions.
- Dataset/benchmark version and license.
- Hardware profile and resource limits.
- Sampling settings, context, max steps, and timeouts.
- Number of trials and seed where possible.
- Raw task outcomes and logs, with sensitive content redacted.
- Aggregate score, confidence/variance, and known limitations.

### 9.2 Coding benchmarks

Build a benchmark adapter architecture and prioritize a small reliable starter suite:

- SWE-bench Verified / Lite for real repository issue resolution.
- LiveCodeBench for time-aware coding evaluation and contamination-aware comparison.
- HumanEval+ and MBPP+ for compact code-generation smoke tests, with contamination caveats.
- RepoBench or repository-level retrieval/code understanding tasks.
- A project-specific test-repair suite built from the user's own repositories, with explicit permission and a strict train/eval split.

For SWE-bench-like tasks, use isolated environments, apply patches, run official tests, and report solved/failed/unresolved rather than relying only on an LLM judge.

### 9.3 Terminal and agentic benchmarks

Evaluate long-horizon tool use and task completion with compatible, licensed environments such as:

- Terminal-Bench 2.0 for terminal-centric tasks.
- GAIA for general assistant tasks involving tools and multimodal information, subject to dataset access/terms.
- Additional maintained tool-use benchmarks where their current format and license are verified.

Measure task success, steps, tool errors, retries, wall time, token cost, safety violations, and recovery behaviour.

### 9.4 Browser and computer-use benchmarks

Where feasible and properly sandboxed, evaluate:

- WebArena or a maintained successor for realistic multi-site browser tasks.
- OSWorld / OSWorld-Verified / current release for desktop and computer-use actions.
- BrowserGym-compatible tasks where useful and maintainable.

Use disposable browser profiles and virtualized or otherwise isolated environments. Never let benchmark agents operate the user's real accounts, personal browser profile, or unrestricted filesystem.

### 9.5 Visual and multimodal benchmarks

Build modular adapters for tasks that match the model's actual modality support:

- MMMU / MMMU-Pro for broad multimodal understanding.
- ScreenSpot or ScreenSpot-Pro for visual GUI grounding where current datasets and terms permit.
- OCR/document/chart understanding suites where licensing and task fit are verified.
- Screenshot-to-action tasks in a controlled test environment.

A text-only GGUF model must not be scored as though it has vision. Use the right vision adapter/projector/preprocessor and test the full pipeline.

### 9.6 UI/UX benchmark category

Create project-specific visual and accessibility regression tests:

- Playwright screenshots at desktop/tablet/mobile breakpoints.
- Route and interaction coverage.
- Keyboard-only navigation.
- Focus order and visible focus.
- Color contrast checks.
- Reduced-motion mode.
- Screen-reader labels and semantic landmarks.
- Layout overflow and responsive regression checks.
- Loading, empty, error, retry, and offline states.

Do not claim a model “passes visual benchmark” based solely on a screenshot similarity score. Use task completion and accessibility checks as separate metrics.

### 9.7 Benchmark runner architecture

Each benchmark adapter must declare:

- ID, version, source, license, supported tasks, modality, environment needs.
- Setup and teardown.
- Dataset and asset hashes.
- Required packages, tools, credentials, or network access.
- Official scoring method.
- Time and memory constraints.
- Safety / isolation requirements.
- Whether the run is official-comparable, approximate, or a project-specific smoke test.

Never label a local subset score as an official leaderboard score. Never scrape leaderboard data in violation of terms. If benchmark assets cannot legally or practically be run locally, show “not run” and link to the official instructions.

### 9.8 Result visualization

Provide:

- Model vs model matrix.
- Base vs fine-tuned delta.
- Coding / agentic / visual / multimodal breakdown.
- Pass/fail task list with trace viewer.
- Latency and throughput charts.
- Memory/time cost chart.
- Failure taxonomy.
- Confidence/variance display.
- Benchmark version and freshness label.
- Exportable JSON, CSV, and Markdown report.

Make comparisons fair: same task set, same harness when possible, matching limits, and clearly stated differences otherwise.

---

## 10. Phase 8 — recursive upgrade and learning loop

### 10.1 The controlled improvement cycle

Build the upgrade loop around evidence and fixed gates, not an unconstrained model that grades itself.

```text
Baseline model + frozen evaluation suite
              ↓
Run tasks and collect observable failures/successes
              ↓
Classify failure causes with evidence
              ↓
Propose candidate data / prompts / tools / adapter changes
              ↓
Validate synthetic data and human-review samples
              ↓
Train candidate adapter or update agent configuration
              ↓
Run validation set and regression suite
              ↓
Run untouched holdout / benchmark gate
              ↓
Independent review + explicit promotion decision
              ↓
Promote candidate OR reject and retain baseline
              ↓
Record lineage and make the next experiment reproducible
```

### 10.2 Failure taxonomy

Classify failures such as:

- Incorrect interpretation of the user request.
- Missing information / failure to ask a clarifying question.
- Incorrect plan.
- Tool selection or argument error.
- Tool execution / environment error.
- Invalid code or patch.
- Test gap / false success claim.
- Hallucinated fact or unsupported conclusion.
- Context loss / long-context failure.
- Visual grounding failure.
- Permission or safety boundary violation.
- Resource exhaustion / timeout.
- Recovery failure.

Each failure label must be supported by a trace, test, or reviewer annotation. The model may propose labels; it cannot silently certify its own labels.

### 10.3 Candidate creation

A candidate can change one or more of:

- Dataset version.
- Synthetic annotation template.
- SFT/LoRA adapter.
- Preference data / preference training configuration.
- System prompt or task prompt.
- Tool schema / tool description.
- Agent planner and retry strategy.
- Runtime settings.
- Quantization/KV-cache profile.

Prefer one controlled variable at a time where possible. Record all combined changes and avoid attributing improvements to the wrong component.

### 10.4 Data flywheel without contamination

- Evaluation failures may become **candidate training data** only after review and provenance capture.
- Never train on the final test/holdout split.
- Never move a failed benchmark item into training and continue reporting the same item as an untouched holdout.
- Maintain a permanently frozen holdout version and a rotating fresh evaluation set where possible.
- Detect duplicates and near duplicates across training/validation/test.
- Preserve original benchmark versions and license constraints.
- Keep synthetic and original data separate in experiment reports.
- Track whether a score gain is due to memorization, prompt change, tool change, or model improvement.

### 10.5 Promotion gates

A candidate can be promoted only when:

1. It has reproducible run metadata and a known parent baseline.
2. Its training data rights and quality checks pass.
3. It passes all required unit/integration tests.
4. It improves the target metric by a configured meaningful threshold, or makes a clearly approved trade-off.
5. It does not regress critical coding, agentic, visual, factuality, privacy, or safety metrics beyond thresholds.
6. Holdout results are evaluated independently of training data.
7. At least one independent validator/oracle or deterministic test supports the result.
8. The user can inspect the diff and approve promotion.

Do not allow the candidate model to approve itself. An LLM judge can provide a secondary signal but must not replace deterministic tests, task oracles, human review, or independent evaluation.

### 10.6 Rollback and model lineage

Store:

- Parent model and adapter hashes.
- Training dataset version/hash.
- Training config and environment.
- Teacher/judge versions and prompts.
- Benchmark reports.
- Promotion decision and approver.
- Exported artifact hash.

Maintain the last known-good model profile. Rollback must restore the previous adapter/model/runtime config without deleting the failed candidate or its report.

### 10.7 Bounded self-healing loops

Use a limited repair loop for bugs and test failures:

1. Reproduce failure.
2. Capture logs and failing test.
3. Diagnose using evidence.
4. Propose the smallest safe patch.
5. Run the narrow test.
6. Run impacted tests.
7. Run full relevant suite.
8. Review the diff for accidental changes.
9. Repeat within a configured retry/time budget.
10. If still failing, stop and report the blocker with exact repro steps.

Never create an infinite “try until perfect” loop. Never hide a failing test, weaken assertions merely to pass, delete a regression test, or change benchmark scoring to inflate results.

---

## 11. Phase 9 — agentic suite and tool orchestration

### 11.1 Agent workspace

Provide a GUI where the user can ask the suite to perform tasks such as:

- Inspect a repository and prepare a plan.
- Import and clean a selected dataset.
- Compare model configurations.
- Run a training preflight.
- Start a benchmark suite.
- Analyze failures and propose candidate improvements.
- Build a report or prepare a release.

The agent must display a plan, tools it intends to use, resources required, permissions requested, expected outputs, and risks before execution.

### 11.2 Tool categories

Use explicit, typed tool adapters:

- Filesystem: read-only inspection, workspace-scoped edits, artifact creation.
- Git: status, diff, branch, local commits, and later explicitly approved pushes.
- Shell: allowlisted commands inside a sandbox; ask before privileged or external actions.
- Model runtime: load/unload, inference, benchmark, health checks.
- Dataset: import, parse, redact, validate, split, export.
- Training: preflight, launch, pause/resume/cancel, checkpoint, export.
- Browser: controlled benchmark browser only; no access to personal profiles by default.
- MCP: optional integrations discovered from local configuration, individually approved and visibly permissioned.

Never expose raw unrestricted filesystem or shell access to an LLM by default.

### 11.3 Agent task lifecycle

Use durable task states:

`queued → planning → awaiting_approval → running → validating → completed`

Additional states:

`paused`, `cancelled`, `failed`, `needs_input`, `rolled_back`.

Each state transition must be persisted. After an application restart, the UI must recover the state of jobs and clearly distinguish recoverable jobs from interrupted jobs.

### 11.4 Autonomy levels

Offer clear modes:

- **Suggest:** plan and recommend; no mutations.
- **Supervised:** can perform approved workspace actions; ask before sensitive operations.
- **Trusted local automation:** allowlisted repeatable actions under user-defined limits; still ask before external network, destructive changes, publishing, or secret access.

Do not make a hidden “full access” mode the default.

---

## 12. Phase 10 — persistence, API, and job reliability

### 12.1 Local persistence

Use SQLite or the existing suitable local database for metadata. Keep large data and model artifacts outside the database.

Suggested entities:

- `machine_profiles`
- `runtime_capabilities`
- `models`
- `model_artifacts`
- `runtime_profiles`
- `datasets`
- `dataset_sources`
- `dataset_records`
- `training_runs`
- `training_checkpoints`
- `benchmark_definitions`
- `benchmark_runs`
- `benchmark_cases`
- `agent_tasks`
- `agent_events`
- `upgrade_candidates`
- `model_lineage`
- `approvals`
- `settings`

Use migrations and schema versions. Back up before schema migration. Make destructive migrations reversible or provide a documented recovery path.

### 12.2 Job runner

Every heavy operation must be a job with:

- Unique ID and type.
- User/agent owner and approval record.
- Input artifact IDs/hashes.
- Resource request and effective limits.
- State, progress, heartbeat, start/end timestamps.
- Logs and structured events.
- Cancellation token.
- Output artifact IDs/hashes.
- Exit status, error type, and recovery hint.

Use atomic writes and durable state transitions. Handle process termination and machine restart.

### 12.3 API quality

- Version local APIs.
- Validate all inputs and outputs with typed schemas.
- Return actionable error codes and safe messages.
- Enforce path allowlists and prevent path traversal.
- Stream progress with sequence IDs.
- Support cancellation and idempotency where appropriate.
- Add request timeouts and payload size limits.
- Avoid returning secrets or full raw conversation records in logs.

---

## 13. Phase 11 — multilingual, responsive, accessible UI

### 13.1 Required languages

Release with:

- German (`de`) — primary initial locale.
- English (`en`).
- French (`fr`).

Use complete translation dictionaries; do not mix untranslated placeholder keys into production screens. Format dates, numbers, file sizes, and units according to locale.

### 13.2 UX requirements

- Clear dashboard hierarchy and progressive disclosure.
- Advanced settings grouped separately from safe defaults.
- Persistent job tray for active training/import/benchmark/agent tasks.
- Toasts only for short status; persistent issues belong in an activity/error panel.
- Empty states explain what the feature does and how to begin.
- Destructive operations use a preview and confirmation.
- Long tasks show progress, current step, elapsed time, cancellation, and logs.
- All charts have labels, units, accessible text summaries, and useful empty states.
- Model download sizes and estimated disk use appear before approval.
- Avoid fake precision in estimates; show ranges and confidence.

### 13.3 Responsive layouts

Test at minimum:

- 360×800 mobile.
- 768×1024 tablet.
- 1280×800 laptop.
- 1440×900 desktop.
- 1920×1080 large desktop.

The model library, benchmark tables, logs, and job details must remain usable without horizontal overflow. Use responsive table strategies rather than shrinking text to unreadability.

### 13.4 Accessibility

- Keyboard access for every action.
- Visible and logical focus order.
- Correct semantic headings and landmarks.
- Screen-reader names for icon-only controls.
- Good contrast in both themes.
- Reduced-motion mode.
- Errors announced accessibly and linked to the relevant field/action.
- Never communicate a status through color alone.

---

## 14. Phase 12 — privacy, security, and safety

### 14.1 Data locality

Default to local storage and local processing. No analytics, session uploads, telemetry, remote model calls, or crash dumps containing user content without explicit opt-in.

### 14.2 Secrets and sensitive data

- Never commit `.env`, API keys, tokens, cookies, private session files, machine profiles, raw datasets, training logs with secrets, or credentials.
- Add robust `.gitignore` patterns and secret scanning.
- Detect and redact secrets before data enters training.
- Keep import originals in a local quarantine area until the user confirms retention.
- Do not read `.ssh`, browser profiles, password stores, cloud credential folders, or unrelated home directories during discovery.
- Do not export OpenCode auth files or any equivalent provider credential store.
- Keep secrets out of prompts, generated datasets, and benchmark artifacts.

### 14.3 Sandbox boundaries

- Run external benchmark environments in isolated processes/containers/VMs where appropriate.
- Docker may be optional for local Windows workflows, but isolation must still exist for untrusted code. Use a suitable alternative or mark the benchmark unsupported.
- Restrict outbound network access by default for benchmark jobs.
- Mount only the working directory and required benchmark assets.
- Prevent imported transcripts from being interpreted as executable instructions.
- Treat all model-generated commands and dataset text as untrusted input.

### 14.4 Auditability

Log actions, not secrets. Record who/what initiated a task, what permissions were granted, what files/artifacts were accessed, and the outcome. Provide local log retention controls and a purge/export feature.

---

## 15. Phase 13 — testing strategy and acceptance matrix

### 15.1 Test pyramid

Build and maintain:

1. Unit tests for parsing, validation, resource budgeting, schema transforms, and state machines.
2. Integration tests for local API, database migrations, worker lifecycle, inference adapters, and dataset pipeline.
3. End-to-end tests for first-run scan, model import, inference, session import, training preflight, benchmark execution, candidate review, and rollback.
4. Hardware-in-the-loop tests on every claimed accelerator/backend.
5. Performance tests for load time, TTFT, throughput, memory, dataset throughput, and UI responsiveness.
6. Security tests for path traversal, CORS/Origin, token handling, arbitrary command access, unsafe archives, secret leakage, and sandbox escape attempts.
7. Regression tests for every fixed bug.

### 15.2 Required acceptance tests

**Repository and machine**

- The app starts without network access after dependencies are installed.
- A machine scan completes even if some probe tools are missing.
- Shared GPU memory is not reported as dedicated VRAM.
- Unknown hardware capability is shown as unknown, not supported.
- A changed resource state causes a fresh budget calculation.

**Model runtime**

- A valid local GGUF is detected and its metadata is displayed.
- Corrupt/incomplete GGUF files are rejected with a useful error.
- A compatible model starts, answers a prompt, streams output, and stops cleanly.
- Unsupported model architecture/backend combinations fail before a misleading “ready” state.
- CPU fallback works where supported.
- A repeated OOM does not trigger an infinite retry loop.

**Data import**

- Each adapter has versioned fixture tests.
- Malformed/partial JSONL is handled safely.
- Duplicate sessions/records are detected.
- Secret/PII scans redact or quarantine risky content.
- A user can inspect and exclude records before training approval.
- Synthetic annotations are labelled and traceable.
- Hidden reasoning is never reconstructed or represented as source content.
- Train/validation/test splits prevent conversation-level leakage.

**Identity Maker**

- One prompt creates an editable identity draft.
- Explicit user facts and generated suggestions remain distinguishable.
- Unsupported facts are not silently invented.
- Negative examples are labelled synthetic.
- User approval is required before exporting/training.

**Training**

- Training preflight catches unsupported hardware/backend/model combinations.
- A tiny smoke job performs a real forward/backward/update cycle where training is supported.
- Loss/metrics are finite and checkpoint restore works.
- The source model remains unchanged.
- A candidate adapter can be exported and reloaded for inference when the backend supports it.
- The system refuses to claim successful training if no parameter update occurred.

**Benchmarks and recursive loop**

- Each benchmark records exact versions/configuration.
- Official and approximate scores are visibly distinguished.
- A failed benchmark cannot be counted as a pass due to a model judge alone.
- Training-data records cannot be silently taken from the frozen holdout.
- A candidate regression blocks automatic promotion.
- Rollback restores a known-good model profile.

**UI**

- All routes render and all visible controls work.
- No dead buttons or placeholder data in production.
- The UI passes responsive, keyboard, and reduced-motion tests.
- German, English, and French translations are complete for all release paths.
- Error, empty, loading, offline, and disconnected states are covered.

**Release**

- Clean local install/build/test succeeds from documented steps.
- No secrets, session exports, datasets, checkpoints, GGUF files, or machine profiles are in the Git diff.
- GitHub CI passes on a clean checkout.
- Vercel build/deploy contains only the intended frontend and public assets.

### 15.3 Quality gates

Use explicit gates rather than “looks good”:

- **Gate A — Audit complete:** machine and repository facts recorded.
- **Gate B — Architecture approved internally:** documented decisions, no unresolved critical security design.
- **Gate C — Local vertical slice:** local UI ↔ local service ↔ model inference works.
- **Gate D — Data pipeline:** consent, import, redaction, review, versioning, and leakage-safe splits work.
- **Gate E — Training/benchmark:** at least one real supported smoke training path or truthful unsupported status; reproducible benchmark runner works.
- **Gate F — Upgrade loop:** candidate/regression/approval/rollback workflow works.
- **Gate G — Local release:** full applicable test suite passes and known limitations are documented.
- **Gate H — GitHub release:** clean publish, CI, docs, and license audit pass.
- **Gate I — Vercel:** frontend deployment verified, local pairing model secure, no local compute accidentally routed to serverless functions.

Do not proceed past a gate with a critical security/privacy failure or a false-success condition.

---

## 16. Phase 14 — local-first implementation plan and execution loops

### 16.1 Recommended order of implementation

Implement the smallest useful end-to-end flow before adding every advanced feature:

**Milestone 1 — audited foundation**

- Complete repository and machine audit.
- Document actual stack and actual hardware.
- Define safe storage roots and local service lifecycle.
- Add health endpoint, typed API, logging, and job state model.
- Add the first-run hardware/capability screen.

**Milestone 2 — local model runtime**

- Add model library with local GGUF import.
- Implement one reliable inference adapter (prefer llama.cpp if compatible).
- Start/stop model process and stream results.
- Add memory governor and CPU fallback.
- Record performance baseline.

**Milestone 3 — polished main UI**

- Build Overview, Machine, Models, Playground, Runs, and Settings.
- Add i18n de/en/fr, theme, responsive design, accessibility, and job tray.
- Remove all placeholder UI before proceeding.

**Milestone 4 — Data Studio**

- Build generic importer first.
- Add OpenCode export and Codex/Claude adapters according to current supported formats.
- Add Antigravity generic/manual import first; add a dedicated adapter only with a documented and testable interface.
- Add redaction, deduplication, normalization, review, schema, and split logic.

**Milestone 5 — synthetic teaching layer and Identity Maker**

- Add evidence-based decision summaries and labelled negative examples.
- Add synthetic provenance and independent validation.
- Build one-shot identity profile and dataset preview.
- Add user approval before any training/export.

**Milestone 6 — Training Lab**

- Implement environment preflight.
- Add a tiny SFT/LoRA smoke test for the actual supported backend.
- Add job lifecycle, checkpointing, monitoring, export, and inference reload.
- Add advanced methods only after a stable baseline.

**Milestone 7 — benchmark suite**

- Add local smoke tests and project-specific benchmarks.
- Add selected public benchmark adapters with licensing and sandboxing checks.
- Add model/agent comparison and reproducibility reports.

**Milestone 8 — Upgrade Lab**

- Add failure classification, candidate creation, dataset lineage, training proposals, regression gates, promotion, and rollback.
- Prove that holdout data remains untouched.

**Milestone 9 — agent workspace**

- Add task planner, visible tool permissions, approval queue, job timeline, and bounded repair loops.
- Make the suite operable through the GUI without granting unrestricted host access.

**Milestone 10 — local release hardening**

- Run all tests, install from a clean environment, test offline mode, test restart/recovery, review performance, scan secrets, and complete docs.
- Generate a release report and capture remaining limitations honestly.

### 16.2 Autonomous work cycle

For every milestone:

```text
INSPECT → PLAN → IMPLEMENT → TEST → REVIEW → FIX → RE-TEST → DOCUMENT
```

For every task:

1. Read the relevant source files and tests.
2. State the concrete expected behaviour.
3. Identify edge cases and failure modes.
4. Implement the smallest coherent change.
5. Add or update tests before considering it complete.
6. Run targeted tests.
7. Run impacted integration tests.
8. Inspect the diff and logs.
9. Update documentation and status.
10. Mark complete only with evidence.

### 16.3 Self-healing budget

- Retry deterministic test/repair loops a finite configured number of times.
- Use a small retry budget for flaky external tooling; do not rerun destructive operations automatically.
- Keep the failing test and original error visible.
- Do not mutate tests merely to make the run green.
- Do not install random dependencies or execute arbitrary web snippets to fix an error.
- If blocked by missing hardware, credentials, license, or platform support, document the exact blocker and continue independent tasks.

---

## 17. Phase 15 — GitHub preparation and push (only after local Gate G)

### 17.1 Repository hygiene

Before any push:

- Review `git status`, all staged files, full diff, and large-file inventory.
- Add/verify `.gitignore` for model files, adapters, checkpoints, datasets, raw sessions, logs, machine profiles, caches, virtual environments, `.env*`, API keys, and temporary artifacts.
- Add a `.env.example` containing names only and safe example values.
- Ensure no user conversation exports or personal information are included.
- Check every dependency's license and any benchmark/model redistribution conditions.
- Run secret scanning and dependency vulnerability checks.
- Confirm no third-party dataset/model assets were copied into the repository without rights.
- Add release notes and a clear limitations section.

Suggested ignored artefact patterns (adapt to real paths):

```gitignore
.env
.env.*
!.env.example
*.gguf
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.onnx
*.model
models/
model-cache/
datasets/
raw-sessions/
imports/
training-runs/
checkpoints/
artifacts/private/
logs/
*.log
MACHINE_PROFILE.json
__pycache__/
.venv/
venv/
node_modules/
dist/
build/
.cache/
```

Do not ignore source fixtures or small synthetic test data that are intentionally safe to distribute. Use explicit allowlists for test fixtures.

### 17.2 Documentation required before push

- `README.md` — clear product description, screenshots, architecture, install, quick start, hardware requirements, privacy, supported features, limitations.
- `docs/INSTALL_WINDOWS.md` — Windows local install and troubleshooting.
- `docs/INSTALL_LINUX.md` and `docs/INSTALL_MACOS.md` if those platforms are claimed.
- `docs/ARCHITECTURE.md` — actual architecture, not a wish list.
- `docs/MODELS_AND_RUNTIMES.md` — tested formats/backends and model licensing reminders.
- `docs/DATA_AND_PRIVACY.md` — import scope, rights, redaction, data storage, deletion, no hidden CoT reconstruction.
- `docs/TRAINING.md` — supported hardware/backend matrix and real training flow.
- `docs/BENCHMARKS.md` — exact suites, versions, scoring, and limitations.
- `docs/SECURITY.md` — local service pairing, sandbox, secrets, and disclosure policy.
- `CHANGELOG.md` and release notes.

### 17.3 Push gate

Only push when:

- Local Gate G is passed.
- The user has authorized publication / the repository destination is clear.
- Git remote and branch are verified.
- No private artefacts or secrets are staged.
- CI config and docs are complete.

Never force-push, rewrite shared history, or publish to a different repository than the one explicitly configured. If authentication is unavailable, prepare local commits and provide exact next steps without claiming the push occurred.

---

## 18. Phase 16 — Vercel deployment (last)

### 18.1 Deployment boundary

Vercel is the final stage and primarily hosts the **web frontend and static public assets**.

Do not deploy the user's local model runner, GPU workloads, model weights, private sessions, dataset store, training workers, or long-lived local job processes to Vercel Functions.

Reasons:

- Serverless functions have finite duration, memory, package-size, and runtime limits that vary by plan and evolve over time.
- Large model inference and training need persistent, machine-specific hardware and long-lived processes.
- Local privacy and hardware access are lost if user content is silently routed through a remote deployment.

Before deployment, re-check the current official Vercel limits and framework support; do not hard-code old plan limits into architecture.

### 18.2 Frontend-to-local-service connectivity

The deployed frontend may connect to a local Foundry companion only through an explicit pairing flow and exact origin allowlist. Test:

- Secure pairing from the deployed HTTPS origin.
- Origin validation and token expiry.
- Browser mixed-content restrictions.
- CORS/preflight handling.
- WebSocket/SSE reconnect and local service offline state.
- A clear fallback when a deployed browser cannot access the local service.

Never use a permissive `Access-Control-Allow-Origin: *` for privileged local endpoints.

### 18.3 Optional future hosted compute

If a future cloud training/inference product is desired, design it as a separate opt-in service with its own authentication, consent, storage, compute cost, privacy disclosures, quotas, and deletion controls. Do not smuggle it into the Vercel frontend deployment.

### 18.4 Vercel release gate

- GitHub main/release branch is stable and CI green.
- Frontend production build passes.
- Environment variables are configured only in the deployment platform and never committed.
- No local private data is part of the build artifact.
- Local-service pairing has been security-tested.
- Production routes, i18n, responsive layouts, offline/disconnected state, and error handling work.
- Rollback deployment is documented and tested.

---

## 19. Observability, reporting, and truthful product language

Every substantial screen should answer four questions:

1. What is happening?
2. Why is it happening?
3. What evidence supports the displayed status/result?
4. What can the user safely do next?

Use precise states:

- `Detected` is not the same as `tested`.
- `Compatible` is not the same as `verified on this machine`.
- `Downloaded` is not the same as `loaded`.
- `Loaded` is not the same as `inference tested`.
- `Training configured` is not the same as `training completed`.
- `Training completed` is not the same as `quality improved`.
- `Benchmark started` is not the same as `benchmark passed`.
- `Candidate improved one score` is not the same as `candidate is globally better`.
- `Local build succeeded` is not the same as `GitHub published` or `Vercel deployed`.

Reports must include actual run IDs and artifact hashes. If a metric is estimated, mark it as estimated. If a result is missing, say “not measured”.

---

## 20. Definition of Done

A feature is done only when all applicable conditions are true:

- Real functionality exists end to end.
- It has explicit input validation and useful error handling.
- It has loading, progress, empty, offline, cancellation, and failure states where applicable.
- It is covered by automated tests.
- It is accessible and responsive.
- It is translated into German, English, and French.
- It respects local-first privacy and permission boundaries.
- It records required provenance and run metadata.
- It does not make claims beyond what is verified.
- Its docs match the current implementation.
- Its regression tests pass.

A feature is **not** done merely because a UI card, settings field, endpoint stub, or mock result exists.

---

## 21. Final agent instruction

Start at **Phase 0**. Inspect the actual repository and machine before choosing the architecture. Write the audit and plan documents before implementation. Then build the smallest real local vertical slice and expand in the milestone order above.

Prioritize a stable local product over a broad but hollow feature list. Make all advanced optimizations measurable and reversible. Treat user sessions and datasets as private source material. Produce evidence for every claim. Use bounded autonomous loops, never infinite ones. Do not push to GitHub until the local release passes. Do not deploy to Vercel until the GitHub release is verified.

**HCS Foundry — Made with ❤️ by timfromhcs and hcsmedia.**

---

## 22. Reference links and technical reading

These links are starting points for the agent's research, not substitutes for re-checking current documentation during implementation. Prefer official project documentation, papers, source repositories, and current release notes. Record the date and exact version used in each implementation decision.

### Inference, GGUF, and accelerator support

- [llama.cpp — official repository and supported backends](https://github.com/ggml-org/llama.cpp)
- [llama.cpp — build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp — model formats and obtaining/quantizing models](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md)
- [llama.cpp — experimental training examples](https://github.com/ggml-org/llama.cpp/tree/master/examples/training)
- [llama.cpp — LoRA export tool](https://github.com/ggml-org/llama.cpp/tree/master/tools/export-lora)
- [Google Research — TurboQuant, QJL, and PolarQuant (March 24, 2026)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

### Training and datasets

- [Hugging Face TRL — SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Hugging Face TRL — PEFT integration](https://huggingface.co/docs/trl/peft_integration)
- [PEFT documentation](https://huggingface.co/docs/peft/index)
- [Unsloth — system and hardware requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)

### Browser inference

- [Transformers.js — WebGPU guide](https://huggingface.co/docs/transformers.js/guides/webgpu)
- [Transformers.js — documentation](https://huggingface.co/docs/transformers.js)
- [WebLLM — local browser inference](https://webllm.mlc.ai/)

### Session import references

- [Claude — export your data](https://support.claude.com/en/articles/9450526-export-your-claude-data)
- [Claude Code — sessions and local history](https://docs.claude.com/en/docs/claude-code/cli-reference)
- [OpenAI Codex — troubleshooting and session transcript locations](https://learn.chatgpt.com/docs/reference/troubleshooting)
- [OpenAI Codex — CLI reference](https://learn.chatgpt.com/docs/cli/reference)
- [OpenCode — CLI and session export](https://opencode.ai/docs/cli/)
- [Google Antigravity — official documentation](https://antigravity.google/docs/)

### Benchmarks

- [SWE-bench Verified](https://www.swebench.com/verified)
- [SWE-bench datasets and variants](https://www.swebench.com/SWE-bench/guides/datasets/)
- [LiveCodeBench — official repository](https://github.com/LiveCodeBench/LiveCodeBench)
- [Terminal-Bench 2.0](https://epoch.ai/benchmarks/terminal-bench)
- [OSWorld — current benchmark and OSWorld 2.0 information](https://os-world.github.io/)
- [WebArena — ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4410c0711e9154a7a2d26f9b3816d1ef-Abstract-Conference.html)
- [GAIA — benchmark paper overview](https://ai.meta.com/research/publications/gaia-a-benchmark-for-general-ai-assistants/)
- [MMMU benchmark](https://mmmu-benchmark.github.io/)

### Deployment

- [Vercel — Functions limitations](https://vercel.com/docs/functions/limitations)
- [Vercel — configuring function duration](https://vercel.com/docs/functions/configuring-functions/duration)
- [Vercel — platform limits](https://vercel.com/docs/limits)

**Implementation reminder:** before using a benchmark, model, dataset, or provider-session source, verify the exact current license, terms, schema, and version. A link in this file is not permission to redistribute or train on the linked content.
