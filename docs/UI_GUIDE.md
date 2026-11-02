# UI guide — what each screen is for

Thirteen routes, one local service, no account. Everything below runs at
<http://127.0.0.1:8770> and is translated into **German, English and French** (the language
switch lives in the top bar and is remembered locally).

| Screen | Question it answers |
|---|---|
| **Overview** | What is this machine, what is loaded, what ran last, what should I do next? |
| **Machine** | Which CPU/GPU/RAM/disk do I really have, and what budget does the governor allow? |
| **Models** | Which models exist locally, what do they cost, which one fits *right now*? |
| **Hub** | What can I pull from Hugging Face, with which files, how big, with which hash? |
| **Playground** | Does the model answer, how fast, with which settings? |
| **Data Studio** | What data do I have, is it legal and clean, what is in it, how do I export it? |
| **Identity Maker** | One prompt → an editable persona → a small, labelled dataset. |
| **Training Lab** | Can this machine train it, what did the last run actually do? |
| **Benchmarks** | Which model/agent configuration wins on *my* tasks, with which evidence? |
| **Pipeline** | Run the whole chain (train → merge → GGUF → quantize) as one job. |
| **Upgrade Lab** | Which failures repeat, what is the candidate, may it be promoted? |
| **Agent Workspace** | What is the agent planning, which tools may it use, what did it do? |
| **Runs & Reports** | Every job with progress, events, logs and artifacts. |
| **Settings** | Language, theme, storage paths, privacy, pairing, runtime binaries. |

## The five screens you will actually live in

### Overview
Tier and budget at a glance, the loaded model with live tokens/s, a getting-started checklist
that links into each step (scan → model → load → data → train → playground), recent runs and any
warnings from the machine scan. The checklist is derived from real state, not from clicks.

### Models
The library with parameter count, quantization, KV bytes/token and a **fit range against live
RAM**. A model is only marked *verified* after it answered a prompt; until then the badge stays
neutral. The detail pane shows the exact llama-server command that would be used.

### Data Studio
Three tabs: **datasets**, **import**, **review**.
- Import: pick files or folders, choose the provider adapter (`auto` detects), confirm the rights
  statement, then import as a job with live progress.
- Dataset: record count, synthetic share, duplicate and redaction counts, split pills, a
  **distribution panel** (task type, language, split) and the manifest.
- Review: record by record, exclude with a reason, everything reversible before freezing.
- Export: **JSONL** (training), **JSON** (inspection), **CSV** (spreadsheets) and a 20-record
  sample — all with the same rights gate.

### Training Lab
Environment state (`unsupported` / `cpu-experimental` / `gpu`), preflight with a memory-aware
preset, recipes, a smoke run, and the run table. Each run shows the **load mode**
(`fp32` / `streaming` / `disk offload`), the **measured peak memory** and, for INT4 runs with a
disk store, a **Merge → GGUF** button that continues the chain as a job.

### Playground
The loaded model, a prompt box, sampling controls, streaming output with a cancel button, and
live metrics (TTFT, tokens/s, context used). If nothing is loaded it says so and links to Models.

Below that (with or without a loaded GGUF model) sits the **browser runtime** card: live WebGPU /
WASM / SIMD / threads / cores / memory chips, the computed safe limits with a reason for each
number, and — if an ONNX export exists under `data/browser-models/<id>/` — a load and generate path
that runs entirely in the tab, served by the local service rather than a CDN. A model above the
limit is refused with numbers. See [BROWSER_RUNTIME.md](BROWSER_RUNTIME.md).

## Keyboard and accessibility

- `Ctrl/⌘ + K` opens the command palette (routes, models, actions).
- A skip link is the first tab stop; every control is reachable by keyboard.
- Focus is always visible; status is never communicated by colour alone (badges carry text).
- `prefers-reduced-motion: reduce` disables transitions.
- Tables scroll horizontally instead of shrinking text; the nav collapses into a drawer below
  900 px.

## Honest states you will see

- "not measured" instead of a fake number,
- "no model loaded" instead of a disabled-looking control that does nothing,
- an explicit reason when a training configuration is refused,
- a red banner when the local service is unreachable (the UI never pretends to be offline-capable
  while showing stale data).

## Screenshots

`docs/screenshots/` holds the current set: `overview`, `models`, `hub`, `data`, `training`,
`pipeline`, `playground`, `palette`, `upgrades`, `settings`, `mobile`.
