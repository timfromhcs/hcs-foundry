# HCS Foundry v4 — Plan

**Written before implementation.** Baseline: v3.0.0 (`a64669a`), CI green.
**Target:** v4.0.0 — full local training, unified auto trainer, full chat-log cleanup, coding/agentic benchmarks with virtual environments, more Vulkan efficiency.

---

## 1. Requested outcomes → feasibility

| # | Request | Verdict | What ships |
|---|---|---|---|
| 1 | "Full train local geht" | **Feasible with a streaming INT4 loader** | Load shards → quantize while reading → 4B INT4 LoRA training without the fp32 peak |
| 2 | Full auto cleanup of chat logs | **Feasible** | One command: discover → import → redact → dedupe → quality filter → freeze → report (+ quarantine copy) |
| 3 | New addons / unified auto trainer | **Feasible** | One entry point that picks SFT/INT4/DPO/distill from model size, free RAM and playbook, then runs train → merge → export → quantize → bench |
| 4 | More Vulkan, more efficient | **Feasible (inference only)** | Auto-apply tuned profiles, sweep inside benchmarks, per-model profile reuse, batching/thread tuning |
| 5 | Coding benchmarks | **Feasible** | Executed codegen suite + a repo-fix suite that applies model patches and runs tests in a venv |
| 6 | Agentic benchmarks | **Feasible** | Multi-step tool tasks in a sandboxed workspace with deterministic oracles |
| 7 | SWE-bench with virtual env | **Partial** | SWE-bench-format adapter + venv isolation + local fixture instances; official dataset/repos are not bundled and are reported as *not run* |
| 8 | Everything better | **Feasible (bounded)** | Efficiency pass, UI/CLI surface, docs, tests |
| 9 | v4 stable | **Feasible** | Tag `v4.0.0`, CI green |

### The key unlock: streaming INT4 loading

Today INT4 training still materialises the fp32 model first (≈17.4 GB for 4B). v4 adds a
loader that:

1. builds the model on the `meta` device (no allocation),
2. streams `model.safetensors` shard by shard,
3. quantizes each target linear weight to INT4 **immediately** and drops the fp32 tensor,
4. assigns the remaining tensors directly,
5. refuses if any parameter is still on `meta` after loading.

Peak memory becomes roughly the INT4 weights (~2 GB for 4B) plus one layer's fp32
temporary, instead of the full fp32 model.

### Honest limits that remain

- **Vulkan training** is still impossible: PyTorch has no Vulkan backend. Vulkan stays an
  inference/tuning path.
- **Official SWE-bench** needs the dataset and repository checkouts. The adapter accepts
  the official JSONL shape and isolates each instance in a venv, but the assets are not
  bundled; without them the suite reports *not run*.
- **A "good" 4B model** still needs far more compute than this machine has. v4 proves the
  pipeline runs end to end at 4B scale for a few steps; it does not claim quality.

---

## 2. Work packages

### W1 — Streaming INT4 loader

- [ ] `load_int4_streaming(model_dir, targets, r, alpha)` using `init_empty_weights`,
      `safetensors.safe_open` and `set_module_tensor_to_device`
- [ ] Bias handling independent of key order, tied-weight handling
- [ ] Post-load check: no parameter left on `meta`
- [ ] `train_int4(..., streaming=True)` uses it
- [ ] Memory measurement (peak RSS) and a manifest field
- [ ] Test on a tiny model and on Qwen3-0.6B
- [ ] Real 4B INT4 run (download 9.32 GB safetensors, 1–2 steps)

**Proof:** peak RSS for 0.6B and 4B; loss values; adapter artifacts.

### W2 — Unified auto trainer

- [ ] `foundry/training/unified.py`: `decide()` returns algorithm + config from
      model size, free RAM, dataset size and playbook
- [ ] Runs: train → best checkpoint → merge adapter → export GGUF → quantize → benchmark
- [ ] Writes one report with every stage and its numbers
- [ ] Refuses with numbers when nothing fits
- [ ] CLI `train unified`, API `/training/unified`, UI button

**Proof:** full local run on 0.6B producing a quantized GGUF and a benchmark score.

### W3 — Full auto chat-log cleanup

- [ ] `foundry/data/cleanup.py`: discover → import all → redact → dedupe →
      quality filter (empty, too short, tool-only) → split → freeze → report
- [ ] Quarantine copy of the redacted import, never touching originals
- [ ] Report: per source files/records/redactions, duplicates, excluded, splits, hash
- [ ] CLI `logs clean`, API `/sources/clean`, UI one-click
- [ ] Test with fixture sources and with the real machine sources

**Proof:** real run over Claude/Codex/Antigravity/OpenCode with counts and dataset hash.

### W4 — Coding & agentic benchmarks with venv isolation

- [ ] `foundry/benchmarks/venv.py`: create a venv, optionally install requirements,
      run a command with timeout and captured output
- [ ] `foundry/benchmarks/repo_fix.py`: local fixture repo with failing tests; the model
      must produce a patch, the runner applies it and executes the suite in a venv
- [ ] `foundry/benchmarks/agentic.py`: multi-step tasks (create, run, fix, verify) with
      deterministic checks and a tool sandbox
- [ ] `foundry/benchmarks/swebench.py`: official JSONL shape adapter; per-instance venv;
      `not_run` with a reason when repo/assets are missing
- [ ] Suite registry entries + UI + CLI
- [ ] Tests: patch applied/failed, venv isolation, timeout, missing assets

**Proof:** real fixture run with a passing and a failing instance; SWE-bench adapter
returns `not_run` honestly.

### W5 — Vulkan efficiency

- [ ] Auto-apply the stored Vulkan profile when loading a model for benchmarks/playground
- [ ] Include the profile (ngl, batch, tok/s) in benchmark metadata
- [ ] `bench` can sweep once and reuse the best profile
- [ ] Efficiency knobs: thread count, batch, context; measured comparison
- [ ] Report "speedup vs CPU" per model

**Proof:** benchmark run that records the tuned profile and a measured speedup.

### W6 — UI / CLI / API

- [ ] CLI: `train unified`, `logs clean`, `bench repo_fix|agentic|swebench`, `tune apply`
- [ ] API: `/training/unified`, `/sources/clean`, `/benchmarks/suites` additions
- [ ] UI: Pipeline page gains "Full local train", "Clean logs", benchmark selector
- [ ] de/en/fr strings

**Proof:** Playwright route/interaction tests.

### W7 — Release

- [ ] All tests green locally and in CI
- [ ] README v4, CHANGELOG v4, evidence files
- [ ] Tag `v4.0.0`, push

---

## 3. Loop protocol

```text
inspect → implement → unit test → integration test → real proof → commit → update this file
```

Bounded repair: 3 attempts per failing test, then document and continue.

## 4. Progress log

| WP | Status | Evidence |
|---|---|---|
| W1 Streaming INT4 | **done, 4B run pending** | `load_int4_streaming` + tests; 0.6B streaming run loss 5.72 → 4.89, 196 layers. 4B download not finished in this session |
| W2 Unified trainer | **done** | `training/unified.py` + `decide()` tests; train → merge → export → quantize chain |
| W3 Log cleanup | **done** | real run: 79 imported, 75 included, 4 excluded, frozen, quarantine copy, originals untouched |
| W4 Benchmarks | **done** | repo_fix + agentic + SWE-bench adapter with per-instance venv; 10 tests |
| W5 Vulkan | **done** | stored profile surfaced in benchmark metadata; `tune apply` CLI; profile API |
| W6 UI/CLI | **done** | Pipeline: Clean logs + Full local train; CLI `logs clean`, `train unified`, `tune apply`, new bench suites |
| W7 Release | **in progress** | 165 backend tests green, docs, push pending |
