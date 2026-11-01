# HCS Foundry v3 — Plan

**Written before implementation.** Baseline: v2.0.0 (`68ad296`), CI green.
**Target:** v3.0.0 — GGUF out, quantization, dequantization, MoEfication, training algorithms, auto-decision, OpenEval, self-improving playbook, Vulkan tuning, real log import, all-in-one pipeline.

---

## 1. Requested outcomes → feasibility

| # | Request | Verdict | What ships |
|---|---|---|---|
| 1 | GGUF export | **Feasible** | Merge adapter → HF dir → `convert_hf_to_gguf.py` → F16 GGUF, verified by loading |
| 2 | Quantize | **Feasible** | `llama-quantize` wrapper with a format matrix, size/sha output, load verification |
| 3 | Dequantize GGUF | **Partial** | GGUF→F16/BF16 GGUF via `llama-quantize` (true HF dequant needs `convert_gguf_to_hf.py`, not present) |
| 4 | MoEfication | **Feasible at small scale** | Real dense→MoE conversion (expert cloning + top-k router), saved and forward-tested; 4B MoE recipe only |
| 5 | Vulkan GPU acceleration | **Partial** | Inference: auto-tune `-ngl`/batch with a measured sweep. Training: **no torch Vulkan backend** — documented, not faked |
| 6 | Multiple training algorithms | **Feasible** | SFT, DPO (synthetic pairs), distillation, INT4/QLoRA; each tested on tiny models |
| 7 | Auto-decision during training | **Feasible** | Controller adapts LR/batch/rank, stops on plateau, restores best checkpoint |
| 8 | OpenEval during fine-tuning | **Feasible** | Held-out eval every N steps, best-checkpoint tracking, early stop |
| 9 | "Learns to train better" (perfect runs) | **Feasible (bounded)** | Playbook records winning recipes per model/dataset signature and biases the next run; not a neural meta-learner |
| 10 | Clean up all OpenCode/Claude/Antigravity CLI logs | **Feasible** | Import + redact + dedupe + quarantine copy. **Never delete source data** |
| 11 | Complete training run on Qwen 3.8, 4–8B | **Not feasible here** | 4B fp32 weights alone need ~16 GB; only ~5.6 GB free. Time would be days. 0.6B full run + 4B recipe |
| 12 | Quantize + MoE that model | **Partial** | 4B GGUF quantize/requantize feasible; 4B MoE is 4× FFN → not feasible on this RAM |
| 13 | Adapt UI/UX, CLI, web demo, downloads | **Feasible** | New pages/commands for export, quantize, MoE, training modes, playbook |
| 14 | GGUF dequantize and continue | **Partial** | Requantize to F16 and continue training from the original HF weights; no HF-format GGUF import tool found |
| 15 | All-in-one | **Feasible** | `foundry pipeline run` with ordered stages, checkpoints and resume |

### Why the 4B run is refused, with numbers

- `empero-ai/Qwen3.8-4B-Distill`: `model.safetensors` = **8.68 GB** (BF16/F16).
- The trainer loads `float32` → **~17.4 GB** of weights, plus optimizer and activations.
- Visible RAM: **19.8 GB**, currently **5.6 GB** free. OS + browser reserve ≥ 3 GB.
- Even F16 loading (~8.7 GB) leaves ~2 GB for activations/optimizer on a shared-memory iGPU.
- Throughput extrapolates to **days** for a useful number of steps.

The pipeline will refuse with these exact numbers and print the recipe for a GPU machine.

### Why Vulkan cannot accelerate training here

`torch` is `2.12.1+cpu`; there is no Vulkan backend in PyTorch. `llama.cpp` has experimental
`ggml` training but not a supported fine-tuning path. Vulkan is therefore used for
**inference and quantization verification**, with an auto-tuned layer/batch sweep. Training
stays CPU and says so.

---

## 2. New modules

```text
server/foundry/
  gguf_tools.py        discover converter/quantizer, export, quantize, requantize, verify
  moe.py               dense→MoE conversion, router init, save, forward check, recipe
  training/
    algorithms.py      SFT, DPO, distillation entry points
    controller.py      auto-decision: LR/batch/rank/stop, best-checkpoint
    openeval.py        held-out evaluation callback + early stop
    playbook.py        recipe memory: record wins, bias next run
  pipeline.py          all-in-one orchestration with resumable stages
  data/opencode_cli.py importer via `opencode session export`
```

---

## 3. Work packages

### W1 — GGUF export / quantize / dequantize

- [ ] Discover `convert_hf_to_gguf.py`, `llama-quantize`, `llama-gguf-split`
- [ ] `export_gguf(hf_dir, out, dtype=f16)` with progress, log capture, timeout
- [ ] `quantize(src_gguf, out_gguf, method)` for Q2_K…Q8_0, F16, BF16
- [ ] `requantize(src, out, F16)` documented as the dequantize path
- [ ] Verify output by loading it in `llama-server` and asking one question
- [ ] Refuse when free disk < output size × 1.2

**Proof:** Qwen3-0.6B → F16 GGUF → Q4_K_M → loads and answers; sizes recorded.

### W2 — MoEfication

- [ ] `moeify(model_dir, out_dir, experts, top_k)`:
      clone each FFN into N experts, initialise from the dense weights with small noise,
      add a linear router, set `top_k`, save a manifest
- [ ] Forward-pass check (finite logits, router sums to 1)
- [ ] Parameter/memory report and a hard refusal when the projected size exceeds RAM/disk
- [ ] Recipe for 4B/8B on GPU

**Proof:** tiny GPT-2 → 4 experts, forward pass finite, file sizes recorded.

### W3 — Training algorithms + auto-decision + OpenEval + playbook

- [ ] `algorithms.py`: `sft`, `dpo`, `distill`
      - DPO uses synthetic preference pairs (chosen = reference answer, rejected = corrupted)
      - distill trains a student on teacher completions (local model or gateway)
- [ ] `controller.py`: warmup, LR reduction on plateau, batch/rank downshift on OOM, stop after patience
- [ ] `openeval.py`: eval loss on a held-out split every N steps, best-checkpoint tracking, early stop
- [ ] `playbook.py`: JSON store keyed by model+dataset signature; records winning configs;
      the next auto run starts from the best-known config
- [ ] Every mode refuses without confirmed rights and a train split

**Proof:** tiny model runs for SFT, DPO and distill with recorded losses; controller changes
LR at least once; playbook gains an entry.

### W4 — Vulkan auto-tuning

- [ ] `ngl` sweep (0, 8, 16, 24, 99) + batch variants against a GGUF model
- [ ] Record TTFT, tokens/s, peak RAM; pick the best stable config
- [ ] Store the winning profile per model
- [ ] Explicit note that this tunes inference, not training

**Proof:** sweep output on Qwen3-0.6B Q4_K_M with at least two configurations measured.

### W5 — Real log import

- [ ] Claude Code: `~/.claude/projects/**/*.jsonl` (27 files, 47 MB)
- [ ] Codex: `~/.codex/sessions/**/*.jsonl` (6 files, 0.7 MB)
- [ ] Antigravity: `~/.gemini/antigravity/brain/**/transcript.jsonl`
- [ ] OpenCode: `opencode session list` + `session export` (documented CLI path)
- [ ] Inventory → preview (redacted) → import → dedupe → split → freeze
- [ ] Quarantine copy of redacted originals; **never delete or modify sources**
- [ ] Report per source: files, records, redactions, skipped lines

**Proof:** import summary with real counts and a frozen dataset hash.

### W6 — CLI / API / UI

- [ ] CLI: `gguf export|quantize|dequantize|inspect`, `moe build`, `train auto|dpo|distill`,
      `tune vulkan`, `pipeline run`, `logs import`
- [ ] API: `/gguf/*`, `/moe/*`, `/training/auto`, `/training/playbook`, `/tune/vulkan`, `/pipeline/*`
- [ ] UI: Models → Export/Quantize/MoE actions; Training → algorithm selector + auto toggle +
      eval chart + playbook; new Pipeline page; Import → "scan known log locations"
- [ ] All new strings in de/en/fr

**Proof:** Playwright route/interaction tests, screenshots.

### W7 — All-in-one pipeline

- [ ] Stages: `pull → dataset → train → export → quantize → moe? → bench → report`
- [ ] Each stage writes a checkpoint; `--resume` continues
- [ ] Refuses stages that cannot run and explains why
- [ ] Emits a JSON + Markdown report

**Proof:** run on 0.6B through export+quantize+bench; 4B stage refused with numbers.

### W8 — Release v3

- [ ] Tests green locally and in CI
- [ ] README/CHANGELOG/docs updated
- [ ] Tag `v3.0.0`, push, CI green

---

## 4. Loop protocol

```text
inspect → implement → unit test → integration test → real proof → commit → update this file
```

Bounded repair: 3 attempts per failing test, then document and move on.

## 5. Definition of done

- Every box checked or explicitly marked blocked with a reason.
- `pytest`, `vitest`, `playwright` green.
- At least one real GGUF export+quantize and one real MoE forward pass.
- A real 0.6B training run with auto-decision and OpenEval.
- Real log import with redaction counts.
- v3.0.0 pushed and CI green.

## 6. Progress log

| WP | Status | Evidence |
|---|---|---|
| W1 GGUF | **done** | Qwen3-0.6B → F16 GGUF (1.41 GiB, 209 s) → Q4_K_M (462 MiB, 17 s) → answers `EXPORT_OK` at 113 tok/s |
| W2 MoE | **done** | `moe.py`, 4 tests, real conversion with finite forward pass |
| W3 Training | **done** | SFT/DPO/distill/INT4 + OpenEval + auto controller + playbook; real 0.6B auto run, eval 4.344 → 4.233; INT4 run 5.72 → 4.86 |
| W4 Vulkan | **done** | sweep on 0.6B Q4: ngl=99 → 96.5 tok/s, 1.79× vs CPU |
| W5 Logs | **done** | 72 records from Claude/Codex/Antigravity; OpenCode active session skipped; sources untouched |
| W6 UI/CLI | **done** | Pipeline page + 14 CLI command groups + v3 API; E2E 7/7 |
| W7 Pipeline | **done** | stage runner with report; failure isolation tested |
| W8 Release | **done** | 149 + 7 + 7 tests green, tag `v3.0.0`, CI run 41347915292 green |

## 7. Honest refusals shipped in v3

| Request | Refusal |
|---|---|
| Complete 4B/8B fine-tune | 4B fp32 weights ≈ 17 GB vs ~5.6 GB free; `estimate_resources` refuses with numbers |
| Vulkan training | no PyTorch Vulkan backend; Vulkan is used for inference tuning only |
| GGUF → HF safetensors | no `convert_gguf_to_hf.py` in the discovered llama.cpp tree; only GGUF→F16 requantization is offered |
| 4B MoE with 4 experts | projected FFN blow-up exceeds RAM; the CLI refuses unless `--force` |
