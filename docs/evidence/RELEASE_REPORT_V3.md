# Release report — HCS Foundry v3.0.0

**Date:** 2026-11-01
**Branch:** `main`
**Tag:** `v3.0.0`
**Repository:** <https://github.com/timfromhcs/hcs-foundry>

## 1. Requested → delivered

| Request | Result |
|---|---|
| Plan first, then execute autonomously | **Done** — `docs/plan/V3_PLAN.md` written before implementation |
| GGUF out | **Done** — 0.6B → F16 GGUF (1.41 GiB, 209 s) |
| Quantize | **Done** — Q4_K_M (462 MiB, ratio 0.321, 17 s) |
| Dequantize GGUF | **Partial** — GGUF→F16 upcast; no HF converter available |
| MoEfication | **Done at small scale** — real conversion + forward check + tests |
| Vulkan GPU acceleration | **Partial** — inference auto-tune 2.1×; no Vulkan training backend |
| Multiple training algorithms | **Done** — SFT, DPO, distillation, INT4/QLoRA |
| Auto-decision during training | **Done** — LR reduction, early stop, best checkpoint |
| OpenEval / "perfect runs" | **Done** — held-out eval + playbook that seeds the next run |
| Clean up OpenCode/Claude/Antigravity logs | **Done** — 72 records imported, originals untouched |
| Complete 4–8B training run | **Refused with numbers** — 4B fp32 load ≈ 17.4 GB vs ~5.6 GB free |
| Quantize + MoE that 4B model | **Partial** — 4B Q4_K_M runs on Vulkan; 4B MoE refused (memory) |
| Adapt UI/UX, CLI, web, downloads | **Done** — Pipeline page, 14 CLI groups, v3 API |
| All-in-one | **Done** — resumable pipeline with report |

## 2. Proofs

### Tests

```
server  149 passed (23 s)     docs/evidence/pytest.txt
web       7 passed            docs/evidence/vitest.txt
e2e       7 passed            docs/evidence/playwright.txt
build    468 kB JS / 133 kB gzip
```

### GGUF pipeline (real)

```
export   1,509,347,136 bytes in 209.4 s
quantize 484,219,712 bytes, ratio 0.321, 17.0 s
verify   llama-server answered "EXPORT_OK", 113.5 tok/s, TTFT 0.79 s
```

### Vulkan sweep (real)

```
ngl=0   51.38 tok/s (CPU)
ngl=99 107.80 tok/s (Vulkan)
speedup 2.1x        docs/evidence/vulkan_sweep.json
```

### Auto training on real imported logs

```
dataset   ds_d642c29dc9bf42a984ce — 72 records (49 train / 19 val / 4 test)
sources   Claude Code 6 files, Codex 6 zstd files, Antigravity 2 transcripts
eval      4.344190 → 4.261183 → 4.233414
best      saved at step 6; playbook entry recorded
```

### INT4 training (real)

```
quantized layers  196
LoRA parameters   2,523,136
loss              5.721064 → 4.859477
eval              4.2595 → 4.1792
peak RSS          4,862 MB (fp32 load peak)
adapter           data/training-runs/v3-int4/adapter/lora_int4.pt
```

### 4B model

```
download  empero-ai/Qwen3.8-4B-Distill-GGUF Q4_K_M, 2,783,446,304 bytes, anonymous
inference loads and answers on Vulkan at context 512
governor  refused context 1024 first (correct); --force allowed the smaller run
```

## 3. Honest refusals

| Request | Reason |
|---|---|
| Complete 4B/8B fine-tune | fp32 load peak ≈ 17.4 GB; ~5.6 GB free; days of compute |
| Vulkan training | PyTorch has no Vulkan backend |
| GGUF → HF safetensors | no `convert_gguf_to_hf.py` in the discovered llama.cpp tree |
| 4B MoE (4 experts) | projected FFN growth exceeds RAM |

## 4. Repository hygiene

- 152 tracked source/doc files plus tests; no weights, datasets, databases, tokens or
  gateway keys tracked.
- `data/`, `dist-exe/`, `build/`, `.playwright-mcp/` ignored.
- The user's session logs were read only; Codex archives were decompressed into the
  git-ignored import area.

## 5. Next steps (documented, not implemented)

1. Streaming INT4 loader (quantize while reading shards) to make 4B INT4 training fit.
2. GGUF → HF conversion (upstream tool or a minimal reader) to continue training from
   an existing GGUF.
3. A real Vulkan/ROCm training backend once PyTorch or another framework supports it.
4. 4B/8B recipes on a GPU machine, using the verified pipeline unchanged.
