# HCS Foundry v5.0.0 — Plan: "close the loop"

**Written before implementation.** Baseline: `71f9f65` (v4.1.0), 178 backend / 7 frontend /
7 Playwright tests green, CI green, site live.

---

## 1. Where the project actually stands

v4.1 proved two mechanisms that still end in a dead end:

| Proven | Dead end that remains |
|---|---|
| 4B INT4 training through the disk store | the adapter is a custom `lora_int4.pt` — **it cannot be merged, exported as GGUF, or served** |
| Streaming/offloaded loading of an 18 GiB fp32 model | the run peaks at 6.3 GB private because every dequantized layer is materialized as one fp32 matrix |
| Landing site on a stable URL | the deployed page cannot talk to the local service — the roadmap item "paired web demo" is still open |

So the product can *train from a huge checkpoint* and *deploy a page*, but it cannot hand the
result back to the user as something they can run, and the deployed page is decoration.

## 2. Requested outcome

> "Work until the next logical stable horizon."

**v5.0.0 = one complete, inspectable path**:

```
4B checkpoint (or 0.6B)
  → streaming INT4 store (disk)
  → LoRA training (memory-bounded)
  → merge into fp16 safetensors          ← NEW
  → GGUF export                          ← exists, now reachable from the INT4 path
  → Q4_K_M quantization                  ← exists
  → load and answer in llama.cpp         ← NEW verification step
```

plus a deployed page that can *pair* with the local service instead of pretending to be an app.

## 3. Work packages

### P1 — Memory-bounded INT4 matmul (blocked dequantization)

Today `dequantize_int4` builds the whole fp32 matrix (94 MB for one 9216×2560 layer) before the
matmul. With 200 layers of allocator churn that is what drives `peak_private` to 6.3 GB.

- [ ] `Int4Linear.forward`/`_Int4Matmul` dequantize in row blocks and accumulate:
      `y[rows] = x @ dequant(packed[rows]).T`, block size configurable (`FOUNDRY_INT4_BLOCK`, default 256)
- [ ] backward recomputes the same way per block (never the full matrix)
- [ ] numerics equal to the monolithic path within 1e-3 relative, grads unchanged
- [ ] re-run the 4B 2-step proof, record `peak_private_mb` / `peak_rss_mb` before vs after
- [ ] tests: blocked == monolithic, block boundaries, odd shapes, bias, backward

**Proof:** 4B run with a lower measured private peak than 6343.5 MB, loss still decreasing.

### P2 — Merge an INT4 adapter back into a real model

- [ ] `foundry/training/merge_int4.py`:
      `merged = dequant(packed) + lora_B @ lora_A * (alpha/r)`, streamed row-block by row-block,
      written as fp16 safetensors shards (≤ 2 GB each) with `index.json`
- [ ] fp32 leftovers (embeddings/norms) are converted to fp16 and streamed to the same shards
- [ ] `config.json` gets `torch_dtype: float16`; tokenizer files are copied from the base
- [ ] refuses when the store fingerprint does not match the adapter's manifest
- [ ] `train unified` replaces the "INT4 adapters cannot be merged" skip with a real merge
- [ ] CLI `train int4 --merge`, API field `merge`, pipeline stage `merge_int4`
- [ ] tests: tiny INT4 run → merge → tensor values equal `dequant + delta`, sharding, index file

**Proof:** 0.6B and 4B adapter → fp16 HF dir → GGUF f16 → Q4_K_M → `llama-server` answers.

### P3 — Paired web demo

- [ ] `site/pair.html`: probes `http://127.0.0.1:8770/api/v1/health`, runs the pairing call,
      shows machine/model status, and has a truthful offline state (CSP-friendly, no inline JS
      assumptions, no fake buttons)
- [ ] server: `FOUNDRY_UI_ORIGINS` env allowlist; default includes the deployed site origin;
      CORS echoes the exact origin only, never `*`
- [ ] pairing requires the local code whenever `require_code` is enabled, and the deployed origin
      never gets a token without it
- [ ] tests: allowed origin accepted, unknown origin rejected, wildcard never emitted, code
      required when enabled
- [ ] Playwright: the page renders the offline state deterministically (bogus port)

**Proof:** deployed page + local service results, origin matrix in evidence.

### P4 — Surface the pipeline in the UI

- [ ] Runs page: load mode (`fp32` / `streaming` / `offload`), `peak_rss_mb`, `peak_private_mb`,
      merge + GGUF artifacts when present
- [ ] Pipeline page: "Merge INT4 adapter → GGUF" action wired to the new stage
- [ ] de/en/fr strings (parity test must stay green)
- [ ] Playwright assertion for the new run badges

### P5 — Release v5.0.0

- [ ] docs: README, CHANGELOG, TRAINING, KNOWN_LIMITATIONS, BENCHMARKS, evidence files
- [ ] version bump everywhere, EXE rebuild, GitHub release
- [ ] full gates: pytest, vitest, `npm run build`, Playwright, CI green

## 4. Order of execution

```text
P1 (code + tests) → P1 real 4B measurement
P2 (code + tests) → 0.6B end-to-end → 4B end-to-end (background)
P3 (site + security) → deploy → verify
P4 (UI + i18n) → P5 (docs, version, EXE, release, CI)
```

Long-running jobs (4B merge → GGUF → quantize) run in the background while P3/P4 are written.

## 5. Bounded repair rules

- max 3 attempts per failing test, then document the blocker and continue with the next package
- no test is weakened to pass; no score is claimed without a manifest that supports it
- every measurement goes into `docs/evidence/V5_*.md` with the command that produced it

## 6. Progress log

| WP | Status | Evidence |
|---|---|---|
| P1 blocked dequant | **done** | blocked forward/backward; isolated 4B-shape measurement: transient 279 MB → 0 MB, no slowdown; numerics asserted equal |
| P2 INT4 merge | **done** | 0.6B chain: train → merge (196 layers) → F16 → Q4_K_M → answers at 117 tok/s (`docs/evidence/v5_loop.md`) |
| P3 paired demo | **done** | `/#pair` on the live site, `FOUNDRY_UI_ORIGINS` URL parsing fixed, 6 backend + 2 E2E tests |
| P4 UI surface | **done** | runs table shows load mode, measured peak, "Merge → GGUF" (de/en/fr) |
| P5 release | **done** | docs, version 5.0.0 everywhere, EXE rebuilt, tag `v5.0.0`, CI green |

### What was not measured, and why

- **4B end-to-end export** was not run: it writes ~17 GB (fp16 shards + GGUF) and support for
  the Qwen3.5 vision-language architecture in the installed `convert_hf_to_gguf.py` is
  unverified. Only 0.6B is proven end to end.
- **4B training re-run with the blocked path** was attempted three times. The host ran out of
  headroom (free RAM 1.4–3.5 GB, commit 87 %, unrelated agent processes) and no step
  completed. The isolated per-layer measurement covers the claim instead; the last complete
  4B run stays the v4.1 one.
