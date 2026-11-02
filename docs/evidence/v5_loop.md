# v5 evidence — closing the loop (INT4 → merged fp16 → GGUF → runs)

Two dead ends from v4.1 are gone:

1. an INT4 adapter could never leave the machine as a runnable model,
2. the per-layer dequantization materialised a full fp32 matrix (94 MB for the 4B MLP).

## 1. Memory-bounded INT4 matmul (P1)

`Int4Linear` now dequantizes one row block at a time and writes into the output buffer
(`foundry/training/int4.py: blocked_linear` / `blocked_grad_input`, autograd function
`_Int4Matmul` recomputes per block in backward). Default block: 256 rows
(`FOUNDRY_INT4_BLOCK`).

Isolated measurement — one 9216×2560 projection (the largest shape in the 4B model),
`x` 128×2560, forward **and** backward in the same process, private bytes sampled every 10 ms:

| block | peak private | Δ vs baseline | duration |
|---|---|---|---|
| full matrix | 1048 MB | **+279 MB** | 0.16 s |
| 1024 | 984 MB | +28 MB | 0.13 s |
| 256 | 984 MB | **+0 MB** | 0.12 s |
| 64 | 965 MB | +4 MB | 0.22 s |

The transient is gone and it is not slower. Numerics are asserted equal by
`server/tests/test_int4.py::test_blocked_linear_matches_monolithic` and
`::test_blocked_gradients_match_monolithic` (max deviation ≤ 1e-4 on gradients).

An end-to-end 4B re-run with the blocked path was attempted twice during this session and
aborted: the host was under memory pressure from unrelated processes (free RAM fell to
1.4–3.5 GB, commit 87 %), the process spent minutes in page faults and no step completed.
No number is claimed for it. The last complete 4B run remains the v4.1 one
(`docs/evidence/v41_int4_4b.md`, loss 4.453 → 3.545, 51 s).

## 2. INT4 adapter → runnable model (P2)

`foundry/training/merge_int4.py` reconstructs what the forward pass actually used:

```
W_merged = dequantize_int4(packed, scales) + (lora_B @ lora_A) * (alpha / r)
```

Non-target tensors (including a vision tower or MTP head the store skipped) come from the
original checkpoint, so the result is a complete HF directory. Tensors are streamed into
≤ 1.5 GB fp16 shards with `model.safetensors.index.json`.

Refusals: store fingerprint/target set mismatch, adapter layer count mismatch, missing
`lora_int4.pt` (`server/tests/test_merge_int4.py`, 5 tests).

### Measured end-to-end chain (Qwen3-0.6B, this machine)

| Step | Command | Result |
|---|---|---|
| 1. INT4 training (disk store) | `train int4 --streaming --offload --steps 2 --seq 128` | 196 layers, loss **5.414693 → 4.308523**, peak RSS 2942 MB, store built on first use |
| 2. Merge | `train merge-int4 --adapter … --out …/merged` | 196 INT4 layers + LoRA → **1 shard, 1.41 GiB fp16, 29.8 s** |
| 3. GGUF export | `gguf export …/merged --out v5-int4-merged-f16.gguf` | **1439 MiB** F16 |
| 4. Quantize | `gguf quantize … --method Q4_K_M` | **462 MiB, ratio 0.321, 13.8 s** |
| 5. Load and answer | `models run mdl_9505130d70b956e278d0 --prompt "Say hello in German and stop."` | `text: "Hallo! Sollte dir helfen, gerne. 😊"` · 128 completion tokens · TTFT 0.998 s · **117.2 tok/s** |

Adapter artifact: `data/training-runs/v5-int4-loop/adapter/lora_int4.pt`
Merged directory: `data/training-runs/v5-int4-loop/merged` (config `dtype: float16`)
GGUF: `data/gguf/v5-int4-merged-f16.gguf`, `data/gguf/v5-int4-merged-Q4_K_M.gguf`

The model is not *better* (two optimisation steps on 53 records cannot do that), but the
artifact is real: it is a normal GGUF, loaded by a normal `llama-server`, answering prompts.

## 3. Paired web demo (P3)

`site/index.html` now carries a `#pair` section that:

- probes `GET /api/v1/health` on `http://127.0.0.1:8770` and reports the actual reason when
  it fails (nothing running, origin not allowlisted, or the browser blocking loopback),
- pairs via `POST /api/v1/system/pair`, keeping the token in the tab's memory only,
- explains the boundary: it can read status, it cannot train or read files without a token.

Server side:

- `FOUNDRY_UI_ORIGINS` is parsed as a URL list (it previously went through `Path`, which
  collapsed `https://`), and `https://hcs-foundry-site.vercel.app` is in the default allowlist.
- CORS echoes the exact origin and never `*`.
- **`OPTIONS` preflights now reach CORS**: previously the security middleware answered them
  with 401 for authenticated paths, so a cross-origin client could never call them.
- **Private Network Access** (`access-control-allow-private-network: true`) is returned when
  the browser asks, which is the protocol-level requirement for public → loopback.
- The pairing code is still enforced when enabled.

### Browser verification

Pairing from a page served on a loopback origin (`http://127.0.0.1:4180`) against the real
service, real Chromium:

```
status: "Paired. Token received for this tab only."
facts:  visible · version "5.0.0" · model "no model loaded"
console errors: none
```

### Measured browser limit (not a bug to hide)

From the **public** origin the same click is refused by Chromium before it reaches the service:

```
Access to fetch at 'http://127.0.0.1:8770/api/v1/health' from origin
'https://hcs-foundry-site.vercel.app' has been blocked by CORS policy:
Permission was denied for this request to access the `loopback` address space.
```

That is Chrome's Local Network Access permission, not the service. The page therefore says
exactly that and points to the local UI (`http://127.0.0.1:8770`). The PNA header removes the
protocol obstacle; the permission prompt remains a browser decision the page cannot make.

```text
server/tests/test_pairing_web.py   8 passed (origins, wildcard, PNA header, pairing code, version)
web/tests/e2e/pair.spec.ts         2 passed (offline state, no token in storage)
```

Live: <https://hcs-foundry-site.vercel.app/#pair>

## 4. Version single-sourcing (found while verifying)

`foundry/config.py` had `VERSION = "4.0.0"` next to `foundry.__version__`, so the API and the
UI reported 4.0.0 while the package said 4.1.0. `VERSION` is now derived from
`foundry.__version__`, and `test_pairing_web.py::test_version_is_single_sourced` fails if the
two ever drift again.
