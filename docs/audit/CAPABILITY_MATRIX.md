# Capability Matrix — this machine

**Measured:** 2026-11-01 on `WIN-UT58NNIMA2L`
**Classification key:** **Verified** = directly observed/tested · **Likely** = strongly supported, not exercised · **Unknown** = not detected · **Unsupported** = known not to work here.

## 1. Hardware

| Capability | Value | Class |
|---|---|---|
| OS | Windows 11 Pro 10.0.26200, x64 | Verified |
| CPU | AMD Ryzen 7 7735HS, 8 physical / 16 logical cores, 3.2 GHz base | Verified |
| Installed RAM | 24 GB (4 × 6 GB LPDDR5-6400) | Verified |
| Visible RAM to OS | 19.79 GB (≈ 4.2 GB reserved as UMA frame buffer) | Verified |
| Available RAM at audit | 7.6 GB | Verified |
| GPU | AMD Radeon 680M (Rembrandt), `PCI\VEN_1002&DEV_1681` | Verified |
| **Dedicated VRAM** | **0 bytes** — integrated GPU, unified memory architecture | Verified |
| Shared GPU memory reported by driver | 4,293,918,720 bytes (BIOS reservation, *not* free app memory) | Verified |
| GPU driver | 32.0.21043.12001 (2026-05-29) | Verified |
| Vulkan | Instance 1.4.357, AMD driver present | Verified |
| CUDA | Not present. `nvidia-smi` on PATH is a ZLUDA-patched binary that reports the AMD GPU; there is no NVIDIA device | Verified |
| ROCm / HIP | Not installed (`rocminfo` missing) | Unsupported here |
| DirectML | Not probed by a tested backend | Unknown |
| NPU | None exposed as a compute accelerator | Unsupported here |
| Disk free | C: 68.3 GB · D: 95.6 GB · E: 238.8 GB | Verified |

### 1.1 UMA budget rule

The governor never treats the 4.29 GB reported `AdapterRAM` as free application memory. It uses:

```
safe_budget = min(
    available_system_ram - os_reserve,
    uma_cap - display_and_os_reserve
)
```

with `os_reserve = max(2 GiB, 15 % of total RAM)` and `uma_cap = reported_shared_gpu_memory`.

## 2. Runtimes and toolchains

| Capability | Value | Class |
|---|---|---|
| Python | 3.14.6 (default), plus 3.13 / 3.12 / 3.11 / 3.10 | Verified |
| Node.js / npm | 22.23.1 / 10.9.8 | Verified |
| Git | 2.54.0 | Verified |
| Docker | 29.6.1 (Docker Desktop) | Verified |
| CMake | present (Python-script shim) | Verified |
| MSVC `cl.exe` | not on PATH | Unknown |
| Rust / Cargo | present | Verified |
| ffmpeg | 9.0 | Verified |
| `uv` | present | Verified |
| `torch` | 2.12.1 **+cpu** | Verified |
| `transformers` / `trl` / `peft` | not installed | Verified |
| `numpy` / `datasets` / `safetensors` / `gguf` | 2.4.6 / 5.0.0 / 0.8.0 / present | Verified |

## 3. Measured inference performance

Test: `Qwen3-VL-2B-Instruct-Q4_K_M.gguf`, `llama-server` Vulkan build 11064, `-ngl 99`, `-c 2048`, 13-token prompt, 2-token reply.

| Metric | Value | Class |
|---|---|---|
| Health endpoint | `{"status":"ok"}` | Verified |
| Cold start to health | < 15 s | Verified |
| Prompt processing | 163 tok/s | Verified |
| Generation | 45.9 tok/s | Verified |
| Time to first token (short prompt) | ≈ 0.1–0.3 s | Verified |
| GPU offload (`-ngl 99`) | works on Vulkan | Verified |

## 4. Derived tier

**Tier 1 — integrated GPU / shared memory**, with a *measured* working Vulkan inference path.

Practical envelope on this machine:

- 1B–4B models at Q4/Q5: comfortable.
- 7B–9B at Q4/Q5: workable with reduced context (≤ 4096) and `-ngl 99`, but memory-tight alongside a browser.
- 12B+ at Q4: only with low context and no other heavy process; expected to be slow and risky.
- 27B+ : not recommended; would exceed the safe budget.
- Training: CPU-only, tiny LoRA smoke experiments at best. No GPU training path on this machine.

## 5. Browser-side capabilities

WebGPU / WASM SIMD / SharedArrayBuffer depend on the user's browser and are **not** asserted here. The frontend performs a runtime feature detection and reports the result in the Machine screen; unknown stays unknown.
