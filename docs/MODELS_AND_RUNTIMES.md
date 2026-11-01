# Models and runtimes

## Supported format

**GGUF** is the only inference format in 0.1.0. The library reads the header (metadata + tensor info) and computes:

- architecture, name, license
- quantization (from `general.file_type`, falling back to the filename)
- parameter count (sum of tensor element counts)
- claimed context length, block count, head counts, head dimension
- KV-cache bytes per token (derived from layers × kv-heads × head-dim × 2 bytes)

Tensor data is never read during a scan. Hashing (`Verify hash`) reads the file in 4 MiB chunks and can be cancelled.

## Runtime discovery

The service looks for `llama-server` in:

1. `FOUNDRY_LLAMA_SERVER` (explicit path)
2. paths registered in Settings
3. `<data>/runtime`
4. `~/.foundry/runtime`, `~/llama.cpp`, `C:/llama.cpp`
5. generic per-drive hints (`<drive>/AI`, `<drive>/llama.cpp`, `<drive>/LLM`)

Discovery is a bounded walk with a time budget. Each candidate is probed with `--version`; the result is `verified` or `unverified`. The backend name is inferred from the path (`vulkan`, `cuda`, `rocm`, `metal`, `cpu`) and cross-checked against detected accelerators when choosing a default.

**No runtime is downloaded automatically.**

## Launch configuration

The supervisor starts the child with a random `--api-key`, an ephemeral loopback port, `--no-webui`, `--metrics`, and the requested:

| Flag | Meaning |
|---|---|
| `-c` | context length |
| `-ngl` | GPU layers |
| `-t` | CPU threads (0 = auto) |
| `-b` / `-ub` | batch / micro-batch |
| `-fa` | flash attention (`auto`/`on`/`off`) |
| `-ctk` / `-ctv` | KV-cache type (`f16`, `q8_0`, `q4_0` when supported) |
| `--mlock` | optional lock |

Unsupported flags cause the child to exit; the service surfaces the child's log tail instead of pretending the model is ready.

## Weight quantization vs KV-cache quantization

These are different things and the UI keeps them separate:

- **Weight quantization** (Q4_K_M, Q5_K_M, Q6_K, Q8_0, …) is a property of the GGUF file.
- **KV-cache quantization** (`-ctk`/`-ctv`) reduces memory used by attention state as context grows. It can affect long-context quality.
- **Context length** is a model/runtime capability, not a free slider. The UI caps the slider at the model's claimed maximum and the governor still refuses configurations that do not fit.

## TurboQuant

Google Research published TurboQuant/QJL/PolarQuant in March 2026. **No discovered `llama-server` build on this machine exposes a TurboQuant KV-cache mode.** HCS Foundry therefore reports it as *not available* and does not advertise it. If a build adds support, it must be implemented behind a versioned adapter and compared against FP16/Q8/Q4 KV modes before being offered.

## Licensing

The library shows `general.license` from the GGUF metadata when present, otherwise "unknown". HCS Foundry never assumes a license. You are responsible for verifying the license of every model you load and of any dataset you train on.

## Practical envelope on the tested machine

AMD Radeon 680M, 19.8 GiB visible RAM, Vulkan:

| Model size | Quantization | Result |
|---|---|---|
| 1–4 B | Q4/Q5/Q8 | comfortable |
| 7–9 B | Q4/Q5 | tight; reduce context to ≤ 4096 |
| 12 B | Q4 | risky; close other apps |
| 27 B+ | any | not recommended |

Measured on `Qwen3-VL-2B-Instruct-Q4_K_M` at 2048 context with `-ngl 99`: 163 tok/s prompt, 46 tok/s generation.
