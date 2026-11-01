# HCS Foundry v2

**Your local AI model lab. Built to learn, tested to improve.**

HCS Foundry is a local-first workbench for finding, running, evaluating and improving language models on your own hardware. It searches Hugging Face, pulls models with or without a token, runs GGUF inference, imports your own agent sessions into a reviewed dataset, fine-tunes a LoRA adapter, benchmarks the result, and only promotes an upgrade when deterministic gates pass.

Nothing leaves the machine by default. There is no telemetry and no cloud fallback.

<p align="center">
  <img src="docs/screenshots/hub.png" alt="Model Hub" width="880" />
</p>

## What is new in v2

| Feature | State |
|---|---|
| **Hugging Face Hub** — search, inspect files, pull with **and without** token, resume, hash verification | working, verified with a real 1.5 GB pull |
| **Model Hub UI** — search, task filters, file picker, download queue | working |
| **Standalone EXE** — one file, bundled API + SPA, opens the browser | built, 50.8 MB, health + UI verified |
| **CLI v2** — `hf`, `models run`, `bench`, `train sft`, `--json` everywhere | working |
| **Real LoRA/SFT backend** — transformers + PEFT, checkpoints, live loss, cancel | verified on Qwen3-0.6B |
| **Command palette** — Ctrl/Cmd+K | working |
| **Mobile bottom navigation** | working |
| **Opt-in cloud teacher** — Vercel AI Gateway, disabled by default | key verified, inference blocked by account billing |
| **Bug-fix pass** | 9 fixes, each with a test |

## Screenshots

| Hub | Command palette |
|---|---|
| ![Hub](docs/screenshots/hub.png) | ![Palette](docs/screenshots/palette.png) |

| Models | Data Studio |
|---|---|
| ![Models](docs/screenshots/models.png) | ![Data Studio](docs/screenshots/data.png) |

| Playground | Training Lab |
|---|---|
| ![Playground](docs/screenshots/playground.png) | ![Training Lab](docs/screenshots/training.png) |

## Quick start

### Option A — standalone executable (Windows)

```powershell
.\dist-exe\HCSFoundry.exe
```

Data lives in `%LOCALAPPDATA%\HCS Foundry`. The browser opens automatically. Fine-tuning is unavailable in the EXE (the ML stack is excluded); everything else works.

### Option B — from source

```powershell
cd server
python -m pip install -e .
python -m foundry.cli serve --port 8770

# second terminal
cd web
npm install
npm run build      # or: npm run dev
```

Open <http://127.0.0.1:8770>.

### Pull your first model

```bash
foundry hf search "qwen3 0.6b" --limit 5
foundry hf pull Qwen/Qwen3-0.6B --file model.safetensors --file tokenizer.json
foundry models run <model-id> --prompt "Say hello"
```

Public repositories need no token. Add one for gated or private models:

```bash
foundry hf token set --token hf_xxx
```

## Training, honestly

A real LoRA run is verified on `Qwen/Qwen3-0.6B`:

```
2 steps · loss 6.0436 → 5.0145 · 2.52M trainable params · adapter saved
```

**A good 3B model cannot be trained on this machine.** It has no CUDA/ROCm device and a
CPU-only torch build. The measured rate projects 15–38 hours for a 200-step 3B run at
512 tokens — before a useful number of epochs. The Training Lab ships the verified
trainer, a `3b-cpu` recipe that warns you, and a `3b-gpu` recipe for real hardware.
No "3B model" is claimed.

## Benchmarks and upgrade gates

- Instruction adherence, executed code generation, latency/throughput.
- An optional LLM judge is advisory only; deterministic checks decide pass/fail.
- The Upgrade Lab refuses promotion without an improvement threshold, no critical
  regression, an intact holdout hash and an explicit human approver.

Verified on this machine: a candidate was evaluated at 50 % vs an 83 % baseline and
**correctly rejected** with a critical secrecy regression.

## Requirements

- Windows 10/11, Linux or macOS, 64-bit.
- Python 3.11+ (tested 3.14.6) for the source install.
- Node.js 20+ (tested 22.23.1) for the frontend.
- A `llama-server` binary for GGUF inference; the app discovers one and never downloads it silently.
- 8 GiB RAM minimum, 16 GiB recommended.

Tested machine: Ryzen 7 7735HS, Radeon 680M (unified memory, **0 B dedicated VRAM**), 19.8 GiB visible RAM, Vulkan. Measured: 163 tok/s prompt, 46 tok/s generation on a 2B Q4_K_M model.

## Testing

```bash
cd server && python -m pytest -q      # 121 tests
cd web && npx vitest run              # 7 tests (incl. i18n parity)
cd web && npx playwright test         # 7 E2E tests (service must run)
```

GitHub Actions runs all three on a clean Ubuntu checkout.

## Vercel

The static frontend can be deployed; local inference never runs on Vercel.
See [docs/DEPLOY_VERCEL.md](docs/DEPLOY_VERCEL.md).

v2 also adds an **opt-in** Vercel AI Gateway teacher for annotations. The provided
credential was an AI Gateway key, not a platform token, so `vercel ai-gateway setup`
could not complete and gateway inference is blocked by missing account billing.
Details: [docs/evidence/vercel_setup.md](docs/evidence/vercel_setup.md).

## Documentation

- [Model Hub](docs/HUB.md) · [CLI](docs/CLI.md) · [Standalone EXE](docs/STANDALONE.md)
- [Architecture](docs/ARCHITECTURE.md) · [Security](docs/SECURITY.md)
- [Models & runtimes](docs/MODELS_AND_RUNTIMES.md) · [Data & privacy](docs/DATA_AND_PRIVACY.md)
- [Training](docs/TRAINING.md) · [Benchmarks](docs/BENCHMARKS.md)
- [v2 plan](docs/plan/V2_PLAN.md) · [Known limitations](docs/architecture/KNOWN_LIMITATIONS.md)
- [Release report v2](docs/evidence/RELEASE_REPORT_V2.md) · [Changelog](CHANGELOG.md)

## License

MIT — see [LICENSE](LICENSE). No model weights, datasets or benchmark assets are bundled.

---

Made with ❤️ by **timfromhcs** and **hcsmedia**.
