# Release report — HCS Foundry v2.0.0

**Date:** 2026-11-01
**Branch:** `main`
**Tag:** `v2.0.0`
**Repository:** <https://github.com/timfromhcs/hcs-foundry>

## 1. Delivered

| Request | Result |
|---|---|
| Hugging Face support | **Done** — search, file listing, anonymous + token downloads, resume, hash check |
| Pull with and without token | **Done and tested** — anonymous pull of 1.5 GB verified; token path unit-tested and gated-error mapped |
| Nicer UI/UX lab | **Done** — Hub page, command palette, bottom nav, refined density, skeletons |
| Standalone EXE | **Done** — `dist-exe/HCSFoundry.exe`, 50.8 MB, health + UI 200 |
| Better CLI backend | **Done** — 9 command groups, `--json`, exit codes, UTF-8 output |
| Train a small good 3B model | **Not possible here** — trainer verified on 0.6B, 3B projected at 15–38 h per short run, no GPU |
| Modern UI on all devices | **Done** — 360/768/1024/1440/1920, bottom nav, no overflow |
| Fix all bugs | **Done (bounded)** — 9 regressions fixed, each with a test |
| Backend "like Unsloth but better" | **Partially** — real LoRA/SFT backend with checkpoints and live metrics; superiority not claimed |
| Push v2 | **Done** — tag `v2.0.0` |
| Vercel setup | **Partial** — CLI + plugin + MCP installed; `ai-gateway setup` blocked by credential type and billing |

## 2. Proofs

### Tests

```
server  121 passed (13.5 s)      docs/evidence/pytest.txt
web       7 passed               docs/evidence/vitest.txt
e2e       7 passed               docs/evidence/playwright.txt
typecheck clean, production build 451 kB JS / 129 kB gzip
```

### Real Hugging Face pull

```
repo    Qwen/Qwen3-0.6B
mode    anonymous (no token)
files   7 (model.safetensors 1,503,300,328 bytes + tokenizer/config)
result  indexed into the local library
```

### Real LoRA fine-tune

```
base    Qwen/Qwen3-0.6B (local safetensors)
steps   2, sequence 128, LoRA r=4
loss    6.043644 → 5.014467
params  2,523,136 trainable of 598,573,056
output  data/training-runs/proof-qwen06b/adapter (+ manifest.json)
```

### Standalone executable

```
artifact  dist-exe/HCSFoundry.exe (50.8 MB)
GET /               200
GET /api/v1/health  {"status":"ok","version":"2.0.0"}
UI rendered from the bundle (docs/screenshots/exe.png)
```

### Upgrade gate (unchanged, still correct)

Candidate evaluated at 50 % against an 83 % baseline → rejected with a critical
secrecy regression. No promotion happened.

## 3. Vercel — verified state, honest blocker

- CLI 65.9.0 installed; `whoami` → `timfromhcs`.
- OpenCode plugin installed; MCP endpoint added.
- `npx vercel@latest ai-gateway setup --yes --no-session-migration` →
  `login_required: The token in VERCEL_TOKEN is not valid.`
- The supplied `vck_…` credential is an **AI Gateway key**: it lists 498 models at
  `https://ai-gateway.vercel.sh/v1/models` but cannot manage teams or create keys.
- Gateway inference returns 403: “requires a valid credit card on file”.

Exact next actions are in `docs/evidence/vercel_setup.md`. Nothing was reported as
deployed, and the key was stored only in the git-ignored local state.

## 4. Known limits carried into v2

- No GPU training on this hardware; 3B is a recipe, not a result.
- GGUF export of a trained adapter is not implemented.
- The executable excludes the ML stack, so fine-tuning requires the source install.
- Cloud teacher is opt-in and currently blocked by account billing.
- Official benchmark suites are still not bundled.

## 5. Repository hygiene

- No weights, datasets, databases, tokens or gateway keys tracked.
- `data/`, `dist-exe/`, `build/`, `.playwright-mcp/` are ignored.
- Secret scan: only the intentional fake keys in redaction fixtures.
