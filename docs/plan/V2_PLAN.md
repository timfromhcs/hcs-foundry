# HCS Foundry v2 — Plan

**Written before implementation.** Status markers are updated as work packages complete.
**Baseline:** v0.1.0, commit `449f338`, CI green.
**Target:** v2.0.0 — Hub integration, standalone EXE, real training backend, modern UI, CLI v2.

---

## 1. Requested outcomes → feasibility

| # | Request | Verdict | What will actually ship |
|---|---|---|---|
| 1 | Hugging Face support | **Feasible** | Search, inspect, download with **and** without token, resume, cancel, disk checks, gated-model errors |
| 2 | Nicer UI/UX lab | **Feasible** | Hub page, command palette, mobile bottom nav, refined tokens, skeletons, better density |
| 3 | Standalone EXE | **Feasible** | PyInstaller one-file build bundling API + SPA, launcher opens the browser |
| 4 | Better CLI backend | **Feasible** | `foundry hf/models/run/bench/train/serve` with `--json` and real exit codes |
| 5 | Train a small **good 3B** model | **Not feasible here** | Stack fixed and verified; real LoRA run proven on a tiny model; 3B recipe + measured time projection. No false "3B trained" claim |
| 6 | HF pull with and without token | **Feasible** | Anonymous works for public repos; token adds gated/private access; both tested |
| 7 | Modern UI on all devices | **Feasible** | 360 / 768 / 1024 / 1440 / 1920 breakpoints, touch targets, no overflow |
| 8 | Fix all bugs | **Feasible (bounded)** | Regression list in §4, each with a test |
| 9 | Backend "like Unsloth but better" | **Partly** | Real LoRA/SFT backend with preflight, checkpoints, resume, live metrics, cancellation. Superiority is **not** claimed; concrete local-first advantages are documented |
| 10 | Push v2 | **Feasible** | Branch `main`, tag `v2.0.0`, CI green |
| 11 | Vercel setup | **Feasible with the provided token** | CLI, plugin/MCP, `ai-gateway setup`, frontend deploy if the account allows it |

### Why a "good 3B model" cannot be produced on this machine

- GPU: Radeon 680M, **unified memory, no CUDA/ROCm**. `torch` is `2.12.1+cpu`.
- A 3B LoRA step at sequence length 256 on this CPU is projected in the tens of seconds.
  A useful SFT run needs thousands of steps → days to weeks of continuous compute.
- The honest deliverable is: a **verified trainer**, a **measured projection**, and a
  ready-to-run 3B recipe for a GPU machine. Claiming otherwise would violate the
  project's own rules.

---

## 2. Architecture changes

```text
server/foundry/
  hf.py                  NEW  Hub client: search, tree, download, token store
  hub_api.py             NEW  (merged into api/models.py) Hub endpoints
  training/
    sft.py               NEW  real LoRA/SFT trainer (transformers + peft)
    backends.py          NEW  backend registry + capability probes
    recipes.py           NEW  3B/tiny presets with measured estimates
  launcher.py            NEW  EXE entry point (bundled SPA, opens browser)
  cli.py                 REWRITE  subcommands + --json
web/src/
  pages/Hub.tsx          NEW
  components/CommandPalette.tsx NEW
  components/BottomNav.tsx      NEW
  styles/app.css         REFINE tokens, density, motion
scripts/build_exe.ps1    NEW
```

---

## 3. Work packages and acceptance criteria

### W1 — Hugging Face client (`foundry/hf.py`)

- [ ] `search(query, task, limit, token)` against `https://huggingface.co/api/models`
- [ ] `tree(repo_id, token)` with file sizes and LFS sha256 where present
- [ ] `download(repo_id, filename, dest, token, progress, cancel)`:
      streaming, `Range` resume, `.part` file, atomic rename, sha256 verify when known
- [ ] Anonymous request sends **no** Authorization header
- [ ] Token stored in `data/state/hf-token.json`, never returned in full (masked)
- [ ] 401/403 → explicit "gated or private; add a token" error
- [ ] Disk-space precheck and path allowlist

**Proof:** unit tests for URL building, header presence/absence, resume, error mapping;
one real anonymous download of a small public file.

### W2 — Hub UI

- [ ] Hub page: search, task filter, sort, result cards, file list, download queue
- [ ] Token panel (set/clear, masked, "test connection")
- [ ] Download progress in the job tray, cancel works
- [ ] "Pull from Hub" entry point from Models
- [ ] Empty/error/offline states

**Proof:** Playwright route test + screenshot.

### W3 — CLI v2

- [ ] `foundry serve|scan|info|token` (existing, cleaned)
- [ ] `foundry hf search <q> [--task] [--limit] [--json]`
- [ ] `foundry hf files <repo> [--json]`
- [ ] `foundry hf pull <repo> [--file GLOB] [--token] [--dest]`
- [ ] `foundry models list|scan|verify|load|unload|run`
- [ ] `foundry bench <suite> [--json]`
- [ ] `foundry train smoke|preflight|sft`
- [ ] `--json` everywhere; non-zero exit on failure
- [ ] `foundry --help` grouped and readable

**Proof:** CLI smoke script captured in `docs/evidence/cli_v2.txt`.

### W4 — Training backend v2

- [ ] Fix the transformers/peft import failure (pin `huggingface-hub<2`)
- [ ] `sft.py`: dataset → chat format → tokenize → LoRA → Trainer with callbacks
- [ ] Checkpoints, resume, cancellation, live loss events
- [ ] Adapter export + reload for inference (documented limits)
- [ ] `preflight` returns measured memory/time estimates
- [ ] Real run on a tiny model, loss curve recorded
- [ ] 3B recipe with honest projection

**Proof:** `docs/evidence/training_v2.md` with loss values and adapter path.

### W5 — Standalone EXE

- [ ] `scripts/build_exe.ps1` (PyInstaller onefile, bundled `web/dist`)
- [ ] `foundry/launcher.py`: finds bundle, starts uvicorn, opens browser, clean shutdown
- [ ] Build artifact runs, health check passes, UI loads from the EXE
- [ ] Document size and startup time

**Proof:** `docs/evidence/exe_build.txt` + screenshot.

### W6 — UI/UX modernization

- [ ] Refined design tokens (spacing, elevation, radii, focus)
- [ ] Command palette (Ctrl/Cmd+K)
- [ ] Mobile bottom navigation
- [ ] Skeletons for lists, better empty states
- [ ] Density and hierarchy pass on all 12 routes
- [ ] No horizontal overflow at 360/768/1024/1440/1920
- [ ] Contrast and focus visible in both themes

**Proof:** Playwright viewport matrix + screenshots.

### W7 — Bug fixes (regression list)

- [ ] Settings changes survive a restart (DB overrides loaded at startup)
- [ ] Model roots from Settings are used by scans without restart
- [ ] Duplicate toast on job completion (single notification)
- [ ] `runtime_profiles` upsert (no 500 on reload of same config)
- [ ] Benchmark `case_id` normalisation (already fixed, add regression test)
- [ ] SimHash signed-64 storage (already fixed, add regression test)
- [ ] Download filename parsing
- [ ] Job output on cancellation
- [ ] Empty-state links never dead
- [ ] Every fix has a test

**Proof:** pytest + Playwright.

### W8 — Vercel

- [ ] Fetch and follow `https://vercel.com/get-started.md`
- [ ] Install CLI, verify `whoami` with the provided token
- [ ] Install guidance (OpenCode plugin or skills)
- [ ] Connect MCP
- [ ] `npx vercel@latest ai-gateway setup`
- [ ] Deploy the static frontend if the account permits
- [ ] Never commit the token; recommend rotation

**Proof:** `docs/evidence/vercel_setup.md` with verified state only.

### W9 — Release v2

- [ ] Full test matrix green locally and in CI
- [ ] README v2 with screenshots and honest limits
- [ ] CHANGELOG v2
- [ ] Tag `v2.0.0`, push `main`

---

## 4. Loop protocol

For every work package:

```text
inspect → implement smallest coherent change → unit test → integration test
→ browser proof (if UI) → fix → re-test → commit → update this file
```

Bounded self-healing: at most 3 repair attempts per failing test, then document the
blocker and continue with independent work.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| HF rate limits during tests | small fixtures, cached downloads, no bulk pulls in CI |
| Token leakage | token only in `data/state` (git-ignored), masked in API, never logged |
| PyInstaller missing hidden imports (uvicorn, pydantic) | explicit `hiddenimports` + build test |
| CPU training too slow | hard `max_steps` cap, tiny model proof, 3B only as recipe |
| Vercel interactive prompts | `VERCEL_TOKEN` env, non-interactive flags, pause only if OAuth is unavoidable |
| Scope creep | must/should split; "can't" items stay documented, not faked |

---

## 6. Definition of done for v2

- All W1–W9 acceptance boxes checked or explicitly marked blocked with a reason.
- `pytest`, `vitest`, `playwright` green.
- EXE built and executed.
- CI green on a clean checkout.
- README and CHANGELOG updated.
- v2.0.0 pushed.

---

## 7. Progress log

| WP | Status | Evidence |
|---|---|---|
| W1 HF client | **done** | `server/foundry/hf.py`, `tests/test_hf.py` (12 tests), live search + 1.5 GB anonymous pull |
| W2 Hub UI | **done** | `web/src/pages/Hub.tsx`, route test, screenshot `hub.png` |
| W3 CLI v2 | **done** | `foundry/cli.py`, `docs/CLI.md`, live `hf search/files`, `models list`, `train smoke` |
| W4 Training v2 | **done, 3B honestly blocked** | `training/sft.py`, real Qwen3-0.6B run, `docs/evidence/v2_training.md` |
| W5 EXE | **done** | `dist-exe/HCSFoundry.exe` 50.8 MB, health + UI 200, `docs/evidence/v2_exe.txt` |
| W6 UI/UX | **done** | command palette, bottom nav, hub grid, skeletons, 7 Playwright tests |
| W7 Bugs | **done** | settings persistence, runtime upsert, case_id, simhash, toasts — each with a test |
| W8 Vercel | **partial** | CLI + plugin + MCP installed; `ai-gateway setup` blocked by token type/billing, see `docs/evidence/vercel_setup.md` |
| W9 Release | **in progress** | tests green, docs updated, tag + push pending |

## 8. Bug fixes shipped in v2

| Bug | Fix | Test |
|---|---|---|
| Settings were lost after restart | `apply_overrides` loads persisted settings at startup | `test_settings_survive_a_restart` |
| Model roots ignored by scans after edit | settings object mutated through one path | same test + scan uses `state.settings` |
| `runtime_profiles` 500 on reload | upsert on conflict | `test_api` load flow |
| Duplicate completion toasts | effects depend on stable `reload` callbacks | manual + E2E |
| Benchmark cases lacked `case_id` | normalised before persistence | `test_run_suite_scores_deterministically` |
| SimHash overflowed SQLite INTEGER | signed-64 conversion + masked Hamming | `test_pipeline` import |
| HF download resumed into duplicate bytes when Range was ignored | restart on `200` | `test_download_restarts_when_server_ignores_range` |
| Windows console crashed on unicode CLI output | force UTF-8 stdout/stderr | `foundry hf search` run |
| `transformers`/`peft` unimportable | pin `huggingface-hub<2` | `test_sft` real run |
