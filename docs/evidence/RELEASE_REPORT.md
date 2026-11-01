# Release report — HCS Foundry 0.1.0

**Date:** 2026-11-01
**Commit:** `88f663e`
**Repository:** <https://github.com/timfromhcs/hcs-foundry> (private)
**Branch:** `main`

## 1. Gate status

| Gate | Requirement | Status | Evidence |
|---|---|---|---|
| A | Audit complete | **Passed** | `docs/audit/REPOSITORY_AUDIT.md`, `CAPABILITY_MATRIX.md` |
| B | Architecture documented | **Passed** | `docs/architecture/DECISIONS.md` (ADR-001…012) |
| C | Local vertical slice | **Passed** | UI → API → `llama-server` → streamed tokens (see §3) |
| D | Data pipeline | **Passed** | import → redact → dedupe → split → freeze → export (see §3) |
| E | Training/benchmark | **Passed (honest)** | real CPU smoke step; fine-tuning reported unsupported |
| F | Upgrade loop | **Passed** | candidate evaluated, **correctly rejected** by the gate |
| G | Local release | **Passed** | 98 + 7 + 6 tests green, clean build |
| H | GitHub release | **Passed** | pushed to `timfromhcs/hcs-foundry`, default branch `main` |
| I | Vercel | **Blocked** | no Vercel credentials or CLI in the environment |

## 2. Test results

```
server:   98 passed in 10.80s          (docs/evidence/pytest.txt)
web:       7 passed                     (docs/evidence/vitest.txt)
e2e:       6 passed (Playwright)        (docs/evidence/playwright.txt)
typecheck: clean (tsc --noEmit)
build:     418.53 kB JS / 121.45 kB gzip
```

Coverage highlights:

- **Security:** token required, foreign `Origin` rejected, foreign `Host` rejected, CSRF header required, JSON-only mutations, path traversal rejected, tokens hashed and revocable.
- **Data:** redaction of provider keys, emails, credentials, private keys and home paths; previews redacted; dedupe; deterministic group splits; freeze requires confirmed rights; export excludes excluded records.
- **Training:** smoke asserts a real parameter delta and loss improvement; preflight blocks a missing stack.
- **Benchmarks:** deterministic checkers; subprocess code execution with timeout; the judge cannot change a case status.
- **Upgrade:** holdout contamination refused; promotion blocked without an approver; critical regression blocks promotion; rollback restores the previous profile.
- **Agent:** no shell tool; `read_file` cannot escape the workspace; unknown tools fail loudly.
- **UI:** every route renders without console errors, keyboard focus order, no horizontal overflow at 390 px, reduced motion, language switch, theme persistence.

## 3. End-to-end verification performed in a real browser

| Flow | Result |
|---|---|
| Onboarding → pair → machine scan | Tier 1, 8c/16t, 19.8 GiB, **0 B dedicated VRAM**, 4.8 GiB safe budget |
| Model scan | 26 GGUF files discovered, metadata parsed, fit estimates computed |
| Load 1B Q8 model (Vulkan, `-ngl 99`) | ready in 1.47 s |
| Chat "Reply with exactly: OK" | `OK`, 62.4 tok/s, 0.8 s TTFT, 50 tokens |
| Import two session files | 2 records, `anthropic_key ×2` and `email ×2` redacted, train split |
| Review → exclude/include | works, redacted content shown |
| Freeze | manifest with dataset hash, honest `no_test_split` warning where applicable |
| Import 20 sessions | 20 records → 16 train / 2 validation / 2 test, holdout hash present |
| Benchmark instruction suite | 5/6 (83 %), failing `counting` case recorded |
| Training smoke | loss 2.1940 → 0.0303, parameter update verified |
| Identity Maker | prompt → card (facts vs proposals) → approve → 7 synthetic rows |
| Upgrade candidate | evaluated at 50 % vs 83 % baseline → **rejected** (−33 %, critical secrecy regression) |
| Agent task | machine scan + model scan executed, summary produced |
| Mobile 390×844 | drawer works, tables become labelled cards, no overflow |
| Light theme + German UI | verified, persisted |

## 4. Measured machine profile

- Windows 11 Pro 10.0.26200, AMD Ryzen 7 7735HS (8C/16T), 24 GB installed / 19.79 GB visible.
- AMD Radeon 680M, unified memory, **no dedicated VRAM**, driver 32.0.21043.12001, Vulkan 1.4.357.
- No NVIDIA device (a patched `nvidia-smi` exists but reports the AMD GPU; the probe correctly says "no NVIDIA device").
- `llama-server` build 11064 (Vulkan) verified.
- Prompt processing 163 tok/s, generation 46 tok/s on a 2B Q4_K_M model.

## 5. What was not done, and why

| Item | Reason |
|---|---|
| Vercel deployment | No `VERCEL_TOKEN`, no CLI session. The static build and `vercel.json` are ready; steps are in `docs/DEPLOY_VERCEL.md`. |
| Real fine-tune | `transformers`/`peft` are present as metadata but **fail to import** (huggingface-hub version mismatch). The app reports `cpu-experimental` and blocks the job. |
| GPU training | No CUDA/ROCm device. |
| TurboQuant | No discovered build exposes it; the capability probe reports *not available*. |
| Official benchmark suites | No assets/isolation/licenses shipped; listed as *not run*. |
| GGUF export | Not implemented in 0.1.0. |

## 6. Repository hygiene

- 120 tracked files, ~7 000 Python and ~8 000 TypeScript lines.
- No weights, datasets, databases, logs, pairing tokens or machine profiles tracked.
- `.gitignore` covers models, checkpoints, datasets, logs, caches, `.env*`, `MACHINE_PROFILE.json`, Playwright artifacts.
- Secret scan of staged files found only intentional fake keys inside redaction test fixtures.
- Largest tracked file is a 282 kB UI screenshot.

## 7. Local state left behind

The service is running on <http://127.0.0.1:8770>. The local (git-ignored) database contains the demo datasets created during verification: two imported session datasets, one identity dataset, four benchmark runs, one rejected upgrade candidate and one agent task. They are synthetic and can be deleted from the Data Studio UI.
