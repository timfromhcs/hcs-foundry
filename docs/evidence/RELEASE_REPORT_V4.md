# Release report — HCS Foundry v4.0.0

**Date:** 2026-11-01
**Branch:** `main`
**Tag:** `v4.0.0`

## Delivered

| Request | Result |
|---|---|
| Plan first | `docs/plan/V4_PLAN.md` written before implementation |
| Full train local | Unified trainer decides SFT vs streaming INT4 and runs train → merge → export → quantize; 0.6B verified, 4B pending download |
| Full auto chat-log cleanup | 79 imported, 75 included, 4 excluded, frozen, quarantine copy, originals untouched |
| Unified auto trainer | `training/unified.py` + `decide()` |
| More Vulkan / efficiency | tuned profile reused in benchmark metadata, `tune apply`, profile API |
| Coding benchmarks | repo-fix suite with per-case venv |
| Agentic benchmarks | JSON file tasks in a sandbox, one executed in a venv |
| SWE-bench with venv | adapter + per-instance venv + command allowlist; official assets not bundled (`not_run`) |
| v4 stable | tag + CI |

## Proofs

```
tests        165 backend, 7 frontend, 7 Playwright
cleanup      79 imported / 75 included / 4 excluded / frozen hash 4c28a267… / quarantine written
streaming    Qwen3-0.6B: 196 layers, loss 5.72 → 4.89, 8.35 s
benchmarks   repo_fix 3 cases, agentic 3 cases, swebench instance passes in venv
```

## Honest gaps

- **4B streaming INT4 run** did not complete: the 9.32 GB safetensors download was still
  in progress at release time. The loader is implemented and proven on 0.6B.
- **Official SWE-bench** needs the dataset and repository checkouts; without them the
  suite reports `not_run` with the reason.
- **Vulkan training** remains impossible (no PyTorch Vulkan backend).

## Vercel

The new AI Gateway key (`vck_…BxfT`) is stored only in the git-ignored local state,
masked in the API and left disabled. It validates for model listing. The earlier key was
replaced; both were pasted into chat and should be rotated.
