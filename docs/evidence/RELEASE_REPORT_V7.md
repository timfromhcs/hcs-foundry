# Release report — HCS Foundry v7.0.0

**Date:** 2026-11-02
**Branch:** `main`
**Tag:** `v7.0.0`
**Baseline:** `cb5f039` (v6.0.0)
**Plan:** `docs/plan/V7_PLAN.md` (written before implementation, updated with the research)

## Theme

Four asks: why is 4B training slow, run it locally with commands **and MCP**, a UI that shows what
happens, a site and docs that behave like a product surface — plus branding.

## Delivered

| # | Ask | Result |
|---|---|---|
| 1 | Make 4B training better/faster | research (`docs/evidence/v7_4b_speed.md`), prefetch **guard**, adaptive block, per-run tuning in the manifest |
| 2 | Local commands + MCP | `foundry mcp serve` (9 typed tools), `foundry run doctor`, `HCSFoundry.exe --mcp/--doctor`, `docs/MCP.md` |
| 3 | Better UI/UX | Data Studio distribution panel + JSONL/JSON/CSV/sample export, tutorial checklist with real state, i18n de/en/fr |
| 4 | Branding, docs, site | logo/wordmark/favicons/OG generated from one source, `docs/TUTORIAL.md`, `docs/UI_GUIDE.md`, `docs/BRAND.md`, docs section + generated changelog + SEO on the site |

## The research, in one table

| Measurement | Value |
|---|---|
| sequential read of the store | **105 MB/s** |
| 4 KiB page-fault read, cold | **66.6 MB/s** |
| the same bytes once resident | 2 660 MB/s |
| one 4B step, working set fits (~11 GB free) | ~25 s |
| one 4B step, working set does not fit | **378 s** (2 steps at seq 128: 756 s) |

Conclusion: on this host the 4B is **I/O-bound**. The release therefore ships a prefetch *guard*
(skip warming when the store cannot stay resident), a tuning report per run, and a doctor hint —
and claims **no speedup**.

## Proofs

```
foundry run doctor
  machine   AMD Ryzen 7 7735HS · 19.8 GiB RAM · AMD Radeon(TM) Graphics
  budget    2.7 GiB safe · 35 models (0 verified)
  runtime   llama-server: yes
  storage   5 INT4 store(s), 11.48 GiB · vram arena 64 MiB
  data      5 datasets (4 frozen)

foundry mcp serve            handshake + machine_scan + models_list verified in-process
                             (tests/test_mcp.py, 8 tests, incl. path rejection and stdio framing)

tests        217 backend · 7 frontend · 9 Playwright E2E
site         logo + docs section + /changelog.html + og.png + robots/sitemap + JSON-LD
```

## Honest gaps

- **4B speed**: guard only; the disk is the limit. Next experiments are written down (batched
  per-layer reads on a cold cache, bf16 leftovers) but not shipped.
- **4B training was not re-run today**: two attempts stalled in page faults with ~4 GB free and no
  step completed, so no new number is claimed; the last complete 4B run remains the v4.1 one.
- The tiered VRAM cache stays **off by default** (v6 measurement).
- Quality is still not claimed: every run is a short smoke run.

## Publication

| Item | Value |
|---|---|
| Tag | `v7.0.0` → commit `b2a4c11` |
| CI | run `41430719802` — green |
| GitHub Release | <https://github.com/timfromhcs/hcs-foundry/releases/tag/v7.0.0> (Latest) |
| Windows EXE | 384,124,715 bytes, built from the tagged commit |
| Landing page | <https://hcs-foundry-site.vercel.app> · <https://hcs-foundry-site.vercel.app/changelog.html> |
