# HCS Foundry v7 — Long-term plan: faster 4B, a real local-first product surface

**Written before implementation.** Baseline: `cb5f039` (v6.0.0), 204 backend / 7 frontend /
9 Playwright tests green, CI green, site live at <https://hcs-foundry-site.vercel.app>.

The request covers six areas. They are ordered by risk to the product, not by how loud they are:
a training run that stalls is worth less than a missing logo, so the engine comes first, then the
automation surface, then UI, docs, branding and deployment.

---

## 1. Where the product actually hurts today

| Observation (measured) | Consequence |
|---|---|
| 4B step time jumps from ~25 s (warm page cache) to minutes when the host is under pressure | the first pass over the 4.57 GB store is the bottleneck, not the matmul |
| `blocked_linear` sizes blocks from a fixed env var | block size is not adapted to free RAM or to the tensor width |
| no way to drive the service from an agent or a script except HTTP | "commands and MCP" is missing |
| Training Lab shows mode/peak but no curve, Data Studio has no statistics | the user cannot see *why* a run behaves as it does |
| landing page documents features but has no tutorial, changelog page or docs index | the deployed site is a brochure, not a product surface |
| brand exists only as a "H" badge in HTML | no logo asset, no OG image, no favicon set |

## 2. Work packages

### P1 — Make 4B training faster and steadier (engine first)

**Status: implemented to the extent the measurements support.**

- [x] **Store prefetch** (`training/store_prefetch.py`): sequential 4 MiB reads warm the page
      cache in a daemon thread; measured **105 MB/s** on the external disk.
- [x] **Adaptive block size**: `choose_block_rows()` targets ~64 MB fp32 per block, clamped to
      256 rows after measurement (see below).
- [x] **Thread/report tuning**: block size, hidden size, available RAM and `torch` thread count
      are recorded in `manifest["tuning"]`.
- [x] **Measurement** (`scripts/store_io_bench.py`, `docs/evidence/v7_4b_speed.md`).
- [ ] **Sequential read path (pread)**: research says this is the right next step, but Windows has
      no `os.pread`; the honest gain over mmap page faults is only visible on a cold cache
      (2 660 MB/s sequential vs 67 MB/s faulted on `rest.f32`). Deferred with the numbers.
- [ ] **bf16 leftovers** (`rest.f32` 2.39 GiB → 1.20 GiB): would cut ~1.2 GB of I/O per pass, but
      needs the embedding kept in bf16 inside the model (a fp32 upcast at attach defeats it).
      Recorded as the next experiment, not shipped.

**Measurements that changed the plan**

| Fact | Number |
|---|---|
| sequential read of the store (prefetch) | **105 MB/s** |
| 4 KiB page-fault read of `rest.f32` (cold) | **66.6 MB/s** |
| same file, sequential 8 MiB reads (warm) | 2 659 MB/s |
| one 4B step at seq 64, store not fitting in RAM | **740 s** (2 steps at seq 128: 756 s) |
| prefetch of 4.36 GB before a step | 41.5 s — but the data is evicted again, so it is wasted |

Conclusion: on this host the 4B is **I/O-bound**, not compute-bound, because the store (4.57 GB)
plus the fp32 leftovers (2.39 GB) do not fit next to the process. Every optimisation that does
not reduce bytes or make them resident cannot fix that. What ships is the prefetch guard, the
honest report and the guidance below; the byte-reduction experiment is documented as the next
step.


### P2 — Run local with commands and MCP

- [ ] `foundry mcp serve` — Model Context Protocol server over stdio exposing typed tools:
      `machine_scan`, `models_list`, `models_run`, `datasets_list`, `datasets_export`,
      `train_int4`, `train_merge`, `gguf_quantize`, `vram_status`, `bench_run`.
      Every tool validates paths against the model/dataset roots, and destructive tools are
      absent by design.
- [ ] `foundry run <recipe>` shortcuts: `run doctor`, `run demo`, `run smoke` (doctor = scan +
      budget + runtime check in one screen; demo = 0.6B end-to-end if the model exists).
- [ ] `docs/MCP.md` with a copy-paste client config, plus tests for the tool schema and one
      end-to-end stdio handshake.
- [ ] EXE: `HCSFoundry.exe --mcp` starts stdio MCP instead of the web server (same binary).

**Acceptance:** an MCP client (tested in-process) lists the tools and calls `machine_scan` and
`models_list` successfully; `foundry run doctor` prints a single actionable report.

### P3 — UI/UX: see what is happening

- [ ] **Training Lab**: block size, VRAM budget and mode as real controls (persisted), a loss
      curve from the run's `eval_history`/loss list, and a per-run "memory" panel that explains
      RSS vs private vs file-backed.
- [ ] **Data Studio**: dataset statistics panel (records, languages, task types, split sizes,
      synthetic ratio) with bars, and export in **JSONL, JSON and CSV** plus a "download 20
      samples" action.
- [ ] **Onboarding/tutorial**: a first-run checklist (scan → model → dataset → smoke run →
      benchmark) that links into the pages, dismissible, state persisted.
- [ ] **Layout**: verify 360/768/1280/1440 for the changed pages; keep tables from overflowing.
- [ ] **i18n**: every new string in de/en/fr, parity test green.
- [ ] Playwright: assertions for the new panels (curve visible after a run, stats panel renders,
      export menu has three formats).

**Acceptance:** rendered evidence (Playwright + screenshots), no dead control, translations
complete.

### P4 — Branding and visual identity

- [ ] `site/assets/logo.svg`, `logo-mark.svg`, favicon set (16/32/apple-touch), OG image
      (`og.png`, 1200×630) generated from the brand mark, used by the site's meta tags and the
      app's header/favicon.
- [ ] A short `docs/BRAND.md` (palette, type, logo usage, do/don't) so the product does not drift.

### P5 — Tutorials and documentation

- [ ] `docs/TUTORIAL.md`: five copy-paste walkthroughs (first run, 0.6B/1.7B/4B training, merge
      to GGUF and serve, data import → export, benchmarks), each with expected output.
- [ ] `docs/MCP.md` (from P2), `docs/UI_GUIDE.md` (screen-by-screen tour with screenshots).
- [ ] README rewrite: what it is, honest capability table, install (EXE + source), first run,
      tutorial links, architecture, limitations, release link.

### P6 — Vercel app, properly

- [ ] Landing page: brand mark, a real "docs" section linking the tutorial/MCP/UI guide, a
      changelog page generated from `CHANGELOG.md`, and a `/security` page summarising the model.
- [ ] SEO: canonical URLs, OG/Twitter tags with the generated image, `robots.txt`, `sitemap.xml`,
      JSON-LD `SoftwareApplication`.
- [ ] Performance: no external fonts, no analytics, keep the page under 60 KB of HTML+CSS, images
      with explicit dimensions.

### P7 — Stable release

- [ ] All gates local (pytest, vitest, build, Playwright), CI green.
- [ ] Version 7.0.0 everywhere, README upgrade, CHANGELOG, release report, evidence files.
- [ ] **Local first**: everything above proven locally, then push to GitHub, then deploy Vercel.
- [ ] EXE rebuilt from the tagged commit, GitHub release with notes and asset.

## 3. Order and push loop

```text
P1 engine (+measure) → push
P2 MCP + run commands (+tests) → push
P3 UI/UX (+E2E) → push
P4/P5 brand + docs → push
P6 site → deploy
P7 release: tag, README, EXE, push, verify CI
```

Rule: no phase starts before the previous one is green locally. Every measurement goes into an
evidence file with the command that produced it. If a phase cannot be finished, it is marked
"partial" with the exact blocker — never silently dropped.

## 4. Explicitly out of scope for this horizon

- Multi-user/server mode, auth beyond the pairing token.
- Cloud training or inference.
- Official benchmark assets.
- Native mobile/desktop apps (the EXE is the desktop story).

## 5. Progress log

| WP | Status | Evidence |
|---|---|---|
| P1 4B speed | open | |
| P2 MCP + run | open | |
| P3 UI/UX | open | |
| P4 branding | open | |
| P5 docs | open | |
| P6 Vercel | open | |
| P7 release | open | |
