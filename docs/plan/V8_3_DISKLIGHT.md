# Plan v8.3 — disk-light operation, opt-in update check, accessibility pass

**Written:** 2026-11-02, after `v7.3.0`
**Sources:** research below was read on 2026-11-02; links are starting points, not claims.
**Rule:** the same one as `V8_STABLE_PLAN.md` — no claim without a gate, bounded self-healing loop.

---

## 0. What the research says (and what it means here)

| Finding | Source | What we do |
|---|---|---|
| Checkpointing is a dominant write load; keeping **two** snapshots and cleaning the rest is the standard fix, and optimizer state roughly doubles every write | CheckFreq (USENIX FAST'21), "Efficient Checkpointing System" (SC-W 2024) | keep `save_total_limit=2` (already), add `save_only_model` since no resume path exists, enforce a minimum save interval |
| 4 KiB page-aligned writes avoid SSD write amplification; delta snapshots cut snapshot bytes ~31 % | moonclip (docs.rs), "Preserving SSD lifetime with delta snapshots" (JPDC) | alignment is the filesystem's job here; we stop *writing* what we do not need instead of implementing deltas |
| Auto-update: never download without explicit consent, do not nag on first launch, ID-less checks, respect a disable flag | AppImage golden rules, SurfSense ADR 0070, memo privacy doc | opt-in, **check only** (no download), no identifier, off by default, visible toggle + "check now" |
| WCAG 2.2: focus visible ≥ 3:1, status messages via live regions, target ≥ 24×24, focus not obscured, reduced motion | W3C WCAG 2.2, Accessible Data Interfaces, A11yPath checklist | announce load/generate status, `aria-busy`, `role="region"` on the panel, keep the existing 2 px focus ring |

## 1. Current state (verified in this repository)

| Item | Value | Where |
|---|---|---|
| `save_steps` | `max(1, config.save_every)`, default `save_every=10` | `training/sft.py:451` |
| `save_total_limit` | 2 | `training/sft.py:452` |
| optimizer state | written on every checkpoint (no `save_only_model`) | `training/sft.py:444` |
| INT4 store reuse | reported (`"reused"`), so a run does not always rewrite 4.6 GB | `training/int4.py:767` |
| update check | does not exist | — |
| a11y | global `:focus-visible`, skip link, reduced motion exist; the browser panel announces nothing while loading | `styles/app.css:101`, `BrowserRuntimePanel.tsx` |

## 2. Phases

**Progress (2026-11-02, v7.4.0):** D1, D2 and D3 are implemented and green (319 backend / 17
frontend / 11 E2E). DG-1 passed locally; DG-2 awaits the CI run for the release commit.


### D1 — Disk-light training (S–M)

- `save_only_model=True` in `SFTConfig`/`TrainingArguments` with the reason in the code: there is no
  resume path, so optimizer/scheduler/RNG state is pure SSD wear.
- `save_steps = max(10, save_every)` so a `save_every=1` experiment cannot write a checkpoint per
  step.
- Measure it: after a run, sum the bytes actually written under the run directory and record
  `extras["disk_bytes_written"]` in the manifest. `foundry train run --json` shows it.
- `docs/DISK_LIGHT.md` with the before/after numbers from a real tiny run.

**Acceptance:** a tiny run with `save_every=1` writes no `optimizer.pt`/`scheduler.pt`/`rng_state.pth`,
keeps at most two checkpoint directories, and reports the byte count. Test asserts both.

### D2 — Opt-in update check (S–M)

- `Settings.update_check: bool = False` (persisted like the other overrides).
- `foundry/update.py::check_for_update(current, fetch=None)` — reads the GitHub releases API for
  this repository, compares versions, returns `{status, current, latest, available, url}`.
  A failure is a reported status, never an exception the UI has to catch.
- `GET /api/v1/update/check` — disabled by default and answers `{"enabled": false, "status": "disabled"}`;
  when enabled it performs exactly one HTTPS request with no identifier and **never downloads**.
- Settings UI: toggle + "Check now" with the result, in de/en/fr.

**Acceptance:** disabled path performs no network call (test patches the fetcher and asserts zero
calls); enabled path is tested with an injected response (newer, same, malformed); a download is
impossible because no code path writes a binary.

### D3 — Accessibility pass on the new panel (S)

- `role="region"` + `aria-label` on the browser card; `aria-busy` while loading.
- Load stage and generation metrics in a `role="status"` live region (pre-mounted, text only).
- Buttons keep native semantics; the model `<select>` already has a label.
- E2E: the region is reachable, the status region exists, and the panel exposes its state as text
  (never colour alone).

**Acceptance:** E2E checks the role/label/status elements; no `outline: none` without a replacement
on an interactive element.

## 3. Loops and gates

Same protocol as `V8_STABLE_PLAN.md` §3: narrow test first, then the module, then the suite; max 5
iterations; never weaken an assertion; a blocked item is recorded with the exact command that would
unblock it.

| Gate | Requires |
|---|---|
| **DG-1** | D1 + D2 + D3 green locally (backend, frontend, E2E) |
| **DG-2** | CI green on `main`; evidence in `docs/evidence/V73_1_DISKLIGHT.md` |

## 4. Explicitly not in this plan

- No delta-snapshot or 4 KiB-aligned writer. The filesystem already aligns; the win here is not
  writing optimizer state, not re-implementing a checkpoint format.
- No automatic download or install. The update check reports a version and a link; installing stays
  a user action.
- No contrast redesign of the whole UI. Only the new panel is in scope; the token palette is
  unchanged.
