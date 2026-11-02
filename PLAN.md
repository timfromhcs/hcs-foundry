# HCS Foundry — PLAN.md: from v7.0.0 to a stable, measured v8.0.0

> **Audience:** the autonomous engineering agent working in this repository.
> **Written:** 2026-11-02, against `main` @ `41b1b32` (v7.0.0 published, working tree clean).
> **Read first:** `AGENTS.md` (product contract — this file never overrides it), then this file top to bottom.
> **Language:** English body for consistency with `docs/`; German summary right below.
> **Status of this file:** a plan, not evidence. Every number in §2 was read from the repo or measured by the
> author of this plan; every number in §3 comes from a cited external source and must be re-checked when you
> implement against it (`AGENTS.md` §22: *a link is not permission and not proof*).

---

## 0. Kurzfassung (Deutsch)

**Wo wir stehen.** v7.0.0 ist veröffentlicht (Tag, CI grün, EXE, Site). Das Produkt kann auf dem Zielrechner
(Ryzen 7 7735HS, 19,8 GiB RAM, keine dedizierte GPU) 0.6B/1.7B/4B per INT4-LoRA trainieren, mergen,
nach GGUF exportieren und beantworten. **Was es nie gezeigt hat:** dass ein Training das Modell *messbar
besser* macht. Alle Läufe sind 2-Schritt-Smoke-Läufe auf 53 Datensätzen; die Benchmark-Suiten haben 6/4/3/3 Fälle,
der Promotion-Gate ist „+5 %“ — bei 6 Fällen entspricht ein einziger gekippter Fall 16,7 %. Außerdem laufen
**53 der 217 Backend-Tests (alle Trainings-Tests) in CI gar nicht**, weil `pip install -e .` ohne `[train]` läuft und
die Dateien per `importorskip` übersprungen werden. „CI grün“ sagt über die Trainer also nichts.

**Was v8.0.0 sein soll:** *„Erste ehrlich gemessene Verbesserung“* — kein neues Feature-Feuerwerk, sondern:

1. **Ein einheitlicher Trainings-Vertrag** (`TrainSpec` → `Trainer` → `RunManifest v2`) für *alle* Methoden
   (SFT, INT4, DPO, Distillation, RFT, später on-policy-Distillation/GRPO), mit **einer** Konformitäts-Testmatrix,
   die in CI mit einem winzigen Zufallsmodell läuft (kein Download) — dieselbe Spec läuft lokal klein, lokal echt,
   später optional remote.
2. **Eine einheitliche Auswertung mit Statistik** (gepaarte Tests, Konfidenzintervalle, Power-Planung),
   die den 5-%-Gate ersetzt.
3. **Unendlich viele verifizierbare Aufgaben** (prozedurale Generatoren mit Prüfern, Split nach Template) →
   Eval-Sets mit n ≥ 200 und ein **Rejection-Sampling-Flywheel**: schnell samplen (GGUF/Vulkan), automatisch
   prüfen, LoRA trainieren (CPU), mergen → GGUF → gepaart messen → Gate → Promote/Rollback.
4. **Ehrliche Engine-Experimente** gegen den I/O-Flaschenhals des 4B (Mikro-Batch statt Gradient-Accumulation,
   schicht-weises Batching, Vorhersage der Schrittzeit im Preflight) — jeweils als falsifizierbare Hypothese.
5. **Release-Zug mit Pushes:** lokal committen → lokal alle Gates → sauberer Checkout-Test → GitHub (CI) →
   Vercel (nur Landing-Site) → Release/EXE. Reihenfolge nie umdrehen (`AGENTS.md` §0.1).

Akzeptanz: §5 (A1–A9). **Ein negatives, sauber dokumentiertes Ergebnis ist ein gültiges Ergebnis** (A4) —
ein behaupteter Gewinn ohne Konfidenzintervall ist es nicht.

---

## 1. Operating contract for this plan

### 1.1 Non-negotiables (restated from `AGENTS.md`, because they get violated first under time pressure)

- **Order:** discover → plan → build locally → validate locally → prepare GitHub → push → Vercel last.
- **No fake success.** "Trained" requires parameter delta > 0 + finite loss + checkpoint. "Improved" requires
  a paired CI that excludes 0 (§6.3). "Pushed/Deployed" requires the remote to confirm it.
- **No silent cloud, no uploads.** Remote compute (Phase 11) is design-only unless the user approves cost,
  credentials and data leaving the machine, per run, in chat.
- **No force-push, no history rewrite, no push to a remote other than `origin` =
  `https://github.com/timfromhcs/hcs-foundry.git`.** Pushing needs the user's authorization in chat for *this*
  release train; if you have none when you reach a push gate, stop and ask (§4.4).
- **Never** commit: `data/`, `.env*`, `docs/audit/MACHINE_PROFILE.json`, weights, adapters, datasets, session
  exports, `dist-exe/`, `build/`. Run the secret scan in §4.3 before every push.
- Never weaken a test, an assertion, a benchmark scorer or a gate to turn a run green.

### 1.2 Machine contention (this has already cost a release)

`docs/evidence/v7_4b_speed.md` and `CHANGELOG.md` (6.0.0, "Not claimed") record 4B runs that stalled at ~3.6 GB
free RAM / 89 % commit because *other agent processes* were running. Therefore:

1. Before any job that needs > 1 GB or > 5 min: `foundry run doctor` and check free RAM + commit charge.
2. Engine experiments (Phase 7) are **machine-exclusive**: run them one at a time, never while another heavy
   session is active. Record free RAM at start in the evidence file.
3. Single writer on the git tree at a time. This PLAN.md was added as an **untracked file** and nothing else was
   touched; your first commit (WI-0.2) should include it.

### 1.3 The loops (all bounded — `AGENTS.md` §10.7, §16.3)

```text
WORK-ITEM LOOP (per WI)          retry budget 3 per failing test, 6 per WI
  read code+tests → write failing test → run (must fail for the right reason)
  → minimal implementation → run narrow test → run impacted tests → inspect diff
  → docs/evidence → commit            (never "fix" by editing the test's expectation)

PHASE LOOP (per Phase)           max 3 cycles
  implement WIs → LOCAL GATE (LG-n) → review own diff (accidental changes, dead code, fake UI)
  → fix → re-run gate → record evidence → merge to main locally → PUSH GATE (PG-n) when marked

EXPERIMENT LOOP (Phase 7, 8)     ≤ 2 h wall per experiment, 3 repeats, then decide
  hypothesis (falsifiable, with numbers) → baseline measurement → change → measurement
  → repeat ×3 → report mean/min/max → keep / drop / "inconclusive" → docs/evidence/v8_*.md
  A shipped optimisation needs a measured win on the real workload; "plausible" is not a win.

ESCALATION
  after the budget: write docs/plan/V8_BLOCKERS.md (what, repro command, what you tried, what you need),
  mark the WI "partial" in §10, continue with independent WIs. Never loop forever, never silently drop.
```

### 1.4 Evidence rule

Every claim that ends up in README/CHANGELOG/site has a file in `docs/evidence/` containing: the exact command,
date, git SHA, machine (free RAM at start), raw numbers, and what was *not* measured. Follow the pattern of
`docs/evidence/v7_4b_speed.md`.

---

## 2. Verified state of the repository (read from the tree on 2026-11-02)

### 2.1 What exists (verified)

| Area | Fact | Where |
|---|---|---|
| Release | v7.0.0 tag → `b2a4c11`; publication facts in `41b1b32`; CI run `41430719802` green; EXE 384,124,715 B | `docs/evidence/RELEASE_REPORT_V7.md` |
| Size | 266 tracked files; 21.6 k LOC Python; 217 backend `def test_` / 7 vitest / 9 Playwright | `git ls-files`, release report |
| Stack | FastAPI + SQLite(WAL) service, React 18/Vite SPA, `llama-server` supervised, PyInstaller EXE (no ML stack), Vercel **site only** | `README.md`, `docs/ARCHITECTURE.md`, `.vercelignore` |
| Trainers | `training/sft.py` (LoRA via transformers/PEFT), `int4.py` (own INT4 + LoRA, no bitsandbytes), `offload.py` (disk store), `merge_int4.py`, `algorithms.py` (`run_auto`, `run_distill`, `run_dpo`), `unified.py` (decide → train → merge → GGUF → quantize), `vram.py` (opt-in cache, measured slower), `playbook.py`, `recipes.py` | `server/foundry/training/` |
| Targets | LoRA targets include MLP (`gate/up/down_proj`) — consistent with §3.1 | `sft.py:33` |
| Defaults | SFT: `micro_batch=1, grad_accum=4, lora_r=8, alpha=16, lr=2e-4, seed=42, max_steps=20`; INT4 auto-config `r=4, seq=128, steps=20` | `sft.py:41-66`, `unified.py:66-73` |
| Upgrade Lab | candidates are **prompt patches** (`PROMPT_PATCHES`), evaluated on the same suite; promote needs `delta ≥ 0.05`, no critical regression (`CRITICAL_CASES={"system_prompt_secrecy"}`), frozen dataset, unchanged holdout hash, human approver | `upgrade.py:35-56, 309-382` |
| Benchmarks | local suites: instruction (6 cases), codegen (4), repo_fix (3), agentic (3), latency; SWE-bench *format* adapter (not run); UI regression via Playwright | `benchmarks/*.py`, `docs/BENCHMARKS.md` |
| MCP | 9 tools: `machine_scan, machine_budget, models_list, datasets_list, datasets_export, vram_status, train_int4, train_merge_int4, gguf_quantize` | `mcp_server.py`, `CHANGELOG.md` 7.0.0 |
| Local models | `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3.8-4B-Distill` (+GGUF) in `data/models/` | disk |

### 2.2 Measured numbers you can build on (from `docs/evidence/`, `README.md`, `CHANGELOG.md`)

| Measurement | Value |
|---|---|
| 0.6B LoRA SFT throughput | ~61.6 tok/s (CPU, fp32) — `docs/TRAINING.md` |
| 0.6B INT4, 2 steps, seq 128 | 20.7 s, peak RSS 2937 MB — `v6_vram.md` |
| 1.7B INT4 end-to-end | loss 5.023 → 4.081, peak RSS 4496 MB, merge 3.80 GiB, Q4_K_M 1223 MiB, answers at 51.4 tok/s — `v6_2b.md` |
| 4B INT4 disk store | store 4.57 GB + `rest.f32` 2.39 GiB; step ≈ 25 s warm vs **378–740 s** when the working set does not fit; sequential read 105 MB/s, page-fault read 66.6 MB/s, resident 2 660 MB/s — `v7_4b_speed.md` |
| 4B activations @ seq 128 | ≈ 84 MB → activation checkpointing is *not* the lever — `V6_PLAN.md` V3 |
| Inference | 0.6B Q4_K_M 113.5 tok/s, TTFT 0.79 s; Vulkan sweep 51.4 → 107.8 tok/s (2.1×) — README |
| GGUF export 0.6B | F16 in 209 s, Q4_K_M in 17 s — README |
| Data | every training run so far = **2 steps on 53 records** — `CHANGELOG.md` 7.0.0 "Not claimed" |

### 2.3 Gaps found while reading everything (these drive the plan)

| ID | Gap (evidence) | Why it matters | Fixed by |
|---|---|---|---|
| **G1** | CI backend job installs `pip install -e .` + `pytest` only (`.github/workflows/ci.yml`); 8 test files `importorskip("transformers"/"torch"/"peft")`: `test_algorithms, test_int4, test_merge_int4, test_moe, test_offload, test_sft, test_unified, test_vram` = **53 of 217 tests skipped in CI** | the whole training stack is untested on GitHub; "CI green" is not evidence for trainers | Phase 1 |
| **G2** | No statistics anywhere (`grep -i 'bootstrap|wilson|pass@k|variance'` finds nothing in `foundry/`; the only `confidence` hits are machine-tier labels); suites have 6/4/3/3 cases; gate = fixed `+5 %` | with n=6 one flipped case = 16.7 pp; the minimum *detectable* effect is ≈ 63 pp (§6.3). The Upgrade Lab cannot distinguish signal from noise | Phase 3 |
| **G3** | Upgrade candidates are prompt patches only; an *adapter* candidate (trained model) is not a candidate kind | the "recursive upgrade loop" of `AGENTS.md` §10 never closes through training | Phase 5 |
| **G4** | No unified trainer contract: `create_sft_run`, `create_int4_run`, `run_auto`, `run_dpo`, `run_distill`, `run_unified` have different signatures, return shapes and manifest keys; `unified.decide()` hard-codes `sft`/`int4` | every new method re-invents progress/cancel/manifest/testing; no shared conformance tests | Phase 2 |
| **G5** | No lockfile for the server (`server/pyproject.toml` uses `>=`); dev runs Python 3.14 (`__pycache__/*cpython-314*`), CI 3.12 | irreproducible environments; a number in `docs/evidence` cannot be tied to library versions except via the manifest | WI-0.3 |
| **G6** | `web/dist/**` and `web/tsconfig.tsbuildinfo` are **tracked** though they are build output (`.gitignore` only ignores root `/dist/`) | noisy diffs, stale bundles in git; `web/vercel.json` builds from source, so untracking is expected to be safe — **verify `scripts/build_exe.ps1` builds the SPA first** | WI-0.4 |
| **G7** | Stale docs: `KNOWN_LIMITATIONS.md` header says 4.1.0 and item 15 says "only 0.6B proven end-to-end" (v6 proved 1.7B); `V7_PLAN.md` §5 progress log still says all WPs "open" | the repo's own honesty rule is violated by its docs | WI-0.5 |
| **G8** | V7 plan promised MCP tools `models_run` and `bench_run`; shipped list has neither | an agent cannot run a benchmark or a prompt through MCP | Phase 9 |
| **G9** | `runtime/llama_server.py` and `gateway.py` expose no token log-probabilities (`grep n_probs|logprobs` → nothing) | on-policy distillation and KL-based quant checks need teacher/reference logprobs | WI-6.1, WI-8.1 |
| **G10** | `SFTConfig` default `micro_batch=1, grad_accum=4`: with the disk store each micro-batch is a full pass over the store, so accumulation multiplies the I/O that already dominates (see §3.8) | likely the cheapest 4B speed-up is unexamined | Phase 7 |
| **G11** | Public benchmark story is dated: `AGENTS.md` §9.2 lists SWE-bench Verified; OpenAI reports it as contaminated and moved to SWE-bench Pro (§3.7) | do not invest in Verified; prefer private/procedural + Pro/Live where feasible | Phase 4, 9 |
| **G12** | No quality claim of any kind has ever been made or tested; no held-out base-vs-tuned comparison exists | the product's central promise ("built to learn, tested to improve") is unproven | Phase 5 |

---

## 3. What the research says — and what we do about it

> Sources were fetched 2026-11-02 via search; several are blog/secondary summaries of papers. Where I could not
> read the primary text I say so. **Re-verify each item against the primary source before you cite it in docs.**
> Dates after my knowledge cutoff (e.g. "as of August 2026") are as reported by the cited page.

### 3.1 LoRA is enough for what we do — if configured right
*Source: Schulman et al., "LoRA Without Regret", Thinking Machines (2025-09-29) and HF TRL docs page of the same name;
Biderman et al. 2024, "LoRA Learns Less and Forgets Less" (arXiv 2405.09673).*

- For SFT on small/medium instruction and reasoning datasets LoRA matches full fine-tuning; in policy-gradient RL
  it matches even at **rank 1**. Optimal LR for LoRA is ~**10×** the full-FT LR and roughly independent of rank;
  apply LoRA to **all** layers incl. MLP (attention-only underperforms).
- LoRA *underperforms* full FT on large domain shifts (code/math, ~100 k pairs+), but **forgets less**; full FT
  learns perturbations of 10–100× higher rank.
- **Do:** keep MLP targets (already so); make **LR the swept hyper-parameter, not rank** (the playbook sweeps
  config as a blob — split LR out); treat rank as a capacity/memory knob. Add a *forgetting guard suite* to every
  training gate (§6.4) — "forgets less" is a hypothesis to measure, not a guarantee.

### 3.2 Evaluation needs error bars
*Source: Miller (Anthropic), "Adding Error Bars to Evals" (arXiv 2411.00640).*

- Report standard errors; compare models with **paired** question-level differences (removes shared difficulty);
  use **clustered** SEs when questions come in groups (clustering inflated SE up to ~3× in the paper's examples);
  plan sample size by power.
- **Do:** implement `paired_compare` and `required_n` (§6.3), make them the only way the Upgrade Lab and any
  training gate can say "better". Group by *template* (§7) so clustering is honest.

### 3.3 Post-training order and data discipline
*Source: Ai2 Tülu 3 (arXiv 2411.15124, blog).*

- Recipe: curate prompts per **target skill** → SFT → DPO (incl. on-policy pairs) → **RLVR** (verifiable rewards),
  with an evaluation suite used for development *and* decontamination: datasets with substantial overlap with
  eval sets are removed/decontaminated (n-gram + embedding matching); skill-specific mixtures are built first, then
  merged and iterated against lagging skills.
- **Do:** keep Foundry's order SFT → (DPO) → verifiable-reward stage; add train↔eval **decontamination** to the
  freeze step (n-gram overlap, §7.4); build the dataset as per-skill mixtures with visible ratios.

### 3.4 Small + clean beats big + noisy, but quantity still matters
*Sources: LIMA (1 k curated samples), Databricks "LIMIT" follow-up, Magpie (arXiv 2406.08464).*

- Small high-quality sets move *style/format* a lot; the data must match the **evaluation paradigm** you care about;
  Magpie: both quality and quantity matter and filtered synthetic data can beat larger noisy sets.
- **Do:** 53 records is far below useful; set a **data-readiness gate** (§7.5): ≥ 300 verified examples per skill
  before a "quality" run is allowed to call itself that; below it the run is labelled `smoke`.

### 3.5 RLVR / GRPO: use the *ideas*, not the CPU run
*Sources: DAPO (arXiv 2503.14476), Dr. GRPO (arXiv 2503.20783), GSPO (arXiv 2507.18071); Cameron Wolfe "GRPO++";
HF blog "How frontier models train on outcomes in 2026" (reports TRL `GRPOTrainer` default `loss_type="dapo"` as of
Aug 2026 and `importance_sampling_level="sequence"` for GSPO); Magistral drops zero-advantage groups.*

- Vanilla GRPO: `A_i = (r_i − mean(r)) / std(r)` per prompt-group, per-sequence length normalisation → length bias.
  **Dr. GRPO:** drop std normalisation and `1/|o_i|`. **DAPO:** clip-higher (ε_high > ε_low), **dynamic sampling**
  (discard groups where all rewards are equal — zero gradient), **token-level** loss aggregation, soft overlong
  penalty. **GSPO:** sequence-level importance ratio (more stable, esp. MoE).
- RL teaches **O(1) bits per episode** (Thinking Machines) and needs on-policy sampling *from the model being
  trained*. On this machine the trainer is CPU-bound at ~60 tok/s (a *training* figure; torch decode is not faster);
  sampling G=8 completions × 200 tokens per prompt from it is ≥ 27 s per prompt before any gradient step. **Not viable as the default path.**
- **Do:** (a) implement the loss/advantage math as **pure, unit-tested functions** (`train/losses.py`, §5.3) so the
  methods are testable on a tiny model in CI and reusable by any backend; (b) ship **RFT / expert iteration**
  (sample with the *fast* GGUF engine → verify → SFT on passes) as the CPU-friendly RLVR approximation (§8);
  (c) put real GRPO behind the `remote` backend (Phase 11), not on this CPU.

### 3.6 On-policy distillation is the efficient way to use a teacher
*Source: Thinking Machines, "On-Policy Distillation" (2025).*

- Student samples, teacher scores **every token** with log-probs; loss = per-token reverse KL
  (`advantage_t = −(log π_student(x_t) − log π_teacher(x_t))`). Reported: recovers an RL-trained policy in ~7–10×
  fewer gradient steps (≈ 50–100× less compute than the RL run); reuses data across epochs without memorising;
  depends on a capable teacher.
- **Do:** WI-6.1 adds `logprobs` (`n_probs`) support to the llama-server client (G9); WI-6.2 adds the
  `onpolicy_distill` method: teacher = a larger local GGUF (e.g. the 4B Q4_K_M, which already answers) scoring the
  student's samples; student = 0.6B/1.7B LoRA. Gate it on tokenizer compatibility (same vocab/family) — refuse
  otherwise with the reason.

### 3.7 Executable environments are the data source for coding agents; public benchmarks age fast
*Sources: SWE-Gym (arXiv 2412.21139, 2.4 k tasks, 11 repos; warns to git-clean to prevent reward hacking);
R2E-Gym (arXiv 2504.07164, 8.1 k procedural environments, hybrid execution-based + execution-free verifiers);
OpenAI "Why we no longer evaluate SWE-bench Verified" (contamination → SWE-bench Pro); SWE-bench Pro V2 page.*

- Tasks with executable tests are simultaneously *training data* (verified trajectories), *rewards*, and *evals*.
- **Do:** our own procedural task families with verifiers (§7) play this role at small scale; the user's own repos
  with tests are a later source (strict split, explicit permission — `AGENTS.md` §9.2). Treat any public benchmark
  as contaminated until proven otherwise; keep the SWE-bench *adapter*, do not chase Verified scores.

### 3.8 The 4B is I/O-bound; the lever is bytes-per-token, not prefetching
*Sources: this repo's `v7_4b_speed.md`; DeepSpeed ZeRO-Inference/ZeRO-Infinity (arXiv 2104.07857) — larger batch
sizes amortise weight fetching; LISA (arXiv 2403.17919) — memory ≈ ½ of full FT, but it saves optimizer memory,
not weight reads.*

- Per optimizer step the store is read `passes × (forward + backward-grad-input)` times, where `passes =
  micro_batches × grad_accum`. At `micro_batch=1, grad_accum=4` that is **4 full passes** for 4 sequences.
- Reading cost is per *pass*, not per token. Tokens per pass = `micro_batch × seq_len`. Activations at seq 128 are
  only ~84 MB (4B), so raising `micro_batch` to 8–16 (and `grad_accum` → 1) should cut I/O per token ~8–16× at
  almost no memory cost. **Hypothesis H1** (Phase 7): throughput in tokens/s rises ≥ 3× at `micro_batch=8` vs `1`
  with the same cold store. Also: **layer-major** scheduling (run all micro-batches through layer *l* before
  layer *l+1*) reads each layer once per pass regardless of micro-batch count, at the price of keeping
  per-layer activations for all micro-batches (H2).
- LISA/GaLore-style methods do **not** address this bottleneck (backward still needs grad-input through every
  layer's weights). Do not implement them for speed. (They may still matter for optimizer memory in full-FT — out of
  scope.)

### 3.9 Quantization quality is measured by distribution drift, not perplexity alone
*Sources: llama.cpp discussion #4110 (KL divergence vs PPL); arXiv 2601.14277 "Which Quantization Should I Use?"
(PPL does not predict GSM8K/IFEval outcomes; Q4_K_M ≈ community sweet spot); Qwen llama.cpp docs (imatrix helps,
`llama-imatrix`).*

- **Do:** after every `gguf quantize`, run a **quant gate**: KL divergence and top-1 *answer-flip rate* vs the F16
  model on a fixed prompt set, plus the task suite delta (paired, §6.3). Use `llama-perplexity --kl-divergence`
  (verify flag names against the installed build before coding). Offer an `imatrix` option for ≤ 3-bit only.

### 3.10 TurboQuant: probe, don't build (as of 2026-10)
*Sources: Google Research blog + paper (arXiv 2504.19874); llama.cpp discussion #20969; secondary report (morphllm.com,
sotaaz.com, Oct 2026): mainline llama.cpp has **no** TurboQuant — feature request closed "not planned", CPU-only PR
#21089 closed unmerged 2026-06-02; a **Hadamard rotation of the KV cache before quantization** (PR #21038, merged
2026-04-01) covers the rotation stage with the stock `q4_0` cache; vLLM ships a TurboQuant backend.*

- Consistent with `KNOWN_LIMITATIONS.md` #7. **Do:** keep the capability probe; add a *measured sweep* of what the
  installed build really offers (`-ctk/-ctv f16|q8_0|q4_0`, `-fa`): KV bytes/token, TTFT, tok/s, and
  long-context retrieval + KL vs f16 (Phase 8). Do not add a fork build to the product.

### 3.11 Unifying trainers: borrow the design, not the dependency
*Sources: comparisons of Unsloth/Axolotl/LLaMA-Factory/torchtune/TRL (secondary, 2026); Axolotl runs SFT, DPO,
GRPO… from one YAML and one dataset pipeline, but **misspelled keys are silently ignored** (a known failure mode).*

- None of these targets CPU + disk-offloaded INT4 on Windows; adopting one wholesale would discard v4–v7.
- **Do:** adopt the *pattern*: one declarative spec, one dataset pipeline, one manifest — with a **strict schema
  (`extra="forbid"`)** so a typo fails loudly (§5.1). Where TRL is installed, use its loss implementations as
  *oracles in tests* (compare numbers), not as a runtime dependency.

### 3.12 Evaluation framework: bridge, don't migrate
*Source: UK AISI Inspect (`inspect-ai`): Task = dataset + solver + scorer, 200+ evals, sandboxes (Docker/K8s/…),
bootstrap CIs, local-model support; used by many labs.*

- **Do (stretch, Phase 9):** an optional `inspect` bridge so community evals can run against the loaded GGUF; keep
  Foundry's deterministic suites as the default. Inspect's sandboxing leans on Docker — Windows without Docker →
  mark such evals "unsupported here", per `AGENTS.md` §14.3.

### 3.13 Base-model choice
*Source: distil labs benchmark (vendor blog; 12 small models × 8 tasks; Qwen3-4B-Instruct-2507 ranked first after
fine-tuning, "fine-tuning matters more than base choice"); Turing Post / PremAI lists for 2026 small models.*

- Vendor data, narrow tasks: use as a prior only. **Do:** the quality proof (Phase 5) uses **Qwen3-0.6B and
  Qwen3-1.7B** (feasible rounds on this CPU); the 4B stays a *scale/mechanism* demonstration. Record the base-model
  decision in an ADR with the measured fit.

### 3.14 Weight-space tools for regression protection
*Sources: NVIDIA model-merging intro (SLERP, TIES, DARE), model-merging survey (arXiv 2603.09938), AdapterSoup.*

- **Do (cheap, measurable):** evaluate each candidate adapter at LoRA scale `s ∈ {0.25, 0.5, 0.75, 1.0}` (merge with
  `W + s·ΔW`) and pick the Pareto point between target gain and forgetting-guard loss. No retraining needed;
  `merge_int4_adapter`/`merge_adapter` get a `scale` parameter (Phase 5). TIES/DARE across *multiple* adapters is
  Phase 12+ (not planned).

---

## 4. Release train, loops and pushes

### 4.1 Versions

| Version | Contents | Push gate |
|---|---|---|
| **v7.1.0** (trust release) | Phases 0–3: honest CI, unified trainer contract, unified eval + stats gate | PG-1 |
| **v7.2.0** (the loop) | Phases 4–6: generators, flywheel, first measured result, on-policy distill | PG-2 |
| **v8.0.0** (stable) | Phases 7–10: engine experiments, quant/KV gates, MCP/UI/docs, release | PG-3 |

Tag only at a push gate. Interim commits stay local (or on a local branch) until the gate passes.

### 4.2 Branching and local flow

```text
main (v7.0.0)
 └─ v8/work                   ← all work; one commit per WI; conventional prefixes (feat/fix/test/docs/chore)
     ├─ after each LOCAL GATE: tag nothing, run §4.3 checks, fast-forward main:  git switch main && git merge --ff-only v8/work
     └─ at PG-n:   §4.3 + §4.4
```

### 4.3 Local gate checklist (run all; paste results into the phase evidence file)

```bash
# backend — with the training extras installed (this is the point of Phase 1)
cd server && python -m pytest -q -rs                      # -rs must show NO training skips after Phase 1
python -m ruff check foundry tests
# frontend
cd ../web && npx tsc --noEmit && npx vitest run && npm run build && npx playwright test
# repo hygiene
cd .. && git status --short                                # clean except intended files
git diff --stat main..v8/work | tail -1
git ls-files | rg -i "\.(gguf|safetensors|pt|bin|ckpt|db)$|\.env|MACHINE_PROFILE" && echo "FORBIDDEN FILE TRACKED"
rg -n -i "hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|vck_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}" --glob '!data/**' .   # must print nothing
# clean-checkout proof (catches "works only on my machine")
git clone --no-local . "$TMP/foundry-clean" && cd "$TMP/foundry-clean/server" && <create venv, install .[train,dev]> && python -m pytest -q
```

### 4.4 Push gate (PG-n) — local first, then cloud, strictly in this order

1. All LG items of the included phases green **locally** and the clean-checkout test green.
2. `CHANGELOG.md`, `README.md` (version badge, capability table), `docs/architecture/KNOWN_LIMITATIONS.md`, release
   report `docs/evidence/RELEASE_REPORT_V<x>.md` updated; version bumped in `server/pyproject.toml` and
   `foundry.__version__` (a test already fails on drift).
3. Confirm with the user in chat (one line): remote `origin`, branch `main`, tag `vX.Y.Z`. No answer → stop at step 2,
   leave local commits, report exact next commands. **Never force-push.**
4. `git push origin main` → `git push origin vX.Y.Z` → watch CI to completion (`gh run watch`); on red: fix forward
   with a new commit, never delete the tag silently — if a tag must move, tell the user first.
5. Only after CI is green: site/landing deploy (`site/` only — `.vercelignore` already allowlists it), verify the
   public URL, `foundry` pairing demo still works.
6. Release: rebuild EXE **from the tagged commit** (`scripts/build_exe.ps1`), GitHub release with notes + asset,
   record the publication facts (commit, CI run id, release URL, EXE size) in a final docs commit — exactly like
   `41b1b32` did for v7.0.0.
7. State in the final message what was *not* done (e.g. "Vercel not deployed: no credentials").

---

## 5. Target: what "stable v8.0.0" means (acceptance criteria)

| # | Criterion | Evidence |
|---|---|---|
| **A1** | CI runs the training conformance matrix (§6.2) on a tiny random model with **0 skips** in training tests, green on GitHub | CI log + `pytest -rs` |
| **A2** | One spec file drives ≥ 5 methods (`sft, int4, dpo, distill, rft`) through `foundry train run`; every run writes a schema-valid `RunManifest v2`; legacy manifests load via `from_legacy` | tests + 5 manifests |
| **A3** | Upgrade/training gates use paired statistics (§6.3); the fixed `+5 %` rule is gone; known-answer unit tests for the stats | `tests/test_stats.py` |
| **A4** | **Either** a reproducible held-out improvement for Qwen3-0.6B/1.7B (paired Δ, 95 % CI lower bound > 0, n ≥ 200 per suite, beats *both* base and best-prompt baseline, guard suite not regressed) **or** a truthful negative report with the data and the diagnosis. No third option. | `docs/evidence/v8_flywheel.md` |
| **A5** | Preflight predicts 4B/1.7B/0.6B step time from a measured sequential-read bandwidth within ±50 % on 3 runs | `v8_engine.md` |
| **A6** | `gguf quantize` reports KL + flip-rate vs F16; KV-cache sweep documented from the installed build | `v8_quant.md` |
| **A7** | MCP exposes `train_plan, train_run, models_run, bench_run, eval_compare` (typed, path-allowlisted, no shell), tested | `tests/test_mcp.py` |
| **A8** | Docs honest and current: README, KNOWN_LIMITATIONS, TRAINING, BENCHMARKS, UI_GUIDE, TUTORIAL (new walkthrough), CHANGELOG | review |
| **A9** | PG-3 completed through GitHub CI, site, EXE release, publication facts recorded — or the exact blocker recorded | release report |

---

## 6. The unified design — one contract for all trains, all tests

### 6.1 Principle

> **One declarative `TrainSpec` in, one `RunManifest` out, one `Trainer` protocol in between, one conformance test
> suite over every registered trainer.** The *same spec* is executed at four **profiles** that differ only in scale:
>
> | Profile | Model | Data | Where | Purpose |
> |---|---|---|---|---|
> | `ci-tiny` | random 2-layer Qwen3-config (≈ 1–2 M params, built in the test, no download) | 16 synthetic rows | GitHub runner, CPU | prove *mechanics*: loss finite, params change, resume, cancel, manifest valid |
> | `local-smoke` | Qwen3-0.6B | 53–300 rows | this machine | prove *integration* with real tokenizer/chat template, GGUF export |
> | `local-real` | Qwen3-0.6B / 1.7B (4B = scale demo) | ≥ 300 verified rows/skill | this machine, hours | prove *quality* with stats (Phase 5) |
> | `remote` | ≥ 4B / GRPO | large | opt-in service only | design-only in this plan (Phase 11) |

### 6.2 Layout (new package `server/foundry/train/`; existing modules are wrapped, not rewritten)

```text
foundry/train/
  spec.py          TrainSpec & friends (pydantic v2, extra="forbid", spec_version)
  manifest.py      RunManifest v2, from_legacy(dict) -> RunManifest, validate()
  base.py          Trainer Protocol, TrainContext, Preflight, RunResult, errors
  registry.py      @register("sft","fp32"), get_trainer(spec) -> Trainer, list_trainers()
  losses.py        pure functions: sft/dpo/reverse_kl/grpo_advantages/dapo_loss (torch only)
  stats.py         paired_compare, required_n, wilson_ci, bootstrap_ci, pass_at_k      (no torch)
  adapters/        sft.py int4.py dpo.py distill.py auto.py   ← thin wrappers over existing code
  methods/         rft.py onpolicy_distill.py grpo.py(remote-only stub with pure losses)
  conformance.py   check functions used by tests (and by `foundry train doctor`)
server/tests/train/
  conftest.py      tiny_model fixture, tiny_dataset fixture, profile fixtures
  test_spec.py test_manifest.py test_losses.py test_stats.py test_conformance_matrix.py
```

### 6.3 Core types (signatures only — you write the bodies; names are binding, other tasks use them)

```python
# spec.py
class Method(str, Enum): SFT="sft"; INT4="int4"; DPO="dpo"; DISTILL="distill"; RFT="rft"; ONPOLICY_DISTILL="onpolicy_distill"; GRPO="grpo"
class Backend(str, Enum): FP32_LORA="fp32_lora"; INT4_STREAM="int4_stream"; INT4_STORE="int4_store"; REMOTE="remote"
class ModelRef(BaseModel):    path: str; revision_hash: str | None = None
class DataRef(BaseModel):     dataset_id: str; manifest_hash: str; split: Literal["train"] = "train"; min_rows: int = 2
class Hyper(BaseModel):       lr: float; lora_r: int; lora_alpha: int; micro_batch: int; grad_accum: int; seq_len: int; max_steps: int; seed: int
class Budget(BaseModel):      max_wall_s: int; max_ram_bytes: int | None = None; max_disk_bytes: int | None = None
class EvalPlan(BaseModel):    suites: list[str]; guard_suites: list[str]; every_n_steps: int = 0; final: bool = True
class StopRules(BaseModel):   patience: int = 3; min_delta: float = 0.0; nan_abort: bool = True
class TrainSpec(BaseModel):   # model_config = ConfigDict(extra="forbid")  ← typos must fail loudly (Axolotl lesson)
    spec_version: Literal[1] = 1
    method: Method; backend: Backend; model: ModelRef; data: DataRef
    hyper: Hyper; budget: Budget; eval: EvalPlan; stop: StopRules
    profile: Literal["ci-tiny","local-smoke","local-real","remote"]
    method_args: dict[str, Any] = {}          # validated per method by the registered trainer's own model

# base.py
@dataclass
class Preflight:  ok: bool; blockers: list[str]; warnings: list[str]; est_ram_bytes: int; est_disk_bytes: int
                  est_step_seconds: tuple[float, float]  # (low, high) — see WI-7.4; (0,0) = unknown, never a guess
class TrainContext:  progress: Callable[[float,str],None]; log: Callable[[str],None]; cancel: threading.Event
                     out_dir: Path; db: Database; settings: Settings; clock: Callable[[],float]
class Trainer(Protocol):
    method: Method; backend: Backend
    def preflight(self, spec: TrainSpec, ctx: TrainContext) -> Preflight: ...
    def run(self, spec: TrainSpec, ctx: TrainContext, *, resume_from: Path | None = None) -> RunResult: ...
class RunResult(BaseModel):  run_id: str; manifest: RunManifest; adapter_dir: Path | None; checkpoints: list[Path]

# manifest.py  (superset of today's three shapes; method-specific data goes in `extras`)
class RunManifest(BaseModel):
    manifest_version: Literal[2] = 2
    run_id: str; spec: TrainSpec; spec_hash: str
    env: EnvInfo               # python, torch, transformers, peft, device, threads, os, git_sha
    base_model_sha256: str; dataset_manifest_hash: str; adapter_sha256: str | None
    steps: int; tokens_seen: int; wall_seconds: float
    losses: list[float]; eval_history: list[EvalPoint]; param_delta_max: float; grads_finite: bool
    memory: MemInfo            # peak_rss_mb, private_mb, file_backed_mb  (existing PeakRSS)
    io: IoInfo | None          # bytes_read_store, passes, seq_read_mbps  (new, WI-7.2)
    tuning: dict; extras: dict
    status: Literal["completed","cancelled","failed","resumed"]; failure: str | None
def from_legacy(manifest: dict) -> RunManifest: ...        # maps sft / int4 / dpo / distill manifests

# stats.py  (pure python; numpy optional)
def wilson_ci(k:int, n:int, z:float=1.96) -> tuple[float,float]: ...
def paired_compare(a: Sequence[float], b: Sequence[float], *, clusters: Sequence[Hashable] | None = None,
                   alpha: float = 0.05, n_boot: int = 10_000, seed: int = 0) -> PairedResult: ...
    # PairedResult: delta, se, ci_low, ci_high, p_value, n, n_discordant, method("mcnemar_exact"|"paired_bootstrap")
def required_n(delta: float, discordance: float, *, alpha: float = 0.05, power: float = 0.80) -> int: ...
    # n ≈ (z_{α/2}+z_{β})² · d / Δ²  ;  Δ=0.10,d=0.20 → 157 ; Δ=0.05,d=0.20 → 628  (ceil)
def pass_at_k(n:int, c:int, k:int) -> float: ...          # unbiased estimator 1 - C(n-c,k)/C(n,k)
```

**Why these numbers matter (the argument for Phase 3).** With per-item paired difference in {−1,0,+1} and
discordance rate `d`, `SE(Δ) ≈ sqrt(d/n)`. For `n = 6, d ≈ 0.3` the 80 %-power minimum detectable effect is
`2.8·sqrt(0.3/6) ≈ 0.63`. Today's gate promotes on `+0.05` — it can only ever react to noise or to enormous effects.

### 6.4 The conformance matrix (the unified *test*)

Every registered trainer must pass the same checks on `ci-tiny` (automatic, CI) and a subset on `local-smoke`
(marker `@pytest.mark.local`, not run in CI):

| ID | Check | How |
|---|---|---|
| C1 | `preflight` is honest | too-large model/budget → `ok=False` with numbers; never raises |
| C2 | real optimisation | `param_delta_max > 0`, LoRA `B ≠ 0`, `grads_finite`, all losses finite |
| C3 | learns something trivial | on a fixed tiny dataset, mean loss of last 25 % of steps < first 25 % (seeded, margin from a calibrated run, not "lower by a hair") |
| C4 | determinism | same spec+seed twice (1 thread) → identical loss list (atol 1e-6) |
| C5 | resume equivalence | 4 steps straight == 2 steps + resume + 2 steps (params `allclose`, atol 1e-5) |
| C6 | cancel | `ctx.cancel.set()` mid-run → status `cancelled` within 5 s, checkpoint written, no partial adapter masquerading as final |
| C7 | source untouched | base-model file hashes identical before/after |
| C8 | manifest valid | validates as `RunManifest v2`; `spec_hash` matches; `from_legacy(old_shape)` round-trips |
| C9 | adapter reloads | adapter loads onto a fresh base and produces logits that differ from the base by > ε; merged weights ≈ base + ΔW |
| C10 | no false success | with `max_steps=0` or all-masked labels the run reports `failed`/`no_update`, never `completed` |
| C11 | budget | exceeding `Budget.max_wall_s` stops cleanly with a reason |
| C12 | typo rejected | unknown spec key → `ValidationError` (strict schema) |

Method-specific **oracle tests** in `test_losses.py` (pure tensors, no model):
- `dpo_loss`: equals `−log σ(β[(π_w−ref_w)−(π_l−ref_l)])` on random log-probs; 0.5·log2 ≈ 0.693 at zero margin; monotone in margin; compare with TRL's value when TRL is importable (skip otherwise, never fail).
- `reverse_kl_loss`: exactly 0 when student == teacher log-probs; non-negative in expectation over student samples.
- `grpo_advantages(mode="grpo")`: per-group mean 0, std 1 (ε-safe); `mode="dr_grpo"`: mean 0, no std scaling.
- `dapo_filter_groups`: groups with identical rewards are dropped; if all dropped → explicit `NoSignal` result (not a silent zero loss).
- `dapo_token_loss`: aggregates over `Σ_i |o_i|` tokens (not per-sequence mean); `clip_high > clip_low` honoured.
- `soft_overlong_penalty`: 0 below the cache threshold, linear to −1 at max length.

### 6.5 One entry point, four surfaces

- CLI: `foundry train run --spec spec.yaml [--profile local-smoke]`, `foundry train plan --spec …` (preflight only),
  `foundry train doctor` (runs C1–C12 on `ci-tiny` locally in < 2 min). Existing commands (`train int4|sft|dpo|…`)
  become **thin aliases that build a `TrainSpec`** and print a one-line deprecation note; remove in the version after.
- API: `POST /api/v1/training/runs {spec}` → job; `GET …/{id}` returns `RunManifest`. Old endpoints delegate.
- MCP: `train_plan`, `train_run` (Phase 9). Old `train_int4` etc. stay one version as aliases.
- UI: Training Lab form generated from the spec schema (no hand-maintained duplicate of defaults); show `Preflight`
  and the predicted step time range; show manifest v2 fields incl. IO/pass count.

---

## 7. Verifiable tasks: unlimited data with answers (Phase 4)

### 7.1 Why

Quality needs n ≥ 200 per eval and ≥ 300 verified training rows per skill (§3.4). Handwriting that is impossible;
**procedural families with programmatic verifiers** make it cheap, leak-free (split by template) and RLVR-ready
(§3.5, §3.7). Multilingual (de/en/fr) variants fit the product.

### 7.2 Task families (each = generator + verifier + metadata; pure Python, deterministic given a seed)

| Family | Prompt | Verifier | Notes |
|---|---|---|---|
| `fmt` | "Answer in exactly N words / only JSON with keys … / all-caps / no commas / start with …" (IFEval-style programmatic constraints) | deterministic checks | stack 1–3 constraints for difficulty |
| `arith` | multi-step word problems, units, German/French number words | exact numeric match | operands random; avoid memorised items by magnitude/seed range |
| `extract` | text in de/en/fr → JSON fields (dates, amounts, names) | JSON schema + field equality | template families per document type |
| `code_synth` | "write `f(…)` that …" from a spec | hidden unit tests in the existing sandbox | reuse `execute_python` (timeout, temp dir) |
| `code_fix` | function + failing tests produced by an **AST mutation** of a correct function | tests pass | mutation operators: off-by-one, wrong operator, swapped args, missing return |
| `toolcall` | pick tool + args from a small schema | JSON validity + arg equality | matches the `agentic` suite format |

### 7.3 Splits (anti-leakage by construction)

- Every item has `family`, `template_id`, `seed`, `difficulty`, `lang`. **Split by `template_id`** (not by instance):
  `train` templates ∩ `dev` templates ∩ `test` templates = ∅. `test` is **frozen once** (hash recorded, never used
  for selection, never regenerated with the same templates).
- Keep a second, *rotating* test set from fresh templates each release (freshness; `AGENTS.md` §10.4).
- Clusters for the stats = `template_id` (§3.2).

### 7.4 Decontamination (Tülu-3 style, applied at freeze time)

Normalised 8-gram overlap and a cheap embedding/SimHash check (existing `data/` SimHash code is reusable) between any
**training row** and **every eval item**; overlap above a threshold blocks `freeze` with a list of offending ids.
Unit test with a planted duplicate.

### 7.5 Readiness gate

A run may be labelled `quality` only if: ≥ 300 verified rows per targeted skill, decontamination passed, eval
`n ≥ required_n(Δ_target, d_measured)`. Otherwise the manifest says `smoke` and the UI shows it.

---

## 8. The flywheel (Phase 5) — the CPU-realistic RLVR loop

```text
 base GGUF (fast, Vulkan ~100 tok/s)          ┌────────────────────────────────────────────┐
   │ sample k=4–8 completions / train prompt  │ ROUND r                                    │
   ▼                                          │  1 SAMPLE   llama-server, T=0.7, n slots   │
 verifier (programmatic, §7)                  │  2 VERIFY   family verifier → pass/fail    │
   │ keep passes; for prompts with ≥1 pass    │  3 SELECT   dedupe, cap per template,      │
   ▼                                          │             keep shortest correct,         │
 SFT dataset  (provenance: model, seed, round)│             DAPO-style: drop all-pass and  │
   │                                          │             all-fail prompts (no signal)   │
   ▼                                          │  4 TRAIN    LoRA (spec, §6) on CPU          │
 LoRA adapter  ──► merge(scale s) ──► GGUF ──►│  5 EVAL     dev+test (paired) + guard suite│
   ▲                                          │  6 GATE     §3.2 stats; human approve      │
   └───────── next round samples from the NEW └────────────────────────────────────────────┘
              model only if the gate passed
```

Budget estimate for 0.6B (derived from measured rates — **re-measure**): sampling 1 000 prompts × k=4 × ~150 tokens
≈ 600 k tokens ≈ 1.5–2 h at ~100 tok/s (more with parallel slots); training ~50 k tokens at ~60 tok/s ≈ 15 min/epoch;
merge+export+quantize ≈ 4 min; eval (n=200 × 3 suites, greedy) ≈ 10 min. ≈ 2–2.5 h per round, 3 rounds overnight.

Required properties:
- **Baselines in every report:** (b0) base, (b1) base + best *prompt-only* patch (what Upgrade Lab did before),
  (b2) SFT on teacher/gateway data if a teacher is configured, (b3) the candidate at LoRA scale
  s ∈ {0.25, 0.5, 0.75, 1.0}. A gain that b1 already achieves is a prompt win, not a training win.
- **Guard suite** (forgetting, §3.1): general instruction/format items *not* in the target skills + the existing
  `system_prompt_secrecy` critical case; non-inferiority test with margin (e.g. lower CI bound of Δ ≥ −0.03).
- **Candidate kind `adapter`** added to the Upgrade Lab: stores spec_hash, run_id, adapter sha, merged/GGUF sha,
  parent model, dataset hashes (train/dev/test), per-item outcomes (needed for paired tests), baseline run ids.
  `promote()` calls `stats.paired_compare` and refuses unless: lower CI bound > 0, `n ≥ min_n`, guard non-inferior,
  holdout hash unchanged, human approver. Rollback restores the previous active profile **and** previous GGUF.
- **No selection on test.** Round-to-round decisions use `dev`; `test` is read once per candidate for the gate.
  A candidate that was tuned against `test` more than once is flagged `test_reused` and cannot promote.

---

## 9. Engine experiments for the 4B (Phase 7) — falsifiable, machine-exclusive

All on the existing store `data/training-runs/v41-int4-4b/int4-store` (4.57 GB) unless noted. Record free RAM, commit
charge and whether the page cache was cold (reboot or `RAMMap` standby flush — document how). 3 repeats.

| Exp | Hypothesis | Change | Pass criterion | If it fails |
|---|---|---|---|---|
| **E1** | I/O per token ∝ 1/(micro_batch·seq): bigger micro-batches amortise store reads (§3.8) | `micro_batch ∈ {1,2,4,8,16}`, `grad_accum=1`, seq 128 | tokens/s ≥ 3× at mb=8 vs mb=1, cold store; peak RSS grows < 1 GB | record the curve; keep defaults; note "not I/O-per-pass bound" |
| **E2** | layer-major scheduling reads each layer once per pass regardless of micro-batch count | process N micro-batches per layer before advancing (forward and backward), keep per-layer activations | ≥ 2× over E1's best at equal tokens/step, memory ≤ budget | drop; document activation memory as the limiter |
| **E3** | `rest.f32` as bf16 halves leftover I/O (2.39 GiB → 1.20 GiB) | keep embedding/head bf16 in the model (no fp32 upcast at attach — the reason v7 deferred it) | step time ↓ ≥ 20 % cold, loss within 1e-3 relative of fp32 over 5 steps | drop; record numerical drift |
| **E4** | sequential, unbuffered reads beat mmap page-faults on a cold cache (2 660 vs 66.6 MB/s observed on the same file) | batched per-layer `readinto` via `io.FileIO` + large aligned buffers; Windows: `FILE_FLAG_SEQUENTIAL_SCAN` | cold-cache pass time ↓ ≥ 30 % | drop |
| **E5** | step time is predictable from bytes and measured bandwidth | `T ≈ passes · (store_bytes + touched_rest_bytes)/B_seq + compute`; `B_seq` measured in preflight by reading ~256 MB of the store | prediction within ±50 % on 0.6B, 1.7B, 4B (3 runs each) | widen the interval; show "unknown" if wider than 4× — never a precise fake ETA |

Deliverables: `scripts/store_io_bench.py` extended; `docs/evidence/v8_engine.md`; `Preflight.est_step_seconds`
(WI-7.4); `foundry run doctor` prints the ETA range and a warning when predicted > `Budget.max_wall_s`.
Ship only what wins; losing experiments are documented (that is also a result).

---

## 10. Work items

Legend: **S** ≈ ≤ 2 h, **M** ≈ half a day, **L** ≈ 1–2 days of focused agent time. "Depends" = hard dependency.
Every WI ends with: tests green, evidence/docs line, commit `type(scope): summary`.

### Phase 0 — Baseline, guardrails, hygiene (S–M)

- [ ] **WI-0.1 Baseline numbers.** Run §4.3 with the project's real interpreter (the system `python` on PATH here lacks
  `fastapi/torch`; use the venv/interpreter the dev loop uses, 3.14). Write `docs/evidence/v8_baseline.md`: test
  counts, skips, durations, library versions (`pip freeze`), free RAM. *Done when:* the file exists and states the
  real skip count (expected 53 once `-rs` is read).
- [ ] **WI-0.2 Branch + this plan.** `git switch -c v8/work`; commit `PLAN.md` as `docs(plan): v8 plan`.
- [ ] **WI-0.3 Pin the environment (G5).** Generate `server/requirements-lock.txt` (and `-train.txt` with CPU torch)
  with `uv pip compile` (or pip-tools); document the 3.14-dev / 3.12-CI mismatch and either align CI to the dev
  version if wheels exist or record why not. CI installs from the lock.
- [ ] **WI-0.4 Tracked build output (G6).** Confirm `scripts/build_exe.ps1` builds the SPA before packaging and
  `web/vercel.json` builds from source; then `git rm --cached -r web/dist web/tsconfig.tsbuildinfo`, add
  `web/dist/` and `*.tsbuildinfo` to `.gitignore`. *Done when:* a clean clone builds the EXE and the SPA.
- [ ] **WI-0.5 Stale docs (G7).** Update `KNOWN_LIMITATIONS.md` header + items 15/16/24 to v7.0.0 facts; V7_PLAN §5
  progress log to the true statuses; add the missing MCP tools to the V7 table as "not shipped".

**LG-0:** WI-0.1…0.5 done; `git status` clean; baseline evidence committed.

### Phase 1 — CI that actually tests training (M) — fixes G1

- [ ] **WI-1.1 `tiny_model` fixture** (`server/tests/train/conftest.py`): build a random-init
  `Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
  vocab_size=512)` + a minimal tokenizer (train a tiny BPE with `tokenizers`, or a char-level tokenizer saved with
  `PreTrainedTokenizerFast`), `save_pretrained` to `tmp_path`. Test: loads, forward works, < 3 s.
- [ ] **WI-1.2 CI backend job installs the train extra** with CPU wheels (`pip install torch --index-url
  https://download.pytorch.org/whl/cpu`, then `pip install -e ".[train,dev]"`), caches pip, runs `pytest -q -rs`
  and **fails if any test in the 8 training files is skipped** (`pytest --strict-markers` + a tiny conftest hook
  `pytest_collection_modifyitems` that errors when `FOUNDRY_REQUIRE_TRAIN=1` and a `training` marker test would skip).
- [ ] **WI-1.3 Make the existing 53 tests pass on the tiny fixture where they now download or need real models**
  (adapt fixtures, not assertions). Mark genuinely local-only tests `@pytest.mark.local` and exclude them from CI
  explicitly (visible, counted), instead of silent `importorskip`.
- [ ] **WI-1.4 Timing budget:** CI backend job ≤ 12 min; if exceeded split `-m "not slow"` / nightly.

**LG-1:** a clean-venv local run using the lock + CPU wheels (§4.3) shows 0 training skips in `pytest -rs`
(`act` as a second check if installed). GitHub itself only confirms at PG-1 — do not push earlier just to see CI.

### Phase 2 — Unified Training Contract (L) — fixes G4. Depends: Phase 1

- [ ] **WI-2.1 `spec.py` + `test_spec.py`:** schema from §6.3; tests: valid spec round-trips JSON/YAML; unknown key
  rejected (C12); `spec_hash` stable under key order; profile defaults produce valid specs.
- [ ] **WI-2.2 `manifest.py` + `from_legacy` + tests:** feed the **real** manifests from `data/training-runs/*/manifest.json`
  copied as small redacted fixtures (strip paths) into `server/tests/fixtures/manifests/`; each maps without data loss
  of loss/steps/memory/offload/vram keys (unknown keys go to `extras`).
- [ ] **WI-2.3 `base.py` + `registry.py`:** Protocol, `register`, `get_trainer`, `list_trainers`; test: registering two
  trainers for the same (method, backend) raises.
- [ ] **WI-2.4 Adapters** wrapping `create_sft_run`, `create_int4_run`, `run_dpo`, `run_distill`, `run_auto`.
  No behaviour change: the adapter builds the legacy config from `TrainSpec`, calls the old function, converts the
  returned manifest with `from_legacy`, and fills the new fields the old code lacked (`param_delta_max`,
  `grads_finite`, `tokens_seen`, `spec_hash`, `env`). Where the old code cannot supply a field, the trainer must be
  changed minimally to record it (small, tested edits in `sft.py`/`int4.py`).
- [ ] **WI-2.5 Conformance suite** (`conformance.py`, `tests/train/test_conformance_matrix.py`): C1–C12 parametrized over
  `list_trainers()`. Expect to *find real bugs* (C4 determinism, C5 resume, C6 cancel latency, C10 no-false-success are
  the likely ones); fix them in the trainers with a regression test each. This WI is the point of the phase.
- [ ] **WI-2.6 `losses.py` + `test_losses.py`:** the pure functions and oracle tests from §6.4. Refactor `run_dpo` to
  call `losses.dpo_loss` (same numbers; pin with a test using a fixed seed before/after).
- [ ] **WI-2.7 Surfaces:** `foundry train run|plan|doctor`; legacy commands become aliases; API
  `POST /training/runs`; `unified.decide()` returns a `TrainSpec` (keeps `playbook` seeding); `run_unified` consumes
  specs. Tests: CLI via `typer`/argparse runner (whatever `cli.py` uses), API test with the tiny fixture.
- [ ] **WI-2.8 Playbook split:** record `lr` separately from other hyper-parameters (§3.1) so the playbook can
  suggest LR for a (model, dataset) pair without pinning rank.

**LG-2:** `foundry train doctor` green (< 2 min); C1–C12 pass for sft, int4, dpo, distill; old CLI commands still work
(existing tests untouched and green); no legacy manifest key lost.

### Phase 3 — Unified evaluation and the statistical gate (M–L) — fixes G2

- [ ] **WI-3.1 `stats.py` + `test_stats.py`:** known-answer tests: Wilson CI vs hand-computed (k=8,n=10 → ≈ (0.490, 0.943));
  exact McNemar p for (b=10,c=2) equals the binomial tail 2·P(X≤2 | n=12,p=.5) = 0.0386; paired bootstrap on a
  constructed case recovers Δ and covers it in ≥ 93 % of 200 simulated resamples; clustered SE > naive SE when
  within-cluster correlation is planted; `required_n(0.10, 0.20) == 157` (±1), `required_n(0.05, 0.20) == 628` (ceil of 627.9);
  `pass_at_k` against brute-force enumeration for small n.
- [ ] **WI-3.2 Per-item outcomes everywhere:** `run_suite` results already hold per-case status; guarantee a stable
  `item_id` and numeric `score ∈ [0,1]` per item and store them (needed for pairing). Migration for older runs: pairs
  only if `item_id`s match, else `fair:false` (existing behaviour in `compare_runs`).
- [ ] **WI-3.3 `compare_runs` gains `paired`:** returns `PairedResult`; UI Benchmarks compare view shows Δ, CI, p, n, and
  an explicit "underpowered: need n≈X" badge using `required_n`.
- [ ] **WI-3.4 Gate rewrite in `upgrade.promote`:** replace `delta < threshold` by the §8 rules; keep `threshold` as
  an *additional* minimum practical effect (`min_effect`, default 0.03), not the significance test. Keep every
  existing test intent: update tests to supply n large enough or assert the new "underpowered" refusal.
  Document the change in CHANGELOG as breaking behaviour.
- [ ] **WI-3.5 Sampling mode:** suites accept `samples_per_item` and `temperature`; report `pass@1` (mean) and
  `pass@k`; seeds recorded.

**LG-3:** with the old 6-case suite the new gate refuses to promote a +16.7 pp flip and says why (regression test).

### → **PG-1 (v7.1.0):** Phases 0–3 complete. Follow §4.4.

### Phase 4 — Verifiable task generators and splits (M–L) — fixes G11 partially. Depends: Phase 3

- [ ] **WI-4.1 `foundry/tasks/` package:** `Task` dataclass (`family, template_id, seed, difficulty, lang, prompt,
  verifier_spec`), registry, `generate(family, n, seed, split)`, `verify(task, completion) -> VerifyResult(score,
  detail)`. Start with `fmt`, `arith`, `extract`; then `code_synth`, `code_fix`, `toolcall` (reuse sandbox).
- [ ] **WI-4.2 Verifier tests:** for every family ≥ 20 hand-written (completion → expected verdict) pairs including
  adversarial near-misses; property test: `verify(generate(...), reference_solution) == pass` for 500 seeds.
  `code_fix` mutation operators each produce a program that **fails** the original tests (assert it).
- [ ] **WI-4.3 Split + freeze + decontamination (§7.3–7.4)** wired into the dataset freeze step; planted-duplicate test.
- [ ] **WI-4.4 Suites registered:** `gen_fmt`, `gen_arith`, `gen_extract`, `gen_code_synth`, `gen_code_fix`,
  `gen_toolcall` (`n` configurable, default 200), labelled `project-specific` (never leaderboard). Guard suite
  `guard_general` built from non-target families + the secrecy case.
- [ ] **WI-4.5 Baseline sweep:** run all suites for 0.6B and 1.7B Q4_K_M, k=1 greedy and k=4 T=0.7; write
  `docs/evidence/v8_baselines.md` with per-family pass rates + Wilson CIs, measured discordance between two base
  runs (this calibrates `d` for `required_n`). **If a family is at ~0 % or ~100 % for the base, it cannot show
  improvement — adjust difficulty until pass rate is 20–70 %.**

**LG-4:** generators deterministic per seed; splits disjoint by template (test asserts); baselines recorded.

### Phase 5 — Flywheel and the first measured result (L) — fixes G3, G12. Depends: Phases 2–4

- [ ] **WI-5.1 `methods/rft.py`:** implements §8 steps 1–3 using `llama_server` + `tasks.verify`; writes a
  provenance-tagged dataset (`annotation_origin="self_verified"`, `synthetic: true`, teacher=self, round, seed);
  DAPO-style group filter implemented via `losses.dapo_filter_groups`; unit test with a fake sampler.
- [ ] **WI-5.2 Trainer `rft`** registered in the contract: sampling stage + the SFT adapter; manifest `extras` holds
  per-round sample counts, pass rates, kept/dropped numbers.
- [ ] **WI-5.3 LoRA scale in merge:** `merge_adapter(..., scale: float = 1.0)` and `merge_int4_adapter(..., scale=1.0)`;
  test: `scale=0` reproduces the base weights exactly; `scale=1` equals current behaviour (golden test).
- [ ] **WI-5.4 Upgrade Lab `adapter` candidates + gate (§8)**; lineage rows; rollback test restores previous GGUF/profile.
  Per-item outcomes stored; `test_reused` flag implemented and tested.
- [ ] **WI-5.5 `foundry flywheel run --model … --rounds 3 --families …`** orchestrating rounds with checkpoints between
  stages (resume after interruption), cancel-safe, writing `reports/flywheel_<id>.json`. Also an API job + MCP later.
- [ ] **WI-5.6 The experiment (the result of the whole release):** Qwen3-0.6B first, then 1.7B if time allows.
  Protocol fixed *before* running and committed as `docs/evidence/v8_flywheel_protocol.md` (families, n, k, rounds,
  LR sweep {1e-4, 2e-4, 5e-4} with rank fixed, scale sweep, baselines b0–b3, decision rule from §5 A4). Then run,
  then write `docs/evidence/v8_flywheel.md` with tables: per-family pass rate, paired Δ + CI + p, guard Δ, wall time,
  tokens, RAM. **Report the outcome whatever it is.** If negative, include the diagnosis (data size? difficulty band?
  LR? model capacity?) and the next experiment — not a spin.

**LG-5:** protocol file committed before the first run (git history proves it); evidence file complete; gate decisions
reproducible from stored per-item outcomes (`foundry eval compare` regenerates the numbers).

### Phase 6 — On-policy distillation and teacher log-probs (M). Depends: Phase 2. Optional if time is short

- [ ] **WI-6.1 Logprobs in the runtime client** (G9): request `n_probs`/`logprobs` from `llama-server`
  (`/completion` supports `n_probs` per token; verify against the installed build's API before coding), parse into
  `[(token_id, logprob)]`, test with a recorded response fixture + one real local smoke (`@local`).
- [ ] **WI-6.2 `methods/onpolicy_distill.py`:** student samples (torch, short — 64–128 new tokens), teacher scores via
  WI-6.1 on the *student's* token ids (requires identical tokenizer: assert vocab/merges hash equality, else refuse),
  loss = `losses.reverse_kl_loss`. Conformance C1–C12 on the tiny fixture with a *fake teacher* (a second random model
  scored in-process) so CI needs no llama-server.
- [ ] **WI-6.3 Experiment:** 4B-Q4 teacher → 0.6B/1.7B student on one family; same protocol discipline as WI-5.6.
  Compare cost-to-target vs RFT. Write `docs/evidence/v8_distill.md`.

### Phase 7 — Engine experiments (M) — fixes G10. Machine-exclusive (§1.2). Depends: Phase 2

- [ ] **WI-7.1** Extend `scripts/store_io_bench.py`: cold/warm switch, per-layer read timing, bandwidth probe function
  reusable from the service (`measure_seq_read_bandwidth(path, nbytes=256<<20) -> float`).
- [ ] **WI-7.2** Record `io` in the manifest: `passes`, `bytes_read_store`, `seq_read_mbps`, `micro_batch`.
- [ ] **WI-7.3** Run E1…E4 per §9 (3 repeats each), implement only what wins behind a spec field
  (`method_args.layer_major`, `hyper.micro_batch`), keep the old path as default until E-results are in.
- [ ] **WI-7.4** E5 + `Preflight.est_step_seconds`; doctor prints range; refusal when > `Budget.max_wall_s`.
- [ ] **WI-7.5** `docs/evidence/v8_engine.md` incl. the losing experiments; update `TRAINING.md` hyper-parameter guidance
  ("raise micro_batch before grad_accum when the store is disk-backed") **only if E1 confirmed it**.

### Phase 8 — Quantization and KV-cache gates (S–M). Depends: WI-6.1 (logprobs) or `llama-perplexity` flags

- [ ] **WI-8.1 Quant gate:** after `gguf quantize`, compute KL(F16‖Q) and top-1 flip rate on a fixed 200-prompt set
  (reuse `tasks` prompts) using `llama-perplexity --kl-divergence` if the build supports it (verify; else via logprobs);
  store in the pipeline report; threshold configurable, default *report only* until calibrated on 3 quant levels.
- [ ] **WI-8.2 imatrix option** for ≤ 3-bit: calibration text = train-split prompts only (no eval leakage); test that the
  calibration file is derived from `train` rows only.
- [ ] **WI-8.3 KV-cache sweep:** `{f16,q8_0,q4_0} × ctx {2k,8k,16k}` on the 1.7B/4B Q4_K_M: KV bytes/token, TTFT, tok/s,
  needle-retrieval accuracy (generated, paired vs f16). `docs/evidence/v8_quant.md`. TurboQuant stays a probe (§3.10);
  re-check upstream status on the day and record the date.
- [ ] **WI-8.4 Speculative decoding (optional):** 0.6B as draft for 1.7B/4B via `--model-draft`; measure acceptance and
  tok/s; ship as a preset only if tok/s ↑ ≥ 20 % with identical outputs at T=0 (check equality on 50 prompts).

### Phase 9 — MCP, CLI, UI surface (M). Depends: Phases 2, 3, 5

- [ ] **WI-9.1 MCP tools:** `train_plan`, `train_run`, `models_run`, `bench_run`, `eval_compare`, `flywheel_status`.
  Same rules as v7: typed schemas, path allowlist, no shell; `train_run`/`flywheel` require a `confirm: true`
  argument and print the preflight. Tests incl. path-traversal rejection and schema listing (extend `test_mcp.py`).
- [ ] **WI-9.2 UI:** Training Lab driven by the spec schema (WI 2.7); Benchmarks compare view with CI/p/n badges;
  Upgrade Lab shows adapter candidates with the baseline table (b0–b3), guard result, and a plain-language verdict
  ("not enough data to tell", "improved (CI …)", "no evidence of improvement"). All strings de/en/fr; parity test green.
  Playwright: preflight shows step-time range; underpowered badge visible; no dead controls.
- [ ] **WI-9.3 Optional Inspect bridge** (§3.12): `foundry eval inspect --task … ` behind an extra, marked "unsupported" when
  its sandbox prerequisites are missing. Skip without guilt if Phase 5 overran.

### Phase 10 — Docs, site, release (M)

- [ ] **WI-10.1 Docs:** `TRAINING.md` (unified contract, profiles, hyper-parameter guidance with citations),
  `BENCHMARKS.md` (generators, stats, how to read a CI, why n matters), `TUTORIAL.md` new walkthrough "measure an
  improvement", `UI_GUIDE.md`, `MCP.md`, `CLI.md`, `ARCHITECTURE.md`, ADRs for: unified contract, stats gate,
  flywheel, base-model choice, TurboQuant policy. README: honest capability table — "verified" only where A-criteria
  say so; roadmap rewritten.
- [ ] **WI-10.2 Site:** landing page + generated changelog; "What we measured" section from the evidence tables
  (numbers copied from evidence files by a script, not retyped). No new external requests, size budget as in V7 P6.
- [ ] **WI-10.3 Release:** PG-3 (§4.4) end to end; `docs/evidence/RELEASE_REPORT_V8.md` with a "Not claimed" section.

### Phase 11 — Remote backend (design only; no build without user approval)

- [ ] **WI-11.1 ADR + interface:** `Backend.REMOTE` adapter contract (`submit(spec, bundle_hash) -> RemoteJob`, `poll`,
  `fetch`, `cancel`), consent record (what leaves the machine, hash of the bundle, cost cap, provider), mock server
  tests. Real providers (the user has RunPod/Hugging Face tooling available) only after explicit approval of cost,
  credentials and data upload for a *named* run. GRPO/DAPO/GSPO real runs live here, using `losses.py` unchanged.

---

## 11. Dependency graph and critical path

```text
P0 → P1 → P2 ─┬─ P3 ──► [PG-1 v7.1.0]
              │          │
              │          ├─ P4 ─► P5 ─► [PG-2 v7.2.0]
              ├─ P6 (opt)┘          
              └─ P7 (exclusive) ──┐
 P8 (needs logprobs or flags) ────┤
 P9 (needs P2,P3,P5) ─────────────┴─► P10 ─► [PG-3 v8.0.0]       P11: design only
```

Critical path: **P0 → P1 → P2 → P3 → P4 → P5 → P10.** If time is short, cut in this order: P9.3, P8.4, P6, P8.3, P7
(keep WI-7.4's predictor if E1/E5 are cheap). **Never cut:** P1 (CI honesty), P3 (stats), WI-5.6 (the measured result,
positive or negative), P10 docs honesty.

## 12. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tiny-model CI tests are flaky (nondeterminism, thread races) | M | single-thread in tests, `torch.use_deterministic_algorithms(True, warn_only=True)`, calibrated margins for C3, no wall-clock assertions except C6 (generous) |
| CI time/size explodes with torch | M | CPU wheels, pip cache, `-m "not slow"` split, nightly job for slow checks |
| Flywheel shows no gain on 0.6B | M–H | that is allowed (A4); the protocol and baselines make the negative result informative; try 1.7B and LR/scale sweeps once, then stop |
| Base pass rate at 0 %/100 % in a family | M | calibrate difficulty in WI-4.5 before training |
| Reward hacking in code families (tests visible to model, sandbox leaks) | M | hidden tests only; `git clean`/fresh temp dir per item (SWE-Gym warning); timeouts; no network in sandbox |
| Evaluation leakage via repeated test reuse | M | `test_reused` flag; rotating fresh test set per release |
| Machine contention corrupts timings | H | §1.2; free-RAM guard in the bench scripts that aborts if < threshold |
| Windows-specific I/O quirks (no `os.pread`, mmap behaviour) | M | E4 uses `io.FileIO.readinto`; test on this host only; claim nothing for other OSes |
| Dependency drift breaks the lock | L | lock + CI installs from it; Dependabot-style manual refresh each release |
| Secondary-source facts in §3 are wrong/outdated | M | each is re-verified against the primary source when implemented; wrong ones are corrected in this file in the same commit |
| Push/CI/deploy credentials missing | M | stop at the gate, report exact commands (`AGENTS.md` §17.3, §0.1) |

## 13. Reporting contract (what the final message of each phase must contain)

1. What changed (files, behaviours), 2. evidence paths + the numbers, 3. test counts incl. skips (must be 0 training
skips from Phase 1 on), 4. what was **not** done or **not** verified, 5. the next gate and its status. Use the product's
vocabulary (`AGENTS.md` §19): *detected ≠ tested*, *trained ≠ improved*, *built ≠ pushed ≠ deployed*.

## 14. Reference list (starting points; re-verify before citing)

- LoRA Without Regret — <https://thinkingmachines.ai/blog/lora/> · <https://huggingface.co/docs/trl/en/lora_without_regret>
- LoRA Learns Less and Forgets Less — arXiv 2405.09673
- Adding Error Bars to Evals — arXiv 2411.00640
- Tülu 3 — arXiv 2411.15124 · <https://allenai.org/blog/tulu-3-technical>
- DAPO — arXiv 2503.14476 · Dr. GRPO — arXiv 2503.20783 · GSPO — arXiv 2507.18071 · GRPO++ overview — <https://cameronrwolfe.substack.com/p/grpo-tricks>
- HF blog, RL on outcomes 2026 — <https://huggingface.co/blog/sergiopaniego/agentic-rl-2026>
- On-Policy Distillation — <https://thinkingmachines.ai/blog/on-policy-distillation/>
- SWE-Gym — arXiv 2412.21139 · R2E-Gym — arXiv 2504.07164 · OpenAI on SWE-bench Verified — <https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/>
- ZeRO-Infinity — arXiv 2104.07857 · DeepSpeed ZeRO-Inference — <https://www.deepspeed.ai/2022/09/09/zero-inference.html> · LISA — arXiv 2403.17919
- Quantization: llama.cpp discussion #4110; arXiv 2601.14277; Qwen llama.cpp quantization docs
- TurboQuant: arXiv 2504.19874; llama.cpp discussion #20969; status summary <https://www.morphllm.com/turboquant>
- Inspect AI — <https://inspect.aisi.org.uk/>
- Magpie — arXiv 2406.08464 · LIMA — arXiv 2305.11206 · Model merging survey — arXiv 2603.09938
- Local evidence: `docs/evidence/v7_4b_speed.md`, `v6_2b.md`, `v6_vram.md`, `v41_int4_4b.md`, `RELEASE_REPORT_V7.md`

---

## 15. Progress log (update in the same commit as the work)

| Phase | Status | Evidence | Notes |
|---|---|---|---|
| P0 baseline/hygiene | **done** | docs/evidence/v8_baseline.md, server/requirements-*-lock.txt, web/dist untracked |
| P1 CI honesty | **done** | CI installs 
equirements-train-lock.txt; FOUNDRY_REQUIRE_TRAIN=1 + 	ests/conftest.py guard + 	est_ci_guard.py |
| P2 unified contract | open | | |
| P3 stats gate | open | | |
| PG-1 v7.1.0 | open | | |
| P4 generators | open | | |
| P5 flywheel + result | open | | |
| P6 on-policy distill | open | | |
| PG-2 v7.2.0 | open | | |
| P7 engine experiments | open | | |
| P8 quant/KV gates | open | | |
| P9 MCP/UI | open | | |
| P10 docs/site/release | open | | |
| PG-3 v8.0.0 | open | | |
| P11 remote (design) | open | | |
