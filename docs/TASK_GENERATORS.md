# Verifiable task generators

The point of these generators is that **the answer is checkable without a judge**. A benchmark that
asks a model to grade itself cannot support a quality claim; a generator with a verifier can.

```bash
foundry tasks generate --family all --seed 1 --per-template 10 --out data/exports/tasks.jsonl
foundry tasks check data/exports/tasks.jsonl
```

Exit code 1 means the data is not ready for a claim (too few held-out items, or a template leaked
across splits).

## Families

| Family | Templates | Verifier |
|---|---|---|
| `arith` | `arith_add`, `arith_mul`, `arith_percent` | exact numeric match after removing grouping separators; trailing prose is rejected |
| `fmt` | `fmt_csv_to_json`, `fmt_json_to_csv`, `fmt_sorted_list` | JSON deep-equality, CSV line equality, or list equality after whitespace normalisation |
| `extract` | `extract_invoice`, `extract_log` | JSON object equality, including field types (`"12"` ≠ `12`) |
| `toolcall` | `tool_weather`, `tool_calendar` | exactly `{tool, arguments}`, same tool name, same argument keys, same values |

Every item is a `TaskItem` with `task_id`, `family`, `template_id`, `split`, `prompt`, `answer`,
`meta`. Generation is deterministic: the same `--seed` produces byte-identical JSONL.

## Splits and leakage

Splits are assigned **per template**, never per row:

```text
templates, ranked by sha256(template)
  first 60 %  -> train
  next  20 %  -> dev
  last  20 %  -> test
```

A test template therefore cannot appear in training — the leak is impossible by construction rather
than by a filter that might silently miss one. `check_template_isolation()` returns every template
that appears in more than one split; a test proves it detects a hand-made violation.

The current map (10 templates) is:

```text
train  fmt_json_to_csv, fmt_csv_to_json, fmt_sorted_list, tool_calendar, tool_weather, extract_log
dev    arith_percent, arith_mul
test   arith_add, extract_invoice
```

## Readiness gate

`readiness()` refuses a dataset that cannot support a paired claim:

- `train >= 60`, `dev >= 20`, `test >= 20` items (defaults, adjustable)
- no template in two splits

With 10 items per template the full set is 100 rows: 60 / 20 / 20 — exactly at the gate. Anything
smaller is reported as blocked with the numbers, not quietly accepted.

## Decontamination

`decontaminate(items, forbidden_prompts)` drops items whose normalised prompt already exists in an
evaluation set and returns `(kept, dropped)`. The drop count is reported; a silent removal would
make a score look better than it is.

## What this is not

- **Not a leaderboard.** These are project-specific tasks with deterministic verifiers, not
  HumanEval/MMLU.
- **Two families are still missing** from the plan: `code_synth` and `code_fix` (both need the
  subprocess test runner wired into the generator API). `toolcall` covers the structured-output
  case in the meantime.
- **No claim yet.** These generators make a measured run possible; `docs/plan/V8_STABLE_PLAN.md`
  S5 is where a base-vs-candidate comparison has to happen.
