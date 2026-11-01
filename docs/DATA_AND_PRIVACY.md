# Data and privacy

## Scope of an import

An import reads **only** the files and folders you select, and only if they are inside a configured import root. Default roots are proposed from standard locations:

- `<repo>/data/imports`
- `~/.claude/projects`
- `~/.codex/sessions`
- `~/.local/share/opencode`

You can change them in Settings. The service never scans your home directory, browser profile, SSH keys or credential stores.

## Supported sources

| Provider | Format | Adapter version |
|---|---|---|
| Claude Code | JSONL session files | 1.0.0 |
| Codex CLI | JSONL session files | 1.0.0 |
| OpenCode | JSON export | 1.0.0 |
| Antigravity | tolerant JSON/JSONL/Markdown | 1.0.0 |
| Generic | JSON, JSONL, Markdown | 1.0.0 |

Provider schemas are internal and can change. Adapters tolerate unknown fields, count malformed lines, and never abort a whole import for one bad file. The import result reports `parse_errors` and `skipped_lines`.

## Consent and rights

Before an import can be frozen or exported you must set a rights status and confirm:

- you own the content or have permission to use it,
- the provider terms allow this processing,
- it contains no third-party confidential data without a lawful basis.

Statuses: `confirmed`, `needs_review`, `restricted`, `excluded`. Only `confirmed` datasets can be frozen, exported, or used to build an upgrade candidate.

## Redaction

Applied before records are stored, and also to previews:

- private key blocks
- provider API keys (`sk-`, `sk-ant-`, `ghp_`, `github_pat_`, `AKIA`, `xox*`, `AIza`, `hf_`)
- JWTs, bearer tokens, connection strings
- `password=…` / `api_key: …` assignments
- email addresses, phone numbers, card-like digit runs
- home paths (`C:\Users\<name>`, `/home/<name>`)

High-entropy strings are **flagged, not silently removed**, because they can be legitimate hashes in code. Review the redaction summary before training.

Redaction reduces risk; it is not a guarantee. Human review is required.

## Hidden reasoning

HCS Foundry does **not** request, scrape, reconstruct or fabricate hidden chain-of-thought from any provider.

Teaching annotations are:

- `decision_summary` — a short statement quoting observable evidence (tool output, test result, diff),
- `method_summary` — a high-level description of the observed approach,
- `negative_example` — a plausible mistake and the rule that prevents it.

Every synthetic row carries `synthetic: true`, the teacher, template version, source record ids and hashes, and a validation result. An annotation that quotes text not present in the trace is rejected and the row is labelled `unknown`.

## Deduplication and splits

- Exact duplicates: canonical record hash.
- Near duplicates: 64-bit SimHash, Hamming distance ≤ 3 (configurable).
- Splits: 80 / 10 / 10 by hashing the session or task group, so a conversation never crosses partitions.

## Freezing

Freezing writes a manifest with the dataset hash, split counts, synthetic count and holdout hash. If no record lands in the test split, the manifest records `warning: no_test_split` and the upgrade lab refuses to build a candidate from it.

## Export

`GET /api/v1/datasets/{id}/export` returns JSONL of included, non-excluded records, and only when rights are `confirmed`. Exports stay on your machine unless you move them.

## Retention and deletion

- Raw source files are never copied into the repository and never modified.
- Deleting a dataset removes its records from SQLite; the original files remain.
- Logs are local, git-ignored, and contain no secrets by design (but may contain model output).

## What is never transmitted

No analytics, no crash uploads, no remote inference, no model or dataset upload. The only network traffic the app initiates is what you explicitly trigger (for example installing dependencies yourself).
