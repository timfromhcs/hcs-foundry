# Benchmarks

All scores produced by HCS Foundry are **project-specific smoke** results. They are not official leaderboard numbers and must not be presented as such.

## Suites in 0.1.0

### 1. Instruction adherence (deterministic)

Six cases with deterministic assertions:

| Case | Assertion |
|---|---|
| `exact_token` | reply is exactly `PONG` |
| `json_only` | reply parses as `{"ok": true, "count": 3}` |
| `german` | reply is detected as German |
| `system_prompt_secrecy` | reply does not contain the system prompt |
| `counting` | reply contains `1, 2, 3, 4, 5` |
| `arithmetic` | reply contains `42` |

### 2. Code generation (executed)

Four small Python tasks. The generated code is extracted from a fenced block, executed in a subprocess with a 15 s timeout inside a temporary directory, and checked against unit tests. A failure records the assertion or exception.

### 3. Latency & throughput (measurement)

Three prompts; records time to first token, completion tokens, tokens/sec and wall time. Reported as a **measurement**, not pass/fail. On the tested machine (2B Q4_K_M, Vulkan): ~62 tokens/s and ~0.8 s TTFT for a short reply.

### 4. UI regression (external)

Playwright from `web/`: routes, keyboard focus, responsive overflow, reduced motion, language switch, theme persistence. The backend reports it as *not runnable here* unless Playwright is installed in `web/`.

## Comparability rules

- Every run stores model id and hash, suite version, sampling settings, hardware profile and comparability label.
- Comparing runs from different suites shows per-case deltas only and marks `fair: false`.
- An LLM judge is stored under `aggregate.judge` as a secondary signal. It can never change a deterministic pass/fail.

## Running

```bash
# from the UI: Benchmarks → Run suite
# or via API
curl -X POST http://127.0.0.1:8770/api/v1/benchmarks/runs \
  -H "Authorization: Bearer $TOKEN" -H "X-Foundry-Client: 1" \
  -H "Content-Type: application/json" \
  -d '{"suite_id":"instruction_adherence","judge":false}'
```

## Reports

`GET /api/v1/benchmarks/runs/{id}/export?format=json|csv|md` returns a report with run id, model hash, suite version, cases and the comparability note.

## Not bundled

SWE-bench, LiveCodeBench, Terminal-Bench, GAIA, WebArena, OSWorld, MMMU and ScreenSpot require assets, isolation and licensing that are not shipped. The UI lists them as *not run* with a link to the official instructions rather than faking a local subset score.
