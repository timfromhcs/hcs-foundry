# Repository Audit — HCS Foundry

**Audit date:** 2026-11-01
**Auditor:** autonomous engineering agent
**Repository root:** `E:\AILOCALTRAIN`
**Status:** Gate A (audit) — complete for the initial build

## 1. Observed facts

| Item | Value | Evidence class |
|---|---|---|
| Repository content | Exactly one file: `AGENTS.md` (85,745 bytes) | Verified |
| Git repository | None (`Is directory a git repo: no`) | Verified |
| Existing frontend | None | Verified |
| Existing backend | None | Verified |
| Existing tests | None | Verified |
| Existing CI | None | Verified |
| Existing docs | None except `AGENTS.md` | Verified |
| Existing lockfiles | None | Verified |
| Existing secrets in tree | None (single Markdown file) | Verified |
| Node toolchain | Node v22.23.1, npm 10.9.8 | Verified |
| Python toolchain | Python 3.14.6 default; 3.13/3.12/3.11/3.10 also present | Verified |
| Git | 2.54.0.windows.1 | Verified |
| GitHub CLI | `gh` 2.x, authenticated as `timfromhcs` (token scopes include `repo`, `workflow`) | Verified |
| Vercel CLI | Not installed | Verified |
| Docker | 29.6.1 present (Docker Desktop) | Verified |

## 2. Consequence

Because the repository is empty, there is no existing stack to preserve and no legacy behaviour to protect. AGENTS.md §2.1 ("use existing tools and conventions when sound") is therefore satisfied by choosing a stack explicitly rather than by inheriting one. The chosen stack is documented in `docs/architecture/DECISIONS.md` (ADR-001 … ADR-012).

## 3. Existing local AI assets on this machine (reused, not copied into the repo)

These were discovered during the machine audit and are relevant because the product must run real inference on day one.

| Asset | Path | Notes |
|---|---|---|
| llama.cpp Vulkan server (build 11064, `0.4.1-dev`) | `D:\AI\bin\llama-vulkan\llama-server.exe` | Verified working with `-ngl 99` |
| llama.cpp Vulkan server (build 10977) | `D:\HCS Chat\runtime\llama.cpp\vulkan\llama-server.exe` | Alternative |
| llama.cpp CPU server | `D:\HCS Chat\runtime\llama.cpp\cpu\llama-server.exe` | CPU fallback |
| Additional llama-server builds | `D:\AI Serving Stack\…`, `D:\PrimeAgent\runtime\llama-vulkan`, `E:\AIMinecarftLearn\llama-turboquant-windows-x64-vulkan`, `E:\Bonsaiserver\Bonsai-demo\bin\vulkan`, `D:\AiMovieMaker\bin\llama-cpp`, `D:\hcsimagegen\bin\llama.cpp` | Discovered, not all validated |
| Local GGUF library | `D:\aimodels\**` | ≥ 25 GGUF files observed, sizes 0.11–6.87 GB |
| Ollama model store | `C:\Users\hcsme\.ollama\models` | 13.1 GB + 6.87 GB blobs present; no `ollama` CLI on PATH |

The application **discovers** these at runtime through the model-library scanner and the runtime-probe; it never hard-codes them and never copies weights into the repository.

## 4. Security posture of the starting point

The starting point has no attack surface beyond a Markdown file. All security design is therefore new work and is specified in `docs/SECURITY.md` and ADR-004/ADR-005:

- loopback-only bind, explicit pairing token, strict `Origin`/`Host` validation,
- no generic "run arbitrary command" endpoint,
- path allowlists for every filesystem operation,
- secrets/PII redaction before anything enters a dataset,
- no telemetry, no cloud calls, no LAN scanning.

## 5. Gaps and risks carried into the build

| Risk | Severity | Mitigation |
|---|---|---|
| Single 16-thread APU with shared memory: a large context can starve the desktop | High | Resource governor + conservative UMA budget + pre-load re-measure |
| `torch` is CPU-only (2.12.1+cpu); no ROCm/CUDA training path | High | Truthful "CPU experimental" training tier; tiny smoke test only; no fake GPU training claims |
| `transformers`/`trl`/`peft` not installed | Medium | Preflight detects and reports; install is an explicit user action |
| Python 3.14 is very new; some ML wheels lag | Medium | Backend pins only packages that import cleanly on 3.14 (verified: fastapi, uvicorn, pydantic, httpx, psutil, numpy, datasets, safetensors) |
| GGUF files from community repos have unknown provenance/licenses | Medium | Model card fields are user-editable; license shown as "unknown" until supplied; never auto-download |
| No Vercel CLI/credentials at audit time | Low | Deployment is the last phase and will be reported honestly if credentials are missing |

## 6. Gate A verdict

**Passed.** Machine and repository facts are recorded, the stack decision is justified by observed constraints, and the critical risks are known and mitigated by design rather than optimism.
