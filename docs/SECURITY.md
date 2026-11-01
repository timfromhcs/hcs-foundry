# Security

## Threat model

The service is privileged: it spawns processes and reads local files. It listens on a TCP port. The realistic attackers are:

1. A malicious web page the user visits (CSRF, DNS rebinding, CORS abuse).
2. Another local process with the same user account (already has equivalent access, but should not be able to silently drive the app).
3. A malicious model file or imported transcript (untrusted input).

## Controls

### Loopback binding

Default `127.0.0.1:8770`. Binding elsewhere requires `FOUNDRY_ALLOW_LAN=1`, and the CLI refuses to start on a non-loopback host without it.

### Pairing

`POST /api/v1/system/pair` mints a `fnd_…` token. Tokens are stored **hashed** (SHA-256) in `data/state/pairing.json`. The browser keeps the token in `sessionStorage`, never in the bundle. `POST /api/v1/system/revoke` invalidates all tokens.

Optional pairing codes: set `require_pairing_code` in the store; the code is written to `data/state/pairing-code.txt` and shown in the service terminal.

### Origin and Host validation

- A present `Origin` must exactly match the allowlist (dev server, preview server, service origin, plus explicitly configured origins).
- `Host` must be a loopback name. This defeats DNS rebinding.
- `Access-Control-Allow-Origin: *` is never emitted.

### CSRF

Mutating requests require:

- `X-Foundry-Client: 1` (a header a cross-site form cannot set without a successful preflight), and
- `Content-Type: application/json`.

### Path allowlist

Every filesystem read resolves the path and requires it to be inside configured roots (`model_roots`, `import_roots`, `data_dir`). Symlinks are resolved before the check. `tests/test_security.py::test_path_allowlist_rejects_traversal` covers this.

### No generic shell

There is no endpoint that accepts an arbitrary command. The agent exposes named tools with typed arguments; only read-only git and file inspection run without approval. `run_benchmark` is marked mutating and requires approval under supervised autonomy.

### Untrusted input

- GGUF parsing reads only the header, with hard limits on string/array/tensor counts.
- Imported transcripts are data, never instructions.
- Generated benchmark code runs in a subprocess with a timeout and a temp directory. **This is not a VM or container.** Do not treat it as a hardened sandbox for untrusted repositories.

### Secrets

- No `.env` files are committed; `.gitignore` covers keys, weights, datasets and logs.
- Redaction runs before records are stored; previews are redacted too.
- Logs contain job metadata and child process output. They can contain model output; they are local and git-ignored.

## What is deliberately not implemented

- LAN access, remote access, multi-user accounts.
- Cloud inference or telemetry.
- Automatic model downloads.
- Automatic agent shell execution.
- Auto-promotion of upgrade candidates.

## Reporting a vulnerability

Open a private security advisory on the repository or contact the maintainers directly. Do not file a public issue for an unpatched vulnerability.

## Verification

```bash
cd server && python -m pytest tests/test_security.py -q
```

The suite asserts: token required, foreign origin rejected, foreign host rejected, client header required, JSON content type required, traversal rejected, tokens hashed and revocable.
