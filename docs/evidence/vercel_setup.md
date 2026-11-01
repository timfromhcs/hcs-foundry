# Vercel setup — verified state

**Date:** 2026-11-01
**Playbook:** <https://vercel.com/get-started.md> (fetched and followed)

## 1. CLI

```
npm install --global vercel@latest
vercel --version  -> 65.9.0
VERCEL_TOKEN=... vercel whoami -> timfromhcs
```

## 2. Guidance and MCP

```
opencode plugin add github:vercel/vercel-plugin
  -> installed, added to C:\Users\hcsme\.config\opencode\opencode.jsonc
opencode mcp add vercel --global --url https://mcp.vercel.com
  -> added to the same config (status: pending until the client reloads)
```

The plugin also installed the Vercel skill pack (ai-gateway, vercel-cli, deployments, etc.).

## 3. AI Gateway

`npx vercel@latest ai-gateway setup` could **not** complete. The reason is precise:

```
vercel ai-gateway setup --yes --no-session-migration
  -> login_required: "The token in VERCEL_TOKEN is not valid."
```

The provided credential has the `vck_` prefix, which is a **Vercel AI Gateway API key**,
not a Vercel platform token. Evidence:

| Probe | Result |
|---|---|
| `GET https://api.vercel.com/v2/user` | 200, user `timfromhcs` |
| `GET https://api.vercel.com/v2/teams` | 403 `forbidden` |
| `GET https://api.vercel.com/v2/user/tokens` | 403 `forbidden` |
| `GET https://ai-gateway.vercel.sh/v1/models` | 200, **498 models** |
| `POST https://ai-gateway.vercel.sh/v1/chat/completions` | 403 “AI Gateway requires a valid credit card on file” |

So the key is valid for the gateway API but the account has no billing method, and it
cannot be used to create gateway keys or manage teams.

## 4. What was wired up anyway

HCS Foundry now has an **opt-in** Vercel AI Gateway teacher:

- key stored locally in `data/state/vercel-gateway.json` (git-ignored, masked in the API),
- 80 models listed through the gateway,
- disabled by default,
- only teaching annotations may use it; chat, training and benchmarks stay local,
- the 403 billing answer is surfaced verbatim in the UI.

The key was stored locally, **not** committed, and should be rotated because it was
pasted into a chat message.

## 5. Blocked steps

| Step | Blocker | Exact next action |
|---|---|---|
| `vercel ai-gateway setup` | needs a platform token, not a gateway key | create a token at <https://vercel.com/account/tokens> and export it as `VERCEL_TOKEN`, or run `vercel login` |
| Gateway inference | account has no payment method | add one at <https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai> |
| Frontend deployment | no linked project and no platform token | `cd web && vercel --prod` after login; set `VITE_FOUNDRY_API` |

Nothing was claimed as deployed.
