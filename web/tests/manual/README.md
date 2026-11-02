# Manual browser checks

These scripts are not part of `npx playwright test` (they need a running service and, for the
public one, a deployed site). They exist so the evidence in `docs/evidence/v5_loop.md` can be
reproduced.

```bash
# 1. local service + the landing page served from a loopback origin
cd web && npx http-server ../site -p 4180 --bind 127.0.0.1   # or: python -m http.server 4180
#    start the service with that origin allowlisted:
#    set FOUNDRY_UI_ORIGINS=http://127.0.0.1:4180,http://127.0.0.1:8770
#    python -m foundry.cli serve --port 8770
node tests/manual/live-pair-local.cjs     # expect "Paired. Token received for this tab only."

# 2. the deployed page from its public HTTPS origin
node tests/manual/live-pair-public.cjs    # expect the Local Network Access block message
```

The second check documents a browser limit, not a product feature: Chromium refuses a public
page's request to `127.0.0.1` unless the user grants Local Network Access. The service answers
the Private Network Access preflight, but the permission stays the browser's decision.
