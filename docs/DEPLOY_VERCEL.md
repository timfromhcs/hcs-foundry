# Deploying the frontend

HCS Foundry is local-first. The hosted deployment contains **only the static SPA**. It never runs inference, training, imports or benchmarks on a serverless function.

## Why the backend is not deployed

- Model inference and training need persistent, machine-specific hardware.
- Serverless functions have duration, memory and package limits.
- Routing user content through a hosted backend would break the privacy promise.

## Build the static frontend

```bash
cd web
VITE_FOUNDRY_API=http://127.0.0.1:8770/api/v1 npm run build
```

`VITE_FOUNDRY_API` must point at the user's local service. If omitted, the SPA uses a relative `/api/v1`, which only works when the service serves the build itself.

## Deploy

```bash
cd web
npx vercel --prod
```

Set `VITE_FOUNDRY_API` in the Vercel project's environment variables. Do not commit it.

## Pair the deployed origin with the local service

1. Note the deployment origin, for example `https://hcs-foundry.vercel.app`.
2. Start the service with that origin allowlisted:

```powershell
$env:FOUNDRY_UI_ORIGINS="https://hcs-foundry.vercel.app"
python -m foundry.cli serve
```

3. Open the deployment, press **Connect to local service**, and pair.

## Browser reality check

An HTTPS page calling `http://127.0.0.1:8770` is subject to mixed-content rules. Chromium treats loopback as a trustworthy origin, but Safari and some configurations block it. If the browser refuses the connection, the reliable fallback is to open the service directly at `http://127.0.0.1:8770`, which serves the same build.

This is documented rather than hidden: a deployed SPA that cannot reach the local service shows the "backend unreachable" state, not a fake dashboard.

## Security checklist before deploying

- [ ] No `.env` or token committed.
- [ ] `VITE_FOUNDRY_API` set only in the platform.
- [ ] The local service allowlists exactly the deployment origin.
- [ ] Pairing works from the deployed origin.
- [ ] The offline/disconnected state renders.
- [ ] A rollback deployment is available.

## Status of this release

The Vercel deployment step requires credentials for a Vercel account. None were available in the build environment, so **no deployment was performed**. The build artifact is verified locally (`web/dist`, 418 kB JS / 121 kB gzip).
