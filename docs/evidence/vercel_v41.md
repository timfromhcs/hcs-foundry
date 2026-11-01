# v4.1 evidence — stable Vercel deployment of the landing site

The v4.0 release shipped `site/` but only deployed it as a temporary deployment
(`https://agile-jasper-pcdb33c.tmp.now`), which is not a stable URL and was still linked
from the README roadmap as "permanent deployment pending".

## What was done

```
vercel link --project hcs-foundry-site          # project created in scope hcsmediacorp
vercel project update hcs-foundry-site --root-directory site --framework other
vercel deploy --prod                            # from the repo root
```

A `.vercelignore` now restricts the upload to `site/` so model weights, datasets,
training runs and the local service never enter a deployment bundle.

## Verified

```
GET https://hcs-foundry-site.vercel.app/                      200  27899 bytes
    Content-Type: text/html; charset=utf-8
    Permissions-Policy: camera=(), microphone=(), geolocation=()
    Referrer-Policy: strict-origin-when-cross-origin
GET https://hcs-foundry-site.vercel.app/assets/hub.png        200
GET https://hcs-foundry-site.vercel.app/assets/pipeline.png   200
<title>HCS Foundry — local AI model lab</title>
```

| Item | Value |
|---|---|
| Project | `hcsmediacorp/hcs-foundry-site` (`prj_wMn2gkhoyeHBEwZYGFm1vKQ9PvML`) |
| Root directory | `site` |
| Production URL | <https://hcs-foundry-site.vercel.app> |
| Deployment | `dpl_Cu3XgcXu22yWQ8yn7Uk3wbAQkQsN` (`READY`, target `production`) |
| Build | static, 2 cores / 8 GB, no serverless functions |
| Git | connected to `timfromhcs/hcs-foundry`, so later pushes rebuild the landing page |

## Boundary kept

The deployment is a **static page only**. There is no Vercel function, no inference, no
training and no local data in the bundle. The local service still binds to `127.0.0.1`
and requires pairing; a deployed page cannot reach it without the explicit pairing flow.
