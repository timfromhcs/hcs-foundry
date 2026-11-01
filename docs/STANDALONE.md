# Standalone executable

HCS Foundry ships as a single Windows executable that contains the API, the built SPA and the Python runtime.

## Build

```powershell
python -m pip install pyinstaller
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1
```

The script builds the frontend, then runs PyInstaller with the SPA bundled as `web/dist`.

Output: `dist-exe/HCSFoundry.exe` (about 51 MB).

## Run

```powershell
.\dist-exe\HCSFoundry.exe
```

- Picks port 8770, or the next free port.
- Opens the browser automatically (`--no-browser` to disable).
- Stores user data in `%LOCALAPPDATA%\HCS Foundry` so the executable can live anywhere, including a read-only folder.
- `--data-dir <path>` overrides the data location.
- `--port <n>` sets the preferred port.

## What the executable contains

- The FastAPI service and all local modules.
- The built React SPA.
- No model weights, no datasets, no API keys.

## What it does not contain

`torch`, `transformers`, `peft` and `datasets` are **excluded** from the bundle to keep it small. Inference, Hub downloads, datasets, benchmarks and the agent work fully. Fine-tuning requires running from source with the training extras installed; the Training Lab reports `requires_stack` in the executable and explains why.

## Verify a build

```powershell
.\dist-exe\HCSFoundry.exe --no-browser --port 8790
curl http://127.0.0.1:8790/api/v1/health
```

Expected: `{"status":"ok", ...}` and the onboarding UI at `http://127.0.0.1:8790/`.
