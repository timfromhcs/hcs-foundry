# Install on Windows

Tested on Windows 11 Pro (build 26200) with Python 3.14.6 and Node 22.

## 1. Prerequisites

```powershell
python --version    # 3.11+
node --version      # 20+
git --version
```

If Python is missing, install it from python.org and tick "Add python.exe to PATH".

## 2. Backend

```powershell
cd server
python -m pip install --upgrade pip
python -m pip install -e .
```

`pip install -e .` installs FastAPI, uvicorn, Pydantic, httpx and psutil. Training extras are optional:

```powershell
python -m pip install -e ".[train]"
```

> Only install `[train]` if you intend to fine-tune. The Training Lab detects whether the stack is importable and reports the truth either way.

## 3. Frontend

```powershell
cd web
npm install
npm run build
```

`npm run build` writes `web/dist`. The service serves it automatically.

For development with hot reload:

```powershell
npm run dev     # http://127.0.0.1:5173, proxying /api to :8770
```

## 4. Run

```powershell
cd server
python -m foundry.cli serve --port 8770
```

Open <http://127.0.0.1:8770>.

## 5. Add a llama-server runtime

The app never downloads a runtime. Either:

- put a `llama-server.exe` under `<repo>\data\runtime\`, or
- open **Machine → Inference runtimes**, enter a folder and press **Scan for runtimes**, then **Add**.

A Vulkan build is recommended on AMD hardware. CPU builds work as a fallback.

## 6. Add models

Open **Models**, press **Scan model folders**. Default roots are discovered automatically (for example `D:\aimodels`, `%USERPROFILE%\.lmstudio\models`). You can change the roots in **Settings**.

## 7. Verify the installation

```powershell
cd server
python -m pytest -q
python -m foundry.cli scan
```

`foundry scan` prints the measured machine profile. `--write` also writes `docs/audit/MACHINE_PROFILE.json`, which is git-ignored.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot reach the local service` | Service not running or on another port | Start `python -m foundry.cli serve` and check the port |
| `No verified llama-server binary` | No runtime registered | Machine → Inference runtimes → add one |
| `does_not_fit` when loading | Configuration exceeds the safe memory budget | Reduce context, choose a smaller quantization, or close applications |
| `401 unauthorized` | Token expired or revoked | Settings → revoke, then reconnect |
| `Host header is not loopback` | Opening the API through a proxy | Use `http://127.0.0.1:8770` directly |
| Model loads but answers are empty | Thinking model consumed the token budget in reasoning | Increase max tokens |
| Port 8770 in use | Another process | `--port 8780` and use that URL |
