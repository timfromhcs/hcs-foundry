# Install on Linux

Tested logic is shared with Windows; the Linux machine probes use `lspci`, `rocm-smi`/`rocminfo`, `vulkaninfo` and `nvidia-smi` when present.

## 1. Prerequisites

```bash
sudo apt install -y python3 python3-venv python3-pip nodejs npm git build-essential
python3 --version   # 3.11+
node --version      # 20+
```

## 2. Backend

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m foundry.cli serve --port 8770
```

## 3. Frontend

```bash
cd web
npm install
npm run build      # or: npm run dev
```

## 4. Runtime

Build or install `llama-server` for your GPU:

```bash
# CPU
cmake -B build && cmake --build build -j
# Vulkan
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
# ROCm (AMD, only where the toolkit supports your GPU)
cmake -B build -DGGML_HIP=ON && cmake --build build -j
```

Then register the binary in **Machine → Inference runtimes**.

## 5. Permissions

The service binds to `127.0.0.1` only. If you run it in a container, publish the port to the host loopback and keep the pairing token out of images.

## Notes

- Wayland/X11 do not matter: the service is headless and only the browser renders.
- For AMD training, ROCm support depends on the exact GPU, driver and PyTorch build. The Training Lab reports `unsupported` rather than guessing.
- `foundry scan` marks anything it cannot probe as `unknown`, never as supported.
