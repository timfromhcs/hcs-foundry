"""Machine capability detection.

Every probe is isolated: a missing utility or a failed command degrades one section
to ``unknown`` and never aborts the scan. Results are explicitly labelled with an
evidence class so the UI cannot present a guess as a measurement.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import psutil

from ..config import Settings

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _run(cmd: list[str] | str, timeout: float = 12.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            shell=isinstance(cmd, str),
            creationflags=CREATE_NO_WINDOW,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, "", str(exc)


def _ps(script: str, timeout: float = 20.0) -> Any:
    """Run PowerShell and parse JSON output. Returns None on any failure."""
    if sys.platform != "win32":
        return None
    prelude = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
    rc, out, _err = _run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", prelude + script],
        timeout=timeout,
    )
    if rc != 0 or not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _dotnet_date(value: Any) -> str | None:
    """Convert PowerShell's ``/Date(1780012800000)/`` to an ISO date."""
    if not isinstance(value, str):
        return None
    match = re.search(r"/Date\((\d+)\)/", value)
    if not match:
        return value[:32]
    import datetime as _dt

    return _dt.datetime.fromtimestamp(int(match.group(1)) / 1000, tz=_dt.timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------

INTEGRATED_PATTERNS = (
    "radeon(tm) graphics",
    "radeon graphics",
    "radeon vega",
    "vega graphics",
    "intel(r) uhd",
    "intel(r) iris",
    "intel(r) hd",
    "apple m1",
    "apple m2",
    "apple m3",
    "apple m4",
)


def probe_os() -> dict:
    uname = platform.uname()
    data = {
        "status": "detected",
        "name": uname.system,
        "release": uname.release,
        "version": uname.version,
        "arch": uname.machine,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "evidence": "platform.uname",
    }
    if sys.platform == "win32":
        osinfo = _ps(
            "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture "
            "| ConvertTo-Json -Compress"
        )
        if isinstance(osinfo, dict):
            data.update(
                {
                    "name": osinfo.get("Caption") or data["name"],
                    "version": osinfo.get("Version") or data["version"],
                    "build": osinfo.get("BuildNumber"),
                    "arch": osinfo.get("OSArchitecture") or data["arch"],
                    "evidence": "Win32_OperatingSystem",
                }
            )
    return data


def probe_cpu() -> dict:
    freq = psutil.cpu_freq()
    data = {
        "status": "detected",
        "model": platform.processor() or "unknown",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_clock_mhz": round(freq.max or 0, 1) if freq else None,
        "arch": platform.machine(),
        "evidence": "psutil",
    }
    if sys.platform == "win32":
        info = _ps(
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed "
            "| ConvertTo-Json -Compress"
        )
        if isinstance(info, dict):
            data.update(
                {
                    "model": info.get("Name") or data["model"],
                    "physical_cores": info.get("NumberOfCores") or data["physical_cores"],
                    "logical_cores": info.get("NumberOfLogicalProcessors") or data["logical_cores"],
                    "max_clock_mhz": info.get("MaxClockSpeed") or data["max_clock_mhz"],
                    "evidence": "Win32_Processor",
                }
            )
    return data


def probe_memory() -> dict:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "status": "detected",
        "total_bytes": int(vm.total),
        "available_bytes": int(vm.available),
        "used_bytes": int(vm.used),
        "percent_used": float(vm.percent),
        "swap_total_bytes": int(swap.total),
        "swap_used_bytes": int(swap.used),
        "evidence": "psutil.virtual_memory",
    }


def probe_gpu_windows() -> list[dict]:
    raw = _ps(
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,DriverDate,"
        "PNPDeviceID,AdapterCompatibility,VideoProcessor | ConvertTo-Json -Compress"
    )
    devices = []
    for item in _as_list(raw):
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "unknown")
        pnp = str(item.get("PNPDeviceID") or "")
        adapter_ram = int(item.get("AdapterRAM") or 0)
        vendor = "AMD" if "VEN_1002" in pnp else "NVIDIA" if "VEN_10DE" in pnp else (
            "Intel" if "VEN_8086" in pnp else str(item.get("AdapterCompatibility") or "unknown")
        )
        integrated = any(pat in name.lower() for pat in INTEGRATED_PATTERNS) or (
            vendor == "AMD" and "VEN_1002" in pnp and adapter_ram <= 6 * 1024**3 and "rx" not in name.lower()
        )
        if "parsec" in name.lower():
            integrated = True  # virtual display adapter, never a compute device
        devices.append(
            {
                "name": name,
                "vendor": vendor,
                "pnp_id": pnp or None,
                "driver_version": item.get("DriverVersion"),
                "driver_date": _dotnet_date(item.get("DriverDate")),
                "reported_adapter_ram_bytes": adapter_ram,
                # Critical: a UMA/iGPU device has no dedicated VRAM. The driver value is
                # a shared-memory carve-out and must never be reported as dedicated.
                "dedicated_vram_bytes": 0 if integrated else adapter_ram,
                "shared_gpu_memory_bytes": adapter_ram if integrated else 0,
                "unified_memory": integrated,
                "virtual_display": "parsec" in name.lower() or "virtual" in name.lower(),
                "evidence": "Win32_VideoController",
            }
        )
    return devices


def probe_gpu_posix() -> list[dict]:
    devices: list[dict] = []
    if sys.platform == "darwin":
        rc, out, _ = _run(["system_profiler", "SPDisplaysDataType"], timeout=20)
        if rc == 0:
            name = None
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Chipset Model:"):
                    name = line.split(":", 1)[1].strip()
                if line.startswith("VRAM (Total):") and name:
                    vram = _parse_size(line.split(":", 1)[1].strip())
                    devices.append(
                        {
                            "name": name,
                            "vendor": "Apple" if "apple" in name.lower() else "unknown",
                            "dedicated_vram_bytes": vram,
                            "shared_gpu_memory_bytes": 0,
                            "unified_memory": "apple" in name.lower(),
                            "evidence": "system_profiler",
                        }
                    )
        return devices
    rc, out, _ = _run(["lspci"], timeout=10)
    if rc == 0:
        for line in out.splitlines():
            if re.search(r"VGA|3D controller|Display controller", line, re.I):
                name = line.split(":", 2)[-1].strip()
                devices.append(
                    {
                        "name": name,
                        "vendor": "AMD" if "amd" in name.lower() or "ati" in name.lower() else (
                            "NVIDIA" if "nvidia" in name.lower() else "Intel" if "intel" in name.lower() else "unknown"
                        ),
                        "dedicated_vram_bytes": None,
                        "shared_gpu_memory_bytes": None,
                        "unified_memory": None,
                        "evidence": "lspci",
                    }
                )
    return devices


def _parse_size(text: str) -> int | None:
    match = re.search(r"([\d.]+)\s*(GB|MB|TB)", text, re.I)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    factor = {"MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]
    return int(value * factor)


def probe_vulkan() -> dict:
    exe = shutil.which("vulkaninfo")
    if not exe:
        return {"status": "unknown", "reason": "vulkaninfo not installed", "evidence": "shutil.which"}
    rc, out, err = _run([exe, "--summary"], timeout=25)
    text = out + err
    if rc != 0 and not text:
        return {"status": "error", "reason": "vulkaninfo failed", "evidence": "vulkaninfo"}
    names = re.findall(r"deviceName\s*=\s*(.+)", text)
    version = None
    m = re.search(r"Vulkan Instance Version:\s*([\d.]+)", text)
    if m:
        version = m.group(1)
    heaps = [int(x) for x in re.findall(r"heapSize\s*=\s*(\d+)", text)]
    return {
        "status": "detected" if names else "unknown",
        "instance_version": version,
        "devices": [n.strip() for n in names],
        "max_heap_bytes": max(heaps) if heaps else None,
        "evidence": "vulkaninfo --summary",
    }


def probe_cuda() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"status": "not_detected", "reason": "nvidia-smi not on PATH", "evidence": "shutil.which"}
    rc, out, err = _run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=15)
    text = (out or "") + (err or "")
    # A vendored/patched nvidia-smi can exist without an NVIDIA device (e.g. ZLUDA)
    # and may ignore --query-gpu, printing its human table instead. Only trust
    # non-table lines that actually name an NVIDIA device with a memory figure.
    device_lines = [
        line.strip()
        for line in out.splitlines()
        if line.strip() and not line.strip().startswith(("|", "+", "="))
    ]
    if rc == 0 and any("nvidia" in line.lower() for line in device_lines) and any(
        "mib" in line.lower() for line in device_lines
    ):
        return {
            "status": "detected",
            "devices": device_lines,
            "evidence": "nvidia-smi --query-gpu",
        }
    return {
        "status": "not_detected",
        "reason": "nvidia-smi present but reports no NVIDIA device",
        "raw": text.strip()[:400],
        "evidence": "nvidia-smi --query-gpu",
    }


def probe_rocm() -> dict:
    exe = shutil.which("rocminfo")
    if not exe:
        return {"status": "not_detected", "reason": "rocminfo not installed", "evidence": "shutil.which"}
    rc, out, _ = _run([exe], timeout=20)
    if rc != 0:
        return {"status": "error", "reason": "rocminfo failed", "evidence": "rocminfo"}
    return {"status": "detected" if "gfx" in out else "unknown", "evidence": "rocminfo"}


def probe_npu() -> dict:
    if sys.platform != "win32":
        return {"status": "unknown", "reason": "NPU probe implemented for Windows only", "evidence": "n/a"}
    raw = _ps(
        "Get-PnpDevice -Class ComputeAccelerator -ErrorAction SilentlyContinue | "
        "Select-Object FriendlyName,Status | ConvertTo-Json -Compress"
    )
    devices = [d for d in _as_list(raw) if isinstance(d, dict)]
    return {
        "status": "detected" if devices else "not_detected",
        "devices": [d.get("FriendlyName") for d in devices],
        "evidence": "Get-PnpDevice ComputeAccelerator",
    }


def probe_directml() -> dict:
    # DirectML is only meaningful when a tested backend uses it. Presence of the DLL
    # alone is not evidence that our inference path can use it.
    dll = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "DirectML.dll"
    if dll.exists():
        return {
            "status": "unknown",
            "reason": "DirectML runtime present, but no Foundry adapter is verified for it",
            "evidence": str(dll),
        }
    return {"status": "not_detected", "reason": "DirectML.dll not found", "evidence": "filesystem"}


def probe_disks() -> list[dict]:
    disks = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        disks.append(
            {
                "mount": part.mountpoint,
                "fstype": part.fstype,
                "total_bytes": usage.total,
                "free_bytes": usage.free,
                "writable": os.access(part.mountpoint, os.W_OK),
            }
        )
    return disks


TOOLCHAIN_COMMANDS = [
    ("python", ["--version"]),
    ("node", ["--version"]),
    ("npm", ["--version"]),
    ("git", ["--version"]),
    ("uv", ["--version"]),
    ("pip", ["--version"]),
    ("cmake", ["--version"]),
    ("cl", []),
    ("rustc", ["--version"]),
    ("cargo", ["--version"]),
    ("docker", ["--version"]),
    ("ffmpeg", ["-version"]),
    ("vulkaninfo", ["--version"]),
    ("rocminfo", []),
    ("ollama", ["--version"]),
]


def probe_toolchains() -> dict:
    result: dict[str, dict] = {}
    for name, args in TOOLCHAIN_COMMANDS:
        path = shutil.which(name)
        if not path:
            result[name] = {"status": "not_detected"}
            continue
        entry = {"status": "detected", "path": path, "version": None}
        if args:
            rc, out, err = _run([path, *args], timeout=8)
            text = (out or err).strip().splitlines()
            if text:
                entry["version"] = text[0][:160]
        result[name] = entry
    # Python packages that matter for training/inference.
    try:
        import importlib.metadata as md

        packages = {}
        for pkg in ("torch", "transformers", "trl", "peft", "datasets", "numpy", "safetensors", "fastapi"):
            try:
                packages[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                packages[pkg] = None
        result["python_packages"] = {"status": "detected", "packages": packages}
    except Exception as exc:  # pragma: no cover
        result["python_packages"] = {"status": "error", "reason": str(exc)}
    return result


PRUNE_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages", "Windows",
    "$Recycle.Bin", "System32", "WinSxS", "AppData", ".cache", "dist", "build",
}


def discover_llama_servers(
    roots: Iterable[Path], *, max_depth: int = 5, time_budget: float = 20.0
) -> list[dict]:
    """Bounded search for llama-server binaries. Never scans the whole disk."""
    deadline = time.time() + time_budget
    names = {"llama-server.exe", "llama-server"}
    found: dict[str, dict] = {}

    for root in roots:
        root = Path(root).expanduser()
        if not root.exists() or not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if time.time() > deadline:
                break
            if len(Path(dirpath).parts) - base_depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d.lower() not in PRUNE_DIRS and not d.startswith(".")]
            for filename in filenames:
                if filename.lower() in names:
                    path = str(Path(dirpath) / filename)
                    if path not in found:
                        found[path] = describe_llama_server(path)
    return list(found.values())


def describe_llama_server(path: str) -> dict:
    rc, out, err = _run([path, "--version"], timeout=15)
    text = (out or "") + (err or "")
    version = None
    build = None
    backend = None
    m = re.search(r"version:\s*([^\s]+)", text)
    if m:
        version = m.group(1)
    m = re.search(r"build\s*(\d+)", text)
    if m:
        build = int(m.group(1))
    m = re.search(r"commit\s*([0-9a-f]{6,})", text)
    if m:
        backend = m.group(1)
    lower = path.lower()
    accel = "cpu"
    if "vulkan" in lower:
        accel = "vulkan"
    elif "cuda" in lower:
        accel = "cuda"
    elif "rocm" in lower or "hip" in lower:
        accel = "rocm"
    elif "metal" in lower:
        accel = "metal"
    return {
        "path": path,
        "version": version,
        "build": build,
        "commit": backend,
        "accelerator": accel,
        "status": "verified" if rc == 0 else "unverified",
        "raw_version": text.strip().splitlines()[0][:200] if text.strip() else None,
    }


def probe_runtimes(settings: Settings) -> dict:
    roots: list[Path] = []
    env_path = os.environ.get("FOUNDRY_LLAMA_SERVER")
    if env_path:
        roots.append(Path(env_path).parent)
    roots.extend(settings.llama_server_candidates)
    roots.append(settings.data_dir / "runtime")
    home = Path.home()
    for candidate in (
        home / ".foundry" / "runtime",
        Path("C:/llama.cpp"),
        home / "llama.cpp",
    ):
        if candidate.exists():
            roots.append(candidate)
    # Generic per-drive hints, not machine-specific absolute paths.
    if sys.platform == "win32":
        for drive in ("C:/", "D:/", "E:/"):
            for hint in ("AI", "llama.cpp", "LLM"):
                p = Path(drive) / hint
                if p.exists():
                    roots.append(p)

    servers = discover_llama_servers(roots, max_depth=5, time_budget=15)
    if env_path and Path(env_path).exists():
        servers.append(describe_llama_server(env_path))

    ollama_path = shutil.which("ollama")
    lmstudio_models = Path.home() / ".lmstudio" / "models"
    return {
        "llama_server": {
            "status": "detected" if servers else "not_detected",
            "candidates": servers,
            "searched_roots": [str(r) for r in roots],
            "evidence": "bounded filesystem walk + --version",
        },
        "ollama": {
            "status": "detected" if ollama_path else ("installed_no_cli" if (Path.home() / ".ollama" / "models").exists() else "not_detected"),
            "path": ollama_path,
            "evidence": "shutil.which / model store",
        },
        "lmstudio": {
            "status": "detected" if lmstudio_models.exists() else "not_detected",
            "models_dir": str(lmstudio_models) if lmstudio_models.exists() else None,
            "evidence": "filesystem",
        },
    }


def compute_tier(gpu_devices: list[dict], accelerators: dict) -> dict:
    discrete = [d for d in gpu_devices if not d.get("unified_memory") and (d.get("dedicated_vram_bytes") or 0) > 0]
    if discrete:
        vram = max(int(d.get("dedicated_vram_bytes") or 0) for d in discrete)
        if vram >= 40 * 1024**3:
            tier, label = 4, "Tier 4 — workstation / multi-GPU"
        elif vram >= 14 * 1024**3:
            tier, label = 3, "Tier 3 — high GPU memory"
        elif vram >= 7 * 1024**3:
            tier, label = 3, "Tier 3 — mid GPU memory"
        else:
            tier, label = 2, "Tier 2 — entry discrete GPU"
        return {"tier": tier, "label": label, "confidence": "verified", "reasons": [f"discrete GPU with {vram} bytes reported VRAM"]}

    unified = [d for d in gpu_devices if d.get("unified_memory")]
    vulkan_ok = accelerators.get("vulkan", {}).get("status") == "detected"
    if unified and vulkan_ok:
        return {
            "tier": 1,
            "label": "Tier 1 — integrated GPU / shared memory",
            "confidence": "verified",
            "reasons": [
                "integrated GPU with unified memory (no dedicated VRAM)",
                "Vulkan runtime detected",
            ],
        }
    if unified:
        return {
            "tier": 1,
            "label": "Tier 1 — integrated GPU / shared memory",
            "confidence": "likely",
            "reasons": ["integrated GPU detected", "GPU compute backend not verified"],
        }
    return {
        "tier": 0,
        "label": "Tier 0 — CPU only",
        "confidence": "verified",
        "reasons": ["no usable GPU device detected"],
    }


def scan_machine(settings: Settings) -> dict:
    started = time.time()
    gpu_devices = probe_gpu_windows() if sys.platform == "win32" else probe_gpu_posix()
    gpu_devices = [d for d in gpu_devices if not d.get("virtual_display")]
    accelerators = {
        "vulkan": probe_vulkan(),
        "cuda": probe_cuda(),
        "rocm": probe_rocm(),
        "directml": probe_directml(),
        "npu": probe_npu(),
    }
    memory = probe_memory()
    profile = {
        "schema_version": "1.0",
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_ms": None,
        "os": probe_os(),
        "cpu": probe_cpu(),
        "memory": memory,
        "gpu": {
            "status": "detected" if gpu_devices else "not_detected",
            "devices": gpu_devices,
            "dedicated_vram_bytes": sum(int(d.get("dedicated_vram_bytes") or 0) for d in gpu_devices),
            "shared_gpu_memory_bytes": sum(int(d.get("shared_gpu_memory_bytes") or 0) for d in gpu_devices),
            "unified_memory": any(d.get("unified_memory") for d in gpu_devices),
            "evidence": "platform GPU query",
        },
        "accelerators": accelerators,
        "disks": probe_disks(),
        "toolchains": probe_toolchains(),
        "runtimes": probe_runtimes(settings),
        "tier": compute_tier(gpu_devices, accelerators),
        "warnings": [],
    }
    if profile["gpu"]["unified_memory"]:
        profile["warnings"].append(
            "Integrated GPU: driver-reported AdapterRAM is shared system memory, not dedicated VRAM."
        )
    if accelerators["cuda"]["status"] != "detected":
        profile["warnings"].append("No NVIDIA CUDA device: GPU-accelerated training is unavailable.")
    if not profile["runtimes"]["llama_server"]["candidates"]:
        profile["warnings"].append("No llama-server binary found. Inference is unavailable until one is added.")
    profile["duration_ms"] = int((time.time() - started) * 1000)
    return profile
