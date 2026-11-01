"""Supervised ``llama-server`` process adapter.

One server at a time. The child process gets a random API key so other local
processes cannot drive it, and its PID is recorded so an orphan can be reaped
after a crash of the Foundry service.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import httpx

from ..db import utc_now

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if sys.platform == "win32" else 0


class RuntimeFailure(RuntimeError):
    pass


@dataclass
class LoadConfig:
    context: int = 4096
    gpu_layers: int = 99
    threads: int = 0
    batch: int = 2048
    ubatch: int = 512
    flash_attn: str = "auto"
    kv_type_k: str = "f16"
    kv_type_v: str = "f16"
    parallel: int = 1
    mlock: bool = False
    extra_args: list[str] = field(default_factory=list)

    def to_public(self) -> dict:
        return asdict(self)


@dataclass
class ServerState:
    status: str = "stopped"  # stopped|starting|ready|error|stopping
    model_id: str | None = None
    model_path: str | None = None
    binary: str | None = None
    port: int | None = None
    pid: int | None = None
    started_at: str | None = None
    ready_at: str | None = None
    config: dict = field(default_factory=dict)
    error: str | None = None
    load_seconds: float | None = None

    def to_public(self) -> dict:
        return asdict(self)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_args(binary: str, model_path: str, cfg: LoadConfig, port: int, api_key: str, alias: str) -> list[str]:
    threads = cfg.threads or max(2, (os.cpu_count() or 4) // 2)
    args = [
        binary,
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--api-key", api_key,
        "-a", alias[:60] or "local-model",
        "-c", str(cfg.context),
        "-ngl", str(cfg.gpu_layers),
        "-t", str(threads),
        "-b", str(cfg.batch),
        "-ub", str(cfg.ubatch),
        "-np", str(cfg.parallel),
        "--no-webui",
        "--metrics",
    ]
    if cfg.flash_attn and cfg.flash_attn != "auto":
        args += ["-fa", cfg.flash_attn]
    if cfg.kv_type_k and cfg.kv_type_k != "f16":
        args += ["-ctk", cfg.kv_type_k]
    if cfg.kv_type_v and cfg.kv_type_v != "f16":
        args += ["-ctv", cfg.kv_type_v]
    if cfg.mlock:
        args += ["--mlock"]
    args += list(cfg.extra_args)
    return args


class LlamaServer:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "runtime.json"
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._state = ServerState()
        self._logs: deque[str] = deque(maxlen=3000)
        self._reader: threading.Thread | None = None
        self._api_key: str | None = None
        self._cancel = threading.Event()

    # -- persistence -------------------------------------------------------
    def _persist(self) -> None:
        try:
            self.state_file.write_text(json.dumps(self._state.to_public(), indent=2), "utf-8")
        except OSError:
            pass

    def _load_persisted(self) -> dict | None:
        if not self.state_file.exists():
            return None
        try:
            return json.loads(self.state_file.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def reap_orphan(self) -> str | None:
        """Kill a child left behind by a previous service instance."""
        previous = self._load_persisted()
        if not previous or not previous.get("pid"):
            return None
        pid = int(previous["pid"])
        if previous.get("status") not in {"ready", "starting"}:
            return None
        if pid == os.getpid():
            return None
        try:
            import psutil

            proc = psutil.Process(pid)
            if proc.is_running() and "llama" in proc.name().lower():
                proc.kill()
                return f"reaped orphan llama-server pid={pid}"
        except Exception:
            return None
        return None

    # -- state -------------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            state = self._state.to_public()
        state["alive"] = self._proc is not None and self._proc.poll() is None
        state["log_tail"] = list(self._logs)[-40:]
        return state

    def logs(self, limit: int = 400) -> list[str]:
        return list(self._logs)[-limit:]

    # -- lifecycle ---------------------------------------------------------
    def load(
        self,
        *,
        model_id: str,
        model_path: str,
        binary: str,
        config: LoadConfig,
        alias: str | None = None,
        on_log: Callable[[str], None] | None = None,
        timeout: float = 300.0,
    ) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeFailure("a model is already loaded; unload it first")
            if not Path(model_path).is_file():
                raise RuntimeFailure(f"model file not found: {model_path}")
            if not Path(binary).is_file():
                raise RuntimeFailure(f"runtime binary not found: {binary}")

            port = free_port()
            api_key = "fnd-" + secrets.token_urlsafe(24)
            args = build_args(binary, model_path, config, port, api_key, alias or model_id)

            self._state = ServerState(
                status="starting",
                model_id=model_id,
                model_path=model_path,
                binary=binary,
                port=port,
                started_at=utc_now(),
                config=config.to_public(),
            )
            self._api_key = api_key
            self._logs.clear()
            self._cancel.clear()
            self._persist()

            started = time.time()
            try:
                proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                )
            except OSError as exc:
                self._state.status = "error"
                self._state.error = str(exc)
                self._persist()
                raise RuntimeFailure(f"could not start llama-server: {exc}") from exc

            self._proc = proc
            self._state.pid = proc.pid
            self._persist()

            def reader() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    text = line.rstrip()
                    if not text:
                        continue
                    self._logs.append(text)
                    if on_log is not None:
                        try:
                            on_log(text)
                        except Exception:
                            pass

            self._reader = threading.Thread(target=reader, name="llama-log", daemon=True)
            self._reader.start()

            ready = self._wait_ready(proc, port, api_key, timeout, on_log)
            if not ready:
                self.unload()
                raise RuntimeFailure(self._state.error or "llama-server did not become healthy")

            self._state.status = "ready"
            self._state.ready_at = utc_now()
            self._state.load_seconds = round(time.time() - started, 2)
            self._persist()
            return self.status()

    def _wait_ready(self, proc, port: int, api_key: str, timeout: float, on_log) -> bool:
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{port}/health"
        headers = {"Authorization": f"Bearer {api_key}"}
        while time.time() < deadline:
            if proc.poll() is not None:
                self._state.status = "error"
                self._state.error = f"llama-server exited with code {proc.returncode}"
                self._persist()
                return False
            try:
                response = httpx.get(url, headers=headers, timeout=2.0)
                if response.status_code == 200 and response.json().get("status") in {"ok", "ready"}:
                    return True
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.4)
        self._state.status = "error"
        self._state.error = f"health check timed out after {timeout:.0f}s"
        self._persist()
        return False

    def unload(self) -> dict:
        with self._lock:
            proc = self._proc
            self._state.status = "stopping"
            self._persist()
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                except OSError:
                    pass
            self._proc = None
            self._api_key = None
            self._state = ServerState(status="stopped")
            self._persist()
            return self.status()

    def shutdown(self) -> None:
        try:
            self.unload()
        except Exception:
            pass

    # -- inference ---------------------------------------------------------
    def _client(self, timeout: float | None = None) -> httpx.Client:
        state = self._state
        if not state.port or not self._api_key:
            raise RuntimeFailure("no model is loaded")
        return httpx.Client(
            base_url=f"http://127.0.0.1:{state.port}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0),
        )

    def props(self) -> dict:
        with self._client(10.0) as client:
            response = client.get("/props")
            response.raise_for_status()
            return response.json()

    def metrics(self) -> dict:
        try:
            with self._client(5.0) as client:
                response = client.get("/metrics")
                if response.status_code != 200:
                    return {}
                values: dict[str, float] = {}
                for line in response.text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    name, _, rest = line.partition(" ")
                    try:
                        values[name.strip()] = float(rest.strip().split(" ")[0])
                    except ValueError:
                        continue
                return values
        except (httpx.HTTPError, RuntimeError_):
            return {}

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[dict]:
        """Yield normalised stream events: ``{"type": "delta"|"done"|"error", ...}``."""
        payload = {
            "model": "local",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if stop:
            payload["stop"] = stop
        cancel = cancel or self._cancel
        started = time.time()
        first_token_at: float | None = None
        usage: dict = {}
        text_parts: list[str] = []

        try:
            with self._client(None) as client:
                with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        body = response.read().decode("utf-8", errors="replace")
                        yield {"type": "error", "message": f"HTTP {response.status_code}: {body[:500]}"}
                        return
                    for line in response.iter_lines():
                        if cancel.is_set():
                            yield {"type": "done", "cancelled": True, "text": "".join(text_parts)}
                            return
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:].strip()
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                        else:
                            continue
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                yield {"type": "reasoning", "text": reasoning}
                            if content:
                                if first_token_at is None:
                                    first_token_at = time.time()
                                text_parts.append(content)
                                yield {"type": "delta", "text": content}
                            if choice.get("finish_reason"):
                                usage.setdefault("finish_reason", choice["finish_reason"])
        except (httpx.HTTPError, RuntimeError_) as exc:
            yield {"type": "error", "message": str(exc)}
            return

        elapsed = time.time() - started
        completion_tokens = int(usage.get("completion_tokens") or len(text_parts))
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        yield {
            "type": "done",
            "text": "".join(text_parts),
            "usage": usage,
            "metrics": {
                "elapsed_seconds": round(elapsed, 3),
                "ttft_seconds": round(first_token_at - started, 3) if first_token_at else None,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "tokens_per_second": round(completion_tokens / elapsed, 2) if elapsed > 0 else None,
            },
        }

    def complete(self, messages: list[dict], **kwargs) -> dict:
        text = ""
        metrics: dict = {}
        for event in self.chat(messages, **kwargs):
            if event["type"] == "delta":
                text += event["text"]
            elif event["type"] == "done":
                metrics = event.get("metrics", {})
            elif event["type"] == "error":
                raise RuntimeFailure(event["message"])
        return {"text": text, "metrics": metrics}

    def cancel(self) -> None:
        self._cancel.set()
