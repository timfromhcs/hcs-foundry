"""Optional Vercel AI Gateway provider.

This is the only cloud path in HCS Foundry and it is **off by default**. It is used
only as a *teacher* for teaching annotations, never for chat, never for training and
never automatically. The key is stored locally, masked in the API and never logged.

If the account has no billing method, the gateway answers 403; the error is surfaced
verbatim instead of being hidden.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_MODEL = "alibaba/qwen-3-14b"


class GatewayError(RuntimeError):
    def __init__(self, code: str, message: str, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


def _path(state_dir: Path) -> Path:
    return Path(state_dir) / "vercel-gateway.json"


def get_config(state_dir: Path) -> dict:
    path = _path(state_dir)
    if not path.exists():
        return {"configured": False, "key": None, "model": DEFAULT_MODEL, "enabled": False}
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"configured": False, "key": None, "model": DEFAULT_MODEL, "enabled": False}
    key = data.get("key")
    return {
        "configured": bool(key),
        "key": key,
        "model": data.get("model") or DEFAULT_MODEL,
        "enabled": bool(data.get("enabled")),
    }


def set_config(state_dir: Path, *, key: str | None, model: str | None = None, enabled: bool | None = None) -> dict:
    path = _path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not key:
        path.unlink(missing_ok=True)
        return get_config(state_dir)
    current = get_config(state_dir)
    payload = {
        "key": key.strip(),
        "model": model or current.get("model") or DEFAULT_MODEL,
        "enabled": current.get("enabled", False) if enabled is None else bool(enabled),
    }
    path.write_text(json.dumps(payload, indent=2), "utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return get_config(state_dir)


def public_status(state_dir: Path) -> dict:
    config = get_config(state_dir)
    key = config["key"]
    return {
        "configured": config["configured"],
        "enabled": config["enabled"],
        "model": config["model"],
        "masked": (f"{key[:4]}…{key[-4:]}" if key and len(key) > 8 else ("*" * len(key) if key else None)),
        "note": "Off by default. Only teaching annotations can use it, and only when enabled.",
    }


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(connect=15.0, read=timeout, write=30.0, pool=15.0))


def _raise(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        message = response.json().get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        message = response.text[:200]
    if response.status_code == 401:
        raise GatewayError("invalid_key", "The AI Gateway key was rejected.", 401)
    if response.status_code == 403:
        raise GatewayError(
            "billing_required",
            message or "The AI Gateway requires a billing method on the Vercel account.",
            403,
        )
    if response.status_code == 429:
        raise GatewayError("rate_limited", "The AI Gateway rate limit was reached.", 429)
    raise GatewayError("http_error", message or f"Gateway HTTP {response.status_code}", response.status_code)


def list_models(key: str, limit: int = 50) -> list[dict]:
    with _client() as client:
        response = client.get(f"{GATEWAY_URL}/models", headers=_headers(key))
        _raise(response)
        data = response.json()
    models = []
    for item in data.get("data", [])[:limit]:
        models.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or item.get("id"),
                "owned_by": item.get("owned_by"),
            }
        )
    return models


def complete(
    key: str,
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> dict:
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with _client(timeout=120.0) as client:
        response = client.post(f"{GATEWAY_URL}/chat/completions", headers=_headers(key), json=payload)
        _raise(response)
        data = response.json()
    choices = data.get("choices") or []
    text = ""
    if choices:
        text = (choices[0].get("message") or {}).get("content") or ""
    return {"text": text, "usage": data.get("usage") or {}, "model": data.get("model") or payload["model"]}


class GatewayTeacher:
    """Adapter matching the enrichment ``llm.complete`` contract."""

    def __init__(self, key: str, model: str, *, state_dir: Path | None = None):
        self.key = key
        self.model = model
        self.teacher_name = f"vercel-ai-gateway:{model}"
        self.state_dir = state_dir

    def complete(self, messages: list[dict], **kwargs) -> dict:
        return complete(
            self.key,
            messages,
            model=self.model,
            temperature=float(kwargs.get("temperature", 0.2)),
            max_tokens=int(kwargs.get("max_tokens", 300)),
        )
