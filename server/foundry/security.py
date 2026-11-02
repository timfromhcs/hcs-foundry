"""Security boundary for the local Foundry service.

Threat model
------------
The service is privileged (it spawns processes and reads local files) and it listens
on a TCP port. A web page the user visits must not be able to drive it. Therefore:

* bind to loopback only (see ``cli``/``main``), unless the user explicitly opts in,
* every request must carry a bearer token minted by an explicit pairing call,
* a present ``Origin`` header must be allowlisted; ``Access-Control-Allow-Origin: *``
  is never emitted,
* the ``Host`` header must be a loopback name (DNS-rebinding defence),
* state-changing requests must carry ``X-Foundry-Client: 1`` and a JSON body, which a
  cross-site HTML form cannot produce without a successful CORS preflight.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from starlette.responses import JSONResponse

TOKEN_PREFIX = "fnd_"
CLIENT_HEADER = "x-foundry-client"
TOKEN_HEADER = "x-foundry-token"
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

PUBLIC_PATHS = {
    "/api/v1/health",
    "/api/v1/system/pair",
    "/api/v1/system/bootstrap",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class PairingStore:
    """Issues and validates long-lived local tokens (stored hashed)."""

    def __init__(self, state_dir: Path, *, require_code: bool = False):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "pairing.json"
        self.code_path = state_dir / "pairing-code.txt"
        self.require_code = require_code
        self._lock = threading.Lock()
        self._data: dict = {"tokens": []}
        self._load()
        if self.require_code:
            self._ensure_code()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {"tokens": []}
        self._data.setdefault("tokens", [])

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), "utf-8")
        tmp.replace(self.path)

    # -- pairing code ------------------------------------------------------
    def _ensure_code(self) -> str:
        if self.code_path.exists():
            code = self.code_path.read_text("utf-8").strip()
            if code:
                return code
        code = f"{secrets.randbelow(900000) + 100000}"
        self.code_path.write_text(code, "utf-8")
        return code

    def pairing_code(self) -> str | None:
        return self._ensure_code() if self.require_code else None

    def verify_code(self, code: str) -> bool:
        expected = self._ensure_code()
        return hmac.compare_digest(str(code).strip(), expected)

    # -- tokens ------------------------------------------------------------
    def issue(self, label: str = "local ui") -> str:
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        with self._lock:
            self._data["tokens"].append(
                {
                    "id": secrets.token_hex(6),
                    "hash": _hash_token(token),
                    "label": label[:64],
                    "created_at": utc_now(),
                    "revoked": False,
                }
            )
            self._save()
        return token

    def validate(self, token: str | None) -> bool:
        if not token:
            return False
        digest = _hash_token(token)
        with self._lock:
            for entry in self._data["tokens"]:
                if not entry.get("revoked") and hmac.compare_digest(entry["hash"], digest):
                    return True
        return False

    def revoke_all(self) -> int:
        with self._lock:
            count = 0
            for entry in self._data["tokens"]:
                if not entry.get("revoked"):
                    entry["revoked"] = True
                    count += 1
            self._save()
        return count

    def public_list(self) -> list[dict]:
        return [
            {
                "id": e["id"],
                "label": e["label"],
                "created_at": e["created_at"],
                "revoked": e.get("revoked", False),
            }
            for e in self._data["tokens"]
        ]


@dataclass
class SecurityPolicy:
    origins: list[str] = field(default_factory=list)
    allow_lan: bool = False
    loopback_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1", "[::1]")

    def origin_allowed(self, origin: str | None) -> bool:
        if not origin:
            return True  # non-browser client; token is still required
        return origin.rstrip("/") in {o.rstrip("/") for o in self.origins}

    def host_allowed(self, host: str | None) -> bool:
        if not host:
            return False
        name = host.split(":")[0].strip().lower()
        if name in self.loopback_hosts:
            return True
        return self.allow_lan


def _header(scope, name: str) -> str | None:
    target = name.lower().encode()
    for key, value in scope.get("headers", []):
        if key.lower() == target:
            return value.decode("latin-1")
    return None


class PrivateNetworkAccessMiddleware:
    """Answer the Private Network Access preflight the browser sends before loopback.

    Chrome blocks ``https://public.example → http://127.0.0.1`` unless the preflight carries
    ``Access-Control-Allow-Private-Network: true``. The header does not widen the security
    boundary: the ``Origin`` allowlist and the pairing token still decide what happens.
    """

    HEADER = b"access-control-allow-private-network"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        wants = (_header(scope, "access-control-request-private-network") or "").lower() == "true"
        if not wants:
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                if not any(key.lower() == self.HEADER for key, _ in headers):
                    headers.append((self.HEADER, b"true"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


class SecurityMiddleware:
    """Raw ASGI middleware (keeps SSE streaming unbuffered)."""

    def __init__(self, app, *, store: PairingStore, policy: SecurityPolicy, public_paths: Iterable[str] = ()):
        self.app = app
        self.store = store
        self.policy = policy
        self.public_paths = set(public_paths) | set(PUBLIC_PATHS)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET").upper()
        if not path.startswith("/api/"):
            return await self.app(scope, receive, send)

        if not self.policy.host_allowed(_header(scope, "host")):
            return await self._reject(scope, receive, send, 403, "host_not_allowed", "Host header is not loopback.")

        if not self.policy.origin_allowed(_header(scope, "origin")):
            return await self._reject(scope, receive, send, 403, "origin_not_allowed", "Origin is not paired with this service.")

        if path in self.public_paths:
            return await self.app(scope, receive, send)

        # A browser preflight carries no token by design; it must reach CORS so the real
        # request can be authorised. The origin is already checked above.
        if method == "OPTIONS":
            return await self.app(scope, receive, send)

        token = self._extract_token(scope)
        if not self.store.validate(token):
            return await self._reject(scope, receive, send, 401, "unauthorized", "Missing or invalid pairing token.")

        if method in MUTATING_METHODS:
            if _header(scope, CLIENT_HEADER) != "1":
                return await self._reject(scope, receive, send, 403, "client_header_missing", "Missing X-Foundry-Client header.")
            content_type = (_header(scope, "content-type") or "").lower()
            if method in {"POST", "PUT", "PATCH"} and "application/json" not in content_type:
                return await self._reject(scope, receive, send, 415, "unsupported_media_type", "JSON body required.")

        return await self.app(scope, receive, send)

    @staticmethod
    def _extract_token(scope) -> str | None:
        auth = _header(scope, "authorization")
        if auth and auth.lower().startswith("bearer "):
            return auth[7:].strip()
        direct = _header(scope, TOKEN_HEADER)
        if direct:
            return direct
        # EventSource cannot set headers; allow the token as a query parameter for GET.
        if scope.get("method", "GET").upper() == "GET":
            query = scope.get("query_string", b"").decode("latin-1")
            for pair in query.split("&"):
                if pair.startswith("token="):
                    from urllib.parse import unquote_plus

                    return unquote_plus(pair[6:])
        return None

    @staticmethod
    async def _reject(scope, receive, send, status: int, code: str, message: str):
        response = JSONResponse({"error": {"code": code, "message": message}}, status_code=status)
        await response(scope, receive, send)


# --------------------------------------------------------------------------
# Path allowlist
# --------------------------------------------------------------------------

class PathNotAllowed(ValueError):
    pass


def resolve_within(raw: str | Path, roots: Iterable[Path]) -> Path:
    """Resolve ``raw`` and require it to live inside one of ``roots``."""
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:  # pragma: no cover - defensive
        raise PathNotAllowed(f"cannot resolve path: {exc}") from exc
    for root in roots:
        try:
            root_resolved = Path(root).expanduser().resolve(strict=False)
        except OSError:
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return resolved
    raise PathNotAllowed(f"path is outside the allowed roots: {raw}")


def ensure_file(raw: str | Path, roots: Iterable[Path]) -> Path:
    path = resolve_within(raw, roots)
    if not path.is_file():
        raise PathNotAllowed(f"not a file: {raw}")
    return path
