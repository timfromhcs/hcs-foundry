"""Opt-in update check — a version comparison, never a download.

Local-first means the app does not phone home by default. When the user turns the check on, exactly
one HTTPS request goes to the public releases feed, carrying no identifier, no machine profile and
no usage data, and the answer is a version plus a link. Installing stays a user action.

A failure is a reported status, not an exception: the Settings screen must be able to say
"unreachable" without a stack trace.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable

RELEASES_URL = "https://api.github.com/repos/timfromhcs/hcs-foundry/releases/latest"
_USER_AGENT = "HCSFoundry-update-check"


def parse_version(value: str) -> tuple[int, ...]:
    """'7.3.0' -> (7, 3, 0); a leading 'v' and pre-release suffixes are tolerated."""
    match = re.match(r"^\s*v?(\d+(?:\.\d+)*)", str(value))
    if not match:
        raise ValueError(f"not a version: {value!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer(latest: str, current: str) -> bool:
    a, b = parse_version(latest), parse_version(current)
    length = max(len(a), len(b))
    a += (0,) * (length - len(a))
    b += (0,) * (length - len(b))
    return a > b


def _fetch_latest(url: str = RELEASES_URL, timeout: float = 6.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        return json.loads(response.read(256 * 1024).decode("utf-8"))


def check_for_update(
    current: str,
    *,
    fetch: Callable[[], dict] | None = None,
) -> dict:
    """Return a status dict. Never raises, never writes, never downloads an artifact."""
    try:
        payload = (fetch or _fetch_latest)()
    except Exception as exc:  # noqa: BLE001 - the network failing is an answer
        return {
            "status": "unreachable",
            "current": current,
            "latest": None,
            "available": False,
            "url": None,
            "message": f"{type(exc).__name__}: {exc}",
        }

    tag = str(payload.get("tag_name") or payload.get("name") or "").strip()
    if not tag:
        return {
            "status": "invalid_response",
            "current": current,
            "latest": None,
            "available": False,
            "url": None,
            "message": "the release feed returned no tag_name",
        }
    try:
        available = is_newer(tag, current)
    except ValueError as exc:
        return {
            "status": "invalid_response",
            "current": current,
            "latest": tag,
            "available": False,
            "url": payload.get("html_url"),
            "message": str(exc),
        }
    return {
        "status": "ok",
        "current": current,
        "latest": tag,
        "available": available,
        "url": payload.get("html_url"),
        "message": "a newer release exists" if available else "this is the newest release",
    }
