"""Discovery and import of agent session sources on this machine.

Sources are only read. Nothing is deleted, moved or modified. OpenCode is imported
through its documented CLI export with ``--sanitize``; file-based sources go through the
normal import pipeline, which redacts before storing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..db import Database
from .pipeline import ImportOptions, import_dataset

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class Source:
    provider: str
    label: str
    kind: str  # "files" | "cli"
    roots: list[Path] = field(default_factory=list)
    patterns: list[str] = field(default_factory=lambda: ["*.jsonl"])
    note: str | None = None

    def to_public(self) -> dict:
        return {
            "provider": self.provider,
            "label": self.label,
            "kind": self.kind,
            "roots": [str(p) for p in self.roots],
            "patterns": self.patterns,
            "note": self.note,
        }


def known_sources(settings: Settings) -> list[Source]:
    home = Path.home()
    return [
        Source(
            provider="claude_code",
            label="Claude Code projects",
            kind="files",
            roots=[home / ".claude" / "projects"],
            patterns=["*.jsonl"],
        ),
        Source(
            provider="codex",
            label="Codex CLI sessions",
            kind="files",
            roots=[home / ".codex" / "sessions"],
            patterns=["*.jsonl", "*.jsonl.zst"],
            note="Compressed sessions are decompressed into the import area; the originals are untouched.",
        ),
        Source(
            provider="antigravity",
            label="Antigravity transcripts",
            kind="files",
            roots=[home / ".gemini" / "antigravity" / "brain"],
            patterns=["transcript*.jsonl"],
        ),
        Source(
            provider="opencode",
            label="OpenCode sessions",
            kind="cli",
            note="Imported through `opencode session export --sanitize`; the active session is skipped.",
        ),
    ]


def _files(source: Source) -> list[Path]:
    found: list[Path] = []
    for root in source.roots:
        if not root.exists():
            continue
        for pattern in source.patterns:
            found.extend(sorted(root.rglob(pattern)))
    # Transcript chunks are duplicates of transcript.jsonl; skip them.
    return [p for p in found if p.is_file() and "chunks" not in p.parts]


def scan(settings: Settings) -> list[dict]:
    result = []
    for source in known_sources(settings):
        if source.kind == "files":
            files = _files(source)
            total = sum(p.stat().st_size for p in files)
            result.append(
                {
                    **source.to_public(),
                    "exists": any(root.exists() for root in source.roots),
                    "files": len(files),
                    "bytes": total,
                }
            )
        else:
            cli = shutil.which("opencode")
            sessions = _opencode_sessions() if cli else []
            result.append(
                {
                    **source.to_public(),
                    "exists": bool(cli),
                    "files": len(sessions),
                    "bytes": 0,
                    "sessions": [{"id": s.get("id"), "title": s.get("title"), "updated": s.get("updated")} for s in sessions[:10]],
                }
            )
    return result


def _run(cmd: list[str], timeout: float = 120.0) -> tuple[int, str, str]:
    executable = cmd[0]
    if sys.platform == "win32" and executable.lower().endswith((".cmd", ".bat")):
        cmd = ["cmd", "/c", *cmd]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, "", str(exc)


def _opencode_sessions() -> list[dict]:
    executable = shutil.which("opencode")
    if not executable:
        return []
    code, out, _ = _run([executable, "session", "list", "--format", "json", "--max-count", "100"], timeout=60)
    if code != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def export_opencode(settings: Settings, *, skip_newest: int = 1) -> list[Path]:
    """Export OpenCode sessions to the local import area (sanitized by the CLI)."""
    sessions = _opencode_sessions()
    if not sessions:
        return []
    executable = shutil.which("opencode") or "opencode"
    sessions.sort(key=lambda item: item.get("updated") or 0, reverse=True)
    selected = sessions[skip_newest:]
    target = settings.data_dir / "imports" / "opencode"
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for session in selected:
        session_id = session.get("id")
        if not session_id:
            continue
        code, out, err = _run([executable, "session", "export", str(session_id), "--sanitize"], timeout=300)
        if code != 0 or not out.strip():
            continue
        path = target / f"{session_id}.json"
        path.write_text(out, encoding="utf-8")
        written.append(path)
    return written


def _prepare_files(source: Source, settings: Settings) -> list[Path]:
    """Return importable files, decompressing .zst copies into the import area."""
    prepared: list[Path] = []
    for path in _files(source):
        if path.suffix.lower() != ".zst":
            prepared.append(path)
            continue
        try:
            import zstandard
        except ImportError:
            continue
        target = settings.data_dir / "imports" / source.provider / path.name.replace(".zst", "")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size == 0:
            with path.open("rb") as raw, target.open("wb") as out:
                decompressor = zstandard.ZstdDecompressor()
                decompressor.copy_stream(raw, out)
        prepared.append(target)
    return prepared


def import_all(
    db: Database,
    settings: Settings,
    providers: list[str],
    *,
    rights_status: str = "needs_review",
    progress: Callable[[float, str], None] | None = None,
    cancel=None,
) -> dict:
    """Import several sources into one dataset (mixed providers)."""
    paths: list[str] = []
    roots: list[Path] = []
    counts: dict[str, int] = {}
    for provider in providers:
        source = next((item for item in known_sources(settings) if item.provider == provider), None)
        if source is None:
            continue
        if source.kind == "files":
            files = _prepare_files(source, settings)
            paths.extend(str(p) for p in files)
            roots.extend(root for root in source.roots if root.exists())
            counts[provider] = len(files)
        else:
            exported = export_opencode(settings)
            paths.extend(str(p) for p in exported)
            roots.append(settings.data_dir / "imports")
            counts[provider] = len(exported)
    if not paths:
        raise ValueError("no importable files were found for the selected sources")
    roots.append(settings.data_dir / "imports")
    result = import_dataset(
        db,
        paths=paths,
        import_roots=roots,
        options=ImportOptions(name="Agent session import", provider="auto", rights_status=rights_status),
        progress=progress,
        cancel=cancel,
    )
    result["files_by_source"] = counts
    result["total_files"] = len(paths)
    return result


def import_source(
    db: Database,
    settings: Settings,
    provider: str,
    *,
    rights_status: str = "needs_review",
    progress: Callable[[float, str], None] | None = None,
    cancel=None,
) -> dict:
    source = next((item for item in known_sources(settings) if item.provider == provider), None)
    if source is None:
        raise ValueError(f"unknown source: {provider}")

    if source.kind == "files":
        files = _prepare_files(source, settings)
        if not files:
            raise ValueError(f"no files found for {provider}")
        paths = [str(p) for p in files]
        roots = [root for root in source.roots if root.exists()] + [settings.data_dir / "imports"]
        result = import_dataset(
            db,
            paths=paths,
            import_roots=roots,
            options=ImportOptions(name=f"{source.label} import", provider=provider, rights_status=rights_status),
            progress=progress,
            cancel=cancel,
        )
        result["source"] = source.to_public()
        result["files"] = len(files)
        return result

    exported = export_opencode(settings)
    if not exported:
        raise ValueError(
            "No OpenCode session was exported. Either none exists, or only the active session exists and it was skipped."
        )
    result = import_dataset(
        db,
        paths=[str(p) for p in exported],
        import_roots=[settings.data_dir / "imports"],
        options=ImportOptions(name=f"{source.label} import", provider="opencode", rights_status=rights_status),
        progress=progress,
        cancel=cancel,
    )
    result["source"] = source.to_public()
    result["files"] = len(exported)
    return result
