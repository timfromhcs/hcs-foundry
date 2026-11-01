"""SWE-bench-format adapter with venv isolation.

The official dataset and repository checkouts are not bundled. The adapter accepts the
official JSONL shape and runs an instance only when the repository is present locally;
otherwise it reports ``not_run`` with the exact reason.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .venv import create_venv, venv_python

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
ALLOWED_COMMANDS = {"python", "python3", "py", "pytest"}


@dataclass
class SWEInstance:
    instance_id: str
    repo_path: str | None
    base_commit: str | None
    problem_statement: str
    test_command: str
    patch: str | None = None


def load_instances(path: str | Path) -> list[SWEInstance]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    instances = []
    for line in source.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        instances.append(
            SWEInstance(
                instance_id=str(data.get("instance_id") or data.get("id") or "unknown"),
                repo_path=data.get("repo_path"),
                base_commit=data.get("base_commit"),
                problem_statement=data.get("problem_statement") or "",
                test_command=data.get("test_command") or "python -m pytest -q",
                patch=data.get("patch"),
            )
        )
    return instances


def run_instance(
    instance: SWEInstance,
    patch_text: str,
    workdir: Path,
    *,
    timeout: float = 180.0,
) -> dict:
    if not instance.repo_path:
        return {"id": instance.instance_id, "status": "not_run", "detail": "instance has no local repo_path"}
    repo = Path(instance.repo_path).expanduser()
    if not repo.is_dir():
        return {"id": instance.instance_id, "status": "not_run", "detail": f"repository not available: {repo}"}
    command = shlex.split(instance.test_command)
    if not command or Path(command[0]).name.lower() not in ALLOWED_COMMANDS:
        return {
            "id": instance.instance_id,
            "status": "not_run",
            "detail": f"test_command not allowlisted: {instance.test_command[:80]}",
        }
    if not patch_text.strip():
        return {"id": instance.instance_id, "status": "fail", "detail": "empty patch"}

    workspace = workdir / instance.instance_id
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    shutil.copytree(repo, workspace, dirs_exist_ok=True)

    patch_file = workspace / "model.patch"
    patch_file.write_text(patch_text, encoding="utf-8")
    try:
        applied = subprocess.run(
            ["git", "apply", "--whitespace=fix", str(patch_file)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"id": instance.instance_id, "status": "fail", "detail": f"git apply failed: {exc}"}
    if applied.returncode != 0:
        return {
            "id": instance.instance_id,
            "status": "fail",
            "detail": f"patch did not apply: {(applied.stderr or applied.stdout)[-300:]}",
        }

    try:
        venv_dir = create_venv(workspace)
    except Exception as exc:  # noqa: BLE001
        return {"id": instance.instance_id, "status": "error", "detail": f"venv: {exc}"}

    executable = command[0].lower()
    if executable in {"python", "python3", "py"}:
        command[0] = str(venv_python(venv_dir))
    elif executable == "pytest":
        command = [str(venv_python(venv_dir)), "-m", "pytest", *command[1:]]

    try:
        run = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return {"id": instance.instance_id, "status": "fail", "detail": f"timeout after {timeout}s"}
    output = ((run.stdout or "") + (run.stderr or ""))[-1500:]
    return {
        "id": instance.instance_id,
        "status": "pass" if run.returncode == 0 else "fail",
        "detail": "tests passed" if run.returncode == 0 else output,
        "command": " ".join(command),
    }


def run_suite(
    instances: list[SWEInstance],
    patch_for,
    workdir: Path,
    *,
    progress=None,
    cancel=None,
) -> tuple[list[dict], dict]:
    cases = []
    for index, instance in enumerate(instances):
        if cancel is not None and getattr(cancel, "is_set", lambda: False)():
            raise InterruptedError("swebench cancelled")
        if progress:
            progress(index / max(1, len(instances)) * 100, f"swebench {instance.instance_id}")
        try:
            patch_text = patch_for(instance)
        except Exception as exc:  # noqa: BLE001
            cases.append({"id": instance.instance_id, "status": "error", "detail": str(exc)[:300]})
            continue
        cases.append(run_instance(instance, patch_text, workdir))
    return cases, {"sandbox": "per-instance copy + venv", "instances": len(instances)}
