"""Agentic benchmark: multi-step file tasks with deterministic oracles.

The model answers with a JSON object describing file writes. A sandboxed runner applies
them in a temp workspace and checks the result. One task also executes the written code
inside a venv.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from .venv import create_venv, run_in_venv

CASES = [
    {
        "id": "write_readme",
        "prompt": (
            "Create a file named hello.txt containing exactly the text 'hello foundry' and nothing else. "
            'Answer with JSON only: {"files": {"path": "content"}}'
        ),
        "files": {"hello.txt": "hello foundry"},
    },
    {
        "id": "update_config",
        "prompt": (
            "The file app.json currently contains {\"port\": 8000, \"debug\": true}. Change the port to 8770 and "
            "set debug to false. Answer with JSON only: {\"files\": {\"app.json\": \"<full new content>\"}}"
        ),
        "files": {"app.json": '{"port": 8770, "debug": false}'},
        "check": "json_equal",
    },
    {
        "id": "write_function",
        "prompt": (
            "Write double.py with a function `double(x)` that returns x*2. "
            'Answer with JSON only: {"files": {"double.py": "<code>"}}'
        ),
        "files": {"double.py": "def double(x):\n    return x * 2\n"},
        "execute": "from double import double\nassert double(4) == 8\nassert double(-2) == -4\nprint('PASS')\n",
    },
]


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the answer")
    return json.loads(candidate[start : end + 1])


def _normalise(content: str) -> str:
    return " ".join(content.split())


def run_case(server, case: dict, workdir: Path, *, timeout: float = 45.0) -> dict:
    try:
        result = server.complete(
            [
                {"role": "system", "content": "You are an agent that writes files. Answer with JSON only."},
                {"role": "user", "content": case["prompt"]},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        payload = _extract_json(result.get("text") or "")
        files = payload.get("files") or {}
        if not isinstance(files, dict) or not files:
            return {"id": case["id"], "status": "fail", "detail": "answer had no files object"}
        for name, content in files.items():
            safe = Path(str(name)).name
            (workdir / safe).write_text(str(content), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return {"id": case["id"], "status": "fail", "detail": f"invalid answer: {exc}"}

    if case.get("check") == "json_equal":
        try:
            actual = json.loads((workdir / "app.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"id": case["id"], "status": "fail", "detail": f"invalid JSON written: {exc}"}
        expected = json.loads(case["files"]["app.json"])
        return {
            "id": case["id"],
            "status": "pass" if actual == expected else "fail",
            "detail": f"got {actual}",
        }

    for name, expected in case["files"].items():
        path = workdir / name
        if not path.exists():
            return {"id": case["id"], "status": "fail", "detail": f"{name} was not created"}
        if _normalise(path.read_text("utf-8")) != _normalise(expected):
            return {"id": case["id"], "status": "fail", "detail": f"{name} content mismatch"}

    if case.get("execute"):
        try:
            venv_dir = create_venv(workdir)
            script = workdir / "check.py"
            script.write_text(case["execute"], encoding="utf-8")
            execution = run_in_venv(venv_dir, script, cwd=workdir, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return {"id": case["id"], "status": "error", "detail": f"execution: {exc}"}
        return {
            "id": case["id"],
            "status": "pass" if execution["ok"] else "fail",
            "detail": "executed successfully" if execution["ok"] else execution.get("output", "")[-300:],
            "duration": execution.get("duration"),
        }

    return {"id": case["id"], "status": "pass", "detail": "files match"}


def run_suite(server, *, progress=None, cancel=None) -> tuple[list[dict], dict]:
    cases: list[dict] = []
    durations: list[float] = []
    with tempfile.TemporaryDirectory(prefix="foundry-agentic-") as tmp:
        for index, case in enumerate(CASES):
            if cancel is not None and getattr(cancel, "is_set", lambda: False)():
                raise InterruptedError("agentic cancelled")
            if progress:
                progress(index / len(CASES) * 100, f"agentic {case['id']}")
            workdir = Path(tmp) / case["id"]
            workdir.mkdir(parents=True, exist_ok=True)
            result = run_case(server, case, workdir)
            cases.append(result)
            if result.get("duration"):
                durations.append(float(result["duration"]))
    metrics = {
        "sandbox": "temp workspace + per-case venv",
        "mean_case_seconds": round(sum(durations) / len(durations), 2) if durations else None,
    }
    return cases, metrics
