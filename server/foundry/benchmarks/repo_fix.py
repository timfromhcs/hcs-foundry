"""Repo-fix benchmark: the model must repair a real file, tests run in a venv."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from .venv import create_venv, run_in_venv

CASES = [
    {
        "id": "fix_add",
        "file": "calc.py",
        "initial": "def add(a, b):\n    return a - b\n",
        "tests": "from calc import add\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\nprint('PASS')\n",
        "prompt": (
            "This file has a bug and its test fails.\n\ncalc.py:\n```python\ndef add(a, b):\n    return a - b\n```\n\n"
            "test output: AssertionError: add(2, 3) returned -1, expected 5\n\n"
            "Return the complete corrected calc.py in a single python code block."
        ),
    },
    {
        "id": "fix_clamp",
        "file": "clamp.py",
        "initial": "def clamp(x, low, high):\n    if x < low:\n        return low\n    return x\n",
        "tests": "from clamp import clamp\nassert clamp(5, 0, 3) == 3\nassert clamp(-1, 0, 3) == 0\nassert clamp(1, 0, 3) == 1\nprint('PASS')\n",
        "prompt": (
            "This file has a bug and its test fails.\n\nclamp.py:\n```python\n"
            "def clamp(x, low, high):\n    if x < low:\n        return low\n    return x\n```\n\n"
            "test output: AssertionError: clamp(5, 0, 3) returned 5, expected 3\n\n"
            "Return the complete corrected clamp.py in a single python code block."
        ),
    },
    {
        "id": "fix_parse",
        "file": "pairs.py",
        "initial": "def parse_pairs(text):\n    return {k: int(v) for k, v in (p.split('=') for p in text.split(';'))}\n",
        "tests": "from pairs import parse_pairs\nassert parse_pairs('') == {}\nassert parse_pairs('a=1') == {'a': 1}\nassert parse_pairs('a=1;b=2') == {'a': 1, 'b': 2}\nprint('PASS')\n",
        "prompt": (
            "This file has a bug and its test fails.\n\npairs.py:\n```python\n"
            "def parse_pairs(text):\n    return {k: int(v) for k, v in (p.split('=') for p in text.split(';'))}\n```\n\n"
            "test output: ValueError: not enough values to unpack (expected 2, got 1) for input ''\n\n"
            "Return the complete corrected pairs.py in a single python code block."
        ),
    },
]


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", text)
    return match.group(1) if match else text


def run_case(server, case: dict, workdir: Path, *, timeout: float = 60.0) -> dict:
    target = workdir / case["file"]
    target.write_text(case["initial"], encoding="utf-8")
    tests = workdir / "test_case.py"
    tests.write_text(case["tests"], encoding="utf-8")
    try:
        result = server.complete(
            [
                {"role": "system", "content": "You are a precise coding assistant. Output only the corrected file."},
                {"role": "user", "content": case["prompt"]},
            ],
            temperature=0.0,
            max_tokens=500,
        )
    except Exception as exc:  # noqa: BLE001
        return {"id": case["id"], "status": "error", "detail": str(exc)}
    code = _extract_code(result.get("text") or "")
    if not code.strip():
        return {"id": case["id"], "status": "fail", "detail": "no code block in the answer"}
    target.write_text(code, encoding="utf-8")
    try:
        venv_dir = create_venv(workdir)
    except Exception as exc:  # noqa: BLE001
        return {"id": case["id"], "status": "error", "detail": f"venv: {exc}"}
    execution = run_in_venv(venv_dir, tests, cwd=workdir, timeout=timeout)
    return {
        "id": case["id"],
        "status": "pass" if execution["ok"] else "fail",
        "detail": "tests passed" if execution["ok"] else execution.get("output", "")[-300:],
        "code": code[:1200],
        "duration": execution.get("duration"),
    }


def run_suite(server, *, progress=None, cancel=None) -> tuple[list[dict], dict]:
    cases: list[dict] = []
    durations: list[float] = []
    with tempfile.TemporaryDirectory(prefix="foundry-repo-fix-") as tmp:
        for index, case in enumerate(CASES):
            if cancel is not None and getattr(cancel, "is_set", lambda: False)():
                raise InterruptedError("repo_fix cancelled")
            if progress:
                progress(index / len(CASES) * 100, f"repo_fix {case['id']}")
            workdir = Path(tmp) / case["id"]
            workdir.mkdir(parents=True, exist_ok=True)
            result = run_case(server, case, workdir)
            cases.append(result)
            if result.get("duration"):
                durations.append(float(result["duration"]))
    metrics = {
        "sandbox": "per-case venv + subprocess",
        "mean_case_seconds": round(sum(durations) / len(durations), 2) if durations else None,
    }
    return cases, metrics
