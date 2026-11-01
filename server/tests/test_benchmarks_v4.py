from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from foundry.benchmarks import agentic, repo_fix, swebench, suite_catalog


class FakeServer:
    def __init__(self, text: str):
        self.text = text

    def complete(self, messages, **kwargs):
        return {"text": self.text, "metrics": {}}


def test_suite_catalog_has_new_suites():
    ids = {suite["id"] for suite in suite_catalog()}
    assert {"repo_fix", "agentic", "swebench"} <= ids
    swe = next(suite for suite in suite_catalog() if suite["id"] == "swebench")
    assert swe["available"] is False
    assert "not bundled" in swe["unavailable_reason"]


def test_repo_fix_case_passes_with_correct_code(tmp_path: Path):
    case = repo_fix.CASES[0]
    fixed = "def add(a, b):\n    return a + b\n"
    result = repo_fix.run_case(FakeServer(f"```python\n{fixed}```"), case, tmp_path)
    assert result["status"] == "pass", result["detail"]


def test_repo_fix_case_fails_with_wrong_code(tmp_path: Path):
    case = repo_fix.CASES[0]
    result = repo_fix.run_case(FakeServer("```python\ndef add(a, b):\n    return a * b\n```"), case, tmp_path)
    assert result["status"] == "fail"


def test_agentic_case_passes(tmp_path: Path):
    case = agentic.CASES[0]
    result = agentic.run_case(FakeServer('{"files": {"hello.txt": "hello foundry"}}'), case, tmp_path)
    assert result["status"] == "pass", result["detail"]


def test_agentic_case_detects_wrong_content(tmp_path: Path):
    case = agentic.CASES[0]
    result = agentic.run_case(FakeServer('{"files": {"hello.txt": "nope"}}'), case, tmp_path)
    assert result["status"] == "fail"


def test_agentic_json_update(tmp_path: Path):
    case = agentic.CASES[1]
    answer = '{"files": {"app.json": "{\\"port\\": 8770, \\"debug\\": false}"}}'
    result = agentic.run_case(FakeServer(answer), case, tmp_path)
    assert result["status"] == "pass", result["detail"]


def _make_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "buggy.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (path / "test_buggy.py").write_text(
        "from buggy import add\nassert add(2, 3) == 5\nprint('PASS')\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=path,
        check=True,
    )
    return path


PATCH = """--- a/buggy.py
+++ b/buggy.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


def test_swebench_instance_runs_in_venv(tmp_path: Path):
    repo = _make_git_repo(tmp_path / "repo")
    instance = swebench.SWEInstance(
        instance_id="local-1",
        repo_path=str(repo),
        base_commit=None,
        problem_statement="add is wrong",
        test_command="python test_buggy.py",
    )
    result = swebench.run_instance(instance, PATCH, tmp_path / "work")
    assert result["status"] == "pass", result["detail"]


def test_swebench_missing_repo_is_not_run(tmp_path: Path):
    instance = swebench.SWEInstance(
        instance_id="missing",
        repo_path=str(tmp_path / "nope"),
        base_commit=None,
        problem_statement="x",
        test_command="python test.py",
    )
    result = swebench.run_instance(instance, PATCH, tmp_path / "work")
    assert result["status"] == "not_run"
    assert "not available" in result["detail"]


def test_swebench_rejects_unsafe_command(tmp_path: Path):
    repo = _make_git_repo(tmp_path / "repo2")
    instance = swebench.SWEInstance(
        instance_id="unsafe",
        repo_path=str(repo),
        base_commit=None,
        problem_statement="x",
        test_command="powershell -Command Remove-Item -Recurse /",
    )
    result = swebench.run_instance(instance, PATCH, tmp_path / "work")
    assert result["status"] == "not_run"
    assert "allowlisted" in result["detail"]


def test_swebench_load_instances(tmp_path: Path):
    path = tmp_path / "instances.jsonl"
    path.write_text(
        '{"instance_id": "a", "repo_path": "x", "problem_statement": "p", "test_command": "python t.py"}\n',
        encoding="utf-8",
    )
    instances = swebench.load_instances(path)
    assert instances[0].instance_id == "a"
