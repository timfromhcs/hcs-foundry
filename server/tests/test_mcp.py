from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from foundry.mcp_server import TOOL_NAMES, McpServer


@pytest.fixture()
def server(db, settings):
    from foundry.state import AppState
    from foundry.events import EventBus
    from foundry.jobs import JobRunner
    from foundry.runtime.llama_server import LlamaServer
    from foundry.security import PairingStore

    state = AppState(
        settings=settings,
        db=db,
        bus=EventBus(),
        runner=JobRunner(db, EventBus()),
        server=LlamaServer(settings.state_dir),
        pairing=PairingStore(settings.state_dir),
    )
    return McpServer(state, log=lambda message: None)


def test_initialize_announces_tools_and_protocol(server):
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    result = response["result"]
    assert result["serverInfo"]["name"] == "hcs-foundry"
    assert result["protocolVersion"]
    assert result["capabilities"]["tools"]["listChanged"] is False
    assert result["serverInfo"]["version"].count(".") == 2


def test_tools_list_is_typed_and_path_safe(server):
    response = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert names == TOOL_NAMES
    assert "shell" not in names and "exec" not in names
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["description"]


def test_machine_scan_returns_a_profile(server):
    response = server.handle(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "machine_scan", "arguments": {}}}
    )
    assert response["result"]["isError"] is False
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["profile"]["os"]["name"]
    assert payload["profile"]["cpu"]["model"]


def test_models_list_and_datasets_list_are_json(server):
    for name in ("models_list", "datasets_list"):
        response = server.handle(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": name, "arguments": {}}}
        )
        payload = json.loads(response["result"]["content"][0]["text"])
        assert "total" in payload


def test_unknown_tool_and_method_are_reported_not_crashed(server):
    unknown = server.handle(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "run_shell", "arguments": {}}}
    )
    assert unknown["error"]["code"] == -32602
    assert "unknown tool" in unknown["error"]["message"]

    missing = server.handle({"jsonrpc": "2.0", "id": 6, "method": "resources/list"})
    assert missing["error"]["code"] == -32601


def test_paths_outside_the_roots_are_refused(server):
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "train_int4", "arguments": {"model_path": "C:/Windows/System32", "dataset_id": "x"}},
        }
    )
    assert "error" in response
    assert "outside" in response["error"]["message"].lower() or "not allowed" in response["error"]["message"].lower()


def test_datasets_export_writes_jsonl(server, db, settings):
    dataset_id = "ds_mcp_export"
    db.insert(
        "datasets",
        {
            "id": dataset_id,
            "name": "mcp export",
            "provider": "generic",
            "status": "frozen",
            "rights_status": "confirmed",
            "record_count": 1,
            "stats": json.dumps({"record_count": 1}),
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    db.insert(
        "dataset_records",
        {
            "id": "rec_mcp_1",
            "dataset_id": dataset_id,
            "record_key": "rec_mcp_1",
            "split": "train",
            "excluded": 0,
            "rights_status": "confirmed",
            "content_hash": "sha256:deadbeef",
            "payload": json.dumps({"task": {"user_request": "hi"}, "trace": []}),
        },
    )
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "datasets_export", "arguments": {"dataset_id": dataset_id}},
        }
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["records"] == 1
    path = Path(payload["path"])
    assert path.exists() and path.read_text("utf-8").strip()


def test_stdio_transport_answers_one_message_per_line(server):
    stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n"
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n"
    )
    stdout = io.StringIO()
    assert server.serve_stdio(stdin=stdin, stdout=stdout) == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert [line["id"] for line in lines] == [1, 2]
    assert lines[0]["result"]["serverInfo"]["name"] == "hcs-foundry"
