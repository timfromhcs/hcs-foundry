"""Model Context Protocol server for HCS Foundry (stdio transport).

An agent (or any MCP client) can drive the same local capabilities the CLI has, without a
browser and without shell access to the host: machine scan, model library, dataset listing and
export, INT4 training, adapter merge, GGUF quantization and the VRAM cache report.

Design rules:

* **One JSON-RPC message per line.** No batching, no framing headers — that is what the stdio
  transport of MCP specifies and what simple clients expect.
* **Typed, small surface.** Every tool has a JSON schema; there is no "run any command" tool.
* **Paths are validated** against the configured model/dataset roots before anything is touched.
* **stdout belongs to the protocol.** Diagnostics go to stderr, so a stray ``print`` cannot
  corrupt a session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "hcs-foundry"


def _log(message: str) -> None:
    print(f"[hcs-foundry-mcp] {message}", file=sys.stderr, flush=True)


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOL_SPECS: list[dict] = [
    {
        "name": "machine_scan",
        "description": "Read-only scan of the host: OS, CPU, RAM, GPU, accelerators, disk, runtimes.",
        "inputSchema": _schema({}),
    },
    {
        "name": "machine_budget",
        "description": "Current memory/disk budget and the governor's safe limits.",
        "inputSchema": _schema({}),
    },
    {
        "name": "models_list",
        "description": "List the local model library (GGUF files with metadata and task class).",
        "inputSchema": _schema(
            {
                "task": {"type": "string", "description": "Optional filter, e.g. 'coding' or 'chat'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
            }
        ),
    },
    {
        "name": "datasets_list",
        "description": "List imported datasets with record counts, status and rights.",
        "inputSchema": _schema({}),
    },
    {
        "name": "datasets_export",
        "description": "Export a frozen dataset as JSONL and return the written path and counts.",
        "inputSchema": _schema(
            {
                "dataset_id": {"type": "string"},
                "format": {"type": "string", "enum": ["jsonl"], "default": "jsonl"},
            },
            ["dataset_id"],
        ),
    },
    {
        "name": "vram_status",
        "description": "Report the tiered cache arena (size on disk, last run's cache statistics).",
        "inputSchema": _schema({}),
    },
    {
        "name": "train_int4",
        "description": (
            "Run INT4 LoRA training on a local HF directory and block until it finishes. "
            "Uses the disk-backed store so 2–4B checkpoints fit; keeps a few steps by default."
        ),
        "inputSchema": _schema(
            {
                "model_path": {"type": "string", "description": "Local HF directory (safetensors)."},
                "dataset_id": {"type": "string"},
                "steps": {"type": "integer", "minimum": 1, "maximum": 50, "default": 1},
                "seq": {"type": "integer", "minimum": 32, "maximum": 2048, "default": 128},
                "rank": {"type": "integer", "minimum": 1, "maximum": 64, "default": 4},
                "offload": {"type": "boolean", "default": True},
            },
            ["model_path", "dataset_id"],
        ),
    },
    {
        "name": "train_merge_int4",
        "description": "Merge an INT4 LoRA adapter into a plain fp16 checkpoint (streamed shards).",
        "inputSchema": _schema(
            {
                "adapter_dir": {"type": "string", "description": "Run dir or adapter dir with lora_int4.pt."},
                "store_dir": {"type": "string"},
                "output_dir": {"type": "string"},
            },
            ["adapter_dir"],
        ),
    },
    {
        "name": "gguf_quantize",
        "description": "Quantize a local GGUF file (Q4_K_M by default) and return size and ratio.",
        "inputSchema": _schema(
            {
                "source": {"type": "string"},
                "method": {"type": "string", "default": "Q4_K_M"},
                "output": {"type": "string"},
            },
            ["source"],
        ),
    },
]

TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}


class McpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class McpServer:
    """Transport-agnostic MCP handler; ``serve_stdio`` wires it to stdin/stdout."""

    def __init__(self, state, *, log: Callable[[str], None] | None = None) -> None:
        self.state = state
        self.log = log or _log
        self.initialized = False

    # ------------------------------------------------------------------ tools
    def _roots(self) -> list[Path]:
        roots = list(self.state.settings.model_roots)
        roots.append(self.state.settings.data_dir)
        return roots

    def _within(self, raw: str) -> Path:
        from .security import resolve_within

        return resolve_within(raw, self._roots())

    def _call(self, name: str, arguments: dict) -> dict:
        if name not in TOOL_NAMES:
            raise McpError(-32602, f"unknown tool: {name}")
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise McpError(-32603, f"tool not implemented: {name}")
        return handler(arguments or {})

    def _tool_machine_scan(self, args: dict) -> dict:
        from .machine import scan_machine

        profile = scan_machine(self.state.settings)
        self.state.db.set_setting("machine_profile", profile)
        return {"profile": profile}

    def _tool_machine_budget(self, args: dict) -> dict:
        from .machine import compute_budget

        profile = self.state.latest_profile() or {}
        return {"budget": compute_budget(profile, self.state.settings).to_public()}

    def _tool_models_list(self, args: dict) -> dict:
        from .library import list_models

        models = list_models(self.state.db)
        task = (args.get("task") or "").strip().lower()
        if task:
            models = [m for m in models if str(m.get("task") or "").lower() == task]
        limit = int(args.get("limit") or 50)
        return {"total": len(models), "models": models[:limit]}

    def _tool_datasets_list(self, args: dict) -> dict:
        rows = self.state.db.query(
            "SELECT id, name, status, rights_status, stats FROM datasets ORDER BY rowid DESC LIMIT 100"
        )
        datasets = []
        for row in rows:
            payload = dict(row)
            try:
                payload["stats"] = json.loads(payload.get("stats") or "{}")
            except (TypeError, ValueError):
                payload["stats"] = {}
            datasets.append(payload)
        return {"total": len(datasets), "datasets": datasets}

    def _tool_datasets_export(self, args: dict) -> dict:
        from .data.pipeline import export_records

        dataset_id = str(args["dataset_id"])
        records = export_records(self.state.db, dataset_id)
        if not records:
            raise McpError(-32602, f"dataset not found or empty: {dataset_id}")
        target = self.state.settings.data_dir / "exports" / f"{dataset_id}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "dataset_id": dataset_id,
            "path": str(target),
            "records": len(records),
            "size_bytes": target.stat().st_size,
        }

    def _tool_vram_status(self, args: dict) -> dict:
        from .training.vram import budget_from_available

        arena = self.state.settings.data_dir / "cache" / "vram"
        index_path = arena / "arena.json"
        arena_file = arena / "arena.bin"
        index = None
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                index = None
        return {
            "arena_dir": str(arena),
            "arena_bytes_on_disk": arena_file.stat().st_size if arena_file.exists() else 0,
            "auto_budget_bytes": budget_from_available(),
            "index": index,
        }

    def _tool_train_int4(self, args: dict) -> dict:
        from .training import SFTConfig, create_int4_run
        from .training.vram import resolve_budget

        model_path = self._within(str(args["model_path"]))
        if not model_path.is_dir():
            raise McpError(-32602, f"model directory not found: {model_path}")
        dataset_id = str(args["dataset_id"])
        if not self.state.db.query("SELECT id FROM datasets WHERE id=?", (dataset_id,)):
            raise McpError(-32602, f"dataset not found: {dataset_id}")
        config = SFTConfig(
            model_path=str(model_path),
            dataset_id=dataset_id,
            max_steps=int(args.get("steps") or 1),
            max_seq_len=int(args.get("seq") or 128),
            lora_r=int(args.get("rank") or 4),
        )
        output = self.state.settings.training_dir / f"mcp_{int(__import__('time').time())}"
        result = create_int4_run(
            self.state.db,
            config,
            job_id=None,
            output_dir=output,
            streaming=True,
            offload=bool(args.get("offload", True)),
            vram_gb=None,
            vram_mode="off" if resolve_budget(None, "auto") == 0 else "auto",
            vram_dir=self.state.settings.data_dir / "cache" / "vram",
        )
        manifest = result["manifest"]
        return {
            "run_id": result["run_id"],
            "output_dir": str(output),
            "steps": manifest.get("steps"),
            "loss_first": manifest.get("loss_first"),
            "loss_last": manifest.get("loss_last"),
            "adapter_dir": manifest.get("adapter_dir"),
            "memory": manifest.get("memory"),
            "offload": bool(manifest.get("offload")),
        }

    def _tool_train_merge_int4(self, args: dict) -> dict:
        from .training.merge_int4 import merge_int4_adapter

        adapter = self._within(str(args["adapter_dir"]))
        adapter_dir = next(
            (candidate for candidate in (adapter, adapter / "adapter") if (candidate / "lora_int4.pt").exists()),
            None,
        )
        if adapter_dir is None:
            raise McpError(-32602, f"no lora_int4.pt under {adapter}")
        store = self._within(str(args["store_dir"])) if args.get("store_dir") else adapter_dir.parent / "int4-store"
        out = self._within(str(args["output_dir"])) if args.get("output_dir") else adapter_dir.parent / "merged"
        result = merge_int4_adapter(store, adapter_dir, out)
        return {k: result[k] for k in ("output", "dtype", "shards", "size_bytes", "quantized_weights_merged")}

    def _tool_gguf_quantize(self, args: dict) -> dict:
        from .gguf_tools import discover, quantize_gguf

        source = self._within(str(args["source"]))
        if not source.is_file():
            raise McpError(-32602, f"file not found: {source}")
        toolchain = discover(self.state.settings)
        method = str(args.get("method") or "Q4_K_M")
        out = self._within(str(args["output"])) if args.get("output") else source.with_name(
            f"{source.stem}-{method}.gguf"
        )
        result = quantize_gguf(source, out, method, toolchain=toolchain)
        return {k: result[k] for k in ("output", "method", "size_bytes", "ratio") if k in result}

    # ------------------------------------------------------------- JSON-RPC
    def handle(self, message: dict) -> dict | None:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}
        if method is None:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32600, "message": "missing method"}}

        try:
            if method == "initialize":
                self.initialized = True
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": SERVER_NAME, "version": self._version()},
                        "instructions": (
                            "Local-first model lab. All tools run on this machine; paths are limited "
                            "to the configured model and data roots."
                        ),
                    },
                }
            if method in ("notifications/initialized", "initialized"):
                return None
            if method == "ping":
                return {"jsonrpc": "2.0", "id": request_id, "result": {}}
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOL_SPECS}}
            if method == "tools/call":
                name = str(params.get("name") or "")
                arguments = params.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise McpError(-32602, "arguments must be an object")
                payload = self._call(name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                        "isError": False,
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }
        except McpError as exc:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": exc.code, "message": exc.message}}
        except Exception as exc:  # noqa: BLE001 - report, do not kill the session
            self.log(f"tool {method} failed: {exc}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"[:500]},
            }

    @staticmethod
    def _version() -> str:
        from . import __version__

        return __version__

    def serve_stdio(self, stdin=None, stdout=None) -> int:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        self.log(f"{SERVER_NAME} {self._version()} MCP server ready ({len(TOOL_SPECS)} tools)")
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
            else:
                response = self.handle(message)
            if response is None:
                continue
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
        return 0


def run_mcp(state) -> int:
    return McpServer(state).serve_stdio()
