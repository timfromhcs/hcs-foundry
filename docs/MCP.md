# MCP — drive HCS Foundry from an agent

HCS Foundry can run as a **Model Context Protocol** server over stdio. Any MCP client (Claude
Desktop, Cline, Continue, an agent runtime, your own script) then has typed access to the local
lab without shell access to the machine.

```bash
foundry mcp serve            # from the source install
HCSFoundry.exe --mcp         # from the standalone build
```

Diagnostics go to **stderr**; stdout carries only JSON-RPC, one message per line.

## What the tools can do

| Tool | Purpose | Writes? |
|---|---|---|
| `machine_scan` | OS, CPU, RAM, GPU, accelerators, disk, runtimes | no |
| `machine_budget` | safe memory/disk budget from the governor | no |
| `models_list` | local library with task class and metadata | no |
| `datasets_list` | datasets with record counts, status, rights | no |
| `datasets_export` | write a frozen dataset to `data/exports/<id>.jsonl` | yes (new file) |
| `vram_status` | tiered-cache arena size and last run's statistics | no |
| `train_int4` | INT4 LoRA training on a local HF directory, blocking | yes (new run dir) |
| `train_merge_int4` | merge an INT4 adapter into fp16 shards | yes (new dir) |
| `gguf_quantize` | quantize a local GGUF (Q4_K_M by default) | yes (new file) |
| `train_plan` | validate a TrainSpec and report the trainer + preflight | no |
| `eval_compare` | paired statistics over two stored benchmark runs | no |

**There is no "run any command" tool.** Paths are validated against the configured model and data
roots, so an agent cannot reach outside them; every write lands in a new directory under
`data/`.

## Client configuration

Most clients take a command + args pair. Example (JSON):

```json
{
  "mcpServers": {
    "hcs-foundry": {
      "command": "python",
      "args": ["-m", "foundry.cli", "mcp", "serve"],
      "cwd": "E:/AILOCALTRAIN/server",
      "env": { "FOUNDRY_DATA_DIR": "E:/AILOCALTRAIN/data" }
    }
  }
}
```

With the standalone build:

```json
{ "mcpServers": { "hcs-foundry": { "command": "C:/Tools/HCSFoundry.exe", "args": ["--mcp"] } } }
```

## Example session

```text
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18",
     "capabilities":{"tools":{"listChanged":false}},
     "serverInfo":{"name":"hcs-foundry","version":"7.0.0"}}}

→ {"jsonrpc":"2.0","id":2,"method":"tools/call",
   "params":{"name":"machine_scan","arguments":{}}}
← {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text",
     "text":"{\"profile\": {\"cpu\": {\"model\": \"AMD Ryzen 7 7735HS\"} ...}}"}],"isError":false}}
```

## Safety model

- The service still binds to `127.0.0.1` when serving HTTP; the MCP transport is a child process
  of the client, so it inherits the user's session — start it yourself, do not let a remote
  party spawn it.
- The pairing token is not needed for MCP: there is no network boundary to protect, only the
  process boundary the operating system already enforces.
- Training and merge block until done. Keep `steps` small when an agent calls them; the tool
  schema caps it at 50 and defaults to 1.
- Every error is returned as JSON-RPC `error` with the server staying alive; a failing tool never
  kills the session.

## Tests

```bash
cd server && python -m pytest tests/test_mcp.py -q     # 8 tests
```

They cover the handshake, the schema (no `shell`/`exec` tool exists), path rejection outside the
roots, JSONL export, unknown-tool handling and the line-based stdio transport.
