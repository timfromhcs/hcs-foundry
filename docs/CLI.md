# CLI

`foundry` works without starting the HTTP service. Every command supports `--json` where it makes sense and exits non-zero on failure.

```text
foundry serve | scan | info | token
foundry hf       search | files | pull | token
foundry models   list | scan | verify | load | unload | run
foundry bench    instruction_adherence | codegen_exec | latency_throughput
foundry train    smoke | preflight | sft | auto | dpo | distill | int4 | unified | plan | run | doctor | merge-int4
foundry run      doctor
foundry mcp      serve
foundry vram     status | clear
foundry datasets list
```

## Spec-driven training (v8)

```bash
foundry train plan   --spec spec.yaml                  # validate + preflight only
foundry train run    --spec spec.yaml [--out DIR]      # execute; writes RunManifest v2
foundry train doctor [--model DIR --dataset ID]        # registry + conformance matrix
```

`plan` prints the `spec_hash`, the selected trainer and every blocker/warning; it exits non-zero
when the spec cannot run. `doctor` without arguments lists the registered `method/backend` pairs
and which stacks are importable; with `--model` and `--dataset` it runs the conformance checks on
a tiny configuration.

## Examples

```bash
# machine report
foundry scan
foundry scan --json --write

# library
foundry models scan --root D:/aimodels
foundry models list
foundry models verify mdl_abc123

# one-shot inference (loads, prompts, unloads)
foundry models run mdl_abc123 --prompt "Explain KV cache quantization in two sentences" --max-tokens 200

# benchmarks against the loaded model
foundry models load mdl_abc123 --context 4096
foundry bench instruction_adherence --json

# training
foundry train smoke --json
foundry train preflight --dataset ds_123 --model E:/models/Qwen3-0.6B
foundry train sft --dataset ds_123 --model E:/models/Qwen3-0.6B --steps 20 --seq 256

# hub
foundry hf search "gemma 4 gguf" --limit 10
foundry hf files unsloth/Qwen3-8B-GGUF
foundry hf pull unsloth/Qwen3-8B-GGUF --file Qwen3-8B-Q4_K_M.gguf
```

## Output modes

- Human output is meant to be read in a terminal.
- `--json` prints a single JSON document and nothing else on stdout.
- Errors go to stderr as `error: …` and set exit code `1` (usage errors use `2`).
- Unicode output is forced to UTF-8 so Windows code pages cannot break it.

## Notes

- `models run` respects the resource governor and refuses a configuration that does not fit unless `--force` is passed.
- `train sft` refuses when the estimated memory exceeds available RAM unless `--force` is passed.
- The CLI and the web UI share the same SQLite database, so a model loaded by the CLI is visible to the service only after it restarts; the service supervises its own child process.
