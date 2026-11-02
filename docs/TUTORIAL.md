# Tutorial — five walkthroughs, copy and paste

Every command below was run on the reference machine (Ryzen 7 7735HS, Radeon 680M, 19.8 GiB RAM,
Windows). Expected output is quoted from those runs; your numbers will differ.

Two ways to run HCS Foundry:

- **Standalone**: `HCSFoundry.exe` → opens <http://127.0.0.1:8770>. API + UI, no ML stack (no
  training) — good for browsing, importing, benchmarking against a local model.
- **From source** (everything): `cd server && python -m pip install -e ".[train]" &&
  python -m foundry.cli serve`. The commands below assume this install, i.e. `foundry` on PATH or
  `python -m foundry.cli`.

---

## 0. First run in one screen

```bash
foundry run doctor
```

```
machine   AMD Ryzen 7 7735HS … · 19.8 GiB RAM · AMD Radeon(TM) Graphics
budget    2.7 GiB safe · 35 models (0 verified)
runtime   llama-server: yes
storage   5 INT4 store(s), 11.48 GiB · vram arena 64 MiB (auto budget 1451 MiB)
data      5 datasets (4 frozen)
next      no model has answered yet: `foundry models verify <id>`
```

`--json` gives the same data for scripts. The same report is available to agents through MCP
(`machine_scan`), see [MCP.md](MCP.md).

---

## 1. A model that answers

```bash
foundry hf search "qwen3 0.6b" --limit 5
foundry hf pull Qwen/Qwen3-0.6B --file model.safetensors --file tokenizer.json \
  --file tokenizer_config.json --file vocab.json --file merges.txt --file config.json
foundry models scan --root data/models
foundry models run <model-id> --prompt "Explain KV-cache quantization in two sentences" --max-tokens 96
```

Expected: the first token arrives after ~1 s and the answer streams at 40–110 tok/s on the iGPU.
`--json` returns `{text, reasoning, metrics}` so you can script it.

---

## 2. Train on your own logs (0.6B, minutes)

```bash
foundry logs clean --json          # discover → redact → dedupe → split → freeze
foundry datasets list              # note the dataset id
foundry train int4 --dataset <id> --model data/models/Qwen/Qwen3-0.6B \
  --steps 2 --seq 128 --streaming --offload --output data/training-runs/demo
```

Expected (reference run):

```
196 layers quantized, 2,523,136 LoRA parameters
int4 step 1: loss=5.4147
int4 step 2: loss=4.3085
manifest: memory.peak_rss_mb ≈ 2900, tuning.block_rows = 256, prefetch skipped/enabled
```

**2B class** (`Qwen3-1.7B`): the same command, ~4.5 GB peak RSS, loss 5.02 → 4.08.
**4B class**: works, but the store (4.57 GB) plus leftovers (2.39 GB) must fit next to the
process or the step time collapses to disk speed — see
[`docs/evidence/v7_4b_speed.md`](evidence/v7_4b_speed.md). If you are tight on RAM:

```bash
foundry train int4 … --steps 1 --seq 64 --store <existing store>
```

---

## 3. Adapter → GGUF → answer

```bash
foundry train merge-int4 --adapter data/training-runs/demo \
  --out data/training-runs/demo/merged
foundry gguf export data/training-runs/demo/merged --out data/gguf/demo-f16.gguf
foundry gguf quantize data/gguf/demo-f16.gguf --method Q4_K_M --out data/gguf/demo-Q4_K_M.gguf
foundry models scan --root data/gguf
foundry models run <new-model-id> --prompt "Say hello in German" --max-tokens 60
```

Expected (0.6B): merge 1.41 GiB fp16 · F16 1439 MiB · Q4_K_M 462 MiB · answer at ~117 tok/s.
`train unified` runs the whole chain (train → merge → export → quantize) for you.

---

## 4. Data in, data out

```bash
# import a folder of Claude Code / Codex / OpenCode / Antigravity sessions, or generic JSONL
foundry logs import --paths "C:/Users/you/.codex/sessions"
foundry datasets list

# export the frozen dataset in three formats
curl -s "$BASE/api/v1/datasets/<id>/export?format=jsonl" -H "Authorization: Bearer $TOKEN" > d.jsonl
curl -s "$BASE/api/v1/datasets/<id>/export?format=json&limit=20" -H "Authorization: Bearer $TOKEN" > sample.json
curl -s "$BASE/api/v1/datasets/<id>/export?format=csv"  -H "Authorization: Bearer $TOKEN" > d.csv
```

In the UI: **Data Studio → dataset → Distribution** shows task types, languages and split sizes;
the export buttons produce JSONL, JSON, CSV and a 20-record sample.

---

## 5. Benchmarks and the upgrade loop

```bash
foundry models run <id> --prompt "seed" --max-tokens 1        # ensures the server is warm
foundry bench instruction_adherence --limit 10
foundry bench codegen_exec --limit 5
```

Results land in **Benchmarks** with the exact model, runtime, config and timings. The Upgrade Lab
turns failures into candidate datasets, but only a human promotion gate moves a candidate into
production.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no_converter` when exporting GGUF | llama.cpp not found | set `FOUNDRY_LLAMA_CONVERT` to `convert_hf_to_gguf.py` |
| a 4B step takes minutes | working set does not fit in RAM | `--seq 64 --steps 1`, close other apps, keep the store on the fastest disk |
| `403 origin_not_allowed` from a browser | the page's origin is not allowlisted | add it to `FOUNDRY_UI_ORIGINS`, restart the service |
| training refuses with "does not fit" | governor budget | it prints the estimate; lower `--seq`, use `--offload`, or a smaller model |
| `pairing code wrong` | code required | `foundry token` or the code printed at startup |
