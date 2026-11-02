# Next action

**Phase:** V8-S5 — measured training flywheel.

**Command to start from:**

```bash
# 1. generate enough data with disjoint templates
cd server
python -m foundry.cli tasks generate --family all --seed 11 --per-template 30 \
  --out ../data/exports/tasks_s5.jsonl        # 300 items, 180/60/60

# 2. baseline: run the base model over the test split, per case
#    (needs a loader adapter that reads the JSONL and writes benchmark_cases rows)

# 3. candidate: RFT on the train split
cat > /tmp/rft.yaml <<'YAML'
spec_version: 1
method: rft
backend: fp32_lora
profile: local-smoke
model: {path: ../data/models/Qwen3-0.6B}
data: {dataset_id: tasks_s5}
hyper: {max_steps: 200, seq_len: 512, lora_r: 8, lora_alpha: 16, lr: 0.0002}
method_args:
  tasks_path: ../data/exports/tasks_s5.jsonl
  samples_per_task: 1
  min_accepted: 60
  max_candidates: 400
  seed: 11
YAML
python -m foundry.cli train run --spec /tmp/rft.yaml --json

# 4. paired comparison of baseline vs candidate on the same test case ids
#    foundry eval_compare / MCP eval_compare, n >= 60 for the test split
```

**Expected result:** either `improved` (CI lower bound > 0, no guard regression, human approval) or
one of `no_material_improvement` / `inconclusive` / `regression`, written to
`docs/evidence/v8_flywheel.md`.

**Blocked by:** a loader that turns generated tasks into benchmark cases (small, next), and CPU time
for 200 steps on 0.6B plus 120 test generations. Both are known; neither is faked.

**Do not start with:** a 4B model, GRPO, or a new serving stack.
