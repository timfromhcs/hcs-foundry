# Disk-light training

The machine has one SSD and no spare. Training that writes optimizer state no resume path can read
is wear for nothing, so the trainer writes the adapter and nothing else.

## What is written now

| Artifact | Written | Why |
|---|---|---|
| adapter weights (`adapter_model.safetensors`) | once, at the end | this is the product of the run |
| tokenizer files | once, with the adapter | needed to reload the adapter |
| checkpoints | at most 2, at least 10 steps apart | crash recovery without a write storm |
| optimizer / scheduler / RNG state | **never** | `save_only_model=True`; no trainer has a resume path |
| INT4 store (4B) | only when it does not exist (`reused` is reported) | a 4.6 GB rewrite per run would dominate |

`save_steps` is floored at 10. A `save_every=1` experiment used to mean one checkpoint per step;
the floor makes that impossible regardless of the caller.

## Measured on the tiny fixture (2026-11-02)

```
run             sft/fp32_lora, 11 steps, seq 24, lora_r 2
disk written    53,732 bytes (adapter + tokenizer + one checkpoint)
optimizer.pt    not written
scheduler.pt    not written
rng_state.pth   not written
```

Reproduce with `server/tests/train/test_disk_light.py`; the run reports
`manifest.extras["disk_bytes_written"]`.

The number is small because the tiny model is small. What matters is the **shape**: the write set no
longer contains optimizer state, which on a real LoRA run is roughly the size of the trainable
parameters again for every checkpoint.

## What is deliberately not implemented

- **Delta snapshots.** Research shows ~31 % fewer snapshot bytes by storing bit deltas between
  checkpoints (JPDC 2018). It is a checkpoint-format project with its own corruption surface, and
  skipping optimizer state saves more than that on this workload.
- **4 KiB page-aligned writes.** The filesystem handles alignment; a userspace writer would have to
  bypass the page cache to matter.
- **Async checkpointing.** Only relevant once checkpoints are expensive; two of them per run is not.

## When this will change

If a resume path is added (conformance check C5), the optimizer state becomes useful again and the
trade-off must be re-measured — not silently reverted. The check reports `not_applicable` today and
would fail if a run claimed resume support without it.
