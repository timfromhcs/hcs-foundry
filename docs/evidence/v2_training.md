# v2 evidence — training

## Environment

- Windows 11 Pro 26200, Ryzen 7 7735HS, Radeon 680M (unified memory), 19.8 GiB visible RAM.
- `torch 2.12.1+cpu` (no CUDA, no ROCm).
- Fixed stack: `huggingface-hub 1.33.0`, `transformers 5.12.1`, `peft 0.19.1`, `datasets 5.0.0`, `accelerate 1.14.0`.
  The previous failure was `huggingface-hub 2.0.0` breaking the transformers/peft imports; pinning `<2` fixed it.

## Smoke optimisation (synthetic, proves the update path)

```
steps 40, loss 2.193977 → 0.030316, max parameter delta 0.2901, device cpu
```

## Real LoRA run on Qwen3-0.6B

Pulled anonymously from the Hub with the new client (`foundry hf pull`), then trained:

```
base              Qwen/Qwen3-0.6B (model.safetensors, 1,503,300,328 bytes)
dataset           ds_b450b9a55b1745679655 (16 train records, rights confirmed)
lora              r=4, alpha=16, targets q/k/v/o/gate/down/up
sequence length   128
batch             1 (grad accum 1)
steps             2
duration          4.16 s
loss              6.043644 → 5.014467
trainable params  2,523,136 / 598,573,056 (0.42 %)
adapter           E:\AILOCALTRAIN\data\training-runs\proof-qwen06b\adapter
manifest          E:\AILOCALTRAIN\data\training-runs\proof-qwen06b\manifest.json
```

## Unit proof (CI-safe, offline)

`server/tests/test_sft.py` builds a 1-layer GPT-2 with a WordLevel tokenizer locally, runs 2 steps, and asserts:

- the run reaches the requested step count,
- a finite loss is recorded,
- the adapter config is written,
- the manifest exists and is labelled `lora_sft`,
- a missing model directory and unconfirmed rights are refused.

## 3B projection

| Basis | Value |
|---|---|
| measured 0.6B CPU throughput | 61.6 tok/s (short sequences) |
| assumed 3B CPU throughput | 6–15 tok/s |
| 200 steps × 512 tokens × grad-accum 8 | 819,200 tokens |
| projected wall time | 15–38 hours |
| verdict | not a practical fine-tune on this hardware |

No 3B weights were trained. No "3B model" claim is made.
