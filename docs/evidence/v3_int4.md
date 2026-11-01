# v3 evidence — INT4 (QLoRA-style) training

Implemented without `bitsandbytes`, so it works on CPU and on this AMD machine.

## Scheme

- Group-wise symmetric absmax quantization, group size 64.
- Two 4-bit values packed per `uint8`; per-group `float32` scales.
- Base weights stay frozen; each target linear becomes `Int4Linear` wrapped by a LoRA
  adapter (`lora_A` random, `lora_B` zero, so the initial delta is exactly zero).
- Only LoRA parameters are trained; the forward pass dequantizes one layer at a time.
- 4 bits + scales ≈ **7× smaller** than fp32 for the frozen weights.

## Real run on Qwen3-0.6B

```
dataset            ds_d642c29dc9bf42a984ce (user's cleaned agent logs, 49 train)
model              Qwen3-0.6B
quantized layers   196
LoRA parameters    2,523,136
steps              2, sequence 128, grad-accum 1, r=4
train loss         5.721064 → 4.859477
eval loss          4.2595 → 4.1792
duration           13.0 s (after load)
peak RSS           4,862 MB   (dominated by the fp32 load peak)
adapter            data/training-runs/v3-int4/adapter/lora_int4.pt
manifest           data/training-runs/v3-int4/manifest.json
```

## Tests

`server/tests/test_int4.py`:

- quantization round-trip relative error < 15 %,
- INT4 storage more than 3× smaller than fp32,
- 2-step training on a tiny GPT-2 (Conv1D-aware conversion), adapter + manifest written,
- missing model refused.

## Honest limits

- **Loading still needs the fp32 peak.** The model is materialised in fp32, then
  quantized layer by layer. A 4B model therefore still needs ~17 GB at load time even
  though training itself would only use ~2 GB of weights. A streaming INT4 loader
  (quantize while reading shards) is the next step and is not implemented.
- **Dequantization costs throughput.** INT4 is a memory trade, not a speed feature.
- **Adapter format is custom** (`lora_int4.pt` + manifest). It is not a PEFT directory;
  loading requires the same conversion code.
