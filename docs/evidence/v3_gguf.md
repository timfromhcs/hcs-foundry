# v3 evidence — GGUF, MoE, Vulkan

## GGUF export and quantization (real, Qwen3-0.6B)

```
source      E:\AILOCALTRAIN\data\models\Qwen\Qwen3-0.6B (model.safetensors 1.5 GB)
converter   D:\AI\llama-src\convert_hf_to_gguf.py
quantizer   D:\AI\bin\llama-vulkan\llama-quantize.exe
F16 output  1,509,347,136 bytes (1.41 GiB) in 209.4 s
Q4_K_M      484,219,712 bytes (462 MiB), ratio 0.321, 17.0 s
verification loaded in llama-server, answered "EXPORT_OK"
             91 tokens, 113.5 tok/s, ttft 0.79 s
```

## MoEfication

`server/tests/test_moe.py` builds a 2-layer GPT-2, converts it to 3 experts with top-2
routing, runs a forward pass, and asserts:

- finite logits,
- `moe_manifest.json` with `trained: false`,
- 2 blocks converted,
- invalid `top_k` and missing sources are refused.

The 4B conversion is refused by `estimate()`: cloning the FFN four times exceeds the
available RAM envelope. `--force` exists for machines with more memory.

## Vulkan auto-tuning (real, Qwen3-0.6B Q4_K_M)

```
ngl=0   (CPU)   53.9 tok/s
ngl=99  (Vulkan) 96.5 tok/s
speedup          1.79x
profile stored   data/settings -> vulkan_profiles[model_id]
```

This tunes inference. PyTorch has no Vulkan backend, so training is unaffected.

## Honest gap: dequantization

`llama-quantize` can raise a GGUF to F16/BF16. There is no `convert_gguf_to_hf.py` in
the discovered llama.cpp tree, so HF-format dequantization is **not** offered. The CLI
labels the operation `upcast`, not `convert`.
