# qkv_bias

Qwen2 **QKV bias-add epilogue** on the fp8 QKV-projection move-out.

```
out[token, c] (bf16) = proj[token, c] (bf16, fp8-GEMM move-out) + bias[c] (fp32)
```

The one op DeepSeek-R1-Distill-Qwen-1.5B (Qwen2) has that TinyLlama / Llama-2 lack:
a learned bias on the Q/K/V projection outputs, broadcast over tokens. Real dims:
hidden `K = 1536`, GQA 12 Q-heads : 2 KV-heads, so QKV width
`N = 1536 + 256 + 256 = 2048`. The GEMM stays fp8; the bias is precision-light
(`fp32`). The SIMT epilogue reads `proj` (bf16), adds `bias[col]`, truncates back
to bf16 — verified tolerance-exact (`tohost 0`).
