# gemma_4norm

Validates the Gemma-2 decoder-layer **norm/residual wiring** — the "sandwich"
4-norm topology from HF `Gemma2DecoderLayer.forward`:

```
residual = h
h = input_layernorm(h)              # NORM 1  (pre-attention)
h = self_attn(h)
h = post_attention_layernorm(h)     # NORM 2  (post-attention)
h = residual + h                    # RESIDUAL 1  (bypasses norms 1 and 2)
residual = h
h = pre_feedforward_layernorm(h)    # NORM 3  (pre-FFN)
h = mlp(h)
h = post_feedforward_layernorm(h)   # NORM 4  (post-FFN)
h = residual + h                    # RESIDUAL 2  (bypasses norms 3 and 4)
```

Each norm is `Gemma2RMSNorm` (`rms(x) * (1.0 + weight)`, `eps = 1e-6`). `attn` and
`mlp` are stubbed as deterministic per-channel transforms — they are covered by
their own kernels; what is under test here is purely which weight applies where
and that each residual carries the pre-norm value across the sublayer. `fp32`,
`8 x 2304`.
