# embed_scale

Gemma-2 **scaled embedding gather** — the model's first op.

```
out[t, :] = TABLE[ids[t], :] * sqrt(D)
```

Gemma scales the token embedding by `sqrt(hidden_size)` (HF
`Gemma2TextScaledWordEmbedding`). At the real hidden `D = 2304` the scale is
`sqrt(2304) = 48.0`, exact in bf16. `fp32`, `T = 16` tokens; an indexed row
gather plus a constant scale. Self-checks against a numpy golden (`tohost 0`).
