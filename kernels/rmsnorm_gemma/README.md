# rmsnorm_gemma

The Gemma-2 flavor of **RMSNorm**.

```
out[r, :] = x / sqrt(mean(x^2) + eps) * (1.0 + gamma),   eps = 1e-6
```

Gemma stores the norm weight zero-centered and applies it as `(1.0 + weight)` —
unlike the Llama / TinyLlama RMSNorm, which applies `gamma` directly with
`eps = 1e-5`. `fp32`, `16 x 2304` (real Gemma-2-2B hidden). Self-checks against a
numpy golden (`tohost 0 = pass`).
