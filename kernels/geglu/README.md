# geglu

The **GeGLU** feed-forward activation (Gemma-2 FFN, SmolVLA vision MLP).

```
C[m, n] = gelu(A[m, n]) * B[m, n]        gelu = gelu_pytorch_tanh
```

`A` = `gate_proj` activation, `B` = `up_proj` output; plain GELU (SmolVLA vision)
is the `B = 1` subset. The golden is the exact numpy tanh-GELU; the kernel uses the
algebraically-identical sigmoid form `x / (1 + exp(-2z))` (one `exp`, one
reciprocal — stays under the register-rename limit), matching to ~1e-7. Verified
at Gemma-2-2B FFN width `N = 9216`.
