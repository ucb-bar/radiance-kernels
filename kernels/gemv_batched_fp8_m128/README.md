# gemv_batched_fp8_m128

TinyLlama **batched-decode projection** on the MX mesh, fp8 weight-stationary,
batch `M = 128`.

```
out[M, N] = X[M, K] @ W[K, N],   K = 2048,  N = 128
```

The largest fp8 batch — every weight loaded into the `16x16` systolic array feeds
`M = 128` MACs, so the projection is firmly compute-bound rather than the
bandwidth-bound GEMV it is at `M = 1`. Square tile here (`128x128`); the family.s non-square path matters for the M=32/64 variants. See [`gemv_batched_fp8_m32`](../gemv_batched_fp8_m32) for the
rationale; fp4 at [`gemv_batched_fp4_m128`](../gemv_batched_fp4_m128).
