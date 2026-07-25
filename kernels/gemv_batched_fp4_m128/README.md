# gemv_batched_fp4_m128

TinyLlama **batched-decode projection** on the MX mesh, **fp4** (e2m1)
weight-stationary, batch `M = 128`.

```
out[M, N] = X[M, K] @ W[K, N],   K = 2048,  N = 128
```

The fp4 counterpart of [`gemv_batched_fp8_m128`](../gemv_batched_fp8_m128): batching
`M = 128` decode steps makes each weight feed `M` MACs, turning the bandwidth-bound
`M = 1` GEMV into a compute-bound mesh GEMM. Square tile here (`128x128`); the family.s non-square path matters for the M=32/64 variants. See [`gemv_batched_fp8_m32`](../gemv_batched_fp8_m32) for the
batch-scaling rationale.
