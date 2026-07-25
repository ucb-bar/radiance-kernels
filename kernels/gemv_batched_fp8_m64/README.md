# gemv_batched_fp8_m64

TinyLlama **batched-decode projection** on the MX mesh, fp8 weight-stationary,
batch `M = 64`.

```
out[M, N] = X[M, K] @ W[K, N],   K = 2048,  N = 128
```

Batching `M` decode steps makes every weight loaded into the `16x16` systolic array
feed `M` MACs, so the projection (bandwidth-bound as an `M = 1` GEMV) becomes
compute-bound on the mesh. The tile is non-square (`TILE_M = 64`, `TILE_N = 128`).
See [`gemv_batched_fp8_m32`](../gemv_batched_fp8_m32) for the batch-scaling
rationale; other batches are `_m32` / `_m128` and fp4
[`gemv_batched_fp4_m128`](../gemv_batched_fp4_m128).
