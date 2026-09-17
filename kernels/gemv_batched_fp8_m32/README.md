# gemv_batched_fp8_m32

TinyLlama **batched-decode projection** on the MX mesh, fp8 weight-stationary,
batch `M = 32`.

```
out[M, N] = X[M, K] @ W[K, N],   K = 2048,  N = 128
```

The key decode lever: at `M = 1` this projection is a SIMT GEMV (~8 cyc/MAC,
weight-**bandwidth**-bound — each weight feeds one MAC). Batching `M` decode steps
makes every weight loaded into the `16x16` systolic array feed `M` MACs, so
arithmetic intensity climbs to ~`M` and the projection becomes **compute**-bound on
the mesh. The tile is non-square (`TILE_M = 32`, `TILE_N = 128`), so this copy of
`mxgemm_lib` differentiates the A/B scale-factor counts to verify `TILE_M != TILE_N`.

Batch variants: `M = 32 / 64 / 128` (this dir, `_m64`, `_m128`) and fp4 at
[`gemv_batched_fp4_m128`](../gemv_batched_fp4_m128).
