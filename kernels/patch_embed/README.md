# patch_embed

SigLIP **patch / conv embedding** (SmolVLA / SmolVLM2-500M vision tower):
`Conv2d(3 -> OC, kernel=16, stride=16)` + bias + learned position embedding.

Because `stride == kernel`, the conv is a non-overlapping **patchify + GEMM**:

```
patch p at grid (gy, gx), K = C*16*16:
  col[p, (c,ky,kx)] = image[c, gy*16+ky, gx*16+kx]
  out[p, oc]        = sum_K col[p, K] * W[oc, K] + bias[oc] + pos[p, oc]
```

The new piece vs a plain GEMM is the strided patchify addressing plus the
per-channel bias and position-embedding add. `fp32`; self-checks against a numpy
golden.

The kernel is **SMEM-staged**: the naive one-output-per-thread version has a
warp's 16 lanes each streaming a distinct 3 KB weight row through the 4 KB
no-landing-pads l0d, thrashing it ~77x over compute-bound. Instead it materializes
the patchified `col` matrix into cluster SMEM once, then maps `lane -> patch`,
`warp -> output channel`, so a warp's lanes share one broadcast weight load and
read distinct `col` rows from SMEM (no DRAM thrash).
