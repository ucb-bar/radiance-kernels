# rope_qkv_fused

A fused **fp4 QKV projection + RoPE**, `head_dim = 64`.

```
out[M, N] (bf16) = RoPE( bf16( X[M, K] @ W[K, N] ), cos, sin )
```

- **Phase 1 (MX):** `C = X @ W` (fp4 e2m1, per-32-K e8m0 scales) is left in the
  scratchpad **without moving out to DRAM**.
- **Phase 2 (SIMT):** a separate schedule applies RoPE to the SMEM-resident `C`
  and writes the rotated bf16 straight to GMEM.

The projection output never round-trips through DRAM — that saved traffic is the
fusion win. RoPE uses the HF Llama `rotate_half` form with `cos`/`sin` duplicated
across halves; one thread per `(m, i)` pair reads both originals and writes both
outputs, so there is no cross-thread read-after-write hazard.
