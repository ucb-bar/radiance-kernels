# rmsnorm_qkv_fused

**RMSNorm fused into the matmul prologue**, ahead of an fp8 MX-Gemmini QKV
projection.

```
out[M, N] (bf16) = ( fp8( RMSNorm(X[M, K]) ) ) @ W[K, N]
```

- **Phase 0 (SIMT prologue — the fusion):** a SIMT schedule reads `X` (bf16),
  computes `RMSNorm` per row, quantizes to fp8 e4m3, and writes the codes directly
  into the exact GMEM buffer the mesh reads its A operand from. The normalized
  activation never round-trips to DRAM as a separate tensor.
- **Phase 1 (MX):** the mesh runs the fp8 matmul on the freshly-normalized A.
- **Phase 2 (SIMT epilogue):** copy the bf16 result out for verification.

The device `fp8_e4m3_to_code` is bit-identical to `lib/golden/mx_fp_math.h` but
libm-free (exponent taken from the float32 field; `rsqrt` is fast-inverse-sqrt +
Newton). Verified **bit-exact** against `mx_golden(fp8(RMSNorm(X)), W)`.
