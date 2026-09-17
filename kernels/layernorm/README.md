# layernorm

SigLIP vision **LayerNorm** (SmolVLA / SmolVLM2-500M vision tower).

```
mean[t]  = (1/D) sum_j x[t, j]
var[t]   = (1/D) sum_j (x[t, j] - mean[t])^2          (biased, torch LayerNorm)
out[t,j] = (x[t, j] - mean[t]) / sqrt(var[t] + eps) * gamma[j] + beta[j]
```

The op the RMSNorm-only text kernels lack: **mean-subtraction** plus a **beta**
bias. `fp32`, one token-row per thread, grid-stride. Self-checks against a numpy
golden (`tohost 0 = pass`).
