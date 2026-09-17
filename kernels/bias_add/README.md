# bias_add

The SigLIP vision projection / MLP **bias-add epilogue** (SmolVLA vision tower).

```
out[t, c] = proj[t, c] + bias[c]      (bias broadcast over token rows)
```

Every vision linear (`q/k/v/out_proj`, `fc1`, `fc2`) carries a learned bias; the
text tower does not. `fp32`, one token-row per thread, grid-stride. Self-checks
against a numpy golden (`tohost 0 = pass`).
