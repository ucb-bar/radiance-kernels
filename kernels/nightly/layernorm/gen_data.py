#!/usr/bin/env python3
"""LayerNorm test data.  y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta over each row,
x [L, D] bf16, gamma and beta [D] bf16, fp32 accumulation, y bf16.  X, gamma, beta in region 1;
Y (poisoned) in region 2; golden in region 3."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nightly_data import Layout, bf16_bits, bf16_to_f32

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=128)
ap.add_argument("--dim", type=int, default=2048)
ap.add_argument("--eps", type=float, default=1e-5)
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
# per-row offset and scale, so the mean is not near zero and the variance differs between rows
x = bf16_to_f32(bf16_bits(rng.standard_normal((a.rows, a.dim)).astype(np.float32)
                          * rng.uniform(0.25, 4.0, (a.rows, 1)).astype(np.float32)
                          + rng.uniform(-2.0, 2.0, (a.rows, 1)).astype(np.float32)))
g = bf16_to_f32(bf16_bits(1.0 + 0.1 * rng.standard_normal(a.dim).astype(np.float32)))
b = bf16_to_f32(bf16_bits(0.1 * rng.standard_normal(a.dim).astype(np.float32)))
xd = x.astype(np.float64)
mean = np.mean(xd, axis=1, keepdims=True)
var = np.mean((xd - mean) ** 2, axis=1, keepdims=True)
y = (xd - mean) / np.sqrt(var + a.eps) * g.astype(np.float64) + b.astype(np.float64)
L = Layout("layernorm")
L.const("LN_ROWS", f"{a.rows}u")
L.const("LN_DIM", f"{a.dim}u")
L.const("LN_EPS", f"{a.eps:.9e}f")
L.add("LN_X", 1, bf16_bits(x), "input bf16 [ROWS][DIM]")
L.add("LN_GAMMA", 1, bf16_bits(g), "gamma bf16 [DIM]")
L.add("LN_BETA", 1, bf16_bits(b), "beta bf16 [DIM]")
L.reserve("LN_Y", 2, 2 * a.rows * a.dim, fill=0xEE, comment="output bf16 (poisoned 0xEEEE)")
L.add("LN_G", 3, bf16_bits(y), "golden bf16")
L.write("data", "layernorm_data.h")
