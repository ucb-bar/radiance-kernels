#!/usr/bin/env python3
"""RMSNorm test data.  y = x / sqrt(mean(x^2) + eps) * gamma, x [L, D] bf16, gamma [D] bf16,
fp32 accumulation, y bf16.  X, gamma in region 1; Y (poisoned) in region 2; golden in region 3."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nightly_data import Layout, bf16_bits, bf16_to_f32

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=128)
ap.add_argument("--dim", type=int, default=2048)
ap.add_argument("--eps", type=float, default=1e-5)
ap.add_argument("--seed", type=int, default=2)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
x = bf16_to_f32(bf16_bits(rng.standard_normal((a.rows, a.dim)).astype(np.float32)
                          * rng.uniform(0.25, 4.0, (a.rows, 1)).astype(np.float32)))
g = bf16_to_f32(bf16_bits(1.0 + 0.1 * rng.standard_normal(a.dim).astype(np.float32)))
xd = x.astype(np.float64)
inv = 1.0 / np.sqrt(np.mean(xd * xd, axis=1, keepdims=True) + a.eps)
y = xd * inv * g.astype(np.float64)
L = Layout("rmsnorm")
L.const("RMS_ROWS", f"{a.rows}u")
L.const("RMS_DIM", f"{a.dim}u")
L.const("RMS_EPS", f"{a.eps:.9e}f")
L.add("RMS_X", 1, bf16_bits(x), "input bf16 [ROWS][DIM]")
L.add("RMS_GAMMA", 1, bf16_bits(g), "gamma bf16 [DIM]")
L.reserve("RMS_Y", 2, 2 * a.rows * a.dim, fill=0xEE, comment="output bf16 (poisoned 0xEEEE)")
L.add("RMS_G", 3, bf16_bits(y), "golden bf16")
L.write("data", "rmsnorm_data.h")
