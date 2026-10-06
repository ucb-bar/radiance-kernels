#!/usr/bin/env python3
"""GELU (tanh approximation) test data: X bf16 [N] in region 1, Y (poisoned) in region 2,
golden bf16 in region 3.  gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from nightly_data import Layout, bf16_bits, bf16_to_f32

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=262144)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
x = bf16_to_f32(bf16_bits(rng.standard_normal(a.n).astype(np.float32) * 2.0))
xd = x.astype(np.float64)
# x / (1 + exp(-2 z)) == 0.5 x (1 + tanh z), but stable for large negative x: there 1 + tanh(z)
# cancels to ~1 significant digit in float64 (x < -6 gave golden values off by many bf16 steps).
z = np.sqrt(2.0 / np.pi) * (xd + 0.044715 * xd ** 3)
y = xd / (1.0 + np.exp(-2.0 * z))
L = Layout("gelu")
L.const("GELU_N", f"{a.n}u")
L.add("GELU_X", 1, bf16_bits(x), "input bf16")
L.reserve("GELU_Y", 2, 2 * a.n, fill=0xEE, comment="output bf16 (poisoned 0xEEEE)")
L.add("GELU_G", 3, bf16_bits(y), "golden bf16")
L.write("data", "gelu_data.h")
