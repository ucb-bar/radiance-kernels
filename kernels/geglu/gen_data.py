#!/usr/bin/env python3
# Generate GeGLU coverage data at Gemma-2-2B FFN dims.
# GeGLU: C = gelu_pytorch_tanh(gate) * up   (Gemma-2 FFN gate activation).
# Real deployed Gemma-2 activation = gelu_pytorch_tanh (the TANH approximation),
# NOT erf-GELU. Golden here is the exact tanh form (numpy tanh), matching the
# real HuggingFace model. The kernel evaluates the algebraically-identical
# sigmoid rearrangement gelu(x) = x / (1 + exp(-2*C*(x + 0.044715 x^3))); the two
# agree to ~1e-7, well within TOLERANCE_REL=1e-3 / TOLERANCE_ABS=2e-4.
# gelu_pytorch_tanh reference and emit come from the shared golden module (lib/golden).
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "lib" / "golden"))
import golden as G

# Gemma-2-2B FFN: intermediate_size (FFN width) = 9216, hidden_size = 2304.
# GeGLU activation is elementwise over the 9216-wide FFN intermediate.
M = 4       # seq / token tile (kept so the single-lane verify + compute
            # finishes under cyclotron's 10M-cycle --timing budget)
N = 9216    # Gemma-2 FFN intermediate width (the real coverage dim)
total = M * N

rng = np.random.default_rng(0)
# gate_proj / up_proj outputs: zero-mean, a few-sigma spread to exercise the
# full GELU range (deep-negative saturation, linear region, positive tail).
A = (rng.standard_normal(total) * 2.0).astype(np.float32)   # gate pre-activation
B = (rng.standard_normal(total) * 1.0).astype(np.float32)   # up_proj output

gold = (G.gelu_tanh(A) * B.astype(np.float64)).astype(np.float32)

with open("data", "w") as f:
    f.write("// generated - GeGLU gelu(A)*B at Gemma-2-2B FFN dims (gelu_pytorch_tanh golden)\n")
    f.write(f"#define VERIFY_COUNT {total}\n")
    f.write(f"static const uint32_t M = {M};\n")
    f.write(f"static const uint32_t N = {N};\n")
    G.emit_f32(f, "A_raw", A, dims=f"[{total}]", trailing_comma=True)
    G.emit_f32(f, "B_raw", B, dims=f"[{total}]", trailing_comma=True)
    G.emit_f32(f, "gold_raw", gold, dims=f"[{total}]", trailing_comma=True)

print(f"wrote data: M={M} N={N} total={total}")
print("A range", float(A.min()), float(A.max()))
print("gold range", float(gold.min()), float(gold.max()))
