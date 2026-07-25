#!/usr/bin/env python3
"""Generate `data` for Qwen2 QKV bias-add EPILOGUE.

The one op DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 arch) has that TinyLlama/Llama-2 do NOT:
a learned bias added to the Q, K, V projection outputs (confirmed from the weight manifest:
self_attn.{q,k,v}_proj.bias present; o_proj has NO bias; no q_norm/k_norm; full RoPE).

  out[token, c] = qkv_proj[token, c] + bias[c]        (bias broadcast over tokens)

Real dims: hidden K=1536, head_dim=128, GQA 12 Q-heads : 2 KV-heads.
  QKV output width N = q(12*128=1536) + k(2*128=256) + v(2*128=256) = 2048.
  bias = concat(q_bias[1536], k_bias[256], v_bias[256]) -- one value per output column,
  i.e. per (head, head_dim) channel; column c belongs to head c//128.

The projection output feeding the epilogue is the GENUINE fp8 MX-Gemmini move-out: proj is
computed by mx_golden (fp8 e4m3 A/B with block scales) exactly as the accelerator produces it
(bf16). The GEMM stays fp8; the bias is precision-light (fp32). This is the SIMT epilogue that
runs on that move-out.

Golden = bf16_trunc( bf16_to_f(proj) + bias_f32 ), matching the kernel's arithmetic. Verify is
tolerance-based in float (a single IEEE add + bf16 truncation; any tie-rounding edge is absorbed
while a broken broadcast / wrong-column bias is still caught).
"""
import math
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MX_GOLDEN = pathlib.Path(__file__).resolve().parents[2] / "lib" / "golden" / "mx_golden"
GROUP = 32
FP8_CODE = 0

# Real DeepSeek-R1-Distill-Qwen-1.5B attention-projection dims.
HEAD_DIM = 128
N_Q_HEADS = 12
N_KV_HEADS = 2
HIDDEN = 1536
M = 64                     # tokens (a prefill tile); bias broadcasts over all M
K = HIDDEN                 # 1536
N_Q = N_Q_HEADS * HEAD_DIM      # 1536
N_KV = N_KV_HEADS * HEAD_DIM     # 256
N = N_Q + 2 * N_KV               # 2048  (q | k | v)
EPS = 0  # unused


def f_to_bf16(x):
    """float32 -> bf16 bits (truncate toward zero on the mantissa)."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u >> 16) & 0xFFFF).astype(np.uint16)


def bf16_to_f(h):
    return (h.astype(np.uint32) << 16).view(np.float32)


def rand_fp8(rng, n):
    exp = rng.integers(0, 8, size=n, dtype=np.uint8)
    mant = rng.integers(0, 8, size=n, dtype=np.uint8)
    sign = rng.integers(0, 2, size=n, dtype=np.uint8)
    return ((sign << 7) | (exp << 3) | mant).astype(np.uint8)


def emit_u16(f, name, dims, arr):
    flat = arr.reshape(-1)
    rows = ["    " + ", ".join(f"0x{v:04x}" for v in flat[i:i + 16]) for i in range(0, flat.size, 16)]
    f.write(f"static const uint16_t {name}{dims} = {{\n")
    f.write(",\n".join(rows))
    f.write("\n};\n")


def emit_u16_global(f, name, dims, arr):
    flat = arr.reshape(-1)
    rows = ["    " + ", ".join(f"0x{v:04x}" for v in flat[i:i + 16]) for i in range(0, flat.size, 16)]
    f.write(f"__global uint16_t {name}{dims} = {{\n")
    f.write(",\n".join(rows))
    f.write("\n};\n")


def emit_f32(f, name, dims, arr):
    flat = arr.astype(np.float32).reshape(-1)
    rows = ["    " + ", ".join(f"{v:.9e}f" for v in flat[i:i + 8]) for i in range(0, flat.size, 8)]
    f.write(f"static const float {name}{dims} = {{\n")
    f.write(",\n".join(rows))
    f.write("\n};\n")


def main():
    GK, GN = K // GROUP, N // GROUP
    rng = np.random.default_rng(0xB1A5)

    # --- fp8 QKV GEMM operands: proj[M][N] = mx_golden(A_fp8, W_fp8) (the genuine move-out) ---
    A = rand_fp8(rng, M * K).reshape(M, K)
    B = rand_fp8(rng, K * N).reshape(K, N)
    SA = rng.integers(0x7B, 0x83, size=(GK, M), dtype=np.uint8)   # A block scales ~2^0
    SB = rng.integers(0x7B, 0x83, size=(GK, N), dtype=np.uint8)   # B block scales ~2^0

    tmp = HERE / "_gen"
    tmp.mkdir(exist_ok=True)
    (tmp / "A.bin").write_bytes(A.tobytes())
    (tmp / "B.bin").write_bytes(B.tobytes())
    (tmp / "SA.bin").write_bytes(SA.tobytes())
    (tmp / "SB.bin").write_bytes(SB.tobytes())
    subprocess.run(
        [str(MX_GOLDEN), str(M), str(N), str(K), str(tmp / "A.bin"), str(tmp / "B.bin"),
         str(tmp / "SA.bin"), str(tmp / "SB.bin"), str(tmp / "C.bin"), str(FP8_CODE)],
        check=True, env={"PATH": "/usr/bin:/bin"})
    proj_bf16 = np.frombuffer((tmp / "C.bin").read_bytes(), dtype="<u2")[: M * N].reshape(M, N).copy()

    # --- learned QKV bias (fp32), one value per output channel, per-head structure ---
    q_bias = (0.30 * rng.standard_normal(N_Q)).astype(np.float32)
    k_bias = (0.30 * rng.standard_normal(N_KV)).astype(np.float32)
    v_bias = (0.30 * rng.standard_normal(N_KV)).astype(np.float32)
    bias = np.concatenate([q_bias, k_bias, v_bias]).astype(np.float32)
    assert bias.shape[0] == N

    # --- golden: bf16_trunc( bf16_to_f(proj) + bias ), broadcast bias over tokens ---
    proj_f = bf16_to_f(proj_bf16)                     # [M,N] exact widen
    out_f = (proj_f + bias[None, :]).astype(np.float32)
    gold = f_to_bf16(out_f)                            # [M,N] bf16

    with open(HERE / "data", "w") as f:
        f.write("// @generated by gen_data.py -- Qwen2 QKV bias-add epilogue on fp8 QKV move-out\n")
        f.write("#include <stdint.h>\n")
        f.write(f"#define QKV_M {M}\n#define QKV_N {N}\n")
        f.write(f"#define QKV_HEAD_DIM {HEAD_DIM}\n#define QKV_N_Q {N_Q}\n#define QKV_N_KV {N_KV}\n")
        f.write(f"#define VERIFY_COUNT {M * N}\n")
        emit_u16(f, "proj_bf16", "[QKV_M * QKV_N]", proj_bf16)
        emit_f32(f, "bias_f32", "[QKV_N]", bias)
        emit_u16_global(f, "out_raw", "[QKV_M * QKV_N]", np.zeros(M * N, dtype=np.uint16))
        emit_u16(f, "gold_raw", "[QKV_M * QKV_N]", gold)
    nz = int((gold != proj_bf16).sum())
    print(f"wrote data M={M} N={N} K={K}  proj!=out(bias changed) {nz}/{M*N} elems")


if __name__ == "__main__":
    if not MX_GOLDEN.exists():
        raise SystemExit(f"build mx_golden first: {MX_GOLDEN}")
    main()
