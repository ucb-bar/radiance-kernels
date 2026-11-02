#!/usr/bin/env python3
"""ref.py: numerical reference for the tapeout transformer encoder layer.

Pre-LN encoder block, S tokens, model width D, H heads of width DH, FFN width F:
    h1  = LN1(x)                       -> MX-FP8 (groups of 32 along D)
    qkv = h1 @ Wqkv                    (MX-FP8 GEMM, bf16 out)  [S][3D]
    o   = attention(q, k, v) per head  (bf16)                   [S][D]
    x1  = x + MXq(o) @ Wo              (bf16)
    h2  = LN2(x1)                      -> MX-FP8
    f   = GELU(h2 @ W1)                -> MX-FP8                [S][F]
    y   = x1 + f @ W2                  (bf16)

Every quantization step uses the device rule (quantize_mx below): one e8m0 scale per 32
consecutive elements, chosen so the block maximum maps into (0.9375, 1.875], then e4m3 with
round-to-nearest-even and subnormals flushed to zero.  Both choices are forced by the tapeout
mesh: larger operands saturate its e4 product/accumulator formats, and any subnormal operand
corrupts its accumulation (see the tapeout-e4m3 notes).

Matmuls here are exact (float64) on the quantized operands; the mesh's own accumulation adds
~4.7% Frobenius error per GEMM on top, which mx_golden models.
"""
import math

import numpy as np

GROUP = 32
TARGET_MAX = 1.875


# ---------------------------------------------------------------- bf16
def to_bf16(x):
    """float -> bf16 (round to nearest even) as float32."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def bf16_bits(x):
    return (to_bf16(x).view(np.uint32) >> 16).astype(np.uint16)


# ---------------------------------------------------------------- MX-FP8
def block_scale_exp(amax):
    """Smallest e with amax * 2^-e <= 1.875 (exact, from the float32 exponent field)."""
    bits = np.asarray(amax, dtype=np.float32).view(np.uint32).astype(np.int64)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & 0x7FFFFF
    return np.where(amax > 0, exp + (man > 0x700000), 0)


def e4m3_encode(v):
    """float32 -> e4m3 code, round to nearest even, subnormals flushed to zero, |v| <= 448."""
    v = np.asarray(v, dtype=np.float32)
    a = np.abs(v).astype(np.float64)
    sign = (np.signbit(v)).astype(np.int64) << 7
    exp = np.floor(np.log2(np.where(a > 0, a, 1.0))).astype(np.int64)
    exp = np.clip(exp, -6, 8)
    mant = a / np.ldexp(1.0, exp) - 1.0                  # in [0, 1) for normals
    q = np.rint(mant * 8.0)                              # numpy rint is half-to-even
    carry = q >= 8
    exp = np.where(carry, exp + 1, exp)
    q = np.where(carry, 0, q).astype(np.int64)
    over = (exp > 8) | ((exp == 8) & (q > 6))
    exp = np.where(over, 8, exp)
    q = np.where(over, 6, q)
    under = a < 2.0 ** -6 * (1 - 1 / 16)                 # rounds below the smallest normal
    code = sign | ((exp + 7) << 3) | q
    return np.where((a == 0) | under, 0, code).astype(np.uint8)


def e4m3_decode(c):
    c = np.asarray(c).astype(np.int64)
    e, m = (c >> 3) & 15, c & 7
    v = np.where(e > 0, np.ldexp(1.0 + m / 8.0, e - 7), np.ldexp(m / 8.0, -6))
    return np.where(c & 0x80, -v, v)


def quantize_mx(x):
    """x [R][C] (C multiple of 32) -> (codes [R][C] uint8, scale codes [R][C/32] uint8)."""
    r, c = x.shape
    xb = np.asarray(x, dtype=np.float32).reshape(r, c // GROUP, GROUP)
    amax = np.abs(xb).max(axis=2)
    e = block_scale_exp(amax)
    codes = e4m3_encode(xb * np.ldexp(1.0, -e)[:, :, None].astype(np.float32))
    return codes.reshape(r, c), np.clip(e + 127, 0, 254).astype(np.uint8)


def dequantize_mx(codes, scales):
    r, c = codes.shape
    v = e4m3_decode(codes).reshape(r, c // GROUP, GROUP)
    return (v * np.ldexp(1.0, scales.astype(np.int64) - 127)[:, :, None]).reshape(r, c)


# ---------------------------------------------------------------- ops
def layernorm(x, gamma, beta, eps=1e-5):
    x = x.astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + np.float32(eps)) * gamma + beta).astype(np.float32)


def gelu_tanh(x):
    x = x.astype(np.float64)
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def mx_matmul_exact(a_codes, a_scales, b_codes, b_scales_t):
    """A [M][K] with scales [M][K/32]; B given as B^T [N][K] with scales [N][K/32]."""
    return dequantize_mx(a_codes, a_scales) @ dequantize_mx(b_codes, b_scales_t).T


def attention(q, k, v):
    """softmax(q k^T / sqrt(d)) v, float64 (the FA kernel's own quantization is not modelled)."""
    s = q @ k.T / math.sqrt(q.shape[1])
    p = np.exp(s - s.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)) @ v


def encoder_layer(x, w, heads):
    """x [S][D] float32 (bf16-valued); w: dict of quantized weights and LN params.
    Returns a dict of every intermediate (bf16-rounded where the device stores bf16)."""
    s_len, d = x.shape
    dh = d // heads
    out = {}
    h1 = layernorm(x, w["ln1_g"], w["ln1_b"])
    out["h1_codes"], out["h1_scales"] = quantize_mx(h1)
    out["qkv"] = to_bf16(mx_matmul_exact(out["h1_codes"], out["h1_scales"], w["wqkv"], w["wqkv_s"]))
    q, k, v = (out["qkv"][:, i * d:(i + 1) * d] for i in range(3))
    o = np.concatenate([attention(q[:, h * dh:(h + 1) * dh], k[:, h * dh:(h + 1) * dh],
                                  v[:, h * dh:(h + 1) * dh]) for h in range(heads)], axis=1)
    out["o"] = to_bf16(o)
    out["o_codes"], out["o_scales"] = quantize_mx(out["o"])
    out["x1"] = to_bf16(x + to_bf16(mx_matmul_exact(out["o_codes"], out["o_scales"], w["wo"], w["wo_s"])))
    h2 = layernorm(out["x1"], w["ln2_g"], w["ln2_b"])
    out["h2_codes"], out["h2_scales"] = quantize_mx(h2)
    f = to_bf16(mx_matmul_exact(out["h2_codes"], out["h2_scales"], w["w1"], w["w1_s"]))
    out["f"] = f
    out["g"] = to_bf16(gelu_tanh(f).astype(np.float32))      # the device stores GELU output as bf16
    out["f_codes"], out["f_scales"] = quantize_mx(out["g"])
    out["y"] = to_bf16(out["x1"] + to_bf16(mx_matmul_exact(out["f_codes"], out["f_scales"], w["w2"], w["w2_s"])))
    return out


def random_layer(s_len=256, d=512, heads=4, ffn=2048, seed=0):
    """Random input and weights, weights quantized with the same rule (stored as W^T rows)."""
    rng = np.random.default_rng(seed)
    x = to_bf16(rng.standard_normal((s_len, d)).astype(np.float32))
    w = {}
    for name, (n_out, n_in) in {"wqkv": (3 * d, d), "wo": (d, d), "w1": (ffn, d), "w2": (d, ffn)}.items():
        wt = (rng.standard_normal((n_out, n_in)) / math.sqrt(n_in)).astype(np.float32)
        w[name], w[name + "_s"] = quantize_mx(wt)        # W^T [N][K], scales [N][K/32]
    for i in (1, 2):
        w[f"ln{i}_g"] = (1.0 + 0.1 * rng.standard_normal(d)).astype(np.float32)
        w[f"ln{i}_b"] = (0.1 * rng.standard_normal(d)).astype(np.float32)
    return x, w


if __name__ == "__main__":
    x, w = random_layer()
    out = encoder_layer(x, w, heads=4)
    for key in ("qkv", "o", "x1", "y"):
        print(f"{key:4s} shape {out[key].shape} rms {np.sqrt(np.mean(out[key].astype(np.float64) ** 2)):.4f}")
    codes = np.concatenate([out[k].reshape(-1) for k in ("h1_codes", "o_codes", "h2_codes", "f_codes")])
    sub = ((codes & 0x78) == 0) & ((codes & 7) != 0)
    print("subnormal codes emitted:", int(sub.sum()), " zeros:", int(((codes & 0x7F) == 0).sum()), "of", codes.size)
