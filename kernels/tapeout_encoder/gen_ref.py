#!/usr/bin/env python3
"""gen_ref.py [out.h]: reference data for kernels/tapeout_encoder (sampled rows per stage).

Reproduces the host's integer-hash inputs (keep in sync with host.cpp and enc.h) and writes the
expected outputs of each implemented stage for a few sampled rows, using ref.py's quantization
rule.  The host compares codes exactly and reports the mismatch count: device float summation
order differs from numpy, so an occasional 1-code difference is expected.
"""
import pathlib
import sys

import numpy as np

import ref

S, D, H, DH, F = 256, 512, 4, 128, 2048
TM = TN = 128
TK = 256
ST_X, ST_LN, ST_WQKV, ST_WO, ST_W1, ST_W2, ST_WSC = 0, 1, 2, 3, 4, 5, 6
ST_QKV_SYN = 7          # synthetic qkv for the split test
WSC_BASE = {ST_WQKV: 0, ST_WO: 1 << 22, ST_W1: 2 << 22, ST_W2: 3 << 22}
SAMPLE_ROWS = [r for blk in range(S // 64) for r in range(blk * 64, blk * 64 + 8)]


def hash32(x):
    x = np.asarray(x, dtype=np.uint64).astype(np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def h(idx, stream):
    return hash32(np.asarray(idx, dtype=np.uint64) * 8 + stream)


def bf16_from_bits(b):
    return (np.asarray(b, dtype=np.uint32) << 16).view(np.float32)


def gen_x():
    hv = h(np.arange(S * D), ST_X)
    bits = ((hv >> 31) << 15) | ((124 + ((hv >> 8) & 3)) << 7) | (hv & 0x7F)
    return bf16_from_bits(bits).reshape(S, D), bits.astype(np.uint16).reshape(S, D)


def gen_ln(which):
    """which 1 or 2 -> (gamma, beta) float32 [D]."""
    d = np.arange(D)
    hg = h(d * 4 + 2 * (which - 1), ST_LN)
    hb = h(d * 4 + 2 * (which - 1) + 1, ST_LN)
    gamma = bf16_from_bits(0x3F80 | (hg & 0x7F))
    beta = bf16_from_bits(((hb >> 31) << 15) | ((120 + ((hb >> 8) & 3)) << 7) | (hb & 0x7F))
    return gamma, beta


def gen_weight(stream, k, n):
    """B-operand weight: e4m3 codes [K][N] (exponent field 1..7) and scales [K/32][N] 0x79..0x7B."""
    hv = h(np.arange(k * n), stream)
    codes = ((((hv >> 6) & 1) << 7) | ((1 + ((hv >> 3) & 0xFF) % 7) << 3) | (hv & 7)).astype(np.uint8)
    hs_ = h(np.arange((k // 32) * n) + WSC_BASE[stream], ST_WSC)
    scales = (0x79 + hs_ % 3).astype(np.uint8)
    return codes.reshape(k, n), scales.reshape(k // 32, n)


ST_GEN_FF, ST_GEN_O, ST_GEN_ATT, ST_GEN_FF2 = 10, 11, 12, 13   # stage tests (EN_TEST 13)


def gen_syn(stream, n):
    """bf16 [n], sign random, exponent 124..128, mantissa random (kernel st_gen)."""
    hv = h(np.arange(n), stream)
    bits = ((hv >> 31) << 15) | ((124 + ((hv >> 8) % 5)) << 7) | (hv & 0x7F)
    return bf16_from_bits(bits)


def gen_qkv_synthetic():
    """bf16 [S][3D], sign random, exponent 124..128, mantissa random (split-stage test input)."""
    hv = h(np.arange(S * 3 * D), ST_QKV_SYN)
    bits = ((hv >> 31) << 15) | ((124 + ((hv >> 8) % 5)) << 7) | (hv & 0x7F)
    return bf16_from_bits(bits).reshape(S, 3 * D)


def mx_matmul_rows(a_codes, a_scales_rowmajor, b_codes, b_scales):
    """Exact float64 C for A codes [M][K] (scales [M][K/32]) and B codes [K][N] (scales [K/32][N])."""
    da = ref.e4m3_decode(a_codes) * np.repeat(np.ldexp(1.0, a_scales_rowmajor.astype(np.int64) - 127), 32, axis=1)
    db = ref.e4m3_decode(b_codes) * np.repeat(np.ldexp(1.0, b_scales.astype(np.int64) - 127), 32, axis=0)
    return da @ db


def device_chain(x, g1, b1, g2, b2, weights, mesh_model=False):
    """The encoder layer exactly as the device stages compute it (quantization points, layouts).
    GEMMs exact (float64) or, with mesh_model, through mx_golden.  Attention exact on the split
    operands.  Returns a dict of bf16-rounded float32 intermediates."""
    import tempfile
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "lib" / "golden"))
    import golden

    def gemm(a_codes, a_scales, w):
        b_codes, b_scales = w
        if not mesh_model:
            return ref.to_bf16(mx_matmul_rows(a_codes, a_scales, b_codes, b_scales).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            c = golden.mx_matmul(a_codes, b_codes, a_scales.T.copy(), b_scales, a_codes.shape[0],
                                 b_codes.shape[1], a_codes.shape[1], tmpdir=tmp)
        return (c.astype(np.uint32) << 16).view(np.float32)

    out = {}
    h1c, h1s = ref.quantize_mx(ref.layernorm(x, g1, b1))
    out["qkv"] = qkv = gemm(h1c, h1s, weights["wqkv"])
    o = np.zeros((S, D), dtype=np.float32)
    for hd in range(H):
        q = ref.dequantize_mx(*ref.quantize_mx(qkv[:, hd * DH:(hd + 1) * DH]))
        k = ref.dequantize_mx(*ref.quantize_mx(qkv[:, D + hd * DH: D + (hd + 1) * DH]))
        vc, vs_ = ref.quantize_mx(qkv[:, 2 * D + hd * DH: 2 * D + (hd + 1) * DH].T.copy())
        o[:, hd * DH:(hd + 1) * DH] = ref.attention(q, k, ref.dequantize_mx(vc, vs_).T)
    out["o"] = o = ref.to_bf16(o)
    oc, os_ = ref.quantize_mx(o)
    out["att"] = att = gemm(oc, os_, weights["wo"])
    out["x1"] = x1 = ref.to_bf16(x + att)
    h2c, h2s = ref.quantize_mx(ref.layernorm(x1, g2, b2))
    ff = gemm(h2c, h2s, weights["w1"])
    out["g"] = g = ref.to_bf16(ref.gelu_tanh(ff).astype(np.float32))
    gc, gs = ref.quantize_mx(g)
    out["ff2"] = ff2 = gemm(gc, gs, weights["w2"])
    out["y"] = ref.to_bf16(x1 + ff2)
    return out


def emit_u16(f, name, arr):
    flat = np.asarray(arr, dtype=np.uint16).reshape(-1)
    f.write(f"static const uint16_t {name}[{flat.size}] = {{\n")
    for i in range(0, flat.size, 16):
        f.write("  " + ",".join(f"0x{v:04x}" for v in flat[i:i + 16]) + ",\n")
    f.write("};\n")


def emit_u8(f, name, arr):
    flat = np.asarray(arr, dtype=np.uint8).reshape(-1)
    f.write(f"static const uint8_t {name}[{flat.size}] = {{\n")
    for i in range(0, flat.size, 32):
        f.write("  " + ",".join(f"{v}" for v in flat[i:i + 32]) + ",\n")
    f.write("};\n")


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "ref_data.h")
    x, _ = gen_x()
    g1, b1 = gen_ln(1)
    h1 = ref.layernorm(x, g1, b1)
    h1_codes, h1_scales = ref.quantize_mx(h1)
    rows = np.array(SAMPLE_ROWS)
    # stage QKV: exact reference on the reference h1, and the mesh model's own error on it
    wq, wq_s = gen_weight(ST_WQKV, D, 3 * D)
    qkv_exact = mx_matmul_rows(h1_codes[rows], h1_scales[rows], wq, wq_s)
    qkv_exact_bits = ref.bf16_bits(qkv_exact.astype(np.float32))
    import tempfile
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "lib" / "golden"))
    import golden
    with tempfile.TemporaryDirectory() as tmp:
        model = golden.mx_matmul(h1_codes[rows], wq, h1_scales[rows].T.copy(), wq_s, len(rows), 3 * D, D, tmpdir=tmp)
    tof = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    ref_qkv_ppm = int(1e6 * np.linalg.norm(tof(model) - tof(qkv_exact_bits)) / np.linalg.norm(tof(qkv_exact_bits)))
    # stage SPLIT on the synthetic qkv: Q / K^T / V codes and scales
    syn = gen_qkv_synthetic()
    with open(out, "w") as f:
        f.write("// @generated by gen_ref.py -- do not edit\n#pragma once\n#include <stdint.h>\n")
        f.write(f"#define REF_NROWS {len(rows)}u\n")
        f.write("static const uint16_t ref_rows[REF_NROWS] = {" + ",".join(map(str, rows)) + "};\n")
        f.write("// stage LNQ1: h1 codes [REF_NROWS][D] and scales [REF_NROWS][D/32]\n")
        emit_u8(f, "ref_h1_codes", h1_codes[rows])
        emit_u8(f, "ref_h1_scales", h1_scales[rows])
        emit_u16(f, "ref_h1_y", ref.bf16_bits(h1[rows].astype(np.float32)))   # LN1 output before quantization
        f.write("// stage QKV: exact qkv (bf16) for the sample rows; mesh-model error vs it (ppm)\n")
        f.write(f"#define REF_QKV_PPM {ref_qkv_ppm}u\n")
        emit_u16(f, "ref_qkv", qkv_exact_bits)
        f.write("// stage SPLIT (synthetic qkv), heads 0 and 3: Q rows / K rows (pre-transpose) for the\n")
        f.write("// sample rows as [head][row][DH] codes + [head][row][DH/32] scales; V columns 0..15 as\n")
        f.write("// [head][d][S] codes + [head][d][S/32] scales\n")
        for nm, off in (("q", 0), ("k", D)):
            codes, scl = [], []
            for hd in (0, 3):
                c_, s_ = ref.quantize_mx(syn[rows][:, off + hd * DH: off + (hd + 1) * DH])
                codes.append(c_); scl.append(s_)
            emit_u8(f, f"ref_split_{nm}", np.stack(codes))
            emit_u8(f, f"ref_split_{nm}s", np.stack(scl))
        codes, scl = [], []
        for hd in (0, 3):
            c_, s_ = ref.quantize_mx(syn[:, 2 * D + hd * DH: 2 * D + hd * DH + 16].T.copy())
            codes.append(c_); scl.append(s_)
        emit_u8(f, "ref_split_v", np.stack(codes))
        emit_u8(f, "ref_split_vs", np.stack(scl))
        # stage FA on the synthetic qkv: exact attention on the dequantized split operands,
        # sample rows of every head, [head][REF_NROWS][DH] bf16
        o_ref = []
        for hd in range(H):
            q = ref.dequantize_mx(*ref.quantize_mx(syn[:, hd * DH:(hd + 1) * DH]))
            k = ref.dequantize_mx(*ref.quantize_mx(syn[:, D + hd * DH: D + (hd + 1) * DH]))
            vc, vs_ = ref.quantize_mx(syn[:, 2 * D + hd * DH: 2 * D + (hd + 1) * DH].T.copy())
            v = ref.dequantize_mx(vc, vs_).T
            o_ref.append(ref.attention(q, k, v)[rows])
        f.write("// stage FA (synthetic qkv): exact attention, [H][REF_NROWS][DH] bf16\n")
        emit_u16(f, "ref_fa_o", ref.bf16_bits(np.stack(o_ref).astype(np.float32)))
        # stage tests (EN_TEST 13): each SIMT stage on synthetic inputs, sample rows
        ff = gen_syn(ST_GEN_FF, S * F).reshape(S, F)
        g = ref.to_bf16(ref.gelu_tanh(ff).astype(np.float32))
        gc, gs = ref.quantize_mx(g)
        o_syn = gen_syn(ST_GEN_O, H * S * DH).reshape(H, S, DH).transpose(1, 0, 2).reshape(S, D)
        oc, os_ = ref.quantize_mx(o_syn)
        att = gen_syn(ST_GEN_ATT, S * D).reshape(S, D)
        x1 = ref.to_bf16(x + att)
        g2, b2 = gen_ln(2)
        h2y = ref.layernorm(x1, g2, b2)
        h2c, h2s = ref.quantize_mx(h2y)
        y = ref.to_bf16(x1 + gen_syn(ST_GEN_FF2, S * D).reshape(S, D))
        f.write("// stage tests (EN_TEST 13): GELU (bf16 and MX codes), O quantize, residual + LN2, residual 2\n")
        emit_u16(f, "ref_stg_g", ref.bf16_bits(g[rows]))
        emit_u8(f, "ref_stg_g_codes", gc[rows])
        emit_u8(f, "ref_stg_g_scales", gs[rows])
        emit_u8(f, "ref_stg_oa_codes", oc[rows])
        emit_u8(f, "ref_stg_oa_scales", os_[rows])
        emit_u16(f, "ref_stg_x1", ref.bf16_bits(x1[rows]))
        emit_u8(f, "ref_stg_h2_codes", h2c[rows])
        emit_u8(f, "ref_stg_h2_scales", h2s[rows])
        emit_u16(f, "ref_stg_h2_y", ref.bf16_bits(h2y[rows].astype(np.float32)))
        emit_u16(f, "ref_stg_y", ref.bf16_bits(y[rows]))
        # full layer: exact device-semantics chain at the sample rows, and the mesh model's error
        weights = {"wqkv": (wq, wq_s), "wo": gen_weight(ST_WO, D, D),
                   "w1": gen_weight(ST_W1, D, F), "w2": gen_weight(ST_W2, F, D)}
        g2, b2 = gen_ln(2)
        exact = device_chain(x, g1, b1, g2, b2, weights)
        model = device_chain(x, g1, b1, g2, b2, weights, mesh_model=True)
        f.write("// full layer (EN_TEST 4): exact device-semantics chain, sample rows, bf16;\n")
        f.write("// REF_<T>_MODEL_PPM = error of the same chain with mx_golden GEMMs vs exact\n")
        for key in ("qkv", "o", "att", "x1", "g", "ff2", "y"):
            e_, m_ = exact[key][rows].astype(np.float64), model[key][rows].astype(np.float64)
            ppm = int(1e6 * np.linalg.norm(m_ - e_) / np.linalg.norm(e_))
            f.write(f"#define REF_{key.upper()}_MODEL_PPM {ppm}u\n")
            emit_u16(f, f"ref_full_{key}", ref.bf16_bits(exact[key][rows]))
            print(f"  full-layer {key:4s}: mesh-model error {ppm} ppm")
    print(f"wrote {out}: {len(rows)} sample rows, mx model qkv error {ref_qkv_ppm} ppm")


if __name__ == "__main__":
    main()
