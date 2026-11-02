#!/usr/bin/env python3
"""gen_data.py M N K TM TN TK [out.h] [--unit-scales]: golden header for kernels/tapeout_gemm.

The host program generates the operands itself (gemm_data.h only carries the shape and the
reference), with the integer hash below; this script reproduces the same bytes and writes C from
exact arithmetic (rounded to bf16) plus the Frobenius error that the MX-Gemmini accumulation model
(lib/golden/mx_golden) has against it.  The tapeout mesh is not bit-exact with mx_golden, but its
error against exact math matches mx_golden's (measured 4.69% vs 4.66%), so the host gates on that.
Keep hash32() and the operand/scale formulas identical to host.cpp.
"""
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "lib" / "golden"))
import golden  # noqa: E402

STREAM_A, STREAM_B, STREAM_SA, STREAM_SB = 0, 1, 2, 3


def hash32(x):
    """lowbias32 integer hash on uint32 arrays; identical to tg_hash32() in host.cpp."""
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def stream(idx, s):
    return hash32(idx.astype(np.uint32) * np.uint32(4) + np.uint32(s))


def fp8_operand(n, s, cols=None):
    """e4m3 codes with exponent field 0..7 (|v| < 2), as golden.rand_fp8 bounds them.
    `cols` (row length) is needed for PATTERN 1's identity A."""
    idx = np.arange(n, dtype=np.uint32)
    h = stream(idx, s)
    if PATTERN == 5:
        if s == STREAM_A:
            r, c = idx // cols, idx % cols
            return np.where(c == r % cols, 0x38, 0x00).astype(np.uint8)
        return np.full(n, 0x38, dtype=np.uint8)
    if PATTERN == 1 and s == STREAM_A:
        r, c = idx // cols, idx % cols
        return np.where(c == r % cols, 0x38, 0x00).astype(np.uint8)
    if PATTERN == 2:
        return np.full(n, 0x38, dtype=np.uint8)
    exp = (h >> 3) & 7
    if PATTERN == 3:
        exp = 1 + (exp % 7)
    mant = h & 7
    if PATTERN == 6 and s == STREAM_A:   # A in [1, 448]: the hardware requantizer's code range
        exp = exp + 8
        mant = np.where((exp == 15) & (mant == 7), 6, mant)   # 0x7F / 0xFF is NaN
    # Subnormal operands (exponent field 0, mantissa != 0) corrupt dense accumulations on the
    # tapeout mesh (+-448 / inf outputs; 0x04-0x07 also decode wrong), so flush them to zero, as an
    # MX-FP8 quantizer for this part must.
    mant = np.where(exp == 0, 0, mant)
    return (((h >> 6) & 1) << 7 | exp << 3 | mant).astype(np.uint8)


UNIT_SCALES = False   # --unit-scales: every scale 0x7F (2^0); must match -DTG_UNIT_SCALES
PATTERN = 0           # --pattern=N, must match -DTG_PATTERN=N (diagnostic operand patterns):
                      #   1: A = identity (1.0 where k == m % K), B hashed  -> C == B rows
                      #   2: A = B = 1.0 everywhere                         -> C == K
                      #   3: hashed operands without subnormals (exponent field 1..7)
                      #   5: scale probe: A = identity, B = 1.0, SA[g][m] = 127 + (m + 5g) % 4,
                      #      SB[g][n] = 127 + 4 * ((n + 3g) % 4); C's exponent names both scales
                      #   6: as 0, but A's exponent field is 8..15 (|a| in [1, 448]): the code
                      #      range the hardware MX requantizer emits (block max in [256, 448])


def scales(n, s, cols=None):
    """e8m0 codes 0x7B..0x82 (2^-4 .. 2^3), or all 0x7F with --unit-scales.
    `cols` (M for A, N for B) is needed for PATTERN 5."""
    if PATTERN == 5:
        i = np.arange(n, dtype=np.int64)
        g, c = i // cols, i % cols
        if s == STREAM_SA:
            return (127 + (c + 5 * g) % 4).astype(np.uint8)
        return (127 + 4 * ((c + 3 * g) % 4)).astype(np.uint8)
    if UNIT_SCALES:
        return np.full(n, 0x7F, dtype=np.uint8)
    return (0x7B + (stream(np.arange(n, dtype=np.uint32), s) & 7)).astype(np.uint8)


def decode_e4m3(codes):
    c = codes.astype(np.int64)
    e, mt = (c >> 3) & 15, c & 7
    v = np.where(e > 0, np.ldexp(1.0 + mt / 8.0, e - 7), np.ldexp(mt / 8.0, -6))
    return np.where(c & 0x80, -v, v)


def exact_matmul(a, b, sa, sb):
    """C = sum over 32-groups of (A_g @ B_g) * 2^(sa+sb-254), in float64."""
    k = a.shape[1]
    da, db = decode_e4m3(a), decode_e4m3(b)
    c = np.zeros((a.shape[0], b.shape[1]))
    for g in range(k // 32):
        part = da[:, g * 32:(g + 1) * 32] @ db[g * 32:(g + 1) * 32, :]
        c += part * np.ldexp(1.0, sa[g].astype(np.int64)[:, None] + sb[g].astype(np.int64)[None, :] - 254)
    return c


def main():
    global UNIT_SCALES, PATTERN
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    UNIT_SCALES = "--unit-scales" in sys.argv
    for x in sys.argv:
        if x.startswith("--pattern="):
            PATTERN = int(x.split("=")[1])
    m, n, k, tm, tn, tk = map(int, args[0:6])
    out = pathlib.Path(args[6] if len(args) > 6 else "gemm_data.h")
    assert m % tm == 0 and n % tn == 0 and k % tk == 0 and tk % 32 == 0
    a = fp8_operand(m * k, STREAM_A, cols=k).reshape(m, k)
    b = fp8_operand(k * n, STREAM_B).reshape(k, n)
    sa = scales((k // 32) * m, STREAM_SA, cols=m).reshape(k // 32, m)   # logical [K/32][M]
    sb = scales((k // 32) * n, STREAM_SB, cols=n).reshape(k // 32, n)   # logical [K/32][N]
    with tempfile.TemporaryDirectory() as tmp:
        c = golden.mx_matmul(a, b, sa, sb, m, n, k, tmpdir=tmp)
    # Exact reference: decoded operands, per-group scales, float64 math, rounded to bf16.
    exact = exact_matmul(a, b, sa, sb).astype(np.float32)
    exact_bf16 = ((exact.view(np.uint32) + 0x7FFF + ((exact.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)
    to_f = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    rs = max(1, -(-(m * n) // 65536))
    ref_ppm = int(1e6 * np.linalg.norm(to_f(c[::rs]) - to_f(exact_bf16[::rs])) /
                  np.linalg.norm(to_f(exact_bf16[::rs])))
    with open(out, "w") as f:
        f.write("// @generated by gen_data.py -- do not edit\n#pragma once\n#include <stdint.h>\n")
        f.write(f"#define TG_M {m}u\n#define TG_N {n}u\n#define TG_K {k}u\n")
        f.write(f"#define TG_TM {tm}u\n#define TG_TN {tn}u\n#define TG_TK {tk}u\n")
        f.write(f"// Frobenius error of mx_golden (the hardware accumulation model) vs exact, ppm\n")
        f.write(f"#define TG_REF_FROB_PPM {ref_ppm}u\n")
        # Large problems: check every RS-th row only, so the host image stays small.
        rs = max(1, -(-(m * n) // 65536))
        f.write(f"// C from exact arithmetic, rounded to bf16, rows 0, {rs}, {2 * rs}, ...\n")
        f.write(f"#define TG_CHECK_ROW_STRIDE {rs}u\n")
        f.write(f"static const uint16_t tg_exact_c[{len(range(0, m, rs))} * TG_N] = {{\n")
        flat = exact_bf16[::rs].reshape(-1)
        for i in range(0, flat.size, 16):
            f.write("  " + ", ".join(f"0x{v:04x}" for v in flat[i:i + 16]) + ",\n")
        f.write("};\n")
    print(f"wrote {out} (M={m} N={n} K={k} tile {tm}x{tn}x{tk}) mx_golden vs exact {ref_ppm} ppm")


if __name__ == "__main__":
    main()
