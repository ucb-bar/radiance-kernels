#!/usr/bin/env python3
"""Emit a padded-128 MX data header for a sub-byte radiance capsule (R6 fp6 / R7 fp4).

The capsule tile is 32x32x32; the proven mxgemm_lib mishandles a lone sub-byte PE tile,
so (exactly as for R5) we pad to a 128x128x128 tile with the real operands in the
top-left corner and extract the top-left 32x32 of C for Y0.

Padding contributes nothing to the top-left output:
  * fp4 padding nibbles = 0 -> code 0 -> 0.0 exactly.
  * fp6 padding nibbles = 0 -> LUT[0] (a small non-zero fp6 value), so the K-padding
    block scales are set to e8m0 code 0. The co-model computes the block exponent as
    clamp(SA+SB-127, 0, 254); 0+0 clamps to 0 -> 2^-127, which drives every padding
    contribution ~1e-40, far below a bf16 ULP of the O(1e3) outputs.

Everything (codes, LUT palette, e8m0 scales, Y0) is taken from the capsule golden.yaml,
which is bit-exact to the MX RTL -- no re-encoding, the golden is the source of truth.

Usage: gen_r6r7_pad.py <capsule_dir> <out_header.h>   (fmt read from golden)
"""
import sys
from pathlib import Path

import yaml

PADW = 128
GROUP = 32


def emit2d(f, ctype, name, rows, cols, data):
    f.write(f"static const {ctype} {name}[{rows}][{cols}] = {{\n")
    for r in range(rows):
        f.write("  {" + ",".join(str(int(x)) for x in data[r]) + "},\n")
    f.write("};\n\n")


def pack96(codes16):
    """16 six-bit codes -> three LE uint32 (96 bits), matching co-model unpack_lut_96bit."""
    v = 0
    for i, c in enumerate(codes16):
        v |= (int(c) & 0x3F) << (6 * i)
    return [v & 0xFFFFFFFF, (v >> 32) & 0xFFFFFFFF, (v >> 64) & 0xFFFFFFFF]


def main():
    capsule_dir, outp = sys.argv[1], sys.argv[2]
    g = yaml.safe_load((Path(capsule_dir) / "golden.yaml").read_text())
    oc = g["oracle_provenance"]["inputs"]["operand_codes"]
    inp = g["oracle_provenance"]["inputs"]
    fmt = oc["fmt"]  # fp6_e3m2 | fp4_e2m1
    is_fp6 = fmt.startswith("fp6")
    M, N, K = oc["M"], oc["N"], oc["K"]
    assert M <= PADW and N <= PADW and K <= PADW
    A = oc["A_bytes"]          # packed [M/2][K] nibble-along-M
    B = oc["B_bytes"]          # packed [K][N/2] nibble-along-N
    SA = inp["SA_e8m0_codes"]  # [GK_real][M]
    SB = inp["SB_e8m0_codes"]  # [GK_real][N]
    GK = PADW // GROUP         # 4

    # A_in_hw [PADW/2][PADW]: real packed A into rows 0..M/2-1, cols 0..K-1.
    Ah = [[0] * PADW for _ in range(PADW // 2)]
    for r in range(M // 2):
        for k in range(K):
            Ah[r][k] = A[r * K + k]
    # B_in [PADW][PADW/2]: real packed B into rows 0..K-1, cols 0..N/2-1.
    Bh = [[0] * (PADW // 2) for _ in range(PADW)]
    for k in range(K):
        for c in range(N // 2):
            Bh[k][c] = B[k * (N // 2) + c]

    # Scales [GK][PADW]. Real K-group 0: golden scales in cols 0..M/N-1, 0x7f elsewhere
    # (M/N-padding output is discarded). Padding K-groups 1..3: e8m0 code 0 -> ~zero.
    As = [[0] * PADW for _ in range(GK)]
    Bs = [[0] * PADW for _ in range(GK)]
    for m in range(PADW):
        As[0][m] = SA[0][m] if m < M else 0x7F
    for n in range(PADW):
        Bs[0][n] = SB[0][n] if n < N else 0x7F

    with open(outp, "w") as f:
        f.write(f"// @generated padded-128 MX data header for {fmt}; real {M}x{N}x{K} in the corner.\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define MATMUL_M {PADW}\n#define MATMUL_K {PADW}\n#define MATMUL_N {PADW}\n")
        f.write(f"#define MATMUL_GK {GK}\n#define MATMUL_GN {GK}\n\n")
        emit2d(f, "uint8_t", "A_in_hw", PADW // 2, PADW, Ah)
        emit2d(f, "uint8_t", "B_in", PADW, PADW // 2, Bh)
        emit2d(f, "uint8_t", "A_scales_row", GK, PADW, As)
        emit2d(f, "uint8_t", "B_scales_col", GK, PADW, Bs)
        if is_fp6:
            # One 16-entry palette replicated to every LUT slot, so whatever li =
            # (row >> granularity) the co-model computes, it reads the same palette.
            pa = pack96(oc["lutA"][0])
            pb = pack96(oc["lutB"][0])
            n_lut = PADW // 2  # load_lut stages TILE>>granularity(=1) = 64 entries
            emit2d(f, "uint32_t", "A_lut", n_lut, 3, [pa] * n_lut)
            emit2d(f, "uint32_t", "B_lut", n_lut, 3, [pb] * n_lut)
            emit2d(f, "uint32_t", "C_lut", n_lut, 3, [[0, 0, 0]] * n_lut)
    print(f"wrote {outp}  fmt={fmt} M={M} N={N} K={K} -> padded {PADW}")


if __name__ == "__main__":
    main()
