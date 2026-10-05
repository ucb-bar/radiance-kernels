#!/usr/bin/env python3
"""MX GEMM test data: C[M,N] (bf16) = A[M,K] x B[K,N], MXFP8 (e4m3) or MXFP4 (e2m1) elements with
one e8m0 scale per 32 K-elements.  Golden from lib/golden/mx_golden (hardware accumulation).

Layout (device addresses in the generated gemm_data.h):
  region 1: A    fp8 [M][K]            | fp4 [M/2][K]   (nibble pairs along M)
            Asc  [MT][KS][TK/32][TM]   A scales pre-arranged per (m-tile, k-step), the order
                                       MX_LOAD_SCALES copies them into the act scale memory
  region 2: B    fp8 [K][N]            | fp4 [K][N/2]   (nibble pairs along N)
            Bsc  [NT][KS][TK/32][TN]
  region 3: C    bf16 [M][N] (poisoned 0xEEEE), G golden bf16 [M][N]
"""
import argparse, os, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parents[2] / "lib" / "golden"))
from nightly_data import Layout
import golden

ap = argparse.ArgumentParser()
ap.add_argument("--fmt", choices=["fp8", "fp4"], default="fp8")
ap.add_argument("--M", type=int, default=512)
ap.add_argument("--N", type=int, default=512)
ap.add_argument("--K", type=int, default=512)
ap.add_argument("--TM", type=int, default=128)
ap.add_argument("--TN", type=int, default=64)
ap.add_argument("--TK", type=int, default=0, help="0: 256 for fp8, 512 for fp4")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--pretile", action="store_true", help="store A/B blocks contiguous per (tile, k-step)")
ap.add_argument("--bdup", action="store_true", help="second copy of B (+ scales) in region 3 for cluster 1")
ap.add_argument("--out", default="data")
ap.add_argument("--header", default="gemm_data.h")
a = ap.parse_args()
TK = a.TK or (256 if a.fmt == "fp8" else 512)
M, N, K, TM, TN = a.M, a.N, a.K, a.TM, a.TN
assert M % TM == 0 and N % TN == 0 and K % TK == 0 and TK % 32 == 0
GK = K // 32
rng = np.random.default_rng(a.seed)
if a.fmt == "fp8":
    A = golden.rand_fp8(rng, M * K).reshape(M, K)
    B = golden.rand_fp8(rng, K * N).reshape(K, N)
else:
    A = golden.pack_axis0(golden.rand_fp4(rng, M * K).reshape(M, K))   # [M/2][K]
    B = golden.pack_axis1(golden.rand_fp4(rng, K * N).reshape(K, N))   # [K][N/2]
SA = rng.integers(0x7B, 0x83, size=(GK, M), dtype=np.uint8)
SB = rng.integers(0x7B, 0x83, size=(GK, N), dtype=np.uint8)
C = golden.mx_matmul(A, B, SA, SB, M, N, K, fmt=a.fmt, tmpdir=str(HERE / "_gen" / a.fmt))

MT, NT, KS, KB = M // TM, N // TN, K // TK, TK // 32
Asc = SA.reshape(KS, KB, MT, TM).transpose(2, 0, 1, 3).copy()   # [MT][KS][KB][TM]
Bsc = SB.reshape(KS, KB, NT, TN).transpose(2, 0, 1, 3).copy()   # [NT][KS][KB][TN]

Ad, Bd = A, B
if a.pretile:   # A [MT][KS][TM/vpb][TK], B [KS][NT][TK][TN/vpb] (bytes; vpb = 2 for fp4)
    vpb = 2 if a.fmt == "fp4" else 1
    Ad = np.ascontiguousarray(A.reshape(MT, TM // vpb, KS, TK).transpose(0, 2, 1, 3))
    Bd = np.ascontiguousarray(B.reshape(KS, TK, NT, TN // vpb).transpose(0, 2, 1, 3))

L = Layout("gemm")
for k, v in dict(GEMM_M=M, GEMM_N=N, GEMM_K=K, GEMM_TM=TM, GEMM_TN=TN, GEMM_TK=TK).items():
    L.const(k, f"{v}u")
L.const("GEMM_FP4", 1 if a.fmt == "fp4" else 0)
L.add("GEMM_A", 1, Ad, "A operand")
L.add("GEMM_ASC", 1, Asc, "A scales [MT][KS][TK/32][TM]")
L.add("GEMM_B", 2, Bd, "B operand")
L.add("GEMM_BSC", 2, Bsc, "B scales [NT][KS][TK/32][TN]")
if a.bdup:   # cluster 1 streams B from another L2 slice / DRAM channel
    L.add("GEMM_B2", 3, Bd, "copy of B for cluster 1")
    L.add("GEMM_BSC2", 3, Bsc, "copy of B scales for cluster 1")
L.const("GEMM_BDUP", 1 if a.bdup else 0)
L.const("GEMM_PRETILE", 1 if a.pretile else 0)
L.reserve("GEMM_C", 3, 2 * M * N, fill=0xEE, comment="C bf16 [M][N], poisoned 0xEEEE")
L.add("GEMM_G", 3, C, "golden C bf16")
L.write(a.out, a.header)
print(f"{a.fmt} M={M} N={N} K={K} tile {TM}x{TN}x{TK}; nonfinite golden: "
      f"{int(np.sum(((C >> 7) & 0xFF) == 0xFF))}")
