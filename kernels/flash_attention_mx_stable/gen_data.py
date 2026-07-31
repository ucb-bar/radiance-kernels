#!/usr/bin/env python3
"""Generate MXFP8 flash-attention kernel input data + golden references as a C header.

Reuses the verified golden model (flash_attention_model.py / fp8_matmul_model.py) so
the kernel inputs are in the *exact* layout the mxgemm Gemmini path consumes:

  QK^T gemm:  A = Q   [Sq, d]      grouped along d  -> A_in[M=Sq][K=d]      (fp8 e4m3)
              B = K^T [d, Sk]      grouped along d  -> B_in[K=d][N=Sk]      (fp8 e4m3)
              A_scales_row[GK=d/32][M=Sq]   B_scales_col[GK=d/32][N=Sk]    (E8M0, transposed)
  golden:     S_raw = mesh(Q@K^T) in bf16 (BEFORE softmax_scale)           -> QK_S_bf16[Sq][Sk]

Also emits Q/K/V (fp8+scales) and the fp32-ref attention output O for later PV stages.
Layout/encoders are identical to fp8_matmul_model.write_c_header_tiled.

Usage: python3 gen_data.py --Sq 128 --Sk 128 --d 64 --out include/fa_data.h
"""
import argparse
import os
import sys

LIB = "/scratch/yrh/ai-workspace/kernel-gen/radiance-kernels/lib/mxgemmini"
sys.path.insert(0, LIB)
import torch
import fp8_matmul_model as G
import flash_attention_model as FA

INPUT_SPEC = G.INPUT_SPEC      # fp8:e4m3
SCALE_SPEC = G.SCALE_SPEC      # fpe8m0
GROUP = G.GROUP                # 32


def fmt_rows(hex_rows):
    return ",\n".join("    { " + ", ".join(f"0x{h}" for h in row) + " }" for row in hex_rows)


def emit_fp8(f, name, mat, dims):
    codes, bits = G.tensor_to_custom_fp_codes(mat, INPUT_SPEC)
    hexr = G.codes_to_hex_rows(codes, bits)
    f.write(f"static const {G.c_type_for_bits(bits)} {name}{dims} = {{\n{fmt_rows(hexr)}\n}};\n\n")


def emit_scale(f, name, mat, dims):
    # E8M0 scale codes: 9-bit (sign+8exp+0man) then drop the sign bit -> uint8 exponent.
    codes, bits = G.tensor_to_custom_fp_codes(mat, SCALE_SPEC)
    bits -= 1
    hexr = G.codes_to_hex_rows(codes, bits)
    f.write(f"static const {G.c_type_for_bits(bits)} {name}{dims} = {{\n{fmt_rows(hexr)}\n}};\n\n")


def emit_bf16(f, name, mat, dims):
    codes, bits = G.tensor_to_custom_fp_codes(mat, "bf16")
    hexr = G.codes_to_hex_rows(codes, bits)
    f.write(f"static const uint16_t {name}{dims} = {{\n{fmt_rows(hexr)}\n}};\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Sq", type=int, default=128)
    ap.add_argument("--Sk", type=int, default=128)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block_n", type=int, default=0, help="key-block size Bk for streaming (0=Sk)")
    ap.add_argument("--out", type=str, default="include/fa_data.h")
    args = ap.parse_args()
    Sq, Sk, d = args.Sq, args.Sk, args.d
    assert d % GROUP == 0 and Sk % GROUP == 0 and Sq % 16 == 0
    Bk = args.block_n or Sk
    assert Sk % Bk == 0 and Bk % GROUP == 0, f"Bk={Bk} must divide Sk and be a multiple of {GROUP}"
    Nblk = Sk // Bk
    softmax_scale = 1.0 / (d ** 0.5)

    Q, K, V = FA.make_inputs(Sq, Sk, d, args.seed, peaked=False)

    # --- MX-quantize operands (same scheme as the golden) ---
    Q_fp8, Q_scales = FA.mx_quantize_cols(Q)          # [Sq,d], [Sq,d/32]
    K_fp8, K_scales = FA.mx_quantize_cols(K)          # [Sk,d], [Sk,d/32]
    Vt_fp8, Vt_scales = FA.mx_quantize_cols(V.t().contiguous())  # group V along Sk

    # QK^T gemm operands: A=Q, B=K^T
    A_in = Q_fp8                                      # [Sq,d]
    B_in = K_fp8.t().contiguous()                     # [d,Sk]
    A_scales_row = Q_scales.t().contiguous()          # [d/32, Sq]
    B_scales_col = K_scales.t().contiguous()          # [d/32, Sk]

    # --- goldens from the verified model ---
    trace = {}
    O_mx = FA.mx_attention_dense(Q, K, V, softmax_scale, trace=trace)
    S_raw = trace["S_raw"]                             # mesh QK^T, bf16, pre-softmax-scale
    O_ref = FA.attention_reference(Q, K, V, softmax_scale)

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    guard = "FA_DATA_H"
    GK = d // GROUP
    with open(out, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
        f.write(f"#define FA_SQ {Sq}\n#define FA_SK {Sk}\n#define FA_D {d}\n")
        f.write(f"#define FA_GK {GK}            // d/32 (QK^T contraction groups)\n")
        f.write(f"#define FA_GKV {Sk // GROUP}  // Sk/32 (PV contraction groups)\n")
        f.write(f"#define FA_BK {Bk}            // streaming key-block size\n")
        f.write(f"#define FA_NBLK {Nblk}        // Sk/Bk key blocks\n")
        f.write(f"#define FA_GKB {Bk // GROUP}  // Bk/32 (per-block PV contraction groups)\n")
        scale_bf16 = int(torch.tensor([softmax_scale], dtype=torch.float32)
                         .to(torch.bfloat16).view(torch.int16).item() & 0xFFFF)
        f.write(f"// softmax_scale = 1/sqrt(d) = {softmax_scale:.8f}\n")
        f.write(f"#define FA_SOFTMAX_SCALE_BF16 0x{scale_bf16:04x}\n\n")

        f.write("// ===== QK^T gemm: S = Q @ K^T (A=Q, B=K^T), contract over d =====\n")
        emit_fp8(f, "QK_A_in", A_in, "[FA_SQ][FA_D]")
        emit_fp8(f, "QK_B_in", B_in, "[FA_D][FA_SK]")
        emit_scale(f, "QK_A_scales_row", A_scales_row, "[FA_GK][FA_SQ]")
        emit_scale(f, "QK_B_scales_col", B_scales_col, "[FA_GK][FA_SK]")
        f.write("// golden mesh QK^T output (bf16, BEFORE * softmax_scale)\n")
        emit_bf16(f, "QK_S_bf16", S_raw, "[FA_SQ][FA_SK]")

        f.write("// ===== raw V (fp8) for the PV stage; grouped along Sk. V blocks are the\n"
                "//       natural row-slices V_in[j*Bk : (j+1)*Bk] / V_scales[j*Bk/32 : ...]. =====\n")
        emit_fp8(f, "V_in", Vt_fp8.t().contiguous(), "[FA_SK][FA_D]")
        emit_scale(f, "V_scales", Vt_scales.t().contiguous(), "[FA_GKV][FA_D]")

        # ===== streaming: K^T blocked as [Nblk][d][Bk] (each block's K_j^T contiguous, so
        #       block j's QK^T gemm reads B=QK_B_blocks[j] with a block-local stride Bk).
        Kt_blocks = torch.stack([K_fp8[j*Bk:(j+1)*Bk].t().contiguous() for j in range(Nblk)])
        Ks_blocks = torch.stack([K_scales[j*Bk:(j+1)*Bk].t().contiguous() for j in range(Nblk)])
        # 2D [Nblk*d][Bk] (not 3D -- a flat {row} list can't init a 3D C array); block j
        # is rows [j*d : (j+1)*d]. Scales likewise [Nblk*GK][Bk], block j at [j*GK].
        f.write("// ===== streaming K^T blocks: QK_B_blocks[j*FA_D : ] = K_j^T [d][Bk] =====\n")
        emit_fp8(f, "QK_B_blocks", Kt_blocks.reshape(Nblk * d, Bk), "[FA_NBLK*FA_D][FA_BK]")
        emit_scale(f, "QK_B_scales_blocks", Ks_blocks.reshape(Nblk * GK, Bk), "[FA_NBLK*FA_GK][FA_BK]")

        f.write("// ===== final reference attention output O (bf16) =====\n")
        emit_bf16(f, "O_ref_bf16", O_ref, "[FA_SQ][FA_D]")
        emit_bf16(f, "O_mx_bf16", O_mx, "[FA_SQ][FA_D]")

        f.write(f"#endif // {guard}\n")

    rel = (O_mx - O_ref).norm().item() / O_ref.norm().item()
    print(f"wrote {out}  (Sq={Sq} Sk={Sk} d={d})  MX-vs-ref rel err = {rel:.3e}")
    print(f"  S_raw range [{S_raw.min():.3f},{S_raw.max():.3f}]  "
          f"O_ref range [{O_ref.min():.3f},{O_ref.max():.3f}]")


if __name__ == "__main__":
    main()
