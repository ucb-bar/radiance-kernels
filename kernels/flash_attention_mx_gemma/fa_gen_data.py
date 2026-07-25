#!/usr/bin/env python3
"""Generate MXFP8 flash-attention kernel input data + goldens for the GQA + causal-mask
(Gemma-2: soft-cap + sliding-window) variant.

GQA: n_q query heads share n_kv KV heads in contiguous groups of grp = n_q // n_kv, so
query head h reads KV head h // grp. Causal mask: query row i has global position
q_pos0 + i; key col c of block j has global position j*Bk + c; entry masked (prob 0) when
key_pos > query_pos. Blocks whose min key position is above every query are skipped.

Data is emitted head-major (head folded into the leading array dimension) so the mesh gemm
core's existing 2-D pointer arithmetic works unchanged:
  QK_A_in            [NQ*Sq][d]                 fp8   (Q per query head)
  QK_A_scales_row    [NQ*GK][Sq]                E8M0  (Q scales, transposed, per query head)
  QK_B_blocks        [NKV*NBU*d][Bk]            fp8   (K_j^T per KV head, used blocks only)
  QK_B_scales_blocks [NKV*NBU*GK][Bk]           E8M0
  V_in               [NKV*NBU*Bk][d]            fp8   (V per KV head, used blocks only)
  V_scales           [NKV*NBU*GKB][d]           E8M0
Only the NBU = blocks_used leading key blocks are emitted (the rest are fully above the
diagonal -> masked to zero -> never read by the kernel).

Usage: python3 fa_gen_data.py --Sq 64 --Sk 256 --d 64 --block_n 64 \
                              --n_q 8 --n_kv 2 --q_pos0 64 --out include/fa_data.h
"""
import argparse
import os
import sys

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib", "mxgemmini")
sys.path.insert(0, LIB)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local flash_attention_model.py
import numpy as np
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
    codes, bits = G.tensor_to_custom_fp_codes(mat, SCALE_SPEC)
    bits -= 1
    hexr = G.codes_to_hex_rows(codes, bits)
    f.write(f"static const {G.c_type_for_bits(bits)} {name}{dims} = {{\n{fmt_rows(hexr)}\n}};\n\n")


def u16_codes(t):
    c, _ = G.tensor_to_custom_fp_codes(t, "bf16")
    return np.array(c, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Sq", type=int, default=64)
    ap.add_argument("--Sk", type=int, default=256)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block_n", type=int, default=64, help="key-block size Bk for streaming")
    ap.add_argument("--n_q", type=int, default=8, help="number of query heads")
    ap.add_argument("--n_kv", type=int, default=2, help="number of KV heads (n_q % n_kv == 0)")
    ap.add_argument("--q_pos0", type=int, default=64, help="global position of query row 0")
    ap.add_argument("--attn_softcap", type=float, default=50.0,
                    help="Gemma attn_logit_softcapping (0 disables). Confirmed 50.0.")
    ap.add_argument("--window", type=int, default=0,
                    help="sliding-window size (0 = full causal). Gemma even layers use 4096.")
    ap.add_argument("--qscale", type=float, default=50.0,
                    help="scale applied to Q so scaled scores reach the tanh-saturating "
                         "regime (|s| >> cap) -- otherwise the soft-cap is a near-linear no-op")
    ap.add_argument("--out", type=str, default="include/fa_data.h")
    args = ap.parse_args()
    Sq, Sk, d = args.Sq, args.Sk, args.d
    n_q, n_kv, q_pos0 = args.n_q, args.n_kv, args.q_pos0
    Bk = args.block_n
    attn_softcap = args.attn_softcap if args.attn_softcap > 0 else None
    window = args.window if args.window > 0 else None
    assert d % GROUP == 0 and Sk % GROUP == 0 and Sq % 16 == 0
    assert Sk % Bk == 0 and Bk % GROUP == 0, f"Bk={Bk} must divide Sk and be a multiple of {GROUP}"
    assert n_q % n_kv == 0, "n_q must be a multiple of n_kv"
    grp = n_q // n_kv
    Nblk = Sk // Bk
    GK = d // GROUP                       # d/32 (QK contraction groups)
    GKB = Bk // GROUP                     # Bk/32 (per-block PV contraction groups)
    nbu = FA.blocks_used(Sq, Sk, q_pos0, Bk)   # leading key blocks not fully above diagonal
    softmax_scale = 1.0 / (d ** 0.5)

    Q, K, V = FA.make_inputs_gqa(Sq, Sk, d, n_q, n_kv, args.seed)
    Q = Q * args.qscale         # push scaled scores into the tanh-saturating regime

    # ---- goldens from the verified Gemma model (soft-cap folded into online softmax +
    #      optional sliding-window mask) ----
    O_ref = FA.attention_reference_gemma(Q, K, V, softmax_scale, q_pos0,
                                         attn_softcap=attn_softcap, window=window)     # fp32
    O_mx = FA.mx_attention_flash_gemma(Q, K, V, softmax_scale, q_pos0, Bk,
                                       attn_softcap=attn_softcap, window=window)       # MX flash
    # report how strongly the soft-cap is exercised (fraction of scaled scores past the cap)
    if attn_softcap is not None:
        _grp = n_q // n_kv
        _sc = []
        for _h in range(n_q):
            _s = softmax_scale * (Q[_h] @ K[_h // _grp].t())
            _sc.append(_s.abs().flatten())
        _sc = torch.cat(_sc)
        print(f"  soft-cap exercise: |scaled score|>cap fraction = "
              f"{(_sc > attn_softcap).float().mean().item():.2f}  "
              f"max|s|={_sc.max().item():.1f}  cap={attn_softcap}")

    # ---- MX-quantize operands per head, keep only the NBU used key blocks ----
    A_rows, As_rows = [], []
    for h in range(n_q):
        qf8, qs = FA.mx_quantize_cols(Q[h])                # [Sq,d], [Sq,GK]
        A_rows.append(qf8)
        As_rows.append(qs.t().contiguous())                # [GK,Sq]
    A_in = torch.cat(A_rows, dim=0)                        # [NQ*Sq, d]
    A_scales_row = torch.cat(As_rows, dim=0)               # [NQ*GK, Sq]

    Kt_rows, Ks_rows, V_rows, Vs_rows = [], [], [], []
    for kv in range(n_kv):
        kf8, ks = FA.mx_quantize_cols(K[kv])               # [Sk,d], [Sk,GK]
        vt_f8, vt_s = FA.mx_quantize_cols(V[kv].t().contiguous())  # group V along Sk
        v_in = vt_f8.t().contiguous()                      # [Sk,d]
        v_s = vt_s.t().contiguous()                        # [Sk/32, d]
        for j in range(nbu):
            Kt_rows.append(kf8[j * Bk:(j + 1) * Bk].t().contiguous())      # [d,Bk]
            Ks_rows.append(ks[j * Bk:(j + 1) * Bk].t().contiguous())       # [GK,Bk]
            V_rows.append(v_in[j * Bk:(j + 1) * Bk])                        # [Bk,d]
            Vs_rows.append(v_s[j * GKB:(j + 1) * GKB])                      # [GKB,d]
    QK_B_blocks = torch.cat(Kt_rows, dim=0)                # [NKV*NBU*d, Bk]
    QK_B_scales_blocks = torch.cat(Ks_rows, dim=0)         # [NKV*NBU*GK, Bk]
    V_in = torch.cat(V_rows, dim=0)                        # [NKV*NBU*Bk, d]
    V_scales = torch.cat(Vs_rows, dim=0)                   # [NKV*NBU*GKB, d]

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    guard = "FA_DATA_H"
    with open(out, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
        f.write(f"#define FA_SQ {Sq}\n#define FA_SK {Sk}\n#define FA_D {d}\n")
        f.write(f"#define FA_GK {GK}            // d/32 (QK^T contraction groups)\n")
        f.write(f"#define FA_GKV {Sk // GROUP}  // Sk/32 (full-seq PV groups)\n")
        f.write(f"#define FA_BK {Bk}            // streaming key-block size\n")
        f.write(f"#define FA_NBLK {Nblk}        // Sk/Bk total key blocks\n")
        f.write(f"#define FA_GKB {GKB}          // Bk/32 (per-block PV contraction groups)\n")
        f.write(f"// ---- GQA + causal parameters ----\n")
        f.write(f"#define FA_NQ {n_q}           // query heads\n")
        f.write(f"#define FA_NKV {n_kv}         // KV heads\n")
        f.write(f"#define FA_GRP {grp}          // query heads per KV head (h -> h/FA_GRP)\n")
        f.write(f"#define FA_QPOS0 {q_pos0}     // global position of query row 0 (causal)\n")
        f.write(f"#define FA_NBLK_USED {nbu}    // leading key blocks not fully above diagonal\n")
        scale_bf16 = int(torch.tensor([softmax_scale], dtype=torch.float32)
                         .to(torch.bfloat16).view(torch.int16).item() & 0xFFFF)
        f.write(f"// softmax_scale = 1/sqrt(d) = {softmax_scale:.8f}\n")
        f.write(f"#define FA_SOFTMAX_SCALE_BF16 0x{scale_bf16:04x}\n")

        def _bf16(x):
            return int(torch.tensor([x], dtype=torch.float32)
                       .to(torch.bfloat16).view(torch.int16).item() & 0xFFFF)
        f.write(f"// ---- Gemma-2 attention soft-cap + sliding window ----\n")
        if attn_softcap is not None:
            f.write(f"#define FA_ATTN_SOFTCAP 1   // attn_logit_softcapping enabled\n")
            f.write(f"// cap = {attn_softcap}: each scaled score s -> cap*tanh(s/cap) BEFORE max/exp\n")
            f.write(f"#define FA_ATTN_SOFTCAP_BF16 0x{_bf16(attn_softcap):04x}     // cap\n")
            f.write(f"#define FA_ATTN_SOFTCAP_INV_BF16 0x{_bf16(1.0/attn_softcap):04x} // 1/cap\n")
        else:
            f.write(f"#define FA_ATTN_SOFTCAP 0\n")
            f.write(f"#define FA_ATTN_SOFTCAP_BF16 0x3c00\n#define FA_ATTN_SOFTCAP_INV_BF16 0x3c00\n")
        if window is not None:
            f.write(f"#define FA_SLIDING 1        // sliding-window attention (Gemma even layers)\n")
            f.write(f"#define FA_WINDOW {window}      // key_pos<=query_pos-WINDOW is masked (too old)\n")
        else:
            f.write(f"#define FA_SLIDING 0\n#define FA_WINDOW 0\n")
        f.write("\n")

        f.write("// ===== QK^T gemm operands: A = Q (per query head, head-major rows) =====\n")
        emit_fp8(f, "QK_A_in", A_in, "[FA_NQ*FA_SQ][FA_D]")
        emit_scale(f, "QK_A_scales_row", A_scales_row, "[FA_NQ*FA_GK][FA_SQ]")
        f.write("// ===== streaming K^T blocks per KV head: block (kv*NBU+j) at rows [.*FA_D] =====\n")
        emit_fp8(f, "QK_B_blocks", QK_B_blocks, "[FA_NKV*FA_NBLK_USED*FA_D][FA_BK]")
        emit_scale(f, "QK_B_scales_blocks", QK_B_scales_blocks, "[FA_NKV*FA_NBLK_USED*FA_GK][FA_BK]")
        f.write("// ===== V per KV head: block (kv*NBU+j) at rows [.*FA_BK] =====\n")
        emit_fp8(f, "V_in", V_in, "[FA_NKV*FA_NBLK_USED*FA_BK][FA_D]")
        emit_scale(f, "V_scales", V_scales, "[FA_NKV*FA_NBLK_USED*FA_GKB][FA_D]")

        # O golden (the MX-flash output the kernel computes, bf16 codes, head-major
        # [NQ*Sq][d]) committed IN the header so the offline O check is reproducible from
        # the PR alone. On-device FP verify of the mesh PV is platform-blocked on the
        # functional model, so O is checked offline via the RTL/SQLite-dump flow.
        f.write("// ===== O golden: MX-flash output the kernel computes (bf16 codes) =====\n")
        _o = np.asarray(u16_codes(O_mx.reshape(n_q * Sq, d)))
        _rows = ",\n".join("    { " + ", ".join(f"0x{int(v):04x}" for v in row) + " }" for row in _o)
        f.write(f"static const uint16_t O_gold[FA_NQ*FA_SQ][FA_D] = {{\n{_rows}\n}};\n\n")
        f.write(f"#endif // {guard}\n")

    # Offline O goldens (same data as the header's O_gold), written next to the header.
    _od = os.path.dirname(os.path.abspath(out))
    np.save(os.path.join(_od, "golden_O_gemma_mx_u16.npy"), u16_codes(O_mx.reshape(n_q * Sq, d)))
    np.save(os.path.join(_od, "golden_O_gemma_ref_u16.npy"), u16_codes(O_ref.reshape(n_q * Sq, d)))

    rel = (O_mx - O_ref).norm().item() / O_ref.norm().item()
    print(f"wrote {out}  (NQ={n_q} NKV={n_kv} grp={grp} Sq={Sq} Sk={Sk} d={d} Bk={Bk} "
          f"q_pos0={q_pos0} NBLK_USED={nbu})")
    print(f"  golden-model rel err  MX-flash(causal+GQA) vs fp32 ref = {rel:.3e}  "
          f"({100*rel:.2f}%)")
    print(f"  O_ref range [{O_ref.min():.3f},{O_ref.max():.3f}]  "
          f"O_mx range [{O_mx.min():.3f},{O_mx.max():.3f}]")


if __name__ == "__main__":
    main()
