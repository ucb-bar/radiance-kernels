#!/usr/bin/env python3
"""Dump golden .npy references for each FA kernel stage (for fa_verify_out.py).

  golden_S_u16.npy      QK^T mesh output S_raw (bf16 codes)            [Sq,Sk]
  golden_Ss_u16.npy     S after *softmax_scale (bf16 codes)           [Sq,Sk]
  golden_m_f32.npy      row max of scaled S                           [Sq]
  golden_l_f32.npy      row sum of exp(S-m)                           [Sq]
  golden_P_u8.npy       softmax probs requantized to fp8 e4m3 codes   [Sq,Sk]
  golden_Pscales_u8.npy E8M0 scale codes for P (per 32-col block)     [Sq,Sk/32]
  golden_O_u16.npy      final attention output O (bf16 codes)         [Sq,d]
"""
import argparse, sys
LIB = "/scratch/yrh/ai-workspace/kernel-gen/radiance-kernels/lib/mxgemmini"
sys.path.insert(0, LIB)
import numpy as np, torch
import fp8_matmul_model as G, flash_attention_model as FA


def u16(t):
    c, _ = G.tensor_to_custom_fp_codes(t, "bf16")
    return np.array(c, dtype=np.uint16)


def u8_fp8(t):
    c, _ = G.tensor_to_custom_fp_codes(t, G.INPUT_SPEC)
    return np.array(c, dtype=np.uint8)


def u8_e8m0(t):
    c, bits = G.tensor_to_custom_fp_codes(t, G.SCALE_SPEC)
    return np.array(c, dtype=np.uint16).astype(np.uint8)  # drop sign bit -> exponent byte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Sq", type=int, default=128)
    ap.add_argument("--Sk", type=int, default=128)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block_n", type=int, default=0, help="key-block size for the flash golden (0=Sk)")
    a = ap.parse_args()
    if a.block_n == 0:
        a.block_n = a.Sk
    sc = 1.0 / (a.d ** 0.5)
    Q, K, V = FA.make_inputs(a.Sq, a.Sk, a.d, a.seed, peaked=False)
    tr = {}
    O = FA.mx_attention_dense(Q, K, V, sc, trace=tr)

    P = tr["P"]                                  # bf16 probs (post exp, pre requant)
    P_fp8, P_scales = FA.mx_quantize_cols(P)     # the fp8 + E8M0 the PV gemm consumes

    np.save("golden_S_u16.npy", u16(tr["S_raw"]))
    np.save("golden_Ss_u16.npy", u16(tr["S"]))
    np.save("golden_m_f32.npy", tr["m"].squeeze(-1).numpy().astype(np.float32))
    np.save("golden_l_f32.npy", tr["l"].squeeze(-1).numpy().astype(np.float32))
    np.save("golden_P_u8.npy", u8_fp8(P_fp8))
    np.save("golden_Pscales_u8.npy", u8_e8m0(P_scales))
    np.save("golden_O_u16.npy", u16(O))

    # Kernel-order O: normalize P by 1/l BEFORE requant+PV (the kernel folds /l into
    # softmax so PV yields final O). O = mx_gemm(requant(P/l), requant(V)).
    Pn = G.q_bf16_rne(tr["P"] / tr["l"])            # standard softmax probs (bf16)
    Pn_fp8, Pn_scales = FA.mx_quantize_cols(Pn)
    Vt_fp8, Vt_scales = FA.mx_quantize_cols(V.t().contiguous())
    O_normfirst = FA.mx_gemm(Pn_fp8, Pn_scales,
                             Vt_fp8.t().contiguous(), Vt_scales.t().contiguous())
    np.save("golden_O_normfirst_u16.npy", u16(O_normfirst))
    np.save("golden_Pnorm_u16.npy", u16(Pn))   # normalized softmax probs (bf16), kernel writes to P_GMEM
    np.save("golden_Pnfp8_u8.npy", u8_fp8(Pn_fp8))  # requantizer output: normalized P in fp8 e4m3 codes
    rel = (O_normfirst - O).norm().item() / O.norm().item()
    print(f"  O_normfirst vs O (norm-after) rel diff = {rel:.3e}")

    # ---- Streaming (flash) golden: online softmax over BLOCK_N key blocks. This is the
    # reference for the streaming kernel (per-block MX requant of P_blk). block_m=Sq (one
    # query block for now). O_flash is the exact target the streaming kernel reproduces.
    O_flash = FA.mx_attention_flash(Q, K, V, sc, block_m=a.Sq, block_n=a.block_n)
    np.save("golden_O_flash_u16.npy", u16(O_flash))
    rel_fd = (O_flash - O_normfirst).norm().item() / O_normfirst.norm().item()
    rel_ff = (O_flash - O).norm().item() / O.norm().item()
    print(f"  O_flash(block_n={a.block_n}) vs O_normfirst rel = {rel_fd:.3e} ; vs fp32-ref O = {rel_ff:.3e}")
    print("wrote goldens:",
          "S", tr["S_raw"].shape, "P", P_fp8.shape, "Pscales", P_scales.shape, "O", O.shape)
    print("  m[:4]=", tr["m"].squeeze(-1)[:4].tolist())
    print("  l[:4]=", tr["l"].squeeze(-1)[:4].tolist())


if __name__ == "__main__":
    main()
