#!/usr/bin/env python3
"""fp6 vs fp8 vs fp4 flash-attention accuracy (GQA + causal, TinyLlama prefill dims).

Isolates the PRECISION effect: identical flash-attention dataflow (streaming online
softmax, per-block MX requant, corr rescale, deferred 1/l), identical block MX scaling,
the ONLY difference is the operand grid the Q/K/V/P tensors are rounded to:
    fp8:e4m3 (3 mantissa) | fp6:e3m2 (2 mantissa) | fp4:e2m1 (1 mantissa).
Reports ||O_mx - O_fp32ref|| / ||O_fp32ref|| per precision -> where fp6 sits vs fp8's ~6%.
"""
import os, sys
import numpy as np
import torch

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib", "mxgemmini")
sys.path.insert(0, LIB)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8_matmul_model as G
import flash_attention_model as FA

GROUP = G.GROUP


def mx_quant_cols(M, spec):
    """Per-row MX quantize over 32-col groups, THEN round scaled values to the `spec` grid
    (mirrors the fp8 path's matrix_mx_requantize + _fp8_round, generalized to any spec)."""
    q, s = G.matrix_mx_requantize(M, spec)
    qf = G.make_fp_quantizer(spec, "nearest")(q)     # actual grid rounding (precision loss)
    return qf, s


def mx_gemm(Aq, As, Bq, Bs):
    return G.tiled_matmul_hwlike(Aq, Bq, As, Bs, verbose=False)


def q_bf16(x):
    return G.q_bf16_rne(x)


def flash_causal_gqa(Q, K, V, softmax_scale, q_pos0, block_n, spec):
    n_q, Sq, d = Q.shape
    n_kv, Sk, _ = K.shape
    grp = n_q // n_kv
    nbu = FA.blocks_used(Sq, Sk, q_pos0, block_n)
    qpos = q_pos0 + torch.arange(Sq).unsqueeze(1)
    outs = []
    for h in range(n_q):
        kv = h // grp
        Qf, Qs = mx_quant_cols(Q[h], spec)
        O = torch.zeros(Sq, d, dtype=torch.float32)
        m_run = torch.full((Sq, 1), float("-inf"))
        l_run = torch.zeros(Sq, 1)
        for j in range(nbu):
            koff = j * block_n
            Kj = K[kv][koff:koff + block_n]
            Vj = V[kv][koff:koff + block_n]
            Kjf, Kjs = mx_quant_cols(Kj, spec)
            Sj_raw = mx_gemm(Qf, Qs, Kjf.t().contiguous(), Kjs.t().contiguous())
            Sj = q_bf16(Sj_raw * softmax_scale)
            kpos = koff + torch.arange(block_n).unsqueeze(0)
            Sj = Sj.masked_fill(kpos > qpos, float("-inf"))
            first = (j == 0)
            bmax = Sj.amax(dim=1, keepdim=True)
            m_new = bmax if first else torch.maximum(m_run, bmax)
            corr = torch.ones_like(bmax) if first else q_bf16(torch.exp(m_run - m_new))
            Pj = q_bf16(torch.exp(Sj - m_new))
            lsum = q_bf16(Pj.sum(dim=1, keepdim=True))
            l_run = q_bf16((torch.zeros_like(l_run) if first else l_run * corr) + lsum)
            Pjf, Pjs = mx_quant_cols(Pj, spec)
            Vjtf, Vjts = mx_quant_cols(Vj.t().contiguous(), spec)
            PVj = mx_gemm(Pjf, Pjs, Vjtf.t().contiguous(), Vjts.t().contiguous())
            O = q_bf16((torch.zeros_like(O) if first else O * corr) + PVj)
            m_run = m_new
        safe_l = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
        outs.append(q_bf16(O / safe_l))
    return torch.stack(outs)


def main():
    Sq, Sk, d = 64, 256, 64
    n_q, n_kv, q_pos0, Bk = 8, 1, 64, 64
    softmax_scale = 1.0 / (d ** 0.5)
    Q, K, V = FA.make_inputs_gqa(Sq, Sk, d, n_q, n_kv, seed=0)
    O_ref = FA.attention_reference_causal_gqa(Q, K, V, softmax_scale, q_pos0)
    print(f"GQA+causal flash attention  Sq={Sq} Sk={Sk} d={d} n_q={n_q} n_kv={n_kv} "
          f"grp={n_q//n_kv} q_pos0={q_pos0} Bk={Bk}")
    print(f"{'spec':<12}{'mantissa':<10}{'rel-err':<12}{'cosine':<12}{'vs fp32 ref'}")
    a_ref = O_ref.flatten().to(torch.float32)
    for spec, mant in [("fp8:e4m3", 3), ("fp6:e3m2", 2), ("fp4:e2m1", 1)]:
        O_mx = flash_causal_gqa(Q, K, V, softmax_scale, q_pos0, Bk, spec)
        rel = (O_mx - O_ref).norm().item() / O_ref.norm().item()
        b = O_mx.flatten().to(torch.float32)
        cos = (a_ref @ b / (a_ref.norm() * b.norm())).item()
        print(f"{spec:<12}{mant:<10}{100*rel:<12.3f}{cos:<12.5f}{'%rel'}")


if __name__ == "__main__":
    main()
