#!/usr/bin/env python3
"""Reconstructed MX-FP8 flash-attention reference model.

This standalone rebuild provides the exact function surface the kernel data/golden
generators (fa_gen_data.py) import:

  make_inputs, mx_quantize_cols, mx_gemm,
  mx_attention_dense (with trace), attention_reference, mx_attention_flash

It layers on the surviving low-level MX matmul model (fp8_matmul_model.py), reusing the
identical hardware-matching mesh matmul + MX column-group quantizer + bf16 rounding, so
the numerics track the Gemmini MX path the kernel drives.
"""
import sys, os
import torch

# fp8_matmul_model.py lives in the mxgemmini lib (the surviving low-level MX model).
_LIB = os.environ.get("MXGEMMINI_LIB",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib", "mxgemmini"))
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
import fp8_matmul_model as G

GROUP = G.GROUP  # 32
INPUT_SPEC = G.INPUT_SPEC  # fp8:e4m3

# Hardware mesh precision schedule (matches fp8_matmul_model __main__ / the Gemmini config).
_PROD = [(4, 3)] * G.TILE
_ACC = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1


def q_bf16(x):
    return G.q_bf16_rne(x)


def _fp8_round(x):
    """Round scaled operand values to e4m3 exactly as the emitted header codes do
    (tensor_to_custom_fp_codes uses round-to-nearest), so the golden multiplies the
    same numbers the kernel reads back from the fp8 header arrays."""
    codes, _ = G.tensor_to_custom_fp_codes(x, INPUT_SPEC)
    # decode e4m3 codes back to float
    e_bits, m_bits = 4, 3
    bias = (1 << (e_bits - 1)) - 1
    arr = torch.zeros(len(codes), len(codes[0]), dtype=torch.float32)
    for r, row in enumerate(codes):
        for c, code in enumerate(row):
            s = (code >> (e_bits + m_bits)) & 1
            ef = (code >> m_bits) & ((1 << e_bits) - 1)
            mant = code & ((1 << m_bits) - 1)
            if ef == 0:
                val = mant / (1 << m_bits) * (2.0 ** (1 - bias))
            else:
                val = (1.0 + mant / (1 << m_bits)) * (2.0 ** (ef - bias))
            arr[r, c] = -val if s else val
    return arr


def make_inputs(Sq, Sk, d, seed=0, peaked=False):
    torch.manual_seed(seed)
    Q = torch.randn(Sq, d, dtype=torch.float32)
    K = torch.randn(Sk, d, dtype=torch.float32)
    V = torch.randn(Sk, d, dtype=torch.float32)
    if peaked:
        Q = Q * 4.0
    return Q, K, V


def mx_quantize_cols(M):
    """Per-row MX quantize over 32-col groups. Returns (scaled_values, scales).
    scaled_values are the operand values fed to the mesh (fp8 domain); scales are the
    per-(row,32col-group) E8M0 power-of-two factors. Mirrors matrix_mx_requantize."""
    q, s = G.matrix_mx_requantize(M, INPUT_SPEC)
    return q, s


def mx_gemm(A_fp8, A_scales, B_fp8, B_scales):
    """MX mesh matmul. A_fp8 [M,K] with A_scales [M,K/32]; B_fp8 [K,N] with B_scales [K/32,N].
    Returns C [M,N] in bf16 (as the mesh accumulates + rescales per tile)."""
    # NOTE: the (4,x)-exponent accumulator schedule (_ACC) was tuned for O(1) demo
    # operands; MX column-group quantization scales real operands up into the fp8 range
    # (|x| up to 448), so their products overflow an exp=4 accumulator -> inf/NaN. The
    # mesh here is modeled with the bf16 product + bf16-accumulate path (the default),
    # which is numerically safe and matches the Gemmini bf16 accumulate.
    A = _fp8_round(A_fp8)
    B = _fp8_round(B_fp8)
    return G.tiled_matmul_hwlike(A, B, A_scales, B_scales, verbose=False)


def attention_reference(Q, K, V, softmax_scale):
    """Pure fp32 softmax attention reference."""
    S = softmax_scale * (Q @ K.t())
    P = torch.softmax(S, dim=1)
    return P @ V


def mx_attention_dense(Q, K, V, softmax_scale, trace=None):
    """Dense (non-streaming) MX attention. Records intermediates in `trace`."""
    if trace is None:
        trace = {}
    Qf8, Qs = mx_quantize_cols(Q)
    Kf8, Ks = mx_quantize_cols(K)
    S_raw = mx_gemm(Qf8, Qs, Kf8.t().contiguous(), Ks.t().contiguous())  # [Sq,Sk] bf16
    trace["S_raw"] = S_raw
    S = q_bf16(S_raw * softmax_scale)
    trace["S"] = S
    m = S.amax(dim=1, keepdim=True)
    trace["m"] = m
    P = q_bf16(torch.exp(S - m))
    trace["P"] = P
    l = q_bf16(P.sum(dim=1, keepdim=True))
    trace["l"] = l
    Pn = q_bf16(P / l)
    Pnf8, Pns = mx_quantize_cols(Pn)
    Vt_f8, Vt_s = mx_quantize_cols(V.t().contiguous())  # group V along Sk
    O = mx_gemm(Pnf8, Pns, Vt_f8.t().contiguous(), Vt_s.t().contiguous())
    return O


def mx_attention_flash(Q, K, V, softmax_scale, block_m=None, block_n=None):
    """Streaming online-softmax MX attention. Mirrors the kernel:
    per key block j, requant the UNNORMALIZED P_j to MX-fp8, rescale the O accumulator by
    corr = exp(m_old - m_new), and defer the 1/l normalization to the finalize step."""
    Sq, d = Q.shape
    Sk = K.shape[0]
    if block_m is None:
        block_m = Sq
    if block_n is None:
        block_n = Sk
    Nblk = Sk // block_n
    Qf8, Qs = mx_quantize_cols(Q)

    O = torch.zeros(Sq, d, dtype=torch.float32)
    m_run = torch.full((Sq, 1), float("-inf"))
    l_run = torch.zeros(Sq, 1)
    for j in range(Nblk):
        Kj = K[j * block_n:(j + 1) * block_n]           # [Bk, d]
        Vj = V[j * block_n:(j + 1) * block_n]           # [Bk, d]
        Kjf8, Kjs = mx_quantize_cols(Kj)
        Sj_raw = mx_gemm(Qf8, Qs, Kjf8.t().contiguous(), Kjs.t().contiguous())  # [Sq,Bk]
        Sj = q_bf16(Sj_raw * softmax_scale)
        bmax = Sj.amax(dim=1, keepdim=True)
        first = (j == 0)
        m_new = bmax if first else torch.maximum(m_run, bmax)
        corr = torch.exp((m_run - m_new)) if not first else torch.ones_like(bmax)
        corr = q_bf16(corr)
        Pj = q_bf16(torch.exp(Sj - m_new))              # unnormalized probs
        lsum = q_bf16(Pj.sum(dim=1, keepdim=True))
        l_run = q_bf16((l_run * corr if not first else torch.zeros_like(l_run)) + lsum)
        # per-block MX requant of the unnormalized probs
        Pjf8, Pjs = mx_quantize_cols(Pj)
        Vjt_f8, Vjt_s = mx_quantize_cols(Vj.t().contiguous())  # group along Bk
        PVj = mx_gemm(Pjf8, Pjs, Vjt_f8.t().contiguous(), Vjt_s.t().contiguous())  # [Sq,d]
        O = q_bf16((O * corr if not first else torch.zeros_like(O)) + PVj)
        m_run = m_new
    O = q_bf16(O / l_run)
    return O


# ===================== GQA + causal-mask extension (TinyLlama prefill) =====================
# GQA: n_q query heads share n_kv KV heads in contiguous groups of grp = n_q // n_kv, i.e.
# query head h reads KV head h // grp. Causal mask: query row i has GLOBAL position
# q_pos0 + i; key col c of block j has GLOBAL position j*block_n + c; the entry is masked
# (set to -inf, prob 0) whenever key_pos > query_pos. Blocks whose minimum key position is
# already above every query position are skipped wholesale ("halve work").

def make_inputs_gqa(Sq, Sk, d, n_q, n_kv, seed=0):
    """Per-head inputs: Q[n_q,Sq,d], K[n_kv,Sk,d], V[n_kv,Sk,d]."""
    torch.manual_seed(seed)
    Q = torch.randn(n_q, Sq, d, dtype=torch.float32)
    K = torch.randn(n_kv, Sk, d, dtype=torch.float32)
    V = torch.randn(n_kv, Sk, d, dtype=torch.float32)
    return Q, K, V


def causal_bool_mask(Sq, Sk, q_pos0):
    """True where the entry must be masked (key_pos > query_pos)."""
    qpos = q_pos0 + torch.arange(Sq).unsqueeze(1)      # [Sq,1]
    kpos = torch.arange(Sk).unsqueeze(0)               # [1,Sk]
    return kpos > qpos


def blocks_used(Sq, Sk, q_pos0, block_n):
    """Number of leading key blocks that are NOT fully above the diagonal."""
    max_qpos = q_pos0 + Sq - 1
    Nblk = Sk // block_n
    return min(Nblk, max_qpos // block_n + 1)


def attention_reference_causal_gqa(Q, K, V, softmax_scale, q_pos0):
    """Pure fp32 causal + GQA softmax attention reference. Returns O [n_q,Sq,d]."""
    n_q, Sq, d = Q.shape
    n_kv, Sk, _ = K.shape
    grp = n_q // n_kv
    mask = causal_bool_mask(Sq, Sk, q_pos0)
    outs = []
    for h in range(n_q):
        kv = h // grp
        S = softmax_scale * (Q[h] @ K[kv].t())         # [Sq,Sk]
        S = S.masked_fill(mask, float("-inf"))
        P = torch.softmax(S, dim=1)
        outs.append(P @ V[kv])
    return torch.stack(outs)


def mx_attention_flash_causal_gqa(Q, K, V, softmax_scale, q_pos0, block_n):
    """Streaming online-softmax MX attention with GQA head sharing + causal masking.
    Mirrors the kernel EXACTLY: block skip above the diagonal, per-element causal mask
    before the block max/exp, per-block MX requant of the unnormalized probs, corr rescale
    of the O accumulator, deferred 1/l. Returns O [n_q,Sq,d] (fp32, bf16-rounded)."""
    n_q, Sq, d = Q.shape
    n_kv, Sk, _ = K.shape
    grp = n_q // n_kv
    nbu = blocks_used(Sq, Sk, q_pos0, block_n)
    qpos = q_pos0 + torch.arange(Sq).unsqueeze(1)      # [Sq,1]
    outs = []
    for h in range(n_q):
        kv = h // grp
        Qf8, Qs = mx_quantize_cols(Q[h])
        O = torch.zeros(Sq, d, dtype=torch.float32)
        m_run = torch.full((Sq, 1), float("-inf"))
        l_run = torch.zeros(Sq, 1)
        for j in range(nbu):
            koff = j * block_n
            Kj = K[kv][koff:koff + block_n]
            Vj = V[kv][koff:koff + block_n]
            Kjf8, Kjs = mx_quantize_cols(Kj)
            Sj_raw = mx_gemm(Qf8, Qs, Kjf8.t().contiguous(), Kjs.t().contiguous())  # [Sq,Bk]
            Sj = q_bf16(Sj_raw * softmax_scale)
            # causal mask within this block (partial blocks only; full blocks -> no-op)
            kpos = koff + torch.arange(block_n).unsqueeze(0)     # [1,Bk]
            Sj = Sj.masked_fill(kpos > qpos, float("-inf"))
            first = (j == 0)
            bmax = Sj.amax(dim=1, keepdim=True)
            m_new = bmax if first else torch.maximum(m_run, bmax)
            corr = torch.ones_like(bmax) if first else q_bf16(torch.exp(m_run - m_new))
            Pj = q_bf16(torch.exp(Sj - m_new))          # masked entries -> exp(-inf) = 0
            lsum = q_bf16(Pj.sum(dim=1, keepdim=True))
            l_run = q_bf16((torch.zeros_like(l_run) if first else l_run * corr) + lsum)
            Pjf8, Pjs = mx_quantize_cols(Pj)
            Vjt_f8, Vjt_s = mx_quantize_cols(Vj.t().contiguous())
            PVj = mx_gemm(Pjf8, Pjs, Vjt_f8.t().contiguous(), Vjt_s.t().contiguous())
            O = q_bf16((torch.zeros_like(O) if first else O * corr) + PVj)
            m_run = m_new
        # rows with no visible key (fully masked, l==0) -> leave at 0 (avoid nan)
        safe_l = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
        O = q_bf16(O / safe_l)
        outs.append(O)
    return torch.stack(outs)
