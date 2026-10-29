"""llama_e2e --exact-emu: the prefill emulation in the device's own arithmetic, for bit-exact goldens.

gen_data.emulate() approximates the device (float GEMMs on dequantized weights, numpy reductions, an ideal exp, softmax
normalized before PV), so device and emulation drift 5 - 20 % apart over a model and the checks can only gate gross
failures.  emulate_exact() follows the kernels operation by operation:
  GEMM          mx_golden (the MX Gemmini in hardware semantics, bit-exact with spike), split over output columns into
                parallel processes (every output element is computed independently)
  PRENORM       kernel.cpp prenorm_rows at 2 / 4 clusters: per lane (16 lanes, word w to lane w % 16) a running fp32 sum of
                lo^2 + hi^2 over its words, then warp_sum (lanes 0..15 in order); inv = 1 / sqrtf(ss / D + eps); codes
                of (h inv) g (quant_group = mx_quant).  At 1 cluster the norm after oproj / down is fused into the GEMM
                (norm_fused) and sums in another order: these goldens target 2 / 4 clusters
  ROPE / bias   as emulate() (rope_row's fp32 operations), quant_group = mx_quant
  ATTN          fa_sa.h: S from the QK GEMM (bf16); m = bf16(block-0 max of S * sc) (exact-max models: of the masked
                row max); per element bf16 a = S sc, x = a - m (clamped at -64 with exact max), P = fexp.h(x) (muon_fp's
                fpEX model), masked keys 0; l in fp32 in the softmax lanes' order (lane = query % 16 visits word
                (k + lane) % 64, k = 0..63, adding P[2i] + P[2i+1]); P requantized by the Gemmini SPAD_REQUANT in
                mx_quant's convention (E8M0 floor(log2 max): mx_golden's out_fmt scaling, max - 8, puts the codes near
                448 and overflows its accumulator); PV by mx_golden;
                O = bf16(bf16(PV) * (1 / l)), quant_group
  SILU          silu_mul in fp32 (exp2_poly), quant_group
Not covered (asserted): Gemma-2 (fa_sa256.h, soft-capping, post-norms).  The residual adds are bf16(x + y) as the device.
"""
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from muon_fp import fexp_bf16

F = np.float32
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "lib", "golden"))
import golden  # noqa: E402

JOBS = max(1, min(24, (os.cpu_count() or 8) // 2))
_TMP = tempfile.mkdtemp(prefix="emu_exact_")


def gemm(ac, asc, wq, cols=None):
    """A codes [M][K] + scales [M][K/32] @ weight (codes [N][K], scales [N][K/32]) -> bf16 bits [M][N], mx_golden"""
    codes, sc = wq
    if cols is not None:
        codes, sc = codes[cols[0]:cols[1]], sc[cols[0]:cols[1]]
    M, K = ac.shape
    N = codes.shape[0]
    step = max(64, -(-N // (JOBS * 2)) // 64 * 64)
    asc_t = np.ascontiguousarray(asc.T)

    def run(i):
        c0, c1 = i, min(i + step, N)
        return golden.mx_matmul(ac, np.ascontiguousarray(codes[c0:c1].T), asc_t, np.ascontiguousarray(sc[c0:c1].T),
                                M, c1 - c0, K, tmpdir=os.path.join(_TMP, f"g{os.getpid()}_{i}"))
    with ThreadPoolExecutor(JOBS) as ex:
        parts = list(ex.map(run, range(0, N, step)))
    return np.concatenate(parts, axis=1)


def norm_exact(g, h, gamma, eps):
    """prenorm_rows (2 / 4 clusters): h bf16 values [M][D] fp32, gamma bf16 values -> fp32 (h inv) g"""
    M, D = h.shape
    t = (h[:, 0::2] * h[:, 0::2]).astype(F) + (h[:, 1::2] * h[:, 1::2]).astype(F)   # lo^2 + hi^2 per word
    W = D // 2
    lanes = t.reshape(M, W // 16, 16)
    s = np.zeros((M, 16), F)
    for i in range(W // 16):                              # each lane's words in ascending order
        s = (s + lanes[:, i, :]).astype(F)
    ss = np.zeros(M, F)
    for l in range(16):                                   # warp_sum: red[0] + red[1] + ... from 0
        ss = (ss + s[:, l]).astype(F)
    inv = (F(1) / np.sqrt(((ss / F(D)).astype(F) + F(eps)).astype(F))).astype(F)
    return ((h * inv[:, None]).astype(F) * g.bf(gamma)[None, :]).astype(F)


def fa_sa_head(g, S_bits, qpos, P0, scale, exact, BK=128):
    """fa_sa.h softmax of one head: S_bits [R][SK] bf16 (rows = queries at positions qpos), -> (P fp32 [R][SK], inv [R])"""
    R, SK = S_bits.shape
    NB = SK // BK
    sc = g.bf(F(scale))                                   # bf16 of LL_FA_SCALE_BITS, RNE
    S = g.bf16_to_f32(S_bits)
    keys = np.arange(SK)[None, :]
    allowed = keys <= qpos[:, None]
    if exact:   # pre-pass: bf16 max over the allowed keys, m = bf16(max sc)
        m = g.bf(np.where(allowed, S, -np.inf).max(axis=1) * sc)
    else:       # key block 0's max (unmasked: P0 >= 128), m = bf16(max sc)
        m = g.bf(S[:, :BK].max(axis=1) * sc)
    a = g.bf(S * sc)
    x = g.bf(a - m[:, None])
    if exact:
        x = np.maximum(x, F(-64.0))
    P = g.bf16_to_f32(fexp_bf16(g.bf16_bits(x)))
    P = np.where(allowed, P, F(0)).astype(F)
    lane = (np.arange(R) % 64) % 16                      # tile row (QT = 64) % 16
    lt = np.zeros(R, F)
    for j in range(NB):
        Pj = P[:, j * BK:(j + 1) * BK]
        pair = (Pj[:, 0::2] + Pj[:, 1::2]).astype(F)     # f32(ea) + f32(eb), word i = keys 2i, 2i + 1
        l = np.zeros(R, F)
        for k in range(64):
            l = (l + pair[np.arange(R), (k + lane) & 63]).astype(F)
        lt = l if j == 0 else (lt + l).astype(F)
    return P, (F(1) / lt).astype(F)


def emulate_exact(g, cfg, M, P0, mdl, qw, cos, sin, ilv=False):
    """gen_data.emulate()'s outputs (h, layer0, layer0_raw, logits) in the device's arithmetic (see the module doc)"""
    assert not cfg.get("gemma") and not cfg.get("attn_cap") and not cfg.get("final_cap"), "--exact-emu: no Gemma-2"
    D, Fd, Hq, Hkv, hd, eps = cfg["D"], cfg["F"], cfg["Hq"], cfg["Hkv"], cfg["hd"], cfg["eps"]
    Dq, Dkv, Gq, SK = Hq * hd, Hkv * hd, Hq // Hkv, P0 + M
    exact = bool(cfg.get("fa_exact"))
    scale = F(1.0 / np.sqrt(cfg.get("qpre", hd)))
    qpos = P0 + np.arange(M)
    out = dict(h=[], layer0={})
    h = g.bf(mdl["x0"])
    hmid = ymlp = None
    for l, w in enumerate(mdl["layers"]):
        q = qw[l]
        if l > 0:
            h = g.bf(hmid + ymlp)
            out["h"].append(h)
        xc, xs = g.mx_quant(norm_exact(g, h, w["g_in"], eps))
        qkv_raw = g.bf16_to_f32(gemm(xc, xs, q["wqkv"]))
        qkv = g.bf(qkv_raw + g.bf(w["bqkv"])[None, :]) if "bqkv" in w else qkv_raw
        Q = g.rope(qkv[:, :Dq].reshape(M, Hq, hd), cos, sin)
        Kn = g.rope(qkv[:, Dq:Dq + Dkv].reshape(M, Hkv, hd), cos, sin)
        Vn = qkv[:, Dq + Dkv:].reshape(M, Hkv, hd)
        qc, qs = g.mx_quant(Q)                                            # [M][Hq][hd]
        kcen = w.get("kcen", np.zeros((Hkv, hd), F))[:, None, :]
        Kc = np.concatenate([np.asarray(w["kc"], F) - kcen, Kn.transpose(1, 0, 2) - kcen], axis=1)
        kc, ks = g.mx_quant(Kc)                                           # [Hkv][SK][hd]
        Vall = np.concatenate([np.asarray(w["vc"], F), Vn.transpose(1, 0, 2)], axis=1)
        vc, vs = g.mx_quant(Vall.transpose(0, 2, 1))                      # [Hkv][hd][SK]: groups along keys
        O = np.zeros((M, Hq, hd), F)
        S0 = None
        for h_ in range(Hq):
            gi = h_ // Gq
            S = gemm(qc[:, h_], qs[:, h_], (kc[gi], ks[gi]))              # [M][SK] bf16 bits
            if h_ == 0:
                S0 = g.bf16_to_f32(S)
            P, inv = fa_sa_head(g, S, qpos, P0, scale, exact)
            pc, ps = g.mx_quant(P)   # SPAD_REQUANT: the device's P codes match mx_quant (decode o: 99 % bf16-exact)
            PV = g.bf16_to_f32(gemm(pc, ps, (vc[gi], vs[gi])))           # [M][hd]
            O[:, h_] = g.bf(PV * inv[:, None])
        O = O.reshape(M, Dq)
        oc, os_ = g.mx_quant(O)
        yattn = g.bf16_to_f32(gemm(oc, os_, q["wo"]))
        hmid = g.bf(h + yattn)
        x2c, x2s = g.mx_quant(norm_exact(g, hmid, w["g_post"], eps))
        gu = g.bf16_to_f32(gemm(x2c, x2s, q["wgu"]))
        guo = gu[:, np.argsort(g.gu_interleave(Fd))] if ilv else gu
        Hh = (g.silu_dev(guo[:, :Fd]) * guo[:, Fd:]).astype(F)
        hc, hs = g.mx_quant(Hh)
        ymlp = g.bf16_to_f32(gemm(hc, hs, q["wd"]))
        if l == 0:
            out["layer0"] = dict(qkv=qkv, s0=S0, o=O, yattn=yattn, hmid=hmid, gu=gu, ymlp=ymlp)
            out["layer0_raw"] = dict(qkv_raw=qkv_raw, Q=Q, Kc=Kc, Vall=Vall, fa0=None)
        print(f"  exact emulation: layer {l} done", flush=True)
    h = g.bf(hmid + ymlp)
    out["h"].append(h)
    fc, fs = g.mx_quant(norm_exact(g, h, mdl["g_final"], eps))
    out["logits"] = g.bf16_to_f32(gemm(fc, fs, qw["wlm"]))
    return out
