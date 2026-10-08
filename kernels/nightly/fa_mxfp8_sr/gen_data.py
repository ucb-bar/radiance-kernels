#!/usr/bin/env python3
"""MXFP8 flash-attention test data and golden, in the kernel's order of operations.

Per head h and 64-row query tile, over key blocks j of BK = 128 keys (d = 128):
  S_j   = Q K_j^T                      MX-FP8 mesh (mx_golden), bf16 out
  s_j   = bf16(S_j * scale)            bf16 multiply (Muon half = bf16)
  m_ref = rowmax(s_0)                  lazy softmax: the reference max is fixed after block 0;
                                       the accumulator's 8-bit exponent absorbs later P > 1
  P_j   = bf16(exp(bf16(s_j - m_ref))) Muon fexp.h
  l    += rowsum(P_j)                  fp32
  P_j  -> MX-FP8 (gemmini requantizer: scale 2^floor(log2 blockmax), e4m3 RNE)
  O     = sum_j P_j V_j                one MX GEMM over K = Sk (accumulator adds per 32-group)
                                       with the keys of each 32-key block in the kernel's order:
                                       even keys, then odd keys (KPERM, see the feeder)
  out   = bf16(O / l)

Device layout (see fa_data.h):
  region 1: Q fp8 [H][SQ][D]; QSC [H*SQ/64][D/32][64] (A scales of each q-tile);
            V fp8 [H][SK][D] (rows in KPERM order); VSC [H][NB][BK/32][D]
  region 2: KT fp8 [H][D][SK] (K transposed, the QK B operand); KSC [H][NB][D/32][BK]
  region 3: O bf16 [H][SQ][D] (poisoned 0xEEEE); G kernel-order golden; R fp32-reference (bf16)
  --spread: odd heads' V, VSC, KT, KSC in region 3 (FA_*1_ADDR, indexed by h / 2)
"""
import argparse, math, os, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parents[2] / "lib" / "golden"))
from nightly_data import Layout, bf16_bits, bf16_to_f32
import golden

ap = argparse.ArgumentParser()
ap.add_argument("--H", type=int, default=2)
ap.add_argument("--SQ", type=int, default=128)
ap.add_argument("--SK", type=int, default=1024)
ap.add_argument("--BK", type=int, default=128, help="key block (the kernel needs 128)")
ap.add_argument("--spread", action="store_true",
                help="odd heads' K, V (and scales) in region 3 (L2 slice 3): on 2 SMs each cluster streams from its own slices")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--novperm", action="store_true", help="keys in natural order (SPAD_REQUANT kernels: no Muon feeder, no even/odd key permutation)")
ap.add_argument("--exact", action="store_true", help="golden for FA_EXACT (run 3a): npu's online softmax in the kernel's order")
ap.add_argument("--npu", default="", help="npu-exploration attention header (e.g. attn_vpu_llama_2h.h): use its MX-FP8\n                codes/scales (Sq x Sk x d, GQA heads packed as rows -> H=1) and its O_REF_F_F32 as the reference")
ap.add_argument("--out", default="data")
ap.add_argument("--header", default="fa_data.h")
ap.add_argument("--dump", default="", help="also save kernel-order intermediates (npz) for debugging")
a = ap.parse_args()
import re
def npu_arr(name, dt):
    m = re.search(r'\b' + name + r'\s*\[[^=]*=\s*\{(.*?)\};', NPU, re.S)
    return np.array([int(x, 0) for x in re.findall(r'0x[0-9a-fA-F]+|\d+', m.group(1))], dt)
NPU = open(a.npu).read() if a.npu else ""
if a.npu:   # npu-exploration's problem: its own GQA packing (one K/V, Sq rows), so H = 1
    a.H = 1
    a.SQ, a.SK = (int(re.search(r'#define ATTN_' + k + r'\s+(\d+)', NPU).group(1)) for k in ("SQ", "SK"))
    D_NPU = int(re.search(r'#define ATTN_D\s+(\d+)', NPU).group(1))
H, SQ, SK, D, BK, QT = a.H, a.SQ, a.SK, (D_NPU if a.npu else 128), a.BK, 64
NB = SK // BK
assert SQ % QT == 0 and SK % BK == 0
scale = np.float32(1.0 / math.sqrt(D))
scale_bf = bf16_to_f32(bf16_bits(np.array([scale], dtype=np.float32)))[0]   # the kernel's bf16 scale
rng = np.random.default_rng(a.seed)


def e4m3_decode(c):
    c = c.astype(np.int32)
    s = np.where(c & 0x80, -1.0, 1.0)
    e = (c >> 3) & 0xF
    m = c & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return s * v


def e4m3_rne(x):
    """float -> e4m3 code, round to nearest even, saturate to 448."""
    x = np.asarray(x, dtype=np.float64)
    sign = (np.signbit(x)).astype(np.int32) << 7
    ax = np.minimum(np.abs(x), 448.0)
    e = np.floor(np.log2(np.maximum(ax, 2.0 ** -9)))
    e = np.maximum(e, -6.0)                        # subnormal range shares exponent -6
    q = ax / 2.0 ** e * 8.0                        # mantissa units (8 = 1.0)
    qr = np.round(q)                               # numpy round = half to even
    carry = qr >= 16
    e = np.where(carry, e + 1, e)
    qr = np.where(carry, qr / 2, qr)
    normal = qr >= 8
    code = np.where(normal, ((e + 7).astype(np.int32) << 3) | (qr.astype(np.int32) - 8),
                    qr.astype(np.int32))          # subnormal: e field 0, mantissa = qr
    code = np.where(ax == 0, 0, code)
    return (code | sign).astype(np.uint8)


def mx_quant_rows(X):
    """MX-quantize along the last axis in 32-blocks: scale 2^floor(log2 max|x|) (max -> [1,2)),
    elements e4m3 RNE.  Returns (codes uint8 same shape, scale codes uint8 [..., n/32])."""
    shp = X.shape
    B = X.reshape(-1, shp[-1] // 32, 32).astype(np.float64)
    amax = np.abs(B).max(axis=-1)
    e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-38))), 0.0)
    codes = e4m3_rne(B / (2.0 ** e)[..., None])
    return codes.reshape(shp), (e + 127).astype(np.uint8).reshape(shp[:-1] + (shp[-1] // 32,))


def mx_dequant_rows(codes, sc):
    v = e4m3_decode(codes).reshape(codes.shape[:-1] + (codes.shape[-1] // 32, 32))
    return (v * (2.0 ** (sc.astype(np.float64) - 127))[..., None]).reshape(codes.shape)


# ---- inputs: Q, K, V ~ N(0,1) (bf16-representable), quantized to MX-FP8 along d / keys ----
Qf = bf16_to_f32(bf16_bits(rng.standard_normal((H, SQ, D)).astype(np.float32)))
Kf = bf16_to_f32(bf16_bits(rng.standard_normal((H, SK, D)).astype(np.float32)))
Vf = bf16_to_f32(bf16_bits(rng.standard_normal((H, SK, D)).astype(np.float32)))
Qc, Qs = mx_quant_rows(Qf)                                   # blocks along d
Kc, Ks = mx_quant_rows(Kf)                                   # blocks along d
Vtc, Vts = mx_quant_rows(np.ascontiguousarray(Vf.transpose(0, 2, 1)))   # blocks along keys
Vc = np.ascontiguousarray(Vtc.transpose(0, 2, 1))            # [H][SK][D]
if a.npu:   # replace with npu-exploration's exact MX-FP8 inputs (TinyLlama); Qf/Kf/Vf are then unused
    Qc = npu_arr("Q_IN", np.uint8).reshape(1, SQ, D)
    Qs = npu_arr("Q_SCALES", np.uint8).reshape(D // 32, SQ).T.reshape(1, SQ, D // 32)
    Kc = np.ascontiguousarray(npu_arr("KT_IN", np.uint8).reshape(D, SK).T).reshape(1, SK, D)
    Ks = np.ascontiguousarray(npu_arr("KT_SCALES", np.uint8).reshape(D // 32, SK).T).reshape(1, SK, D // 32)
    Vc = npu_arr("V_IN", np.uint8).reshape(1, SK, D)
    Vts = np.ascontiguousarray(npu_arr("V_SCALES", np.uint8).reshape(SK // 32, D).T).reshape(1, D, SK // 32)
# The feeder puts element 2l of a 32-element block into beat 0 and 2l+1 into beat 1, so the
# requantized P (and PV's A operand) has the block's even keys first, then its odd keys.  V's rows and
# the golden's P columns get the same order (block scales do not depend on the order).
blk = np.concatenate([np.arange(0, 32, 2), np.arange(1, 32, 2)])
KPERM = (np.arange(0, SK, 32)[:, None] + blk[None, :]).reshape(-1)
if a.novperm:   # SPAD_REQUANT reads P in natural key order: no permutation of V or of the golden's P columns
    KPERM = np.arange(SK)
Vs = np.ascontiguousarray(Vts.transpose(0, 2, 1))            # [H][SK/32][D]

O_g = np.zeros((H, SQ, D), dtype=np.uint16)
inter = {}
O_r = np.zeros((H, SQ, D), dtype=np.uint16)
tmp = HERE / "_gen"
for h in range(H):
    # QK^T for the whole head at once: mesh golden, A = Q [SQ][D], B = K^T [D][SK]
    S = golden.mx_matmul(Qc[h], np.ascontiguousarray(Kc[h].T), np.ascontiguousarray(Qs[h].T),
                         np.ascontiguousarray(Ks[h].T), SQ, SK, D, tmpdir=str(tmp / "qk"))
    s = bf16_to_f32(bf16_bits(bf16_to_f32(S) * scale_bf))     # bf16(S * bf16(scale)), fmul.h
    if a.exact:   # FA_EXACT: per key block j, m_j = max(m_{j-1}, rowmax s_j), a_j = exp(m_{j-1} - m_j),
        # l = l * a_j + sum P_j (f32), O_j = MX(P_j) V_j (mesh, bf16), O = bf16(bf16(O * a_j) + O_j); out = bf16(O / l)
        bfr = lambda x: bf16_to_f32(bf16_bits(np.asarray(x, dtype=np.float32)))
        m = l = O = None
        Pcs, Pss = [], []
        for j in range(NB):
            sj = s[:, j * BK:(j + 1) * BK]
            mx_j = sj.max(axis=1, keepdims=True)
            if j == 0:
                m_new, aj = mx_j, np.zeros((SQ, 1), np.float32)
            else:
                m_new = np.maximum(m, mx_j)
                aj = bfr(np.exp(bfr(m - m_new).astype(np.float64)))
            m = m_new
            Pj = bfr(np.exp(bfr(sj - m).astype(np.float64)))
            lb = Pj.astype(np.float32).sum(axis=1, dtype=np.float32)
            l = lb if j == 0 else (l * aj[:, 0]).astype(np.float32) + lb
            Pcj, Psj = mx_quant_rows(Pj)                      # [SQ][BK], [SQ][BK/32]
            kp = KPERM[j * BK:(j + 1) * BK]
            Oj = bf16_to_f32(golden.mx_matmul(np.ascontiguousarray(Pcj[:, kp - j * BK]), np.ascontiguousarray(Vc[h][kp]),
                                              np.ascontiguousarray(Psj.T), np.ascontiguousarray(Vs[h][j * BK // 32:(j + 1) * BK // 32]),
                                              SQ, D, BK, tmpdir=str(tmp / "pv")))
            O = Oj if j == 0 else bfr(bfr(O * aj) + Oj)
            Pcs.append(Pcj); Pss.append(Psj)
        Pc, Ps, P = np.concatenate(Pcs, 1), np.concatenate(Pss, 1), None
        Ou = bf16_bits(O)
        O_g[h] = bf16_bits(O * (np.float32(1.0) / l)[:, None])
    else:
        m_ref = s[:, :BK].max(axis=1, keepdims=True)
        t = bf16_to_f32(bf16_bits(s - m_ref))
        P = bf16_to_f32(bf16_bits(np.exp(t.astype(np.float64)).astype(np.float32)))
        l = P.astype(np.float32).sum(axis=1, dtype=np.float32)
        Pc, Ps = mx_quant_rows(P)                                 # [SQ][SK], [SQ][SK/32]
        Ou = golden.mx_matmul(np.ascontiguousarray(Pc[:, KPERM]), np.ascontiguousarray(Vc[h][KPERM]),
                              np.ascontiguousarray(Ps.T), Vs[h], SQ, D, SK,
                              tmpdir=str(tmp / "pv"))
        O_g[h] = bf16_bits(bf16_to_f32(Ou) / l[:, None])
    inter.update({f"S{h}": S, f"P{h}": bf16_bits(P) if P is not None else 0, f"Pc{h}": Pc, f"Ps{h}": Ps, f"Ou{h}": Ou,
                  f"l{h}": l, f"mref{h}": (m if a.exact else m_ref)[:, 0]})
    # fp32 reference on the unquantized inputs
    Sr = (Qf[h].astype(np.float64) @ Kf[h].astype(np.float64).T) * float(scale)
    Pr = np.exp(Sr - Sr.max(axis=1, keepdims=True))
    O_r[h] = bf16_bits((Pr @ Vf[h].astype(np.float64)) / Pr.sum(axis=1, keepdims=True))
    if a.npu:   # npu's fp64 attention reference (O_REF_F_F32), the one its accuracy line uses
        O_r[h] = bf16_bits(npu_arr("O_REF_F_F32", np.uint32).view(np.float32).reshape(SQ, D))


def frob(x, ref):
    x, ref = bf16_to_f32(x).astype(np.float64), bf16_to_f32(ref).astype(np.float64)
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


if a.dump:
    np.savez(a.dump, Qc=Qc, Qs=Qs, Kc=Kc, Ks=Ks, Vc=Vc, Vs=Vs, O_g=O_g, **inter)

L = Layout("fa")
for k, v in dict(FA_H=H, FA_SQ=SQ, FA_SK=SK, FA_D=D, FA_BK=BK).items():
    L.const(k, f"{v}u")
L.const("FA_SCALE_BITS", f"0x{np.float32(scale).view(np.uint32):08x}u", "1/sqrt(D) as f32 bits")
L.add("FA_Q", 1, Qc, "Q fp8 [H][SQ][D]")
QSC = Qs.reshape(H, SQ // QT, QT, D // 32).transpose(0, 1, 3, 2)   # [H][qt][D/32][64]
L.add("FA_QSC", 1, np.ascontiguousarray(QSC), "Q scales [H*SQ/64][D/32][64]")
VD = np.ascontiguousarray(Vc[:, KPERM])                       # [H][SK][D], keys in KPERM order
VSC = Vs.reshape(H, NB, BK // 32, D)                          # [H][NB][BK/32][D]
KTD = np.ascontiguousarray(Kc.transpose(0, 2, 1))             # [H][D][SK]
KSC = Ks.reshape(H, NB, BK, D // 32).transpose(0, 1, 3, 2)   # [H][NB][D/32][BK]
L.const("FA_SPREAD", 1 if a.spread else 0)
L.const("FA_VPERM", 0 if a.novperm else 1)   # V rows (and the golden's P columns) in KPERM order
if a.spread:   # even heads: V region 1, K region 2; odd heads: both in region 3 (head index h / 2)
    L.add("FA_V", 1, np.ascontiguousarray(VD[0::2]), "V fp8, even heads [H/2][SK][D]")
    L.add("FA_VSC", 1, np.ascontiguousarray(VSC[0::2]), "V scales, even heads")
    L.add("FA_KT", 2, np.ascontiguousarray(KTD[0::2]), "K^T fp8, even heads [H/2][D][SK]")
    L.add("FA_KSC", 2, np.ascontiguousarray(KSC[0::2]), "K scales, even heads")
    L.add("FA_V1", 3, np.ascontiguousarray(VD[1::2]), "V fp8, odd heads")
    L.add("FA_VSC1", 3, np.ascontiguousarray(VSC[1::2]), "V scales, odd heads")
    L.add("FA_KT1", 3, np.ascontiguousarray(KTD[1::2]), "K^T fp8, odd heads")
    L.add("FA_KSC1", 3, np.ascontiguousarray(KSC[1::2]), "K scales, odd heads")
else:
    L.add("FA_V", 1, VD, "V fp8 [H][SK][D] (keys in KPERM order)")
    L.add("FA_VSC", 1, np.ascontiguousarray(VSC), "V scales [H][NB][BK/32][D]")
    L.add("FA_KT", 2, KTD, "K^T fp8 [H][D][SK]")
    L.add("FA_KSC", 2, np.ascontiguousarray(KSC), "K scales [H][NB][D/32][BK]")
L.reserve("FA_O", 3, 2 * H * SQ * D, fill=0xEE, comment="O bf16 [H][SQ][D], poisoned")
L.add("FA_G", 3, O_g, "kernel-order golden O bf16")
L.add("FA_R", 3, O_r, "fp32-reference O (bf16)")
L.write(a.out, a.header)
print(f"H={H} SQ={SQ} SK={SK} D={D}: golden vs fp32 reference Frobenius {100 * frob(O_g, O_r):.3f}%")
