#!/usr/bin/env python3
"""llama_e2e: data and phase program for a whole Llama-family model on Radiance (Muon + MxGemmini).

One prefill chunk of M tokens at positions P0 .. P0+M-1 attends over a KV cache of P0 earlier tokens
(SK = P0 + M keys, causal inside the chunk).  Every decoder layer, the final norm and the LM head run on
the device as a list of phases (kernel.cpp interprets it):

  norm1   PRENORM  h_l = h_mid(l-1) + y_mlp(l-1) (bf16);  xn1 = MX(rmsnorm(h_l) * g_in)     (layer 0: h_0 = x0)
  qkv     GEMM     [Q | K | V] = xn1 @ Wqkv
  rope    ROPE     Q, K = rope(Q), rope(K) -> MX;  K -> K^T cache, V -> V cache (keys P0..P0+M-1)
  qk      GEMM     per KV head (batch): S = Q_g @ K^T, the Gq query heads of the group stacked along M
  softmax SOFTMAX  P = softmax(S / sqrt(hd)) causal -> MX (rows normalized before quantization)
  pv      GEMM     per KV head: O = P @ V, written into O [M][Hq*hd]
  oquant  EWQ      O -> MX
  oproj   GEMM     y_attn = O @ Wo
  norm2   PRENORM  h_mid = h_l + y_attn;  xn2 = MX(rmsnorm(h_mid) * g_post)
  gateup  GEMM     [G | U] = xn2 @ Wgu
  silu    EWQ      H = MX(silu(G) * U)
  down    GEMM     y_mlp = H @ Wd
  final   PRENORM  h_L = h_mid + y_mlp;  xf = MX(rmsnorm(h_L) * g_final);   lmhead GEMM logits = xf @ Wlm

MX = E4M3 codes with one E8M0 scale per 32 elements along the GEMM's K, scale = 2^floor(log2(amax)) (block max in
[1, 2): the MX Gemmini's tile accumulator overflows on codes near 448), codes by RNE (fp8_code_rshift), which the
device encoder reproduces bit for bit.

Models:
  tinyllama  TinyLlama-1.1B-Chat-v1.0, REAL weights, real text (wikitext-2), KV cache from the HF model.
             Checks: per-layer hidden state and logits vs the HF fp32 model and vs a numpy emulation of
             this quantized pipeline; top-1 agreement.
  llama7b    Llama-2-7B shapes, SYNTHETIC: random weights filled on the device before timing (one layer's
             weights shared by all 32 layers, one KV cache shared by all layers).  Performance only.
  smoke      small random model for RTL simulation, every stage checked against the emulation.

Memory: GEMM operands that only gemmini reads are reused across layers; every buffer that Muon reads is
written once per run (its L0d / L1 are not coherent with writes from gemmini or other clusters), except
the S ring (2 sets).  Arena buffers take no space in the ELF.
"""
import argparse, os, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from nightly_data import bf16_bits, bf16_to_f32

REGION_BASE = {1: 0x20000000, 2: 0x40000000, 3: 0x60000000}
REGION_END = {1: 0x40000000, 2: 0x60000000, 3: 0x7F000000}   # warp stacks from 0x7F00_0000
ALIGN = 256
TN = 64                         # GEMM n-tile (4 PE tiles)
TK = 256                        # GEMM k-step for the activation / weight GEMMs and PV

MODELS = {
    "tinyllama": dict(D=2048, F=5632, Hq=32, Hkv=4, hd=64, L=22, V=32000, eps=1e-5, theta=10000.0,
                      kind="real", hf="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    "llama7b": dict(D=4096, F=11008, Hq=32, Hkv=32, hd=128, L=32, V=32000, eps=1e-5, theta=10000.0,
                    kind="synthetic"),
    "smoke": dict(D=256, F=512, Hq=4, Hkv=2, hd=64, L=2, V=512, eps=1e-5, theta=10000.0, kind="random"),
    # smallest model with the ATTN (flash attention) phase: hd 128, >= 6 key blocks (use --ctx 768)
    "fasmoke": dict(D=256, F=512, Hq=2, Hkv=2, hd=128, L=1, V=512, eps=1e-5, theta=10000.0, kind="random"),
}

OPS = ["init", "norm1", "qkv", "rope", "qk", "softmax", "pv", "oquant", "oproj", "norm2", "gateup",
       "silu", "down", "final", "lmhead", "gemm", "gelu", "attn", "rope_k", "rope_v"]
OP = {n: i for i, n in enumerate(OPS)}
T_NONE, T_GEMM, T_PRENORM, T_ROPE, T_SOFTMAX, T_EWQ, T_FILL, T_SILU, T_ACT, T_FA = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
FA_BK, FA_QT = 128, 64          # ATTN phase: key block, query tile (fa_phase.h)
PW = 32                         # words per phase descriptor


# ================================ MX E4M3 (bit-exact with the device) ================================
def _rshift_rne(v, sh):
    sh = np.minimum(sh, 40)
    low = v & ((np.int64(1) << sh) - 1)
    half = np.int64(1) << (sh - 1)
    q = v >> sh
    return q + ((low > half) | ((low == half) & ((q & 1) == 1))).astype(np.int64)


def fp8_encode(x):
    """fp32 -> e4m3 code, RNE, saturating to +-448 (golden.fp8_code_rshift, vectorized)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    sign = (u >> 31) & 1
    ax = u & 0x7FFFFFFF
    e = ((ax >> 23) & 0xFF) - 127
    sig = (np.int64(1) << 23) | (ax & 0x7FFFFF)
    q = _rshift_rne(sig, np.maximum(14 - e, 1))
    out_sub = np.where(q >= 8, 8 | (q - 8), q)
    r = _rshift_rne(sig, np.full_like(sig, 20))
    carry = r == 16
    r = np.where(carry, 8, r)
    expf = e + carry + 7
    mant = r - 8
    sat = (expf > 15) | ((expf == 15) & (mant == 7))
    out = np.where(sat, 0x7E, (expf << 3) | mant)
    out = np.where(e < -6, out_sub, out)
    out = np.where(ax >= 0x43E00000, 0x7E, out)
    out = np.where(ax == 0, 0, out)
    return ((sign << 7) | out).astype(np.uint8)


_E4M3 = np.zeros(256, dtype=np.float32)
for _c in range(256):
    _s, _e, _m = _c >> 7, (_c >> 3) & 15, _c & 7
    _v = (_m / 8.0) * 2.0 ** -6 if _e == 0 else (1 + _m / 8.0) * 2.0 ** (_e - 7)
    if _e == 15 and _m == 7:
        _v = np.nan
    _E4M3[_c] = -_v if _s else _v


def block_exp(amax):
    """floor(log2(amax)), at least -126: the block maximum maps into [1, 2).  The MX Gemmini accumulates a 16-deep tile
    in a reduced-range format; codes near the E4M3 maximum overflow it (mx_golden: all NaN with 448-scaled blocks)."""
    b = np.ascontiguousarray(amax, dtype=np.float32).view(np.uint32).astype(np.int64)
    e = ((b >> 23) & 0xFF) - 127
    return np.clip(e, -126, 127)


def mx_quant(x):
    """Quantize along the last axis in groups of 32: (codes uint8 [...,K], scales uint8 [...,K/32])."""
    x = np.asarray(x, dtype=np.float32)
    g = x.reshape(x.shape[:-1] + (x.shape[-1] // 32, 32))
    e = block_exp(np.max(np.abs(g), axis=-1))
    y = np.ldexp(g, -e[..., None].astype(np.int32)).astype(np.float32)
    return fp8_encode(y).reshape(x.shape), (e + 127).astype(np.uint8)


def mx_dequant(codes, scales):
    v = _E4M3[codes].reshape(codes.shape[:-1] + (codes.shape[-1] // 32, 32))
    return np.ldexp(v, scales[..., None].astype(np.int32) - 127).reshape(codes.shape).astype(np.float32)


def bf(x):
    """round to bf16, back to fp32"""
    return bf16_to_f32(bf16_bits(x))


# ================================ device layouts ================================
def pack_a(codes, scales, TM, TKa):
    """A [R][K] -> pretiled [MT][KS][TM][TKa], scales [R][K/32] -> [MT][KS][TKa/32][TM]"""
    R, K = codes.shape
    MT, KS = R // TM, K // TKa
    a = codes.reshape(MT, TM, KS, TKa).transpose(0, 2, 1, 3)
    s = scales.T.reshape(KS, TKa // 32, MT, TM).transpose(2, 0, 1, 3)
    return np.ascontiguousarray(a), np.ascontiguousarray(s)


def pack_b(codes_kn, scales_gn, TKb):
    """B [K][N] codes, scales [K/32][N] -> pretiled [KS][NT][TKb][TN], scales [NT][KS][TKb/32][TN]"""
    K, N = codes_kn.shape
    KS, NT = K // TKb, N // TN
    b = codes_kn.reshape(KS, TKb, NT, TN).transpose(0, 2, 1, 3)
    s = scales_gn.reshape(KS, TKb // 32, NT, TN).transpose(2, 0, 1, 3)
    return np.ascontiguousarray(b), np.ascontiguousarray(s)


def quant_weight(w_out_in):
    """nn.Linear weight [out][in] (y = x W^T) -> B = W^T [in][out] MX along `in`, pretiled"""
    codes, sc = mx_quant(np.asarray(w_out_in, dtype=np.float32))   # groups along `in`, per output
    return pack_b(codes.T, sc.T, TK), (codes, sc)


def deq_weight(q):
    codes, sc = q
    return mx_dequant(codes, sc).T                                 # [in][out]


# ================================ memory plan ================================
class Mem:
    """Blob tensors (loaded with the ELF) per region, then arena buffers (address only) after them."""

    def __init__(self, name):
        self.name = name
        self.blobs = {1: bytearray(), 2: bytearray(), 3: bytearray()}
        self.arena = None
        self.defs = []

    def _pad(self, r):
        self.blobs[r].extend(b"\0" * ((-len(self.blobs[r])) % ALIGN))

    def add(self, region, array, sym=None):
        assert self.arena is None, "blob tensors before arena buffers"
        self._pad(region)
        addr = REGION_BASE[region] + len(self.blobs[region])
        self.blobs[region].extend(np.ascontiguousarray(array).tobytes())
        assert REGION_BASE[region] + len(self.blobs[region]) <= REGION_END[region], f"region {region} full"
        if sym:
            self.defs.append((f"{sym}_ADDR", f"0x{addr:08x}u"))
        return addr

    def alloc(self, nbytes, sym=None):
        if self.arena is None:
            self.arena = {r: REGION_BASE[r] + len(b) for r, b in self.blobs.items()}
        nbytes = (nbytes + ALIGN - 1) // ALIGN * ALIGN
        for r in (3, 2, 1):
            a = (self.arena[r] + ALIGN - 1) // ALIGN * ALIGN
            if a + nbytes <= REGION_END[r]:
                self.arena[r] = a + nbytes
                if sym:
                    self.defs.append((f"{sym}_ADDR", f"0x{a:08x}u"))
                return a
        raise SystemExit(f"out of GPU memory allocating {nbytes} bytes")

    def write(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        paths = []
        for r, blob in self.blobs.items():
            p = os.path.join(outdir, f"{self.name}.region{r}.bin")
            with open(p, "wb") as f:
                f.write(bytes(blob) if blob else b"\0" * ALIGN)
            paths.append(p)
        return paths


def f32_bits(x):
    return int(np.float32(x).view(np.uint32))


# ================================ model data ================================
def rope_tables(M, P0, hd, theta):
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.outer(np.arange(P0, P0 + M, dtype=np.float64), inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)   # [M][hd/2]


def rope(x, cos, sin):
    """x [M][H][hd] fp32, rotate_half convention (HF Llama)"""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(np.float32)


def load_hf(cfg, M, P0, layers):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], torch_dtype=torch.float32)
    model.eval()
    import glob
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet")))
    text = "".join(pq.read_table(files[0]).column("text").to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[:, : P0 + M + 1]
    full = model(ids[:, : P0 + M], output_hidden_states=True)
    past = model(ids[:, :P0], use_cache=True).past_key_values

    def kv(l):
        if hasattr(past, "key_cache"):
            return past.key_cache[l], past.value_cache[l]
        if hasattr(past, "layers"):
            return past.layers[l].keys, past.layers[l].values
        return past[l]

    sd = model.state_dict()
    W = []
    for l in range(layers):
        p = f"model.layers.{l}."
        g = lambda n: sd[p + n].float().numpy()
        k, v = kv(l)
        W.append(dict(
            wqkv=np.concatenate([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                 g("self_attn.v_proj.weight")], 0),
            wo=g("self_attn.o_proj.weight"),
            wgu=np.concatenate([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")], 0),
            wd=g("mlp.down_proj.weight"),
            g_in=g("input_layernorm.weight"), g_post=g("post_attention_layernorm.weight"),
            kc=k[0].float().numpy(), vc=v[0].float().numpy()))          # [Hkv][P0][hd]
    hs = [full.hidden_states[i][0, P0:].float().numpy() for i in range(len(full.hidden_states))]
    x0 = sd["model.embed_tokens.weight"][ids[0, P0:P0 + M]].float().numpy()
    return dict(layers=W, g_final=sd["model.norm.weight"].float().numpy(),
                wlm=sd["lm_head.weight"].float().numpy(), x0=x0, hidden=hs,
                logits=full.logits[0, P0:].float().numpy(), labels=ids[0, P0 + 1:P0 + M + 1].numpy())


def random_model(cfg, M, P0, layers, rng):
    D, F, Hq, Hkv, hd, V = cfg["D"], cfg["F"], cfg["Hq"], cfg["Hkv"], cfg["hd"], cfg["V"]
    Dq, Dkv = Hq * hd, Hkv * hd
    W = []
    for _ in range(layers):
        W.append(dict(
            wqkv=rng.normal(0, D ** -0.5, (Dq + 2 * Dkv, D)), wo=rng.normal(0, Dq ** -0.5, (D, Dq)),
            wgu=rng.normal(0, D ** -0.5, (2 * F, D)), wd=rng.normal(0, F ** -0.5, (D, F)),
            g_in=rng.uniform(0.5, 1.5, D), g_post=rng.uniform(0.5, 1.5, D),
            kc=rng.normal(0, 1, (Hkv, P0, hd)), vc=rng.normal(0, 1, (Hkv, P0, hd))))
    return dict(layers=W, g_final=rng.uniform(0.5, 1.5, D), wlm=rng.normal(0, D ** -0.5, (V, D)),
                x0=rng.normal(0, 1, (M, D)), hidden=None, logits=None, labels=None)


def gu_interleave(F):
    """Wgu output order with SILU fused into the gateup GEMM: [G_j | U_j] per 64 columns (C tiles 2j, 2j+1)"""
    j = np.arange(F // TN)[:, None] * TN + np.arange(TN)[None, :]
    return np.concatenate([j, F + j], axis=1).reshape(-1)


# ================================ emulation of the device pipeline ================================
def emulate(cfg, M, P0, mdl, qw, cos, sin, keep_layer0=True, fa=False, ilv=False):
    D, F, Hq, Hkv, hd, eps = cfg["D"], cfg["F"], cfg["Hq"], cfg["Hkv"], cfg["hd"], cfg["eps"]
    Dq, Dkv, Gq, SK = Hq * hd, Hkv * hd, Hq // Hkv, P0 + M
    out = dict(h=[], layer0={})

    def norm_q(h, gamma):
        ss = np.sum(h.astype(np.float32) ** 2, axis=1, dtype=np.float32)
        inv = (np.float32(1) / np.sqrt(ss / np.float32(D) + np.float32(eps))).astype(np.float32)
        y = (h * inv[:, None]).astype(np.float32) * bf(gamma)[None, :]
        return mx_dequant(*mx_quant(y))

    h = bf(mdl["x0"])
    hmid = ymlp = None
    for l, w in enumerate(mdl["layers"]):
        q = qw[l]
        if l > 0:
            h = bf(hmid + ymlp)
            out["h"].append(h)
        xn = norm_q(h, w["g_in"])
        qkv = bf(xn @ deq_weight(q["wqkv"]))
        Q = rope(qkv[:, :Dq].reshape(M, Hq, hd), cos, sin)
        Kn = rope(qkv[:, Dq:Dq + Dkv].reshape(M, Hkv, hd), cos, sin)
        Vn = qkv[:, Dq + Dkv:].reshape(M, Hkv, hd)
        Qd = mx_dequant(*mx_quant(Q))                                        # [M][Hq][hd]
        Kd = np.concatenate([mx_dequant(*mx_quant(np.asarray(w["kc"], np.float32))),
                             mx_dequant(*mx_quant(Kn.transpose(1, 0, 2)))], axis=1)   # [Hkv][SK][hd]
        Vall = np.concatenate([np.asarray(w["vc"], np.float32), Vn.transpose(1, 0, 2)], axis=1)
        Vd = mx_dequant(*mx_quant(Vall.transpose(0, 2, 1))).transpose(0, 2, 1)       # groups along keys
        scale = np.float32(1.0 / np.sqrt(hd))
        mask = np.arange(SK)[None, :] <= (P0 + np.arange(M))[:, None]
        O = np.zeros((M, Hq, hd), np.float32)
        S0 = None
        for h_ in range(Hq):
            g = h_ // Gq
            S = bf(Qd[:, h_, :] @ Kd[g].T)
            if h_ == 0:
                S0 = S
            if fa:   # ATTN phase: max of key block 0 only, unnormalized bf16 P, O = (P V) / l
                s = bf(S * bf(scale))
                m = s[:, :FA_BK].max(axis=1, keepdims=True)
                e = np.where(mask, bf(np.exp(bf(s - m))), 0).astype(np.float32)
                lsum = np.sum(e, axis=1, dtype=np.float32)
                O[:, h_, :] = bf(bf(mx_dequant(*mx_quant(e)) @ Vd[g]) * (np.float32(1) / lsum)[:, None])
                continue
            z = np.where(mask, S * scale, -np.inf).astype(np.float32)
            m = np.max(z, axis=1, keepdims=True)
            e = np.where(mask, bf(np.exp(bf(z - m))), 0).astype(np.float32)
            p = e * (np.float32(1) / np.sum(e, axis=1, keepdims=True, dtype=np.float32))
            O[:, h_, :] = bf(mx_dequant(*mx_quant(p)) @ Vd[g])
        O = O.reshape(M, Dq)
        yattn = bf(mx_dequant(*mx_quant(O)) @ deq_weight(q["wo"]))
        hmid = bf(h + yattn)
        xn2 = norm_q(hmid, w["g_post"])
        gu = bf(xn2 @ deq_weight(q["wgu"]))
        guo = gu[:, np.argsort(gu_interleave(F))] if ilv else gu         # back to [G | U]
        G, U = guo[:, :F], guo[:, F:]
        en = bf(np.exp(bf(-G)))
        Hh = (G / (np.float32(1) + en)).astype(np.float32) * U
        ymlp = bf(mx_dequant(*mx_quant(Hh)) @ deq_weight(q["wd"]))
        if l == 0 and keep_layer0:
            out["layer0"] = dict(qkv=qkv, s0=S0, o=O, yattn=yattn, hmid=hmid, gu=gu, ymlp=ymlp)
    h = bf(hmid + ymlp)
    out["h"].append(h)
    xf = norm_q(h, mdl["g_final"])
    out["logits"] = bf(xf @ deq_weight(qw["wlm"]))
    return out


# ================================ debug: layer-0 stage goldens ================================
def debug_l0_goldens(cfg, M, P0, mdl, qw, cos, sin, kv0, TM):
    """Layer 0 in device formats.  Every stage starts from the previous stage's golden; GEMMs by mx_golden
    (bit-exact with the hardware), the SIMT ops by the same algorithms as the device (exp-based ones may differ
    in a few codes)."""
    sys.path.insert(0, str(HERE.parents[2] / "lib" / "golden"))
    import golden
    D, F, Hq, Hkv, hd, eps = cfg["D"], cfg["F"], cfg["Hq"], cfg["Hkv"], cfg["hd"], cfg["eps"]
    Dq, Dkv, Gq, SK = Hq * hd, Hkv * hd, Hq // Hkv, P0 + M
    w, q = mdl["layers"][0], qw[0]
    out = {}
    tmp = HERE / "_gen" / "dbg"

    def mm(ac, asc, wq):                                  # A codes [M][K] + scales [M][K/32] @ weight (out, in)
        codes, sc = wq
        return golden.mx_matmul(ac, np.ascontiguousarray(codes.T), np.ascontiguousarray(asc.T),
                                np.ascontiguousarray(sc.T), ac.shape[0], codes.shape[0], ac.shape[1], tmpdir=str(tmp))

    def norm(h, gamma):
        ss = np.sum(h * h, axis=1, dtype=np.float32)
        inv = (np.float32(1) / np.sqrt(ss / np.float32(D) + np.float32(eps))).astype(np.float32)
        return mx_quant((h * inv[:, None]).astype(np.float32) * bf(gamma)[None, :])

    h0 = bf(mdl["x0"])
    out["h0"] = bf16_bits(h0)
    xc, xs = norm(h0, w["g_in"])
    out["xn1"], out["xn1_sc"] = pack_a(xc, xs, TM, TK)
    qkv = mm(xc, xs, q["wqkv"])                           # bf16 bits [M][NQKV]
    out["qkv"] = qkv
    qf = bf16_to_f32(qkv)
    Q = rope(qf[:, :Dq].reshape(M, Hq, hd), cos, sin)
    Kn = rope(qf[:, Dq:Dq + Dkv].reshape(M, Hkv, hd), cos, sin)
    Vn = qf[:, Dq + Dkv:].reshape(M, Hkv, hd)
    qc, qs = mx_quant(Q)
    kc, ks = kv0["kcodes"].copy(), kv0["ksc"].copy()      # [Hkv][SK][hd], [Hkv][SK][hd/32]
    c_, s_ = mx_quant(Kn.transpose(1, 0, 2))
    kc[:, P0:], ks[:, P0:] = c_, s_
    vc, vs = kv0["vcodes"].copy(), kv0["vsc"].copy()      # [Hkv][hd][SK], [Hkv][hd][SK/32]
    c_, s_ = mx_quant(Vn.transpose(1, 2, 0))
    vc[:, :, P0:], vs[:, :, P0 // 32:] = c_, s_
    qa, qsa, kb, kbs, vb, vbs, S_all, Sb = [], [], [], [], [], [], [], []
    scale = np.float32(1.0 / np.sqrt(hd))
    mask = np.arange(SK)[None, :] <= (P0 + np.arange(M))[:, None]
    pc_all, ps_all, O = [], [], np.zeros((M, Dq), np.uint16)
    for g in range(Hkv):
        rows = np.concatenate([qc[:, g * Gq + j] for j in range(Gq)], 0)
        rsc = np.concatenate([qs[:, g * Gq + j] for j in range(Gq)], 0)
        a_, b_ = pack_a(rows, rsc, TM, hd); qa.append(a_); qsa.append(b_)
        a_, b_ = pack_b(kc[g].T, ks[g].T, hd); kb.append(a_); kbs.append(b_)
        a_, b_ = pack_b(vc[g].T, vs[g].T, TK); vb.append(a_); vbs.append(b_)
        S = golden.mx_matmul(rows, np.ascontiguousarray(kc[g].T), np.ascontiguousarray(rsc.T),
                             np.ascontiguousarray(ks[g].T), Gq * M, SK, hd, tmpdir=str(tmp))
        Sb.append(S)
        Sf = bf16_to_f32(S)
        mrow = np.tile(mask, (Gq, 1))
        z = np.where(mrow, Sf * scale, -np.inf).astype(np.float32)
        m = np.max(z, axis=1, keepdims=True)
        e = np.where(mrow, bf(np.exp(bf(Sf * scale - m))), 0).astype(np.float32)
        p = e * (np.float32(1) / np.sum(e, axis=1, keepdims=True, dtype=np.float32))
        pc, ps = mx_quant(p)
        a_, b_ = pack_a(pc, ps, TM, TK); pc_all.append(a_); ps_all.append(b_)
        Og = golden.mx_matmul(pc, np.ascontiguousarray(vc[g].T), np.ascontiguousarray(ps.T),
                              np.ascontiguousarray(vs[g].T), Gq * M, hd, SK, tmpdir=str(tmp))
        for j in range(Gq):
            O[:, (g * Gq + j) * hd:(g * Gq + j + 1) * hd] = Og[j * M:(j + 1) * M]
    out["q"], out["q_sc"] = np.stack(qa), np.stack(qsa)
    out["kc"], out["kc_sc"], out["vc"], out["vc_sc"] = np.stack(kb), np.stack(kbs), np.stack(vb), np.stack(vbs)
    out["s"] = np.stack(Sb)
    out["p"], out["p_sc"] = np.stack(pc_all), np.stack(ps_all)
    out["o"] = O
    oc, os_ = mx_quant(bf16_to_f32(O))
    out["omx"], out["omx_sc"] = pack_a(oc, os_, TM, TK)
    y = mm(oc, os_, q["wo"])
    out["yattn"] = y
    hmid = bf(h0 + bf16_to_f32(y))
    out["hmid"] = bf16_bits(hmid)
    x2c, x2s = norm(hmid, w["g_post"])
    out["xn2"], out["xn2_sc"] = pack_a(x2c, x2s, TM, TK)
    gu = mm(x2c, x2s, q["wgu"])
    out["gu"] = gu
    guf = bf16_to_f32(gu)
    G_, U_ = guf[:, :F], guf[:, F:]
    Hh = (G_ / (np.float32(1) + bf(np.exp(bf(-G_))))).astype(np.float32) * U_
    hc, hs = mx_quant(Hh)
    out["hmx"], out["hmx_sc"] = pack_a(hc, hs, TM, TK)
    out["ymlp"] = mm(hc, hs, q["wd"])
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


# ================================ phase program ================================
class Prog:
    def __init__(self):
        self.ph = []

    def add(self, typ, layer, op, *words):
        d = [typ, ((layer & 0xFF) << 8) | OP[op]] + [int(w) & 0xFFFFFFFF for w in words]
        assert len(d) <= PW, (op, len(d))
        self.ph.append(d + [0] * (PW - len(d)))

    def gemm(self, layer, op, a, asc, a_bs, asc_bs, b, bsc, b_bs, bsc_bs, c, c_bs, c_rs, c_ms,
             batch, MT, NT, KS, TM, TKg, fuse=()):
        """fuse: work for warps 1.. while the GEMM runs (kernel FUSE_*): (1, H codes, H scales, F, M, TM, TK) SILU,
        (2, ROPE descriptor address) ROPE, (3, x, hout) PRENORM accumulation"""
        assert (TM // 16) * 4 * (TKg // 16) >= 64, "compute loop too short for the scale-load ordering"
        assert not fuse or (batch * MT * NT % 2 == 0 and NT <= 509), "fused GEMMs split tile pairs, table <= 509"
        self.add(T_GEMM, layer, op, a, asc, a_bs, asc_bs, b, bsc, b_bs, bsc_bs, c, c_bs, c_rs, c_ms,
                 batch, MT, NT, KS, TM, TKg, *fuse)

    def array(self):
        return np.array(self.ph, dtype=np.uint32).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="smoke")
    ap.add_argument("--M", type=int, default=None, help="chunk tokens (64 or 128)")
    ap.add_argument("--ctx", type=int, default=None, help="SK = KV cache + chunk (multiple of 256)")
    ap.add_argument("--layers", type=int, default=None, help="decoder layers (default: the model's)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--attn", choices=["auto", "fa", "gemm"], default="auto",
                    help="attention: fa = one ATTN phase (flash attention, hd 128), gemm = qk / softmax / pv phases")
    ap.add_argument("--no-emulate", action="store_true", help="skip the quantized-pipeline reference")
    ap.add_argument("--debug-l0", action="store_true",
                    help="layer 0 gets its own MX operand buffers and a stage-by-stage check, each stage's golden "
                         "computed from the previous stage's golden (GEMMs by mx_golden)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--header", default="llama_data.h")
    a = ap.parse_args()

    cfg = dict(MODELS[a.model])
    M = a.M or (64 if a.model == "smoke" else 128)
    SK = a.ctx or (256 if a.model == "smoke" else 1024)
    L = a.layers or cfg["L"]
    P0 = SK - M
    D, F, Hq, Hkv, hd, V = cfg["D"], cfg["F"], cfg["Hq"], cfg["Hkv"], cfg["hd"], cfg["V"]
    Dq, Dkv, Gq, NQKV = Hq * hd, Hkv * hd, Hq // Hkv, Hq * hd + 2 * Hkv * hd
    TM = M
    assert M in (64, 128) and SK % 256 == 0 and P0 % 32 == 0 and P0 >= 0
    assert D % TK == 0 and F % TK == 0 and Dq % TK == 0 and hd in (64, 128)
    for n in (NQKV, D, 2 * F, V, SK):
        assert n % TN == 0, n
    synthetic = cfg["kind"] == "synthetic"
    rng = np.random.default_rng(a.seed)
    fa_ok = (hd == 128 and SK % FA_BK == 0 and SK // FA_BK >= 6 and P0 >= FA_BK and (Hq * M // FA_QT) % 4 == 0
             and P0 % 64 == 0 and M % 64 == 0)
    fa = a.attn == "fa" or (a.attn == "auto" and fa_ok)
    assert fa_ok or not fa, "ATTN phase needs hd 128, >= 6 key blocks of 128, P0 >= 128, Hq M / 64 % 4 == 0"
    assert not (fa and a.debug_l0), "--debug-l0 checks the qk / softmax / pv phases"
    NBk = SK // FA_BK
    fuse = fa                                            # SILU fused into the gateup GEMM (with the ATTN path)
    assert not fuse or (F % TK == 0 and (2 * F // TN) % 8 == 0 and M % 32 == 0 and M <= 256)

    cos, sin = rope_tables(M, P0, hd, cfg["theta"])
    if cfg["kind"] == "real":
        mdl = load_hf(cfg, M, P0, L)
    elif cfg["kind"] == "random":
        mdl = random_model(cfg, M, P0, L, rng)
    else:
        mdl = None

    mem = Mem("llama")
    prog = Prog()
    checks = []          # (name, dev, gold, n, scale_log2)
    KSv = SK // TK

    # ---------------- blob: weights, caches, inputs ----------------
    def put_weight(q, region):
        (bt, bs), _ = q
        return mem.add(region, bt), mem.add(region, bs)

    lw = []              # per layer: dict of device addresses
    qw = {}              # quantized weights, for the emulation
    # one shared layer and one shared KV cache for the synthetic model (device-filled, see the arena)
    sizes = dict(wqkv=(D * NQKV, NQKV * D // 32), wo=(Dq * D, D * Dq // 32),
                 wgu=(D * 2 * F, 2 * F * D // 32), wd=(F * D, D * F // 32))
    prog_addr = mem.add(3, np.zeros(PW * 1024, np.uint32))   # patched once the arena is laid out
    if not synthetic:
        for l, w in enumerate(mdl["layers"]):
            region = 1 if l < (L + 1) // 2 else 2
            q = {k: quant_weight(w[k][gu_interleave(F)] if k == "wgu" and fuse else w[k])
                 for k in ("wqkv", "wo", "wgu", "wd")}
            qw[l] = {k: v[1] for k, v in q.items()}
            d = {}
            for k in ("wqkv", "wo", "wgu", "wd"):
                d[k], d[k + "_sc"] = put_weight(q[k], region)
            d["g_in"] = mem.add(3, bf16_bits(w["g_in"]))
            d["g_post"] = mem.add(3, bf16_bits(w["g_post"]))
            # KV cache: K^T [hd][SK] as B (KS=1, TK=hd), V [SK][hd] as B (TK=256), cached keys 0..P0-1
            kc = np.zeros((Hkv, SK, hd), np.float32)
            vc = np.zeros((Hkv, SK, hd), np.float32)
            kc[:, :P0] = w["kc"]
            vc[:, :P0] = w["vc"]
            kcodes, ksc = mx_quant(kc)                                # groups along hd per key
            vcodes, vsc = mx_quant(vc.transpose(0, 2, 1))             # groups along keys per d
            if fa:   # K^T [hd][SK], scales [NB][hd/32][128]; V [SK][hd], scales [NB][128/32][hd]
                kb = [kcodes[g].T for g in range(Hkv)]
                kbs = [ksc[g].reshape(NBk, FA_BK, hd // 32).transpose(0, 2, 1) for g in range(Hkv)]
                vb = [vcodes[g].T for g in range(Hkv)]
                vbs = [vsc[g].T.reshape(NBk, FA_BK // 32, hd) for g in range(Hkv)]
            else:
                kb, kbs = zip(*[pack_b(kcodes[g].T, ksc[g].T, hd) for g in range(Hkv)])
                vb, vbs = zip(*[pack_b(vcodes[g].T, vsc[g].T, TK) for g in range(Hkv)])
            d["kc"] = mem.add(3, np.stack(kb))
            d["kc_sc"] = mem.add(3, np.stack(kbs))
            d["vc"] = mem.add(3, np.stack(vb))
            d["vc_sc"] = mem.add(3, np.stack(vbs))
            if l == 0:
                kv0 = dict(kcodes=kcodes, ksc=ksc, vcodes=vcodes, vsc=vsc)
            lw.append(d)
        ql = quant_weight(mdl["wlm"])
        qw["wlm"] = ql[1]
        wlm, wlm_sc = put_weight(ql, 3)
        g_final = mem.add(3, bf16_bits(mdl["g_final"]))
        x0 = mem.add(3, bf16_bits(mdl["x0"]))
    if synthetic:
        x0 = mem.add(3, bf16_bits(rng.normal(0, 1, (M, D))))
        gam = mem.add(3, bf16_bits(rng.uniform(0.5, 1.5, (2 * L + 1, D))))
    zeros = mem.add(3, np.zeros(M * D, np.uint16))          # layer 0's norm adds zero
    cos_a = mem.add(3, cos)
    sin_a = mem.add(3, sin)

    # references (in the ELF)
    emu = None
    if mdl is not None and not a.no_emulate:
        print("emulating the quantized pipeline ...", flush=True)
        emu = emulate(cfg, M, P0, mdl, qw, cos, sin, fa=fa, ilv=fuse)
    gold = {}
    if emu is not None:
        for l, hh in enumerate(emu["h"]):
            gold[f"emu_h{l + 1}"] = mem.add(3, bf16_bits(hh))
        gold["emu_logits"] = mem.add(3, bf16_bits(emu["logits"]))
        if a.model == "smoke":
            for k, v in emu["layer0"].items():
                gold[f"emu_l0_{k}"] = mem.add(3, bf16_bits(v))
    dbg = None
    if a.debug_l0:
        assert mdl is not None and not synthetic
        print("debug-l0: stage goldens (mx_golden) ...", flush=True)
        dbg = debug_l0_goldens(cfg, M, P0, mdl, qw, cos, sin, kv0, TM)
        dbg_gold = {k: mem.add(3, v) for k, v in dbg.items()}
    if mdl is not None and mdl["hidden"] is not None:
        for l in range(1, L):
            gold[f"hf_h{l}"] = mem.add(3, bf16_bits(mdl["hidden"][l]))
        gold["hf_logits"] = mem.add(3, bf16_bits(mdl["logits"]))
        gold["labels"] = mem.add(3, mdl["labels"].astype(np.uint32))

    # ---------------- arena ----------------
    al = mem.alloc
    if synthetic:
        sw = {}
        for k, (nb, ns) in sizes.items():
            sw[k], sw[k + "_sc"] = al(nb), al(ns)
        kvb = Hkv * SK * hd
        sw["kc"], sw["kc_sc"], sw["vc"], sw["vc_sc"] = al(kvb), al(kvb // 32), al(kvb), al(kvb // 32)
        wlm, wlm_sc = al(D * V), al(V * D // 32)
        lw = []
        for l in range(L):
            d = dict(sw)
            d["g_in"], d["g_post"] = gam + (2 * l) * D * 2, gam + (2 * l + 1) * D * 2
            lw.append(d)
        g_final = gam + 2 * L * D * 2
        fills = [(sw[k], sizes[k][0], 0, 0x9E3779B9 + i) for i, k in enumerate(sizes)]
        fills += [(sw[k + "_sc"], sizes[k][1], 1, 0x7A7A7A7A) for k in sizes]          # weights ~ +-2 * 2^-5
        fills += [(sw["kc"], kvb, 0, 0x1234567), (sw["vc"], kvb, 0, 0x7654321),
                  (sw["kc_sc"], kvb // 32, 1, 0x7F7F7F7F), (sw["vc_sc"], kvb // 32, 1, 0x7F7F7F7F),
                  (wlm, D * V, 0, 0xABCDEF1), (wlm_sc, V * D // 32, 1, 0x7A7A7A7A)]
        for base, nb, mode, val in fills:
            assert nb % 4 == 0
            prog.add(T_FILL, 255, "init", base, nb, mode, val)

    xn = al(M * D), al(M * D // 32)                      # MX A operands (gemmini-read, reused)
    xf = al(M * D), al(M * D // 32)
    qmx = al(Hq * M * hd), al(Hq * M * hd // 32)
    pmx = al(Hq * M * SK), al(Hq * M * SK // 32)
    omx = al(M * Dq), al(M * Dq // 32)
    hmx = al(M * F), al(M * F // 32)
    if dbg is not None:   # layer 0's own MX operands, kept for the host check
        l0mx = dict(xn1=(al(M * D), al(M * D // 32)), q=(al(Hq * M * hd), al(Hq * M * hd // 32)),
                    p=(al(Hq * M * SK), al(Hq * M * SK // 32)), omx=(al(M * Dq), al(M * Dq // 32)),
                    xn2=(al(M * D), al(M * D // 32)), hmx=(al(M * F), al(M * F // 32)))
    s_ring = [al(Hq * M * SK * 2) for _ in range(2)]     # Muon-read, 2 sets
    perf = al(64 * 4 * 1024)                          # [phase][cluster][start, end]; phases < 1024
    logits = al(M * V * 2)
    hbuf = [al(M * D * 2) for _ in range(L + 1)]         # h_0 (= x0 + 0) .. h_L (write once)
    act = []
    for l in range(L):
        act.append(dict(qkv=al(M * NQKV * 2), o=al(M * Dq * 2), yattn=al(M * D * 2), hmid=al(M * D * 2),
                        gu=al(M * 2 * F * 2), ymlp=al(M * D * 2)))

    # ---------------- phases ----------------
    eps = f32_bits(cfg["eps"])
    xn_all, qmx_all, pmx_all, omx_all, hmx_all = xn, qmx, pmx, omx, hmx
    for l in range(L):
        w, ac = lw[l], act[l]
        xn2 = xn
        if dbg is not None and l == 0:
            xn, qmx, pmx, omx, hmx = l0mx["xn1"], l0mx["q"], l0mx["p"], l0mx["omx"], l0mx["hmx"]
            xn2 = l0mx["xn2"]
        elif dbg is not None:
            xn, qmx, pmx, omx, hmx = xn_all, qmx_all, pmx_all, omx_all, hmx_all
            xn2 = xn
        if l == 0:
            prog.add(T_PRENORM, l, "norm1", x0, zeros, hbuf[0], w["g_in"], xn[0], xn[1], D, M, TM, TK, eps)
        else:
            pa = act[l - 1]
            prog.add(T_PRENORM, l, "norm1", pa["hmid"], pa["ymlp"], hbuf[l], w["g_in"], xn[0], xn[1],
                     D, M, TM, TK, eps, int(fuse))
        rope_args = (ac["qkv"], NQKV, Hq, Hkv, hd, cos_a, sin_a, qmx[0], qmx[1], Gq * M * hd, Gq * M * hd // 32,
                     w["kc"], w["kc_sc"], SK * hd, SK * hd // 32, w["vc"], w["vc_sc"], SK * hd, SK * hd // 32,
                     P0, M, SK, FA_QT if fa else TM, int(fa), 0)
        # fused: ROPE runs on warps 1.. under the qkv GEMM; its descriptor rides in the no-op phase right after it
        rope_desc = prog_addr + (len(prog.ph) + 1) * PW * 4
        prog.gemm(l, "qkv", xn[0], xn[1], 0, 0, w["wqkv"], w["wqkv_sc"], 0, 0, ac["qkv"], 0, NQKV * 2,
                  TM * NQKV * 2, 1, 1, NQKV // TN, D // TK, TM, TK, fuse=(2, rope_desc) if fuse else ())
        prog.add(T_NONE if fuse else T_ROPE, l, "rope", *rope_args)
        if fa:
            # O goes straight into oproj's MX A operand (no oquant phase)
            prog.add(T_FA, l, "attn", qmx[0], qmx[1], w["kc"], w["kc_sc"], w["vc"], w["vc_sc"], omx[0], Hq, Gq, M,
                     SK, P0, 0, f32_bits(1.0 / np.sqrt(hd)), SK * hd, SK * hd // 32, SK * hd, SK * hd // 32,
                     omx[1], TM, TK, Dq // TK)
        else:
            s = s_ring[l & 1]
            prog.gemm(l, "qk", qmx[0], qmx[1], Gq * M * hd, Gq * M * hd // 32, w["kc"], w["kc_sc"],
                      SK * hd, SK * hd // 32, s, Gq * M * SK * 2, SK * 2, TM * SK * 2,
                      Hkv, Gq * M // TM, SK // TN, 1, TM, hd)
            prog.add(T_SOFTMAX, l, "softmax", s, Gq * M * SK * 2, SK, Gq * M, Hkv, pmx[0], pmx[1],
                     Gq * M * SK, Gq * M * SK // 32, f32_bits(1.0 / np.sqrt(hd)), P0, M, TM, TK)
            prog.gemm(l, "pv", pmx[0], pmx[1], Gq * M * SK, Gq * M * SK // 32, w["vc"], w["vc_sc"],
                      SK * hd, SK * hd // 32, ac["o"], Gq * hd * 2, Dq * 2, hd * 2,
                      Hkv, Gq * M // TM, hd // TN, KSv, TM, TK)
        if not fa:
            prog.add(T_EWQ, l, "oquant", ac["o"], Dq, 0, Dq, M, omx[0], omx[1], TM, TK)
        prog.gemm(l, "oproj", omx[0], omx[1], 0, 0, w["wo"], w["wo_sc"], 0, 0, ac["yattn"], 0, D * 2,
                  TM * D * 2, 1, 1, D // TN, Dq // TK, TM, TK, fuse=(3, hbuf[l], ac["hmid"]) if fuse else ())
        prog.add(T_PRENORM, l, "norm2", hbuf[l], ac["yattn"], ac["hmid"], w["g_post"], xn2[0], xn2[1],
                 D, M, TM, TK, eps, int(fuse))
        prog.gemm(l, "gateup", xn2[0], xn2[1], 0, 0, w["wgu"], w["wgu_sc"], 0, 0, ac["gu"], 0, 2 * F * 2,
                  TM * 2 * F * 2, 1, 1, 2 * F // TN, D // TK, TM, TK,
                  fuse=(1, hmx[0], hmx[1], F, M, TM, TK) if fuse else ())
        if not fuse:
            prog.add(T_SILU, l, "silu", ac["gu"], 2 * F, 1, F, M, hmx[0], hmx[1], TM, TK)
        # fused: the next norm1 (or final)'s h = h_mid + y_mlp and row sums under the down GEMM
        prog.gemm(l, "down", hmx[0], hmx[1], 0, 0, w["wd"], w["wd_sc"], 0, 0, ac["ymlp"], 0, D * 2,
                  TM * D * 2, 1, 1, D // TN, F // TK, TM, TK, fuse=(3, ac["hmid"], hbuf[l + 1]) if fuse else ())
    if dbg is not None:
        xn, qmx, pmx, omx, hmx = xn_all, qmx_all, pmx_all, omx_all, hmx_all
    pa = act[L - 1]
    prog.add(T_PRENORM, 255, "final", pa["hmid"], pa["ymlp"], hbuf[L], g_final, xf[0], xf[1],
             D, M, TM, TK, eps, int(fuse))
    prog.gemm(255, "lmhead", xf[0], xf[1], 0, 0, wlm, wlm_sc, 0, 0, logits, 0, V * 2, TM * V * 2,
              1, 1, V // TN, D // TK, TM, TK)
    NP = len(prog.ph)
    assert NP <= 1024

    # checks: (name, device buffer, golden, elements, fixed-point scale[, bytes, tolerance])
    if dbg is not None:
        l0, kq = act[0], lw[0]
        G = dbg_gold
        def bchk(name, dev, key, tol=0):
            checks.append((f"DBG {name} (bytes)", dev, G[key], dbg[key].nbytes, 8, 1, tol))
        def hchk(name, dev, key, tol=0):
            checks.append((f"DBG {name} (bf16)", dev, G[key], dbg[key].size, 12, 0, tol))
        bchk("1 norm1 xn codes", l0mx["xn1"][0], "xn1")
        bchk("1 norm1 xn scales", l0mx["xn1"][1], "xn1_sc")
        hchk("1 norm1 h0", hbuf[0], "h0")
        hchk("2 qkv = xn1 @ Wqkv", l0["qkv"], "qkv")
        bchk("3 rope Q codes", l0mx["q"][0], "q")
        bchk("3 rope Q scales", l0mx["q"][1], "q_sc")
        bchk("3 rope K^T cache codes", kq["kc"], "kc")
        bchk("3 rope K^T cache scales", kq["kc_sc"], "kc_sc")
        bchk("3 rope V cache codes", kq["vc"], "vc")
        bchk("3 rope V cache scales", kq["vc_sc"], "vc_sc")
        hchk("4 qk S = Q @ K^T", s_ring[0], "s")
        bchk("5 softmax P codes", l0mx["p"][0], "p", dbg["p"].nbytes // 100)
        bchk("5 softmax P scales", l0mx["p"][1], "p_sc", dbg["p_sc"].nbytes // 100)
        hchk("6 pv O = P @ V", l0["o"], "o", 200)
        bchk("7 oquant codes", l0mx["omx"][0], "omx", dbg["omx"].nbytes // 50)
        bchk("7 oquant scales", l0mx["omx"][1], "omx_sc", dbg["omx_sc"].nbytes // 50)
        hchk("8 oproj y = O @ Wo", l0["yattn"], "yattn", 200)
        hchk("9 norm2 h_mid", l0["hmid"], "hmid", 200)
        bchk("9 norm2 xn2 codes", l0mx["xn2"][0], "xn2", dbg["xn2"].nbytes // 50)
        hchk("10 gateup = xn2 @ Wgu", l0["gu"], "gu", 200)
        bchk("11 silu H codes", l0mx["hmx"][0], "hmx", dbg["hmx"].nbytes // 50)
        hchk("12 down = H @ Wd", l0["ymlp"], "ymlp", 300)
    if emu is not None:
        if a.model == "smoke":
            l0 = act[0]
            for k, dev, n in (("qkv", l0["qkv"], M * NQKV), ("s0", s_ring[0], M * SK), ("o", l0["o"], M * Dq),
                              ("yattn", l0["yattn"], M * D), ("hmid", l0["hmid"], M * D),
                              ("gu", l0["gu"], M * 2 * F), ("ymlp", l0["ymlp"], M * D)):
                checks.append((f"L0 {k} vs emulation", dev, gold[f"emu_l0_{k}"], n, 12))
        for l in range(1, L + 1):
            checks.append((f"h{l} vs emulation", hbuf[l], gold[f"emu_h{l}"], M * D, 8))
            if f"hf_h{l}" in gold:
                checks.append((f"h{l} vs HF fp32", hbuf[l], gold[f"hf_h{l}"], M * D, 8))
        checks.append(("logits vs emulation", logits, gold["emu_logits"], M * V, 10))
    if "hf_logits" in gold and L == cfg["L"]:
        checks.append(("logits vs HF fp32", logits, gold["hf_logits"], M * V, 10))

    # patch the program into its reserved blob slot
    pa_ = prog.array().tobytes()
    off = prog_addr - REGION_BASE[3]
    mem.blobs[3][off:off + len(pa_)] = pa_

    paths = mem.write(a.out)
    nfill = sum(1 for p in prog.ph if p[0] == T_FILL)
    with open(a.header, "w") as f:
        f.write(f"// generated by llama_e2e/gen_data.py --model {a.model}; do not edit\n")
        f.write("#ifndef LLAMA_DATA_H\n#define LLAMA_DATA_H\n#include <stdint.h>\n")
        for k, v in dict(LL_M=M, LL_SK=SK, LL_P0=P0, LL_L=L, LL_D=D, LL_F=F, LL_HQ=Hq, LL_HKV=Hkv,
                         LL_HD=hd, LL_V=V, LL_NP=NP, LL_NFILL=nfill, LL_PW=PW, LL_TN=TN,
                         LL_SYNTHETIC=int(synthetic)).items():
            f.write(f"#define {k} {v}u\n")
        f.write(f'#define LL_MODEL "{a.model}"\n')
        f.write(f"#define LL_PROG_ADDR 0x{prog_addr:08x}u\n#define LL_PERF_ADDR 0x{perf:08x}u\n")
        f.write(f"#define LL_LOGITS_ADDR 0x{logits:08x}u\n")
        full = L == cfg["L"]                             # HF logits only mean something for the whole model
        f.write(f"#define LL_LABELS_ADDR 0x{gold.get('labels', 0) if full else 0:08x}u\n")
        f.write(f"#define LL_HF_LOGITS_ADDR 0x{gold.get('hf_logits', 0) if full else 0:08x}u\n")
        f.write(f"#define LL_EMU_LOGITS_ADDR 0x{gold.get('emu_logits', 0):08x}u\n")
        f.write("#define LL_T_GEMM 1u\n#define LL_T_PRENORM 2u\n#define LL_T_ROPE 3u\n#define LL_T_SOFTMAX 4u\n"
                "#define LL_T_EWQ 5u\n#define LL_T_FILL 6u\n#define LL_T_SILU 7u\n#define LL_T_ACT 8u\n#define LL_T_FA 9u\n")
        f.write("#ifdef LL_HOST\n")
        f.write("static const char *const LL_OPS[] = {" + ", ".join(f'"{o}"' for o in OPS) + "};\n")
        f.write(f"#define LL_NOPS {len(OPS)}u\n")
        f.write(f"#define LL_NCHECKS {len(checks)}u\n")
        f.write("#define LL_CPU_ONLY 0\n#define LL_NCSUM 0u\nstatic const uint32_t LL_CSUM[1][3] = {{0, 0, 0}};\n")
        f.write("struct LlCheck { const char *name; uint32_t dev, gold, n; int scale_log2; int bytes; uint32_t tol; };\n")
        f.write("static const LlCheck LL_CHECKS[] = {\n")
        for c in checks:
            name, dev, g, n, sc = c[:5]
            kind, tol = (c[5], c[6]) if len(c) > 5 else (0, 0)
            f.write(f'  {{"{name}", 0x{dev:08x}u, 0x{g:08x}u, {n}u, {sc}, {kind}, {tol}u}},\n')
        f.write("  {0, 0, 0, 0, 0, 0, 0}\n};\n#endif\n#endif\n")
    used = {r: (mem.arena[r] - REGION_BASE[r]) >> 20 for r in (1, 2, 3)}
    blob = {r: len(b) >> 20 for r, b in mem.blobs.items()}
    print(f"{a.model}: M={M} SK={SK} (cache {P0}) L={L} phases={NP} (fill {nfill}); "
          f"blob MiB {blob}, used MiB {used}")
    if emu is not None and mdl.get("labels") is not None:
        top = np.argmax(emu["logits"], axis=1)
        print(f"emulation: top-1 vs HF {np.mean(top == np.argmax(mdl['logits'], 1)) * 100:.1f}%, "
              f"vs next token {np.mean(top == mdl['labels']) * 100:.1f}% "
              f"(HF itself {np.mean(np.argmax(mdl['logits'], 1) == mdl['labels']) * 100:.1f}%)")


if __name__ == "__main__":
    main()
