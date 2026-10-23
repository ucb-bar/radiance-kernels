#!/usr/bin/env python3
"""llama_e2e/gen_vla.py: SmolVLA (lerobot/smolvla_base) on Radiance: data and phase program for kernel.cpp.

One action chunk (50 x 32, 10 flow-matching steps) from 3 camera images, a 6-D state and an instruction, every stage on
the device.  The reference is lerobot's SmolVLAPolicy in fp32; `forward(q=False)` below re-implements it in numpy
(checked against lerobot's actions here), `forward(q=True)` is the same pipeline with the device's quantization points
(MX E4M3 at every GEMM input, bf16 at every GEMM output, the device's exp / GELU / SiLU): the emulation the host checks
are judged against.

Device pipeline (rows padded to 64-multiples, widths to 256-multiples; norms over the true width):
  vision   3 images x 1024 patches = 3072 rows: patch GEMM, AFFINE (bias + position), 12 x [LN1, qkv GEMM, ROPE (q/k/v
           bias, identity rotation, FA layouts) x 3 images, ATTN (36 heads, no mask), out_proj GEMM, LN2 (+ out_proj
           bias), fc1 GEMM, GELU(x + b) quantize, fc2 GEMM], post-LayerNorm (+ fc2 bias, bf16 out)
  connect  GATHERQ (pixel shuffle) -> connector GEMM -> prefix rows 0..191, AFFINE x sqrt(960); state GEMM + AFFINE
           (bias) -> prefix row 204; instruction embeddings (host, x sqrt(960)) in rows 192..203
  prefix   256 rows (205 valid: images, instruction, state; padding last), 16 SmolLM2 layers on the qk / softmax / pv
           GEMM path with a key-mask table; K / V cached for the expert (keys 0..255 of a 512-key cache); odd layers also
           keep K (A operand, from ROPE) and V (EWQ) as the cross-attention sources
  cross    8 odd layers: K' = K Wk^T, V' = V Wv^T (expert projections), ROPE (identity) into K'^T / V' caches
  denoise  10 x [EULER (x += dt v, quantize), time-MLP GEMM (action_in_proj folded in, per-step bias = time embedding),
           SiLU(x + c) quantize, mlp_out GEMM, 16 expert layers (even: self-attention over [prefix ; suffix keys
           256..], odd: cross-attention over K' / V'), final norm, action_out GEMM], EULER
"""
import argparse, math, os, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_data as g                                     # MX / bf16 helpers, Mem, layouts
from gen_data import bf, bf16_bits, mx_quant, mx_dequant, f32_bits

F32 = np.float32


# ================================ inputs and weights (lerobot) ================================
def load_reference(seed=0, text="pick up the red cube and place it in the bowl\n"):
    """lerobot SmolVLAPolicy (fp32), fixed synthetic inputs, its action chunk; returns numpy weights + inputs."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
    torch.manual_seed(seed)
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").float().eval()
    cfg, m = policy.config, policy.model
    rng = np.random.default_rng(seed)
    batch = {}
    for key in cfg.image_features:                       # smooth synthetic scenes in [0, 1]
        yy, xx = np.mgrid[0:256, 0:256] / 255.0
        img = np.stack([0.5 + 0.5 * np.sin(6 * xx + rng.uniform(0, 6)), yy, 0.5 + 0.4 * np.cos(5 * yy * xx)], 0)
        batch[key] = torch.tensor(img[None], dtype=torch.float32)
    batch["observation.state"] = torch.tensor(rng.normal(0, 1, (1, 6)), dtype=torch.float32)
    tok = m.vlm_with_expert.processor.tokenizer
    enc = tok([text], padding="max_length", max_length=cfg.tokenizer_max_length, return_tensors="pt")
    lang, lmask = enc["input_ids"], enc["attention_mask"].bool()
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim)
    with torch.no_grad():
        actions = m.sample_actions(images, img_masks, lang, lmask, state, noise=noise)
    sd = {k: v.float().numpy() for k, v in policy.state_dict().items()}
    P = "model.vlm_with_expert."
    vlm, ex = m.vlm_with_expert.get_vlm_model(), m.vlm_with_expert.lm_expert
    nvalid = int(lmask.sum())
    steps = cfg.num_steps
    times = [1.0 - s / steps for s in range(steps)]
    temb = [create_sinusoidal_pos_embedding(torch.tensor([t], dtype=torch.float32), m.vlm_with_expert.expert_hidden_size,
                                            cfg.min_period, cfg.max_period, device=torch.device("cpu")).float().numpy()[0]
            for t in times]
    # pixel shuffle as an index map: output row o of an image, 768-chunk c <- patch tab[o][c]
    idx = torch.arange(1024, dtype=torch.float32)[None, :, None].expand(1, 1024, 768).contiguous()
    shuf = vlm.connector.pixel_shuffle(idx, vlm.connector.scale_factor)[0].numpy()        # [64][12288]
    tab = shuf.reshape(64, 16, 768)[:, :, 0].astype(np.int64)
    assert (shuf.reshape(64, 16, 768) == tab[:, :, None]).all(), "pixel shuffle keeps 768-chunks"
    return dict(
        sd=sd, P=P, images=np.stack([i[0].numpy() for i in images]), state=state[0].numpy(), noise=noise[0].numpy(),
        lang_ids=lang[0, :nvalid].numpy(), actions=actions[0].numpy(), temb=temb, dt=-1.0 / steps, shuffle=tab,
        eps_v=vlm.vision_model.config.layer_norm_eps, eps_t=vlm.text_model.config.rms_norm_eps,
        eps_e=ex.config.rms_norm_eps, n_vision=len(vlm.vision_model.encoder.layers),
        n_text=len(vlm.text_model.layers), n_expert=len(ex.layers), self_every=m.vlm_with_expert.self_attn_every_n_layers)


# ================================ the numpy pipeline ================================
class Ops:
    """q = False: fp32 (the lerobot reference); q = True: the device's quantization points."""

    def __init__(self, q):
        self.q = q
        self.wcache = {}

    def mx(self, x):
        """MX along the last axis; a width that is not a multiple of 32 is zero-padded as on the device (the 720-wide
        expert rows are 768 there: the group across column 720 includes the zeros)"""
        if not self.q:
            return np.asarray(x, F32)
        x = np.asarray(x, F32)
        K = x.shape[-1]
        if K % 32:
            x = np.concatenate([x, np.zeros(x.shape[:-1] + (32 - K % 32,), F32)], axis=-1)
        return mx_dequant(*mx_quant(x))[..., :K]

    def bf(self, x):
        return bf(x) if self.q else np.asarray(x, F32)

    def gemm(self, x, w, key=None):
        """x [M][K] @ W[out][in]^T, MX operands, bf16 out.  key: the weight's name, so it is quantized once"""
        if key is None:
            return self.bf(self.mx(x) @ self.mx(w).T)
        if key not in self.wcache:
            self.wcache[key] = self.mx(w).T
        return self.bf(self.mx(x) @ self.wcache[key])

    def gelu(self, x):
        return g.gelu_dev(x) if self.q else F32(0.5) * x * (1 + np.tanh(F32(0.7978845608) * (x + F32(0.044715) * x ** 3)))

    def silu(self, x):
        return g.silu_dev(x) if self.q else (x / (1 + np.exp(-x.astype(np.float64)))).astype(F32)

    def exp(self, x):                                      # the device's fexp on bf16(x), widened to fp32
        return bf(np.exp(bf(x))) if self.q else np.exp(x)


def rope(x, pos, wavelength=10000.0):
    """lerobot apply_rope: x [L][H][hd], positions [L] (fp32)"""
    half = x.shape[-1] // 2
    ts = F32(wavelength) ** ((F32(2.0) / x.shape[-1]) * np.arange(half, dtype=F32))
    rad = (np.asarray(pos, F32)[:, None] / ts[None, :])[:, None, :]
    c, s = np.cos(rad).astype(F32), np.sin(rad).astype(F32)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(F32)


def rope_tables(pos, hd, wavelength=10000.0):
    """cos / sin [L][hd/2] fp32, as rope() uses them"""
    half = hd // 2
    ts = F32(wavelength) ** ((F32(2.0) / hd) * np.arange(half, dtype=F32))
    rad = np.asarray(pos, F32)[:, None] / ts[None, :]
    return np.cos(rad).astype(F32), np.sin(rad).astype(F32)


def rms(o, h, gamma, dt, eps):
    ss = np.sum(h.astype(F32) ** 2, axis=-1, dtype=F32)
    inv = (F32(1) / np.sqrt(ss / F32(dt) + F32(eps))).astype(F32)
    return (h * inv[:, None]).astype(F32) * o.bf(gamma)[None, :]


def lnorm(o, h, gamma, beta, eps):
    mean = (np.sum(h, axis=-1, dtype=F32) / F32(h.shape[-1])).astype(F32)
    c = (h - mean[:, None]).astype(F32)
    var = (np.sum(c * c, axis=-1, dtype=F32) / F32(h.shape[-1])).astype(F32)
    inv = (F32(1) / np.sqrt(var + F32(eps))).astype(F32)
    return (c * inv[:, None]).astype(F32) * o.bf(gamma)[None, :] + o.bf(beta)[None, :]


def attention(o, Q, K, V, mask, scale, fa=False):
    """Q [Lq][H][hd], K / V [Lk][Hkv][hd], mask [Lq][Lk] bool -> O [Lq][H * hd].  Device paths: fa = ATTN phase (exact
    max, unnormalized P, O / l); else qk GEMM / SOFTMAX (normalized P) / pv GEMM."""
    Lq, H, hd = Q.shape
    Hkv = K.shape[1]
    Gq = H // Hkv
    Qd = o.mx(Q)                                           # groups along hd
    Kd = o.mx(K)
    Vd = np.swapaxes(o.mx(np.swapaxes(V, 0, 2)), 0, 2)     # groups along keys (per head, per d)
    out = np.zeros((Lq, H, hd), F32)
    for h in range(H):
        gk = h // Gq
        S = o.bf(Qd[:, h, :] @ Kd[:, gk, :].T)
        if not o.q:
            z = np.where(mask, S * F32(scale), -np.inf)
            p = np.exp(z - z.max(axis=1, keepdims=True))
            out[:, h] = (p / p.sum(axis=1, keepdims=True)) @ V[:, gk, :]
            continue
        if fa:                                             # upstream lazy softmax: the max of key block 0
            s = bf(S * bf(F32(scale)))
            m = s[:, :128].max(axis=1, keepdims=True)
            e = np.where(mask, bf(np.exp(bf(s - m))), 0).astype(F32)
            lsum = np.sum(e, axis=1, dtype=F32)
            PV = bf(o.mx(e) @ Vd[:, gk, :])
            out[:, h] = bf(PV * (F32(1) / lsum)[:, None])
            if h == 0 and "S" not in FATRACE:              # the first ATTN head (vision image 0, layer 0): --fa-dump
                FATRACE.update(S=np.asarray(S, F32), P=e, m=m[:, 0], l=lsum, PV=PV)
        else:
            z = (S * F32(scale)).astype(F32)
            m = np.where(mask, z, -np.inf).max(axis=1, keepdims=True)
            e = np.where(mask, bf(np.exp(bf(z - m))), 0).astype(F32)
            p = e * (F32(1) / np.sum(e, axis=1, keepdims=True, dtype=F32))
            out[:, h] = bf(o.mx(p) @ Vd[:, gk, :])
    return out.reshape(Lq, H * hd)


FA_DUMP = False
FATRACE = {}  # quantized pass: the first ATTN head's S / P / m / l / unnormalized O (gen_vla.py --fa-dump)
VTRACE = {}   # quantized pass: vision layer 0's device buffers per image (the host's per-stage checks)


def vision(o, R, img):
    """one image [3][512][512] -> post-LayerNorm features [1024][768]"""
    tr = (lambda k, v: VTRACE.setdefault(k, []).append(np.asarray(v, F32))) if o.q else (lambda k, v: None)
    sd, P = R["sd"], R["P"] + "vlm.model.vision_model."
    w = lambda n: sd[P + n]
    patches = img.reshape(3, 32, 16, 32, 16).transpose(1, 3, 0, 2, 4).reshape(1024, 768)
    pc = o.gemm(patches, w("embeddings.patch_embedding.weight").reshape(768, 768), "v.patch")
    h = o.bf(pc + o.bf(w("embeddings.patch_embedding.bias")[None, :] + w("embeddings.position_embedding.weight")))
    tr("patch", pc); tr("h0", h)
    mask = np.ones((1024, 1024), bool)
    add, badd = np.zeros_like(h), np.zeros(768, F32)
    for l in range(R["n_vision"]):
        p = f"encoder.layers.{l}."
        h = o.bf(h + (add + o.bf(badd)))
        x = lnorm(o, h, w(p + "layer_norm1.weight"), w(p + "layer_norm1.bias"), R["eps_v"])
        qkv = o.gemm(x, np.concatenate([w(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0), f"v{l}.qkv")
        if l == 0: tr("qkv", qkv)
        qkv = o.bf(qkv + o.bf(np.concatenate([w(p + f"self_attn.{n}_proj.bias") for n in "qkv"]))[None, :])
        Q, K, V = (qkv[:, i * 768:(i + 1) * 768].reshape(1024, 12, 64) for i in range(3))
        if l == 0: tr("Q", Q); tr("K", K); tr("V", V)
        att = attention(o, Q, K, V, mask, 1 / 8, fa=True)
        if l == 0: tr("att", att)
        y = o.gemm(att, w(p + "self_attn.out_proj.weight"), f"v{l}.o")
        hmid = o.bf(h + (y + o.bf(w(p + "self_attn.out_proj.bias"))))
        if l == 0: tr("y", y); tr("hmid", hmid)
        x2 = lnorm(o, hmid, w(p + "layer_norm2.weight"), w(p + "layer_norm2.bias"), R["eps_v"])
        f1 = o.gemm(x2, w(p + "mlp.fc1.weight"), f"v{l}.fc1")
        a = o.gelu((f1 + o.bf(w(p + "mlp.fc1.bias"))).astype(F32))
        h, add, badd = hmid, o.gemm(a, w(p + "mlp.fc2.weight"), f"v{l}.fc2"), w(p + "mlp.fc2.bias")
        if l == 0: tr("f1", f1); tr("f2", add)
    h = o.bf(h + (add + o.bf(badd)))
    return o.bf(lnorm(o, h, w("post_layernorm.weight"), w("post_layernorm.bias"), R["eps_v"]))


def forward(R, q):
    """the whole action chunk; returns intermediates (device row order) and the actions"""
    o = Ops(q)
    sd, P = R["sd"], R["P"]
    out = {}
    # ---- prefix embeddings, device row order: 192 image tokens, the instruction, the state, padding ----
    feats = np.stack([vision(o, R, R["images"][i]) for i in range(3)])                  # [3][1024][768]
    out["vision"] = feats.reshape(3072, 768)
    nl = len(R["lang_ids"])
    Lv = 192 + nl + 1                                       # valid prefix tokens (205)
    Mp = 256
    X0 = np.zeros((Mp, 960), F32)
    wc = sd[P + "vlm.model.connector.modality_projection.proj.weight"]                  # [960][12288]
    for i in range(3):
        gathered = feats[i][R["shuffle"]].reshape(64, 12288)
        X0[64 * i:64 * (i + 1)] = o.bf(o.gemm(gathered, wc, "conn") * F32(math.sqrt(960)))
    emb = sd[P + "vlm.model.text_model.embed_tokens.weight"]
    X0[192:192 + nl] = o.bf(emb[R["lang_ids"]] * F32(math.sqrt(960)))
    sp = o.gemm(R["state"][None, :], sd["model.state_proj.weight"], "state")
    X0[192 + nl] = o.bf(sp[0] + o.bf(sd["model.state_proj.bias"]))
    out["x0"] = X0
    pos = np.concatenate([np.arange(Lv), np.zeros(Mp - Lv)]).astype(F32)
    mask = np.zeros((Mp, Mp), bool)
    mask[:, :Lv - 1] = True                                 # image + instruction keys
    mask[Lv - 1, Lv - 1] = True                             # only the state row sees the state key
    for r in range(Lv, Mp):                                 # padding rows: as the state row (never used)
        mask[r, :Lv] = True
    # ---- the 16 SmolLM2 layers over the prefix, K / V cached (post-RoPE) ----
    T = P + "vlm.model.text_model.layers."
    h, add = X0, np.zeros_like(X0)
    caches = []
    for l in range(R["n_text"]):
        p = f"{T}{l}."
        w = lambda n: sd[p + n]
        h = o.bf(h + add)
        x = rms(o, h, w("input_layernorm.weight"), 960, R["eps_t"])
        qkv = o.gemm(x, np.concatenate([w(f"self_attn.{n}_proj.weight") for n in "qkv"], 0), f"t{l}.qkv")
        Q = rope(qkv[:, :960].reshape(Mp, 15, 64), pos)
        K = rope(qkv[:, 960:1280].reshape(Mp, 5, 64), pos)
        V = qkv[:, 1280:].reshape(Mp, 5, 64)
        caches.append((K, V))
        att = attention(o, Q, K, V, mask, 1 / 8)
        hmid = o.bf(h + o.gemm(att, w("self_attn.o_proj.weight"), f"t{l}.o"))
        x2 = rms(o, hmid, w("post_attention_layernorm.weight"), 960, R["eps_t"])
        gu = o.gemm(x2, np.concatenate([w("mlp.gate_proj.weight"), w("mlp.up_proj.weight")], 0), f"t{l}.gu")
        hh = (o.silu(gu[:, :2560]) * gu[:, 2560:]).astype(F32)
        h, add = hmid, o.gemm(hh, w("mlp.down_proj.weight"), f"t{l}.d")
    out["prefix_h"] = o.bf(h + add)
    # ---- expert: cross-attention K' / V' once, then the 10 Euler steps ----
    E = P + "lm_expert.layers."
    cross = {}
    for l in range(R["n_expert"]):
        if l % R["self_every"] != 0:
            K, V = caches[l]
            Kp = o.gemm(K.reshape(Mp, 320), sd[f"{E}{l}.self_attn.k_proj.weight"], f"c{l}.k").reshape(Mp, 5, 64)
            Vp = o.gemm(V.reshape(Mp, 320), sd[f"{E}{l}.self_attn.v_proj.weight"], f"c{l}.v").reshape(Mp, 5, 64)
            cross[l] = (Kp, Vp)
    Ms, NA = 64, 50
    w_ai, b_ai = sd["model.action_in_proj.weight"], sd["model.action_in_proj.bias"]           # [720][32]
    w_in, b_in = sd["model.action_time_mlp_in.weight"], sd["model.action_time_mlp_in.bias"]   # [720][1440]
    w_out, b_out = sd["model.action_time_mlp_out.weight"], sd["model.action_time_mlp_out.bias"]
    w_fold = (w_in[:, :720].astype(np.float64) @ w_ai.astype(np.float64)).astype(F32)       # [720][32]
    c_base = (w_in[:, :720].astype(np.float64) @ b_ai.astype(np.float64) + b_in).astype(F32)
    spos = np.concatenate([Lv + np.arange(NA), np.zeros(Ms - NA)]).astype(F32)
    cpos = np.concatenate([np.arange(NA), np.zeros(Ms - NA)]).astype(F32)
    smask = np.zeros((Ms, 256 + Ms), bool)                  # [prefix keys 0..255 | suffix keys 256..]
    smask[:, :Lv] = True
    for t in range(Ms):
        smask[t, 256:256 + min(t, NA - 1) + 1] = True
    cmask = np.zeros((Ms, Mp), bool)
    cmask[:, :Lv] = True
    x = np.zeros((Ms, 32), F32)
    x[:NA] = R["noise"]
    out["x"] = [x.copy()]
    for s in range(len(R["temb"])):
        c_s = (c_base + (w_in[:, 720:].astype(np.float64) @ R["temb"][s].astype(np.float64))).astype(F32)
        t1 = o.gemm(x, w_fold, "fold")
        e = o.gemm(o.silu((t1 + o.bf(c_s)[None, :]).astype(F32)), w_out, "mlp_out")
        h, add = np.zeros((Ms, 720), F32), (e + o.bf(b_out)[None, :]).astype(F32)
        for l in range(R["n_expert"]):
            p = f"{E}{l}."
            w = lambda n: sd[p + n]
            h = o.bf(h + add)
            xn = rms(o, h, w("input_layernorm.weight"), 720, R["eps_e"])
            if l % R["self_every"] == 0:
                qkv = o.gemm(xn, np.concatenate([w(f"self_attn.{n}_proj.weight") for n in "qkv"], 0), f"e{l}.qkv")
                Q = rope(qkv[:, :960].reshape(Ms, 15, 64), spos)
                Ks = rope(qkv[:, 960:1280].reshape(Ms, 5, 64), spos)
                Vs = qkv[:, 1280:].reshape(Ms, 5, 64)
                K0, V0 = caches[l]
                att = attention(o, Q, np.concatenate([K0, Ks]), np.concatenate([V0, Vs]), smask, 1 / 8)
            else:
                qq = o.gemm(xn, w("self_attn.q_proj.weight"), f"e{l}.q")
                Q = rope(qq.reshape(Ms, 15, 64), cpos)
                att = attention(o, Q, cross[l][0], cross[l][1], cmask, 1 / 8)
            hmid = o.bf(h + o.gemm(att, w("self_attn.o_proj.weight"), f"e{l}.o"))
            x2 = rms(o, hmid, w("post_attention_layernorm.weight"), 720, R["eps_e"])
            gu = o.gemm(x2, np.concatenate([w("mlp.gate_proj.weight"), w("mlp.up_proj.weight")], 0), f"e{l}.gu")
            hh = (o.silu(gu[:, :2048]) * gu[:, 2048:]).astype(F32)
            h, add = hmid, o.gemm(hh, w("mlp.down_proj.weight"), f"e{l}.d")
        hf = rms(o, o.bf(h + add), sd[P + "lm_expert.norm.weight"], 720, R["eps_e"])
        v = o.gemm(hf, sd["model.action_out_proj.weight"], "act_out")
        x = x.copy()
        x[:NA] = (x[:NA] + F32(R["dt"]) * (v[:NA] + sd["model.action_out_proj.bias"][None, :])).astype(F32)
        out["x"].append(x.copy())
    out["actions"] = x[:NA]
    return out


# ================================ device program ================================
OPS = ["init", "vpatch", "vaffine", "vnorm1", "vqkv", "vrope", "vattn", "vo", "vnorm2", "vfc1", "vgelu", "vfc2", "vpost",
       "shuffle", "conn", "scale", "state", "state_b",
       "pnorm1", "pqkv", "prope", "pvsrc", "pqk", "psoftmax", "ppv", "pgather", "po", "pnorm2", "pgu", "psilu", "pdown",
       "pfinal", "xk", "xv", "xrope",
       "euler", "tin", "tsilu", "tout", "enorm1", "eqkv", "erope", "eqk", "esoftmax", "epv", "egather", "eo", "enorm2",
       "egu", "esilu", "edown", "efinal", "aout", "final", "lmhead", "softcap"]
OP = {n: i for i, n in enumerate(OPS)}
T_NONE, T_GEMM, T_PRENORM, T_ROPE, T_SOFTMAX, T_EWQ, T_FILL, T_SILU, T_ACT, T_FA = range(10)
T_AFFINE, T_GATHERQ, T_EULER = 10, 11, 12
PW, TN, TK = 32, 64, 256
NOROW = 0xFFFFFFFF                                          # GATHERQ zero chunk


class Prog:
    def __init__(self):
        self.ph = []

    def add(self, typ, layer, op, *words):
        d = [typ, ((layer & 0xFF) << 8) | OP[op]] + [int(w) & 0xFFFFFFFF for w in words]
        assert len(d) <= PW, (op, len(d))
        self.ph.append(d + [0] * (PW - len(d)))

    def gemm(self, layer, op, a, asc, a_bs, asc_bs, b, bsc, b_bs, bsc_bs, c, c_bs, c_rs, c_ms, batch, MT, NT, KS, TM, TKg):
        assert (TM // 16) * 4 * (TKg // 16) >= 64, "compute loop too short for the scale-load ordering"
        self.add(T_GEMM, layer, op, a, asc, a_bs, asc_bs, b, bsc, b_bs, bsc_bs, c, c_bs, c_rs, c_ms, batch, MT, NT, KS, TM, TKg)

    def dense(self, layer, op, A, W, c, c_rs, M, N, K, TM):
        """C [M][N] (row stride c_rs bytes) = A (MX, TM-row tiles, K / 256 k-steps) @ W (pretiled [K][N])"""
        assert M % TM == 0 and N % TN == 0 and K % TK == 0, (op, M, N, K)
        self.gemm(layer, op, A[0], A[1], 0, 0, W[0], W[1], 0, 0, c, 0, c_rs, TM * c_rs, 1, M // TM, N // TN, K // TK, TM, TK)

    def array(self):
        return np.array(self.ph, dtype=np.uint32).reshape(-1)


def pad2(w, n, k):
    out = np.zeros((n, k), F32)
    out[:w.shape[0], :w.shape[1]] = w
    return out


def build(R, emu, ref, outdir, header):
    sd, P = R["sd"], R["P"]
    V, T, E = P + "vlm.model.vision_model.", P + "vlm.model.text_model.layers.", P + "lm_expert.layers."
    mem = g.Mem("llama")
    prog = Prog()
    nl = len(R["lang_ids"])
    Lv = 192 + nl + 1
    Mv, Mp, Ms, NA = 3072, 256, 64, 50
    Dp, De = 1024, 768                                      # 960 / 720 padded
    SKe = 512                                               # prefix + suffix key cache
    eps_v, eps_t, eps_e = (f32_bits(R[k]) for k in ("eps_v", "eps_t", "eps_e"))
    s18 = f32_bits(1 / 8)

    def wq(w, n, k):                                        # weight [out][in] -> pretiled B, zero-padded to [n][k]
        codes, sc = mx_quant(pad2(np.asarray(w, F32), n, k))
        bt, bs = g.pack_b(np.ascontiguousarray(codes.T), np.ascontiguousarray(sc.T), TK)
        return mem.add_any(bt), mem.add_any(bs)

    def vec(x, n, f32=False):                               # a padded row vector (bf16, or f32)
        v = np.zeros(n, F32)
        v[:len(x)] = x
        return mem.add(3, v if f32 else bf16_bits(v))

    def cs(pos, hd=64, identity=False):                     # ROPE cos / sin tables [rows][hd/2]
        c, s_ = rope_tables(pos, hd)
        if identity:
            c, s_ = np.ones_like(c), np.zeros_like(s_)
        return mem.add(3, c), mem.add(3, s_)

    def mask_tab(lims):                                     # u32 [M][2] = lim1, lim2
        return mem.add(3, np.array(lims, np.uint32).reshape(-1))

    prog_addr = mem.add(3, np.zeros(PW * 3072, np.uint32))  # patched once laid out

    # ---------------- weights ----------------
    w = lambda n: sd[n]
    vw = []
    W_patch = wq(sd[V + "embeddings.patch_embedding.weight"].reshape(768, 768), 768, 768)
    tab_pos = mem.add(3, bf16_bits(sd[V + "embeddings.patch_embedding.bias"][None, :] +
                                   sd[V + "embeddings.position_embedding.weight"]))
    for l in range(R["n_vision"]):
        p = f"{V}encoder.layers.{l}."
        vw.append(dict(
            ln1g=vec(w(p + "layer_norm1.weight"), 768), ln1b=vec(w(p + "layer_norm1.bias"), 768),
            ln2g=vec(w(p + "layer_norm2.weight"), 768), ln2b=vec(w(p + "layer_norm2.bias"), 768),
            qkv=wq(np.concatenate([w(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0), 2304, 768),
            bqkv=vec(np.concatenate([w(p + f"self_attn.{n}_proj.bias") for n in "qkv"]), 2304),
            o=wq(w(p + "self_attn.out_proj.weight"), 768, 768), bo=vec(w(p + "self_attn.out_proj.bias"), 768),
            fc1=wq(w(p + "mlp.fc1.weight"), 3072, 768), b1=vec(w(p + "mlp.fc1.bias"), 3072),
            fc2=wq(w(p + "mlp.fc2.weight"), 768, 3072), b2=vec(w(p + "mlp.fc2.bias"), 768)))
    postg, postb = vec(sd[V + "post_layernorm.weight"], 768), vec(sd[V + "post_layernorm.bias"], 768)
    W_conn = wq(sd[P + "vlm.model.connector.modality_projection.proj.weight"], Dp, 12288)
    W_state, b_state = wq(sd["model.state_proj.weight"], Dp, 256), vec(sd["model.state_proj.bias"], Dp)
    tw = []
    for l in range(R["n_text"]):
        p = f"{T}{l}."
        tw.append(dict(
            g1=vec(w(p + "input_layernorm.weight"), Dp), g2=vec(w(p + "post_attention_layernorm.weight"), Dp),
            qkv=wq(np.concatenate([w(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0), 1600, Dp),
            o=wq(w(p + "self_attn.o_proj.weight"), Dp, Dp),
            gu=wq(np.concatenate([w(p + "mlp.gate_proj.weight"), w(p + "mlp.up_proj.weight")], 0), 5120, Dp),
            d=wq(w(p + "mlp.down_proj.weight"), Dp, 2560)))
    g_tfinal = vec(sd[P + "vlm.model.text_model.norm.weight"], Dp)
    ew = []
    for l in range(R["n_expert"]):
        p = f"{E}{l}."
        selfl = l % R["self_every"] == 0
        d = dict(self=selfl, g1=vec(w(p + "input_layernorm.weight"), De),
                 g2=vec(w(p + "post_attention_layernorm.weight"), De),
                 o=wq(w(p + "self_attn.o_proj.weight"), De, Dp),
                 gu=wq(np.concatenate([w(p + "mlp.gate_proj.weight"), w(p + "mlp.up_proj.weight")], 0), 4096, De),
                 d=wq(w(p + "mlp.down_proj.weight"), De, 2048))
        if selfl:
            d["qkv"] = wq(np.concatenate([w(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0), 1600, De)
        else:
            d["q"] = wq(w(p + "self_attn.q_proj.weight"), 960, De)
            d["k"] = wq(w(p + "self_attn.k_proj.weight"), 320, 512)
            d["v"] = wq(w(p + "self_attn.v_proj.weight"), 320, 512)
        ew.append(d)
    g_efinal = vec(sd[P + "lm_expert.norm.weight"], De)
    w_ai, b_ai = sd["model.action_in_proj.weight"], sd["model.action_in_proj.bias"]
    w_in, b_in = sd["model.action_time_mlp_in.weight"], sd["model.action_time_mlp_in.bias"]
    W_fold = wq((w_in[:, :720].astype(np.float64) @ w_ai.astype(np.float64)).astype(F32), De, 256)
    c_base = (w_in[:, :720].astype(np.float64) @ b_ai.astype(np.float64) + b_in).astype(F32)
    c_steps = [vec((c_base + w_in[:, 720:].astype(np.float64) @ t.astype(np.float64)).astype(F32), De) for t in R["temb"]]
    W_tout, b_tout = wq(sd["model.action_time_mlp_out.weight"], De, De), vec(sd["model.action_time_mlp_out.bias"], De)
    W_aout, b_aout = wq(sd["model.action_out_proj.weight"], 64, De), vec(sd["model.action_out_proj.bias"], 32, f32=True)

    # ---------------- inputs, tables, references ----------------
    patches = np.concatenate([R["images"][i].reshape(3, 32, 16, 32, 16).transpose(1, 3, 0, 2, 4).reshape(1024, 768)
                              for i in range(3)])
    pc, ps = mx_quant(patches)
    A_patch = tuple(mem.add_any(x) for x in g.pack_a(pc, ps, 128, TK))
    X0 = np.zeros((Mp, Dp), F32)
    X0[192:192 + nl, :960] = bf(sd[P + "vlm.model.text_model.embed_tokens.weight"][R["lang_ids"]] * F32(math.sqrt(960)))
    x0_addr = mem.add(3, bf16_bits(X0))
    st = np.zeros((Ms, 256), F32)
    st[0, :32] = R["state"]
    sc_, ss_ = mx_quant(st)
    A_state = tuple(mem.add(3, x) for x in g.pack_a(sc_, ss_, 64, TK))
    noise = np.zeros((Ms, 32), F32)
    noise[:NA] = R["noise"]
    x_bufs = [mem.add(3, noise)]
    zeros_e = mem.add(3, np.zeros((Ms, De), np.uint16))
    gtab = np.concatenate([i * 1024 + R["shuffle"] for i in range(3)])                  # [192][16]
    shuffle_tab = mem.add(3, gtab.astype(np.uint32).reshape(-1))

    def head_tab(M):                                        # o_proj operand rows t <- head-major O rows h M + t
        t_ = np.full((M, 16), NOROW, np.uint32)
        for h in range(15):
            t_[:, h] = h * M + np.arange(M)
        return mem.add(3, t_.reshape(-1))
    htab_p, htab_e = head_tab(Mp), head_tab(Ms)
    vcos = cs(np.zeros(1024), identity=True)
    ppos = np.concatenate([np.arange(Lv), np.zeros(Mp - Lv)])
    pcos = cs(ppos)
    scos = cs(np.concatenate([Lv + np.arange(NA), np.zeros(Ms - NA)]))
    ccos = cs(np.concatenate([np.arange(NA), np.zeros(Ms - NA)]))
    xcos = cs(np.zeros(Mp), identity=True)
    ptab = mask_tab([[Lv - 2 if t < Lv - 1 else Lv - 1, 0] for t in range(Mp)])
    stab = mask_tab([[Lv - 1, 256 + min(t, NA - 1)] for t in range(Ms)])
    ctab = mask_tab([[Lv - 1, 0] for t in range(Ms)])
    gold = {}
    for name, val in (("vision", emu["vision"]), ("prefix", np.pad(emu["prefix_h"][:Lv], ((0, 0), (0, 64)))),
                      ("x1", emu["x"][1]), ("x10", emu["x"][-1]), ("ref_x10", ref["x"][-1]),
                      ("ref_vision", ref["vision"]), ("ref_prefix", np.pad(ref["prefix_h"][:Lv], ((0, 0), (0, 64))))):
        gold[name] = mem.add(3, bf16_bits(val))
    fdg = {}
    if FA_DUMP and FATRACE:   # tile 0 (rows 0..63) of the first vision ATTN phase's head 0
        NR, f0 = 64, FATRACE
        blk = lambda x: np.ascontiguousarray(np.asarray(x[:NR], F32).reshape(NR, 8, 128).transpose(1, 0, 2))
        fdg = dict(S=mem.add(3, bf16_bits(blk(f0["S"]))), P=mem.add(3, bf16_bits(blk(f0["P"]))),
                   m=mem.add(3, np.asarray(f0["m"][:NR], F32).view(np.uint32) & np.uint32(0xFFFF0000)),
                   l=mem.add(3, np.asarray(f0["l"][:NR], F32).view(np.uint32) & np.uint32(0xFFFF0000)),
                   PV=mem.add(3, bf16_bits(f0["PV"][:NR])))
    for k, v in VTRACE.items():                             # 3 images stacked, as the device's rows
        if k not in ("Q", "K", "V", "att"):
            gold["vt_" + k] = mem.add(3, bf16_bits(np.concatenate(v, 0)))
    # the vision ROPE's FA operands, all 36 heads (image-major): Q codes [h][1024][64], scales [h][16][2][64];
    # K^T codes [h][64][1024], scales [h][8][2][128]; V codes [h][1024][64], scales [h][8][4][64]
    vrg = {}
    if "att" in VTRACE:   # the vision ATTN output as o_proj's MX A operand (TM 128, TK 256): O_v
        oc, osc = mx_quant(np.concatenate(VTRACE["att"], 0).astype(np.float32))
        oa, oas = g.pack_a(oc, osc, 128, TK)
        vrg["O"], vrg["O_sc"] = (mem.add(3, oa), oa.size), (mem.add(3, oas), oas.size)
    if "Q" in VTRACE:
        Qa = np.concatenate([q.transpose(1, 0, 2) for q in VTRACE["Q"]], 0)          # [36][1024][64]
        Ka = np.concatenate([k.transpose(1, 0, 2) for k in VTRACE["K"]], 0)
        Va = np.concatenate([v.transpose(1, 0, 2) for v in VTRACE["V"]], 0)
        qc, qs = mx_quant(Qa)
        kc_, ks_ = mx_quant(Ka)
        vc_, vs_ = mx_quant(Va.transpose(0, 2, 1))                                     # [36][64][1024]
        for k, v in (("Q", qc), ("Q_sc", qs.reshape(36, 16, 64, 2).transpose(0, 1, 3, 2)),
                     ("K", kc_.transpose(0, 2, 1)), ("K_sc", ks_.reshape(36, 8, 128, 2).transpose(0, 1, 3, 2)),
                     ("V", vc_.transpose(0, 2, 1)), ("V_sc", vs_.reshape(36, 64, 8, 4).transpose(0, 2, 3, 1))):
            v = np.ascontiguousarray(v)
            vrg[k] = (mem.add(3, v), v.size)

    # ---------------- arena ----------------
    al = mem.alloc

    def amx(M, K):
        return al(M * K), al(M * K // 32)
    fills = []

    def zfill(addr, nbytes):
        fills.append((addr, nbytes))
        return addr
    vbuf = lambda: al(Mv * 768 * 2)
    hv = [vbuf()]                                           # vision h per layer (h0 = patch embedding)
    v_patch_c = vbuf()
    A_v = amx(Mv, 768)
    Q_v, K_v, V_v = amx(36 * 1024, 64), amx(36 * 1024, 64), amx(36 * 1024, 64)
    O_v = amx(Mv, 768)
    A_fc2 = amx(Mv, 3072)
    mrow = al(4 * 576 * 64 * 4)
    fa_dump = al(262656 + 64 * 64 * 2 + 64 * 4) if fdg else 0   # + 1 / l per row
    vfeat = vbuf()
    A_conn = amx(192, 12288)
    st_c = al(Ms * Dp * 2)
    A_p, A_pf = amx(Mp, Dp), amx(Mp, 2560)
    Qp = amx(5 * 768, 64)
    Pp = amx(5 * 768, 256)
    A_po = amx(Mp, Dp)
    kc = [(zfill(al(5 * SKe * 64), 5 * SKe * 64), zfill(al(5 * SKe * 2), 5 * SKe * 2)) for _ in range(R["n_text"])]
    vc = [(zfill(al(5 * SKe * 64), 5 * SKe * 64), zfill(al(5 * SKe * 2), 5 * SKe * 2)) for _ in range(R["n_text"])]
    ksrc, vsrc_ = {}, {}
    for l in range(R["n_text"]):
        if not ew[l]["self"]:
            ksrc[l] = tuple(zfill(al(n), n) for n in (Mp * 512, Mp * 512 // 32))
            vsrc_[l] = tuple(zfill(al(n), n) for n in (Mp * 512, Mp * 512 // 32))
    xkc = {l: (al(5 * Mp * 64), al(5 * Mp * 2)) for l in ksrc}
    xvc = {l: (al(5 * Mp * 64), al(5 * Mp * 2)) for l in ksrc}
    A_x = tuple(zfill(al(n), n) for n in (Ms * 256, Ms * 256 // 32))
    A_e, A_t, A_ef = amx(Ms, De), amx(Ms, De), amx(Ms, 2048)
    Qe = amx(5 * 192, 64)
    Pe = amx(5 * 192, 512)
    A_eo = amx(Ms, Dp)
    perf = None

    # ---------------- init (FILL, before timing) ----------------
    for a_, nb in fills:
        prog.add(T_FILL, 255, "init", a_, nb, 1, 0)
    nfill = len(prog.ph)
    perf = al(64 * 4 * 3072)

    # ---------------- vision ----------------
    prog.dense(255, "vpatch", A_patch, W_patch, v_patch_c, 768 * 2, Mv, 768, 768, 128)
    prog.add(T_AFFINE, 255, "vaffine", v_patch_c, 1536, hv[0], 1536, Mv, 768, f32_bits(1.0), 0, tab_pos, 1024)
    add_prev, bias_prev = None, 0
    v0 = dict(patch=(v_patch_c, 768), h0=(hv[0], 768))      # vision layer 0's buffers (per-stage checks)
    for l, d in enumerate(vw):
        h_in = hv[-1]
        hn = vbuf()
        prog.add(T_PRENORM, l, "vnorm1", h_in, add_prev if add_prev else h_in, hn, d["ln1g"], A_v[0], A_v[1], 768, Mv,
                 128, TK, eps_v, 0, 0, 0, d["ln1b"], bias_prev, 0)
        if add_prev is None:                                # layer 0: no branch to add (x + x would double it)
            prog.ph[-1][3] = zeros_v_addr = al(Mv * 768 * 2)
            fills.append((zeros_v_addr, Mv * 768 * 2))
        qkv = al(Mv * 2304 * 2)
        prog.dense(l, "vqkv", A_v, d["qkv"], qkv, 2304 * 2, Mv, 2304, 768, 128)
        for i in range(3):                                  # q / k / v bias, identity rotation, ATTN layouts
            hb = i * 12
            prog.add(T_ROPE, l, "vrope", qkv + i * 1024 * 2304 * 2, 2304, 12, 12, 64, vcos[0], vcos[1],
                     Q_v[0] + hb * 65536, Q_v[1] + hb * 2048, 65536, 2048, K_v[0] + hb * 65536, K_v[1] + hb * 2048,
                     65536, 2048, V_v[0] + hb * 65536, V_v[1] + hb * 2048, 65536, 2048, 0, 1024, 1024, 64, 1, 0,
                     d["bqkv"], 1, 1, 0, 0)                     # d[28] = 1: no rotation (rope_qk_norot)
        # 9 ATTN phases of 4 heads (64 query tiles: the softmax lanes keep their row maxima on the stack, not in DRAM
        # as one 36-head phase did, whose output broke on FireSim); head hb writes O rows (hb / 12) 1024 + t, columns
        # (hb % 12) 64 + d (d[28], d[29]).  d[27] = 0: block-0 max.
        for hb in range(0, 36, 4):
            prog.add(T_FA, l, "vattn", Q_v[0] + hb * 65536, Q_v[1] + hb * 2048, K_v[0] + hb * 65536,
                     K_v[1] + hb * 2048, V_v[0] + hb * 65536, V_v[1] + hb * 2048, O_v[0], 4, 1, 1024, 1024, 1024, 0,
                     s18, 65536, 2048, 65536, 2048, O_v[1], 128, TK, 3, 64, 0, 4, 0, (hb // 12) * 1024, (hb % 12) * 64,
                     fa_dump if (l == 0 and hb == 0) else 0)    # d[30]: --fa-dump
        y = vbuf()
        prog.dense(l, "vo", O_v, d["o"], y, 1536, Mv, 768, 768, 128)
        hm = vbuf()
        prog.add(T_PRENORM, l, "vnorm2", hn, y, hm, d["ln2g"], A_v[0], A_v[1], 768, Mv, 128, TK, eps_v, 0, 0, 0,
                 d["ln2b"], d["bo"], 0)
        f1 = al(Mv * 3072 * 2)
        prog.dense(l, "vfc1", A_v, d["fc1"], f1, 3072 * 2, Mv, 3072, 768, 128)
        prog.add(T_SILU, l, "vgelu", f1, 3072, 0, 3072, Mv, A_fc2[0], A_fc2[1], 128, TK, 3, d["b1"], 0)
        f2 = vbuf()
        prog.dense(l, "vfc2", A_fc2, d["fc2"], f2, 1536, Mv, 768, 3072, 128)
        hv.append(hm)
        add_prev, bias_prev = f2, d["b2"]
        if l == 0:
            v0.update(qkv=(qkv, 2304), y=(y, 768), hmid=(hm, 768), f1=(f1, 3072), f2=(f2, 768))
    hlast = vbuf()
    prog.add(T_PRENORM, 255, "vpost", hv[-1], add_prev, hlast, postg, vfeat, 0, 768, Mv, 128, TK, eps_v, 0, 0, 0,
             postb, bias_prev, 1)

    # ---------------- connector, state, prefix assembly ----------------
    prog.add(T_GATHERQ, 255, "shuffle", vfeat, 1536, shuffle_tab, 16, 768, 192, A_conn[0], A_conn[1], 64, TK)
    prog.dense(255, "conn", A_conn, W_conn, x0_addr, Dp * 2, 192, Dp, 12288, 64)
    prog.add(T_AFFINE, 255, "scale", x0_addr, Dp * 2, x0_addr, Dp * 2, 192, Dp, f32_bits(math.sqrt(960)), 0, 0, 0)
    prog.dense(255, "state", A_state, W_state, st_c, Dp * 2, Ms, Dp, 256, 64)
    prog.add(T_AFFINE, 255, "state_b", st_c, Dp * 2, x0_addr + (Lv - 1) * Dp * 2, Dp * 2, 1, Dp, f32_bits(1.0),
             b_state, 0, 0)

    # ---------------- prefix: 16 SmolLM2 layers ----------------
    zeros_p = al(Mp * Dp * 2)
    fills.append((zeros_p, Mp * Dp * 2))
    h, add = x0_addr, zeros_p
    pfx = {}
    for l, d in enumerate(tw):
        hn = al(Mp * Dp * 2)
        prog.add(T_PRENORM, l, "pnorm1", h, add, hn, d["g1"], A_p[0], A_p[1], Dp, Mp, 128, TK, eps_t, 0, 0, 960, 0, 0, 0)
        qkv = al(Mp * 1600 * 2)
        prog.dense(l, "pqkv", A_p, d["qkv"], qkv, 3200, Mp, 1600, Dp, 128)
        ka = ksrc.get(l, (0, 0))
        prog.add(T_ROPE, l, "prope", qkv, 1600, 15, 5, 64, pcos[0], pcos[1], Qp[0], Qp[1], 768 * 64, 768 * 2,
                 kc[l][0], kc[l][1], SKe * 64, SKe * 2, vc[l][0], vc[l][1], SKe * 64, SKe * 2, 0, Mp, SKe, 128, 0, 0,
                 0, 0, 3, ka[0], ka[1])
        if l in vsrc_:                                      # raw V of this layer: the expert's cross-attention source
            prog.add(T_EWQ, l, "pvsrc", qkv + 1280 * 2, 1600, 0, 320, Mp, vsrc_[l][0], vsrc_[l][1], 128, TK, 0, 0, 2)
        S = al(5 * 768 * 256 * 2)
        prog.gemm(l, "pqk", Qp[0], Qp[1], 768 * 64, 768 * 2, kc[l][0], kc[l][1], SKe * 64, SKe * 2, S, 768 * 256 * 2,
                  512, 128 * 512, 5, 6, 4, 1, 128, 64)
        prog.add(T_SOFTMAX, l, "psoftmax", S, 768 * 256 * 2, 256, 768, 5, Pp[0], Pp[1], 768 * 256, 768 * 8, s18, 0,
                 Mp, 128, TK, 0, 0, ptab, NOROW)
        O = al(15 * Mp * 64 * 2)
        prog.gemm(l, "ppv", Pp[0], Pp[1], 768 * 256, 768 * 8, vc[l][0], vc[l][1], SKe * 64, SKe * 2, O,
                  768 * 64 * 2, 128, 128 * 128, 5, 6, 1, 1, 128, TK)
        prog.add(T_GATHERQ, l, "pgather", O, 128, htab_p, 16, 64, Mp, A_po[0], A_po[1], 128, TK)
        y = al(Mp * Dp * 2)
        prog.dense(l, "po", A_po, d["o"], y, Dp * 2, Mp, Dp, Dp, 128)
        hm = al(Mp * Dp * 2)
        prog.add(T_PRENORM, l, "pnorm2", hn, y, hm, d["g2"], A_p[0], A_p[1], Dp, Mp, 128, TK, eps_t, 0, 0, 960, 0, 0, 0)
        gu = al(Mp * 5120 * 2)
        prog.dense(l, "pgu", A_p, d["gu"], gu, 5120 * 2, Mp, 5120, Dp, 128)
        prog.add(T_SILU, l, "psilu", gu, 5120, 0, 2560, Mp, A_pf[0], A_pf[1], 128, TK, 0, 0, 0)
        ym = al(Mp * Dp * 2)
        prog.dense(l, "pdown", A_pf, d["d"], ym, Dp * 2, Mp, Dp, 2560, 128)
        h, add = hm, ym
    pfinal = al(Mp * Dp * 2)
    prog.add(T_PRENORM, 255, "pfinal", h, add, pfinal, g_tfinal, A_p[0], A_p[1], Dp, Mp, 128, TK, eps_t, 0, 0, 960,
             0, 0, 0)

    # ---------------- cross-attention K' / V' (once) ----------------
    for l in sorted(ksrc):
        kv = al(Mp * 640 * 2)
        prog.dense(l, "xk", ksrc[l], ew[l]["k"], kv, 1280, Mp, 320, 512, 128)
        prog.dense(l, "xv", vsrc_[l], ew[l]["v"], kv + 640, 1280, Mp, 320, 512, 128)
        prog.add(T_ROPE, l, "xrope", kv, 640, 0, 5, 64, xcos[0], xcos[1], 0, 0, 0, 0, xkc[l][0], xkc[l][1], Mp * 64,
                 Mp * 2, xvc[l][0], xvc[l][1], Mp * 64, Mp * 2, 0, Mp, Mp, 128, 0, 0, 0, 0, 3, 0, 0)

    # ---------------- denoising ----------------
    dt = f32_bits(R["dt"])
    xbf = [al(Ms * 32 * 2) for _ in range(len(R["temb"]) + 1)]
    for s_ in range(len(R["temb"])):
        x_bufs.append(al(Ms * 32 * 4))
    prog.add(T_EULER, 255, "euler", x_bufs[0], 0, 0, f32_bits(0.0), x_bufs[0], A_x[0], A_x[1], 64, TK, 1, NA, xbf[0], 0)
    for s_, cst in enumerate(c_steps):
        t1 = al(Ms * De * 2)
        prog.dense(s_, "tin", A_x, W_fold, t1, De * 2, Ms, De, 256, 64)
        prog.add(T_SILU, s_, "tsilu", t1, De, 0, De, Ms, A_t[0], A_t[1], 64, TK, 2, cst, 0)
        emb = al(Ms * De * 2)
        prog.dense(s_, "tout", A_t, W_tout, emb, De * 2, Ms, De, De, 64)
        h, add, bias = zeros_e, emb, b_tout
        for l, d in enumerate(ew):
            hn = al(Ms * De * 2)
            prog.add(T_PRENORM, l, "enorm1", h, add, hn, d["g1"], A_e[0], A_e[1], De, Ms, 64, TK, eps_e, 0, 0, 720,
                     0, bias, 0)
            bias = 0
            if d["self"]:
                qkv = al(Ms * 1600 * 2)
                prog.dense(l, "eqkv", A_e, d["qkv"], qkv, 3200, Ms, 1600, De, 64)
                prog.add(T_ROPE, l, "erope", qkv, 1600, 15, 5, 64, scos[0], scos[1], Qe[0], Qe[1], 192 * 64, 192 * 2,
                         kc[l][0], kc[l][1], SKe * 64, SKe * 2, vc[l][0], vc[l][1], SKe * 64, SKe * 2, 256, Ms, SKe, 64,
                         0, 0, 0, 0, 3, 0, 0)
                kb, kbs, vb, vbs, SK, tab, b2, NT, KSv = kc[l][0], kc[l][1], vc[l][0], vc[l][1], SKe, stab, 256, 8, 2
                kb_bs, ksc_bs = SKe * 64, SKe * 2
            else:
                qq = al(Ms * 960 * 2)
                prog.dense(l, "eqkv", A_e, d["q"], qq, 1920, Ms, 960, De, 64)
                prog.add(T_ROPE, l, "erope", qq, 960, 15, 0, 64, ccos[0], ccos[1], Qe[0], Qe[1], 192 * 64, 192 * 2,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, Ms, Mp, 64, 0, 0, 0, 0, 3, 0, 0)
                kb, kbs, vb, vbs, SK, tab, b2, NT, KSv = xkc[l][0], xkc[l][1], xvc[l][0], xvc[l][1], Mp, ctab, NOROW, 4, 1
                kb_bs, ksc_bs = Mp * 64, Mp * 2
            S = al(5 * 192 * SK * 2)
            prog.gemm(l, "eqk", Qe[0], Qe[1], 192 * 64, 192 * 2, kb, kbs, kb_bs, ksc_bs, S, 192 * SK * 2, SK * 2,
                      64 * SK * 2, 5, 3, NT, 1, 64, 64)
            prog.add(T_SOFTMAX, l, "esoftmax", S, 192 * SK * 2, SK, 192, 5, Pe[0], Pe[1], 192 * 512, 192 * 16, s18, 0,
                     Ms, 64, TK, 0, 0, tab, b2)
            O = al(15 * Ms * 64 * 2)
            prog.gemm(l, "epv", Pe[0], Pe[1], 192 * 512, 192 * 16, vb, vbs, kb_bs, ksc_bs, O, 192 * 64 * 2, 128,
                      64 * 128, 5, 3, 1, KSv, 64, TK)
            prog.add(T_GATHERQ, l, "egather", O, 128, htab_e, 16, 64, Ms, A_eo[0], A_eo[1], 64, TK)
            y = al(Ms * De * 2)
            prog.dense(l, "eo", A_eo, d["o"], y, De * 2, Ms, De, Dp, 64)
            hm = al(Ms * De * 2)
            prog.add(T_PRENORM, l, "enorm2", hn, y, hm, d["g2"], A_e[0], A_e[1], De, Ms, 64, TK, eps_e, 0, 0, 720,
                     0, 0, 0)
            gu = al(Ms * 4096 * 2)
            prog.dense(l, "egu", A_e, d["gu"], gu, 8192, Ms, 4096, De, 64)
            prog.add(T_SILU, l, "esilu", gu, 4096, 0, 2048, Ms, A_ef[0], A_ef[1], 64, TK, 0, 0, 0)
            ym = al(Ms * De * 2)
            prog.dense(l, "edown", A_ef, d["d"], ym, De * 2, Ms, De, 2048, 64)
            h, add = hm, ym
        hf = al(Ms * De * 2)
        prog.add(T_PRENORM, 255, "efinal", h, add, hf, g_efinal, A_e[0], A_e[1], De, Ms, 64, TK, eps_e, 0, 0, 720,
                 0, 0, 0)
        vout = al(Ms * 64 * 2)
        prog.dense(255, "aout", A_e, W_aout, vout, 128, Ms, 64, De, 64)
        prog.add(T_EULER, s_, "euler", x_bufs[s_], vout, 128, dt, x_bufs[s_ + 1], A_x[0], A_x[1], 64, TK, 1, NA,
                 xbf[s_ + 1], b_aout)

    # the vision layer-0 zero branch was registered after the init phases were emitted: insert its FILL up front
    extra = [f for f in fills[nfill:]]
    for a_, nb in extra:
        prog.ph.insert(0, [T_FILL, (255 << 8) | OP["init"], a_, nb, 1, 0] + [0] * (PW - 6))
    nfill += len(extra)
    NP = len(prog.ph)
    assert NP <= 3072, NP

    # checks: (name, device, gold, elements, scale_log2, kind, tol in 0.01 %)
    # tolerances (relative Frobenius error, 0.01 %) from FireSim runs of the full model at 1 / 2 / 4 SMs plus margin:
    # device vs emulation 37 / 22 / 0.8 / 9.4-11.7 %, vs fp32 48 / 33 / - / 11-13 % (the emulation itself is 38 / 23 /
    # - / 7.7 % from fp32: E4M3 MX at every GEMM input; the rest is the MX Gemmini's reduced-precision accumulation)
    checks = [("vision features vs emulation", vfeat, gold["vision"], Mv * 768, 10, 0, 4500),
              ("vision features vs fp32 reference", vfeat, gold["ref_vision"], Mv * 768, 10, 0, 5500),
              ("prefix hidden vs emulation", pfinal, gold["prefix"], Lv * Dp, 8, 0, 3000),
              ("prefix hidden vs fp32 reference", pfinal, gold["ref_prefix"], Lv * Dp, 8, 0, 4000),
              ("x after step 1 vs emulation", xbf[1], gold["x1"], Ms * 32, 12, 0, 300),
              ("actions vs emulation", xbf[-1], gold["x10"], Ms * 32, 12, 0, 1800),
              ("actions vs fp32 reference", xbf[-1], gold["ref_x10"], Ms * 32, 12, 0, 1800)]
    # report only (any error passes): where in vision layer 0 the device leaves the emulation
    if fdg:   # report only
        for j in range(8):
            checks.append((f"FA0 S block {j}", fa_dump + j * 64 * 256, fdg["S"] + j * 64 * 256, 64 * 128, 8, 0, 1000000))
        for j in range(8):
            checks.append((f"FA0 P block {j}", fa_dump + 131072 + j * 64 * 256, fdg["P"] + j * 64 * 256, 64 * 128, 8, 0,
                           1000000))
        checks += [("FA0 m", fa_dump + 262144, fdg["m"], 128, 8, 0, 1000000),
                   ("FA0 l", fa_dump + 262400, fdg["l"], 128, 8, 0, 1000000),
                   ("FA0 O unnormalized", fa_dump + 262656, fdg["PV"], 64 * 64, 8, 0, 1000000)]
    for k, dev in (("O", O_v[0]), ("O_sc", O_v[1]), ("Q", Q_v[0]), ("Q_sc", Q_v[1]), ("K", K_v[0]), ("K_sc", K_v[1]), ("V", V_v[0]), ("V_sc", V_v[1])):
        if k in vrg:
            checks.append((f"V0 rope {k} (bytes)", dev, vrg[k][0], vrg[k][1], 8, 1, vrg[k][1]))
    checks += [(f"V0 {k} vs emulation", v0[k][0], gold["vt_" + k], Mv * v0[k][1], 10, 0, 1000000)
               for k in ("patch", "h0", "qkv", "y", "hmid", "f1", "f2") if "vt_" + k in gold]
    pa_ = prog.array().tobytes()
    off = prog_addr - g.REGION_BASE[3]
    mem.blobs[3][off:off + len(pa_)] = pa_
    mem.write(outdir)
    with open(header, "w") as f:
        f.write("// generated by llama_e2e/gen_vla.py --model smolvla; do not edit\n")
        f.write("#ifndef LLAMA_DATA_H\n#define LLAMA_DATA_H\n#include <stdint.h>\n")
        for k, v_ in dict(LL_M=Mp, LL_SK=SKe, LL_P0=0, LL_L=R["n_text"], LL_D=960, LL_F=2560, LL_HQ=15, LL_HKV=5,
                          LL_HD=64, LL_V=0, LL_NP=NP, LL_NFILL=nfill, LL_PW=PW, LL_TN=TN, LL_SYNTHETIC=0).items():
            f.write(f"#define {k} {v_}u\n")
        f.write('#define LL_MODEL "smolvla"\n')
        f.write(f"#define LL_PROG_ADDR 0x{prog_addr:08x}u\n#define LL_PERF_ADDR 0x{perf:08x}u\n")
        f.write("#define LL_LOGITS_ADDR 0x00000000u\n#define LL_LABELS_ADDR 0x00000000u\n")
        f.write("#define LL_HF_LOGITS_ADDR 0x00000000u\n#define LL_EMU_LOGITS_ADDR 0x00000000u\n")
        for n_, v_ in (("GEMM", 1), ("PRENORM", 2), ("ROPE", 3), ("SOFTMAX", 4), ("EWQ", 5), ("FILL", 6), ("SILU", 7),
                       ("ACT", 8), ("FA", 9), ("AFFINE", 10), ("GATHERQ", 11), ("EULER", 12)):
            f.write(f"#define LL_T_{n_} {v_}u\n")
        f.write("#ifdef LL_HOST\n")
        # raw bytes the host prints device / golden (debug): the ATTN output operand's first codes and scales, y row 0
        hexd = []
        if "O" in vrg:
            hexd = [("O_v row0 codes [0:64]", O_v[0], vrg["O"][0], 64),
                    ("O_v row1 codes [0:64]", O_v[0] + 256, vrg["O"][0] + 256, 64),
                    ("O_v scales g0 rows 0..63", O_v[1], vrg["O_sc"][0], 64),
                    ("O_v scales g1 rows 0..63", O_v[1] + 128, vrg["O_sc"][0] + 128, 64),
                    ("y row 0 [0:16] bf16", v0["y"][0], gold["vt_y"], 32)]
            if fdg:   # tile 0's unnormalized O rows 0, 1 (bf16, 64 each) and 1 / l rows 0..7 (f32), device only
                hexd += [("FA0 O row0 bf16", fa_dump + 262656, fdg["PV"], 128),
                         ("FA0 O row1 bf16", fa_dump + 262656 + 128, fdg["PV"] + 128, 128),
                         ("FA0 inv rows 0..7 f32 (gold: l)", fa_dump + 262656 + 64 * 128, fdg["l"], 32)]
        f.write(f"#define LL_HEX_N {len(hexd)}u\nstruct LlHex {{ const char *name; uint32_t dev, gold, n; }};\n")
        f.write("static const LlHex LL_HEX[] = {" + "".join(f'{{"{n_}", 0x{d_:08x}u, 0x{g_:08x}u, {k_}u}}, '
                                                         for n_, d_, g_, k_ in hexd) + "{0, 0, 0, 0}};\n")
        f.write("static const char *const LL_OPS[] = {" + ", ".join(f'"{o}"' for o in OPS) + "};\n")
        f.write(f"#define LL_NOPS {len(OPS)}u\n#define LL_NCHECKS {len(checks)}u\n")
        f.write("#define LL_CPU_ONLY 0\n#define LL_NCSUM 0u\nstatic const uint32_t LL_CSUM[1][3] = {{0, 0, 0}};\n")
        f.write("struct LlCheck { const char *name; uint32_t dev, gold, n; int scale_log2; int bytes; uint32_t tol; };\n")
        f.write("static const LlCheck LL_CHECKS[] = {\n")
        for c in checks:
            f.write(f'  {{"{c[0]}", 0x{c[1]:08x}u, 0x{c[2]:08x}u, {c[3]}u, {c[4]}, {c[5]}, {c[6]}u}},\n')
        f.write("  {0, 0, 0, 0, 0, 0, 0}\n};\n#endif\n#endif\n")
    used = {r: (mem.arena[r] - g.REGION_BASE[r]) >> 20 for r in mem.arena}
    blob = {r: len(b) >> 20 for r, b in mem.blobs.items()}
    print(f"smolvla: phases={NP} (init {nfill}); blob MiB {blob}, used MiB {used}", flush=True)


def compare(name, a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
    print(f"  {name}: frob_rel_err={err * 100:.2f}% max_abs={np.abs(a - b).max():.4f}", flush=True)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="validate the numpy pipeline against lerobot and exit")
    ap.add_argument("--out", default="data")
    ap.add_argument("--header", default="llama_data.h")
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--fa-dump", action="store_true",
                    help="build with -DLL_FA_DUMP=1: the first vision ATTN phase dumps tile 0 (S, P, m, l, O), checked vs "
                         "the emulation (report only)")
    ap.add_argument("--layers", type=int, default=None,
                    help="truncate the vision, prefix and expert stacks to N layers each (the expert's layer l reads the "
                         "prefix layer l cache; N = 1: only self-attention, N >= 2 also cross-attention); the fp32 "
                         "reference is then the numpy pipeline (it matches lerobot exactly for the full model)")
    a = ap.parse_args()
    global FA_DUMP
    FA_DUMP = a.fa_dump
    print("lerobot reference ...", flush=True)
    R = load_reference()
    if a.layers:
        for k in ("n_vision", "n_text", "n_expert"):
            R[k] = min(R[k], a.layers)
        print(f"truncated to {a.layers} layer(s) per stack", flush=True)
    print("numpy fp32 pipeline" + (" (truncated: no lerobot comparison)" if a.layers else " vs lerobot:"), flush=True)
    ref = forward(R, q=False)
    if not a.layers:
        compare("actions", ref["actions"], R["actions"])
    print("device emulation (quantized) vs lerobot:", flush=True)
    emu = forward(R, q=True)
    compare("vision features", emu["vision"], ref["vision"])
    compare("prefix hidden (valid rows)", emu["prefix_h"][:205], ref["prefix_h"][:205])
    compare("x after step 1", emu["x"][1][:50], ref["x"][1][:50])
    compare("actions", emu["actions"], ref["actions"])
    if a.check_only:
        return
    build(R, emu, ref, a.out, a.header)


if __name__ == "__main__":
    main()
