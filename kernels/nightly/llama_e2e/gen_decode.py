#!/usr/bin/env python3
"""llama_e2e decode: B sequences decode S tokens each after a P-token context, every stage on the device.

gen_data.py's models, MX formats and phase types, with the batch of B sequences as the GEMM rows (M = B = 16).  Each
sequence has its own KV cache (from the HF model for real weights) of capacity SKc; step t feeds token P + t of every
sequence (teacher forcing with the text), appends its key / value to that sequence's cache and produces the next-token
logits [B][V].  Per step and layer:

  norm1   PRENORM  h = h_mid + y_mlp (layer 0: x_t);  xn1 = MX(rmsnorm(h) * g_in)
  qkv     GEMM     [Q | K | V] = xn1 @ Wqkv                                   [B][NQKV]
  drope   DROPE    per (sequence b, KV head g): Q heads -> rows 0..Gq-1 of QK batch b Hkv + g; key -> column pos of the
                   K^T cache; V group of pos requantized (scales along keys) from this layer's earlier steps
  qk      GEMM     batch B Hkv: S = Q_bg @ K_bg^T, TMq rows (Gq used), keys 0..pos rounded up to 64
  softmax SOFTMAX  rows 0..Gq-1 of each batch (d[18]), mask key <= pos (M = 1, P0 = pos), MX -> P (16-row tiles)
  pv      GEMM     batch B Hkv: O_bg = P_bg @ V_bg, 16 rows x hd per batch
  oquant  EWQ      gather (d[12]): O [B][Dq] from the per-batch rows, MX
  oproj, norm2, gateup, silu, down   as gen_data.py (unfused: M = 16 is below the fused paths' 32-row minimum)
  final, lmhead (+ softcap)          logits of step t -> logits[t] [B][V]

Attention runs as GEMMs with the query heads of one KV head as rows: Gq of TMq rows are real (TMq = 64 / 32 / 16 for
hd 64 / 128 / 256, the GEMM's minimum compute-loop length), so QK does TMq / Gq times the useful work; the cost of
decode attention is reading the caches anyway.  P0 = P must be a multiple of 32 (DROPE's V groups start at P).

Speculative-decoding verify (--verify K): each step takes T = K + 1 tokens per sequence (the K drafts and the token
before them; here the text's own tokens, teacher forcing), so the GEMMs have B T rows (row b T + j) and read every weight
once per T tokens; attention batch (b, g) has Gq T query rows (row jh T + j, prefill's head-major layout: SOFTMAX masks
row r at key P0 + r % T with M = T), and DROPE appends the step's T keys.  The host or a draft model would accept the
longest matching prefix; the device cost of a step does not depend on how many are accepted.

Checks: logits of every step vs the HF fp32 model (top-1 >= 85 %, same gate as prefill) and vs a numpy emulation of
this pipeline; the last hidden state of every step vs the emulation.  Synthetic models (llama7b, gemma2): device-filled
weights and caches (one layer's, shared), finite logits only.
"""
import argparse, glob, os, sys
import numpy as np

from gen_data import (MODELS, PW, TN, TK, T_GEMM, T_PRENORM, T_SOFTMAX, T_EWQ, T_FILL, T_SILU, T_ACT, Mem, Prog, OPS,
                      REGION_BASE, mx_quant, mx_dequant, pack_b, quant_weight, deq_weight, bf, bf_scalar, bf16_bits,
                      f32_bits, rope_tables, rope, key_center, silu_dev, gelu_dev, tanh_dev)

T_DROPE = 10
NP_MAX = 4096                    # phase slots (program and perf table)


def load_hf_decode(cfg, B, P, S, layers):
    """B text windows of P + S + 1 tokens: the P-token KV cache, the S decode inputs, HF's logits at those steps"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow.parquet as pq
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], dtype=torch.float32)
    model.eval()
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    files = sorted(glob.glob(os.path.join(
        hub, "datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet")))
    assert files, f"wikitext-2 test split not in {hub}"
    text = "".join(pq.read_table(files[0]).column("text").to_pylist())
    n = P + S + 1
    allids = tok(text, return_tensors="pt").input_ids[0]
    assert allids.numel() >= B * n
    ids = torch.stack([allids[b * n:(b + 1) * n] for b in range(B)])        # window 0 = the prefill test's text
    Lm = model.config.num_hidden_layers
    kc = [[] for _ in range(layers)]
    vc = [[] for _ in range(layers)]
    logits = []
    for c0 in range(0, B, 4):                                                # 4 sequences per forward
        out = model.model(input_ids=ids[c0:c0 + 4, :P + S], use_cache=True)
        logits.append(model.lm_head(out.last_hidden_state[:, P:P + S]).float().numpy())   # [4][S][V]
        past = out.past_key_values
        for l in range(layers):
            if hasattr(past, "layers"):
                k, v = past.layers[l].keys, past.layers[l].values
            elif hasattr(past, "key_cache"):
                k, v = past.key_cache[l], past.value_cache[l]
            else:
                k, v = past[l]
            kc[l].append(k[:, :, :P].float().numpy())                        # [4][Hkv][P][hd]
            vc[l].append(v[:, :, :P].float().numpy())
        del out, past
    sd = model.state_dict()
    W = []
    for l in range(layers):
        p = f"model.layers.{l}."
        g = lambda nm: sd[p + nm].float().numpy()
        W.append(dict(
            wqkv=np.concatenate([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                 g("self_attn.v_proj.weight")], 0),
            wo=g("self_attn.o_proj.weight"),
            wgu=np.concatenate([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")], 0),
            wd=g("mlp.down_proj.weight"),
            g_in=g("input_layernorm.weight"), g_post=g("post_attention_layernorm.weight"),
            kc=np.concatenate(kc[l], 0), vc=np.concatenate(vc[l], 0)))      # [B][Hkv][P][hd]
        if cfg.get("bias"):
            W[-1]["bqkv"] = np.concatenate([g("self_attn.q_proj.bias"), g("self_attn.k_proj.bias"),
                                            g("self_attn.v_proj.bias")], 0)
    emb = sd["model.embed_tokens.weight"].float().numpy()
    x0 = np.stack([emb[ids[:, P + t].numpy()] for t in range(S)])           # [S][B][D]
    wlm = sd["lm_head.weight"] if "lm_head.weight" in sd else sd["model.embed_tokens.weight"]
    return dict(layers=W, g_final=sd["model.norm.weight"].float().numpy(), wlm=wlm.float().numpy(), x0=x0,
                logits=np.concatenate(logits, 0).transpose(1, 0, 2),           # [S][B][V]
                labels=ids[:, P + 1:P + S + 1].numpy().T, full=layers == Lm)  # [S][B]


def random_decode(cfg, B, P, S, layers, rng):
    D, F, Hq, Hkv, hd, V = cfg["D"], cfg["F"], cfg["Hq"], cfg["Hkv"], cfg["hd"], cfg["V"]
    Dq, Dkv = Hq * hd, Hkv * hd
    W = []
    for _ in range(layers):
        W.append(dict(
            wqkv=rng.normal(0, D ** -0.5, (Dq + 2 * Dkv, D)), wo=rng.normal(0, Dq ** -0.5, (D, Dq)),
            wgu=rng.normal(0, D ** -0.5, (2 * F, D)), wd=rng.normal(0, F ** -0.5, (D, F)),
            g_in=rng.uniform(0.5, 1.5, D), g_post=rng.uniform(0.5, 1.5, D),
            kc=rng.normal(0, 1, (B, Hkv, P, hd)), vc=rng.normal(0, 1, (B, Hkv, P, hd))))
        if cfg.get("bias"):
            W[-1]["bqkv"] = rng.normal(0, 0.5, Dq + 2 * Dkv)
        if cfg.get("gemma"):
            W[-1]["g_pa"], W[-1]["g_pf"] = rng.uniform(0.5, 1.5, D), rng.uniform(0.5, 1.5, D)
    x0 = rng.normal(0, 1, (S, B, D)) * (bf_scalar(np.sqrt(D)) if cfg.get("gemma") else 1.0)
    return dict(layers=W, g_final=rng.uniform(0.5, 1.5, D), wlm=rng.normal(0, D ** -0.5, (V, D)), x0=x0,
                logits=None, labels=None, full=False)


def emulate_decode(cfg, B, P, S, SKc, mdl, qw, T=1):
    """the device pipeline in numpy (gen_data.emulate's operations, per sequence caches, unfused paths).  Step t feeds
    T tokens per sequence (positions P + t T + j, rows b T + j of mdl["x0"][t] [B T][D]); token j sees keys <= its own."""
    D, F, Hq, Hkv, hd, eps = cfg["D"], cfg["F"], cfg["Hq"], cfg["Hkv"], cfg["hd"], cfg["eps"]
    Dq, Dkv, Gq, R = Hq * hd, Hkv * hd, Hq // Hkv, B * T
    gemma, acap, fcap = cfg.get("gemma", False), cfg.get("attn_cap"), cfg.get("final_cap")
    scale = np.float32(1.0 / np.sqrt(cfg.get("qpre", hd)))

    def norm_f(h, gamma):
        ss = np.sum(h.astype(np.float32) ** 2, axis=1, dtype=np.float32)
        inv = (np.float32(1) / np.sqrt(ss / np.float32(D) + np.float32(eps))).astype(np.float32)
        return (h * inv[:, None]).astype(np.float32) * bf(gamma)[None, :]

    def norm_q(h, gamma):
        return mx_dequant(*mx_quant(norm_f(h, gamma)))

    def resid(x, y, g2):
        return bf(x + (bf(norm_f(y, g2)) if g2 is not None else y))

    layers = mdl["layers"]
    Kc, Vc = [], []
    for w in layers:
        kcen = w.get("kcen", np.zeros((Hkv, hd), np.float32))
        k = np.zeros((B, Hkv, SKc, hd), np.float32)
        v = np.zeros((B, Hkv, SKc, hd), np.float32)
        k[:, :, :P] = np.asarray(w["kc"], np.float32) - kcen[None, :, None, :]
        v[:, :, :P] = w["vc"]
        Kc.append(k)
        Vc.append(v)
    out = dict(logits=[], h=[])
    for t in range(S):
        p0 = P + t * T
        last = p0 + T - 1
        cos, sin = rope_tables(T, p0, hd, cfg["theta"])
        cos, sin = np.tile(cos, (B, 1)), np.tile(sin, (B, 1))                      # row b T + j: position p0 + j
        valid = np.arange(last + 1)[None, :] <= p0 + np.arange(T)[:, None]         # [T][keys]
        h = bf(mdl["x0"][t])
        hmid = ymlp = None
        for l, w in enumerate(layers):
            q = qw[l]
            if l > 0:
                h = resid(hmid, ymlp, layers[l - 1].get("g_pf") if gemma else None)
            xn = norm_q(h, w["g_in"])
            qkv = bf(xn @ deq_weight(q["wqkv"]))
            if "bqkv" in w:
                qkv = bf(qkv + bf(w["bqkv"])[None, :])
            Q = rope(qkv[:, :Dq].reshape(R, Hq, hd), cos, sin)
            kcen = w.get("kcen", np.zeros((Hkv, hd), np.float32))
            Kn = rope(qkv[:, Dq:Dq + Dkv].reshape(R, Hkv, hd), cos, sin) - kcen[None]
            Kc[l][:, :, p0:last + 1] = Kn.reshape(B, T, Hkv, hd).transpose(0, 2, 1, 3)
            Vc[l][:, :, p0:last + 1] = qkv[:, Dq + Dkv:].reshape(B, T, Hkv, hd).transpose(0, 2, 1, 3)
            Qd = mx_dequant(*mx_quant(Q)).reshape(B, T, Hkv, Gq, hd)
            Kd = mx_dequant(*mx_quant(Kc[l][:, :, :last + 1]))                     # groups along hd
            Vd = mx_dequant(*mx_quant(Vc[l].transpose(0, 1, 3, 2))).transpose(0, 1, 3, 2)   # groups along keys
            Sx = bf(np.einsum("btgjd,bgkd->bgjtk", Qd, Kd, dtype=np.float32))      # [B][Hkv][Gq][T][keys]
            if acap:
                z = (np.float32(acap) * tanh_dev(Sx * np.float32(scale / acap))).astype(np.float32)
            else:
                z = (Sx * scale).astype(np.float32)
            m = np.where(valid, z, -np.inf).max(axis=-1, keepdims=True)
            e = np.where(valid, bf(np.exp(bf(z - m))), 0).astype(np.float32)
            p = (e * (np.float32(1) / np.sum(e, axis=-1, keepdims=True, dtype=np.float32))).astype(np.float32)
            Pf = np.zeros((B, Hkv, Gq, T, SKc), np.float32)
            Pf[..., :last + 1] = p
            O = bf(np.einsum("bgjtk,bgkd->btgjd", mx_dequant(*mx_quant(Pf)), Vd, dtype=np.float32)).reshape(R, Dq)
            yattn = bf(mx_dequant(*mx_quant(O)) @ deq_weight(q["wo"]))
            hmid = resid(h, yattn, w.get("g_pa") if gemma else None)
            xn2 = norm_q(hmid, w["g_post"])
            gu = bf(xn2 @ deq_weight(q["wgu"]))
            G, U = gu[:, :F], gu[:, F:]
            Hh = ((gelu_dev(G) if gemma else silu_dev(G)) * U).astype(np.float32)
            ymlp = bf(mx_dequant(*mx_quant(Hh)) @ deq_weight(q["wd"]))
        h = resid(hmid, ymlp, layers[-1].get("g_pf") if gemma else None)
        out["h"].append(h)
        lg = bf(mx_dequant(*mx_quant(norm_f(h, mdl["g_final"]))) @ deq_weight(qw["wlm"]))
        if fcap:
            lg = bf(np.float32(fcap) * tanh_dev(lg * np.float32(1.0 / fcap)))
        out["logits"].append(lg)
    return out


def check_l0_goldens(cfg, B, P, S, SKk, SKc, TMp, mdl, qw, T=1):
    """--check-l0: layer 0 in device formats with every GEMM through mx_golden (bit-exact with the MX Gemmini), each
    stage from the previous stage's golden.  Layer 0 of step t depends only on x_t and layer 0's cache, so: qkv of every
    step, the K^T / V caches after all S appends (device byte layouts), and step 0's O (PV output, [NB][TMp][hd] with the
    padded rows 0), yattn, h_mid, gate/up, down.  T tokens per sequence and step: rows b T + j (O: jh T + j)."""
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "lib" / "golden"))
    import golden
    from gen_data import pack_b as pb, bf16_to_f32
    D, F, Hq, Hkv, hd, eps = cfg["D"], cfg["F"], cfg["Hq"], cfg["Hkv"], cfg["hd"], cfg["eps"]
    Dq, Dkv, Gq, R = Hq * hd, Hkv * hd, Hq // Hkv, B * T
    tmp = pathlib.Path(__file__).resolve().parent / "_gen" / "dbg_dec"

    def gm(xc, xs, wc, ws):   # A codes [M][K] + scales [M][K/32] @ B given as (codes [N][K], scales [N][K/32])
        M0, K = xc.shape
        N0 = wc.shape[0]
        M, N = -(-M0 // 16) * 16, -(-N0 // 64) * 64
        xc, xs = np.pad(xc, ((0, M - M0), (0, 0))), np.pad(xs, ((0, M - M0), (0, 0)), constant_values=1)
        wc, ws = np.pad(wc, ((0, N - N0), (0, 0))), np.pad(ws, ((0, N - N0), (0, 0)), constant_values=1)
        return golden.mx_matmul(np.ascontiguousarray(xc), np.ascontiguousarray(wc.T), np.ascontiguousarray(xs.T),
                                np.ascontiguousarray(ws.T), M, N, K, tmpdir=str(tmp))[:M0, :N0]

    def dense(x, wq):
        return gm(*mx_quant(np.asarray(x, np.float32)), *wq)

    def norm_q(h, gamma):
        ss = np.sum(h * h, axis=1, dtype=np.float32)
        inv = (np.float32(1) / np.sqrt(ss / np.float32(D) + np.float32(eps))).astype(np.float32)
        return (h * inv[:, None]).astype(np.float32) * bf(gamma)[None, :]

    w, q = mdl["layers"][0], qw[0]
    kcen = w.get("kcen", np.zeros((Hkv, hd), np.float32))
    K = np.zeros((B, Hkv, SKk, hd), np.float32)
    V = np.zeros((B, Hkv, SKc, hd), np.float32)
    K[:, :, :P] = w["kc"] - kcen[None, :, None, :]
    V[:, :, :P] = w["vc"]
    out = {}
    for t in range(S):
        p0 = P + t * T
        last = p0 + T - 1
        cos, sin = rope_tables(T, p0, hd, cfg["theta"])
        cos, sin = np.tile(cos, (B, 1)), np.tile(sin, (B, 1))
        h = bf(mdl["x0"][t])
        qkv_bits = dense(norm_q(h, w["g_in"]), q["wqkv"])
        out[f"qkv{t}"] = qkv_bits
        qkv = bf16_to_f32(qkv_bits)
        if "bqkv" in w:
            qkv = bf(qkv + bf(w["bqkv"])[None, :])
        Q = rope(qkv[:, :Dq].reshape(R, Hq, hd), cos, sin)
        Kn = rope(qkv[:, Dq:Dq + Dkv].reshape(R, Hkv, hd), cos, sin) - kcen[None]
        K[:, :, p0:last + 1] = Kn.reshape(B, T, Hkv, hd).transpose(0, 2, 1, 3)
        V[:, :, p0:last + 1] = qkv[:, Dq + Dkv:].reshape(B, T, Hkv, hd).transpose(0, 2, 1, 3)
        if t:
            continue
        kc_, ks_ = mx_quant(K[:, :, :last + 1])
        vc_, vs_ = mx_quant(V.transpose(0, 1, 3, 2))
        qc, qs = mx_quant(Q.reshape(B, T, Hkv, Gq, hd).transpose(0, 2, 3, 1, 4).reshape(B, Hkv, Gq * T, hd))
        scale = np.float32(1.0 / np.sqrt(cfg.get("qpre", hd)))
        acap = cfg.get("attn_cap")
        valid = np.arange(last + 1)[None, :] <= p0 + np.arange(Gq * T)[:, None] % T      # row jh T + j
        O = np.zeros((B * Hkv, TMp, hd), np.uint16)
        for b in range(B):
            for g in range(Hkv):
                Sx = bf16_to_f32(gm(qc[b, g], qs[b, g], kc_[b, g], ks_[b, g]))
                z = ((np.float32(acap) * tanh_dev(Sx * np.float32(scale / acap))) if acap else Sx * scale).astype(np.float32)
                m = np.where(valid, z, -np.inf).max(axis=1, keepdims=True)
                e = np.where(valid, bf(np.exp(bf(z - m))), 0).astype(np.float32)
                p = np.zeros((Gq * T, SKc), np.float32)
                p[:, :last + 1] = e * (np.float32(1) / np.sum(e, axis=1, keepdims=True, dtype=np.float32))
                O[b * Hkv + g, :Gq * T] = gm(*mx_quant(p), vc_[b, g], vs_[b, g])
        out["o0"] = O
        Ob = bf16_to_f32(O[:, :Gq * T].reshape(B, Hkv, Gq, T, hd)).transpose(0, 3, 1, 2, 4).reshape(R, Dq)
        out["yattn0"] = dense(Ob, q["wo"])
        hmid = bf(h + bf16_to_f32(out["yattn0"]))
        out["hmid0"] = bf16_bits(hmid)
        out["gu0"] = dense(norm_q(hmid, w["g_post"]), q["wgu"])
        gu = bf16_to_f32(out["gu0"])
        G, U = gu[:, :F], gu[:, F:]
        out["ymlp0"] = dense(((gelu_dev(G) if cfg.get("gemma") else silu_dev(G)) * U).astype(np.float32), q["wd"])
    # the caches after S appends, device layouts (as the blob)
    kcodes, ksc = mx_quant(K)
    vcodes, vsc = mx_quant(V.transpose(0, 1, 3, 2))
    kb, kbs, vb, vbs = [], [], [], []
    for b in range(B):
        for g in range(Hkv):
            x_, y_ = pb(kcodes[b, g].T, ksc[b, g].T, hd); kb.append(x_); kbs.append(y_)
            x_, y_ = pb(vcodes[b, g].T, vsc[b, g].T, TK); vb.append(x_); vbs.append(y_)
    out["kc"], out["kc_sc"], out["vc"], out["vc_sc"] = np.stack(kb), np.stack(kbs), np.stack(vb), np.stack(vbs)
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="smoke")
    ap.add_argument("--batch", type=int, default=16, help="sequences (GEMM rows; 16)")
    ap.add_argument("--steps", type=int, default=4, help="decode steps (with --verify: verify steps)")
    ap.add_argument("--verify", type=int, default=0, metavar="K",
                    help="speculative-decoding verify: each step takes K drafted tokens + 1 per sequence (T = K + 1 "
                         "rows per sequence, causal among them); the drafts are the text's tokens (teacher forcing)")
    ap.add_argument("--ctx", type=int, default=None, help="cached context tokens P (multiple of 32; default 1024)")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-emulate", action="store_true")
    ap.add_argument("--lm-chunks", type=int, default=1)
    ap.add_argument("--random", action="store_true",
                    help="random weights and caches at the model's shapes (emulated and checked like a real model)")
    ap.add_argument("--vocab", type=int, default=None,
                    help="LM head over the first N vocabulary rows only (quick builds: the host's logits checks scale "
                         "with B S V; the HF comparison is dropped)")
    ap.add_argument("--check-l0", action="store_true",
                    help="report-only layer-0 checks against mx_golden stage goldens: qkv of every step, the K^T / V "
                         "caches after the appends, step 0's O, yattn, h_mid, gate/up, down")
    ap.add_argument("--out", default="data")
    ap.add_argument("--header", default="llama_data.h")
    a = ap.parse_args()

    cfg = dict(MODELS[a.model])
    if a.random:
        cfg["kind"] = "random"
    B, S, T = a.batch, a.steps, a.verify + 1
    R = B * T                                            # GEMM rows per step: sequence b's token j = row b T + j
    ST = S * T                                           # tokens per sequence
    P = a.ctx or (256 if cfg["kind"] == "random" else 1024)
    L = a.layers or cfg["L"]
    D, F, Hq, Hkv, hd, V = cfg["D"], cfg["F"], cfg["Hq"], cfg["Hkv"], cfg["hd"], a.vocab or cfg["V"]
    Dq, Dkv, Gq, NQKV = Hq * hd, Hkv * hd, Hq // Hkv, Hq * hd + 2 * Hkv * hd
    gemma, bias = cfg.get("gemma", False), cfg.get("bias", False)
    acap, fcap = cfg.get("attn_cap"), cfg.get("final_cap")
    synthetic = cfg["kind"] == "synthetic"
    pow2 = lambda n: 1 << (n - 1).bit_length()
    VR = Gq * T                                          # attention rows per batch: head jh, token j = row jh T + j
    TMq = max({64: 64, 128: 32, 256: 16}[hd], pow2(VR))  # QK rows: (TMq / 16) 4 (hd / 16) >= 64 (Prog.gemm)
    TMp = max(16, pow2(VR))                              # P / O rows per batch (TK = 256)
    SKk = -(-(P + ST) // 64) * 64                        # K^T cache keys (64-key tiles)
    SKc = -(-(P + ST) // 256) * 256                      # V cache keys, S / P columns (256-key k-steps)
    NB = B * Hkv                                         # attention batches
    assert B == 16, "the GEMM row tile is 16"
    assert T in (1, 2, 4, 8, 16, 32) and R <= 128, "a step's keys in one V group; dense GEMM rows one tile"
    assert P % 32 == 0 and VR <= 128 and SKc * 6 <= 8192, "DROPE groups, P rows, SOFTMAX row stage"
    assert D % TK == 0 and F % TK == 0 and Dq % TK == 0 and D <= 4096
    for n_ in (NQKV, D, 2 * F, V):
        assert n_ % TN == 0, n_
    rng = np.random.default_rng(a.seed)

    if cfg["kind"] == "real":
        print(f"loading {cfg['hf']} and running HF on {B} x {P + S} tokens ...", flush=True)
        mdl = load_hf_decode(cfg, B, P, ST, L)
    elif cfg["kind"] == "random":
        mdl = random_decode(cfg, B, P, ST, L, rng)
    else:
        mdl = None
    if mdl and a.vocab:
        mdl["wlm"], mdl["full"] = mdl["wlm"][:V], False
    if mdl:   # token-major [S T][B] -> device rows per step [S][B T] (row b T + j)
        rows = lambda x: np.ascontiguousarray(np.asarray(x).reshape(S, T, B, *np.shape(x)[2:]).swapaxes(1, 2)
                                              .reshape(S, R, *np.shape(x)[2:]))
        mdl["x0"] = rows(mdl["x0"])
        if mdl["logits"] is not None:
            mdl["logits"], mdl["labels"] = rows(mdl["logits"]), rows(mdl["labels"])
    for w in (mdl["layers"] if mdl else []):
        if "bqkv" in w:
            w["kcen"] = key_center(cfg, w["bqkv"], P)

    mem = Mem("llama")
    prog = Prog()
    checks = []
    prog_addr = mem.add(3, np.zeros(PW * NP_MAX, np.uint32))
    nlm = max(-(-D * V // (192 << 20)), a.lm_chunks)
    vc_ = -(-V // nlm // TN) * TN
    lm_chunks = [(c0, min(c0 + vc_, V)) for c0 in range(0, V, vc_)]
    kcb, vcb = NB * SKk * hd, NB * SKc * hd              # one layer's K^T / V cache bytes

    # ---------------- blob ----------------
    lw, qw = [], {}
    if not synthetic:
        for l, w in enumerate(mdl["layers"]):
            q = {k: quant_weight(w[k]) for k in ("wqkv", "wo", "wgu", "wd")}
            qw[l] = {k: v[1] for k, v in q.items()}
            d = {}
            for k in ("wqkv", "wo", "wgu", "wd"):
                (bt_, bs_), _ = q[k]
                d[k], d[k + "_sc"] = mem.add_any(bt_), mem.add_any(bs_)
            del q
            d["g_in"], d["g_post"] = mem.add(3, bf16_bits(w["g_in"])), mem.add(3, bf16_bits(w["g_post"]))
            for k in ("g_pa", "g_pf"):
                if k in w:
                    d[k] = mem.add(3, bf16_bits(w[k]))
            if "bqkv" in w:
                d["bqkv"], d["kcen"] = mem.add(3, bf16_bits(w["bqkv"])), mem.add(3, w["kcen"])
            # per (sequence, KV head): K^T [hd][SKk] pretiled (TK = hd), V [SKc][hd] pretiled (TK = 256)
            kc = np.zeros((B, Hkv, SKk, hd), np.float32)
            vc = np.zeros((B, Hkv, SKc, hd), np.float32)
            kc[:, :, :P] = w["kc"] - (w["kcen"][None, :, None, :] if "kcen" in w else 0)
            vc[:, :, :P] = w["vc"]
            kcodes, ksc = mx_quant(kc)
            vcodes, vsc = mx_quant(vc.transpose(0, 1, 3, 2))
            kb, kbs, vb, vbs = [], [], [], []
            for b in range(B):
                for g in range(Hkv):
                    x_, y_ = pack_b(kcodes[b, g].T, ksc[b, g].T, hd); kb.append(x_); kbs.append(y_)
                    x_, y_ = pack_b(vcodes[b, g].T, vsc[b, g].T, TK); vb.append(x_); vbs.append(y_)
            d["kc"], d["kc_sc"] = mem.add_any(np.stack(kb)), mem.add_any(np.stack(kbs))
            d["vc"], d["vc_sc"] = mem.add_any(np.stack(vb)), mem.add_any(np.stack(vbs))
            lw.append(d)
        lcodes, lsc = mx_quant(np.asarray(mdl["wlm"], np.float32))
        qw["wlm"] = (lcodes, lsc)
        wlm = []
        for c0, c1 in lm_chunks:
            bt_, bs_ = pack_b(np.ascontiguousarray(lcodes[c0:c1].T), np.ascontiguousarray(lsc[c0:c1].T), TK)
            wlm.append((mem.add_any(bt_), mem.add_any(bs_)))
        del lcodes, lsc
        g_final = mem.add(3, bf16_bits(mdl["g_final"]))
        x0 = mem.add(3, bf16_bits(mdl["x0"]))                                     # [S][B T][D]
    else:
        x0 = mem.add(3, bf16_bits(rng.normal(0, 1, (S, R, D)) * (bf_scalar(np.sqrt(D)) if gemma else 1.0)))
        ngam = 4 * L + 1 if gemma else 2 * L + 1
        gam = mem.add(3, bf16_bits(rng.uniform(0.5, 1.5, (ngam, D))))
    zeros = mem.add(3, np.zeros(R * D, np.uint16))
    cos_t, sin_t = rope_tables(ST, P, hd, cfg["theta"])
    cos_a, sin_a = mem.add(3, cos_t), mem.add(3, sin_t)   # [S T][hd/2], row = position - P: DROPE reads earlier rows

    emu = None
    if mdl is not None and not a.no_emulate:
        print("emulating the quantized decode ...", flush=True)
        emu = emulate_decode(cfg, B, P, S, SKc, mdl, qw, T)
    gold = {}
    if emu is not None:
        gold["emu_logits"] = mem.add(3, bf16_bits(np.stack(emu["logits"])))
        gold["emu_h"] = [mem.add(3, bf16_bits(h)) for h in emu["h"]]
    l0g = {}
    if a.check_l0:
        assert mdl is not None and not synthetic
        print("check-l0: layer-0 stage goldens (mx_golden) ...", flush=True)
        l0g = {k: (mem.add(3, v), v) for k, v in check_l0_goldens(cfg, B, P, S, SKk, SKc, TMp, mdl, qw, T).items()}
    if mdl is not None and mdl["logits"] is not None:
        gold["hf_logits"] = mem.add(3, bf16_bits(mdl["logits"]))
        gold["labels"] = mem.add(3, mdl["labels"].astype(np.uint32))

    # ---------------- arena ----------------
    al = mem.alloc
    if synthetic:
        sizes = dict(wqkv=(D * NQKV, NQKV * D // 32), wo=(Dq * D, D * Dq // 32),
                     wgu=(D * 2 * F, 2 * F * D // 32), wd=(F * D, D * F // 32))
        sw = {}
        for k, (nb, ns) in sizes.items():
            sw[k], sw[k + "_sc"] = al(nb), al(ns)
        sw["kc"], sw["kc_sc"], sw["vc"], sw["vc_sc"] = al(kcb), al(kcb // 32), al(vcb), al(vcb // 32)
        wlm = [(al(D * (c1 - c0)), al((c1 - c0) * D // 32)) for c0, c1 in lm_chunks]
        npl = 4 if gemma else 2
        for l in range(L):
            d = dict(sw)
            d["g_in"], d["g_post"] = gam + (npl * l) * D * 2, gam + (npl * l + 1) * D * 2
            if gemma:
                d["g_pa"], d["g_pf"] = gam + (npl * l + 2) * D * 2, gam + (npl * l + 3) * D * 2
            lw.append(d)
        g_final = gam + npl * L * D * 2
        fills = [(sw[k], sizes[k][0], 0, 0x9E3779B9 + i) for i, k in enumerate(sizes)]
        fills += [(sw[k + "_sc"], sizes[k][1], 1, 0x7A7A7A7A) for k in sizes]
        fills += [(sw["kc"], kcb, 0, 0x1234567), (sw["vc"], vcb, 0, 0x7654321),
                  (sw["kc_sc"], kcb // 32, 1, 0x7F7F7F7F), (sw["vc_sc"], vcb // 32, 1, 0x7F7F7F7F)]
        for i, ((c0, c1), (wb, ws)) in enumerate(zip(lm_chunks, wlm)):
            fills += [(wb, D * (c1 - c0), 0, 0xABCDEF1 + i), (ws, (c1 - c0) * D // 32, 1, 0x7A7A7A7A)]
    else:
        fills = []
    xn = al(R * D), al(R * D // 32)                      # MX A operands (gemmini-read, reused)
    xf = al(R * D), al(R * D // 32)
    qmx = al(NB * TMq * hd), al(NB * TMq * hd // 32)      # rows >= Gq T stay zero (filled once)
    pmx = al(NB * TMp * SKc), al(NB * TMp * SKc // 32)
    omx = al(R * Dq), al(R * Dq // 32)
    hmx = al(R * F), al(R * F // 32)
    fills += [(qmx[0], NB * TMq * hd, 1, 0), (qmx[1], NB * TMq * hd // 32, 1, 0),
              (pmx[0], NB * TMp * SKc, 1, 0), (pmx[1], NB * TMp * SKc // 32, 1, 0)]
    for base, nb, mode, val in fills:
        assert nb % 4 == 0
        prog.add(T_FILL, 255, "init", base, nb, mode, val)
    s_ring = [al(NB * TMq * SKc * 2) for _ in range(2)]   # Muon-read, 2 sets (as gen_data.py)
    perf = al(64 * 4 * NP_MAX)
    logits = al(S * R * V * 2)
    qstep = R * NQKV * 2
    # every buffer Muon reads is written once per run: per (step, layer); qkv steps of a layer contiguous (DROPE's V)
    qkv_l = [al(S * qstep) for _ in range(L)]
    hbuf = [[al(R * D * 2) for _ in range(L + 1)] for _ in range(S)]
    act = [[dict(qkv=qkv_l[l] + t * qstep, o=al(NB * TMp * hd * 2), yattn=al(R * D * 2), hmid=al(R * D * 2),
                 gu=al(R * 2 * F * 2), ymlp=al(R * D * 2)) for l in range(L)] for t in range(S)]

    # ---------------- phases ----------------
    eps = f32_bits(cfg["eps"])
    ascale = 1.0 / np.sqrt(cfg.get("qpre", hd))
    cap_args = (f32_bits(ascale / acap), f32_bits(acap)) if acap else (0, 0)
    act_kind = 1 if gemma else 0
    pf = lambda l: lw[l].get("g_pf", 0) if gemma else 0
    for t in range(S):
        pos = P + t * T                                  # the step's first position; its tokens pos .. pos + T - 1
        NTq = -(-(pos + T) // 64)                        # QK key tiles up to the last
        for l in range(L):
            w, ac = lw[l], act[t][l]
            if l == 0:
                prog.add(T_PRENORM, l, "norm1", x0 + t * R * D * 2, zeros, hbuf[t][0], w["g_in"], xn[0], xn[1],
                         D, R, R, TK, eps, 0, 0)
            else:
                pa = act[t][l - 1]
                prog.add(T_PRENORM, l, "norm1", pa["hmid"], pa["ymlp"], hbuf[t][l], w["g_in"], xn[0], xn[1],
                         D, R, R, TK, eps, 0, pf(l - 1))
            prog.gemm(l, "qkv", xn[0], xn[1], 0, 0, w["wqkv"], w["wqkv_sc"], 0, 0, ac["qkv"], 0, NQKV * 2,
                      R * NQKV * 2, 1, 1, NQKV // TN, D // TK, R, TK)
            prog.add(T_DROPE, l, "drope", ac["qkv"], NQKV, Hq, Hkv, hd, cos_a, sin_a, qmx[0], qmx[1],
                     TMq * hd, TMq * hd // 32, w["kc"], w["kc_sc"], SKk * hd, SKk * hd // 32, w["vc"], w["vc_sc"],
                     SKc * hd, SKc * hd // 32, pos, qstep, SKc, TMq, P, B, w.get("bqkv", 0), w.get("kcen", 0), T)
            s = s_ring[(t * L + l) & 1]
            prog.gemm(l, "qk", qmx[0], qmx[1], TMq * hd, TMq * hd // 32, w["kc"], w["kc_sc"], SKk * hd,
                      SKk * hd // 32, s, TMq * SKc * 2, SKc * 2, 0, NB, 1, NTq, 1, TMq, hd)
            prog.add(T_SOFTMAX, l, "softmax", s, TMq * SKc * 2, SKc, TMp, NB, pmx[0], pmx[1], TMp * SKc,
                     TMp * SKc // 32, f32_bits(ascale), pos, T, TMp, TK, *cap_args, VR, 1)   # d[18] Gq T rows, d[19] decode
            prog.gemm(l, "pv", pmx[0], pmx[1], TMp * SKc, TMp * SKc // 32, w["vc"], w["vc_sc"], SKc * hd,
                      SKc * hd // 32, ac["o"], TMp * hd * 2, hd * 2, 0, NB, 1, hd // TN, SKc // TK, TMp, TK)
            prog.add(T_EWQ, l, "oquant", ac["o"], Dq, 0, Dq, R, omx[0], omx[1], R, TK, 0, hd, Gq, Hkv, TMp, T)
            prog.gemm(l, "oproj", omx[0], omx[1], 0, 0, w["wo"], w["wo_sc"], 0, 0, ac["yattn"], 0, D * 2,
                      R * D * 2, 1, 1, D // TN, Dq // TK, R, TK)
            prog.add(T_PRENORM, l, "norm2", hbuf[t][l], ac["yattn"], ac["hmid"], w["g_post"], xn[0], xn[1],
                     D, R, R, TK, eps, 0, w.get("g_pa", 0) if gemma else 0)
            prog.gemm(l, "gateup", xn[0], xn[1], 0, 0, w["wgu"], w["wgu_sc"], 0, 0, ac["gu"], 0, 2 * F * 2,
                      R * 2 * F * 2, 1, 1, 2 * F // TN, D // TK, R, TK)
            prog.add(T_SILU, l, "silu", ac["gu"], 2 * F, 1, F, R, hmx[0], hmx[1], R, TK, act_kind)
            prog.gemm(l, "down", hmx[0], hmx[1], 0, 0, w["wd"], w["wd_sc"], 0, 0, ac["ymlp"], 0, D * 2,
                      R * D * 2, 1, 1, D // TN, F // TK, R, TK)
        pa = act[t][L - 1]
        prog.add(T_PRENORM, 255, "final", pa["hmid"], pa["ymlp"], hbuf[t][L], g_final, xf[0], xf[1],
                 D, R, R, TK, eps, 0, pf(L - 1))
        lg = logits + t * R * V * 2
        for (c0, c1), (wb, ws) in zip(lm_chunks, wlm):
            prog.gemm(255, "lmhead", xf[0], xf[1], 0, 0, wb, ws, 0, 0, lg + c0 * 2, 0, V * 2, R * V * 2,
                      1, 1, (c1 - c0) // TN, D // TK, R, TK)
        if fcap:
            prog.add(T_ACT, 255, "softcap", lg, lg, R * V // 2, 1, f32_bits(fcap), f32_bits(1.0 / fcap))
    NP = len(prog.ph)
    assert NP <= NP_MAX, NP

    if emu is not None:
        # gross-failure gate (Frobenius error in 0.01 % units): the device's fexp / reduction order differ from the
        # emulation's, random models included (smoke: 9.4 % h, 10.4 % logits on FireSim, runs 624 / 627)
        htol, ltol = 3000, 2000
        for t in range(S):
            checks.append((f"step {t} h{L} vs emulation", hbuf[t][L], gold["emu_h"][t], R * D, 8, 0, htol))
        checks.append(("logits vs emulation", logits, gold["emu_logits"], S * R * V, 10, 0, ltol))
    if l0g:   # report only (any error passes): bf16 checks scale 8, byte checks allow every byte
        for t in range(S):
            checks.append((f"L0 step {t} qkv vs golden", act[t][0]["qkv"], l0g[f"qkv{t}"][0], R * NQKV, 8, 0, 1000000))
        for k, nm in (("kc", "K^T cache codes"), ("kc_sc", "K^T cache scales"), ("vc", "V cache codes"),
                      ("vc_sc", "V cache scales")):
            n_ = l0g[k][1].size
            checks.append((f"L0 {nm} after {S} appends (bytes)", lw[0][k], l0g[k][0], n_, 8, 1, n_))
        for k, dev, n_ in (("o0", act[0][0]["o"], NB * TMp * hd), ("yattn0", act[0][0]["yattn"], R * D),
                           ("hmid0", act[0][0]["hmid"], R * D), ("gu0", act[0][0]["gu"], R * 2 * F),
                           ("ymlp0", act[0][0]["ymlp"], R * D)):
            checks.append((f"L0 step 0 {k[:-1]} vs golden", dev, l0g[k][0], n_, 8, 0, 1000000))
    if "hf_logits" in gold and mdl["full"]:
        checks.append(("logits vs HF fp32", logits, gold["hf_logits"], S * R * V, 10, 0, 2000))

    pa_ = prog.array().tobytes()
    off = prog_addr - REGION_BASE[3]
    mem.blobs[3][off:off + len(pa_)] = pa_
    mem.write(a.out)
    nfill = sum(1 for p_ in prog.ph if p_[0] == T_FILL)
    full = mdl is not None and mdl["full"]
    with open(a.header, "w") as f:
        f.write(f"// generated by llama_e2e/gen_decode.py --model {a.model}; do not edit\n")
        f.write("#ifndef LLAMA_DATA_H\n#define LLAMA_DATA_H\n#include <stdint.h>\n")
        # LL_M: logits rows (steps x sequences x tokens) for the host's top-1 checks
        for k, v in dict(LL_M=S * R, LL_SK=SKc, LL_P0=P, LL_L=L, LL_D=D, LL_F=F, LL_HQ=Hq, LL_HKV=Hkv,
                         LL_HD=hd, LL_V=V, LL_NP=NP, LL_NFILL=nfill, LL_PW=PW, LL_TN=TN,
                         LL_SYNTHETIC=int(synthetic), LL_STEPS=S, LL_BATCH=B, LL_VERIFY_T=T).items():
            f.write(f"#define {k} {v}u\n")
        f.write(f'#define LL_MODEL "{a.model} decode"\n')
        f.write(f"#define LL_PROG_ADDR 0x{prog_addr:08x}u\n#define LL_PERF_ADDR 0x{perf:08x}u\n")
        f.write(f"#define LL_LOGITS_ADDR 0x{logits:08x}u\n")
        f.write(f"#define LL_LABELS_ADDR 0x{gold.get('labels', 0) if full else 0:08x}u\n")
        f.write(f"#define LL_HF_LOGITS_ADDR 0x{gold.get('hf_logits', 0) if full else 0:08x}u\n")
        f.write(f"#define LL_EMU_LOGITS_ADDR 0x{gold.get('emu_logits', 0):08x}u\n")
        f.write("#define LL_T_GEMM 1u\n#define LL_T_PRENORM 2u\n#define LL_T_ROPE 3u\n#define LL_T_SOFTMAX 4u\n"
                "#define LL_T_EWQ 5u\n#define LL_T_FILL 6u\n#define LL_T_SILU 7u\n#define LL_T_ACT 8u\n#define LL_T_FA 9u\n"
                f"#define LL_T_DROPE {T_DROPE}u\n")
        f.write("#ifdef LL_HOST\n")
        f.write("static const char *const LL_OPS[] = {" + ", ".join(f'"{o}"' for o in OPS) + "};\n")
        f.write(f"#define LL_NOPS {len(OPS)}u\n#define LL_NCHECKS {len(checks)}u\n")
        f.write("#define LL_CPU_ONLY 0\n#define LL_NCSUM 0u\nstatic const uint32_t LL_CSUM[1][3] = {{0, 0, 0}};\n")
        f.write("struct LlCheck { const char *name; uint32_t dev, gold, n; int scale_log2; int bytes; uint32_t tol; };\n")
        f.write("static const LlCheck LL_CHECKS[] = {\n")
        for c in checks:
            name, dev, g, n, sc, kind, tol = c
            f.write(f'  {{"{name}", 0x{dev:08x}u, 0x{g:08x}u, {n}u, {sc}, {kind}, {tol}u}},\n')
        f.write("  {0, 0, 0, 0, 0, 0, 0}\n};\n#endif\n#endif\n")
    used = {r: (mem.arena[r] - REGION_BASE[r]) >> 20 for r in REGION_BASE}
    blob = {r: len(b_) >> 20 for r, b_ in mem.blobs.items()}
    print(f"{a.model} decode: B={B} steps={S} T={T} (QK rows {TMq}, P rows {TMp}) ctx={P} (K cache {SKk}, V cache {SKc}) L={L} phases={NP} "
          f"(fill {nfill}); blob MiB {blob}, used MiB {used}")
    if emu is not None and mdl["logits"] is not None:
        e = np.stack(emu["logits"])
        print(f"emulation: top-1 vs HF {np.mean(e.argmax(-1) == mdl['logits'].argmax(-1)) * 100:.1f}%, "
              f"vs next token {np.mean(e.argmax(-1) == mdl['labels']) * 100:.1f}% "
              f"(HF itself {np.mean(mdl['logits'].argmax(-1) == mdl['labels']) * 100:.1f}%)")


if __name__ == "__main__":
    main()
