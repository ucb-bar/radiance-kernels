#!/usr/bin/env python3
"""llama_e2e/gen_vla_dec.py: SmolVLA's action expert alone ("decode"), a batch of B observations.

The vision encoder and the SmolLM2 prefix ran before (gen_vla.py's prefill): each sample's prefix K / V caches (the
16 layers' post-RoPE keys and values, keys 0..255) and the cross-attention K' / V' of the odd expert layers are inputs
here, in the device layouts the prefill leaves them in (from the quantized emulation, so they are what the device's
prefill produces up to its own rounding).  The device runs the 10 flow-matching steps for all B samples at once:

  rows     B x 64 (50 actions + padding per sample) = 1024: every dense GEMM (time MLP, q/k/v, o, gate/up, down,
           action out) is one GEMM over the batch
  self     ROPE with d[31] = 64 (rope_fast.h): sample s's Q -> attention batches s 5 + g, its suffix keys / values ->
           keys 256.. of sample s's caches; QK / SOFTMAX (the prefill's mask tables, row t = r % 64) / PV batched over
           B x 5 KV heads; GATHERQ back to [1024][15 x 64]
  cross    Q only, over sample s's K' / V'
  EULER    a warp per sample (d[15] = B)

Checks: x after step 1 and the actions of every sample vs the emulation and vs lerobot (fp32), gen_vla.py's
tolerances.  Sample b: gen_vla.load_reference(seed=b) (its own images, state and noise; the same instruction).
"""
import argparse, math, sys
import numpy as np

import gen_data as g
import gen_vla as gv
from gen_data import bf16_bits, mx_quant, f32_bits

F32 = np.float32
PW, TN, TK = gv.PW, gv.TN, gv.TK
T_GEMM, T_PRENORM, T_ROPE, T_SOFTMAX, T_FILL, T_SILU = gv.T_GEMM, gv.T_PRENORM, gv.T_ROPE, gv.T_SOFTMAX, gv.T_FILL, gv.T_SILU
T_GATHERQ, T_EULER, NOROW, OP, OPS = gv.T_GATHERQ, gv.T_EULER, gv.NOROW, gv.OP, gv.OPS
RING = 4                         # Muon-read per-layer buffers: a ring of RING sets (as gen_data.py's S ring)


def build(R, samples, outdir, header):
    sd, P = R["sd"], R["P"]
    E = P + "lm_expert.layers."
    B = len(samples)
    mem = g.Mem("llama")
    prog = gv.Prog()
    nl = len(R["lang_ids"])
    Lv = 192 + nl + 1
    Mp, Ms, NA = 256, 64, 50
    MB = B * Ms                                             # batch rows
    TME = 128                                               # dense GEMM row tile
    Dp, De = 1024, 768
    SKe = 512
    NBt = B * 5                                             # attention batches
    eps_e = f32_bits(R["eps_e"])
    s18 = f32_bits(1 / 8)

    def wq(w, n, k):
        codes, sc = mx_quant(gv.pad2(np.asarray(w, F32), n, k))
        bt, bs = g.pack_b(np.ascontiguousarray(codes.T), np.ascontiguousarray(sc.T), TK)
        return mem.add_any(bt), mem.add_any(bs)

    def vec(x, n, f32=False):
        v = np.zeros(n, F32)
        v[:len(x)] = x
        return mem.add(3, v if f32 else bf16_bits(v))

    def cs(pos, hd=64):
        c, s_ = gv.rope_tables(pos, hd)
        return mem.add(3, c), mem.add(3, s_)

    prog_addr = mem.add(3, np.zeros(PW * 3072, np.uint32))

    # ---------------- weights (expert only) ----------------
    w = lambda n: sd[n]
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
        ew.append(d)
    g_efinal = vec(sd[P + "lm_expert.norm.weight"], De)
    w_ai, b_ai = sd["model.action_in_proj.weight"], sd["model.action_in_proj.bias"]
    w_in, b_in = sd["model.action_time_mlp_in.weight"], sd["model.action_time_mlp_in.bias"]
    W_fold = wq((w_in[:, :720].astype(np.float64) @ w_ai.astype(np.float64)).astype(F32), De, 256)
    c_base = (w_in[:, :720].astype(np.float64) @ b_ai.astype(np.float64) + b_in).astype(F32)
    c_steps = [vec((c_base + w_in[:, 720:].astype(np.float64) @ t.astype(np.float64)).astype(F32), De) for t in R["temb"]]
    W_tout, b_tout = wq(sd["model.action_time_mlp_out.weight"], De, De), vec(sd["model.action_time_mlp_out.bias"], De)
    W_aout, b_aout = wq(sd["model.action_out_proj.weight"], 64, De), vec(sd["model.action_out_proj.bias"], 32, f32=True)

    # ---------------- per-sample caches (device layouts of gen_vla.py's prefill) ----------------
    # self layers: K^T [512][64] pretiled (TK 64) per (sample, KV head), keys 0..255 = the prefix; V [512][64] (TK 256)
    # cross layers: K'^T / V' over the 256 prefix rows
    kc, vc, xkc, xvc = {}, {}, {}, {}
    for l in range(R["n_expert"]):
        if ew[l]["self"]:
            K = np.zeros((B, 5, SKe, 64), F32)
            V = np.zeros((B, 5, SKe, 64), F32)
            for b, smp in enumerate(samples):
                K0, V0 = smp["caches"][l]
                K[b, :, :Mp], V[b, :, :Mp] = K0.transpose(1, 0, 2), V0.transpose(1, 0, 2)
            NK = SKe
        else:
            K = np.stack([smp["cross"][l][0].transpose(1, 0, 2) for smp in samples]).astype(F32)   # [B][5][256][64]
            V = np.stack([smp["cross"][l][1].transpose(1, 0, 2) for smp in samples]).astype(F32)
            NK = Mp
        kcodes, ksc = mx_quant(K)                                    # along hd per key
        vcodes, vsc = mx_quant(V.transpose(0, 1, 3, 2))              # along keys per d
        kb, kbs, vb, vbs = [], [], [], []
        for b in range(B):
            for h in range(5):
                x_, y_ = g.pack_b(kcodes[b, h].T, ksc[b, h].T, 64); kb.append(x_); kbs.append(y_)
                x_, y_ = g.pack_b(vcodes[b, h].T, vsc[b, h].T, TK); vb.append(x_); vbs.append(y_)
        pair = ((mem.add_any(np.stack(kb)), mem.add_any(np.stack(kbs))),
                (mem.add_any(np.stack(vb)), mem.add_any(np.stack(vbs))))
        assert np.stack(kb)[0].size == NK * 64 and np.stack(vb)[0].size == NK * 64
        if ew[l]["self"]:
            kc[l], vc[l] = pair
        else:
            xkc[l], xvc[l] = pair

    # ---------------- inputs, tables, references ----------------
    noise = np.zeros((MB, 32), F32)
    for b, smp in enumerate(samples):
        noise[b * Ms:b * Ms + NA] = smp["noise"]
    x_bufs = [mem.add(3, noise)]
    zeros_e = mem.add(3, np.zeros((MB, De), np.uint16))
    ht = np.full((MB, 16), NOROW, np.uint32)                 # o_proj operand row b 64 + t, head h <- O row
    for b in range(B):
        for h in range(15):
            ht[b * Ms:(b + 1) * Ms, h] = (b * 5 + h // 3) * 192 + (h % 3) * 64 + np.arange(Ms)
    htab_e = mem.add(3, ht.reshape(-1))
    scos = cs(np.tile(np.concatenate([Lv + np.arange(NA), np.zeros(Ms - NA)]), B))
    ccos = cs(np.tile(np.concatenate([np.arange(NA), np.zeros(Ms - NA)]), B))
    stab = mem.add(3, np.array([[Lv - 1, 256 + min(t, NA - 1)] for t in range(Ms)], np.uint32).reshape(-1))
    ctab = mem.add(3, np.array([[Lv - 1, 0] for t in range(Ms)], np.uint32).reshape(-1))
    pad = lambda x: np.concatenate([np.pad(np.asarray(s_, F32), ((0, Ms - len(s_)), (0, 0))) for s_ in x])
    gold = {k: mem.add(3, bf16_bits(v)) for k, v in (
        ("x1", pad([smp["x1"][:NA] for smp in samples])), ("x10", pad([smp["actions"] for smp in samples])),
        ("ref_x10", pad([smp["ref_actions"] for smp in samples])))}

    # ---------------- arena ----------------
    al = mem.alloc

    def amx(M, K):
        return al(M * K), al(M * K // 32)
    A_x = amx(MB, 256)
    A_e, A_t, A_ef, A_eo = amx(MB, De), amx(MB, De), amx(MB, 2048), amx(MB, Dp)
    Qe = amx(NBt * 192, 64)
    Pe = amx(NBt * 192, 512)
    fills = [(A_x[0], MB * 256), (A_x[1], MB * 256 // 32), (Qe[0], NBt * 192 * 64), (Qe[1], NBt * 192 * 2),
             (Pe[0], NBt * 192 * 512), (Pe[1], NBt * 192 * 16)]
    for l in kc:   # the suffix keys 256..511 of the self caches start at 0 (blob zeros); nothing else to clear
        pass
    for a_, nb in fills:
        prog.add(T_FILL, 255, "init", a_, nb, 1, 0)
    nfill = len(prog.ph)
    perf = al(64 * 4 * 3072)
    ring = lambda nbytes: [al(nbytes) for _ in range(RING)]
    R_hn, R_hm, R_y, R_ym = (ring(MB * De * 2) for _ in range(4))
    R_qkv, R_O = ring(MB * 1600 * 2), ring(NBt * 192 * 64 * 2)
    R_S, R_gu = ring(NBt * 192 * SKe * 2), ring(MB * 4096 * 2)
    R_t1, R_emb, R_hf, R_vout = ring(MB * De * 2), ring(MB * De * 2), ring(MB * De * 2), ring(MB * 64 * 2)
    xbf = [al(MB * 32 * 2) for _ in range(len(R["temb"]) + 1)]
    for s_ in range(len(R["temb"])):
        x_bufs.append(al(MB * 32 * 4))

    # ---------------- denoising ----------------
    dt = f32_bits(R["dt"])
    prog.add(T_EULER, 255, "euler", x_bufs[0], 0, 0, f32_bits(0.0), x_bufs[0], A_x[0], A_x[1], TME, TK, 1, NA, xbf[0], 0, B)
    gl = 0                                                  # global layer counter (ring index)
    for s_, cst in enumerate(c_steps):
        r_ = s_ % RING
        t1 = R_t1[r_]
        prog.dense(s_, "tin", A_x, W_fold, t1, De * 2, MB, De, 256, TME)
        prog.add(T_SILU, s_, "tsilu", t1, De, 0, De, MB, A_t[0], A_t[1], TME, TK, 2, cst, 0)
        emb = R_emb[r_]
        prog.dense(s_, "tout", A_t, W_tout, emb, De * 2, MB, De, De, TME)
        h, add, bias = zeros_e, emb, b_tout
        for l, d in enumerate(ew):
            k_ = gl % RING
            gl += 1
            hn = R_hn[k_]
            prog.add(T_PRENORM, l, "enorm1", h, add, hn, d["g1"], A_e[0], A_e[1], De, MB, TME, TK, eps_e, 0, 0, 720,
                     0, bias, 0)
            bias = 0
            if d["self"]:
                qkv = R_qkv[k_]
                prog.dense(l, "eqkv", A_e, d["qkv"], qkv, 3200, MB, 1600, De, TME)
                prog.add(T_ROPE, l, "erope", qkv, 1600, 15, 5, 64, scos[0], scos[1], Qe[0], Qe[1], 192 * 64, 192 * 2,
                         kc[l][0], kc[l][1], SKe * 64, SKe * 2, vc[l][0], vc[l][1], SKe * 64, SKe * 2, 256, MB, SKe, 64,
                         0, 0, 0, 0, 3, 0, Ms)                                   # d[31]: 64 rows per sample
                kb, kbs, vb, vbs, SK, tab, b2, NT, KSv = kc[l][0], kc[l][1], vc[l][0], vc[l][1], SKe, stab, 256, 8, 2
                kb_bs, ksc_bs = SKe * 64, SKe * 2
            else:
                qq = R_qkv[k_]
                prog.dense(l, "eqkv", A_e, d["q"], qq, 1920, MB, 960, De, TME)
                prog.add(T_ROPE, l, "erope", qq, 960, 15, 0, 64, ccos[0], ccos[1], Qe[0], Qe[1], 192 * 64, 192 * 2,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, MB, Mp, 64, 0, 0, 0, 0, 3, 0, Ms)
                kb, kbs, vb, vbs, SK, tab, b2, NT, KSv = xkc[l][0], xkc[l][1], xvc[l][0], xvc[l][1], Mp, ctab, NOROW, 4, 1
                kb_bs, ksc_bs = Mp * 64, Mp * 2
            S = R_S[k_]
            prog.gemm(l, "eqk", Qe[0], Qe[1], 192 * 64, 192 * 2, kb, kbs, kb_bs, ksc_bs, S, 192 * SK * 2, SK * 2,
                      64 * SK * 2, NBt, 3, NT, 1, 64, 64)
            prog.add(T_SOFTMAX, l, "esoftmax", S, 192 * SK * 2, SK, 192, NBt, Pe[0], Pe[1], 192 * 512, 192 * 16, s18, 0,
                     Ms, 64, TK, 0, 0, tab, b2)
            O = R_O[k_]
            prog.gemm(l, "epv", Pe[0], Pe[1], 192 * 512, 192 * 16, vb, vbs, kb_bs, ksc_bs, O, 192 * 64 * 2, 128,
                      64 * 128, NBt, 3, 1, KSv, 64, TK)
            prog.add(T_GATHERQ, l, "egather", O, 128, htab_e, 16, 64, MB, A_eo[0], A_eo[1], TME, TK)
            y = R_y[k_]
            prog.dense(l, "eo", A_eo, d["o"], y, De * 2, MB, De, Dp, TME)
            hm = R_hm[k_]
            prog.add(T_PRENORM, l, "enorm2", hn, y, hm, d["g2"], A_e[0], A_e[1], De, MB, TME, TK, eps_e, 0, 0, 720,
                     0, 0, 0)
            gu = R_gu[k_]
            prog.dense(l, "egu", A_e, d["gu"], gu, 8192, MB, 4096, De, TME)
            prog.add(T_SILU, l, "esilu", gu, 4096, 0, 2048, MB, A_ef[0], A_ef[1], TME, TK, 0, 0, 0)
            ym = R_ym[k_]
            prog.dense(l, "edown", A_ef, d["d"], ym, De * 2, MB, De, 2048, TME)
            h, add = hm, ym
        hf = R_hf[r_]
        prog.add(T_PRENORM, 255, "efinal", h, add, hf, g_efinal, A_e[0], A_e[1], De, MB, TME, TK, eps_e, 0, 0, 720,
                 0, 0, 0)
        vout = R_vout[r_]
        prog.dense(255, "aout", A_e, W_aout, vout, 128, MB, 64, De, TME)
        prog.add(T_EULER, s_, "euler", x_bufs[s_], vout, 128, dt, x_bufs[s_ + 1], A_x[0], A_x[1], TME, TK, 1, NA,
                 xbf[s_ + 1], b_aout, B)
    NP = len(prog.ph)
    assert NP <= 3072, NP

    checks = [("x after step 1 vs emulation", xbf[1], gold["x1"], MB * 32, 12, 0, 300),
              ("actions vs emulation", xbf[-1], gold["x10"], MB * 32, 12, 0, 1800),
              ("actions vs fp32 reference", xbf[-1], gold["ref_x10"], MB * 32, 12, 0, 1800)]
    pa_ = prog.array().tobytes()
    off = prog_addr - g.REGION_BASE[3]
    mem.blobs[3][off:off + len(pa_)] = pa_
    mem.write(outdir)
    with open(header, "w") as f:
        f.write("// generated by llama_e2e/gen_vla_dec.py; do not edit\n")
        f.write("#ifndef LLAMA_DATA_H\n#define LLAMA_DATA_H\n#include <stdint.h>\n")
        # LL_M: action rows; the host's per-layer figure counts one expert layer per denoising step
        for k, v_ in dict(LL_M=MB, LL_SK=SKe, LL_P0=0, LL_L=R["n_expert"], LL_D=720, LL_F=2048, LL_HQ=15, LL_HKV=5,
                          LL_HD=64, LL_V=0, LL_NP=NP, LL_NFILL=nfill, LL_PW=PW, LL_TN=TN, LL_SYNTHETIC=0,
                          LL_STEPS=len(R["temb"]), LL_BATCH=B).items():
            f.write(f"#define {k} {v_}u\n")
        f.write('#define LL_MODEL "smolvla expert decode"\n')
        f.write(f"#define LL_PROG_ADDR 0x{prog_addr:08x}u\n#define LL_PERF_ADDR 0x{perf:08x}u\n")
        f.write("#define LL_LOGITS_ADDR 0x00000000u\n#define LL_LABELS_ADDR 0x00000000u\n")
        f.write("#define LL_HF_LOGITS_ADDR 0x00000000u\n#define LL_EMU_LOGITS_ADDR 0x00000000u\n")
        for n_, v_ in (("GEMM", 1), ("PRENORM", 2), ("ROPE", 3), ("SOFTMAX", 4), ("EWQ", 5), ("FILL", 6), ("SILU", 7),
                       ("ACT", 8), ("FA", 9), ("AFFINE", 10), ("GATHERQ", 11), ("EULER", 12)):
            f.write(f"#define LL_T_{n_} {v_}u\n")
        f.write("#ifdef LL_HOST\n")
        f.write("#define LL_HEX_N 0u\nstruct LlHex { const char *name; uint32_t dev, gold, n; };\n")
        f.write("static const LlHex LL_HEX[] = {{0, 0, 0, 0}};\n")
        f.write("static const char *const LL_OPS[] = {" + ", ".join(f'"{o}"' for o in OPS) + "};\n")
        f.write(f"#define LL_NOPS {len(OPS)}u\n#define LL_NCHECKS {len(checks)}u\n")
        f.write("#define LL_CPU_ONLY 0\n#define LL_NCSUM 0u\nstatic const uint32_t LL_CSUM[1][3] = {{0, 0, 0}};\n")
        f.write("struct LlCheck { const char *name; uint32_t dev, gold, n; int scale_log2; int bytes; uint32_t tol; };\n")
        f.write("static const LlCheck LL_CHECKS[] = {\n")
        for c in checks:
            f.write(f'  {{"{c[0]}", 0x{c[1]:08x}u, 0x{c[2]:08x}u, {c[3]}u, {c[4]}, {c[5]}, {c[6]}u}},\n')
        f.write("  {0, 0, 0, 0, 0, 0, 0}\n};\n#endif\n#endif\n")
    used = {r: (mem.arena[r] - g.REGION_BASE[r]) >> 20 for r in mem.arena}
    blob = {r: len(b_) >> 20 for r, b_ in mem.blobs.items()}
    print(f"smolvla expert decode: B={B} phases={NP} (init {nfill}); blob MiB {blob}, used MiB {used}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--out", default="data")
    ap.add_argument("--header", default="llama_data.h")
    ap.add_argument("--layers", type=int, default=None,
                    help="truncate the stacks to N layers each, as gen_vla.py --layers (N >= 2 keeps a cross-attention "
                         "layer); the fp32 reference is then the numpy pipeline instead of lerobot")
    a = ap.parse_args()
    samples = []
    R = None
    for b in range(a.batch):
        print(f"sample {b}: lerobot reference and the quantized pipeline ...", flush=True)
        Rb = gv.load_reference(seed=b, policy=R["policy"] if R else None)
        if R is None:
            R = Rb
            if a.layers:
                for k in ("n_vision", "n_text", "n_expert"):
                    R[k] = min(R[k], a.layers)
        else:   # same weights and instruction: keep sample 0's dict, this sample's inputs and reference
            R = dict(R)
            for k in ("images", "state", "noise", "actions"):
                R[k] = Rb[k]
        del Rb
        emu = gv.forward(R, q=True)
        gv.VTRACE.clear()                                   # vision layer-0 traces (gen_vla.py's checks), not used here
        ref = gv.forward(R, q=False)["actions"] if a.layers else R["actions"]   # truncated: no lerobot model to match
        gv.VTRACE.clear()
        gv.compare(f"sample {b} actions (emulation vs {'fp32 numpy' if a.layers else 'lerobot'})", emu["actions"], ref)
        samples.append(dict(caches=emu["caches"], cross=emu["cross"], noise=R["noise"], x1=emu["x"][1],
                            actions=emu["actions"], ref_actions=ref))
        assert emu["Lv"] == 192 + len(R["lang_ids"]) + 1
    build(R, samples, a.out, a.header)


if __name__ == "__main__":
    main()
