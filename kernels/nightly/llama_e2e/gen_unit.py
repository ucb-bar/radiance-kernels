#!/usr/bin/env python3
"""llama_e2e unit tests for bring-up on FireSim (same kernel.cpp / host.cpp as the models):

  cputest   host (Rocket) only, the GPU is never launched: 256 KiB of random words in each GPU region blob; the host
            reads them back through the GPU address hash and compares checksums.  Checks LoadMem of a hashed ELF
            and host access to GPU memory.
  gemmact   gemmini + Muon: GEMMs (operands from the ELF, C bit-exact vs the hardware-exact mx_golden), each
            followed by GELU on Muon (bf16 -> bf16, whole-word stores, one warp per 64-byte line).
  m2g       Muon -> gemmini handoff through global memory: Muon fills an MX A operand (FILL, whole-line stores),
            gemmini multiplies it (C1); control C2 uses the same A from the ELF; C3 uses a Muon-filled A followed by 1 MiB
            of other Muon writes (evicting anything still cached) before the GEMM.  All bit-exact vs mx_golden.
  muontest  Muon only, no gemmini: one phase of each SIMT row op (PRENORM, EWQ, SILU, ROPE, SOFTMAX) on small random
            inputs; every output byte is compared with a numpy golden of the device algorithm.  PRENORM, EWQ and ROPE
            are expected bit-exact; SILU and SOFTMAX use the hardware exp (fexp.h) and may differ in a few codes.
"""
import argparse, os, sys
import numpy as np
sys.argv_saved = sys.argv
import gen_data as g

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["cputest", "muontest", "gemmact", "m2g", "silufuse"], required=True)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--shape", default="32,256,256,4,2,64,256", help="muontest: M,D,F,Hq,Hkv,hd,SK")
ap.add_argument("--fa", action="store_true", help="muontest: ROPE in the ATTN layout (hd 128; Q TM 64, K^T [hd][SK], V [SK][hd])")
ap.add_argument("--out", default="data")
ap.add_argument("--header", default="llama_data.h")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
mem = g.Mem("llama")
prog = g.Prog()
prog_addr = mem.add(3, np.zeros(g.PW * 1024, np.uint32))
checks = []                 # (name, dev, gold, n, scale_log2, bytes)
csum = []

def gelu_golden(c_bits):
    """the device gelu (nightly gelu kernel math) in float32, step by step"""
    f32 = np.float32
    x = g.bf16_to_f32(c_bits)
    z = (f32(0.7978845608) * (x + (f32(0.044715) * x * x) * x)).astype(f32)
    t = (f32(-2.0 * 1.4426950409) * z).astype(f32)
    t = np.minimum(np.maximum(t, f32(-126)), f32(126))
    n = np.floor(t).astype(np.int32)
    fr = (t - n.astype(f32)).astype(f32)
    p = f32(1.8775767e-3)
    for c in (8.9893397e-3, 5.5826318e-2, 2.4015361e-1, 6.9315308e-1, 9.9999994e-1):
        p = (p * fr + f32(c)).astype(f32)
    e = (p.view(np.uint32).astype(np.int64) + (n.astype(np.int64) << 23)).astype(np.uint32).view(f32)
    return g.bf16_bits((x / (f32(1) + e)).astype(f32))


if a.model == "gemmact":
    sys.path.insert(0, str(g.HERE.parents[2] / "lib" / "golden"))
    import golden
    # (batch, M, N, K, TK): the smoke QKV shape (TM 64, one k-step), TM 128 with 4 k-steps, batched K = TK = 64
    shapes = [(1, 64, 512, 256, 256), (1, 128, 512, 1024, 256), (2, 64, 256, 64, 64)]
    tests = []
    for si, (B, M, N, K, TKg) in enumerate(shapes):
        As, Bs, SAs, SBs, Cs = [], [], [], [], []
        for _ in range(B):
            A = golden.rand_fp8(rng, M * K).reshape(M, K)
            Bm = golden.rand_fp8(rng, K * N).reshape(K, N)
            SA = rng.integers(0x7B, 0x83, size=(K // 32, M), dtype=np.uint8)
            SB = rng.integers(0x7B, 0x83, size=(K // 32, N), dtype=np.uint8)
            Cs.append(golden.mx_matmul(A, Bm, SA, SB, M, N, K, tmpdir=str(g.HERE / "_gen" / f"gemmact{si}")))
            ap_, asp_ = g.pack_a(A, SA.T, M, TKg)
            bp_, bsp_ = g.pack_b(Bm, SB, TKg)
            As.append(ap_); SAs.append(asp_); Bs.append(bp_); SBs.append(bsp_)
        C = np.stack(Cs)
        tests.append((B, M, N, K, TKg, [mem.add(1, np.stack(x)) for x in (As, SAs, Bs, SBs)], mem.add(3, C),
                      mem.add(3, gelu_golden(C))))
    perf = mem.alloc(64 * 4 * 1024)
    for si, (B, M, N, K, TKg, (aa, sa, ba, sba), cg, yg) in enumerate(tests):
        c = mem.alloc(B * M * N * 2)
        y = mem.alloc(B * M * N * 2)
        prog.gemm(si, "gemm", aa, sa, M * K, M * K // 32, ba, sba, K * N, K * N // 32, c, M * N * 2, N * 2, M * N * 2,
                  B, 1, N // g.TN, K // TKg, M, TKg)
        prog.add(g.T_ACT, si, "gelu", c, y, B * M * N // 2)
        name = f"B{B} M{M} N{N} K{K} TK{TKg}"
        checks.append((f"C   {name} vs mx_golden", c, cg, B * M * N, 6, 0, 0))
        checks.append((f"gelu {name} vs numpy", y, yg, B * M * N, 6, 0, 100))   # <= 1% Frobenius
elif a.model == "silufuse":
    # gateup GEMM with SILU fused (warps 1.. quantize H as C tiles land): C [M][2F] with [G_j | U_j] per 64 columns
    sys.path.insert(0, str(g.HERE.parents[2] / "lib" / "golden"))
    import golden
    M, K, F, TKg = 128, 1024, 512, 256
    N = 2 * F
    A = golden.rand_fp8(rng, M * K).reshape(M, K)
    Bm = golden.rand_fp8(rng, K * N).reshape(K, N)
    SA = rng.integers(0x7B, 0x83, size=(K // 32, M), dtype=np.uint8)
    SB = rng.integers(0x7B, 0x83, size=(K // 32, N), dtype=np.uint8)
    C = golden.mx_matmul(A, Bm, SA, SB, M, N, K, tmpdir=str(g.HERE / "_gen" / "silufuse"))
    ap_, asp_ = g.pack_a(A, SA.T, M, TKg)
    bp_, bsp_ = g.pack_b(Bm, SB, TKg)
    aa, sa, ba, sba = (mem.add(1, x) for x in (ap_, asp_, bp_, bsp_))
    cg = mem.add(3, C)
    Cf = g.bf16_to_f32(C).reshape(M, F // 64, 2, 64)
    G_, U_ = Cf[:, :, 0].reshape(M, F), Cf[:, :, 1].reshape(M, F)
    hh = (G_ / (np.float32(1) + g.bf(np.exp(g.bf(-G_))))).astype(np.float32) * U_
    hq, hs = g.pack_a(*g.mx_quant(hh), M, TKg)
    hqg, hsg = mem.add(3, hq), mem.add(3, hs)
    perf = mem.alloc(64 * 4 * 1024)
    c = mem.alloc(M * N * 2)
    hc, hsc = mem.alloc(hq.nbytes), mem.alloc(hs.nbytes)
    prog.gemm(0, "gateup", aa, sa, 0, 0, ba, sba, 0, 0, c, 0, N * 2, M * N * 2, 1, 1, N // g.TN, K // TKg, M, TKg,
              fuse=(1, hc, hsc, F, M, M, TKg))
    checks.append(("C vs mx_golden", c, cg, M * N, 6, 0, 0))
    checks.append(("H codes (fused silu)", hc, hqg, hq.nbytes, 8, 1, hq.nbytes // 100))
    checks.append(("H scales (fused silu)", hsc, hsg, hs.nbytes, 8, 1, 0))
elif a.model == "m2g":
    sys.path.insert(0, str(g.HERE.parents[2] / "lib" / "golden"))
    import golden
    M, N, K = 64, 256, 256
    A = np.full((M, K), 0x38, np.uint8)                       # e4m3 1.0
    SA = np.full((K // 32, M), 0x7F, np.uint8)                # 2^0
    Bm = golden.rand_fp8(rng, K * N).reshape(K, N)
    SB = rng.integers(0x7B, 0x83, size=(K // 32, N), dtype=np.uint8)
    C = golden.mx_matmul(A, Bm, SA, SB, M, N, K, tmpdir=str(g.HERE / "_gen" / "m2g"))
    ap_, asp_ = g.pack_a(A, SA.T, M, 256)
    bp_, bsp_ = g.pack_b(Bm, SB, 256)
    a_elf, sa_elf = mem.add(1, ap_), mem.add(1, asp_)
    b_elf, sb_elf = mem.add(1, bp_), mem.add(1, bsp_)
    cg = mem.add(3, C)
    perf = mem.alloc(64 * 4 * 1024)
    a1, s1 = mem.alloc(M * K), mem.alloc(M * K // 32)
    a3, s3 = mem.alloc(M * K), mem.alloc(M * K // 32)
    junk = mem.alloc(1 << 20)
    c1, c2, c3 = (mem.alloc(M * N * 2) for _ in range(3))
    def gemm(si, aa, sa, c):
        prog.gemm(si, "gemm", aa, sa, 0, 0, b_elf, sb_elf, 0, 0, c, 0, N * 2, M * N * 2, 1, 1, N // g.TN, K // 256, M, 256)
    prog.add(g.T_FILL, 0, "init", a1, M * K, 1, 0x38383838)
    prog.add(g.T_FILL, 0, "init", s1, M * K // 32, 1, 0x7F7F7F7F)
    gemm(0, a1, s1, c1)
    gemm(1, a_elf, sa_elf, c2)
    prog.add(g.T_FILL, 2, "init", a3, M * K, 1, 0x38383838)
    prog.add(g.T_FILL, 2, "init", s3, M * K // 32, 1, 0x7F7F7F7F)
    prog.add(g.T_FILL, 2, "init", junk, 1 << 20, 1, 0x5A5A5A5A)
    gemm(2, a3, s3, c3)
    checks.append(("C1 Muon-written A, GEMM right after", c1, cg, M * N, 6, 0, 0))
    checks.append(("C2 control: same A from the ELF", c2, cg, M * N, 6, 0, 0))
    checks.append(("C3 Muon-written A, then 1 MiB of other writes", c3, cg, M * N, 6, 0, 0))
elif a.model == "cputest":
    for r in (1, 2, 3):
        w = rng.integers(0, 1 << 32, size=(256 << 10) // 4, dtype=np.uint64).astype(np.uint32)
        addr = mem.add(r, w)
        csum.append((addr, w.size, int(np.sum(w, dtype=np.uint64)) & 0xFFFFFFFF))
    perf = mem.alloc(4096)
else:
    M, D, F, Hq, Hkv, hd, SK = (int(v) for v in a.shape.split(","))
    P0, Gq, Dq, Dkv = SK - M, Hq // Hkv, Hq * hd, Hkv * hd
    NQKV = Dq + 2 * Dkv
    TK, TM = 256, M
    bfb = lambda x: g.bf16_bits(x)
    eps = np.float32(1e-5)
    # inputs
    x = rng.normal(0, 1, (M, D)); add = rng.normal(0, 1, (M, D)); gam = rng.uniform(0.5, 1.5, D)
    o = rng.normal(0, 2, (M, Dq)); gu = rng.normal(0, 2, (M, 2 * F)); qkv = rng.normal(0, 1, (M, NQKV))
    S = rng.normal(0, 4, (M, SK))
    cos, sin = g.rope_tables(M, P0, hd, 10000.0)
    xa, adda, gama = mem.add(3, bfb(x)), mem.add(3, bfb(add)), mem.add(3, bfb(gam))
    oa, gua, qkva, sa = mem.add(3, bfb(o)), mem.add(3, bfb(gu)), mem.add(3, bfb(qkv)), mem.add(3, bfb(S))
    cosa, sina = mem.add(3, cos), mem.add(3, sin)
    # K / V caches start zeroed (only the chunk keys are written)
    kcache = mem.add(3, np.zeros(Hkv * SK * hd, np.uint8)); ksc = mem.add(3, np.zeros(Hkv * SK * hd // 32, np.uint8))
    vcache = mem.add(3, np.zeros(Hkv * SK * hd, np.uint8)); vsc = mem.add(3, np.zeros(Hkv * SK * hd // 32, np.uint8))

    # goldens (device algorithms)
    h = g.bf(g.bf(x) + g.bf(add))
    ss = np.sum(h * h, axis=1, dtype=np.float32)
    inv = (np.float32(1) / np.sqrt(ss / np.float32(D) + eps)).astype(np.float32)
    y = (h * inv[:, None]).astype(np.float32) * g.bf(gam)[None, :]
    gold = {}
    gold["h"] = bfb(h)
    gold["xn"], gold["xn_sc"] = g.pack_a(*g.mx_quant(y), TM, TK)
    gold["oq"], gold["oq_sc"] = g.pack_a(*g.mx_quant(g.bf(o)), TM, TK)
    G_, U_ = g.bf(gu[:, :F]), g.bf(gu[:, F:])
    hh = (G_ / (np.float32(1) + g.bf(np.exp(g.bf(-G_))))).astype(np.float32) * U_
    gold["hq"], gold["hq_sc"] = g.pack_a(*g.mx_quant(hh), TM, TK)
    qb = g.bf(qkv)
    Q = g.rope(qb[:, :Dq].reshape(M, Hq, hd), cos, sin)
    Kn = g.rope(qb[:, Dq:Dq + Dkv].reshape(M, Hkv, hd), cos, sin)
    Vn = qb[:, Dq + Dkv:].reshape(M, Hkv, hd)
    qc, qs = g.mx_quant(Q)                                   # [M][Hq][hd], [M][Hq][hd/32]
    qa_l, qsc_l = [], []
    for grp in range(Hkv):
        rows = np.concatenate([qc[:, grp * Gq + j] for j in range(Gq)], 0)          # row (j, t)
        rsc = np.concatenate([qs[:, grp * Gq + j] for j in range(Gq)], 0)
        pa_, ps_ = g.pack_a(rows, rsc, 64 if a.fa else TM, hd)
        qa_l.append(pa_); qsc_l.append(ps_)
    gold["q"], gold["q_sc"] = np.stack(qa_l), np.stack(qsc_l)
    kb_l, kbs_l, vb_l, vbs_l = [], [], [], []
    for hk in range(Hkv):
        kc = np.zeros((SK, hd), np.uint8); kcs = np.zeros((SK, hd // 32), np.uint8)
        c_, s_ = g.mx_quant(Kn[:, hk]); kc[P0:], kcs[P0:] = c_, s_
        vc = np.zeros((hd, SK), np.uint8); vcs = np.zeros((hd, SK // 32), np.uint8)
        c_, s_ = g.mx_quant(Vn[:, hk].T); vc[:, P0:], vcs[:, P0 // 32:] = c_, s_
        if a.fa:   # K^T [hd][SK], scales [SK/128][hd/32][128]; V [SK][hd], scales [SK/128][4][hd]
            kb_l.append(kc.T.copy()); kbs_l.append(kcs.reshape(SK // 128, 128, hd // 32).transpose(0, 2, 1).copy())
            vb_l.append(vc.T.copy()); vbs_l.append(vcs.T.reshape(SK // 128, 4, hd).copy())
            continue
        b_, bs_ = g.pack_b(kc.T, kcs.T, hd); kb_l.append(b_); kbs_l.append(bs_)
        b_, bs_ = g.pack_b(vc.T, vcs.T, TK); vb_l.append(b_); vbs_l.append(bs_)
    gold["k"], gold["k_sc"], gold["v"], gold["v_sc"] = np.stack(kb_l), np.stack(kbs_l), np.stack(vb_l), np.stack(vbs_l)
    scale = np.float32(1.0 / np.sqrt(hd))
    Sb = g.bf(S)
    mask = np.arange(SK)[None, :] <= (P0 + np.arange(M))[:, None]
    z = np.where(mask, Sb * scale, -np.inf).astype(np.float32)
    m = np.max(z, axis=1, keepdims=True)
    e = np.where(mask, g.bf(np.exp(g.bf(Sb * scale - m))), 0).astype(np.float32)
    p = e * (np.float32(1) / np.sum(e, axis=1, keepdims=True, dtype=np.float32))
    gold["p"], gold["p_sc"] = g.pack_a(*g.mx_quant(p), TM, TK)
    ga = {k: mem.add(3, v) for k, v in gold.items()}

    # outputs (arena)
    out = {k: mem.alloc(np.asarray(v).nbytes) for k, v in gold.items() if k not in ("k", "k_sc", "v", "v_sc")}
    out.update(k=kcache, k_sc=ksc, v=vcache, v_sc=vsc)
    perf = mem.alloc(64 * 4 * 1024)
    E = g.f32_bits(1e-5)
    prog.add(g.T_PRENORM, 0, "norm1", xa, adda, out["h"], gama, out["xn"], out["xn_sc"], D, M, TM, TK, E)
    prog.add(g.T_EWQ, 0, "oquant", oa, Dq, 0, Dq, M, out["oq"], out["oq_sc"], TM, TK)
    prog.add(g.T_SILU, 0, "silu", gua, 2 * F, 1, F, M, out["hq"], out["hq_sc"], TM, TK)
    prog.add(g.T_ROPE, 0, "rope", qkva, NQKV, Hq, Hkv, hd, cosa, sina, out["q"], out["q_sc"], Gq * M * hd,
             Gq * M * hd // 32, kcache, ksc, SK * hd, SK * hd // 32, vcache, vsc, SK * hd, SK * hd // 32, P0, M, SK,
             64 if a.fa else TM, int(a.fa))
    prog.add(g.T_SOFTMAX, 0, "softmax", sa, M * SK * 2, SK, M, 1, out["p"], out["p_sc"], M * SK, M * SK // 32,
             g.f32_bits(1.0 / np.sqrt(hd)), P0, M, TM, TK)
    for k in gold:
        n = np.asarray(gold[k]).nbytes
        tol = n // 100 if k in ("hq", "p") else 0          # silu / softmax codes go through the hardware exp
        checks.append((f"{k} ({'bf16' if k == 'h' else 'bytes'})", out[k], ga[k], n // 2 if k == "h" else n, 8,
                       0 if k == "h" else 1, tol))

pa_ = prog.array().tobytes() if prog.ph else b""
mem.blobs[3][prog_addr - g.REGION_BASE[3]:prog_addr - g.REGION_BASE[3] + len(pa_)] = pa_
mem.write(a.out)
with open(a.header, "w") as f:
    f.write(f"// generated by llama_e2e/gen_unit.py --model {a.model}; do not edit\n")
    f.write("#ifndef LLAMA_DATA_H\n#define LLAMA_DATA_H\n#include <stdint.h>\n")
    shp = dict(zip("M D F Hq Hkv hd SK".split(), (int(v) for v in a.shape.split(","))))
    for k, v in dict(LL_M=shp["M"], LL_SK=shp["SK"], LL_P0=shp["SK"] - shp["M"], LL_L=1, LL_D=shp["D"], LL_F=shp["F"],
                     LL_HQ=shp["Hq"], LL_HKV=shp["Hkv"], LL_HD=shp["hd"],
                     LL_V=0, LL_NP=len(prog.ph), LL_NFILL=0, LL_PW=g.PW, LL_TN=g.TN, LL_SYNTHETIC=0).items():
        f.write(f"#define {k} {v}u\n")
    f.write(f'#define LL_MODEL "{a.model}"\n')
    f.write(f"#define LL_PROG_ADDR 0x{prog_addr:08x}u\n#define LL_PERF_ADDR 0x{perf:08x}u\n")
    f.write("#define LL_LOGITS_ADDR 0u\n#define LL_LABELS_ADDR 0u\n#define LL_HF_LOGITS_ADDR 0u\n#define LL_EMU_LOGITS_ADDR 0u\n")
    f.write("#define LL_T_GEMM 1u\n#define LL_T_PRENORM 2u\n#define LL_T_ROPE 3u\n#define LL_T_SOFTMAX 4u\n"
            "#define LL_T_EWQ 5u\n#define LL_T_FILL 6u\n#define LL_T_SILU 7u\n#define LL_T_ACT 8u\n#define LL_T_FA 9u\n")
    f.write("#ifdef LL_HOST\n")
    f.write("static const char *const LL_OPS[] = {" + ", ".join(f'"{o}"' for o in g.OPS) + "};\n")
    f.write(f"#define LL_NOPS {len(g.OPS)}u\n#define LL_NCHECKS {len(checks)}u\n")
    f.write(f"#define LL_CPU_ONLY {1 if a.model == 'cputest' else 0}\n#define LL_NCSUM {len(csum)}u\n")
    f.write("static const uint32_t LL_CSUM[][3] = {" + ", ".join(f"{{0x{x:08x}u, {n}u, 0x{c:08x}u}}" for x, n, c in csum)
            + (", " if csum else "") + "{0, 0, 0}};\n")
    f.write("struct LlCheck { const char *name; uint32_t dev, gold, n; int scale_log2; int bytes; uint32_t tol; };\n")
    f.write("static const LlCheck LL_CHECKS[] = {\n")
    for name, dev, gd, n, sc, b, tol in checks:
        f.write(f'  {{"{name}", 0x{dev:08x}u, 0x{gd:08x}u, {n}u, {sc}, {b}, {tol}u}},\n')
    f.write("  {0, 0, 0, 0, 0, 0, 0}\n};\n#endif\n#endif\n")
print(f"{a.model}: phases={len(prog.ph)} checks={len(checks)} checksummed regions={len(csum)}")
