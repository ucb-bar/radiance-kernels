// llama_e2e host (Rocket, integer only): launch, step the clusters through the phase program, then report the
// per-phase cycle counts the GPU stamped (mcycle at each phase's start and end, per cluster) and check results.
//
// Phase handshake (printBuf slots 20 / 21): GO = p + 1 lets every cluster start phase p; each posts DONE = p + 1.
// With several clusters the host releases phase p + 1 only after every cluster posted DONE = p + 1, which is the
// cross-cluster barrier between phases.  With one cluster GO = NP up front: no host round trips.
#define LL_HOST 1
#include <nightly/host.h>
#include "llama_data.h"

static constexpr uint32_t NC = NIGHTLY_CLUSTERS;
static constexpr uint32_t PB_GO = 20, PB_DONE = 21;

static inline uint32_t rd32(uint32_t dev) { return *nightly_gpu_ptr<uint32_t>(dev); }
static inline uint32_t desc(uint32_t p, uint32_t w) { return rd32(LL_PROG_ADDR + (p * LL_PW + w) * 4); }
static inline uint32_t stamp(uint32_t p, uint32_t c, uint32_t e) { return rd32(LL_PERF_ADDR + ((p * NC + c) * 16 + e) * 4); }

static uint64_t gemm_macs(uint32_t p) {
  if (desc(p, 0) == LL_T_FA) return 2ull * desc(p, 9) * desc(p, 11) * desc(p, 12) * 128;   // QK + PV: H SQ SK d
  if (desc(p, 0) != LL_T_GEMM) return 0;
  // batch * MT * NT * KS * TM * TN * TK
  return (uint64_t)desc(p, 14) * desc(p, 15) * desc(p, 16) * desc(p, 17) * desc(p, 18) * LL_TN * desc(p, 19);
}

// bf16 bits -> order-preserving integer
static inline int32_t bf_key(uint16_t b) { return (b & 0x8000) ? -(int32_t)(b & 0x7FFF) : (int32_t)b; }

static void put_pct(uint64_t bp) { nh_putfix(bp, 2); nh_putc('%'); }

int main() {
  nh_puts("llama_e2e " LL_MODEL ": M="); nh_putu(LL_M); nh_puts(" tokens at positions "); nh_putu(LL_P0);
  nh_puts(".."); nh_putu(LL_P0 + LL_M - 1); nh_puts(" (KV cache "); nh_putu(LL_P0); nh_puts(", SK="); nh_putu(LL_SK);
  nh_puts(") layers="); nh_putu(LL_L); nh_puts(" D="); nh_putu(LL_D); nh_puts(" F="); nh_putu(LL_F);
  nh_puts(" heads="); nh_putu(LL_HQ); nh_putc('/'); nh_putu(LL_HKV); nh_puts(" hd="); nh_putu(LL_HD);
  nh_puts(" vocab="); nh_putu(LL_V); nh_puts(" phases="); nh_putu(LL_NP); nh_puts(" (init "); nh_putu(LL_NFILL);
  nh_puts(") clusters="); nh_putu(NC); nh_puts("\n"); nh_flush();

#if LL_CPU_ONLY
  {   // cputest: the GPU is never launched; read every blob back through the GPU address hash
    int ok = 1;
    for (uint32_t i = 0; i < LL_NCSUM; i++) {
      const uint64_t t0 = nightly_rdcycle();
      uint32_t sum = 0;
      for (uint32_t w = 0; w < LL_CSUM[i][1]; w++) sum += rd32(LL_CSUM[i][0] + 4 * w);
      const uint64_t t1 = nightly_rdcycle();
      nh_puts("region blob at "); nh_putx(LL_CSUM[i][0]); nh_puts(": "); nh_putu(LL_CSUM[i][1] * 4ull); nh_puts(" bytes checksum ");
      nh_putx(sum); nh_puts(sum == LL_CSUM[i][2] ? " OK" : " MISMATCH (expected "); if (sum != LL_CSUM[i][2]) { nh_putx(LL_CSUM[i][2]); nh_puts(")"); }
      nh_puts(" host_cycles="); nh_putu(t1 - t0); nh_puts("\n"); nh_flush();
      ok &= sum == LL_CSUM[i][2];
    }
    nh_puts(ok ? "PASS\n" : "FAIL\n"); nh_flush();
    exit(ok ? 0 : 1);
  }
#endif
  for (uint32_t cl = 0; cl < NC; cl++) { nightly_pb_write(cl, PB_GO, NC == 1 ? LL_NP : 0); nightly_pb_write(cl, PB_DONE, 0); }
  nightly_host_launch(NC);
  if (NC > 1) {
    for (uint32_t p = 0; p < LL_NP; p++) {
      for (uint32_t cl = 0; cl < NC; cl++) nightly_pb_write(cl, PB_GO, p + 1);
      for (uint32_t cl = 0; cl < NC; cl++) {
        uint64_t polls = 0;
        while (nightly_pb_read(cl, PB_DONE) < p + 1) {
          if (++polls == 400000000ull) {
            nh_puts("timeout: phase "); nh_putu(p); nh_puts(" cluster "); nh_putu(cl); nh_puts("\n"); nh_flush();
            exit(2);
          }
        }
      }
      if ((p & 15) == 15) { nh_puts("phase "); nh_putu(p + 1); nh_puts("/"); nh_putu(LL_NP); nh_puts("\n"); nh_flush(); }
    }
  }
  if (!nightly_host_wait(NC, 2000000000ull)) { nh_puts("timeout waiting for the kernel end\n"); nh_flush(); exit(2); }
  nightly_host_report(NC);

  // ---- per-phase cycles: span = latest end - earliest start over the clusters (mcycle deltas, wrap-safe) ----
  static uint64_t op_cyc[LL_NOPS], op_mac[LL_NOPS];
  static uint32_t op_n[LL_NOPS];
  uint64_t total = 0, busy = 0, macs = 0;
  uint32_t prev_start = 0;
  nh_puts("phase layer op cycles [MACs mesh_util]\n");
  for (uint32_t p = 0; p < LL_NP; p++) {
    const uint32_t s0 = stamp(p, 0, 0);
    int32_t lo = 0, hi = 0;
    for (uint32_t c = 0; c < NC; c++) {
      const int32_t s = (int32_t)(stamp(p, c, 0) - s0), e = (int32_t)(stamp(p, c, 1) - s0);
      if (s < lo) lo = s;
      if (e > hi) hi = e;
    }
    const uint32_t start = s0 + (uint32_t)lo, span = (uint32_t)(hi - lo);
    const uint32_t tag = desc(p, 1), op = tag & 0xFF, layer = (tag >> 8) & 0xFF;
    const uint64_t m = gemm_macs(p);
    if (p >= LL_NFILL) {
      if (p > LL_NFILL) total += (uint32_t)(start - prev_start);
      prev_start = start;
      busy += span;
      macs += m;
      if (op < LL_NOPS) { op_cyc[op] += span; op_mac[op] += m; op_n[op]++; }
      if (p == LL_NP - 1) total += span;
    }
    nh_putu(p); nh_putc(' ');
    if (layer == 255) nh_putc('-'); else nh_putu(layer);
    nh_putc(' '); nh_puts(op < LL_NOPS ? LL_OPS[op] : "?"); nh_putc(' '); nh_putu(span);
    if (m) { nh_putc(' '); nh_putu(m); nh_putc(' '); put_pct(nightly_util_bp(m, 256ull * NC, span)); }
    nh_putc('\n');
    if ((p & 31) == 31) nh_flush();
  }
  nh_puts("== per op (all layers): op count cycles share [mesh_util]\n");
  for (uint32_t o = 1; o < LL_NOPS; o++) {
    if (!op_n[o]) continue;
    nh_puts(LL_OPS[o]); nh_putc(' '); nh_putu(op_n[o]); nh_putc(' '); nh_putu(op_cyc[o]); nh_putc(' ');
    put_pct(op_cyc[o] * 10000ull / (total ? total : 1));
    if (op_mac[o]) { nh_putc(' '); put_pct(nightly_util_bp(op_mac[o], 256ull * NC, op_cyc[o])); }
    nh_putc('\n');
  }
  nh_puts("== model (first phase after init to the end of the LM head)\n");
  nh_puts("model_cycles="); nh_putu(total); nh_puts(" phase_cycles="); nh_putu(busy);
  nh_puts(" handshake_cycles="); nh_putu(total - busy); nh_puts("\n");
  nh_puts("cycles_per_layer="); nh_putu((total - op_cyc[13] - op_cyc[14]) / LL_L);
  nh_puts(" cycles_per_token="); nh_putu(total / LL_M); nh_puts("\n");
  nh_puts("gemm_macs="); nh_putu(macs); nh_puts(" mesh_util_model="); put_pct(nightly_util_bp(macs, 256ull * NC, total));
  nh_puts("\n"); nh_flush();

  // ---- checks ----
  int pass = 1;
  for (uint32_t i = 0; i < LL_NCHECKS; i++) {
    const LlCheck &c = LL_CHECKS[i];
    if (c.bytes) {
      const volatile uint8_t *gp = nightly_gpu_ptr<uint8_t>(c.dev), *rp = nightly_gpu_ptr<uint8_t>(c.gold);
      uint32_t bad = 0, first = 0xFFFFFFFFu;
      for (uint32_t k = 0; k < c.n; k++)
        if (gp[k] != rp[k]) { if (bad == 0) first = k; bad++; }
      nh_puts(c.name); nh_puts(": mismatched_bytes="); nh_putu(bad); nh_putc('/'); nh_putu(c.n);
      if (c.tol) { nh_puts(" (allowed "); nh_putu(c.tol); nh_puts(")"); }
      if (bad) { nh_puts(" first_at="); nh_putu(first); nh_puts(" got="); nh_putx(gp[first], 2); nh_puts(" want="); nh_putx(rp[first], 2); }
      nh_putc('\n'); nh_flush();
      if (bad > c.tol) pass = 0;
      continue;
    }
    const NhCmp r = nh_compare_bf16(nightly_gpu_ptr<uint16_t>(c.dev), (const uint16_t *)nightly_gpu_ptr<uint16_t>(c.gold),
                                    c.n, c.scale_log2);
    nh_print_cmp(c.name, r);
    nh_flush();
    // tol 0: bit-exact required; tol > 0: relative Frobenius error <= tol (units of 0.01 %)
    if (r.nonfinite || (c.tol ? r.err_bp > c.tol : r.exact != r.n)) pass = 0;
  }
  if (LL_LOGITS_ADDR) {   // logits: finite, top-1 agreement
    const volatile uint16_t *lg = nightly_gpu_ptr<uint16_t>(LL_LOGITS_ADDR);
    uint32_t nonfin = 0, agree_hf = 0, agree_emu = 0, agree_label = 0, hf_label = 0;
    for (uint32_t t = 0; t < LL_M; t++) {
      uint32_t best = 0, bh = 0, be = 0;
      int32_t kb = -0x7FFFFFFF, kh = kb, ke = kb;
      for (uint32_t v = 0; v < LL_V; v++) {
        const uint16_t x = lg[t * LL_V + v];
        if (((x >> 7) & 0xFF) == 0xFF) nonfin++;
        if (bf_key(x) > kb) { kb = bf_key(x); best = v; }
        if (LL_HF_LOGITS_ADDR) {
          const uint16_t y = nightly_gpu_ptr<uint16_t>(LL_HF_LOGITS_ADDR)[t * LL_V + v];
          if (bf_key(y) > kh) { kh = bf_key(y); bh = v; }
        }
        if (LL_EMU_LOGITS_ADDR) {
          const uint16_t y = nightly_gpu_ptr<uint16_t>(LL_EMU_LOGITS_ADDR)[t * LL_V + v];
          if (bf_key(y) > ke) { ke = bf_key(y); be = v; }
        }
      }
      agree_hf += LL_HF_LOGITS_ADDR && best == bh;
      agree_emu += LL_EMU_LOGITS_ADDR && best == be;
      if (LL_LABELS_ADDR) {
        const uint32_t lab = rd32(LL_LABELS_ADDR + t * 4);
        agree_label += best == lab;
        hf_label += bh == lab;
      }
    }
    nh_puts("logits: nonfinite="); nh_putu(nonfin);
    if (LL_EMU_LOGITS_ADDR) { nh_puts(" top1_vs_emulation="); nh_putu(agree_emu); nh_putc('/'); nh_putu(LL_M); }
    if (LL_HF_LOGITS_ADDR) { nh_puts(" top1_vs_HF="); nh_putu(agree_hf); nh_putc('/'); nh_putu(LL_M); }
    if (LL_LABELS_ADDR) {
      nh_puts(" top1_vs_next_token="); nh_putu(agree_label); nh_putc('/'); nh_putu(LL_M);
      nh_puts(" (HF fp32: "); nh_putu(hf_label); nh_puts(")");
    }
    nh_puts("\n");
    if (nonfin) pass = 0;
  }
  nh_puts(pass ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(pass ? 0 : 1);
}
