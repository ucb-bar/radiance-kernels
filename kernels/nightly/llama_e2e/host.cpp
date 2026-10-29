// llama_e2e host (Rocket, integer only): launch, step the clusters through the phase program, then report the
// per-phase cycle counts the GPU stamped (mcycle at each phase's start and end, per cluster) and check results.
//
// Phase handshake (printBuf slots 20 / 21): GO = p + 1 lets every cluster start phase p; each posts DONE = p + 1.
// With several clusters the host releases phase p + 1 only after every cluster posted DONE = p + 1, which is the
// cross-cluster barrier between phases.  With one cluster GO = NP up front: no host round trips.
#define LL_HOST 1
#include <nightly/host.h>
#include "llama_data.h"
// per-phase host watchdog (multi-cluster): polls of a cluster's DONE word before "timeout: phase p cluster c" (~20 target
// cycles a poll); debug builds lower it in the data header so a hang is located in about a minute
#ifndef LL_WATCHDOG_POLLS
#define LL_WATCHDOG_POLLS 400000000ull
#endif

static constexpr uint32_t NC = NIGHTLY_CLUSTERS;
static constexpr uint32_t PB_GO = 20, PB_DONE = 21;

static inline uint32_t rd32(uint32_t dev) { return *nightly_gpu_ptr<uint32_t>(dev); }
static inline uint32_t desc(uint32_t p, uint32_t w) { return rd32(LL_PROG_ADDR + (p * LL_PW + w) * 4); }
static inline uint32_t stamp(uint32_t p, uint32_t c, uint32_t e) { return rd32(LL_PERF_ADDR + ((p * NC + c) * 16 + e) * 4); }

// ---- offline checks (LL_OFFLINE, default off): write the checked device buffers to files on the simulation host and
// let check_offline.py compare them there.  The Rocket host has no FPU and reads GPU DRAM slowly: on-host checks of a
// 152k-word vocabulary took longer than the kernel itself.  Files land in the FireSim driver's working directory (the
// sim slot) through fesvr's syscall proxy: libgloss's open() is a stub and htif_syscall() passes only 3 arguments, so
// openat goes through this 7-argument copy of the same magic-memory protocol (libgloss misc/htif.c).
// Off by default: on FireSim, fesvr reads the buffers over serial TSI, 1 KiB per round trip and one round trip per
// tsibridge step (2M target cycles), so a 1 MiB write did not finish in 10 minutes.  Usable under spike, or on FireSim
// with a far smaller +fesvr-step-size.
#ifndef LL_OFFLINE
#define LL_OFFLINE 0
#endif
extern "C" volatile uint64_t tohost, fromhost;
static long ll_syscall(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3 = 0, uint64_t a4 = 0) {
  volatile uint64_t buf[8] __attribute__((aligned(64))) = {n, a0, a1, a2, a3, a4, 0, 0};
  asm volatile("fence" ::: "memory");
  tohost = (uint64_t)(uintptr_t)buf;                     // HTIF_TOHOST(device 0 = syscall, cmd 0, payload)
  while (fromhost == 0) {}
  fromhost = 0;
  asm volatile("fence" ::: "memory");
  return (long)buf[0];
}
static constexpr uint64_t SYS_OPENAT = 56, SYS_CLOSE = 57, SYS_WRITE = 64;
static constexpr uint64_t AT_CWD = (uint64_t)-100, LNX_WRONLY_CREAT_TRUNC = 0x241;   // Linux flag values: fesvr passes them through
static uint32_t ll_strlen(const char *s) { uint32_t n = 0; while (s[n]) n++; return n; }
static int ll_open(const char *name) {
  return (int)ll_syscall(SYS_OPENAT, AT_CWD, (uint64_t)(uintptr_t)name, ll_strlen(name) + 1, LNX_WRONLY_CREAT_TRUNC, 0644);
}
static int ll_write_all(int fd, uint64_t addr, uint64_t n) {   // addr: any host address (GPU DRAM included)
  while (n) {
    const uint64_t k = n < (1u << 20) ? n : (1u << 20);
    const long w = ll_syscall(SYS_WRITE, (uint64_t)fd, addr, k);
    if (w <= 0) return 0;
    addr += (uint64_t)w; n -= (uint64_t)w;
  }
  return 1;
}
// text into the manifest: a small line buffer, written per line
static char ll_line[512];
static uint32_t ll_ll = 0;
static void lc(char c) { if (ll_ll < sizeof(ll_line)) ll_line[ll_ll++] = c; }
static void ls_(const char *s) { while (*s) lc(*s++); }
static void lu(uint64_t v) { char t[24]; int n = 0; do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v); while (n) lc(t[--n]); }
static void li(int64_t v) { if (v < 0) { lc('-'); v = -v; } lu((uint64_t)v); }
static int lflush(int fd) { const int ok = ll_write_all(fd, (uint64_t)(uintptr_t)ll_line, ll_ll); ll_ll = 0; return ok; }
// Writes lldump_<i>.bin per check (device bytes) and lldump.txt (the manifest check_offline.py reads).  Returns 0 if
// the host could not create a file (the caller then runs the on-host checks).
static int ll_offline_dump() {
  const int mf = ll_open("lldump.txt");
  if (mf < 0) return 0;
  int ok = 1;
  ls_("lldump 1 model="); ls_(LL_MODEL); ls_(" M="); lu(LL_M); ls_(" V="); lu(LL_V); ls_(" clusters="); lu(NC); lc('\n');
  ok &= lflush(mf);
  char fname[24] = "lldump_000.bin", lgname[24] = "lldump_logits.bin";
  bool lg_done = false;                                  // the logits already dumped by a check of the same buffer
  for (uint32_t i = 0; i < LL_NCHECKS && ok; i++) {
    const LlCheck &c = LL_CHECKS[i];
    const uint64_t bytes = c.bytes ? (uint64_t)c.n : 2ull * c.n;
    fname[7] = (char)('0' + i / 100); fname[8] = (char)('0' + i / 10 % 10); fname[9] = (char)('0' + i % 10);
    const int fd = ll_open(fname);
    if (fd < 0) { ok = 0; break; }
#if defined(LL_DUMP_TEST) && LL_DUMP_TEST
    ok &= ll_write_all(fd, (uint64_t)(uintptr_t)nightly_gpu_ptr<uint8_t>(c.gold), bytes);   // spike test: gold as device
#else
    ok &= ll_write_all(fd, (uint64_t)(uintptr_t)nightly_gpu_ptr<uint8_t>(c.dev), bytes);
#endif
    ll_syscall(SYS_CLOSE, (uint64_t)fd, 0, 0);
    if (LL_LOGITS_ADDR && c.dev == LL_LOGITS_ADDR && !c.bytes && c.n == LL_M * LL_V && !lg_done) {
      for (uint32_t k = 0; k < sizeof(fname); k++) lgname[k] = fname[k];
      lg_done = true;
    }
    ls_("check|"); ls_(c.name); lc('|'); ls_(fname); lc('|'); lu(c.dev); lc('|'); lu(c.gold); lc('|'); lu(c.n); lc('|');
    li(c.scale_log2); lc('|'); lu((uint64_t)c.bytes); lc('|'); lu(c.tol); lc('\n');
    ok &= lflush(mf);
  }
  if (LL_LOGITS_ADDR && ok) {
    const int fd = lg_done ? 0 : ll_open(lgname);
    if (fd < 0) ok = 0;
    else if (!lg_done) {
#if defined(LL_DUMP_TEST) && LL_DUMP_TEST
      const uint32_t src = LL_EMU_LOGITS_ADDR ? LL_EMU_LOGITS_ADDR : LL_LOGITS_ADDR;
#else
      const uint32_t src = LL_LOGITS_ADDR;
#endif
      ok &= ll_write_all(fd, (uint64_t)(uintptr_t)nightly_gpu_ptr<uint8_t>(src), 2ull * LL_M * LL_V);
      ll_syscall(SYS_CLOSE, (uint64_t)fd, 0, 0);
    }
    if (ok) {
      ls_("logits|"); ls_(lgname); lc('|'); lu(LL_LOGITS_ADDR); lc('|'); lu(LL_M); lc('|'); lu(LL_V); lc('|');
      lu(LL_HF_LOGITS_ADDR); lc('|'); lu(LL_EMU_LOGITS_ADDR); lc('|'); lu(LL_LABELS_ADDR); lc('\n');
      ok &= lflush(mf);
    }
  }
  ls_(ok ? "end ok\n" : "end incomplete\n");
  lflush(mf);
  ll_syscall(SYS_CLOSE, (uint64_t)mf, 0, 0);
  return ok;
}

static uint64_t gemm_macs(uint32_t p) {
  if (desc(p, 0) == LL_T_FA)   // QK + PV: H SQ SK d (useful MACs; the max pre-pass's QK is not counted)
    return 2ull * desc(p, 9) * desc(p, 11) * desc(p, 12) * (desc(p, 24) ? desc(p, 24) : 128);
  if (desc(p, 0) != LL_T_GEMM) return 0;
  // batch * MT * NT * KS * TM * TN * TK
  return (uint64_t)desc(p, 14) * desc(p, 15) * desc(p, 16) * desc(p, 17) * desc(p, 18) * LL_TN * desc(p, 19);
}

// bf16 bits -> order-preserving integer
#if defined(LL_DROPE_DBG) && LL_DROPE_DBG
// debug: every warp's progress word (kernel.cpp dbg_put: printBuf slot 24 + 2 w, its update count in 25 + 2 w)
#ifndef LL_DBG_WARPS
#define LL_DBG_WARPS 8u                                  // warps per cluster (MU_NUM_CORES x OCC)
#endif
static void dbg_dump(const char *tag) {
  static const char *const SEC[16] = {"phase_start", "unit", "Q_row", "Q_group", "K_word", "K_group", "K_empty",
                                      "K_stored", "V_chunk", "V_amax", "V_staged", "V_stored", "drope_done",
                                      "before_fence", "after_fence", "?"};
  for (uint32_t c = 0; c < NC; c++)
    for (uint32_t w = 0; w < LL_DBG_WARPS; w++) {
      const uint32_t v = nightly_pb_read(c, 24 + 2 * w), n = nightly_pb_read(c, 25 + 2 * w);
      nh_puts(tag); nh_puts(" cluster "); nh_putu(c); nh_puts(" warp "); nh_putu(w); nh_puts(": "); nh_putx(v, 8);
      nh_putc(' '); nh_puts(SEC[v >> 28]); nh_puts(" x="); nh_putu((v >> 16) & 0xFFF); nh_puts(" a=");
      nh_putu((v >> 8) & 0xFF); nh_puts(" b="); nh_putu(v & 0xFF); nh_puts(" count="); nh_putu(n); nh_putc('\n');
    }
  nh_flush();
}
#endif
static inline int32_t bf_key(uint16_t b) { return (b & 0x8000) ? -(int32_t)(b & 0x7FFF) : (int32_t)b; }

static void put_pct(uint64_t bp) { nh_putfix(bp, 2); nh_putc('%'); }

int main() {
  nh_puts("llama_e2e " LL_MODEL ": M="); nh_putu(LL_M); nh_puts(" tokens at positions "); nh_putu(LL_P0);
  nh_puts(".."); nh_putu(LL_P0 + LL_M - 1); nh_puts(" (KV cache "); nh_putu(LL_P0); nh_puts(", SK="); nh_putu(LL_SK);
  nh_puts(") layers="); nh_putu(LL_L); nh_puts(" D="); nh_putu(LL_D); nh_puts(" F="); nh_putu(LL_F);
  nh_puts(" heads="); nh_putu(LL_HQ); nh_putc('/'); nh_putu(LL_HKV); nh_puts(" hd="); nh_putu(LL_HD);
  nh_puts(" vocab="); nh_putu(LL_V); nh_puts(" phases="); nh_putu(LL_NP); nh_puts(" (init "); nh_putu(LL_NFILL);
  nh_puts(") clusters="); nh_putu(NC); nh_puts("\n"); nh_flush();

#if defined(LL_DUMP_TEST) && LL_DUMP_TEST
  // spike test of the offline-check path: no GPU; dumps the gold buffers as if they were the device's
  { const int ok = ll_offline_dump(); nh_puts(ok ? "OFFLINE_DUMP ok\n" : "OFFLINE_DUMP failed\n"); nh_flush(); exit(ok ? 0 : 1); }
#endif
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
#if defined(LL_DROPE_DBG) && LL_DROPE_DBG
  for (uint32_t cl = 0; cl < NC; cl++)                   // the warps' progress slots (kernel.cpp dbg_put)
    for (uint32_t w = 0; w < 2 * LL_DBG_WARPS; w++) nightly_pb_write(cl, 24 + w, 0);
#endif
  nightly_host_launch(NC);
#if defined(LL_DROPE_DBG) && LL_DROPE_DBG
  // debug: the per-phase watchdog at one cluster too (GO = NP is already written: the cluster runs straight through)
  if (true) {
#else
  if (NC > 1) {
#endif
    for (uint32_t p = 0; p < LL_NP; p++) {
#if defined(LL_DROPE_DBG) && LL_DROPE_DBG
      if (NC > 1)
#endif
      for (uint32_t cl = 0; cl < NC; cl++) nightly_pb_write(cl, PB_GO, p + 1);
#if defined(LL_PROGRESS) && LL_PROGRESS
      // progress print (-DLL_PROGRESS=1), after releasing phase p.  Off by default: one HTIF round trip on FireSim is
      // longer than most phases, so the clusters finish and wait for the host (TinyLlama 4 SMs: 344 M idle cycles).
      if (p > 0 && (p & 15) == 0) { nh_puts("phase "); nh_putu(p); nh_puts("/"); nh_putu(LL_NP); nh_puts("\n"); nh_flush(); }
#endif
      for (uint32_t cl = 0; cl < NC; cl++) {
        uint64_t polls = 0;
        while (nightly_pb_read(cl, PB_DONE) < p + 1) {
          if (++polls == LL_WATCHDOG_POLLS) {
            nh_puts("timeout: phase "); nh_putu(p); nh_puts(" cluster "); nh_putu(cl); nh_puts("\n");
            for (uint32_t c2 = 0; c2 < NC; c2++) {   // every cluster's progress: phases done (= index of the one it is in)
              nh_puts("  cluster "); nh_putu(c2); nh_puts(" done="); nh_putu(nightly_pb_read(c2, PB_DONE)); nh_puts("\n");
            }
            // cluster 0 writes its heartbeat (= p) right after it sees GO for phase p: heartbeat == the stuck phase
            // means it started that phase (hang inside it or in its epilogue), heartbeat == p - 1 means it never got
            // past the GO poll.  Host-side only: the device ELF and its timing are unchanged.
            nh_puts("heartbeat (cluster 0) "); nh_putx(nightly_pb_read(0, 48), 8); nh_putc('\n');
            nh_flush();
#if defined(LL_DROPE_DBG) && LL_DROPE_DBG
            dbg_dump("dbg1");
            for (uint32_t i = 0; i < 1000000u; i++) (void)nightly_pb_read(0, PB_DONE);   // a while, then again
            dbg_dump("dbg2");
#endif
            exit(2);
          }
        }
      }
    }
  }
#if defined(LL_FA_DUMP) && LL_FA_DUMP
  // debug builds: a short wait, then the checks regardless (the dump is in DRAM even if the kernel never ends)
  if (!nightly_host_wait(NC, 3000000ull)) { nh_puts("timeout waiting for the kernel end (continuing to the checks)\n"); nh_flush(); }
  else nightly_host_report(NC);
#else
  if (!nightly_host_wait(NC, 2000000000ull)) { nh_puts("timeout waiting for the kernel end\n"); nh_flush(); exit(2); }
  nightly_host_report(NC);
#endif

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
#if defined(LL_FA_PROF) && LL_FA_PROF
  // FA role totals (fa_phase.h PROF_*), per cluster, summed over the ATTN phases
  nh_puts("== fa_prof (cycles summed over ATTN phases) cluster: prologue | producer: retire_wait nd_wait bar_s_wait issue "
          "bar_p_wait epilogue | tail | softmax(w2): bar_s_wait compute bar_p_wait | norm_o(w1): bar_n_wait compute bar_nd_wait\n");
  for (uint32_t c = 0; c < NC; c++) {
    uint64_t acc[14] = {0};
    for (uint32_t p = LL_NFILL; p < LL_NP; p++)
      if (desc(p, 0) == LL_T_FA)
        for (uint32_t k = 0; k < 14; k++) acc[k] += stamp(p, c, 2 + k);
    nh_puts("fa_prof c"); nh_putu(c);
    for (uint32_t k = 0; k < 14; k++) {
      if (k == 1 || k == 7 || k == 8 || k == 11) nh_puts(" |");
      nh_putc(' '); nh_putu(acc[k]);
    }
    nh_putc('\n');
  }
  nh_flush();
#endif
  nh_puts("== model (first phase after init to the end of the LM head)\n");
  nh_puts("model_cycles="); nh_putu(total); nh_puts(" phase_cycles="); nh_putu(busy);
  nh_puts(" handshake_cycles="); nh_putu(total - busy); nh_puts("\n");
  // per layer: without final, lmhead and softcap (ops 13, 14, 20)
#ifdef LL_STEPS
  // decode (gen_decode.py): LL_M = steps x sequences; a layer is counted once per step
  nh_puts("decode: batch="); nh_putu(LL_BATCH); nh_puts(" steps="); nh_putu(LL_STEPS);
#if defined(LL_VERIFY_T) && LL_VERIFY_T > 1
  nh_puts(" verify_tokens_per_step="); nh_putu(LL_VERIFY_T);   // speculative verify: T rows per sequence and step
#endif
  nh_puts(" cycles_per_step="); nh_putu(total / LL_STEPS);
  nh_puts(" cycles_per_layer="); nh_putu((total - op_cyc[13] - op_cyc[14] - (LL_NOPS > 20 ? op_cyc[20] : 0)) / (LL_L * LL_STEPS));
#else
  nh_puts("cycles_per_layer="); nh_putu((total - op_cyc[13] - op_cyc[14] - (LL_NOPS > 20 ? op_cyc[20] : 0)) / LL_L);
#endif
  nh_puts(" cycles_per_token="); nh_putu(total / LL_M); nh_puts("\n");
  nh_puts("gemm_macs="); nh_putu(macs); nh_puts(" mesh_util_model="); put_pct(nightly_util_bp(macs, 256ull * NC, total));
  nh_puts("\n"); nh_flush();

#if defined(LL_FA_DUMP) && LL_FA_DUMP && defined(LL_FA_DUMP_ADDR)
  // tile 0 rows 0..7 as bf16 hex, device / golden: m (scaled max), l, the pre-pass's raw masked block maxima j = 0..7
  for (uint32_t r = 0; r < 8; r++) {
    nh_puts("FA0 row "); nh_putu(r);
    nh_puts(" m "); nh_putx(rd32(LL_FA_DUMP_ADDR + 262144 + 4 * r) >> 16, 4); nh_putc('/'); nh_putx(rd32(LL_FA_DUMP_M + 4 * r) >> 16, 4);
    nh_puts(" l "); nh_putx(rd32(LL_FA_DUMP_ADDR + 262400 + 4 * r) >> 16, 4); nh_putc('/'); nh_putx(rd32(LL_FA_DUMP_L + 4 * r) >> 16, 4);
    nh_puts(" bm");
    for (uint32_t j = 0; j < 8; j++) {
      nh_putc(' '); nh_putx(rd32(LL_FA_DUMP_ADDR + 410112 + (j * 64 + r) * 4) >> 16, 4);
      nh_putc('/'); nh_putx(rd32(LL_FA_DUMP_BM + (j * 64 + r) * 4) >> 16, 4);
    }
    nh_putc('\n');
  }
  nh_flush();
#endif
#ifdef LL_HEX_N
  for (uint32_t i = 0; i < LL_HEX_N; i++) {   // debug: raw device / golden bytes
    for (uint32_t which = 0; which < 2; which++) {
      nh_puts("HEX "); nh_puts(LL_HEX[i].name); nh_puts(which ? " gold:" : " dev: ");
      const volatile uint8_t *b = nightly_gpu_ptr<uint8_t>(which ? LL_HEX[i].gold : LL_HEX[i].dev);
      for (uint32_t k = 0; k < LL_HEX[i].n; k++) { nh_putc(' '); nh_putx(b[k], 2); }
      nh_putc('\n');
    }
  }
  nh_flush();
#endif
#if LL_OFFLINE
  {
    const uint64_t t0 = nightly_rdcycle();
    if (ll_offline_dump()) {
      nh_puts("OFFLINE_DUMP ok host_cycles="); nh_putu(nightly_rdcycle() - t0);
      nh_puts(" (checks deferred to check_offline.py)\n"); nh_flush();
      exit(0);
    }
    nh_puts("OFFLINE_DUMP failed: checking on the host\n"); nh_flush();
  }
#endif
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
    // real weights: the model's predictions must agree with the fp32 HF model (>= 85 % top-1; the quantized pipeline
    // itself reaches ~90 % on DeepSeek, so a 90 % gate failed correct runs)
    if (LL_HF_LOGITS_ADDR && agree_hf * 20 < LL_M * 17) pass = 0;
  }
  nh_puts(pass ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(pass ? 0 : 1);
}
