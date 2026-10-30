// rpbench: the llama_e2e ROPE phase on cyclotron (no host: mu_schedule straight from main).  One case per build
// (CASE=gemma2 | tinyllama | deepseek | llama7b, gen_data.py), M = 128 tokens at P0 = 128 of an SK = 256 cache.
//   BENCH_MODE 0  nothing (boot + epilogue cycles, subtracted from the others)
//              1  OLD: hd 64 / 256 the committed rope_qk_fast + rope_v (warp per unit, gw of ngw); llama7b rope_fa +
//                 rope_v_fa; deepseek the fused rope_*_unit<true> (bias, key center), every unit in turn
//              2  NEW: ../rope_fast.h (core-owned units, own of nown, the core's warps split them)
//              3  OLD then NEW for every simulated cluster, then every output byte compared; prints the counts
// SIM_NC clusters are simulated on cyclotron's one: SIM_TB = the cluster this build plays (modes 1 / 2: the phase
// takes the slowest cluster's time), or -1 = all of them in turn (mode 3).
// Cyclotron does not advance mcycle: time a build by the simulator's cycle count minus mode 0's.
#include <stdint.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#ifndef NIGHTLY_OCC
#define NIGHTLY_OCC 4
#endif
extern "C" uint32_t __mu_num_warps = NIGHTLY_OCC;
extern "C" uint32_t __mu_cluster_from_pb = 0;           // see ../smbench/kernel.cpp: no host to wait for
__attribute__((noinline, convergent)) static void nightly_kernel_end(uint32_t tid, uint32_t tpb) {
  mu_barrier(4, tpb / MU_NUM_THREADS);
  if (tid < MU_NUM_CORES * MU_NUM_THREADS) mu_fence();
  else asm volatile("nop");
}
#include RP_CASE_H

#ifndef BENCH_MODE
#define BENCH_MODE 3
#endif
#ifndef SIM_NC
#define SIM_NC 1
#endif
#ifndef SIM_TB
#define SIM_TB -1
#endif

constexpr uint32_t WARPS = MU_NUM_CORES * NIGHTLY_OCC;
constexpr uint32_t BAR_PH = 1, BAR_FL = 3;               // cyclotron: barrier ids < 8 (higher ids are other tasks)
constexpr uint32_t RED_SM = 0x1C000;
constexpr uint32_t FUSE_STAGE_BYTES = 0x1000;            // the fused units' per-warp SMEM (kernel.cpp)
static inline uint32_t opaque(uint32_t x) { asm volatile("" : "+r"(x)); return x; }
__attribute__((noinline, convergent)) static void bar(uint32_t id, uint32_t warps) { mu_barrier(id, warps); }

// ---- kernel.cpp, extracted at build time (see the Makefile) ----
#include "build/helpers.h"
namespace old {
#include "build/rope_old.h"
}
#undef LL_ROPE_FAST_H
namespace nw {
#include "../rope_fast.h"
}

// ---- harness ----
constexpr uint32_t HD = RP_HD, HQ = RP_HQ, HKV = RP_HKV, M = RP_M, SK = RP_SK, GQ = HQ / HKV;
constexpr uint32_t NQKV = (HQ + 2 * HKV) * HD;
constexpr uint32_t OUT = 0x40000000u, VSTRIDE = 0x800000u;
// [variant] Q codes, Q scales, K^T codes, K scales, V codes, V scales (1 MiB each)
static inline uint32_t out(uint32_t v, uint32_t k) { return OUT + v * VSTRIDE + k * 0x100000u; }
constexpr uint32_t out_bytes[6] = {HQ * M * HD, HQ * M * HD / 32, HKV * SK * HD, HKV * SK * HD / 32, HKV * SK * HD,
                                   HKV * SK * HD / 32};
static_assert(HKV * SK * HD <= 0x100000u, "rpbench output slot");
static uint32_t desc[2][32] __attribute__((aligned(64)));
static uint32_t result[16] __attribute__((aligned(64)));

static void make_desc(uint32_t *d, uint32_t v) {          // gen_data.py's ROPE descriptor (rope_args)
  for (uint32_t i = 0; i < 32; i++) d[i] = 0;
  d[2] = RP_QKV; d[3] = NQKV; d[4] = HQ; d[5] = HKV; d[6] = HD; d[7] = RP_COS; d[8] = RP_SIN;
  d[9] = out(v, 0); d[10] = out(v, 1); d[11] = GQ * M * HD; d[12] = GQ * M * HD / 32;
  d[13] = out(v, 2); d[14] = out(v, 3); d[15] = SK * HD; d[16] = SK * HD / 32;
  d[17] = out(v, 4); d[18] = out(v, 5); d[19] = SK * HD; d[20] = SK * HD / 32;
  d[21] = RP_P0; d[22] = M; d[23] = SK; d[24] = RP_TM; d[25] = 1; d[26] = 0;
  d[27] = RP_HAS_BIAS ? RP_BIAS : 0; d[28] = RP_HAS_BIAS ? RP_KCEN : 0;
}

static void putc_(char ch) { *(volatile uint8_t *)0xFF080000u = (uint8_t)ch; }
static void puts_(const char *s) { while (*s) putc_(*s++); }
static void putu(uint32_t v) {                           // hex; no runtime-indexed local array (they alias across lanes)
  putc_('0'); putc_('x');
  for (int sh = 28; sh >= 0; sh -= 4) putc_("0123456789abcdef"[(v >> sh) & 15u]);
}

static void phase_end(uint32_t warp, uint32_t warps) {
  bar(BAR_PH, warps);
  if (opaque(warp < MU_NUM_CORES)) mu_fence();
  else asm volatile("nop");
  bar(BAR_FL, warps);
}

// the OLD path of this case, as cluster tb of SIM_NC
static void run_old(const volatile uint32_t *d, uint32_t gw, uint32_t ngw, uint32_t warp, uint32_t lane,
                    volatile __shared uint32_t *stage) {
  if (HD == 128 && RP_HAS_BIAS) {                            // rope_fused's unit list, no tile waits
    const uint32_t NQ = HQ * (M / 4), NK = HKV * (M / 16), NV = HKV * (M / 32) * 4, base = warp * FUSE_STAGE_BYTES;
    for (uint32_t u = gw; u < NQ + NK + NV; u += ngw) {
      const uint32_t isq = opaque(u < NQ), isk = opaque(u >= NQ) & opaque(u < NQ + NK), isv = opaque(u >= NQ + NK);
      const uint32_t vk = u - NQ, vv = u - NQ - NK;
      const uint32_t a0 = isq ? u / (M / 4) : isk ? vk / (M / 16) : vv / ((M / 32) * 4);
      const uint32_t a1 = isq ? (u % (M / 4)) * 4 : isk ? (vk % (M / 16)) * 16 : (vv / 4) % (M / 32);
      if (isq) rope_q_unit<true>(d, a0, a1, lane, base);
      else asm volatile("nop");
      if (isk) rope_k_unit<true>(d, a0, a1, lane, base);
      else asm volatile("nop");
      if (isv) rope_v_unit<true>(d, a0, a1, vv % 4, lane, base);
      else asm volatile("nop");
    }
  } else if (HD == 128) {
    rope_fa(d, gw, ngw, lane, warp, 0);
    rope_v_fa(d, gw, ngw, lane, warp);
  } else {
    old::rope_qk_fast<HD / 32, false, true>(d, gw, ngw, lane, stage);
    old::rope_v(d, gw, ngw, lane, stage);
  }
}
static void run_new(const volatile uint32_t *d, uint32_t own, uint32_t nown, uint32_t wi, uint32_t nwo, uint32_t lane,
                    volatile __shared uint32_t *stage) {
  nw::rope_qk_fast<HD / 32, RP_HAS_BIAS != 0, true, RP_HAS_BIAS != 0>(d, own, nown, wi, nwo, lane, stage);
  nw::rope_v<RP_HAS_BIAS != 0>(d, own, nown, wi, nwo, lane, stage);
}

static void run(uint32_t v, uint32_t tid, uint32_t tpb) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  volatile __shared uint32_t *stage = sm32(STAGE_SM + warp * STAGE_BYTES);
  const volatile uint32_t *d = desc[v];
  if (tid == 0) make_desc((uint32_t *)desc[v], v);
  else asm volatile("nop");
  mu_fence();
  bar(2, warps);
  for (uint32_t tb = SIM_TB < 0 ? 0 : SIM_TB; tb < (SIM_TB < 0 ? SIM_NC : SIM_TB + 1); tb++) {
    const uint32_t gw = tb * WARPS + warp, ngw = SIM_NC * WARPS;
    const uint32_t own = tb * MU_NUM_CORES + warp % MU_NUM_CORES, nown = SIM_NC * MU_NUM_CORES;
    const uint32_t wi = warp / MU_NUM_CORES, nwo = WARPS / MU_NUM_CORES;
    if (v == 0) run_old(d, gw, ngw, warp, lane, stage);
    else run_new(d, own, nown, wi, nwo, lane, stage);
    phase_end(warp, warps);
  }
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t) {
  const uint32_t warp = tid / MU_NUM_THREADS;
  if (BENCH_MODE & 1) run(0, tid, tpb);
  if (BENCH_MODE & 2) run(1, tid, tpb);
  if (BENCH_MODE == 3 && warp == 0) {                    // compare: warp 0, lanes stride the words
    const uint32_t lane = tid % MU_NUM_THREADS;
    volatile __shared uint32_t *cnt = sm32(CW_SM);
    for (uint32_t k = 0; k < 6; k++) {
      uint32_t bad = 0, sum = 0;
      for (uint32_t i = lane; i < out_bytes[k] / 4; i += MU_NUM_THREADS) {
        const uint32_t a = g32(out(0, k))[i], b = g32(out(1, k))[i];
        bad += a != b;
        sum += b;
      }
      cnt[lane] = bad; cnt[16 + lane] = sum;
      mu_fence_smem();
      if (lane == 0) {
        uint32_t tb_ = 0, ts = 0;
        for (uint32_t j = 0; j < 16; j++) { tb_ += cnt[j]; ts += cnt[16 + j]; }
        result[2 * k] = tb_; result[2 * k + 1] = ts;
      } else asm volatile("nop");
      mu_fence_smem();
    }
    if (lane == 0) {
      puts_("rpbench words differing (checksum): Q ");
      putu(result[0]); puts_(" ("); putu(result[1]); puts_(") Qsc "); putu(result[2]); puts_(" ("); putu(result[3]);
      puts_(") K "); putu(result[4]); puts_(" ("); putu(result[5]); puts_(") Ksc "); putu(result[6]); puts_(" (");
      putu(result[7]); puts_(") V "); putu(result[8]); puts_(" ("); putu(result[9]); puts_(") Vsc "); putu(result[10]);
      puts_(" ("); putu(result[11]); puts_(")\n");
    } else asm volatile("nop");
  } else asm volatile("nop");
  nightly_kernel_end(tid, tpb);
}

int main() {
  mu_schedule(entry, nullptr, NIGHTLY_OCC);
  return 0;
}
