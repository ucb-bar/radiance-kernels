// smbench: the llama_e2e SOFTMAX phases on cyclotron (no host: mu_schedule straight from main).
// Cases (BENCH_CASES bits): 0 SmolVLA expert self-attention (SK 512, mask table), 1 cross-attention (SK 256),
// 2 LLM decode (kernel.cpp, SK 1024, causal).
//   BENCH_MODE 0  nothing (boot + epilogue cycles, subtracted from the others)
//              1  OLD: SmolVLA a copy of kernel_vla.cpp's softmax_phase as of 37f69cf; decode kernel.cpp's
//                 softmax_phase<false> (still the Gemma-2 path)
//              2  NEW: kernel_vla.cpp's softmax_phase, kernel.cpp's softmax_bf (both extracted from the sources)
//              3  OLD then NEW, then every P code and scale byte compared; prints the mismatch counts
// Cyclotron does not advance mcycle: time a mode by the simulator's cycle count minus mode 0's.
// The helpers below are verbatim copies of kernel_vla.cpp's.
// nightly/device.h's runtime symbols without its host protocol: __mu_cluster_from_pb = 1 makes mu_start.S wait for a
// printBuf cluster tag that only the nightly host writes (cyclotron has no host)
#include <stdint.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#ifndef NIGHTLY_OCC
#define NIGHTLY_OCC 4
#endif
extern "C" uint32_t __mu_num_warps = NIGHTLY_OCC;
extern "C" uint32_t __mu_cluster_from_pb = 0;
__attribute__((noinline, convergent)) static void nightly_kernel_end(uint32_t tid, uint32_t tpb) {
  mu_barrier(4, tpb / MU_NUM_THREADS);
  if (tid < MU_NUM_CORES * MU_NUM_THREADS) mu_fence();
  else asm volatile("nop");
}
#include "sm_data.h"

#ifndef BENCH_MODE
#define BENCH_MODE 3
#endif
#ifndef BENCH_CASES
#define BENCH_CASES 7
#endif

constexpr uint32_t WARPS = MU_NUM_CORES * NIGHTLY_OCC;
constexpr uint32_t BAR_PH = 1, BAR_FL = 3;               // cyclotron: barrier ids < 8 (higher ids are other tasks)
constexpr uint32_t RED_SM = 0x1C000;
static inline uint32_t opaque(uint32_t x) { asm volatile("" : "+r"(x)); return x; }
__attribute__((noinline, convergent)) static void bar(uint32_t id, uint32_t warps) { mu_barrier(id, warps); }

// ---- kernel_vla.cpp helpers (verbatim) ----
static inline float f_bits(uint32_t u) { return __builtin_bit_cast(float, u); }
static inline uint32_t bits_f(float f) { return __builtin_bit_cast(uint32_t, f); }
static inline float lo(uint32_t w) { return f_bits(w << 16); }
static inline float hi(uint32_t w) { return f_bits(w & 0xFFFF0000u); }
static inline uint32_t bf16(float f) {
  const uint32_t u = bits_f(f);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
static inline float fabs_(float f) { return f_bits(bits_f(f) & 0x7FFFFFFFu); }
static inline float fexp_bf(float x) {
  x = __builtin_fmaxf(x, -64.0f);
  const _Float16 r = mu_fexp(__builtin_bit_cast(_Float16, (uint16_t)bf16(x)));
  return f_bits((uint32_t)__builtin_bit_cast(uint16_t, r) << 16);
}
static inline uint32_t mask_if(uint32_t c) { return opaque(0u - c); }
static inline uint32_t sel(uint32_t c, uint32_t a, uint32_t b) { const uint32_t m = mask_if(c); return (a & m) | (b & ~m); }
static inline uint32_t bit(bool c) { return opaque((uint32_t)c); }
static inline int32_t block_exp(float amax) {
  const int32_t e = (int32_t)((bits_f(amax) >> 23) & 0xFF) - 127;
  return (int32_t)sel((uint32_t)(e < -126), (uint32_t)-126, (uint32_t)e);
}
static inline float pow2_neg(int32_t e) { return f_bits((uint32_t)(127 - e) << 23); }
static inline uint32_t fp8(float x) {
  const uint32_t u = bits_f(x), s = (u >> 24) & 0x80u, ax = u & 0x7FFFFFFFu;
  const uint32_t norm = ((ax + 0x7FFFFu + ((ax >> 20) & 1u)) >> 20) - (120u << 3);
  const uint32_t sub = bits_f(f_bits(ax) * 512.0f + 8388608.0f) - 0x4B000000u;
  return s | sel(bit(ax < 0x3C800000u), sub, norm);
}
static inline uint32_t pack4(float a, float b, float c, float d) {
  return fp8(a) | (fp8(b) << 8) | (fp8(c) << 16) | (fp8(d) << 24);
}
static inline uint32_t a_code(uint32_t a, uint32_t TM, uint32_t TK, uint32_t KS, uint32_t r, uint32_t k) {
  return a + (((r / TM) * KS + k / TK) * TM + r % TM) * TK + k % TK;
}
static inline uint32_t a_scale(uint32_t asc, uint32_t TM, uint32_t TK, uint32_t KS, uint32_t r, uint32_t k) {
  return asc + (((r / TM) * KS + k / TK) * (TK / 32) + (k % TK) / 32) * TM + r % TM;
}
constexpr uint32_t STAGE_SM = 0, STAGE_BYTES = 8192;
constexpr uint32_t GAMMA_SM = WARPS * STAGE_BYTES;
constexpr uint32_t CW_SM = RED_SM + WARPS * 64;
constexpr uint32_t SB_SM = GAMMA_SM + 8192, SB_BYTES = 2048;
static inline const uint32_t *g32(uint32_t a) { return (const uint32_t *)a; }
static inline volatile __shared uint32_t *sm32(uint32_t a) { return (volatile __shared uint32_t *)a; }
static inline float warp_sum(volatile __shared float *red, uint32_t lane, float v) {
  red[lane] = v;
  mu_fence_smem();
  float s = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < MU_NUM_THREADS; j++) s += red[j];
  mu_fence_smem();
  return s;
}
static inline float warp_max(volatile __shared float *red, uint32_t lane, float v) {
  red[lane] = v;
  mu_fence_smem();
  float m = red[0];
#pragma unroll
  for (uint32_t j = 1; j < MU_NUM_THREADS; j++) m = __builtin_fmaxf(m, red[j]);
  mu_fence_smem();
  return m;
}
static inline void mx_group2(volatile __shared float *red, volatile __shared uint32_t *cw, uint32_t lane, float v0,
                             float v1, uint32_t code_addr, volatile __shared uint32_t *sbw, uint32_t rr) {
  const float amax = warp_max(red, lane, __builtin_fmaxf(fabs_(v0), fabs_(v1)));
  const int32_t e = block_exp(amax);
  const float sc = pow2_neg(e);
  cw[lane] = fp8(v0 * sc) | (fp8(v1 * sc) << 8);
  mu_fence_smem();
  if (lane < 8) *(uint32_t *)(code_addr + 4 * lane) = cw[2 * lane] | (cw[2 * lane + 1] << 16);
  else asm volatile("nop");
  mu_fence_smem();
  if (lane == 0) *sbw = (rr ? *sbw : 0u) | ((uint32_t)(e + 127) << (8 * rr));
  else asm volatile("nop");
}
static inline void sb_flush(volatile __shared uint32_t *sb, uint32_t lane, uint32_t groups, uint32_t g0, uint32_t asc,
                            uint32_t TM, uint32_t TK, uint32_t KS, uint32_t rb) {
  mu_fence_smem();
  for (uint32_t g = lane; g < groups; g += MU_NUM_THREADS)
    *(uint32_t *)a_scale(asc, TM, TK, KS, rb, (g0 + g) * 32) = sb[g];
  mu_fence_smem();
}
static inline float exp2_poly(float t) {
  t = __builtin_fminf(__builtin_fmaxf(t, -126.0f), 126.0f);
  int32_t n;
  asm("fcvt.w.s %0, %1, rdn" : "=r"(n) : "r"(t));
  const float f = t - (float)n;
  float p = 1.8775767e-3f;
  p = p * f + 8.9893397e-3f;
  p = p * f + 5.5826318e-2f;
  p = p * f + 2.4015361e-1f;
  p = p * f + 6.9315308e-1f;
  p = p * f + 9.9999994e-1f;
  return f_bits(bits_f(p) + ((uint32_t)n << 23));
}
static inline float tanh_f(float y) { return 1.0f - 2.0f / (1.0f + exp2_poly(2.0f * 1.4426950409f * y)); }
struct F2 { float a, b; };
template <typename P, typename W>
__attribute__((always_inline)) static inline void quant_group(P pair, uint32_t lane, W *cbg, uint32_t scale_addr) {
  float amax = 0.f;
#pragma unroll 4
  for (uint32_t jj = 0; jj < 16; jj++) {
    const F2 p = pair((jj + lane) & 15u);
    amax = __builtin_fmaxf(amax, __builtin_fmaxf(fabs_(p.a), fabs_(p.b)));
  }
  const int32_t e = block_exp(amax);
  const float sc = pow2_neg(e);
#pragma unroll 2
  for (uint32_t jj = 0; jj < 8; jj++) {
    const uint32_t j = (jj + (lane >> 1)) & 7u;
    const F2 p0 = pair(2 * j), p1 = pair(2 * j + 1);
    cbg[j] = pack4(p0.a * sc, p0.b * sc, p1.a * sc, p1.b * sc);
  }
  *(volatile uint8_t *)scale_addr = (uint8_t)(e + 127);
}
static inline void cb_flush(volatile __shared uint32_t *cb, uint32_t lane, uint32_t ng, uint32_t g0, uint32_t A,
                            uint32_t TM, uint32_t TK, uint32_t KS, uint32_t r) {
  mu_fence_smem();
  for (uint32_t w = lane; w < ng * 8; w += MU_NUM_THREADS) *(uint32_t *)a_code(A, TM, TK, KS, r, 32 * g0 + 4 * w) = cb[w];
  mu_fence_smem();
}

// ---- OLD: kernel_vla.cpp's softmax_phase as of 37f69cf (a warp per output group) ----
static inline float masked_old(float v, uint32_t valid) { return f_bits(bits_f(v) & mask_if(valid)); }
template <bool CAP>
static inline float sm_logit_old(float s, float scale, float k1, float cap) { return CAP ? cap * tanh_f(s * k1) : s * scale; }
template <bool CAP>
static void softmax_old(const volatile uint32_t *d, uint32_t own, uint32_t nown, uint32_t wi, uint32_t nwo,
                        uint32_t lane, volatile __shared float *red, volatile __shared uint32_t *cw,
                        volatile __shared uint32_t *stage, volatile __shared uint32_t *sb) {
  const uint32_t s = d[2], s_bs = d[3], SK = d[4], R = d[5], batch = d[6];
  const uint32_t pa = d[7], pasc = d[8], pa_bs = d[9], pasc_bs = d[10];
  const float scale = f_bits(d[11]), k1 = f_bits(d[16]), cap = f_bits(d[17]);
  const uint32_t P0 = d[12], M = d[13], TM = d[14], TK = d[15], KSp = SK / TK, W = SK / 2, NGT = SK / 32;
  const uint32_t MTb = R / TM;
  const float NEG = f_bits(0xFF800000u);
  for (uint32_t u = own; u < batch * MTb; u += nown) {
    const uint32_t bt = u / MTb, mt = u % MTb;
    const uint32_t A = pa + bt * pa_bs, ASC = pasc + bt * pasc_bs;
    for (uint32_t rb = mt * TM + 4 * wi; rb < (mt + 1) * TM; rb += 4 * nwo) {
      for (uint32_t rr = 0; rr < 4; rr++) {
        const uint32_t r = rb + rr, t = r % M;
        const uint32_t tab = d[18], lim1 = tab ? g32(tab)[2 * t] : P0 + t, lim2 = tab ? g32(tab)[2 * t + 1] : 0u;
        const uint32_t b2 = tab ? d[19] : 0xFFFFFFFFu;
        auto ok = [&](uint32_t k) __attribute__((always_inline)) {
          return bit(k <= lim1) | (bit(k >= b2) & bit(k <= lim2));
        };
        const uint32_t *S = g32(s + bt * s_bs + r * SK * 2);
        float m = NEG;
        for (uint32_t w0 = lane; w0 < W; w0 += 8 * MU_NUM_THREADS) {
          uint32_t sv[8];
#pragma unroll
          for (uint32_t k = 0; k < 8; k++) sv[k] = S[w0 + k * MU_NUM_THREADS];
#pragma unroll
          for (uint32_t k = 0; k < 8; k++) {
            const uint32_t w = w0 + k * MU_NUM_THREADS;
            stage[w] = sv[k];
            m = __builtin_fmaxf(m, f_bits(sel(ok(2 * w), bits_f(sm_logit_old<CAP>(lo(sv[k]), scale, k1, cap)), bits_f(NEG))));
            m = __builtin_fmaxf(m, f_bits(sel(ok(2 * w + 1), bits_f(sm_logit_old<CAP>(hi(sv[k]), scale, k1, cap)), bits_f(NEG))));
          }
        }
        m = warp_max(red, lane, m);
        float sum = 0.f;
        for (uint32_t w = lane; w < W; w += MU_NUM_THREADS) {
          const uint32_t sv = stage[w];
          const float ea = masked_old(fexp_bf(sm_logit_old<CAP>(lo(sv), scale, k1, cap) - m), ok(2 * w));
          const float eb = masked_old(fexp_bf(sm_logit_old<CAP>(hi(sv), scale, k1, cap) - m), ok(2 * w + 1));
          sum += ea + eb;
          stage[w] = (bits_f(ea) >> 16) | (bits_f(eb) & 0xFFFF0000u);
        }
        sum = warp_sum(red, lane, sum);
        const float inv = 1.0f / sum;
        for (uint32_t g = 0; g < NGT; g++) {
          const uint32_t ev = stage[g * 16 + lane];
          mx_group2(red, cw, lane, lo(ev) * inv, hi(ev) * inv, a_code(A, TM, TK, KSp, r, g * 32), sb + g, rr);
        }
      }
      sb_flush(sb, lane, NGT, 0, ASC, TM, TK, KSp, rb);
    }
  }
}

// ---- NEW: the SOFTMAX sections of kernel_vla.cpp and kernel.cpp, extracted at build time (see the Makefile) ----
namespace vla {
#include "build/softmax_vla.h"
}
namespace llm {
#include "build/softmax_llm.h"
}
constexpr uint32_t NCASE = 3;
static inline uint32_t case_sk(uint32_t c) { return c == 0 ? 512u : c == 1 ? 256u : 1024u; }
static inline uint32_t case_rows(uint32_t c) { return c == 2 ? 16u * 16u : SM_B * SM_R; }   // rows in the buffers

// ---- harness ----
static uint32_t desc[2][NCASE][32] __attribute__((aligned(64)));   // [variant][case]: no line read before it is written
static uint32_t result[16] __attribute__((aligned(64)));   // 3 per case
constexpr uint32_t OUT = 0x40000000u;                    // [variant][case]: codes (<= 512 KiB), then scales
static inline uint32_t out_codes(uint32_t v, uint32_t c) { return OUT + v * 0x400000u + c * 0x100000u; }
static inline uint32_t out_scales(uint32_t v, uint32_t c) { return out_codes(v, c) + 0x80000u; }

static void make_desc(uint32_t *d, uint32_t c, uint32_t v) {
  const uint32_t SK = case_sk(c);
  for (uint32_t i = 0; i < 32; i++) d[i] = 0;
  if (c == 2) {                                          // gen_decode.py's SOFTMAX (TinyLlama: TMp 16, Gq 8, T 1)
    d[2] = SM_S_DEC; d[3] = 16 * SK * 2; d[4] = SK; d[5] = 16; d[6] = 16;
    d[7] = out_codes(v, c); d[8] = out_scales(v, c); d[9] = 16 * SK; d[10] = 16 * SK / 32;
    d[11] = 0x3E000000u; d[12] = 1000; d[13] = 1; d[14] = 16; d[15] = 256; d[18] = 8; d[19] = 1;
    return;
  }
  d[2] = c ? SM_S_CROSS : SM_S_SELF;
  d[3] = SM_R * SK * 2;
  d[4] = SK;
  d[5] = SM_R;
  d[6] = SM_B;
  d[7] = out_codes(v, c);
  d[8] = out_scales(v, c);
  d[9] = SM_R * SK;
  d[10] = SM_R * SK / 32;
  d[11] = 0x3E000000u;                                   // 1 / 8
  d[12] = 0;
  d[13] = SM_M;
  d[14] = SM_TM;
  d[15] = SM_TK;
  d[18] = c ? SM_CTAB : SM_STAB;
  d[19] = c ? SM_NOROW : 256u;
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

static void run(uint32_t v, uint32_t tid, uint32_t tpb) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  volatile __shared float *red = (volatile __shared float *)(RED_SM + warp * 64);
  volatile __shared uint32_t *stage = sm32(STAGE_SM + warp * STAGE_BYTES);
  volatile __shared uint32_t *cw = sm32(CW_SM + warp * 128);
  volatile __shared uint32_t *sb = sm32(SB_SM + warp * SB_BYTES);
  const uint32_t own = warp % MU_NUM_CORES, nown = MU_NUM_CORES, wi = warp / MU_NUM_CORES, nwo = WARPS / MU_NUM_CORES;
  for (uint32_t c = 0; c < NCASE; c++) {
    if (!((BENCH_CASES >> c) & 1)) continue;
    const volatile uint32_t *d = desc[v][c];
    if (tid == 0) make_desc((uint32_t *)desc[v][c], c, v);
    else asm volatile("nop");
    mu_fence();
    bar(2, warps);
    if (c == 2) {                                        // kernel.cpp decode: a unit per warp (sdec)
      if (v == 0) llm::softmax_phase<false>(d, warp, WARPS, 0, 1, lane, red, stage);
      else llm::softmax_bf<false>(d, warp, WARPS, 0, 1, lane, red, stage, sb);
    } else if (v == 0) softmax_old<false>(d, own, nown, wi, nwo, lane, red, cw, stage, sb);
    else if (c == 0) vla::softmax_phase<false, false>(d, own, nown, wi, nwo, lane, red, stage, sb);
    else vla::softmax_phase<false, true>(d, own, nown, wi, nwo, lane, red, stage, sb);
    phase_end(warp, warps);
  }
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS;
  if (BENCH_MODE & 1) run(0, tid, tpb);
  if (BENCH_MODE & 2) run(1, tid, tpb);
  if (BENCH_MODE == 3 && warp == 0) {                    // compare: warp 0, lanes stride the words
    const uint32_t lane = tid % MU_NUM_THREADS;
    volatile __shared uint32_t *cnt = sm32(CW_SM);
    for (uint32_t c = 0; c < NCASE; c++) {
      const uint32_t SK = case_sk(c), nc = case_rows(c) * SK / 4, ns = case_rows(c) * SK / 32 / 4;
      uint32_t bad_c = 0, bad_s = 0, sum = 0;
      if ((BENCH_CASES >> c) & 1) {
        for (uint32_t i = lane; i < nc; i += MU_NUM_THREADS) {
          const uint32_t a = g32(out_codes(0, c))[i], b = g32(out_codes(1, c))[i];
          bad_c += a != b;
          sum += b;
        }
        for (uint32_t i = lane; i < ns; i += MU_NUM_THREADS) {
          const uint32_t a = g32(out_scales(0, c))[i], b = g32(out_scales(1, c))[i];
          bad_s += a != b;
          sum += b;
        }
      }
      cnt[lane] = bad_c; cnt[16 + lane] = bad_s; cnt[32 + lane] = sum;
      mu_fence_smem();
      if (lane == 0) {
        uint32_t tc = 0, ts = 0, tsum = 0;
        for (uint32_t j = 0; j < 16; j++) { tc += cnt[j]; ts += cnt[16 + j]; tsum += cnt[32 + j]; }
        result[3 * c] = tc; result[3 * c + 1] = ts; result[3 * c + 2] = tsum;
      } else asm volatile("nop");
      mu_fence_smem();
    }
    if (lane == 0) {
      puts_("smbench self: code words differing ");
      putu(result[0]); puts_(" scale words differing "); putu(result[1]); puts_(" checksum "); putu(result[2]);
      puts_("\nsmbench cross: code words differing ");
      putu(result[3]); puts_(" scale words differing "); putu(result[4]); puts_(" checksum "); putu(result[5]);
      puts_("\nsmbench decode: code words differing ");
      putu(result[6]); puts_(" scale words differing "); putu(result[7]); puts_(" checksum "); putu(result[8]);
      puts_("\n");
    } else asm volatile("nop");
  } else asm volatile("nop");
  nightly_kernel_end(tid, tpb);
}

int main() {
  mu_schedule(entry, nullptr, NIGHTLY_OCC);
  return 0;
}
