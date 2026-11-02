// kernel.cpp: GPU side of the tapeout transformer encoder layer.
//
// One image holds every stage; the host launches it once per stage with ENArgs::stage set, and
// every launch ends in the tapeout epilogue (which also drains this launch's stores to DRAM, so
// the next stage's DMA or loads see them).
//
// SIMT conventions (2 cores, occupancy EN_OCC warps per core, 16 lanes per warp):
//   * L0d is per core and not coherent, so every 64-byte output line is written by one core.
//     Row blocks of 64 rows are owned by core (block % 2).
//   * MX-FP8 quantization: one e8m0 scale per 32 elements, block max mapped into (0.9375, 1.875],
//     e4m3 round-to-nearest-even, subnormals flushed to zero (see ref.py for why).

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <rad_tapeout.h>
#include <radiance.h>
#include <stdint.h>

#include "include/gemmini.h"
#include <mxgemmini_mmio.h>

#include "enc.h"
#include "../tapeout_gemm/mxg.hpp"

#define EN_OCC 3u   // warps per core (4 x ~63 registers per warp exceeds the renamer budget)
// EN_LN_OCC4: the LayerNorm launches run 4 warps per core.  The renamer budget per core is the union
// of written registers x occupancy (scripts/regs.py, keep <= ~216), so their code may write at most
// 54 registers.  ln_stage.cpp compiles this file again with EN_LN_TU and x54-x63 reserved
// (-target-feature +reserve-xN, Makefile) and provides the three LN stage functions (en_st_*); a per-function
// target("reserve-xN") attribute is silently ignored by this LLVM (checked 2026-10-21).
#ifdef EN_LN_OCC4
#define EN_LN_OCC 4u
#else
#define EN_LN_OCC EN_OCC
#endif
#define EN_LN_WARPS (EN_LN_OCC * 2u)

static inline uint32_t mcycle() {
  uint32_t c; asm volatile("csrr %0, mcycle" : "=r"(c)); return c;
}
static inline uint32_t f2u(float f) { union { float f; uint32_t u; } c; c.f = f; return c.u; }
static inline float u2f(uint32_t u) { union { float f; uint32_t u; } c; c.u = u; return c.f; }
static inline float bf16_to_f(uint32_t h) { return u2f(h << 16); }

// fp32 reciprocal square root: bit-trick seed + 3 Newton steps (no libm).
static inline float rsqrt_f(float x) {
  float y = u2f(0x5f3759dfu - (f2u(x) >> 1));
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

// All-ones if a < b (unsigned), else 0.  In inline asm so the optimizer cannot recognise the
// select pattern it is used in: LLVM turns a visible select into a branch, and on Muon every
// data-dependent branch is a divergence region (vx_split/vx_join, IPDOM stack).
static inline uint32_t mask_lt(uint32_t a, uint32_t b) {
  uint32_t m;
  asm("sltu %0, %1, %2\n\tneg %0, %0" : "=r"(m) : "r"(a), "r"(b));
  return m;
}
static inline uint32_t sel(uint32_t m, uint32_t x, uint32_t y) { return (x & ~m) | (y & m); }   // m ? y : x

// Integer replacements for the FP32 "short" ops (fsgnj/fabs/fneg, fmin/fmax, fcvt).  On the
// tapeout core, with several warps on the FPU, an fabs.s result was written to another warp's
// physical register and the owner's scoreboard never cleared: the warp stalled for good and every
// other warp's fence.s waited on it (metasim enc_m5, 2026-10-12; LN1/OQ/GELU hung on the board).
// FP arithmetic (add/mul/fma/div) is unaffected.  In asm so LLVM cannot fold them back.
static inline uint32_t and_asm(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("and %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
static inline uint32_t xor_asm(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("xor %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
static inline float fma_f(float a, float b, float c) {   // a * b + c, one rounding (fmadd.s)
  float r;
  asm("fmadd.s %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
static inline uint32_t abs_bits(float x) { return and_asm(f2u(x), 0x7FFFFFFFu); }   // |x| as bits
static inline float neg_f(float x) { return u2f(xor_asm(f2u(x), 0x80000000u)); }
static inline uint32_t umax(uint32_t a, uint32_t b) { return sel(mask_lt(a, b), a, b); }
static inline int imin(int a, int b) { return (int)sel(mask_lt((uint32_t)(b ^ INT32_MIN), (uint32_t)(a ^ INT32_MIN)), (uint32_t)a, (uint32_t)b); }
static inline int imax(int a, int b) { return (int)sel(mask_lt((uint32_t)(a ^ INT32_MIN), (uint32_t)(b ^ INT32_MIN)), (uint32_t)a, (uint32_t)b); }

// Block scale exponent: smallest e with amax * 2^-e <= 1.875 (0 for an all-zero block).
// `b` is the bit pattern of |amax|.
static inline int block_exp(uint32_t b) {
  const uint32_t e = (b >> 23) - 127u + (mask_lt(0x700000u, b & 0x7FFFFFu) & 1u);
  return (int)(e & mask_lt(0u, b));
}
static inline float block_inv(int e) { return u2f((uint32_t)(127 - e) << 23); }

// e4m3 encode of a block-scaled value (|v| <= 1.875): round to nearest even, subnormals flushed to
// zero (values below 15/16 * 2^-6 become 0; [15/16, 1) * 2^-6 rounds up to the smallest normal).
static inline uint32_t e4m3_encode(float v) {
  // |v| <= 1.875 here (the block scale guarantees it), so there is no saturation case.  Rounding
  // at bit 20 carries [15/16, 1) * 2^-6 into the smallest normal 0x08 by itself; every code below
  // 8 (subnormal or a wrapped negative) is flushed.  13 instructions.
  const uint32_t u = f2u(v), a = and_asm(u, 0x7FFFFFFFu);
  const int32_t c = (int32_t)((a + 0x7FFFFu + ((a >> 20) & 1u)) >> 20) - (int32_t)(120u << 3);
  uint32_t keep;
  asm("slti %0, %1, 8\n\taddi %0, %0, -1" : "=r"(keep) : "r"(c));   // c >= 8 ? ~0 : 0
  return ((uint32_t)c | ((u >> 24) & 0x80u)) & keep;
}
// Four values -> one little-endian code word.
static inline uint32_t e4m3_pack4(float a, float b, float c, float d) {
  return e4m3_encode(a) | (e4m3_encode(b) << 8) | (e4m3_encode(c) << 16) | (e4m3_encode(d) << 24);
}


// Passed by value: 4 bytes travel in one register.  Never pass a pointer or reference to a stack
// object into an outlined function: the compiler interleaves the lanes' stacks
// (riscv-stack-word-stride), scales only offsets it knows are frame accesses, and the callee's
// field offsets then land in a sibling lane's slot.
struct Lane {
  uint8_t lane, warp_in_core, core, nwarps;   // nwarps = warps per core
};
static inline Lane lane_of(uint32_t tid, uint32_t nthreads) {
  const uint32_t w = tid / MU_NUM_THREADS;
  return {(uint8_t)(tid % MU_NUM_THREADS), (uint8_t)(w / MU_NUM_CORES), (uint8_t)(w % MU_NUM_CORES),
          (uint8_t)(nthreads / MU_NUM_THREADS / MU_NUM_CORES)};
}

// Byte offset of the scale for (row, 32-group g) in the tile-blocked A-operand scale layout
// [M/TM][K/TK][TK/32][TM] of a matrix with `cols` columns.
static inline uint32_t a_scale_offset(uint32_t row, uint32_t g, uint32_t cols) {
  const uint32_t groups_per_kt = EN_TK / 32, kt = g / groups_per_kt, gi = g % groups_per_kt;
  return (((row / EN_TM) * (cols / EN_TK) + kt) * groups_per_kt + gi) * EN_TM + row % EN_TM;
}

// bf16 fused multiply-add on the low halves (Muon .h is bf16; RNE); the result's upper 16 bits are
// not defined
static inline uint32_t fma_h(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("fmadd.h %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
static inline uint32_t bf16_rne(float f) {
  const uint32_t u = f2u(f);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

// exp(x) without libm: 2^k * poly(r), k = round(x / ln 2) clamped to [-126, 127] so the scale never
// leaves the exponent field.  Rounding through the 1.5 * 2^23 magic constant; FP arithmetic only.
static inline float exp_f(float x) {
  const float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f, MAGIC = 12582912.0f;
  const int k = imax(imin((int)(f2u(x * LOG2E + MAGIC) - 0x4B400000u), 127), -126);
  const float kf = u2f(0x4B400000u + (uint32_t)k) - MAGIC;
  const float r = x - kf * LN2;
  const float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));
  return p * u2f((uint32_t)(k + 127) << 23);
}
// tanh-approximation GELU in its sigmoid form: 0.5x(1 + tanh(z)) = x / (1 + exp(-2z)),
// 2z = u = 1.5957691 (x + 0.044715 x^3).  fp32 fdiv.s costs ~80 cycles per warp instruction on this
// core (issue probe, kernels/tapeout_requant RQ_MODE_ISSUE), so: exp(-u) with the bf16 exp unit
// (fnexp.h, ~5.5 cycles), 1/(1 + e) with the bf16 divider (fdiv.h, ~5.2) plus one fp32 Newton step.
// u is clamped at -80 so exp(-u) stays finite in bf16.  Error vs the exact formula is below one
// bf16 ulp of the result for |u| < ~4 and shrinks in absolute terms beyond.
static inline float gelu_f(float x) {
  float u = x * fma_f(0.07135481627f, x * x, 1.5957691216057308f);
  const uint32_t ub = f2u(u);
  const uint32_t low = mask_lt(0xC2A00000u, ub) & mask_lt(0x7FFFFFFFu, ub);   // u < -80 (negative, |u| > 80)
  const uint32_t uh = (sel(low, ub, 0xC2A00000u) + 0x7FFFu + ((ub >> 16) & 1u)) >> 16;   // bf16 RNE
  uint32_t eh;
  asm("fnexp.h %0, %1" : "=r"(eh) : "r"(uh));
  const float d = 1.0f + u2f(eh << 16);
  uint32_t rh;
  asm("fdiv.h %0, %1, %2" : "=r"(rh) : "r"(0x3F80u), "r"(f2u(d) >> 16));
  const float r0 = u2f(rh << 16);
  float t;   // 2 - d r0
  asm("fnmsub.s %0, %1, %2, %3" : "=r"(t) : "r"(d), "r"(r0), "r"(2.0f));
  return x * (r0 * t);
}

// GELU in bf16 arithmetic on one bf16 value (low 16 bits of x; upper bits are not read).  Muon's
// ".h" format is bf16 (radiance FPPipe.scala maps FP16 requests to BF16), so
//   g = x / (1 + exp(-u)),  u = x (1.59375 + 0.0713 x^2)   (constants rounded to bf16)
// costs 7 FPU instructions.  u is clamped at -80 (fmax.h; the fp32 short-op defect does not affect
// half ops) so exp(-u) stays finite.  Error: a few bf16 ulps, far below the e4m3 step that follows.
#define GELU_C1 0x3FCCu   // 1.5957691 in bf16
#define GELU_C2 0x3D92u   // 0.0713548 in bf16
#define GELU_LO 0xC2A0u   // -80
#ifdef EN_GELU_BREAK   // check falsification: 1 + exp(-u) becomes 2 + exp(-u)
#define BF16_ONE 0x4000u
#else
#define BF16_ONE 0x3F80u
#endif
static inline uint32_t gelu_bf16(uint32_t x) {
  uint32_t x2, u, e, d, g;
  asm("fmul.h %0, %1, %1" : "=r"(x2) : "r"(x));
  asm("fmadd.h %0, %1, %2, %3" : "=r"(u) : "r"(x2), "r"(GELU_C2), "r"(GELU_C1));
  asm("fmul.h %0, %1, %2" : "=r"(u) : "r"(u), "r"(x));
  asm("fmax.h %0, %1, %2" : "=r"(u) : "r"(u), "r"(GELU_LO));
  asm("fnexp.h %0, %1" : "=r"(e) : "r"(u));
  asm("fadd.h %0, %1, %2" : "=r"(d) : "r"(e), "r"(BF16_ONE));
#ifdef EN_GELU_RCP
  // EN_GELU_RCP: fdiv.h issues once per ~20 cycles per core (EN_TEST 24 uprobe, fyv: 2048 per warp
  // x 3 warps in 122.9k cycles; fmadd.h ~2 per cycle), so x / d is x * (1 / d) with the reciprocal
  // from an integer seed (0x7EF3 - bits(d), ~12% error) and two Newton steps r = r (2 - d r) in bf16.
  // d = 1 + exp(-u) is in [1, exp(80)], so the seed stays positive and normal.  Within ~1 bf16 ulp
  // of the correctly rounded quotient.
  uint32_t r, t;
  asm("sub %0, %1, %2" : "=r"(r) : "r"(0x7EF3u), "r"(and_asm(d, 0xFFFFu)));
  asm("fnmsub.h %0, %1, %2, %3" : "=r"(t) : "r"(d), "r"(r), "r"(0x4000u));
  asm("fmul.h %0, %1, %2" : "=r"(r) : "r"(r), "r"(t));
  asm("fnmsub.h %0, %1, %2, %3" : "=r"(t) : "r"(d), "r"(r), "r"(0x4000u));
  asm("fmul.h %0, %1, %2" : "=r"(r) : "r"(r), "r"(t));
  asm("fmul.h %0, %1, %2" : "=r"(g) : "r"(x), "r"(r));
#else
  asm("fdiv.h %0, %1, %2" : "=r"(g) : "r"(x), "r"(d));
#endif
  return g;
}
// Two bf16 values of one word.
static inline uint32_t gelu_bf16x2(uint32_t w) {
  const uint32_t lo = gelu_bf16(w), hi = gelu_bf16(w >> 16);
  return and_asm(lo, 0xFFFFu) | (hi << 16);
}
// EN_GELU_ILP: the 8 bf16 values of 4 words, each GELU stage issued for all 8 before the next.
// The in-order core otherwise runs one ~140-cycle dependent chain (fmul, fmadd, fmul, fmax, fnexp,
// fadd, fdiv) at a time per value pair (metasim ms_t18ug disassembly); the same operations per
// value, so bit-identical to gelu_bf16x2 (with EN_GELU_RCP, to its reciprocal form).
#ifdef EN_GELU_ASM8
// EN_GELU_ASM8: the same 7 operations per value as gelu_bf16 (bit-identical), for 8 values in one asm
// block, stage-major, every stage writing the other of two 8-register banks (a, b).  The Muon
// hazard unit (radiance Hazard.scala) keeps an instruction out of the reservation station while
// its destination register has a pending write (WAW) or a pending read (WAR) and admits each warp
// in order, so the compiler's in-place register reuse (a = f(a), and reuse of a register right
// after its last read) serialized each warp on one ~140-cycle chain: perf counters (EN_FX_HPM,
// board fzhp1) show ~60% WAR and ~40% WAW stall cycles per warp in the UP+GELU epilogue, IPC 0.53.
// Here a register is rewritten only two stages after the stage that read it.
#ifdef EN_GELU_RCP
// With EN_GELU_RCP: the reciprocal form of gelu_bf16 (integer seed 0x7EF3 - d, two Newton steps
// r = r (2 - d r), g = x r), 4 values per block in three 4-register banks (a, b, c): no divider.
__attribute__((always_inline)) static inline void gelu_bf16x4(uint32_t &w0, uint32_t &w1) {
  uint32_t h0, h1, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3;
#define G4_S(op, d, s1, s2) op " %[" #d "0], %[" #s1 "0]" s2 "\n\t" op " %[" #d "1], %[" #s1 "1]" s2 "\n\t" \
  op " %[" #d "2], %[" #s1 "2]" s2 "\n\t" op " %[" #d "3], %[" #s1 "3]" s2 "\n\t"
  asm("srli %[h0], %[w0], 16\n\tsrli %[h1], %[w1], 16\n\t"
      "fmul.h %[a0], %[w0], %[w0]\n\tfmul.h %[a1], %[h0], %[h0]\n\tfmul.h %[a2], %[w1], %[w1]\n\tfmul.h %[a3], %[h1], %[h1]\n\t"
      G4_S("fmadd.h", b, a, ", %[gc2], %[gc1]")
      "fmul.h %[a0], %[b0], %[w0]\n\tfmul.h %[a1], %[b1], %[h0]\n\tfmul.h %[a2], %[b2], %[w1]\n\tfmul.h %[a3], %[b3], %[h1]\n\t"
      G4_S("fmax.h", b, a, ", %[lo]")
      G4_S("fnexp.h", a, b, "")
      G4_S("fadd.h", b, a, ", %[one]")            // d in b
      G4_S("slli", c, b, ", 16")
      G4_S("srli", c, c, ", 16")
      "sub %[a0], %[k], %[c0]\n\tsub %[a1], %[k], %[c1]\n\tsub %[a2], %[k], %[c2]\n\tsub %[a3], %[k], %[c3]\n\t"   // r in a
      // two Newton steps: c = 2 - d r (fnmsub.h rd, d, r, 2), r = r c
      "fnmsub.h %[c0], %[b0], %[a0], %[two]\n\tfnmsub.h %[c1], %[b1], %[a1], %[two]\n\t"
      "fnmsub.h %[c2], %[b2], %[a2], %[two]\n\tfnmsub.h %[c3], %[b3], %[a3], %[two]\n\t"
      "fmul.h %[a0], %[a0], %[c0]\n\tfmul.h %[a1], %[a1], %[c1]\n\tfmul.h %[a2], %[a2], %[c2]\n\tfmul.h %[a3], %[a3], %[c3]\n\t"
      "fnmsub.h %[c0], %[b0], %[a0], %[two]\n\tfnmsub.h %[c1], %[b1], %[a1], %[two]\n\t"
      "fnmsub.h %[c2], %[b2], %[a2], %[two]\n\tfnmsub.h %[c3], %[b3], %[a3], %[two]\n\t"
      "fmul.h %[a0], %[a0], %[c0]\n\tfmul.h %[a1], %[a1], %[c1]\n\tfmul.h %[a2], %[a2], %[c2]\n\tfmul.h %[a3], %[a3], %[c3]\n\t"
      // g = x r (into b), then w_j = lo16(g_2j) | g_2j+1 << 16
      "fmul.h %[b0], %[w0], %[a0]\n\tfmul.h %[b1], %[h0], %[a1]\n\tfmul.h %[b2], %[w1], %[a2]\n\tfmul.h %[b3], %[h1], %[a3]\n\t"
      "slli %[b0], %[b0], 16\n\tslli %[b2], %[b2], 16\n\tslli %[b1], %[b1], 16\n\tslli %[b3], %[b3], 16\n\t"
      "srli %[b0], %[b0], 16\n\tsrli %[b2], %[b2], 16\n\t"
      "or %[w0], %[b0], %[b1]\n\tor %[w1], %[b2], %[b3]"
      : [w0] "+r"(w0), [w1] "+r"(w1), [h0] "=&r"(h0), [h1] "=&r"(h1),
        [a0] "=&r"(a0), [a1] "=&r"(a1), [a2] "=&r"(a2), [a3] "=&r"(a3),
        [b0] "=&r"(b0), [b1] "=&r"(b1), [b2] "=&r"(b2), [b3] "=&r"(b3),
        [c0] "=&r"(c0), [c1] "=&r"(c1), [c2] "=&r"(c2), [c3] "=&r"(c3)
      : [gc2] "r"(GELU_C2), [gc1] "r"(GELU_C1), [lo] "r"(GELU_LO), [one] "r"(BF16_ONE), [k] "r"(0x7EF3u), [two] "r"(0x4000u));
#undef G4_S
}
__attribute__((always_inline)) static inline void gelu_bf16x8(uint32_t &w0, uint32_t &w1, uint32_t &w2, uint32_t &w3) {
  gelu_bf16x4(w0, w1);
  gelu_bf16x4(w2, w3);
}
#else
__attribute__((always_inline)) static inline void gelu_bf16x8(uint32_t &w0, uint32_t &w1, uint32_t &w2, uint32_t &w3) {
  uint32_t h0, h1, h2, h3, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;
#define G8_S(op, d, s1, s2) op " %[" #d "0], %[" #s1 "0]" s2 "\n\t" op " %[" #d "1], %[" #s1 "1]" s2 "\n\t" \
  op " %[" #d "2], %[" #s1 "2]" s2 "\n\t" op " %[" #d "3], %[" #s1 "3]" s2 "\n\t" \
  op " %[" #d "4], %[" #s1 "4]" s2 "\n\t" op " %[" #d "5], %[" #s1 "5]" s2 "\n\t" \
  op " %[" #d "6], %[" #s1 "6]" s2 "\n\t" op " %[" #d "7], %[" #s1 "7]" s2 "\n\t"
  asm("srli %[h0], %[w0], 16\n\tsrli %[h1], %[w1], 16\n\tsrli %[h2], %[w2], 16\n\tsrli %[h3], %[w3], 16\n\t"
      // x_k: w0, h0, w1, h1, w2, h2, w3, h3 (fmul.h etc. read the low 16 bits)
      "fmul.h %[a0], %[w0], %[w0]\n\tfmul.h %[a1], %[h0], %[h0]\n\tfmul.h %[a2], %[w1], %[w1]\n\tfmul.h %[a3], %[h1], %[h1]\n\t"
      "fmul.h %[a4], %[w2], %[w2]\n\tfmul.h %[a5], %[h2], %[h2]\n\tfmul.h %[a6], %[w3], %[w3]\n\tfmul.h %[a7], %[h3], %[h3]\n\t"
      G8_S("fmadd.h", b, a, ", %[c2], %[c1]")
      "fmul.h %[a0], %[b0], %[w0]\n\tfmul.h %[a1], %[b1], %[h0]\n\tfmul.h %[a2], %[b2], %[w1]\n\tfmul.h %[a3], %[b3], %[h1]\n\t"
      "fmul.h %[a4], %[b4], %[w2]\n\tfmul.h %[a5], %[b5], %[h2]\n\tfmul.h %[a6], %[b6], %[w3]\n\tfmul.h %[a7], %[b7], %[h3]\n\t"
      G8_S("fmax.h", b, a, ", %[lo]")
      G8_S("fnexp.h", a, b, "")
      G8_S("fadd.h", b, a, ", %[one]")
      "fdiv.h %[a0], %[w0], %[b0]\n\tfdiv.h %[a1], %[h0], %[b1]\n\tfdiv.h %[a2], %[w1], %[b2]\n\tfdiv.h %[a3], %[h1], %[b3]\n\t"
      "fdiv.h %[a4], %[w2], %[b4]\n\tfdiv.h %[a5], %[h2], %[b5]\n\tfdiv.h %[a6], %[w3], %[b6]\n\tfdiv.h %[a7], %[h3], %[b7]\n\t"
      // w_j = lo16(a_2j) | a_2j+1 << 16 (in place: these registers wait only on their own fdiv.h)
      "slli %[a0], %[a0], 16\n\tslli %[a2], %[a2], 16\n\tslli %[a4], %[a4], 16\n\tslli %[a6], %[a6], 16\n\t"
      "srli %[a0], %[a0], 16\n\tsrli %[a2], %[a2], 16\n\tsrli %[a4], %[a4], 16\n\tsrli %[a6], %[a6], 16\n\t"
      "slli %[a1], %[a1], 16\n\tslli %[a3], %[a3], 16\n\tslli %[a5], %[a5], 16\n\tslli %[a7], %[a7], 16\n\t"
      "or %[w0], %[a0], %[a1]\n\tor %[w1], %[a2], %[a3]\n\tor %[w2], %[a4], %[a5]\n\tor %[w3], %[a6], %[a7]"
      : [w0] "+r"(w0), [w1] "+r"(w1), [w2] "+r"(w2), [w3] "+r"(w3),
        [h0] "=&r"(h0), [h1] "=&r"(h1), [h2] "=&r"(h2), [h3] "=&r"(h3),
        [a0] "=&r"(a0), [a1] "=&r"(a1), [a2] "=&r"(a2), [a3] "=&r"(a3), [a4] "=&r"(a4), [a5] "=&r"(a5), [a6] "=&r"(a6), [a7] "=&r"(a7),
        [b0] "=&r"(b0), [b1] "=&r"(b1), [b2] "=&r"(b2), [b3] "=&r"(b3), [b4] "=&r"(b4), [b5] "=&r"(b5), [b6] "=&r"(b6), [b7] "=&r"(b7)
      : [c2] "r"(GELU_C2), [c1] "r"(GELU_C1), [lo] "r"(GELU_LO), [one] "r"(BF16_ONE));
#undef G8_S
}
#endif  // EN_GELU_RCP
#else
__attribute__((always_inline)) static inline void gelu_bf16x8(uint32_t &w0, uint32_t &w1, uint32_t &w2, uint32_t &w3) {
  uint32_t x[8] = {w0, w0 >> 16, w1, w1 >> 16, w2, w2 >> 16, w3, w3 >> 16};
  uint32_t a[8], b[8];
#pragma unroll
  for (uint32_t j = 0; j < 8u; j++) asm("fmul.h %0, %1, %1" : "=r"(a[j]) : "r"(x[j]));
#pragma unroll
  for (uint32_t j = 0; j < 8u; j++) asm("fmadd.h %0, %1, %2, %3" : "=r"(a[j]) : "r"(a[j]), "r"(GELU_C2), "r"(GELU_C1));
#pragma unroll
  for (uint32_t j = 0; j < 8u; j++) asm("fmul.h %0, %1, %2" : "=r"(a[j]) : "r"(a[j]), "r"(x[j]));
#pragma unroll
  for (uint32_t j = 0; j < 8u; j++) asm("fmax.h %0, %1, %2" : "=r"(a[j]) : "r"(a[j]), "r"(GELU_LO));
#pragma unroll
  for (uint32_t j = 0; j < 8u; j++) asm("fnexp.h %0, %1" : "=r"(a[j]) : "r"(a[j]));
#pragma unroll
  for (uint32_t j = 0; j < 8u; j++) asm("fadd.h %0, %1, %2" : "=r"(a[j]) : "r"(a[j]), "r"(BF16_ONE));
#ifdef EN_GELU_RCP
#pragma unroll
#ifdef EN_GELU_SEEDNOMASK
  // EN_GELU_SEEDNOMASK: the FPU sign-extends half results to 32 bits (radiance FPPipe.scala
  // signExtendFp16Lanes) and d = 1 + exp(-u) > 0, so its upper 16 bits are already zero
  for (uint32_t j = 0; j < 8u; j++) asm("sub %0, %1, %2" : "=r"(b[j]) : "r"(0x7EF3u), "r"(a[j]));
#else
  for (uint32_t j = 0; j < 8u; j++) asm("sub %0, %1, %2" : "=r"(b[j]) : "r"(0x7EF3u), "r"(and_asm(a[j], 0xFFFFu)));
#endif
#pragma unroll
  for (uint32_t s = 0; s < 2u; s++) {
    uint32_t t[8];
#pragma unroll
    for (uint32_t j = 0; j < 8u; j++) asm("fnmsub.h %0, %1, %2, %3" : "=r"(t[j]) : "r"(a[j]), "r"(b[j]), "r"(0x4000u));
#pragma unroll
    for (uint32_t j = 0; j < 8u; j++) asm("fmul.h %0, %1, %2" : "=r"(b[j]) : "r"(b[j]), "r"(t[j]));
  }
#pragma unroll
  for (uint32_t j = 0; j < 8u; j++) asm("fmul.h %0, %1, %2" : "=r"(b[j]) : "r"(x[j]), "r"(b[j]));
#else
#pragma unroll
  for (uint32_t j = 0; j < 8u; j++) asm("fdiv.h %0, %1, %2" : "=r"(b[j]) : "r"(x[j]), "r"(a[j]));
#endif
  w0 = and_asm(b[0], 0xFFFFu) | (b[1] << 16);
  w1 = and_asm(b[2], 0xFFFFu) | (b[3] << 16);
  w2 = and_asm(b[4], 0xFFFFu) | (b[5] << 16);
  w3 = and_asm(b[6], 0xFFFFu) | (b[7] << 16);
}
#endif  // EN_GELU_ASM8

// ================================ SIMT stages ====================================================
// Global memory: every warp load/store instruction covers one 64-byte line (16 lanes x one word,
// consecutive), and every output line is written whole by one warp.  No sub-word global stores.
// Compute runs on SMEM staging with one lane per 32-element group; strides are padded so the 16
// lanes of one access hit 16 different subbanks.  Measured reasons (board 09): lane-per-group
// global access touched 16 lines per instruction and took LN1 to 2.2M cycles, and byte stores
// from several warps into one line lost data (split stage, 2026-10-12).
//
// SMEM map (cluster-shared, 128 KB; each launch owns all of it):
#define SM_GAMMA   0x00000u   // f32 [D], word w at w + w/32
#define SM_BETA    0x00840u
#define SM_WARP0   0x01100u   // per-warp staging, slot = core * nwarps + warp_in_core
#define SM_WARP_SZ 0x3400u
#define SM_SC      0x15000u   // row-scale staging: byte (group g, row r) at g * SC_STRIDE + r
#define SC_STRIDE  260u       // EN_S + 4: consecutive groups land in different subbanks

template <typename T> static inline __shared T *smem(uint32_t addr) {
  return reinterpret_cast<__shared T *>(addr);
}
static inline uint32_t warp_slot(Lane L) { return L.core * L.nwarps + L.warp_in_core; }
static inline uint32_t total_warps(Lane L) { return L.nwarps * MU_NUM_CORES; }
// Row of the r-th row of this core's 64-row blocks (block b belongs to core b % 2).
static inline uint32_t core_row(Lane L, uint32_t j) { return ((j / 64u) * 2u + L.core) * 64u + j % 64u; }

#ifdef EN_RES_HADD
// EN_RES_HADD: two bf16 adds on the half FPU (.h is bf16 on Muon; reads the low 16 bits only)
// instead of fp32 add + integer RNE per half.  The fp32 sum of two bf16 values is exact unless
// their exponents differ by more than 16, so both forms give the correctly rounded bf16 sum
// except in double-rounding corner cases; EN_HASH_OUT compares the full buffers.
static inline uint32_t bf16x2_add(uint32_t a, uint32_t b) {
  uint32_t lo, hi;
  asm("fadd.h %0, %1, %2" : "=r"(lo) : "r"(a), "r"(b));
  asm("fadd.h %0, %1, %2" : "=r"(hi) : "r"(a >> 16), "r"(b >> 16));
  return and_asm(lo, 0xFFFFu) | (hi << 16);
}
#else
static inline uint32_t bf16x2_add(uint32_t a, uint32_t b) {
  return bf16_rne(bf16_to_f(a & 0xFFFFu) + bf16_to_f(b & 0xFFFFu)) |
         (bf16_rne(bf16_to_f(a >> 16) + bf16_to_f(b >> 16)) << 16);
}
#endif

// After a cluster barrier: copy this core's row-scale staging to the tile-blocked A-operand
// scale layout.  For a 64-row block and group g the 64 scales are one 64-byte line.
static void scales_out(Lane L, uint32_t groups, uint32_t cols, uint32_t out_s_addr) {
  const __shared uint32_t *sc = smem<uint32_t>(SM_SC);
  uint32_t *dst = reinterpret_cast<uint32_t *>(out_s_addr);
  for (uint32_t line = L.warp_in_core; line < 2u * groups; line += L.nwarps) {
    const uint32_t g = line % groups, blk = (line / groups) * 2u + L.core;
    dst[a_scale_offset(blk * 64u, g, cols) / 4u + L.lane] = sc[(g * SC_STRIDE + blk * 64u) / 4u + L.lane];
  }
}

// ---- row quantization core: 4 rows x 512 columns in SMEM -------------------------------------
// xb: 4 rows, stride RQ_XS words, word w at w + w/16 (lane l's group = words 16l..16l+15 at 17l)
// ob: 4 rows of codes, stride RQ_OS words, word w at w + w/8 (lane l's 8 words at 9l)
#define RQ_XS 272u
#define RQ_OS 144u

// Codes of 4 rows (128 words each) -> DRAM rows starting at `dst` with row stride `ld` words.
static inline void codes_out(uint32_t lane, const __shared uint32_t *ob, uint32_t *dst, uint32_t ld) {
#pragma unroll
  for (uint32_t rr = 0; rr < 4u; rr++)
#pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
      const uint32_t w = 16u * k + lane;
      dst[rr * ld + w] = ob[rr * RQ_OS + w + 2u * k + lane / 8u];
    }
}
// SWAR e4m3 encode of two bf16 values (one word), block-scaled by 2^-e: codes at bits 0..7 and
// 16..23.  Same codes as e4m3_encode(x * 2^-e) for normal bf16 x: with A = |x| bits (15 wide)
// that rule is c = ((A + 7 + lsb) >> 4) - (8 e + 960), lsb = bit 4 of A, and c < 8 -> 0.
// p2 holds 0x1000 - (8 e + 968) in both halves: bit 12 of a field of t + p2 is set iff c >= 8,
// and that field + 8 has c in its low byte.  17 instructions for two values.
static inline uint32_t e4m3_swar2(uint32_t w, uint32_t p2) {
  const uint32_t a = w & 0x7FFF7FFFu;
  uint32_t t = a + 0x00070007u + ((a >> 4) & 0x00010001u);
  t = ((t >> 4) & 0x0FFF0FFFu) + p2;
  const uint32_t k = (t >> 12) & 0x00010001u;
  return ((t + 0x00080008u) | ((w >> 8) & 0x00800080u)) & ((k << 8) - k);
}
static inline uint32_t fmax_h(uint32_t a, uint32_t b) {   // bf16 max of the low halves
  uint32_t r;
  asm("fmax.h %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
static inline uint32_t swar_p(int e) { return 0x1000u - (uint32_t)(8 * e + 968); }   // one half of p2
static inline uint32_t swar_p2(int e) { const uint32_t p = swar_p(e); return p | (p << 16); }
// Two e4m3_swar2 results (4 codes at bits 0, 16 of r0 and r1) -> one little-endian code word.
static inline uint32_t swar_pack4(uint32_t r0, uint32_t r1) {
  return ((r0 | (r0 >> 8)) & 0xFFFFu) | ((r1 | (r1 >> 8)) << 16);
}
// Banked-register forms of the block max and the SWAR encoder (see EN_GELU_ASM8 for why: the hazard
// unit gates RS admission on WAR / WAW of the destination register).  Same operations, same
// results.  swar_max4: m_j = max(m_j, |lo(w_j)|, |hi(w_j)|) on the bf16 FPU, four independent
// accumulators (the block max is order-independent: all inputs are finite magnitudes).
__attribute__((always_inline)) static inline void swar_max4(uint32_t &m0, uint32_t &m1, uint32_t &m2, uint32_t &m3,
                                                            uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
  uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
  asm("and %[a0], %[w0], %[m1k]\n\tand %[a1], %[w1], %[m1k]\n\tand %[a2], %[w2], %[m1k]\n\tand %[a3], %[w3], %[m1k]\n\t"
      "srli %[b0], %[a0], 16\n\tsrli %[b1], %[a1], 16\n\tsrli %[b2], %[a2], 16\n\tsrli %[b3], %[a3], 16\n\t"
      "fmax.h %[b0], %[a0], %[b0]\n\tfmax.h %[b1], %[a1], %[b1]\n\tfmax.h %[b2], %[a2], %[b2]\n\tfmax.h %[b3], %[a3], %[b3]\n\t"
      "fmax.h %[m0], %[m0], %[b0]\n\tfmax.h %[m1], %[m1], %[b1]\n\tfmax.h %[m2], %[m2], %[b2]\n\tfmax.h %[m3], %[m3], %[b3]"
      : [m0] "+r"(m0), [m1] "+r"(m1), [m2] "+r"(m2), [m3] "+r"(m3),
        [a0] "=&r"(a0), [a1] "=&r"(a1), [a2] "=&r"(a2), [a3] "=&r"(a3),
        [b0] "=&r"(b0), [b1] "=&r"(b1), [b2] "=&r"(b2), [b3] "=&r"(b3)
      : [w0] "r"(w0), [w1] "r"(w1), [w2] "r"(w2), [w3] "r"(w3), [m1k] "r"(0x7FFF7FFFu));
}
// e4m3_enc4: e4m3_swar2 of four words and swar_pack4 of the results: o0 = pack(w0, w1), o1 = pack(w2, w3).
// Stage-major over the four words, each stage writing a bank (A, B, C, D) no in-flight
// instruction of the previous stage still reads.
__attribute__((always_inline)) static inline void e4m3_enc4(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t p2,
                                                            uint32_t &o0, uint32_t &o1) {
  uint32_t A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3, D0, D1, D2, D3;
#define E4_S(op, d, s1, s2) op " %[" #d "0], %[" #s1 "0]" s2 "\n\t" op " %[" #d "1], %[" #s1 "1]" s2 "\n\t" \
  op " %[" #d "2], %[" #s1 "2]" s2 "\n\t" op " %[" #d "3], %[" #s1 "3]" s2 "\n\t"
#define E4_T(op, d, s1, s2) op " %[" #d "0], %[" #s1 "0], %[" #s2 "0]\n\t" op " %[" #d "1], %[" #s1 "1], %[" #s2 "1]\n\t" \
  op " %[" #d "2], %[" #s1 "2], %[" #s2 "2]\n\t" op " %[" #d "3], %[" #s1 "3], %[" #s2 "3]\n\t"
  asm(E4_S("and", A, w, ", %[km1]")      // a = |w|
      E4_S("srli", B, A, ", 4")
      E4_S("and", C, B, ", %[kc1]")      // lsb
      E4_S("add", B, A, ", %[kc7]")
      E4_T("add", A, B, C)               // a + 7 + lsb
      E4_S("srli", C, A, ", 4")
      E4_S("and", B, C, ", %[km2]")
      E4_S("add", A, B, ", %[p2]")       // t
      E4_S("srli", B, A, ", 12")
      E4_S("and", C, B, ", %[kc1]")      // k
      E4_S("slli", B, C, ", 8")
      E4_T("sub", D, B, C)               // (k << 8) - k
      E4_S("add", B, A, ", %[kc8]")      // t + 8
      E4_S("srli", C, w, ", 8")
      E4_S("and", A, C, ", %[ksg]")      // sign bits
      E4_T("or", C, B, A)
      E4_T("and", B, C, D)               // r = e4m3_swar2(w)
      E4_S("srli", C, B, ", 8")
      E4_T("or", D, B, C)                // codes at bits 0..15 (garbage above for 0, 2)
      "slli %[D0], %[D0], 16\n\tslli %[D2], %[D2], 16\n\tslli %[D1], %[D1], 16\n\tslli %[D3], %[D3], 16\n\t"
      "srli %[D0], %[D0], 16\n\tsrli %[D2], %[D2], 16\n\t"
      "or %[o0], %[D0], %[D1]\n\tor %[o1], %[D2], %[D3]"
      : [o0] "=&r"(o0), [o1] "=&r"(o1),
        [A0] "=&r"(A0), [A1] "=&r"(A1), [A2] "=&r"(A2), [A3] "=&r"(A3), [B0] "=&r"(B0), [B1] "=&r"(B1), [B2] "=&r"(B2), [B3] "=&r"(B3),
        [C0] "=&r"(C0), [C1] "=&r"(C1), [C2] "=&r"(C2), [C3] "=&r"(C3), [D0] "=&r"(D0), [D1] "=&r"(D1), [D2] "=&r"(D2), [D3] "=&r"(D3)
      : [w0] "r"(w0), [w1] "r"(w1), [w2] "r"(w2), [w3] "r"(w3), [p2] "r"(p2), [km1] "r"(0x7FFF7FFFu), [kc1] "r"(0x00010001u),
        [kc7] "r"(0x00070007u), [km2] "r"(0x0FFF0FFFu), [kc8] "r"(0x00080008u), [ksg] "r"(0x00800080u));
#undef E4_S
#undef E4_T
}
// Quantize lane l's group (32 bf16 at xg[0..15]) into og[0..7]; returns the scale code.
static inline uint32_t quant_group_bf16(const __shared uint32_t *xg, __shared uint32_t *og) {
  // One batch of independent SMEM loads (their latencies overlap).  Through a volatile pointer:
  // otherwise LLVM emits memcpy into a stack array, which the strided stack corrupts.
  const volatile __shared uint32_t *xv = xg;
  uint32_t xw[16];
#pragma unroll
  for (uint32_t i = 0; i < 16u; i++) xw[i] = xv[i];
#ifdef EN_QUANT_FP32
  uint32_t amax = 0;   // |x| bits; bf16 magnitudes order like the values
#pragma unroll
  for (uint32_t i = 0; i < 16u; i++)
    amax = umax(amax, umax(and_asm(xw[i] << 16, 0x7FFFFFFFu), and_asm(xw[i], 0x7FFF0000u)));
  const int e = block_exp(amax);
  const float inv = block_inv(e);
#pragma unroll
  for (uint32_t i = 0; i < 8u; i++)
    og[i] = e4m3_pack4(bf16_to_f(xw[2u * i] & 0xFFFFu) * inv, bf16_to_f(xw[2u * i] >> 16) * inv,
                       bf16_to_f(xw[2u * i + 1u] & 0xFFFFu) * inv, bf16_to_f(xw[2u * i + 1u] >> 16) * inv);
#else
  // block max with the bf16 FPU (half ops are not affected by the fp32 short-op defect)
  uint32_t mx = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16u; i++) {
    const uint32_t a = xw[i] & 0x7FFF7FFFu;
    mx = fmax_h(fmax_h(mx, a), a >> 16);
  }
  const int e = block_exp(mx << 16);
  const uint32_t p2 = swar_p2(e);
#pragma unroll
  for (uint32_t i = 0; i < 8u; i++) {
    og[i] = swar_pack4(e4m3_swar2(xw[2u * i], p2), e4m3_swar2(xw[2u * i + 1u], p2));
  }
#endif
  return (uint32_t)(e + 127);
}

// ---- stage: QKV split into the FA_ENC operand layouts ------------------------------------------
// Units (tensor, head, 64-row block): 48, dealt round-robin to the warps of both cores.  Every
// output line belongs to exactly one unit.  Q and K are quantized along d (groups of 32 columns),
// V along keys (groups of 32 rows).  Lane mapping for Q/K: lane l owns group g = l % 4 of the 4
// rows 4q..4q+3 (q = l / 4) of a 16-row sub-tile, so its 4 scales pack into one word and, for K,
// the codes of 4 consecutive keys pack into one K^T word.
#define SP_XS 65u    // sub-tile row stride (words): 64 + 1
#define SP_OS 41u    // Q code row stride (words): word 8g + i at 9g + i
#define SP_WARP_SZ 0x4400u   // per-warp SMEM for this stage (it uses neither gamma nor SM_SC)
__attribute__((noinline)) static void qkv_split(Lane L) {
  const uint32_t l = L.lane;
  const uint32_t gw = L.warp_in_core * MU_NUM_CORES + L.core;
  __shared uint32_t *xb = smem<uint32_t>(warp_slot(L) * SP_WARP_SZ);   // 32 x SP_XS
  __shared uint32_t *stg = xb + 32u * SP_XS;                                       // codes
  __shared uint32_t *ssc = stg + 2112u;                                            // scales
  const uint32_t *qkv = reinterpret_cast<const uint32_t *>(EN_QKV);
  constexpr uint32_t LDW = 3u * EN_D / 2u;   // qkv row stride in words
  const uint32_t g = l % 4u, q = l / 4u;
  for (uint32_t u = gw; u < 48u; u += total_warps(L)) {
    const uint32_t t = u / 16u, h = (u / 4u) % 4u, blk = u % 4u;
    const uint32_t col0 = (t * EN_D + h * EN_DH) / 2u;   // first word of this head's slice
    if (t < 2u) {
      for (uint32_t s = 0; s < 4u; s++) {   // 16-row sub-tiles
        const uint32_t row0 = blk * 64u + s * 16u;
#pragma unroll 4
        for (uint32_t rr = 0; rr < 16u; rr++)
#pragma unroll
          for (uint32_t k = 0; k < 4u; k++)
            xb[rr * SP_XS + 16u * k + l] = qkv[(row0 + rr) * LDW + col0 + 16u * k + l];
        mu_fence_smem();
        float inv[4];
        uint32_t scw = 0;
#pragma unroll
        for (uint32_t m = 0; m < 4u; m++) {
          const __shared uint32_t *xg = xb + (4u * q + m) * SP_XS + 16u * g;
          uint32_t amax = 0;
#pragma unroll
          for (uint32_t i = 0; i < 16u; i++)
            amax = umax(amax, umax(and_asm(xg[i] << 16, 0x7FFFFFFFu), and_asm(xg[i], 0x7FFF0000u)));
          const int e = block_exp(amax);
          inv[m] = block_inv(e);
          scw |= (uint32_t)(e + 127) << (8u * m);
        }
        ssc[g * 16u + s * 4u + q] = scw;   // scales of rows 4q..4q+3 (row % 64 order)
        if (t == 0u) {
#pragma unroll
          for (uint32_t m = 0; m < 4u; m++) {
            const __shared uint32_t *xg = xb + (4u * q + m) * SP_XS + 16u * g;
            __shared uint32_t *og = stg + (4u * q + m) * SP_OS + 9u * g;
#pragma unroll
            for (uint32_t i = 0; i < 8u; i++) {
              const uint32_t w0 = xg[2u * i], w1 = xg[2u * i + 1u];
              og[i] = e4m3_pack4(bf16_to_f(w0 & 0xFFFFu) * inv[m], bf16_to_f(w0 >> 16) * inv[m],
                                 bf16_to_f(w1 & 0xFFFFu) * inv[m], bf16_to_f(w1 >> 16) * inv[m]);
            }
          }
          mu_fence_smem();
          uint32_t *qd = reinterpret_cast<uint32_t *>(EN_FA_Q) + (h * EN_S + row0) * (EN_DH / 4u);
#pragma unroll 4
          for (uint32_t rr = 0; rr < 16u; rr++)
#pragma unroll
            for (uint32_t k = 0; k < 2u; k++) {
              const uint32_t w = 16u * k + l;
              qd[rr * (EN_DH / 4u) + w] = stg[rr * SP_OS + 9u * (w / 8u) + w % 8u];
            }
        } else {
          // K^T staging: word (d, key/4) at d * 16 + (d / 32) * 4 + key / 4, key = 16s + 4q + m
#pragma unroll 4
          for (uint32_t i = 0; i < 16u; i++) {
            uint32_t lo = 0, hi = 0;
#pragma unroll
            for (uint32_t m = 0; m < 4u; m++) {
              const uint32_t w = xb[(4u * q + m) * SP_XS + 16u * g + i];
              lo |= e4m3_encode(bf16_to_f(w & 0xFFFFu) * inv[m]) << (8u * m);
              hi |= e4m3_encode(bf16_to_f(w >> 16) * inv[m]) << (8u * m);
            }
            const uint32_t d = 32u * g + 2u * i;
            stg[d * 16u + g * 4u + s * 4u + q] = lo;
            stg[(d + 1u) * 16u + g * 4u + s * 4u + q] = hi;
          }
        }
        mu_fence_smem();
      }
      // scales: Q -> QS[h][blk][g][row % 64], K -> KTS[h][g][key]; staging word g * 16 + row/4
      if (t == 0u) {
        uint32_t *qs = reinterpret_cast<uint32_t *>(EN_FA_QS) + (h * 4u + blk) * 64u;
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++) qs[16u * k + l] = ssc[16u * k + l];
      } else {
        uint32_t *kts = reinterpret_cast<uint32_t *>(EN_FA_KTS);
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++) kts[(h * 4u + k) * 64u + blk * 16u + l] = ssc[16u * k + l];
        uint32_t *kt = reinterpret_cast<uint32_t *>(EN_FA_KT) + h * EN_DH * (EN_S / 4u) + blk * 16u;
#pragma unroll 4
        for (uint32_t d = 0; d < EN_DH; d++) kt[d * (EN_S / 4u) + l] = stg[d * 16u + (d / 32u) * 4u + l];
      }
    } else {
      // V: two 32-key groups per unit; lane l owns word-columns l + 16k (columns 2wc, 2wc + 1)
      for (uint32_t hb = 0; hb < 2u; hb++) {
        const uint32_t row0 = blk * 64u + hb * 32u;
#pragma unroll 4
        for (uint32_t rr = 0; rr < 32u; rr++)
#pragma unroll
          for (uint32_t k = 0; k < 4u; k++)
            xb[rr * SP_XS + 16u * k + l] = qkv[(row0 + rr) * LDW + col0 + 16u * k + l];
        mu_fence_smem();
        __shared uint16_t *vc = reinterpret_cast<__shared uint16_t *>(stg);   // [32 rows][64 halfwords]
        __shared uint16_t *vs = reinterpret_cast<__shared uint16_t *>(ssc);   // [64 halfwords]
        for (uint32_t k = 0; k < 4u; k++) {
          const uint32_t wc = 16u * k + l;
          uint32_t a0 = 0, a1 = 0;
#pragma unroll 8
          for (uint32_t rr = 0; rr < 32u; rr++) {
            const uint32_t w = xb[rr * SP_XS + wc];
            a0 = umax(a0, and_asm(w << 16, 0x7FFFFFFFu));
            a1 = umax(a1, and_asm(w, 0x7FFF0000u));
          }
          const int e0 = block_exp(a0), e1 = block_exp(a1);
          const float i0 = block_inv(e0), i1 = block_inv(e1);
          vs[wc] = (uint16_t)((uint32_t)(e0 + 127) | ((uint32_t)(e1 + 127) << 8));
#pragma unroll 8
          for (uint32_t rr = 0; rr < 32u; rr++) {
            const uint32_t w = xb[rr * SP_XS + wc];
            vc[rr * 64u + wc] = (uint16_t)(e4m3_encode(bf16_to_f(w & 0xFFFFu) * i0) |
                                           (e4m3_encode(bf16_to_f(w >> 16) * i1) << 8));
          }
        }
        mu_fence_smem();
        uint32_t *vd = reinterpret_cast<uint32_t *>(EN_FA_V) + (h * EN_S + row0) * (EN_DH / 4u);
#pragma unroll 4
        for (uint32_t rr = 0; rr < 32u; rr++)
#pragma unroll
          for (uint32_t k = 0; k < 2u; k++) vd[rr * (EN_DH / 4u) + 16u * k + l] = stg[rr * 32u + 16u * k + l];
        uint32_t *vsd = reinterpret_cast<uint32_t *>(EN_FA_VS) + (h * (EN_S / 32u) + row0 / 32u) * (EN_DH / 4u);
#pragma unroll
        for (uint32_t k = 0; k < 2u; k++) vsd[16u * k + l] = ssc[16u * k + l];
        mu_fence_smem();
      }
    }
  }
}

// ---- DMA between DRAM and SMEM (thread 0 only) ---------------------------------------------------
// SIMT global access runs at ~0.57 B/cycle (coalescer limit; split 1.69M cycles); the Gemmini DMA
// moves 256-byte chunks at ~6 B/cycle and is byte-exact both ways (kernels/tapeout_requant
// RQ_MODE_DMA16, board + metasim).  Scratchpad row r is SMEM byte 16 r.  Every command here is
// 16 B wide; config_ld / config_st set the DRAM row stride.
static inline void dma_cfg_in(uint32_t stride) {
  gemmini_extended3_config_ld(stride, MVIN_SCALE_IDENTITY, false, 0);
}
static inline void dma_cfg_out(uint32_t stride) { gemmini_config_st(stride); }
// rows x 16 B: DRAM rows at the configured stride <-> consecutive SMEM rows from `smem`
static inline void dma_in(uint32_t smem, uint32_t dram, uint32_t rows) {
  gemmini_extended_mvin(rad_device_to_host_address(dram), smem / 16u, 16u, rows);
}
// 2-block move-in: rows x 32 B DRAM rows (at the configured stride) -> block 0 (bytes 0-15) in SMEM
// rows smem/16 .., block 1 (bytes 16-31) DIM rows later.  The tapeout's move-in cols field is 6 bits,
// so 2 blocks is the maximum; 64 KB moves in 3.3x faster than with 1-block commands (EN_TEST 22).
// With stride 32 a contiguous 512-byte group lands chunk-permuted: even 16-byte chunks in the first
// 16 SMEM rows, odd chunks in the next 16.  dma_out2 at the same stride undoes it on the way out.
static inline void dma_in2(uint32_t smem, uint32_t dram) {
  gemmini_extended_mvin(rad_device_to_host_address(dram), smem / 16u, 2u * DIM, 16u);
}
// 2-block move-out, the inverse of dma_in2 at the same stride: SMEM rows smem/16 .. (block 0) and
// DIM rows later (block 1) -> 16 DRAM rows of 32 B.  64 KB moves out 3.4x faster than with 1-block
// commands (EN_TEST 22).  At stride 32 it returns a dma_in2-permuted 512-byte group to contiguous
// DRAM (needs dma_cfg_out(32)).
static inline void dma_out2(uint32_t dram, uint32_t smem) {
  gemmini_extended_mvout(rad_device_to_host_address(dram), smem / 16u, 2u * DIM, 16u);
}
static inline void dma_out(uint32_t dram, uint32_t smem, uint32_t rows) {
  gemmini_extended_mvout(rad_device_to_host_address(dram), smem / 16u, 16u, rows);
}
// Contiguous copy SMEM -> DRAM in 16-row (256 B) commands (needs dma_cfg_out(16)); larger
// commands are unverified (a 64-row probe hung on the board).
static inline void dma_out_contig(uint32_t dram, uint32_t smem, uint32_t bytes) {
  for (uint32_t off = 0; off < bytes; off += 256u)
    dma_out(dram + off, smem + off, bytes - off < 256u ? (bytes - off) / 16u : 16u);
}
static inline void dma_wait() { mxg::drain(); }
static void dma_setup() {
  dma_wait();   // a previous launch's Gemmini work may still be running (soft reset does not stop it)
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              mxg::FORMAT_FP8, mxg::FORMAT_FP8, mxg::FORMAT_FULL, false);
  gemmini_fence();
}

// tid == 0, recomputed by a volatile asm at every use.  A plain `tid == 0` lets LLVM duplicate code per
// branch: it unswitched residual2's round loop on it (and, with only the loop tests opaque, cloned
// the function body at the dma_setup test): lanes 1-15 of warp 0 ran every round (barriers
// included) inside a divergent region before lane 0 ran its copy, and lane 0 then waited at
// barriers no other warp reached (board hang, tp_enc_h13).  A volatile asm cannot be hoisted.
__attribute__((always_inline)) static inline bool is_lead(uint32_t tid) {
  uint32_t r;
  asm volatile("seqz %0, %1" : "=r"(r) : "r"(tid));
  return r != 0;
}

// This cluster's rank in the split (EN_CLUSTERS, enc.h): the cluster id CSR (0xCD0) on the
// two-cluster system; always 0 for a one-cluster build.  Core ids are cluster-local.
__attribute__((always_inline)) static inline uint32_t en_cluster() {
#if EN_CLUSTERS > 1
  uint32_t v;
  asm volatile("csrr %0, %1" : "=r"(v) : "i"(MU_CSR_CLUSTER_ID));
  return v;
#else
  return 0u;
#endif
}

// This cluster's share of a GEMM (EN_CLUSTERS > 1): rows [c M/2, (c+1) M/2) by default, or with
// EN_SPLIT_N columns [c N/2, (c+1) N/2) (each cluster then reads half of B, the larger operand in
// every encoder GEMM, and all of A).  A and B scales are tile-blocked, so a row block of A's or a
// column block of B's scales is contiguous.  n stays the row stride of B and C (Problem::ncols).
static inline mxg::Problem en_split(mxg::Problem p) {
  const uint32_t c0 = p.c;
#if EN_CLUSTERS > 1
  const uint32_t cl = en_cluster();
#ifdef EN_SPLIT_N
  const uint32_t cols = p.n / EN_CLUSTERS;
  p.b += cl * cols;
  p.c += cl * cols * 2u;
  p.sb += cl * cols * (p.k / 32u);
  p.ncols = cols;
#else
  const uint32_t rows = p.m / EN_CLUSTERS;
  p.a += cl * rows * p.k;
  p.c += cl * rows * p.n * 2u;
  p.sa += cl * rows * (p.k / 32u);
  p.m = rows;
#endif
#endif
  if (p.res) p.res += p.c - c0;   // the residual has C's layout
  return p;
}

#ifndef EN_DEFER_OUT
#ifdef EN_LN_FP32
#define EN_DEFER_OUT false   // the fp32 LN y scratch needs the LN_Y room the second codes buffer uses
#else
#define EN_DEFER_OUT true
#endif
#endif
// ---- stage: QKV split with DMA ---------------------------------------------------------------
// Half-units u in [0, 96): tensor t = u / 32 (Q, K, V), head h = (u / 8) % 4, 32-row block
// hb = u % 8.  Rounds of up to 6 half-units, one per warp: thread 0 moves the inputs in, every
// warp quantizes its tile from SMEM into a staging area laid out for DRAM, thread 0 moves it out.
#define SD_SLOT  0x3200u    // per-warp SMEM: input 8 KB at +0, codes 4 KB at +0x2000, scales at +0x3000
#define SD_IN    0x0000u
#define SD_OUT   0x2000u
#define SD_SC    0x3000u
// The DMA issue functions: every thread calls them, all but thread 0 return at once (the FA
// kernel's pattern).  A divergent `if (tid == 0) { loops with break }` at the call site leaked
// IPDOM entries ("ipdom stack is full", metasim enc_m8).
__attribute__((noinline)) static void split_dma_in(uint32_t round, uint32_t nslots, uint32_t tid) {
  if (tid != 0) return;
  dma_cfg_in(16u);
  const uint32_t ns = 96u - round * nslots < nslots ? 96u - round * nslots : nslots;
  for (uint32_t s = 0; s < ns; s++) {
    const uint32_t u = round * nslots + s;
    const uint32_t t = u / 32u, h = (u / 8u) % 4u, row0 = (u % 8u) * 32u;
    const uint32_t src = EN_QKV + (row0 * 3u * EN_D + t * EN_D + h * EN_DH) * 2u;
    for (uint32_t r = 0; r < 32u; r++)   // one 256-byte row slice per command
      dma_in(s * SD_SLOT + SD_IN + r * 256u, src + r * 3u * EN_D * 2u, 16u);
  }
  dma_wait();
}
__attribute__((noinline)) static void split_dma_out(uint32_t round, uint32_t nslots, uint32_t tid) {
  if (tid != 0) return;
  const uint32_t ns = 96u - round * nslots < nslots ? 96u - round * nslots : nslots;
  for (uint32_t s = 0; s < ns; s++) {
    const uint32_t u = round * nslots + s;
    const uint32_t t = u / 32u, h = (u / 8u) % 4u, row0 = (u % 8u) * 32u;
    const uint32_t base = s * SD_SLOT;
    if (t == 0u) {        // Q rows: 32 x 128 B contiguous; QS[h][row0/64][g][row0%64 ..+32]
      dma_cfg_out(16u);
      dma_out_contig(EN_FA_Q + (h * EN_S + row0) * EN_DH, base + SD_OUT, 32u * EN_DH);
      for (uint32_t g = 0; g < 4u; g++)
        dma_out(EN_FA_QS + h * EN_S * 4u + (row0 / 64u) * 256u + g * 64u + row0 % 64u,
                base + SD_SC + g * 32u, 2u);
    } else if (t == 1u) { // K^T[h][d][row0 ..+32]: staging [c][d][16 keys], c = key / 16
      dma_cfg_out(EN_S);
      for (uint32_t c = 0; c < 2u; c++)
        for (uint32_t d0 = 0; d0 < EN_DH; d0 += 16u)
          dma_out(EN_FA_KT + (h * EN_DH + d0) * EN_S + row0 + 16u * c, base + SD_OUT + (c * 128u + d0) * 16u, 16u);
      dma_cfg_out(16u);
      for (uint32_t g = 0; g < 4u; g++)   // KTS[h][g][row0 ..+32]
        dma_out(EN_FA_KTS + (h * 4u + g) * EN_S + row0, base + SD_SC + g * 32u, 2u);
    } else {              // V rows: 32 x 128 B contiguous; VS[h][row0/32][128]
      dma_cfg_out(16u);
      dma_out_contig(EN_FA_V + (h * EN_S + row0) * EN_DH, base + SD_OUT, 32u * EN_DH);
      dma_out(EN_FA_VS + (h * (EN_S / 32u) + row0 / 32u) * EN_DH, base + SD_SC, 8u);
    }
  }
}
#ifdef EN_QUANT_FP32
// Block scales of the 4 rows 4q..4q+3 (group g) of a 16-row sub-tile; returns the packed scale
// word, inverse scales in inv[] (fully unrolled by the callers, so inv stays in registers).
#define SPLIT_SCALES(xb, s, q, g, inv, scw)                                                       \
  do {                                                                                          \
    _Pragma("unroll") for (uint32_t m = 0; m < 4u; m++) {                                       \
      const __shared uint32_t *xg = (xb) + (16u * (s) + 4u * (q) + m) * 64u + 16u * (g);         \
      uint32_t amax = 0;                                                                        \
      _Pragma("unroll") for (uint32_t i = 0; i < 16u; i++)                                      \
        amax = umax(amax, umax(and_asm(xg[i] << 16, 0x7FFFFFFFu), and_asm(xg[i], 0x7FFF0000u)));  \
      const int e = block_exp(amax);                                                            \
      (inv)[m] = block_inv(e);                                                                  \
      (scw) |= (uint32_t)(e + 127) << (8u * m);                                                 \
    }                                                                                           \
  } while (0)

// Q tile (32 rows x 128): codes row-major, scales [g][32 rows].
__attribute__((noinline)) static void split_work_q(const __shared uint32_t *xb, __shared uint32_t *ob,
                                                   __shared uint32_t *sb, uint32_t l) {
  const uint32_t g = l % 4u, q = l / 4u;
  for (uint32_t s = 0; s < 2u; s++) {
    float inv[4];
    uint32_t scw = 0;
    SPLIT_SCALES(xb, s, q, g, inv, scw);
    sb[g * 8u + s * 4u + q] = scw;
#pragma unroll
    for (uint32_t m = 0; m < 4u; m++) {
      const __shared uint32_t *xg = xb + (16u * s + 4u * q + m) * 64u + 16u * g;
      __shared uint32_t *og = ob + (16u * s + 4u * q + m) * 32u + 8u * g;
#pragma unroll
      for (uint32_t i = 0; i < 8u; i++) {
        const uint32_t w0 = xg[2u * i], w1 = xg[2u * i + 1u];
        og[i] = e4m3_pack4(bf16_to_f(w0 & 0xFFFFu) * inv[m], bf16_to_f(w0 >> 16) * inv[m],
                           bf16_to_f(w1 & 0xFFFFu) * inv[m], bf16_to_f(w1 >> 16) * inv[m]);
      }
    }
  }
}
// K tile: codes transposed, staging [c = key / 16][d][16 keys]; scales [g][32 keys].
__attribute__((noinline)) static void split_work_k(const __shared uint32_t *xb, __shared uint32_t *ob,
                                                   __shared uint32_t *sb, uint32_t l) {
  const uint32_t g = l % 4u, q = l / 4u;
  for (uint32_t s = 0; s < 2u; s++) {
    float inv[4];
    uint32_t scw = 0;
    SPLIT_SCALES(xb, s, q, g, inv, scw);
    sb[g * 8u + s * 4u + q] = scw;
#pragma unroll 4
    for (uint32_t i = 0; i < 16u; i++) {
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (uint32_t m = 0; m < 4u; m++) {
        const uint32_t w = xb[(16u * s + 4u * q + m) * 64u + 16u * g + i];
        lo |= e4m3_encode(bf16_to_f(w & 0xFFFFu) * inv[m]) << (8u * m);
        hi |= e4m3_encode(bf16_to_f(w >> 16) * inv[m]) << (8u * m);
      }
      const uint32_t d = 32u * g + 2u * i;
      ob[(s * 128u + d) * 4u + q] = lo;
      ob[(s * 128u + d + 1u) * 4u + q] = hi;
    }
  }
}
// V tile: groups of 32 keys = the whole tile; lane owns word-columns l + 16k.
__attribute__((noinline)) static void split_work_v(const __shared uint32_t *xb, __shared uint32_t *ob,
                                                   __shared uint32_t *sb, uint32_t l) {
  __shared uint16_t *vc = reinterpret_cast<__shared uint16_t *>(ob);   // [32 rows][64 halfwords]
  __shared uint16_t *vs = reinterpret_cast<__shared uint16_t *>(sb);
  for (uint32_t k = 0; k < 4u; k++) {
    const uint32_t wc = 16u * k + l;
    uint32_t a0 = 0, a1 = 0;
#pragma unroll 8
    for (uint32_t rr = 0; rr < 32u; rr++) {
      const uint32_t w = xb[rr * 64u + wc];
      a0 = umax(a0, and_asm(w << 16, 0x7FFFFFFFu));
      a1 = umax(a1, and_asm(w, 0x7FFF0000u));
    }
    const int e0 = block_exp(a0), e1 = block_exp(a1);
    const float i0 = block_inv(e0), i1 = block_inv(e1);
    vs[wc] = (uint16_t)((uint32_t)(e0 + 127) | ((uint32_t)(e1 + 127) << 8));
#pragma unroll 8
    for (uint32_t rr = 0; rr < 32u; rr++) {
      const uint32_t w = xb[rr * 64u + wc];
      vc[rr * 64u + wc] = (uint16_t)(e4m3_encode(bf16_to_f(w & 0xFFFFu) * i0) |
                                     (e4m3_encode(bf16_to_f(w >> 16) * i1) << 8));
    }
  }
}
#else
// Block scales of the 4 rows 4q..4q+3 (group g) of a 16-row sub-tile: packed scale word, and the
// e4m3_swar2 constants in p2[] (fully unrolled by the callers, so p2 stays in registers).
#define SPLIT_SCALES(xb, s, q, g, p2, scw)                                                        \
  do {                                                                                          \
    _Pragma("unroll") for (uint32_t m = 0; m < 4u; m++) {                                       \
      const __shared uint32_t *xg = (xb) + (16u * (s) + 4u * (q) + m) * 64u + 16u * (g);         \
      uint32_t mx = 0;                                                                          \
      _Pragma("unroll") for (uint32_t i = 0; i < 16u; i++) {                                    \
        const uint32_t a = xg[i] & 0x7FFF7FFFu;                                                 \
        mx = fmax_h(fmax_h(mx, a), a >> 16);                                                    \
      }                                                                                         \
      const int e = block_exp(mx << 16);                                                        \
      (p2)[m] = swar_p2(e);                                                                     \
      (scw) |= (uint32_t)(e + 127) << (8u * m);                                                 \
    }                                                                                           \
  } while (0)

// Q tile (32 rows x 128): codes row-major, scales [g][32 rows].
__attribute__((noinline)) static void split_work_q(const __shared uint32_t *xb, __shared uint32_t *ob,
                                                   __shared uint32_t *sb, uint32_t l) {
  const uint32_t g = l % 4u, q = l / 4u;
  for (uint32_t s = 0; s < 2u; s++) {
    uint32_t p2[4];
    uint32_t scw = 0;
    SPLIT_SCALES(xb, s, q, g, p2, scw);
    sb[g * 8u + s * 4u + q] = scw;
#pragma unroll
    for (uint32_t m = 0; m < 4u; m++) {
      const __shared uint32_t *xg = xb + (16u * s + 4u * q + m) * 64u + 16u * g;
      __shared uint32_t *og = ob + (16u * s + 4u * q + m) * 32u + 8u * g;
#pragma unroll
      for (uint32_t i = 0; i < 8u; i++)
        og[i] = swar_pack4(e4m3_swar2(xg[2u * i], p2[m]), e4m3_swar2(xg[2u * i + 1u], p2[m]));
    }
  }
}
// K tile: codes transposed, staging [c = key / 16][d][16 keys]; scales [g][32 keys].  Word i of
// rows m = 0..3 gives d = 32g + 2i (low halves) and d + 1 (high halves); a K^T word holds the
// 4 keys' codes of one d.
__attribute__((noinline)) static void split_work_k(const __shared uint32_t *xb, __shared uint32_t *ob,
                                                   __shared uint32_t *sb, uint32_t l) {
  const uint32_t g = l % 4u, q = l / 4u;
  for (uint32_t s = 0; s < 2u; s++) {
    uint32_t p2[4];
    uint32_t scw = 0;
    SPLIT_SCALES(xb, s, q, g, p2, scw);
    sb[g * 8u + s * 4u + q] = scw;
    const __shared uint32_t *xr = xb + (16u * s + 4u * q) * 64u + 16u * g;
#pragma unroll 4
    for (uint32_t i = 0; i < 16u; i++) {
      const uint32_t r0 = e4m3_swar2(xr[i], p2[0]), r1 = e4m3_swar2(xr[64u + i], p2[1]);
      const uint32_t r2 = e4m3_swar2(xr[128u + i], p2[2]), r3 = e4m3_swar2(xr[192u + i], p2[3]);
      const uint32_t t01 = r0 | (r1 << 8), t23 = r2 | (r3 << 8);   // [d m0, d m1, d+1 m0, d+1 m1]
      const uint32_t d = 32u * g + 2u * i;
      ob[(s * 128u + d) * 4u + q] = (t01 & 0xFFFFu) | (t23 << 16);
      ob[(s * 128u + d + 1u) * 4u + q] = (t01 >> 16) | (t23 & 0xFFFF0000u);
    }
  }
}
// V tile: groups of 32 keys = the whole tile; lane owns word-columns l + 16k (two columns each,
// with their own scales: e4m3_swar2 takes a different constant per half).
__attribute__((noinline)) static void split_work_v(const __shared uint32_t *xb, __shared uint32_t *ob,
                                                   __shared uint32_t *sb, uint32_t l) {
  __shared uint16_t *vc = reinterpret_cast<__shared uint16_t *>(ob);   // [32 rows][64 halfwords]
  __shared uint16_t *vs = reinterpret_cast<__shared uint16_t *>(sb);
  for (uint32_t k = 0; k < 4u; k++) {
    const uint32_t wc = 16u * k + l;
    uint32_t m0 = 0, m1 = 0;
#pragma unroll 8
    for (uint32_t rr = 0; rr < 32u; rr++) {
      const uint32_t a = xb[rr * 64u + wc] & 0x7FFF7FFFu;
      m0 = fmax_h(m0, a);
      m1 = fmax_h(m1, a >> 16);
    }
    const int e0 = block_exp(m0 << 16), e1 = block_exp(m1 << 16);
    const uint32_t p2 = swar_p(e0) | (swar_p(e1) << 16);
    vs[wc] = (uint16_t)((uint32_t)(e0 + 127) | ((uint32_t)(e1 + 127) << 8));
#pragma unroll 8
    for (uint32_t rr = 0; rr < 32u; rr++) {
      const uint32_t r = e4m3_swar2(xb[rr * 64u + wc], p2);
      vc[rr * 64u + wc] = (uint16_t)(r | (r >> 8));
    }
  }
}
#endif
// Rounds of 6 half-units (96 = 16 x 6, EN_OCC = 3 on 2 cores): every warp has an item each round.
__attribute__((noinline)) static void qkv_split_dma(Lane L, uint32_t tid) {
  const uint32_t l = L.lane, nw = total_warps(L);
  const uint32_t gw = L.warp_in_core * MU_NUM_CORES + L.core;
  if (is_lead(tid)) dma_setup(); else asm volatile("nop");
  for (uint32_t round = 0; round < 96u / nw; round++) {
    split_dma_in(round, nw, tid);
    mu_barrier(4, nw);
    const uint32_t u = round * nw + gw, t = u / 32u;
    const __shared uint32_t *xb = smem<uint32_t>(gw * SD_SLOT + SD_IN);
    __shared uint32_t *ob = smem<uint32_t>(gw * SD_SLOT + SD_OUT);
    __shared uint32_t *sb = smem<uint32_t>(gw * SD_SLOT + SD_SC);
    if (t == 0u) split_work_q(xb, ob, sb, l);
    else if (t == 1u) split_work_k(xb, ob, sb, l);
    else split_work_v(xb, ob, sb, l);
    mu_fence_smem();
    mu_barrier(5, nw);
    split_dma_out(round, nw, tid);
  }
  if (is_lead(tid)) dma_wait(); else asm volatile("nop");
}

// ---- DMA round driver ---------------------------------------------------------------------------
// Items are dealt one per warp per round; input staging is double-buffered (buf = round & 1).
//   round r: thread 0 waits for every DMA (in(r), out(r-1)); barrier; thread 0 issues in(r+1) into
//   the other buffer WITHOUT waiting (it overlaps this round's work); every warp runs work(r);
//   barrier; thread 0 issues out(r).
// Hazards: in(r+1) writes buffer (r+1)&1, last read by work(r-1) and out(r-1), both finished by the
// wait at the top of round r.  work(r) writes the single output staging, last read by out(r-1).
// always_inline here and on every lambda: an outlined callee would get a pointer to the caller's
// stack (the closure), which the strided stack breaks (memory note stack-pointer-escape-broken).
// DEFER: out(r) is issued in round r+1 right after the first barrier, ahead of in(r+2), so it
// overlaps work(r+1) instead of the next round's wait.  The reservation station orders a move-in
// after an earlier move-out of overlapping SMEM rows (ReservationStation.scala deps_st, WAR), so
// in-place outputs (res2's y, LN's x1) stay safe; staging the work writes with SIMT stores must be
// double-buffered by the stage (index buf), since SIMT stores are invisible to that ordering.
template <bool DEFER, typename In, typename Work, typename Out>
__attribute__((always_inline)) static inline void dma_rounds(Lane L, uint32_t tid, uint32_t nitems, In in, Work work, Out out) {
  // two clusters: cluster c takes items [c n/2, (c+1) n/2); the lambdas get global item indices
  const uint32_t ibase = en_cluster() * (nitems / EN_CLUSTERS);
  nitems /= EN_CLUSTERS;
  const uint32_t nwall = total_warps(L), gw0 = L.warp_in_core * MU_NUM_CORES + L.core;
#ifdef EN_DMA_LEAD
  // EN_DMA_LEAD: warp 0 (thread 0 issues every DMA command) takes no work item, so its blocking
  // command issue overlaps the other warps' work instead of adding to the round (fya: nearly all of
  // a stage's "in" time is thread 0 blocked in command issue).  The worker test is hidden from
  // LLVM (volatile asm) so it cannot unswitch the barrier loop on it.
  uint32_t gw_opaque;
  asm volatile("mv %0, %1" : "=r"(gw_opaque) : "r"(gw0));
  const bool worker = gw_opaque != 0u;
  const uint32_t nw = nwall - 1u, gw = gw0 - 1u;   // workers; gw is unused on warp 0
#else
  const bool worker = true;
  const uint32_t nw = nwall, gw = gw0;
#endif
  const uint32_t nrounds = (nitems + nw - 1u) / nw;
  uint32_t p_in = 0, p_bar = 0, p_work = 0, p_out = 0;   // thread 0's phase cycles (EN_PB_PROF)
  uint32_t p_iss = 0;   // the part of p_in spent issuing out(r-1) / in(r+1) (EN_PB_PROF + 4)
#ifdef EN_RND_PROF   // per warp: cycles in work + fence, to postbox slot EN_PB_UPROBE + gw0 (cluster 0)
  uint32_t p_wk = 0;
#endif
  if (is_lead(tid)) {
    for (uint32_t s = 0; s < nw && s < nitems; s++) in(s, ibase + s, 0u);
  } else {
    asm volatile("nop");
  }
  for (uint32_t r = 0; r < nrounds; r++) {
    const uint32_t c0 = mcycle();
    if (is_lead(tid)) dma_wait(); else asm volatile("nop");
    const uint32_t c1 = mcycle();
    mu_barrier(4, nwall);
    const uint32_t c2 = mcycle();
    if (is_lead(tid)) {
#ifndef EN_RND_NOIN   // EN_RND_NOIN: timing probe only (no DMA under the work; outputs are wrong)
      if (DEFER && r > 0)
        for (uint32_t s = 0; s < nw && (r - 1u) * nw + s < nitems; s++) out(s, ibase + (r - 1u) * nw + s, (r - 1u) & 1u);
      for (uint32_t s = 0; s < nw && (r + 1u) * nw + s < nitems; s++) in(s, ibase + (r + 1u) * nw + s, (r + 1u) & 1u);
#endif
    } else {
      asm volatile("nop");
    }
    const uint32_t c3 = mcycle();
    if (worker && r * nw + gw < nitems) work(gw, ibase + r * nw + gw, r & 1u); else asm volatile("nop");
    mu_fence_smem();
    const uint32_t c4 = mcycle();
#ifdef EN_RND_PROF
    p_wk += c4 - c3;   // this warp's work + fence, every warp
#endif
    mu_barrier(5, nwall);
    const uint32_t c5 = mcycle();
    if (!DEFER && is_lead(tid)) {
      for (uint32_t s = 0; s < nw && r * nw + s < nitems; s++) out(s, ibase + r * nw + s, r & 1u);
    } else {
      asm volatile("nop");
    }
    const uint32_t c6 = mcycle();
    p_iss += c3 - c2;
    p_in += (c1 - c0) + (c3 - c2); p_bar += (c2 - c1) + (c5 - c4); p_work += c4 - c3; p_out += c6 - c5;
  }
  const uint32_t c7 = mcycle();
  if (DEFER && nrounds > 0 && is_lead(tid)) {   // the last round's outputs
    const uint32_t r = nrounds - 1u;
    for (uint32_t s = 0; s < nw && r * nw + s < nitems; s++) out(s, ibase + r * nw + s, r & 1u);
  } else {
    asm volatile("nop");
  }
  if (is_lead(tid)) dma_wait(); else asm volatile("nop");
  p_out += mcycle() - c7;
#ifdef EN_RND_PROF
  if (L.lane == 0u) rad_pb_put(EN_PB_UPROBE + gw0, p_wk); else asm volatile("nop");
#endif
  if (is_lead(tid)) {
    rad_pb_put(EN_PB_PROF + 0u, p_in);
    rad_pb_put(EN_PB_PROF + 1u, p_bar);
    rad_pb_put(EN_PB_PROF + 2u, p_work);
    rad_pb_put(EN_PB_PROF + 3u, p_out);
    rad_pb_put(EN_PB_PROF + 4u, p_iss);
  } else {
    asm volatile("nop");
  }
}

// Row staging: a bf16 row of 512 elements = 4 chunks of 256 B, chunk c at c * 272 B (16 B gap),
// so lane l's 32-element group (words 16l..16l+15) starts at word 16l + 4(l/4): 16 subbanks apart.
#define RX_CHUNK 272u                 // bytes per staged 256-byte chunk
#define RX_ROW   (4u * RX_CHUNK)      // one 512-element row
static inline uint32_t rx_word(uint32_t w) { return w + 4u * (w / 64u); }   // row word -> staged word
// Scale staging for the whole launch: byte (group g, row r) at g * DSC_STRIDE + r (16-B aligned rows)
#ifdef EN_LN_OCC4   // 7 LN slots (4 warps per core, 2-row) end at 0x16500: gamma / beta and DSC move up
#define DSC        0x17A00u
#else
#define DSC        0x14800u
#endif
#define DSC_STRIDE 272u
// After the last round: thread 0 moves the tile-blocked A-operand scales out, one 64-byte line
// (4 SMEM rows) per (64-row block, group).
static void dma_scales_out(uint32_t groups, uint32_t cols, uint32_t out_s) {
  dma_cfg_out(16u);
  const uint32_t nb = EN_S / 64u / EN_CLUSTERS, b0 = en_cluster() * nb;   // this cluster's row blocks
  for (uint32_t blk = b0; blk < b0 + nb; blk++)
    for (uint32_t g = 0; g < groups; g++)
      dma_out(out_s + a_scale_offset(blk * 64u, g, cols), DSC + g * DSC_STRIDE + blk * 64u, 4u);
  dma_wait();
}

// ---- stage: LayerNorm + MX-FP8 quantize (RES: x1 = x + res first, written to x1_addr) -----------
// Item = 2 rows.  Slot: input buffers 0 and 1 (each X: 2 staged rows | R: 2 res rows), then
// C (2 x 512 B codes), Y (16 lanes x 33 floats), reduction scratch.
#define LN_IN     0x1080u   // one input buffer
#define LN_X      0x0000u
#define LN_R      0x0880u
#define LN_C      0x2100u   // codes, 2 x 512 B; with EN_DEFER_OUT two buffers (0x400 each)
#if EN_DEFER_OUT
#define LN_Y      0x2900u   // y scratch: 16 lanes x 17 words (bf16 pairs)
#else
#define LN_Y      0x2500u
#endif
#define LN_RED    0x2D80u
#ifdef EN_LN_2ROW
#define LN_Y2     0x2E80u   // EN_LN_2ROW: row 1's y scratch (16 lanes x 17 words)
#define LN_SLOT   0x3300u
#else
#define LN_SLOT   0x2E80u
#endif
#define LNW_HALF  528u      // EN_LN_WIDE: a 512-byte permuted half row + 16-byte gap
#define LNW_ROW   (2u * LNW_HALF)
static_assert(2u * LNW_ROW <= LN_R - LN_X, "EN_LN_WIDE x rows fit the LN_X region");
#ifdef EN_LN_OCC4
#define LN_GB     0x16800u   // gamma, beta: lane l's 32 floats at 144 l bytes (36-word stride)
#else
#define LN_GB     0x13400u   // gamma, beta: lane l's 32 floats at 144 l bytes (36-word stride)
#endif
static_assert(LN_GB + 2u * 16u * 144u <= DSC || DSC + 16u * DSC_STRIDE <= LN_GB, "LN gamma / beta clear of DSC");
// EN_LN_HMATH: gamma / beta as bf16 pairs, lane l's 16 words at 68 l bytes (17-word stride), beta
// table 16 x 68 bytes later; filled once per launch by warp 0 from the fp32 staging at LN_GB
#define LN_GBH    (DSC + 16u * DSC_STRIDE)
#ifdef EN_DMA_LEAD
static_assert((EN_LN_WARPS - 1u) * LN_SLOT <= LN_GB, "LN worker slots below LN_GB");   // warp 0 takes no slot
#else
static_assert(EN_LN_WARPS * LN_SLOT <= LN_GB, "LN slots below LN_GB");
#endif
template <bool RES>
__attribute__((noinline, flatten)) static void ln_quant(Lane L, uint32_t tid, uint32_t x_addr, uint32_t res_addr,
                                               uint32_t x1_addr, uint32_t g_addr, uint32_t b_addr,
                                               uint32_t out_addr, uint32_t out_s_addr) {
  static_assert(EN_D == 512u, "4 x 256-byte chunks per row");
  const uint32_t l = L.lane;
  if (is_lead(tid)) {
    dma_setup();
    dma_cfg_in(16u);
    for (uint32_t ln = 0; ln < 16u; ln++) {   // 16 lanes x 128 B, each lane's slice 144 B apart
      dma_in(LN_GB + ln * 144u, g_addr + ln * 128u, 8u);
      dma_in(LN_GB + 16u * 144u + ln * 144u, b_addr + ln * 128u, 8u);
    }
    dma_wait();
  } else {
    asm volatile("nop");
  }
  const float eps = u2f(EN_LN_EPS_BITS), inv_d = 1.0f / (float)EN_D;
  const __shared float *gam = smem<float>(LN_GB + l * 144u), *bet = smem<float>(LN_GB + 16u * 144u + l * 144u);
#ifdef EN_LN_HMATH
  if (warp_slot(L) == 0u) {   // warp 0 (thread 0 waited for the gamma / beta DMA above)
    __shared uint32_t *ghw = smem<uint32_t>(LN_GBH) + 17u * l, *bhw = smem<uint32_t>(LN_GBH + 16u * 68u) + 17u * l;
    for (uint32_t i = 0; i < 16u; i++) {
      ghw[i] = bf16_rne(gam[2u * i]) | (bf16_rne(gam[2u * i + 1u]) << 16);
      bhw[i] = bf16_rne(bet[2u * i]) | (bf16_rne(bet[2u * i + 1u]) << 16);
    }
    mu_fence_smem();
  } else {
    asm volatile("nop");
  }
#endif
  __shared uint8_t *sc = smem<uint8_t>(DSC);
#ifdef EN_LN_PROF   // worker slot 0, lane 0: cumulative cycles of res-add | stats | reduce | pass 1 | pass 2
  uint32_t lp[5] = {0, 0, 0, 0, 0};
#define LNP(j, c) lp[j] += mcycle() - (c)
#else
#define LNP(j, c) ((void)0)
#endif
  dma_rounds<EN_DEFER_OUT>(L, tid, EN_S / 2u,
    [&](uint32_t s, uint32_t it, uint32_t buf) __attribute__((always_inline)) {
      const uint32_t base = s * LN_SLOT + buf * LN_IN;
#ifdef EN_LN_WIDE
      // 2-block move-ins (dma_in2): each 512-byte half row lands chunk-permuted; x halves are
      // LNW_HALF bytes apart (16-byte gap, subbank spread), the residual's are contiguous
      dma_cfg_in(32u);
      for (uint32_t rr = 0; rr < 2u; rr++)
        for (uint32_t g = 0; g < 2u; g++) {
          const uint32_t off = (2u * it + rr) * EN_D * 2u + g * 512u;
          dma_in2(base + LN_X + rr * LNW_ROW + g * LNW_HALF, x_addr + off);
          if (RES) dma_in2(base + LN_R + rr * 1024u + g * 512u, res_addr + off);
        }
#else
      dma_cfg_in(16u);
      for (uint32_t rr = 0; rr < 2u; rr++)
        for (uint32_t c = 0; c < 4u; c++) {
          const uint32_t off = ((2u * it + rr) * EN_D + c * 128u) * 2u;
          dma_in(base + LN_X + rr * RX_ROW + c * RX_CHUNK, x_addr + off, 16u);
          if (RES) dma_in(base + LN_R + rr * 1024u + c * 256u, res_addr + off, 16u);
        }
#endif
    },
    [&](uint32_t s, uint32_t it, uint32_t buf) __attribute__((always_inline)) {
      __shared uint32_t *xb = smem<uint32_t>(s * LN_SLOT + buf * LN_IN + LN_X);
      uint32_t lc = mcycle();
      if (RES) {
        const __shared uint32_t *rb = smem<uint32_t>(s * LN_SLOT + buf * LN_IN + LN_R);
#pragma unroll 4
        for (uint32_t k = 0; k < 32u; k++) {   // 2 rows x 256 words, coalesced
          const uint32_t rr = k / 16u, w = 16u * (k % 16u) + l;
#ifdef EN_LN_WIDE   // same chunk permutation in x and the residual: add position by position
          const uint32_t xi = rr * (LNW_ROW / 4u) + (w / 128u) * (LNW_HALF / 4u) + w % 128u;
          xb[xi] = bf16x2_add(xb[xi], rb[rr * 256u + w]);
#else
          xb[rr * (RX_ROW / 4u) + rx_word(w)] = bf16x2_add(xb[rr * (RX_ROW / 4u) + rx_word(w)], rb[rr * 256u + w]);
#endif
        }
        mu_fence_smem();
      }
      LNP(0, lc); lc = mcycle();
      __shared float *red = smem<float>(s * LN_SLOT + LN_RED);   // 2 x 32 floats
#ifdef EN_LN_FP32
      __shared float *yb = smem<float>(s * LN_SLOT + LN_Y) + 33u * l;   // conflict-free stride
#else
      __shared float *yb = smem<float>(s * LN_SLOT + LN_Y) + 17u * l;   // 16 words used; conflict-free stride
#endif
#if defined(EN_LN_2ROW) && defined(EN_LN_WIDE) && !defined(EN_LN_FP32)
      // EN_LN_2ROW: the item's two rows interleaved phase by phase.  Every per-row expression and
      // its order are the same as the one-row loop below (bit-exact); the two rows' dependent
      // chains (sums, reductions, block max) now overlap, and gamma / beta are loaded once for both.
#ifndef LN_UNROLL_S   // EN_LN_2ROW unroll factors (code size vs ILP; the icache is 16 KB direct mapped)
#define LN_UNROLL_S 2    // fyp: S=2, P=1 beat S=16, P=4 by ~4% (code size)
#endif
#ifndef LN_UNROLL_P
#define LN_UNROLL_P 1
#endif
      {
        const __shared uint32_t *xh0 = xb + (l / 8u) * (LNW_HALF / 4u) + 8u * (l % 8u);
        const __shared uint32_t *xh1 = xh0 + LNW_ROW / 4u;
        const volatile __shared uint32_t *xa0 = xh0, *xa1 = xh0 + 64u, *xc0 = xh1, *xc1 = xh1 + 64u;
#define XV0(i) ((((i) & 4u) ? xa1 : xa0)[((i) / 8u) * 4u + (i) % 4u])
#define XV1(i) ((((i) & 4u) ? xc1 : xc0)[((i) / 8u) * 4u + (i) % 4u])
        float s1a = 0.0f, s2a = 0.0f, s1b = 0.0f, s2b = 0.0f;
#pragma unroll LN_UNROLL_S
        for (uint32_t i = 0; i < 16u; i++) {
          const uint32_t w0 = XV0(i), w1 = XV1(i);
          const float a = bf16_to_f(w0 & 0xFFFFu), b = bf16_to_f(w0 >> 16);
          const float c = bf16_to_f(w1 & 0xFFFFu), d = bf16_to_f(w1 >> 16);
          s1a += a + b;
          s2a += a * a + b * b;
          s1b += c + d;
          s2b += c * c + d * d;
        }
        __shared float *rd0 = red, *rd1 = red + 32u;
        rd0[l] = s1a; rd0[16u + l] = s2a;
        rd1[l] = s1b; rd1[16u + l] = s2b;
        mu_fence_smem();
        float t1a = 0.0f, t2a = 0.0f, t1b = 0.0f, t2b = 0.0f;
#pragma unroll LN_UNROLL_S
        for (uint32_t j = 0; j < 16u; j++) {
          t1a += rd0[j]; t2a += rd0[16u + j];
          t1b += rd1[j]; t2b += rd1[16u + j];
        }
        const float meana = t1a * inv_d, meanb = t1b * inv_d;
        float m2a = meana * meana, m2b = meanb * meanb;
        asm("" : "+r"(m2a));
        asm("" : "+r"(m2b));
        const float rstda = rsqrt_f(t2a * inv_d - m2a + eps), rstdb = rsqrt_f(t2b * inv_d - m2b + eps);
        const float nmra = 0.0f - meana * rstda, nmrb = 0.0f - meanb * rstdb;
        __shared uint32_t *yw0 = reinterpret_cast<__shared uint32_t *>(yb);
        __shared uint32_t *yw1 = smem<uint32_t>(s * LN_SLOT + LN_Y2) + 17u * l;
        uint32_t mxa = 0, mxb = 0;
#ifdef EN_LN_HMATH
        // EN_LN_HMATH: y = (x rstd - mean rstd) gamma + beta as two bf16 FMAs per element (rstd,
        // -mean rstd, gamma, beta rounded to bf16), replacing the fp32 FMAs and the integer bf16 RNE.
        // Not bit-exact with the fp32 path; gated by the dequantized-error checks.
        const uint32_t rha = bf16_rne(rstda), nha = bf16_rne(nmra), rhb = bf16_rne(rstdb), nhb = bf16_rne(nmrb);
        const __shared uint32_t *gh = smem<uint32_t>(LN_GBH) + 17u * l, *bh = smem<uint32_t>(LN_GBH + 16u * 68u) + 17u * l;
#endif
#pragma unroll LN_UNROLL_P
        for (uint32_t i = 0; i < 16u; i++) {
          const uint32_t w0 = XV0(i), w1 = XV1(i);
#ifdef EN_LN_HMATH
          const uint32_t gw = gh[i], bw = bh[i], gw1 = gw >> 16, bw1 = bw >> 16;
          const uint32_t va = and_asm(fma_h(fma_h(w0, rha, nha), gw, bw), 0xFFFFu) |
                              (fma_h(fma_h(w0 >> 16, rha, nha), gw1, bw1) << 16);
          const uint32_t vb = and_asm(fma_h(fma_h(w1, rhb, nhb), gw, bw), 0xFFFFu) |
                              (fma_h(fma_h(w1 >> 16, rhb, nhb), gw1, bw1) << 16);
#else
          const float g0 = gam[2u * i], g1 = gam[2u * i + 1u], b0 = bet[2u * i], b1 = bet[2u * i + 1u];
          const uint32_t u0 = f2u(fma_f(fma_f(bf16_to_f(w0 & 0xFFFFu), rstda, nmra), g0, b0));
          const uint32_t u1 = f2u(fma_f(fma_f(bf16_to_f(w0 >> 16), rstda, nmra), g1, b1));
          const uint32_t u2 = f2u(fma_f(fma_f(bf16_to_f(w1 & 0xFFFFu), rstdb, nmrb), g0, b0));
          const uint32_t u3 = f2u(fma_f(fma_f(bf16_to_f(w1 >> 16), rstdb, nmrb), g1, b1));
          const uint32_t va = ((u0 + 0x7FFFu + ((u0 >> 16) & 1u)) >> 16) |
                              ((u1 + 0x7FFFu + ((u1 >> 16) & 1u)) & 0xFFFF0000u);
          const uint32_t vb = ((u2 + 0x7FFFu + ((u2 >> 16) & 1u)) >> 16) |
                              ((u3 + 0x7FFFu + ((u3 >> 16) & 1u)) & 0xFFFF0000u);
#endif
          yw0[i] = va;
          yw1[i] = vb;
          const uint32_t aa = va & 0x7FFF7FFFu, ab = vb & 0x7FFF7FFFu;
          mxa = fmax_h(fmax_h(mxa, aa), aa >> 16);
          mxb = fmax_h(fmax_h(mxb, ab), ab >> 16);
        }
#undef XV0
#undef XV1
        const int ea = block_exp(mxa << 16), eb = block_exp(mxb << 16);
        const uint32_t p2a = swar_p2(ea), p2b = swar_p2(eb);
        __shared uint32_t *og = smem<uint32_t>(s * LN_SLOT + LN_C + (EN_DEFER_OUT ? buf * 0x400u : 0u)) + 8u * l;
#pragma unroll LN_UNROLL_P
        for (uint32_t i = 0; i < 8u; i++) {
          og[i] = swar_pack4(e4m3_swar2(yw0[2u * i], p2a), e4m3_swar2(yw0[2u * i + 1u], p2a));
          og[128u + i] = swar_pack4(e4m3_swar2(yw1[2u * i], p2b), e4m3_swar2(yw1[2u * i + 1u], p2b));
        }
        sc[l * DSC_STRIDE + 2u * it] = (uint8_t)(ea + 127);
        sc[l * DSC_STRIDE + 2u * it + 1u] = (uint8_t)(eb + 127);
      }
#else
      for (uint32_t rr = 0; rr < 2u; rr++) {
#ifdef EN_LN_WIDE
        // lane l's 64-byte group = chunks 4j..4j+3 (j = l % 8) of half row l / 8.  In the permuted
        // half, chunks 4j, 4j+2 sit at words 8j..8j+7 and chunks 4j+1, 4j+3 at 64 + 8j..: element
        // word i is at xp[i & 4 ? 1 : 0][(i / 8) * 4 + i % 4]
        const __shared uint32_t *xh = xb + rr * (LNW_ROW / 4u) + (l / 8u) * (LNW_HALF / 4u) + 8u * (l % 8u);
        const volatile __shared uint32_t *xp0 = xh, *xp1 = xh + 64u;
#define XV(i) ((((i) & 4u) ? xp1 : xp0)[((i) / 8u) * 4u + (i) % 4u])
#else
        const __shared uint32_t *xg = xb + rr * (RX_ROW / 4u) + 16u * l + 4u * (l / 4u);
        const volatile __shared uint32_t *xv = xg;   // volatile: see quant_group_bf16
#define XV(i) (xv[i])
#endif
        uint32_t xw[16];
#pragma unroll
        for (uint32_t i = 0; i < 16u; i++) xw[i] = XV(i);
        float s1 = 0.0f, s2 = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < 16u; i++) {
          const float a = bf16_to_f(xw[i] & 0xFFFFu), b = bf16_to_f(xw[i] >> 16);
          s1 += a + b;
          s2 += a * a + b * b;
        }
        LNP(1, lc); lc = mcycle();
        __shared float *rd = red + (rr & 1u) * 32u;
        rd[l] = s1;
        rd[16u + l] = s2;
        mu_fence_smem();
        float t1 = 0.0f, t2 = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < 16u; j++) { t1 += rd[j]; t2 += rd[16u + j]; }
        const float mean = t1 * inv_d;
        float m2 = mean * mean;
        asm("" : "+r"(m2));
        const float rstd = rsqrt_f(t2 * inv_d - m2 + eps);
        const float nmr = 0.0f - mean * rstd;   // y = fma(fma(x, rstd, -mean rstd), gamma, beta)
        asm volatile("" :: "r"(nmr), "r"(rstd));
        LNP(2, lc); lc = mcycle();
#ifndef EN_LN_FP32
        // pass 1: y = (x - mean) rstd gamma + beta in fp32, rounded to bf16 (RNE), two per word in
        // this lane's scratch; block max on the bf16 FPU.  pass 2: the SWAR encoder (as
        // quant_group_bf16).  The extra bf16 rounding before e4m3 changes some codes vs the fp32
        // rule; the host gates LN outputs by dequantized error (check_a_frob).
        __shared uint32_t *yw = reinterpret_cast<__shared uint32_t *>(yb);
        uint32_t mx = 0;
#pragma unroll 4
        for (uint32_t i = 0; i < 16u; i++) {
          const float g0 = gam[2u * i], g1 = gam[2u * i + 1u], b0 = bet[2u * i], b1 = bet[2u * i + 1u];
          const uint32_t w = XV(i);   // reloaded: keeping xw live here exhausts the 64 registers
          const uint32_t u0 = f2u(fma_f(fma_f(bf16_to_f(w & 0xFFFFu), rstd, nmr), g0, b0));
          const uint32_t u1 = f2u(fma_f(fma_f(bf16_to_f(w >> 16), rstd, nmr), g1, b1));
          const uint32_t v = ((u0 + 0x7FFFu + ((u0 >> 16) & 1u)) >> 16) |
                             ((u1 + 0x7FFFu + ((u1 >> 16) & 1u)) & 0xFFFF0000u);
          yw[i] = v;
          const uint32_t a = v & 0x7FFF7FFFu;
          mx = fmax_h(fmax_h(mx, a), a >> 16);
        }
        asm volatile("" :: "r"(mx));
        LNP(3, lc); lc = mcycle();
        const int e = block_exp(mx << 16);
        const uint32_t p2 = swar_p2(e);
        __shared uint32_t *og = smem<uint32_t>(s * LN_SLOT + LN_C + (EN_DEFER_OUT ? buf * 0x400u : 0u)) + rr * 128u + 8u * l;
#pragma unroll 4
        for (uint32_t i = 0; i < 8u; i++)
          og[i] = swar_pack4(e4m3_swar2(yw[2u * i], p2), e4m3_swar2(yw[2u * i + 1u], p2));
        LNP(4, lc); lc = mcycle();
#else
        // pass 1: y = (x - mean) rstd gamma + beta, kept in SMEM; block max
        uint32_t amax = 0;
#pragma unroll 4
        for (uint32_t i = 0; i < 16u; i++) {
          const float g0 = gam[2u * i], g1 = gam[2u * i + 1u], b0 = bet[2u * i], b1 = bet[2u * i + 1u];
          const uint32_t w = XV(i);   // reloaded: keeping xw live here exhausts the 64 registers
          const float y0 = fma_f(fma_f(bf16_to_f(w & 0xFFFFu), rstd, nmr), g0, b0);
          const float y1 = fma_f(fma_f(bf16_to_f(w >> 16), rstd, nmr), g1, b1);
          yb[2u * i] = y0;
          yb[2u * i + 1u] = y1;
          amax = umax(amax, umax(abs_bits(y0), abs_bits(y1)));
        }
        const int e = block_exp(amax);
        const float inv = block_inv(e);
        // pass 2: encode from the y scratch (this lane's own stores: program order)
        __shared uint32_t *og = smem<uint32_t>(s * LN_SLOT + LN_C + (EN_DEFER_OUT ? buf * 0x400u : 0u)) + rr * 128u + 8u * l;
#pragma unroll 2
        for (uint32_t i = 0; i < 8u; i++)
          og[i] = e4m3_pack4(yb[4u * i] * inv, yb[4u * i + 1u] * inv, yb[4u * i + 2u] * inv, yb[4u * i + 3u] * inv);
#endif
        sc[l * DSC_STRIDE + 2u * it + rr] = (uint8_t)(e + 127);
      }
#endif
      (void)lc;
#ifdef EN_LN_PROF
      if (s == 0u && l == 0u)
        for (uint32_t j = 0; j < 5u; j++) rad_pb_put(EN_PB_UPROBE + j, lp[j]);
#endif
    },
    [&](uint32_t s, uint32_t it, uint32_t buf) __attribute__((always_inline)) {
      dma_cfg_out(16u);
      dma_out_contig(out_addr + 2u * it * EN_D, s * LN_SLOT + LN_C + (EN_DEFER_OUT ? buf * 0x400u : 0u), 2u * EN_D);
#ifdef EN_LN_WIDE   // x1 sits in dma_in2's permuted layout: 2-block move-outs at stride 32 undo it
      if (RES) {
        dma_cfg_out(32u);
        for (uint32_t rr = 0; rr < 2u; rr++)
          for (uint32_t g = 0; g < 2u; g++)
            dma_out2(x1_addr + (2u * it + rr) * EN_D * 2u + g * 512u,
                     s * LN_SLOT + buf * LN_IN + LN_X + rr * LNW_ROW + g * LNW_HALF);
        dma_cfg_out(16u);
      }
#else
      if (RES)
        for (uint32_t rr = 0; rr < 2u; rr++)
          for (uint32_t c = 0; c < 4u; c++)
            dma_out(x1_addr + ((2u * it + rr) * EN_D + c * 128u) * 2u,
                    s * LN_SLOT + buf * LN_IN + LN_X + rr * RX_ROW + c * RX_CHUNK, 16u);
#endif
    });
  if (is_lead(tid)) dma_scales_out(EN_D / 32u, EN_D, out_s_addr); else asm volatile("nop");
}

// ---- stage: attention output (FA layout, heads apart) -> out-projection A operand --------------
#define OQ_IN   0x1100u   // one input buffer: 4 staged rows
#define OQ_C    0x2200u   // codes 4 x 512 B, two buffers (EN_DEFER_OUT)
#define OQ_SLOT 0x3200u
__attribute__((noinline, flatten)) static void o_quant(Lane L, uint32_t tid) {
  const uint32_t l = L.lane;
  __shared uint8_t *sc = smem<uint8_t>(DSC);
  if (is_lead(tid)) dma_setup(); else asm volatile("nop");
  dma_rounds<EN_DEFER_OUT>(L, tid, EN_S / 4u,
    [&](uint32_t s, uint32_t it, uint32_t buf) __attribute__((always_inline)) {
      dma_cfg_in(16u);
      for (uint32_t rr = 0; rr < 4u; rr++)
        for (uint32_t h = 0; h < EN_H; h++)   // head h's 128 values = chunk h of the row
          dma_in(s * OQ_SLOT + buf * OQ_IN + rr * RX_ROW + h * RX_CHUNK,
                 EN_FA_O + (h * EN_S + 4u * it + rr) * EN_DH * 2u, 16u);
    },
    [&](uint32_t s, uint32_t it, uint32_t buf) __attribute__((always_inline)) {
      const __shared uint32_t *xb = smem<uint32_t>(s * OQ_SLOT + buf * OQ_IN);
      for (uint32_t rr = 0; rr < 4u; rr++)
        sc[l * DSC_STRIDE + 4u * it + rr] = (uint8_t)quant_group_bf16(
            xb + rr * (RX_ROW / 4u) + 16u * l + 4u * (l / 4u), smem<uint32_t>(s * OQ_SLOT + OQ_C + buf * 0x800u) + rr * 128u + 8u * l);
    },
    [&](uint32_t s, uint32_t it, uint32_t buf) __attribute__((always_inline)) {
      dma_cfg_out(16u);
      dma_out_contig(EN_OA + 4u * it * EN_D, s * OQ_SLOT + OQ_C + buf * 0x800u, 4u * EN_D);
    });
  if (is_lead(tid)) dma_scales_out(EN_D / 32u, EN_D, EN_OA_S); else asm volatile("nop");
}

// ---- stage: GELU (bf16, written back in place) + quantize -> FFN-down A operand ------------------
// Item = one row of F = 2048: 16 staged chunks; lane l owns groups l + 16k.
#define GQ_IN   0x1100u   // one input buffer: 16 staged chunks (GELU is written back in place)
#define GQ_C    0x2200u   // codes 2 KB, two buffers (EN_DEFER_OUT)
#define GQ_SLOT 0x3200u
__attribute__((noinline, flatten)) static void gelu_quant(Lane L, uint32_t tid) {
  const uint32_t l = L.lane;
  __shared uint8_t *sc = smem<uint8_t>(DSC);
  if (is_lead(tid)) dma_setup(); else asm volatile("nop");
  dma_rounds<EN_DEFER_OUT>(L, tid, EN_S,
    [&](uint32_t s, uint32_t r, uint32_t buf) __attribute__((always_inline)) {
      dma_cfg_in(16u);
      for (uint32_t c = 0; c < EN_F / 128u; c++)
        dma_in(s * GQ_SLOT + buf * GQ_IN + c * RX_CHUNK, EN_FF + (r * EN_F + c * 128u) * 2u, 16u);
    },
    [&](uint32_t s, uint32_t r, uint32_t buf) __attribute__((always_inline)) {
      __shared uint32_t *xb = smem<uint32_t>(s * GQ_SLOT + buf * GQ_IN);
      for (uint32_t k = 0; k < 4u; k++) {
        const uint32_t G = l + 16u * k;
        __shared uint32_t *xg = xb + 16u * G + 4u * (G / 4u);
#if defined(EN_GELU_ILP) && !defined(EN_GELU_FP32)
#pragma unroll 1
        for (uint32_t i = 0; i < 16u; i += 4u) {
          uint32_t w0 = xg[i], w1 = xg[i + 1u], w2 = xg[i + 2u], w3 = xg[i + 3u];
          gelu_bf16x8(w0, w1, w2, w3);
          xg[i] = w0; xg[i + 1u] = w1; xg[i + 2u] = w2; xg[i + 3u] = w3;
        }
#else
#pragma unroll 4
        for (uint32_t i = 0; i < 16u; i++) {
          const uint32_t w = xg[i];
#ifdef EN_GELU_FP32
          xg[i] = bf16_rne(gelu_f(bf16_to_f(w & 0xFFFFu))) | (bf16_rne(gelu_f(bf16_to_f(w >> 16))) << 16);
#else
          xg[i] = gelu_bf16x2(w);
#endif
        }
#endif
        sc[G * DSC_STRIDE + r] = (uint8_t)quant_group_bf16(xg, smem<uint32_t>(s * GQ_SLOT + GQ_C + buf * 0x800u) + 8u * G);
      }
    },
    [&](uint32_t s, uint32_t r, uint32_t buf) __attribute__((always_inline)) {
      dma_cfg_out(16u);
      dma_out_contig(EN_G + r * EN_F, s * GQ_SLOT + GQ_C + buf * 0x800u, EN_F);
#ifndef EN_NO_GELU_WB   // the bf16 GELU output is only read by the host checks
      for (uint32_t c = 0; c < EN_F / 128u; c++)
        dma_out(EN_FF + (r * EN_F + c * 128u) * 2u, s * GQ_SLOT + buf * GQ_IN + c * RX_CHUNK, 16u);
#else
      (void)buf;
#endif
    });
  if (is_lead(tid)) dma_scales_out(EN_F / 32u, EN_F, EN_G_S); else asm volatile("nop");
}

// ---- stage: y = x1 + ff2 -------------------------------------------------------------------------
#define R2_IN   0x2000u   // one input buffer: A (4 rows of x1, then y) | B (4 rows of ff2)
#define R2_SLOT 0x4000u
__attribute__((noinline, flatten)) static void residual2(Lane L, uint32_t tid) {
  const uint32_t l = L.lane;
  if (is_lead(tid)) dma_setup(); else asm volatile("nop");
  dma_rounds<EN_DEFER_OUT>(L, tid, EN_S / 4u,
    [&](uint32_t s, uint32_t it, uint32_t buf) __attribute__((always_inline)) {
#ifdef EN_R2_WIDE
      // 2-block move-ins; x1 and ff2 get the same chunk permutation, the add is elementwise, and
      // out() restores the order
      dma_cfg_in(32u);
      for (uint32_t g = 0; g < 8u; g++) {   // 4 rows x 1 KB = 8 groups of 512 B
        dma_in2(s * R2_SLOT + buf * R2_IN + g * 512u, EN_X1 + 4u * it * EN_D * 2u + g * 512u);
        dma_in2(s * R2_SLOT + buf * R2_IN + 0x1000u + g * 512u, EN_FF2 + 4u * it * EN_D * 2u + g * 512u);
      }
#else
      dma_cfg_in(16u);
      for (uint32_t c = 0; c < 16u; c++) {   // 4 rows x 1 KB, contiguous
        dma_in(s * R2_SLOT + buf * R2_IN + c * 256u, EN_X1 + 4u * it * EN_D * 2u + c * 256u, 16u);
        dma_in(s * R2_SLOT + buf * R2_IN + 0x1000u + c * 256u, EN_FF2 + 4u * it * EN_D * 2u + c * 256u, 16u);
      }
#endif
    },
    [&](uint32_t s, uint32_t, uint32_t buf) __attribute__((always_inline)) {
      __shared uint32_t *a = smem<uint32_t>(s * R2_SLOT + buf * R2_IN);
      const __shared uint32_t *b = smem<uint32_t>(s * R2_SLOT + buf * R2_IN + 0x1000u);
#pragma unroll 4
      for (uint32_t k = 0; k < 64u; k++) a[16u * k + l] = bf16x2_add(a[16u * k + l], b[16u * k + l]);
    },
    [&](uint32_t s, uint32_t it, uint32_t buf) __attribute__((always_inline)) {
#ifdef EN_R2_WIDE
      dma_cfg_out(32u);
      for (uint32_t g = 0; g < 8u; g++)
        dma_out2(EN_Y + 4u * it * EN_D * 2u + g * 512u, s * R2_SLOT + buf * R2_IN + g * 512u);
      dma_cfg_out(16u);   // the other stages' default
#else
      dma_cfg_out(16u);
      dma_out_contig(EN_Y + 4u * it * EN_D * 2u, s * R2_SLOT + buf * R2_IN, 4u * EN_D * 2u);
#endif
    });
}

// ---- GEMM with a SIMT epilogue per output tile (fused stages) -------------------------------------
// Thread 0 runs the GEMM with TK = 128; the last k-step of every 128x128 output tile also stores C
// (bf16, row-major, 256 B per row) into SMEM (mxg::C_SPAD_ROW).  The other warps ("workers") then
// run the stage's epilogue on that tile and thread 0 moves the epilogue's output to DRAM.  C never
// reaches DRAM.  SMEM with TK = 128: A [0,16K) [32K,48K), B [80K,96K) [112K,128K), C [48K,80K);
// free for the epilogue: FX_LO = [16K,32K) and FX_HI = [96K,111K).
// SERIALIZED: the epilogue never runs under a matmul.  Gemmini tracks its in-flight scratchpad reads
// in a 4-entry queue without backpressure (Scratchpad.scala:220); dense SIMT SMEM traffic slows the
// shared SMEM's read responses enough to overflow it (metasim enc_m18a/c assert, enc_m18b with idle
// workers completes; the overlapped version hung on the board).  Per tile T:
//   warp 0: k-steps of T, the last one storing C, drain | barrier 4 | barrier 5 | out(T), drain
//   workers: barrier 4 | work(T) | barrier 5
// During work(T) the only Gemmini activity is the move-in of tile T+1's first k-step (SMEM writes,
// which are backpressured, as in the dma_rounds stages).  Workers idle at a barrier never poll SMEM:
// SIMT SMEM traffic also starves the accumulator -> SMEM writer of the C store.
// A stage is a struct with: tiles; problem() (tk = 128); pre(T); work(T, lane, w, nwk); out(T).
// EN_FX_WORK_NOINLINE: one out-of-line copy of St::work instead of two inlined ones (warp 0 and the
// workers each call it in fx_stage); the fused stage's code then fits the 16 KB direct-mapped
// icache next to warp 0's GEMM-issue code.  Scalar arguments only (no stack-pointer escape).
#ifdef EN_FX_WORK_NOINLINE
#define FX_WORK_ATTR __attribute__((noinline))
#else
#define FX_WORK_ATTR
#endif
// EN_UG_WORK_NOINLINE: UpGelu::work only (with -muon-hazard-rename, hzn2: UP+GELU -22.7k, but
// QKV split +11.2k when its work is out of line too)
#if defined(EN_FX_WORK_NOINLINE) || defined(EN_UG_WORK_NOINLINE)
#define UG_WORK_ATTR __attribute__((noinline))
#else
#define UG_WORK_ATTR
#endif
#define FX_C      (mxg::C_SPAD_ROW * 16u)   // 0xC000 (0x8000 with MXG_BANKSPLIT)
#ifdef MXG_BANKSPLIT
// bank 0 = A buffers, bank 1 = C, bank 2 = epilogue outputs, bank 3 = B buffers
#define FX_LO     0x10000u
#define FX_HI     0x14000u
static_assert(FX_C == 0x8000u, "C slot in bank 1");
#else
#define FX_LO     0x4000u
#define FX_HI     0x18000u
static_assert(FX_C == 0xC000u, "C slot between the A and B buffers of TK = 128");
#endif

// Thread 0's GEMM pieces.  Every warp-0 lane calls them; all but thread 0 return at once (the
// split stage's DMA pattern), so the barriers in fx_stage stay in converged code.
template <typename St> static inline mxg::Tiling fx_tiling() { return mxg::tiling_of(St::problem()); }
// St::out without the drain, out of line: inlined into the k-step loop, QkvSplit::out's branches
// trip an LLVM UnifyLoopExits assertion (2026-10-23).  Thread 0 only (called from fx_tile).
template <typename St> __attribute__((noinline)) static void fx_out_issue(uint32_t T) { St::out(T); }
template <typename St>
__attribute__((always_inline)) static inline void fx_step(const mxg::Problem &p, const mxg::Tiling &t, uint32_t s, bool store_c,
                                                         uint32_t out_T = ~0u) {
  // p is this cluster's sub-problem (en_split); s counts its own k-steps
  const uint32_t k_tiles = p.k / p.tk, n_tiles = mxg::ncols_of(p) / p.tn;
  const uint32_t steps = (St::tiles / EN_CLUSTERS) * k_tiles;
  const uint32_t kt = s % k_tiles;
  const bool more = s + 1 < steps;
  mxg::select_scales(p, t, s);
#ifdef MXG_MVIN_WIDE   // 64-128 move-in commands can block thread 0 on a full load queue: matmul first
  mxg::matmul_tile(t, s, kt == 0, store_c);
#endif
  // EN_FX_NOMVIN / EN_FX_NOSCALES: timing diagnostics (wrong results) that skip the next step's
  // operand move-ins / scale fill
#ifndef EN_FX_NOMVIN
  if (more) {
    const uint32_t n = s + 1, tile = n / k_tiles;
    mxg::mvin_tiles(t, p.a + (tile / n_tiles) * p.tm * p.k + (n % k_tiles) * p.tk,
                    p.b + (tile % n_tiles) * p.tn + (n % k_tiles) * p.tk * p.n, n);
  }
#endif
#ifndef MXG_MVIN_WIDE
  mxg::matmul_tile(t, s, kt == 0, store_c);
#endif
  // EN_FX_OUT_OVL: the previous tile's output move-outs go in AFTER this step's LOOP_WS commands (a
  // LOOP_WS write behind a full command queue inflates runningLoops: memory note
  // gemmini-runningloops-overcount) and drain with this step, under its matmul.  The workers are
  // parked at barrier 4, and the next epilogue writes FX_LO only after the tile's last drain.
  if (out_T != ~0u) fx_out_issue<St>(out_T);
#ifndef EN_FX_NOSCALES
  if (more) {
    const uint32_t n = s + 1, tile = n / k_tiles;
    mxg::load_tile_scales(p, tile / n_tiles, tile % n_tiles, n % k_tiles, n);
    mu_fence_smem();
  }
#endif
  mxg::drain();
}
template <typename St>
__attribute__((noinline)) static void fx_prologue(uint32_t tid) {
  if (tid != 0) return;
  mxg::drain();
  const mxg::Problem p = en_split(St::problem());
  const mxg::Tiling t = mxg::tiling_of(p);
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              mxg::FORMAT_FP8, mxg::FORMAT_FP8, mxg::FORMAT_FULL, false);
  gemmini_extended3_config_ld(t.lda, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(t.ldb, MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(16u);   // as mxg MXG_C_SPAD; out(T) must leave the store stride at 16
  gemmini_mxquant_config_mvout(mxg::host_addr(p.scratch), t.i, t.j, t.k, 0, 0, 1);
  gemmini_loop_ws_config_bounds(t.i, t.j, t.k, 0, 0, 0);
  gemmini_fence();
  gemmini_loop_ws_config_bounds(t.i, t.j, t.k, 0, 0, 0);
  gemmini_fence();
  mxg::mvin_tiles(t, p.a, p.b, 0);   // this cluster's sub-problem, local step 0
  mxg::load_tile_scales(p, 0, 0, 0, 0);
  mu_fence_smem();
  mxg::drain();
}
template <typename St>
__attribute__((noinline)) static void fx_tile(uint32_t tid, uint32_t Tl, uint32_t T, uint32_t prev_T = ~0u) {   // local tile Tl = global T
  if (tid != 0) return;
  const mxg::Problem p = en_split(St::problem());
  const mxg::Tiling t = mxg::tiling_of(p);
  const uint32_t k_tiles = p.k / p.tk;
  St::pre(T);   // epilogue inputs (DMA writes into FX_LO / FX_HI; the workers are at a barrier)
  for (uint32_t kt = 0; kt < k_tiles; kt++) fx_step<St>(p, t, Tl * k_tiles + kt, kt + 1 == k_tiles, kt == 0u ? prev_T : ~0u);
}
// Global tile index (the full problem's row-block-major order, which work/out/pre use) of this
// cluster's local tile Tl.
template <typename St>
static inline uint32_t fx_global(uint32_t Tl) {
#if EN_CLUSTERS > 1 && defined(EN_SPLIT_N)
  const mxg::Problem f = St::problem();
  const uint32_t nfull = f.n / f.tn, nh = nfull / EN_CLUSTERS;
  return (Tl / nh) * nfull + en_cluster() * nh + Tl % nh;
#else
  return en_cluster() * (St::tiles / EN_CLUSTERS) + Tl;
#endif
}
template <typename St>
__attribute__((noinline)) static void fx_out(uint32_t tid, uint32_t T) {
  if (tid != 0) return;
  St::out(T);
  mxg::drain();
}
// EN_FX_OUT_HALF: stages with work_range / out_half move the first half of a tile's output out
// while the workers compute the second half (DMA under SIMT work, no matmul running: the
// dma_rounds pattern).  Worker index nwk - 1 (= warp 0, which issues the move-outs) gets the
// fewest second-half iterations.
template <typename St> struct fx_halves { static constexpr bool v = false; };
template <typename St> __attribute__((noinline)) static void fx_out_half(uint32_t tid, uint32_t T, uint32_t h) {
  if (tid != 0) return;
  St::out_half(T, h);
  if (h == 1u) mxg::drain(); else asm volatile("nop");
}
// Epilogue worker index of a worker warp (slot 1 .. nw - 1; warp 0 takes nwk).  Slices go
// round-robin over the indices, so with 32 slices on 6 warps indices 0 and 1 take 6 and the rest 5.
// Slots 0-2 are core 0 and 3-5 core 1: EN_FX_WBAL swaps the indices of slots 2 and 3 so each core
// runs 16 slices (default: core 0 runs 17, core 1 runs 15).
static inline uint32_t fx_wk(uint32_t slot) {
#ifdef EN_FX_WBAL
  return slot - 1u + (slot == 2u ? 1u : 0u) - (slot == 3u ? 1u : 0u);
#else
  return slot - 1u;
#endif
}
#if defined(EN_FX_HPM) || defined(EN_STAGE_HPM)
// EN_FX_HPM / EN_STAGE_HPM: lane 0 of each core's first warp (tid 0 = core 0, tid 16 = core 1) stores the core's
// perf counters (mcycle, minstret, mhpmcounter3-24: cycles decoded / dispatched / eligible / issued,
// then per warp 0-2 stalls WAW, WAR, scoreboard, RS full, busy, busy LSU; radiance CSR.scala
// PerfCSRs) to EN_HPM + ((cluster * 2 + core) * 2 + idx) * 128.  Diagnostic (EN_TEST 18).
#define HPM_RD(n) ({ uint32_t v_; asm volatile("csrr %0, %1" : "=r"(v_) : "i"(n)); v_; })
__attribute__((noinline)) static void fx_hpm_snap(uint32_t tid, uint32_t idx) {
  if ((tid & ~16u) != 0u) return;
  volatile uint32_t *o = reinterpret_cast<volatile uint32_t *>(EN_HPM + ((en_cluster() * 2u + tid / 16u) * 2u + idx) * 128u);
  const uint32_t c = HPM_RD(0xB00), r = HPM_RD(0xB02);
  const uint32_t h3 = HPM_RD(0xB03), h4 = HPM_RD(0xB04), h5 = HPM_RD(0xB05), h6 = HPM_RD(0xB06);
  const uint32_t h7 = HPM_RD(0xB07), h8 = HPM_RD(0xB08), h9 = HPM_RD(0xB09), h10 = HPM_RD(0xB0A);
  const uint32_t h11 = HPM_RD(0xB0B), h12 = HPM_RD(0xB0C), h13 = HPM_RD(0xB0D), h14 = HPM_RD(0xB0E);
  const uint32_t h15 = HPM_RD(0xB0F), h16 = HPM_RD(0xB10), h17 = HPM_RD(0xB11), h18 = HPM_RD(0xB12);
  const uint32_t h19 = HPM_RD(0xB13), h20 = HPM_RD(0xB14), h21 = HPM_RD(0xB15), h22 = HPM_RD(0xB16);
  const uint32_t h23 = HPM_RD(0xB17), h24 = HPM_RD(0xB18);
  o[0] = c; o[1] = r; o[2] = h3; o[3] = h4; o[4] = h5; o[5] = h6; o[6] = h7; o[7] = h8; o[8] = h9;
  o[9] = h10; o[10] = h11; o[11] = h12; o[12] = h13; o[13] = h14; o[14] = h15; o[15] = h16;
  o[16] = h17; o[17] = h18; o[18] = h19; o[19] = h20; o[20] = h21; o[21] = h22; o[22] = h23; o[23] = h24;
}
#endif
#ifdef EN_FX_HPM
#ifndef EN_FX_HPM_TILE
#define EN_FX_HPM_TILE 2u
#endif
#define FX_HPM(Tl, idx) do { if ((Tl) == EN_FX_HPM_TILE) fx_hpm_snap(tid, (idx)); } while (0)
#else
#define FX_HPM(Tl, idx) do { } while (0)
#endif
template <typename St>
__attribute__((noinline)) static void fx_stage(uint32_t tid, uint32_t n) {
  const Lane L = lane_of(tid, n);
  const uint32_t nw = total_warps(L), nwk = nw - 1u, slot = warp_slot(L);
  // Warp 0 drives the GEMM and, while the mesh is idle, works on the epilogue as the last worker.
#ifdef EN_FX_MAXTILES   // diagnostic (waveform runs): only the first EN_FX_MAXTILES tiles
  const uint32_t LT = St::tiles / EN_CLUSTERS < EN_FX_MAXTILES ? St::tiles / EN_CLUSTERS : EN_FX_MAXTILES;
#else
  const uint32_t LT = St::tiles / EN_CLUSTERS;   // this cluster's tiles
#endif
  if (slot == 0u) {
    fx_prologue<St>(tid);
#ifdef EN_FX_PROF   // warp 0: k-steps | wait at barrier 4 | own epilogue work | wait at barrier 5 | out
    uint32_t fp[5] = {0, 0, 0, 0, 0};
#endif
#ifdef EN_FX_OUT_OVL
    uint32_t prev_T = ~0u;   // its out is issued under the next tile's first k-step
#endif
    for (uint32_t Tl = 0; Tl < LT; Tl++) {
      const uint32_t T = fx_global<St>(Tl);
#ifdef EN_FX_PROF
      const uint32_t c0 = mcycle();
#endif
#ifdef EN_FX_OUT_OVL
      fx_tile<St>(tid, Tl, T, prev_T);
      prev_T = T;
#else
      fx_tile<St>(tid, Tl, T);
#endif
#ifdef EN_FX_PROF
      const uint32_t c1 = mcycle();
#endif
      mu_barrier(4, nw);   // C(T) in SMEM
#ifdef EN_FX_PROF
      const uint32_t c2 = mcycle();
#endif
      FX_HPM(Tl, 0u);
      if constexpr (fx_halves<St>::v) {
        St::work_range(T, L.lane, nwk, nw, 0u, EN_TM / 8u);
        mu_fence_smem();
        mu_barrier(7, nw);   // rows 0-63 done: their move-out runs under the second half
        fx_out_half<St>(tid, T, 0u);
        St::work_range(T, L.lane, nwk, nw, EN_TM / 8u, EN_TM / 4u);
      } else {
        St::work(T, L.lane, nwk, nw);
      }
      mu_fence_smem();
      FX_HPM(Tl, 1u);
#ifdef EN_FX_PROF
      const uint32_t c3 = mcycle();
#endif
      mu_barrier(5, nw);   // everyone done with T
#ifdef EN_FX_PROF
      const uint32_t c4 = mcycle();
#endif
#ifndef EN_FX_OUT_OVL
      if constexpr (fx_halves<St>::v) fx_out_half<St>(tid, T, 1u);
      else fx_out<St>(tid, T);
#endif
#ifdef EN_FX_PROF
      const uint32_t c5 = mcycle();
      fp[0] += c1 - c0; fp[1] += c2 - c1; fp[2] += c3 - c2; fp[3] += c4 - c3; fp[4] += c5 - c4;
#endif
    }
#ifdef EN_FX_OUT_OVL
    fx_out<St>(tid, prev_T);   // the last tile's out, drained
#endif
#ifdef EN_FX_PROF
    if (is_lead(tid)) { for (uint32_t j = 0; j < 5u; j++) rad_pb_put(EN_PB_UPROBE + j, fp[j]); } else { asm volatile("nop"); }
#endif
  } else {
    for (uint32_t Tl = 0; Tl < LT; Tl++) {
      const uint32_t T = fx_global<St>(Tl);
      mu_barrier(4, nw);
      FX_HPM(Tl, 0u);
      if constexpr (fx_halves<St>::v) {
        St::work_range(T, L.lane, fx_wk(slot), nw, 0u, EN_TM / 8u);
        mu_fence_smem();
        mu_barrier(7, nw);
        St::work_range(T, L.lane, fx_wk(slot), nw, EN_TM / 8u, EN_TM / 4u);
      } else {
        St::work(T, L.lane, fx_wk(slot), nw);
      }
      mu_fence_smem();
      FX_HPM(Tl, 1u);
      mu_barrier(5, nw);
    }
  }
}

// ---- EN_FX_OVL: the epilogue of tile T-1 runs under tile T's k-steps ------------------------------
// Needs MXG_BANKSPLIT (operands in banks 0 / 3, C in bank 1, outputs in bank 2) and the overlap rules
// (memory note smem-overlap-rules): workers LOAD only from bank 1 (C), which no matmul operand or
// move-out reads, and STORE to bank 2; while they mix loads and stores, no warp reads the control
// bus, so thread 0 only ISSUES the k-step's commands and drains (MMIO polls) after the barrier that
// ends the workers' chunk.  Per k-step kt of tile T:
//   barrier 4 | thread 0: issue k-step kt (no drain)        || workers: chunk kt of T-1's epilogue
//             | (kt = 3: also issue T-1's output move-outs)  || (kt = 3: idle: the step stores C(T))
//   barrier 5 | thread 0: drain                              || workers: wait at the next barrier 4
// One more tile of rounds after the last tile finishes its epilogue.  Workers = warps 1 .. nw-1.
template <typename St> __attribute__((noinline)) static void fxo_issue(uint32_t tid, uint32_t Tl, uint32_t kt) {
  if (tid != 0) return;
  const mxg::Problem p = en_split(St::problem());
  const mxg::Tiling t = mxg::tiling_of(p);
  const uint32_t k_tiles = p.k / p.tk, n_tiles = mxg::ncols_of(p) / p.tn;
  const uint32_t steps = (St::tiles / EN_CLUSTERS) * k_tiles, s = Tl * k_tiles + kt;
  const bool more = s + 1 < steps;
  if (kt == 0u) St::pre(fx_global<St>(Tl));
  mxg::select_scales(p, t, s);
#ifdef MXG_MVIN_WIDE   // as fx_step: matmul first
  mxg::matmul_tile(t, s, kt == 0, kt + 1u == k_tiles);
#endif
  if (more) {
    const uint32_t nx = s + 1, tile = nx / k_tiles;
    mxg::mvin_tiles(t, p.a + (tile / n_tiles) * p.tm * p.k + (nx % k_tiles) * p.tk,
                    p.b + (tile % n_tiles) * p.tn + (nx % k_tiles) * p.tk * p.n, nx);
  }
#ifndef MXG_MVIN_WIDE
  mxg::matmul_tile(t, s, kt == 0, kt + 1u == k_tiles);
#endif
#ifndef EN_FXO_SCL_LATE
  if (more) {
    const uint32_t nx = s + 1, tile = nx / k_tiles;
    mxg::load_tile_scales(p, tile / n_tiles, tile % n_tiles, nx % k_tiles, nx);
    mu_fence_smem();
  }
#endif
}
// EN_FXO_SCL_LATE: the next step's scale-SRAM writes (control-bus stores) after the drain, while
// the workers wait at a barrier, instead of under their SMEM phases (diagnosis of the ovl2 stall)
template <typename St> __attribute__((noinline)) static void fxo_scales(uint32_t tid, uint32_t Tl, uint32_t kt) {
  if (tid != 0) return;
#ifdef EN_FXO_SCL_LATE
  const mxg::Problem p = en_split(St::problem());
  const uint32_t k_tiles = p.k / p.tk, n_tiles = mxg::ncols_of(p) / p.tn;
  const uint32_t steps = (St::tiles / EN_CLUSTERS) * k_tiles, s = Tl * k_tiles + kt;
  if (s + 1 < steps) {
    const uint32_t nx = s + 1, tile = nx / k_tiles;
    mxg::load_tile_scales(p, tile / n_tiles, tile % n_tiles, nx % k_tiles, nx);
    mu_fence_smem();
  }
#else
  (void)Tl; (void)kt;
#endif
}
template <typename St> __attribute__((noinline)) static void fxo_out(uint32_t tid, uint32_t T) {
  if (tid != 0) return;
  St::out(T);   // issued after this k-step's LOOP_WS commands; drained after barrier 5
}
__attribute__((noinline)) static void fxo_drain(uint32_t tid) {
  if (tid != 0) return;
  mxg::drain();
}
template <typename St>
__attribute__((noinline)) static void fx_stage_ovl(uint32_t tid, uint32_t n) {
  const Lane L = lane_of(tid, n);
  const uint32_t nw = total_warps(L), slot = warp_slot(L), nwk = nw - 1u;
  constexpr uint32_t K = EN_D / 128u, IT = EN_TM / 4u, CH = (IT + 2u) / 3u;   // 3 chunks of 4-row slices
  static_assert(K == 4u, "chunks in k-steps 0-2, C(T) stored in k-step 3");
  const uint32_t LT = St::tiles / EN_CLUSTERS;
  if (slot == 0u) {
    fx_prologue<St>(tid);
    for (uint32_t Tl = 0; Tl <= LT; Tl++)
      for (uint32_t kt = 0; kt < K; kt++) {
        mu_barrier(4, nw);
        if (Tl < LT) fxo_issue<St>(tid, Tl, kt); else asm volatile("nop");
        if (kt + 1u == K && Tl > 0u) fxo_out<St>(tid, fx_global<St>(Tl - 1u)); else asm volatile("nop");
        mu_barrier(5, nw);
        fxo_drain(tid);
      }
  } else {
    uint32_t wk;
    asm volatile("mv %0, %1" : "=r"(wk) : "r"(slot - 1u));   // hidden from LLVM (no unswitching)
    for (uint32_t Tl = 0; Tl <= LT; Tl++)
      for (uint32_t kt = 0; kt < K; kt++) {
        mu_barrier(4, nw);
        if (Tl > 0u && kt + 1u < K) {
          const uint32_t it1 = (kt + 1u) * CH < IT ? (kt + 1u) * CH : IT;
          St::work_range(fx_global<St>(Tl - 1u), L.lane, wk, nwk, kt * CH, it1);
          mu_fence_smem();
        } else {
          asm volatile("nop");
        }
        mu_barrier(5, nw);
      }
  }
}

// ---- EN_FX_OVL2: as EN_FX_OVL, but the workers never have SMEM loads and stores in flight together
// (mixed traffic under a matmul fails on silicon even without MMIO polls: board fzc n1 / p4).  Each
// worker round: every worker loads one 4-row slice of C(T-1) (bank 1) into registers, fence.s, worker
// barrier 8; computes and stores codes / scale (bank 2), fence.s, worker barrier 9.  Rounds per k-step
// RPK = {3, 2, 2, 0} (32 slices on 5 workers; none in k-step 3, which stores C(T) into bank 1).
#ifdef EN_UPGELU_CF
#define FXO_PIN_SAVES() asm volatile("" ::: "s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21","s22","s23","s24","s25","s26","s27")
// One round: every worker one 4-row slice, in four phases (loads / GELU + stores / loads / code
// stores) separated by worker barriers.  Inlined into fxo_tile_work.
template <typename St>
__attribute__((always_inline)) static inline void fxo_round(uint32_t it, uint32_t lane, uint32_t nwk) {
  constexpr uint32_t IT = EN_TM / 4u;
  uint32_t xw[16];
  int e = 0;
  if (it < IT) St::load_it(it, lane, xw); else asm volatile("nop");
  mu_fence_smem();
  mu_barrier(8, nwk);
  if (it < IT) e = St::gelu_it(it, lane, xw); else asm volatile("nop");
  mu_fence_smem();
  mu_barrier(9, nwk);
  if (it < IT) St::load_it(it, lane, xw); else asm volatile("nop");
  mu_fence_smem();
  mu_barrier(8, nwk);
  if (it < IT) St::code_it(it, lane, xw, e); else asm volatile("nop");
  mu_fence_smem();
  mu_barrier(9, nwk);
}
// A worker's whole tile (the four k-steps' barriers 4 / 5 and the 7 rounds {3, 2, 2, 0}) as one
// call.  As a function per round, the 29 callee-saved registers went to the GLOBAL stack and back
// on every round: ~7k cycles per round of a ~6k-cycle round (board fzs: same with the mesh idle).
// Now once per tile, and outside every SMEM phase: the saves come before the first barrier 4 (an asm
// that clobbers every callee-saved register pins the save point to the entry; shrink-wrapping had
// moved it past the first barrier), the restores after the last barrier 5.  it0 = IT skips the
// slices (the first tile, before any C exists).
#ifdef EN_FXO_NOOVL_PROBE   // timing / correctness probe: thread 0 drains the k-step before barrier 7
#define FXO_PROBE_WAIT() mu_barrier(7, nw)
#else
#define FXO_PROBE_WAIT() do {} while (0)
#endif
#ifdef EN_FX_PROF
struct FxoTileProf { uint32_t a, b; };   // rounds, scale_fix (a0:a1)
#define FXO_TW_RET FxoTileProf
#define FXO_T(v) const uint32_t v = mcycle()
#else
#define FXO_TW_RET void
#define FXO_T(v) do {} while (0)
#endif
template <typename St>
__attribute__((noinline)) static FXO_TW_RET fxo_tile_work(uint32_t it0, uint32_t wk, uint32_t lane, uint32_t nwk, uint32_t nw) {
  FXO_PIN_SAVES();
  mu_barrier(4, nw);   // k-step 0
  FXO_PROBE_WAIT();
  // ONE rolled copy of the round: inlined 7 times the function was ~30 KB, beyond the 16 KB
  // direct-mapped icache; it ran ~14k cycles per ~6k-cycle round and the GELU output was wrong
  // (board fzu): a warp stalled on fetch with SMEM responses pending also backs up the D path the
  // mesh's operand reads wait on.  Rounds 0-2 in k-step 0, 3-4 in k-step 1, 5-6 in k-step 2.
#ifdef EN_FX_PROF
  uint32_t tr = 0;
#endif
#pragma clang loop unroll(disable)
  for (uint32_t r = 0; r < 7u; r++) {
    if (r == 3u || r == 5u) {   // the next k-step
      mu_barrier(5, nw);
      mu_barrier(4, nw);
  FXO_PROBE_WAIT();
    } else {
      asm volatile("nop");
    }
    FXO_T(c0);
    fxo_round<St>(it0 + r * nwk + wk, lane, nwk);
#ifdef EN_FX_PROF
    tr += mcycle() - c0;
#endif
  }
  FXO_T(c5);
  St::scale_fix(lane, wk, nwk);
  FXO_T(c6);
  mu_barrier(5, nw);
  mu_barrier(4, nw);   // k-step 3: C(T) is stored into the C slot, no rounds
  FXO_PROBE_WAIT();
  mu_barrier(5, nw);
  // barrier 6: thread 0 has drained k-step 3 (mesh idle) and waits here, so the restores below and
  // the next call's saves (GLOBAL stack traffic) never run under a matmul (fzt: with the restores
  // right after barrier 5, one of the two clusters hung)
  mu_barrier(6, nw);
  FXO_PIN_SAVES();
#ifdef EN_FX_PROF
  return {tr, c6 - c5};
#endif
}
#endif
template <typename St>
__attribute__((noinline)) static void fx_stage_ovl2(uint32_t tid, uint32_t n) {
  const Lane L = lane_of(tid, n);
  const uint32_t nw = total_warps(L), slot = warp_slot(L), nwk = nw - 1u;
  constexpr uint32_t K = EN_D / 128u, IT = EN_TM / 4u;
  static_assert(K == 4u, "rounds in k-steps 0-2, C(T) stored in k-step 3");
  const uint32_t LT = St::tiles / EN_CLUSTERS;
  if (slot == 0u) {
    fx_prologue<St>(tid);
#ifdef EN_FX_PROF   // thread 0 per k-step: issue | wait at barrier 5 | drain + scales | wait at barrier 4
    uint32_t fp[4] = {0, 0, 0, 0};
#endif
    for (uint32_t Tl = 0; Tl <= LT; Tl++)
      for (uint32_t kt = 0; kt < K; kt++) {
#ifdef EN_FX_PROF
        const uint32_t c0 = mcycle();
#endif
        mu_barrier(4, nw);
#ifdef EN_FX_PROF
        const uint32_t c1 = mcycle();
#endif
        if (Tl < LT) fxo_issue<St>(tid, Tl, kt); else asm volatile("nop");
        if (kt + 1u == K && Tl > 0u) fxo_out<St>(tid, fx_global<St>(Tl - 1u)); else asm volatile("nop");
#ifdef EN_FXO_NOOVL_PROBE   // timing probe: the k-step's commands finish before the rounds start
        fxo_drain(tid);
        mu_barrier(7, nw);
#endif
#ifdef EN_FX_PROF
        const uint32_t c2 = mcycle();
#endif
        mu_barrier(5, nw);
#ifdef EN_FX_PROF
        const uint32_t c3 = mcycle();
#endif
        fxo_drain(tid);
        if (Tl < LT) fxo_scales<St>(tid, Tl, kt); else asm volatile("nop");
#ifdef EN_UPGELU_CF
        if (kt + 1u == K) mu_barrier(6, nw); else asm volatile("nop");   // mesh idle: worker stack traffic
#endif
#ifdef EN_FX_PROF
        const uint32_t c4 = mcycle();
        fp[0] += c2 - c1; fp[1] += c3 - c2; fp[2] += c4 - c3; fp[3] += c1 - c0;
#endif
      }
#ifdef EN_FX_PROF
    if (is_lead(tid)) { for (uint32_t j = 0; j < 4u; j++) rad_pb_put(EN_PB_UPROBE + j, fp[j]); } else { asm volatile("nop"); }
#endif
  } else {
    uint32_t wk;
    asm volatile("mv %0, %1" : "=r"(wk) : "r"(slot - 1u));   // hidden from LLVM (no unswitching)
#ifdef EN_UPGELU_CF
    static_assert(7u * 5u >= IT, "7 rounds of 5 workers cover the tile's slices");
#ifdef EN_FX_PROF
    uint32_t wt = 0, ws[4] = {0, 0, 0, 0};
#endif
    for (uint32_t Tl = 0; Tl <= LT; Tl++) {
#ifdef EN_FX_PROF
      const uint32_t c0 = mcycle();
      const FxoTileProf q = fxo_tile_work<St>(Tl > 0u ? 0u : IT, wk, L.lane, nwk, nw);
      wt += mcycle() - c0;
      ws[0] += q.a; ws[3] += q.b;
#else
      fxo_tile_work<St>(Tl > 0u ? 0u : IT, wk, L.lane, nwk, nw);
#endif
    }
#ifdef EN_FX_PROF
    if (wk == 0u && L.lane == 0u) {
      rad_pb_put(EN_PB_UPROBE + 5u, wt);
      for (uint32_t j = 0; j < 4u; j++) rad_pb_put(EN_PB_UPROBE + 8u + j, ws[j]);
    } else {
      asm volatile("nop");
    }
#endif
#else
#ifdef EN_FX_PROF   // worker 0 per k-step: wait at barrier 4 | rounds | (unused) | wait at barrier 5
    uint32_t wp[4] = {0, 0, 0, 0};
#endif
    for (uint32_t Tl = 0; Tl <= LT; Tl++) {
      uint32_t base = 0;   // first slice of this round
      for (uint32_t kt = 0; kt < K; kt++) {
#ifdef EN_FX_PROF
        const uint32_t c0 = mcycle();
#endif
        mu_barrier(4, nw);
#ifdef EN_FXO_NOOVL_PROBE
        mu_barrier(7, nw);
#endif
#ifdef EN_FX_PROF
        const uint32_t c1 = mcycle();
#endif
        const uint32_t rounds = (Tl > 0u) ? (kt == 0u ? 3u : (kt + 1u < K ? 2u : 0u)) : 0u;
        for (uint32_t r = 0; r < rounds; r++) {
          const uint32_t it = base + wk;
          uint32_t xw[16];
          if (it < IT) St::load_it(it, L.lane, xw); else asm volatile("nop");
          mu_fence_smem();
          mu_barrier(8, nwk);
          if (it < IT) St::finish_it(it, L.lane, xw); else asm volatile("nop");
          mu_fence_smem();
          mu_barrier(9, nwk);
          base += nwk;
        }
#ifdef EN_FX_PROF
        const uint32_t c2 = mcycle();
#endif
        mu_barrier(5, nw);
#ifdef EN_FX_PROF
        const uint32_t c3 = mcycle();
        wp[0] += c1 - c0; wp[1] += c2 - c1; wp[3] += c3 - c2;
#endif
      }
    }
#ifdef EN_FX_PROF
    if (wk == 0u && L.lane == 0u) { for (uint32_t j = 0; j < 4u; j++) rad_pb_put(EN_PB_UPROBE + 4u + j, wp[j]); } else { asm volatile("nop"); }
#endif
#endif
  }
}

// ---- stage: FFN up GEMM + GELU + quantize (ff stays in SMEM) -------------------------------------
// Worker w takes 4-row slices it = w, w + nwk, ... of the tile; lane l handles group g = l % 4 of
// row 4 it + k (k = l / 4).  Its loads are rotated by 4k words (natural word (i + 4k) & 15 lands in
// xw[i]) so the 16 lanes hit 16 different subbanks; the code words go back to their natural places,
// which is conflict-free for the same reason.  Codes [128 rows][128 B] at FX_LO, scales
// [4 groups][128 rows] at FX_HI.
struct UpGelu {
  static constexpr uint32_t tiles = (EN_S / EN_TM) * (EN_F / EN_TN);   // 32
  static void pre(uint32_t) {}
  static mxg::Problem problem() {
    return {EN_H2, EN_W1, EN_FF, EN_H2_S, EN_W1_S, EN_S, EN_F, EN_D, EN_TM, EN_TN, 128u, EN_SCRATCH};
  }
  UG_WORK_ATTR static void work(uint32_t T, uint32_t l, uint32_t w, uint32_t nwk) { work_range(T, l, w, nwk, 0u, EN_TM / 4u); }
  // iterations it0 + w, it0 + w + nwk, ... below it1 (4-row slices)
  // one 4-row slice (iteration it): load_it reads C into registers (SMEM loads only), finish_it
  // computes GELU + quantize and writes codes / scale (SMEM stores only); EN_FX_OVL2 runs the two
  // in separate phases so no warp has SMEM loads and stores in flight together
#ifndef EN_UPGELU_CF
  __attribute__((always_inline)) static inline void load_it(uint32_t it, uint32_t l, uint32_t (&xw)[16]) {
    const uint32_t k = l / 4u, g = l % 4u, row = 4u * it + k;
    // rotation by whole 16-byte chunks: four chunk pointers, four fixed-offset loads each
    const uint32_t cbase = FX_C + row * 256u + g * 64u;
#pragma unroll
    for (uint32_t c = 0; c < 4u; c++) {
      const volatile __shared uint32_t *cp = smem<uint32_t>(cbase + ((c + k) & 3u) * 16u);
#pragma unroll
      for (uint32_t j = 0; j < 4u; j++) xw[4u * c + j] = cp[j];
    }
  }
  __attribute__((always_inline)) static inline void finish_it(uint32_t it, uint32_t l, uint32_t (&xw)[16]) {
    __shared uint8_t *sc = smem<uint8_t>(FX_HI);
    const uint32_t k = l / 4u, g = l % 4u, row = 4u * it + k;
    uint32_t mx = 0;
#if defined(EN_UPGELU_ASMQ)
    // EN_UPGELU_ASMQ: banked asm for GELU (EN_GELU_ASM8), block max and encoder
#pragma unroll
    for (uint32_t i = 0; i < 16u; i += 4u) gelu_bf16x8(xw[i], xw[i + 1u], xw[i + 2u], xw[i + 3u]);
    {
      uint32_t m0 = 0, m1 = 0, m2 = 0, m3 = 0;
#pragma unroll
      for (uint32_t i = 0; i < 16u; i += 4u) swar_max4(m0, m1, m2, m3, xw[i], xw[i + 1u], xw[i + 2u], xw[i + 3u]);
      mx = fmax_h(fmax_h(m0, m1), fmax_h(m2, m3));
    }
#elif defined(EN_GELU_ILP)
#pragma unroll
    for (uint32_t i = 0; i < 16u; i += 4u) gelu_bf16x8(xw[i], xw[i + 1u], xw[i + 2u], xw[i + 3u]);
#pragma unroll
    for (uint32_t i = 0; i < 16u; i++) {
      const uint32_t a = xw[i] & 0x7FFF7FFFu;
      mx = fmax_h(fmax_h(mx, a), a >> 16);
    }
#else
#pragma unroll
    for (uint32_t i = 0; i < 16u; i++) {
      xw[i] = gelu_bf16x2(xw[i]);
      const uint32_t a = xw[i] & 0x7FFF7FFFu;
      mx = fmax_h(fmax_h(mx, a), a >> 16);
    }
#endif
    const int e = block_exp(mx << 16);
    const uint32_t p2 = swar_p2(e);
#ifdef EN_UPGELU_WIDE
    // codes in the 2-block move-out layout (out()): 16-byte chunk ch of tile row r at SMEM row
    // ((r / 16) * 8 + ch) * 16 + r % 16 of FX_LO.  This lane's 8-byte piece p of group g is chunk
    // 2g + p / 2, bytes (p % 2) * 8 ..
    const uint32_t obase = FX_LO + ((row / 16u) * 8u + 2u * g) * 256u + (row % 16u) * 16u;
#pragma unroll
    for (uint32_t c = 0; c < 4u; c++) {   // code words 2c, 2c+1 of the rotated order -> piece (c + k) & 3
      const uint32_t p = (c + k) & 3u;
      __shared uint32_t *op = smem<uint32_t>(obase + (p / 2u) * 256u + (p % 2u) * 8u);
#ifdef EN_UPGELU_ASMQ
      uint32_t c0, c1;
      e4m3_enc4(xw[4u * c], xw[4u * c + 1u], xw[4u * c + 2u], xw[4u * c + 3u], p2, c0, c1);
      op[0] = c0;
      op[1] = c1;
#else
      op[0] = swar_pack4(e4m3_swar2(xw[4u * c], p2), e4m3_swar2(xw[4u * c + 1u], p2));
      op[1] = swar_pack4(e4m3_swar2(xw[4u * c + 2u], p2), e4m3_swar2(xw[4u * c + 3u], p2));
#endif
    }
#else
    const uint32_t obase = FX_LO + row * 128u + g * 32u;
#pragma unroll
    for (uint32_t c = 0; c < 4u; c++) {   // code words 2c, 2c+1 of the rotated order -> chunk (c + k) & 3
      __shared uint32_t *op = smem<uint32_t>(obase + ((c + k) & 3u) * 8u);
      op[0] = swar_pack4(e4m3_swar2(xw[4u * c], p2), e4m3_swar2(xw[4u * c + 1u], p2));
      op[1] = swar_pack4(e4m3_swar2(xw[4u * c + 2u], p2), e4m3_swar2(xw[4u * c + 3u], p2));
    }
#endif
    sc[g * 128u + row] = (uint8_t)(e + 127);
  }
#else   // EN_UPGELU_CF
  // EN_UPGELU_CF: every SMEM instruction of a slice touches 16 different subbanks, so the 16 lanes
  // of a warp complete together.  SMEM is 4 banks x 16 subbanks of 4 B (subbank = (addr / 4) % 16
  // within a 32 KB bank).  The lanes of a warp that complete out of step stall mesh operand reads
  // in every bank and the mesh then drops a row (RS 568 deadlock or wrong C; runs/overlap-probe
  // NOTES.md), so this is the addressing that SIMT work under a matmul needs.  The chunk rotation
  // above still puts lanes g = 0..3 of one row in one subbank.
  // Lane l = (k = l / 4, g = l % 4) still owns group g of row 4 it + k (the block max stays in the
  // lane).  Pair step m reads word pair pi = (m + l) & 7 of the group; lanes 8-15 read the odd word
  // first, so one load instruction covers 16 different word offsets.
  //
  // A slice runs in four phases, each with a small live set (at 64 GPRs the one-pass form spilled
  // ~60 GLOBAL stack accesses per slice, 300k cycles on the layer, board fzl): load_it (16 loads),
  // gelu_it (GELU + block max, the GELU words back in place, the scale), load_it again, code_it
  // (codes).  Phases alternate SMEM loads and stores, so EN_FX_OVL2 separates them by barriers.
  static inline uint32_t cf_pi(uint32_t m, uint32_t l) { return (m + l) & 7u; }
  // LLVM hoists lane-only offsets out of the slice loop and spills them; a volatile move per phase
  // keeps them inside it
  static inline uint32_t cf_opq(uint32_t v) {
    uint32_t r;
    asm volatile("mv %0, %1" : "=r"(r) : "r"(v));
    return r;
  }
  // address of xw[i] of lane l's group in slice it
  static inline uint32_t cf_word(uint32_t it, uint32_t l, uint32_t i) {
    const uint32_t row = 4u * it + l / 4u;
    return FX_C + row * 256u + (l % 4u) * 64u + cf_pi(i / 2u, l) * 8u + (((i & 1u) ^ (l >> 3)) << 2);
  }
  __attribute__((always_inline)) static inline void load_it(uint32_t it, uint32_t l0, uint32_t (&xw)[16]) {
    const uint32_t l = cf_opq(l0);
#pragma unroll
    for (uint32_t i = 0; i < 16u; i++)
      xw[i] = *reinterpret_cast<const volatile __shared uint32_t *>(smem<uint32_t>(cf_word(it, l, i)));
  }
#ifdef EN_FX_OVL2
  // scale bytes: one word per lane, word (l + it) & 15 of slice it's 64 B line (a byte store per
  // lane into the [g][128] layout puts all 16 lanes in one subbank); scale_fix moves them
  static constexpr uint32_t FX_SCW = FX_HI + 0x400u;
#endif
  // GELU in place + block max; returns the block exponent
  __attribute__((always_inline)) static inline int gelu_it(uint32_t it, uint32_t l0, uint32_t (&xw)[16]) {
    uint32_t mx = 0;
#pragma unroll
    for (uint32_t i = 0; i < 16u; i++) {
      const uint32_t y = gelu_bf16x2(xw[i]);
      *reinterpret_cast<volatile __shared uint32_t *>(smem<uint32_t>(cf_word(it, cf_opq(l0), i))) = y;
      const uint32_t a = y & 0x7FFF7FFFu;
      mx = fmax_h(fmax_h(mx, a), a >> 16);
    }
    const int e = block_exp(mx << 16);
    const uint32_t l = cf_opq(l0);
#ifdef EN_FX_OVL2
    *smem<uint32_t>(FX_SCW + it * 64u + ((l + it) & 15u) * 4u) = (uint32_t)(e + 127);
#else
    smem<uint8_t>(FX_HI)[(l % 4u) * 128u + 4u * it + l / 4u] = (uint8_t)(e + 127);   // mesh idle
#endif
    return e;
  }
  // code word pi of the group (4 codes of words 2 pi, 2 pi + 1) in the 2-block move-out layout:
  // chunk 2g + pi / 4 of tile row `row`, bytes (pi % 4) * 4; subbank 4 k + (m + g) % 4
  __attribute__((always_inline)) static inline void code_it(uint32_t it, uint32_t l0, const uint32_t (&xw)[16], int e) {
    const uint32_t p2 = swar_p2(e);
    const uint32_t l = cf_opq(l0), k = l / 4u, g = l % 4u, row = 4u * it + k;
    const uint32_t sw = 0u - (l >> 3);   // all ones where xw[2m] holds the odd word
    const uint32_t obase = FX_LO + ((row / 16u) * 8u + 2u * g) * 256u + (row % 16u) * 16u;
#pragma unroll
    for (uint32_t m = 0; m < 8u; m++) {
      const uint32_t a = e4m3_swar2(xw[2u * m], p2), b = e4m3_swar2(xw[2u * m + 1u], p2);
      const uint32_t d = and_asm(a ^ b, sw);
      const uint32_t cw = cf_pi(m, l);
      *smem<uint32_t>(obase + (cw / 4u) * 256u + (cw % 4u) * 4u) = swar_pack4(a ^ d, b ^ d);
    }
  }
#ifdef EN_FX_OVL2
  // After a tile's last round: workers 0 and 1 (lane l: slice it = 16 wk + l) gather the 4 rows' bytes
  // of each group into the word out() reads (FX_HI + g * 128 + it * 4).  Loads take staged word
  // (4 k + g + it) & 15 (subbank (4 k + g + l) % 16 across lanes), stores word g * 32 + it
  // (subbank l); loads and stores in separate barrier phases, as the rounds.
  __attribute__((noinline)) static void scale_fix(uint32_t l, uint32_t wk, uint32_t nwk) {
    const uint32_t it = 16u * wk + l;
    uint32_t v[16];
    if (wk < 2u) {
#pragma unroll
      for (uint32_t j = 0; j < 16u; j++)   // j = 4 g + k
        v[j] = *reinterpret_cast<const volatile __shared uint32_t *>(
            smem<uint32_t>(FX_SCW + it * 64u + ((4u * (j % 4u) + j / 4u + it) & 15u) * 4u));
    } else {
      asm volatile("nop");
    }
    mu_fence_smem();
    mu_barrier(8, nwk);
    if (wk < 2u) {
#pragma unroll
      for (uint32_t g = 0; g < 4u; g++)
        *smem<uint32_t>(FX_HI + g * 128u + it * 4u) =
            v[4u * g] | (v[4u * g + 1u] << 8) | (v[4u * g + 2u] << 16) | (v[4u * g + 3u] << 24);
    } else {
      asm volatile("nop");
    }
    mu_fence_smem();
    mu_barrier(9, nwk);
  }
#endif
#endif  // EN_UPGELU_CF
  __attribute__((always_inline)) static inline void work_range(uint32_t T, uint32_t l, uint32_t w, uint32_t nwk,
                                                               uint32_t it0, uint32_t it1) {
    (void)T;
    for (uint32_t it = it0 + w; it < it1; it += nwk) {
      uint32_t xw[16];
#ifdef EN_UPGELU_CF
      load_it(it, l, xw);
      const int e = gelu_it(it, l, xw);
      load_it(it, l, xw);
      code_it(it, l, xw, e);
#else
      load_it(it, l, xw);
      finish_it(it, l, xw);
#endif
    }
  }
  // EN_FX_OUT_HALF: codes of rows 64 h .. 64 h + 63 (h = 0 under the second half's epilogue; h = 1
  // with the scales after it)
  static void out_half(uint32_t T, uint32_t h) {
    const uint32_t mi = T / (EN_F / EN_TN), nj = T % (EN_F / EN_TN);
#ifdef EN_UPGELU_WIDE
    dma_cfg_out(EN_F);
    for (uint32_t rg = 4u * h; rg < 4u * h + 4u; rg++)
      for (uint32_t cp = 0; cp < 4u; cp++)
        dma_out2(EN_G + (mi * EN_TM + 16u * rg) * EN_F + nj * EN_TN + 32u * cp, FX_LO + (rg * 8u + 2u * cp) * 256u);
    dma_cfg_out(16u);
#else
    for (uint32_t r = 64u * h; r < 64u * h + 64u; r++)
      dma_out(EN_G + (mi * EN_TM + r) * EN_F + nj * EN_TN, FX_LO + r * 128u, 8u);
#endif
    if (h == 1u)
      for (uint32_t g = 0; g < 4u; g++)
        dma_out(EN_G_S + a_scale_offset(mi * EN_TM, nj * 4u + g, EN_F), FX_HI + g * 128u, 8u);
  }
  static void out(uint32_t T) {
    const uint32_t mi = T / (EN_F / EN_TN), nj = T % (EN_F / EN_TN);
#ifdef EN_UPGELU_WIDE
    // 32 2-block move-outs of 16 rows x 32 B (chunks 2cp, 2cp + 1 of row group rg) at stride EN_F,
    // instead of 128 1-block move-outs of one 128-byte row each
    dma_cfg_out(EN_F);
    for (uint32_t rg = 0; rg < EN_TM / 16u; rg++)
      for (uint32_t cp = 0; cp < 4u; cp++)
        dma_out2(EN_G + (mi * EN_TM + 16u * rg) * EN_F + nj * EN_TN + 32u * cp, FX_LO + (rg * 8u + 2u * cp) * 256u);
    dma_cfg_out(16u);
#else
    for (uint32_t r = 0; r < EN_TM; r++)
      dma_out(EN_G + (mi * EN_TM + r) * EN_F + nj * EN_TN, FX_LO + r * 128u, 8u);
#endif
    for (uint32_t g = 0; g < 4u; g++)
      dma_out(EN_G_S + a_scale_offset(mi * EN_TM, nj * 4u + g, EN_F), FX_HI + g * 128u, 8u);
  }
};
#ifdef EN_FX_PIPE
// ---- EN_FX_PIPE: UP + GELU with the epilogue under the next tile's matmuls -----------------------
// The serialized fx_stage leaves the mesh idle for the whole epilogue (~25k cycles per tile).  Here
// the epilogue of tile T-1 runs during tile T's four k-steps, one 32-row quarter per k-step, and
// never touches SMEM while the mesh runs (dense SIMT SMEM traffic under a matmul deadlocks
// Gemmini; memory note gemmini-mvout-vs-mesh-overflow).  Per k-step of tile T:
//   barrier 4 (the previous k-step has drained: mesh idle)
//   workers 0-7: load quarter kt of C(T-1) from SMEM into registers (lane: one 32-element group)
//   fence.s, barrier 5
//   warp 0: issue + drain k-step kt of T (the last one stores C(T) over C(T-1): every quarter of
//           C(T-1) has been read by then)   ||   workers: GELU + quantize from registers, codes and
//           scales to DRAM with global stores (the launch epilogue drains L0d / L1)
// After the last tile, four more quarter rounds run without a matmul.
__attribute__((always_inline)) static inline uint32_t fxp_opaque(uint32_t v) {
  uint32_t r;
  asm volatile("mv %0, %1" : "=r"(r) : "r"(v));
  return r;
}
__attribute__((noinline)) static void fxp_scales_out(uint32_t tid, uint32_t T) {
  if (tid != 0) return;
  const uint32_t mi = T / (EN_F / EN_TN), nj = T % (EN_F / EN_TN);
  for (uint32_t g = 0; g < 4u; g++)
    dma_out(EN_G_S + a_scale_offset(mi * EN_TM, nj * 4u + g, EN_F), FX_HI + g * 128u, 8u);
}
__attribute__((noinline)) static void fxp_drain(uint32_t tid) {
  if (tid != 0) return;
  mxg::drain();
}
__attribute__((noinline)) static void fxp_step(uint32_t tid, uint32_t s, bool store_c) {
  if (tid != 0) return;
  const mxg::Problem p = en_split(UpGelu::problem());
  const mxg::Tiling t = mxg::tiling_of(p);
  fx_step<UpGelu>(p, t, s, store_c);
}
// lane (k = l / 4, g = l % 4): tile row 32 q + 4 wk + k, group g; words rotated by 4k as in work()
__attribute__((always_inline)) static inline void fxp_load(uint32_t q, uint32_t wk, uint32_t l, uint32_t *xw) {
  const uint32_t k = l / 4u, g = l % 4u, row = 32u * q + 4u * wk + k;
  const uint32_t cbase = FX_C + row * 256u + g * 64u;
#pragma unroll
  for (uint32_t c = 0; c < 4u; c++) {
    const volatile __shared uint32_t *cp = smem<uint32_t>(cbase + ((c + k) & 3u) * 16u);
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++) xw[4u * c + j] = cp[j];
  }
}
// Scale bytes: a global byte store does not land correctly (board fye: 453/512 scales wrong, all
// word-stored codes right), so they are staged in SMEM (FX_HI, as work()) in the next mesh-idle
// window and thread 0 moves a tile's scales out once they are complete.
__attribute__((always_inline)) static inline uint32_t fxp_compute(uint32_t T, uint32_t q, uint32_t wk, uint32_t l, uint32_t *xw) {
  const uint32_t k = l / 4u, g = l % 4u, row = 32u * q + 4u * wk + k;
  const uint32_t mi = T / (EN_F / EN_TN), nj = T % (EN_F / EN_TN);
  uint32_t mx = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16u; i++) {
    xw[i] = gelu_bf16x2(xw[i]);
    const uint32_t a = xw[i] & 0x7FFF7FFFu;
    mx = fmax_h(fmax_h(mx, a), a >> 16);
  }
  const int e = block_exp(mx << 16);
  const uint32_t p2 = swar_p2(e);
  volatile uint32_t *og = reinterpret_cast<volatile uint32_t *>(EN_G + (mi * EN_TM + row) * EN_F + nj * EN_TN + g * 32u);
#pragma unroll
  for (uint32_t c = 0; c < 4u; c++) {   // xw[4c..4c+3] are natural words 4m..4m+3, m = (c + k) & 3
    const uint32_t m = (c + k) & 3u;
    og[2u * m] = swar_pack4(e4m3_swar2(xw[4u * c], p2), e4m3_swar2(xw[4u * c + 1u], p2));
    og[2u * m + 1u] = swar_pack4(e4m3_swar2(xw[4u * c + 2u], p2), e4m3_swar2(xw[4u * c + 3u], p2));
  }
  return (uint32_t)(e + 127);   // the scale byte goes to SMEM in the next mesh-idle window
}
__attribute__((noinline)) static void st_upgelu(uint32_t tid, uint32_t n) {
  const Lane L = lane_of(tid, n);
  const uint32_t nw = total_warps(L), slot = warp_slot(L);
  constexpr uint32_t K = EN_D / 128u;   // k-steps per tile (TK = 128)
  static_assert(K == 4u && EN_TM == 128u, "one 32-row quarter of C per k-step");
  const uint32_t LT = UpGelu::tiles / EN_CLUSTERS;
  // Tile X's quarter q is computed in round (X + 1, q); its scale bytes reach SMEM in the next
  // round's window, so tile X's scales are complete after the window of round (X + 2, 0) and move
  // out right after it (drained with that round's k-step, before tile X + 1 writes the buffer).
  // One extra window after the rounds finishes the last tile.
  if (slot == 0u) {
    if (tid == 0u) fx_prologue<UpGelu>(tid); else asm volatile("nop");
    uint32_t p_b4 = 0, p_b5 = 0, p_step = 0;   // thread 0: waiting for compute, for loads, k-steps
    for (uint32_t Tl = 0; Tl <= LT; Tl++)
      for (uint32_t kt = 0; kt < K; kt++) {
        const uint32_t c0 = mcycle();
        mu_barrier(4, nw);
        const uint32_t c1 = mcycle();
        mu_barrier(5, nw);
        const uint32_t c2 = mcycle();
        if (kt == 0u && Tl >= 2u) fxp_scales_out(tid, fx_global<UpGelu>(Tl - 2u)); else asm volatile("nop");
        if (Tl < LT) fxp_step(tid, Tl * K + kt, kt + 1u == K); else fxp_drain(tid);   // tail: drain the scale move-out
        p_b4 += c1 - c0; p_b5 += c2 - c1; p_step += mcycle() - c2;
      }
    if (is_lead(tid)) {   // host prints these as prof in= (barrier 4) bar= (barrier 5) work= (k-steps)
      rad_pb_put(EN_PB_PROF + 0u, p_b4);
      rad_pb_put(EN_PB_PROF + 1u, p_b5);
      rad_pb_put(EN_PB_PROF + 2u, p_step);
    } else {
      asm volatile("nop");
    }
    mu_barrier(4, nw);
    mu_barrier(5, nw);
    fxp_scales_out(tid, fx_global<UpGelu>(LT - 1u));
    fxp_drain(tid);
  } else {
    const uint32_t wk = fxp_opaque(slot - 1u);   // hidden from LLVM: no unswitching on it
    __shared uint8_t *sc = smem<uint8_t>(FX_HI);
    uint32_t xw[16];
    uint32_t pend = 0;   // 0x100 | scale byte of the quarter computed last round (0: none)
    uint32_t prow = 0;
    const uint32_t g = L.lane % 4u;
    for (uint32_t Tl = 0; Tl <= LT; Tl++)
      for (uint32_t kt = 0; kt < K; kt++) {
        mu_barrier(4, nw);
        if (pend) sc[g * 128u + prow] = (uint8_t)pend; else asm volatile("nop");
        const bool have = Tl > 0u && wk < 8u;
        if (have) fxp_load(kt, wk, L.lane, xw); else asm volatile("nop");
        mu_fence_smem();
        mu_barrier(5, nw);
        pend = 0;
        if (have) {
          pend = 0x100u | fxp_compute(fx_global<UpGelu>(Tl - 1u), kt, wk, L.lane, xw);
          prow = 32u * kt + 4u * wk + L.lane / 4u;
        } else {
          asm volatile("nop");
        }
      }
    mu_barrier(4, nw);
    if (pend) sc[g * 128u + prow] = (uint8_t)pend; else asm volatile("nop");
    mu_fence_smem();
    mu_barrier(5, nw);
  }
}
#else
#ifdef EN_FX_OUT_HALF
template <> struct fx_halves<UpGelu> { static constexpr bool v = true; };
#endif
#if defined(EN_FX_OVL) || defined(EN_FX_OVL2)
#ifndef MXG_BANKSPLIT
#error "EN_FX_OVL / EN_FX_OVL2 need MXG_BANKSPLIT"
#endif
#if defined(EN_UPGELU_CF) && !defined(EN_UPGELU_WIDE)
#error "EN_UPGELU_CF writes the 2-block (EN_UPGELU_WIDE) code layout"
#endif
#ifdef EN_FX_OVL2
__attribute__((noinline)) static void st_upgelu(uint32_t tid, uint32_t n) { fx_stage_ovl2<UpGelu>(tid, n); }
#else
__attribute__((noinline)) static void st_upgelu(uint32_t tid, uint32_t n) { fx_stage_ovl<UpGelu>(tid, n); }
#endif
#else
__attribute__((noinline)) static void st_upgelu(uint32_t tid, uint32_t n) { fx_stage<UpGelu>(tid, n); }
#endif
#endif

// ---- stage: QKV GEMM + split into the FA_ENC operand layouts (qkv stays in SMEM) ----------------
// Output tile (mi, nj): tensor t = nj / 4 (Q, K, V), head h = nj % 4, rows 128 mi ..+128 = four
// 32-row half-units q, each quantized by one worker with the split stage's functions.  Half-unit q
// stages its codes at base(q) and scales at base(q) + 0x1000 (FX_LO holds q = 0, 1; FX_HI q = 2, 3).
struct QkvSplit {
  static constexpr uint32_t tiles = (EN_S / EN_TM) * (3u * EN_D / EN_TN);   // 24
  static void pre(uint32_t) {}
  static mxg::Problem problem() {
    return {EN_H1, EN_WQKV, EN_QKV, EN_H1_S, EN_WQKV_S, EN_S, 3u * EN_D, EN_D, EN_TM, EN_TN, 128u, EN_SCRATCH};
  }
  static uint32_t base(uint32_t q) { return (q < 2u ? FX_LO : FX_HI) + (q & 1u) * 0x1400u; }
  FX_WORK_ATTR static void work(uint32_t T, uint32_t l, uint32_t w, uint32_t nwk) {
    const uint32_t t = (T % 12u) / 4u;
    for (uint32_t q = 0; q < 4u; q++) {
      if ((q + T) % nwk != w) continue;   // half-units rotate over the workers from tile to tile
      const __shared uint32_t *xb = smem<uint32_t>(FX_C + q * 32u * 256u);
      __shared uint32_t *ob = smem<uint32_t>(base(q));
      __shared uint32_t *sb = smem<uint32_t>(base(q) + 0x1000u);
      if (t == 0u) split_work_q(xb, ob, sb, l);
      else if (t == 1u) split_work_k(xb, ob, sb, l);
      else split_work_v(xb, ob, sb, l);
    }
  }
  static void out(uint32_t T) {   // as split_dma_out; leaves the store stride at 16
    const uint32_t mi = T / 12u, nj = T % 12u, t = nj / 4u, h = nj % 4u;
    for (uint32_t q = 0; q < 4u; q++) {
      const uint32_t row0 = (mi * 4u + q) * 32u, b = base(q);
      if (t == 0u) {
        dma_out_contig(EN_FA_Q + (h * EN_S + row0) * EN_DH, b, 32u * EN_DH);
        for (uint32_t g = 0; g < 4u; g++)
          dma_out(EN_FA_QS + h * EN_S * 4u + (row0 / 64u) * 256u + g * 64u + row0 % 64u, b + 0x1000u + g * 32u, 2u);
      } else if (t == 1u) {
        dma_cfg_out(EN_S);
        for (uint32_t c = 0; c < 2u; c++)
          for (uint32_t d0 = 0; d0 < EN_DH; d0 += 16u)
            dma_out(EN_FA_KT + (h * EN_DH + d0) * EN_S + row0 + 16u * c, b + (c * 128u + d0) * 16u, 16u);
        dma_cfg_out(16u);
        for (uint32_t g = 0; g < 4u; g++)
          dma_out(EN_FA_KTS + (h * 4u + g) * EN_S + row0, b + 0x1000u + g * 32u, 2u);
      } else {
        dma_out_contig(EN_FA_V + (h * EN_S + row0) * EN_DH, b, 32u * EN_DH);
        dma_out(EN_FA_VS + (h * (EN_S / 32u) + row0 / 32u) * EN_DH, b + 0x1000u, 8u);
      }
    }
  }
};
__attribute__((noinline)) static void st_qkvsplit(uint32_t tid, uint32_t n) { fx_stage<QkvSplit>(tid, n); }

// ---- stages: GEMM + residual add (out = res + C; C stays in SMEM) ---------------------------------
// pre(T) moves the residual tile in (rows 0-63 to FX_LO, 64-127 to FX_HI, 256 B per row); the
// workers add it to C in place (bf16x2_add: the same rounding as residual2 / ln_quant<true>) and
// out(T) moves the sum out.  N = D for both users, so a row of a tile is 256 B of a 1 KB row.
//   DownRes:  y  = x1 + g @ W2     (replaces down_gemm + res2)
//   OprojRes: x1 = x  + MXq(o) @ Wo (replaces oproj_gemm and the residual half of res1_lnq2)
template <uint32_t A, uint32_t B, uint32_t C, uint32_t AS, uint32_t BS, uint32_t K, uint32_t RES, uint32_t OUT>
struct GemmRes {
  static constexpr uint32_t tiles = (EN_S / EN_TM) * (EN_D / EN_TN);   // 8
  static mxg::Problem problem() { return {A, B, C, AS, BS, EN_S, EN_D, K, EN_TM, EN_TN, 128u, EN_SCRATCH}; }
  static uint32_t res_row(uint32_t r) { return (r < 64u ? FX_LO : FX_HI) + (r % 64u) * 256u; }
  static uint32_t dram_row(uint32_t base, uint32_t T, uint32_t r) {
    return base + (((T / (EN_D / EN_TN)) * EN_TM + r) * EN_D + (T % (EN_D / EN_TN)) * EN_TN) * 2u;
  }
  // Through load state 0 with a 16-byte stride, restored to the A stride afterwards: the loop FSM
  // issues plain LOAD commands and takes the A row stride from state 0.  Nothing is in flight here
  // (every k-step and out() drains).  A move-in through load state 2 (mvin3) hung (metasim
  // enc_m18d / enc_m18o, board tp_enc_h4k).
  static void pre(uint32_t T) {
    mxg::drain();
    dma_cfg_in(16u);
    for (uint32_t r = 0; r < EN_TM; r++) dma_in(res_row(r), dram_row(RES, T, r), 16u);
    mxg::drain();
    gemmini_extended3_config_ld(mxg::tiling_of(problem()).lda, MVIN_SCALE_IDENTITY, false, 0);
  }
  static void work(uint32_t T, uint32_t l, uint32_t w, uint32_t nwk) {
    (void)T;
    __shared uint32_t *c = smem<uint32_t>(FX_C);
#pragma unroll 4
    for (uint32_t idx = 16u * w + l; idx < EN_TM * 64u; idx += 16u * nwk) {
      const __shared uint32_t *x = smem<uint32_t>(res_row(idx / 64u));
      c[idx] = bf16x2_add(c[idx], x[idx % 64u]);
    }
  }
  static void out(uint32_t T) {
    for (uint32_t r = 0; r < EN_TM; r++) dma_out(dram_row(OUT, T, r), FX_C + r * 256u, 16u);
  }
};
using DownRes = GemmRes<EN_G, EN_W2, EN_FF2, EN_G_S, EN_W2_S, EN_F, EN_X1, EN_Y>;
using OprojRes = GemmRes<EN_OA, EN_WO, EN_ATT, EN_OA_S, EN_WO_S, EN_D, EN_X, EN_X1>;
__attribute__((noinline)) static void st_downres(uint32_t tid, uint32_t n) { fx_stage<DownRes>(tid, n); }
__attribute__((noinline)) static void st_oprojres(uint32_t tid, uint32_t n) { fx_stage<OprojRes>(tid, n); }

// The Problem is built here, not passed in: see the note on Lane.
static inline mxg::Problem gemm_problem(uint32_t stage) {
  switch (stage) {
    case EN_STAGE_QKV:
      return {EN_H1, EN_WQKV, EN_QKV, EN_H1_S, EN_WQKV_S, EN_S, 3u * EN_D, EN_D, EN_TM, EN_TN, EN_TK, EN_SCRATCH};
    case EN_STAGE_OPROJ:
      return {EN_OA, EN_WO, EN_ATT, EN_OA_S, EN_WO_S, EN_S, EN_D, EN_D, EN_TM, EN_TN, EN_TK, EN_SCRATCH};
    case EN_STAGE_UP:
      return {EN_H2, EN_W1, EN_FF, EN_H2_S, EN_W1_S, EN_S, EN_F, EN_D, EN_TM, EN_TN, EN_TK, EN_SCRATCH};
    default:   // EN_STAGE_DOWN
      return {EN_G, EN_W2, EN_FF2, EN_G_S, EN_W2_S, EN_S, EN_D, EN_F, EN_TM, EN_TN, EN_TK, EN_SCRATCH};
  }
}
// EN_DOWN_CSPAD: the down GEMM (K = 2048) runs at TK = 128 with C staged in the SMEM C slot (bank 1,
// MXG_BANKSPLIT) and moved out under the next tile instead of a drained ACCMEM move-out per tile:
// 2-cluster down shape 708.7k -> 628.0k (board fzx / fzy).  Not for the K = 512 GEMMs: their
// move-ins already saturate DRAM on two clusters and the overlapped move-out only adds traffic (QKV
// shape 556.2k -> 576.4k).
__attribute__((noinline)) static void gemm(uint32_t tid, uint32_t stage) {
  if (tid != 0) return;
  mxg::drain();   // see dma_setup
#ifdef EN_DOWN_CSPAD
#ifndef MXG_BANKSPLIT
#error "EN_DOWN_CSPAD needs MXG_BANKSPLIT (the C slot must not share a bank with the operand buffers)"
#endif
  if (stage == EN_STAGE_DOWN) {
    mxg::Problem p = en_split(gemm_problem(stage));
    p.tk = 128u;
    const mxg::Tiling t = mxg::tiling_of(p);
    mxg::configure(p, t, true);
    mxg::run_pipe<true>(p, t, [](uint32_t) {});
    return;
  }
#endif
#if defined(EN_DOWN_RESACC) || defined(EN_OPROJ_RESACC)
  // EN_DOWN_RESACC: y = x1 + g @ W2 straight from the down GEMM (x1 added in ACCMEM before each
  // tile's move-out, mxg::resacc_tile); the res2 stage is not run and EN_FF2 is not written.
  // EN_OPROJ_RESACC: x1 = x + MXq(o) @ Wo straight from the out-projection; LN2 then runs alone
  // (st_lnq2 instead of st_res1ln) and EN_ATT is not written.  Both adds are the same bf16 RNE add
  // as the SIMT residuals (EN_RES_HADD), so x1 / h2 / y stay bit-exact (board fzr1 for down).
#ifndef MXG_RESACC
#error "EN_DOWN_RESACC / EN_OPROJ_RESACC need MXG_RESACC"
#endif
  mxg::Problem pr = gemm_problem(stage);
#ifdef EN_DOWN_RESACC
  if (stage == EN_STAGE_DOWN) { pr.c = EN_Y; pr.res = EN_X1; }
#endif
#ifdef EN_OPROJ_RESACC
  if (stage == EN_STAGE_OPROJ) { pr.c = EN_X1; pr.res = EN_X; }
#endif
  const mxg::Problem p = en_split(pr);
#else
  const mxg::Problem p = en_split(gemm_problem(stage));
#endif
  const mxg::Tiling t = mxg::tiling_of(p);
  mxg::configure(p, t, false);
  mxg::run_pipe<false>(p, t, [](uint32_t) {});
}

// ---- stage dispatch ------------------------------------------------------------------------------
// A table of functions, not a switch: `stage` is a per-thread load, so LLVM treats every case test
// as divergent and nests a vx_split per case; the split stage started 6 IPDOM entries deep and its
// DMA code overflowed the 8-entry stack (metasim enc_m9b).  A uniform indirect call opens none.
typedef void (*StageFn)(uint32_t tid, uint32_t nthreads);
__attribute__((noinline)) static void st_none(uint32_t, uint32_t) {}
#if defined(EN_LN_OCC4) && !defined(EN_LN_TU)
// the LN stages come from ln_stage.cpp (this file with fewer allocatable registers)
void en_st_lnq1(uint32_t tid, uint32_t n);
void en_st_lnq2(uint32_t tid, uint32_t n);
void en_st_res1ln(uint32_t tid, uint32_t n);
#define st_lnq1 en_st_lnq1
#define st_lnq2 en_st_lnq2
#define st_res1ln en_st_res1ln
#else
#ifdef EN_LN_TU
#define EN_LN_LINK
#define st_lnq1 en_st_lnq1
#define st_lnq2 en_st_lnq2
#define st_res1ln en_st_res1ln
#else
#define EN_LN_LINK static
#endif
__attribute__((noinline)) EN_LN_LINK void st_lnq1(uint32_t tid, uint32_t n) {
  ln_quant<false>(lane_of(tid, n), tid, EN_X, 0u, 0u, EN_LN1_G, EN_LN1_B, EN_H1, EN_H1_S);
}
__attribute__((noinline)) EN_LN_LINK void st_lnq2(uint32_t tid, uint32_t n) {   // after OprojRes wrote x1
  ln_quant<false>(lane_of(tid, n), tid, EN_X1, 0u, 0u, EN_LN2_G, EN_LN2_B, EN_H2, EN_H2_S);
}
__attribute__((noinline)) EN_LN_LINK void st_res1ln(uint32_t tid, uint32_t n) {
  ln_quant<true>(lane_of(tid, n), tid, EN_X, EN_ATT, EN_X1, EN_LN2_G, EN_LN2_B, EN_H2, EN_H2_S);
}
#endif
__attribute__((noinline)) static void st_oq(uint32_t tid, uint32_t n) { o_quant(lane_of(tid, n), tid); }
__attribute__((noinline)) static void st_geluq(uint32_t tid, uint32_t n) { gelu_quant(lane_of(tid, n), tid); }
__attribute__((noinline)) static void st_res2(uint32_t tid, uint32_t n) { residual2(lane_of(tid, n), tid); }
__attribute__((noinline)) static void st_split(uint32_t tid, uint32_t n) {
#ifdef EN_SPLIT_SIMT
  qkv_split(lane_of(tid, n));
#else
  qkv_split_dma(lane_of(tid, n), tid);
#endif
}
__attribute__((noinline)) static void st_qkv(uint32_t tid, uint32_t) { gemm(tid, EN_STAGE_QKV); }
__attribute__((noinline)) static void st_oproj(uint32_t tid, uint32_t) { gemm(tid, EN_STAGE_OPROJ); }
__attribute__((noinline)) static void st_up(uint32_t tid, uint32_t) { gemm(tid, EN_STAGE_UP); }
__attribute__((noinline)) static void st_down(uint32_t tid, uint32_t) { gemm(tid, EN_STAGE_DOWN); }
__attribute__((noinline)) static void st_genqkv(uint32_t tid, uint32_t nthreads) {   // metasim
  uint32_t *q = reinterpret_cast<uint32_t *>(EN_QKV);
  for (uint32_t w = tid + en_cluster() * nthreads; w < EN_S * 3u * EN_D / 2u; w += nthreads * EN_CLUSTERS) {
    uint32_t v = 0;
    for (uint32_t j = 0; j < 2u; j++) {
      uint32_t h = (2u * w + j) * 8u + 7u;
      h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
      v |= (((h >> 31) << 15) | ((124u + ((h >> 8) % 5u)) << 7) | (h & 0x7Fu)) << (16u * j);
    }
    q[w] = v;
  }
}
__attribute__((noinline)) static void st_genx(uint32_t tid, uint32_t nthreads) {   // metasim
  uint32_t *xw = reinterpret_cast<uint32_t *>(EN_X);
  for (uint32_t w = tid + en_cluster() * nthreads; w < EN_S * EN_D / 2u; w += nthreads * EN_CLUSTERS) {
    uint32_t v = 0;
    for (uint32_t j = 0; j < 2u; j++) {
      uint32_t h = (2u * w + j) * 8u + EN_ST_X;
      h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
      v |= (((h >> 31) << 15) | ((124u + ((h >> 8) & 3u)) << 7) | (h & 0x7Fu)) << (16u * j);
    }
    xw[w] = v;
  }
}
__attribute__((noinline)) static void st_clear(uint32_t tid, uint32_t nthreads) {
  const volatile ENArgs *a = reinterpret_cast<const volatile ENArgs *>(EN_ARGS);
  const uint32_t lo = a->clear_lo, hi = a->clear_hi ? a->clear_hi : 0x20000u;
  for (uint32_t w = lo / 4u + tid; w < hi / 4u; w += nthreads) smem<uint32_t>(0)[w] = 0;
  mu_fence_smem();
}
__attribute__((noinline)) static void st_gen(uint32_t tid, uint32_t nthreads) {   // metasim stage tests
  const volatile ENArgs *a = reinterpret_cast<const volatile ENArgs *>(EN_ARGS);
  uint32_t *dst = reinterpret_cast<uint32_t *>(a->gen_addr);
  const uint32_t words = a->gen_elems / 2u, stream = a->gen_stream;
  for (uint32_t w = tid + en_cluster() * nthreads; w < words; w += nthreads * EN_CLUSTERS) {
    uint32_t v = 0;
    for (uint32_t j = 0; j < 2u; j++) {
      uint32_t h = (2u * w + j) * 8u + stream;
      h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
      v |= (((h >> 31) << 15) | ((124u + ((h >> 8) % 5u)) << 7) | (h & 0x7Fu)) << (16u * j);
    }
    dst[w] = v;
  }
}
// metasim: MX weight codes (normal e4m3, |x| < 2, as host write_weight) at gen_addr (gen_elems
// bytes) and scales 0x79..0x7B at gen_saddr (gen_elems / 32 bytes); no host writes needed.
__attribute__((noinline)) static void st_genw(uint32_t tid, uint32_t nthreads) {
  const volatile ENArgs *a = reinterpret_cast<const volatile ENArgs *>(EN_ARGS);
  uint32_t *dst = reinterpret_cast<uint32_t *>(a->gen_addr);
  uint32_t *sdst = reinterpret_cast<uint32_t *>(a->gen_saddr);
  const uint32_t words = a->gen_elems / 4u, stream = a->gen_stream;
  for (uint32_t w = tid + en_cluster() * nthreads; w < words; w += nthreads * EN_CLUSTERS) {
    uint32_t v = 0;
    for (uint32_t j = 0; j < 4u; j++) {
      uint32_t h = (4u * w + j) * 8u + stream;
      h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
      v |= ((((h >> 6) & 1u) << 7) | ((1u + ((h >> 3) & 0xFFu) % 7u) << 3) | (h & 7u)) << (8u * j);
    }
    dst[w] = v;
  }
  for (uint32_t w = tid + en_cluster() * nthreads; w < words / 32u; w += nthreads * EN_CLUSTERS) {
    uint32_t v = 0;
    for (uint32_t j = 0; j < 4u; j++) {
      uint32_t h = (4u * w + j) * 8u + stream + 1u;
      h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
      v |= (0x79u + h % 3u) << (8u * j);
    }
    sdst[w] = v;
  }
}
// EN_STAGE_DMAPROBE: the same 16-row x 128-byte DRAM tile moved into SMEM with 4-block commands
// (FA_MVIN_WIDE's fa_mvin_A pattern) and with 1-block commands, both moved back out unchanged.
// Expected SMEM layout for both: block k (bytes 16k..16k+15 of every row) at SMEM rows 16k..16k+15.
__attribute__((noinline)) static void dma_probe() {
  constexpr uint32_t ROW_W = 4096u, ROW_N = 4352u, ROW_2 = 4608u;   // SMEM 0x10000, 0x11000, 0x12000
  dma_setup();
  gemmini_config_ld(128u);   // as fa_mvin_A: stride 128 B, block stride DIM
  for (uint32_t k = 0; k < 8u; k += 4u)
    gemmini_extended_mvin(rad_device_to_host_address(EN_PROBE_SRC + 16u * k), ROW_W + 16u * k, 4u * DIM, DIM);
  for (uint32_t k = 0; k < 8u; k += 2u)
    gemmini_extended_mvin(rad_device_to_host_address(EN_PROBE_SRC + 16u * k), ROW_2 + 16u * k, 2u * DIM, DIM);
  for (uint32_t k = 0; k < 8u; k++)
    gemmini_extended_mvin(rad_device_to_host_address(EN_PROBE_SRC + 16u * k), ROW_N + 16u * k, DIM, DIM);
  gemmini_fence();
  dma_cfg_out(16u);
  dma_out_contig(EN_PROBE_W, ROW_W * 16u, 2048u);
  dma_out_contig(EN_PROBE_N, ROW_N * 16u, 2048u);
  dma_out_contig(EN_PROBE_2, ROW_2 * 16u, 2048u);
  gemmini_fence();
  // timing: 64 KB DRAM -> SMEM as 256 1-block (256 B) vs 128 2-block (512 B) commands, and 64 KB
  // SMEM -> DRAM as 256 move-outs; the SMEM image of the 2-block copy is chunk-permuted (unused)
  volatile uint32_t *tp = reinterpret_cast<volatile uint32_t *>(EN_PROBE_T + 16u * en_cluster());
  mxg::drain();
  const uint32_t t0 = mcycle();
  gemmini_config_ld(16u);
  for (uint32_t k = 0; k < 256u; k++)
    gemmini_extended_mvin(rad_device_to_host_address(EN_FA_V + 256u * k), 16u * k, DIM, 16u);
  mxg::drain();
  const uint32_t t1 = mcycle();
  gemmini_config_ld(32u);
  for (uint32_t k = 0; k < 128u; k++)
    gemmini_extended_mvin(rad_device_to_host_address(EN_FA_V + 512u * k), 32u * k, 2u * DIM, 16u);
  mxg::drain();
  const uint32_t t2 = mcycle();
  dma_cfg_out(16u);
  dma_out_contig(EN_PROBE_T + 0x10000u, 0u, 65536u);
  mxg::drain();
  const uint32_t t3 = mcycle();
  // 2-block move-outs: layout (the narrow SMEM image at ROW_N back to a 16 x 128 B tile, expected
  // equal to the source) and 64 KB timing
  gemmini_config_st(128u);
  for (uint32_t k = 0; k < 8u; k += 2u)
    gemmini_extended_mvout(rad_device_to_host_address(EN_PROBE_M + 16u * k), ROW_N + 16u * k, 2u * DIM, 16u);
  mxg::drain();
  const uint32_t t4 = mcycle();
  gemmini_config_st(32u);
  for (uint32_t k = 0; k < 128u; k++)
    gemmini_extended_mvout(rad_device_to_host_address(EN_PROBE_T + 0x10000u + 512u * k), 32u * k, 2u * DIM, 16u);
  mxg::drain();
  const uint32_t t5 = mcycle();
  dma_cfg_out(16u);
  tp[0] = t1 - t0; tp[1] = t2 - t1; tp[2] = t3 - t2; tp[3] = t5 - t4;
}
__attribute__((noinline)) static void st_dmaprobe(uint32_t tid, uint32_t) {
  if (is_lead(tid)) dma_probe();
  else asm volatile("nop");
}
// EN_STAGE_UPROBE: every warp runs each loop at once (barriers between tests); warp 0 of core 0
// records cycles.  Each loop is 256 iterations x 8 independent instructions = 2048 per warp.
#define UP_FMA_S(a) asm volatile("fmadd.s %0, %0, %1, %2" : "+r"(a) : "r"(m), "r"(c))
#define UP_FMA_H(a) asm volatile("fmadd.h %0, %0, %1, %2" : "+r"(a) : "r"(m), "r"(c))
#define UP_XOR(a)   asm volatile("xor %0, %0, %1" : "+r"(a) : "r"(m))
__attribute__((noinline)) static void st_uprobe(uint32_t tid, uint32_t n) {
  const Lane L = lane_of(tid, n);
  const uint32_t nw = total_warps(L);
  const uint32_t m = 0x3F800000u + tid, c = 0x3F000000u;   // fp32 ~1.0, 0.5 (bf16 pairs for .h)
  uint32_t a0 = tid, a1 = tid + 1, a2 = tid + 2, a3 = tid + 3, a4 = tid + 4, a5 = tid + 5, a6 = tid + 6, a7 = tid + 7;
  uint32_t t[13];
  mu_barrier(6, nw); t[0] = mcycle();
  for (uint32_t i = 0; i < 256u; i++) { UP_FMA_S(a0); UP_FMA_S(a1); UP_FMA_S(a2); UP_FMA_S(a3); UP_FMA_S(a4); UP_FMA_S(a5); UP_FMA_S(a6); UP_FMA_S(a7); }
  mu_barrier(6, nw); t[1] = mcycle();
  for (uint32_t i = 0; i < 256u; i++) { UP_FMA_H(a0); UP_FMA_H(a1); UP_FMA_H(a2); UP_FMA_H(a3); UP_FMA_H(a4); UP_FMA_H(a5); UP_FMA_H(a6); UP_FMA_H(a7); }
  mu_barrier(6, nw); t[2] = mcycle();
  for (uint32_t i = 0; i < 256u; i++) { UP_XOR(a0); UP_XOR(a1); UP_XOR(a2); UP_XOR(a3); UP_XOR(a4); UP_XOR(a5); UP_XOR(a6); UP_XOR(a7); }
  mu_barrier(6, nw); t[3] = mcycle();
  {
    const volatile __shared uint32_t *s = smem<uint32_t>(0x8000u + 1024u * warp_slot(L)) + L.lane;
    uint32_t acc = 0;
    for (uint32_t i = 0; i < 256u; i++) {
      const uint32_t v0 = s[0], v1 = s[16], v2 = s[32], v3 = s[48], v4 = s[64], v5 = s[80], v6 = s[96], v7 = s[112];
      acc += v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7;
    }
    a0 += acc;
  }
  mu_barrier(6, nw); t[4] = mcycle();
  for (uint32_t i = 0; i < 2048u; i++) UP_FMA_S(a1);   // one dependent chain
  mu_barrier(6, nw); t[5] = mcycle();
  for (uint32_t i = 0; i < 2048u; i++) UP_FMA_H(a2);   // dependent bf16 chain
  mu_barrier(6, nw); t[6] = mcycle();
  {   // dependent SMEM load chain: each lane chases a cycle of 64 words in its warp's slot
    volatile __shared uint32_t *s = smem<uint32_t>(0x8000u + 1024u * warp_slot(L));
    if (L.lane == 0u) for (uint32_t j = 0; j < 64u; j++) s[16u * j] = 16u * ((j + 1u) & 63u);
    mu_fence_smem();
    uint32_t p = L.lane;
    for (uint32_t i = 0; i < 512u; i++) p = s[p & ~15u] + L.lane;
    a3 += p;
  }
  mu_barrier(6, nw); t[7] = mcycle();
  {   // dependent global load chain (lines this warp owns, written first)
    volatile uint32_t *g = reinterpret_cast<volatile uint32_t *>(EN_PROBE_SRC + 0x20000u + 4096u * warp_slot(L));
    if (L.lane == 0u) for (uint32_t j = 0; j < 64u; j++) g[16u * j] = 16u * ((j + 1u) & 63u);
    uint32_t p = L.lane;
    for (uint32_t i = 0; i < 512u; i++) p = g[p & ~15u] + L.lane;
    a4 += p;
  }
  mu_barrier(6, nw); t[8] = mcycle();
  {   // fnexp.h / fdiv.h: 256 x 8 independent, then a dependent chain of 512 (bf16 operands near 1)
    uint32_t e0 = 0x3F80u + (tid & 7u), e1 = e0 + 1u, e2 = e0 + 2u, e3 = e0 + 3u, e4 = e0 + 4u, e5 = e0 + 5u, e6 = e0 + 6u, e7 = e0 + 7u;
#define UP_EXP(a) asm volatile("fnexp.h %0, %0" : "+r"(a))
#define UP_DIV(a) asm volatile("fdiv.h %0, %0, %1" : "+r"(a) : "r"(0x3F81u))
    for (uint32_t i = 0; i < 256u; i++) { UP_EXP(e0); UP_EXP(e1); UP_EXP(e2); UP_EXP(e3); UP_EXP(e4); UP_EXP(e5); UP_EXP(e6); UP_EXP(e7); }
    mu_barrier(6, nw); t[9] = mcycle();
    for (uint32_t i = 0; i < 256u; i++) { UP_DIV(e0); UP_DIV(e1); UP_DIV(e2); UP_DIV(e3); UP_DIV(e4); UP_DIV(e5); UP_DIV(e6); UP_DIV(e7); }
    mu_barrier(6, nw); t[10] = mcycle();
    for (uint32_t i = 0; i < 512u; i++) UP_EXP(e0);
    mu_barrier(6, nw); t[11] = mcycle();
    for (uint32_t i = 0; i < 512u; i++) UP_DIV(e1);
    mu_barrier(6, nw); t[12] = mcycle();
    a5 += e0 ^ e1 ^ e2 ^ e3 ^ e4 ^ e5 ^ e6 ^ e7;
  }
  if (is_lead(tid)) {
    for (uint32_t j = 0; j < 12u; j++) rad_pb_put(EN_PB_UPROBE + j, t[j + 1] - t[j]);
    rad_pb_put(EN_PB_UPROBE + 12u, a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7);   // keep the results live
  } else {
    asm volatile("nop");
  }
}
// EN_STAGE_FILL / EN_STAGE_HASH: whole-buffer coverage checks.  Every warp of both clusters takes
// 64-byte lines L = wg, wg + nwg, ...; lane l handles word 16 L + l, so every line is written or
// read whole by one warp (no shared-line stores).  The launch epilogue drains L0d / L1, so the fill
// reaches DRAM and the hash launch starts with no stale lines cached.
__attribute__((noinline)) static void st_fill(uint32_t tid, uint32_t n) {
  const Lane L = lane_of(tid, n);
  const volatile ENArgs *a = reinterpret_cast<const volatile ENArgs *>(EN_ARGS);
  volatile uint32_t *p = reinterpret_cast<volatile uint32_t *>(a->gen_addr);
  const uint32_t lines = a->gen_elems / 64u, seed = a->gen_stream;
  const uint32_t nwg = total_warps(L) * EN_CLUSTERS, wg = en_cluster() * total_warps(L) + warp_slot(L);
  for (uint32_t ln = wg; ln < lines; ln += nwg) {
    const uint32_t i = 16u * ln + L.lane;
    p[i] = (seed ^ (i * 0x9E3779B9u)) | 0x7F807F80u;   // bf16 pairs with exponent 0xFF
  }
}
__attribute__((noinline)) static void st_hash(uint32_t tid, uint32_t n) {
  const Lane L = lane_of(tid, n);
  const volatile ENArgs *a = reinterpret_cast<const volatile ENArgs *>(EN_ARGS);
  const volatile uint32_t *p = reinterpret_cast<const volatile uint32_t *>(a->gen_addr);
  const uint32_t lines = a->gen_elems / 64u;
  const uint32_t nwg = total_warps(L) * EN_CLUSTERS, wg = en_cluster() * total_warps(L) + warp_slot(L);
  uint32_t h = 0;
  for (uint32_t ln = wg; ln < lines; ln += nwg) {
    const uint32_t i = 16u * ln + L.lane;
    h += ((p[i] ^ (i * 0x9E3779B9u)) * 0x85EBCA6Bu) ^ (i >> 3);
  }
  reinterpret_cast<volatile uint32_t *>(EN_HASH_OUT)[16u * wg + L.lane] = h;
}
#ifndef EN_LN_TU   // ln_stage.cpp needs only the LN stage functions
static StageFn const kStages[] = {
  st_none,   st_lnq1,  st_qkv,   st_split,  st_none /* FA: own image entry */, st_oq,  st_oproj,
  st_res1ln, st_up,    st_geluq, st_down,   st_res2,  st_clear, st_genqkv, st_genx,  st_gen,
  st_upgelu, st_qkvsplit, st_genw, st_downres, st_oprojres, st_lnq2, st_dmaprobe, st_uprobe,
  st_fill, st_hash,
};
static_assert(EN_STAGE_HASH == 25 && sizeof(kStages) / sizeof(kStages[0]) == 26, "kStages follows ENStage");

// Cluster id (CSR 0xCD0).  On the 2-cluster tapeout system (FireAxe pair) the host's soft reset
// releases every core of both clusters and core ids are cluster-local, so a one-cluster image runs
// on both; cluster 1 skips the stage (and FA) but still takes the barrier and the epilogue, so its
// cores park normally.  The host polls cluster 0 only.
static inline uint32_t cluster_id() {
  uint32_t v;
  asm volatile("csrr %0, %1" : "=r"(v) : "i"(MU_CSR_CLUSTER_ID));
  return v;
}
static void encoder_entry(void *raw_args, uint32_t tid, uint32_t nthreads, uint32_t) {
  const uint32_t stage = reinterpret_cast<const volatile ENArgs *>(raw_args)->stage;
  if (is_lead(tid)) rad_pb_put(EN_PB_T_START, mcycle());
  else asm volatile("nop");
  const uint32_t n = sizeof(kStages) / sizeof(kStages[0]);
  const uint32_t st = sel(mask_lt(stage, n), 0u, stage) & (cluster_id() < EN_CLUSTERS ? ~0u : 0u);   // idle clusters -> st_none
#ifdef EN_STAGE_HPM   // perf counters of each core around the whole stage (first warp's window)
  fx_hpm_snap(tid, 0u);
#endif
  kStages[st](tid, nthreads);   // out of range -> st_none
#ifdef EN_STAGE_HPM
  fx_hpm_snap(tid, 1u);
#endif
  mu_barrier(1, nthreads / MU_NUM_THREADS);
  if (is_lead(tid)) rad_pb_put(EN_PB_T_END, mcycle());
  else asm volatile("nop");
  rad_tapeout_epilogue(tid, nthreads / MU_NUM_THREADS);
}

extern int fa_image_main();   // fa_stage.cpp: the FA kernel's own launch (mu_schedule(fa_entry))
#ifdef EN_STAGE_HPM
// the attention launch (fa_image_main, not encoder_entry) snapshots the counters through this
void en_hpm_snap(uint32_t tid, uint32_t idx) { fx_hpm_snap(tid, idx); }
#endif

int main() {
  rad_tapeout_begin();
  if (reinterpret_cast<const volatile ENArgs *>(EN_ARGS)->stage == EN_STAGE_FA && cluster_id() < EN_CLUSTERS)
    return fa_image_main();
  const uint32_t st = reinterpret_cast<const volatile ENArgs *>(EN_ARGS)->stage;
  const bool ln = st == EN_STAGE_LNQ1 || st == EN_STAGE_RES1LN || st == EN_STAGE_LNQ2;
  mu_schedule(encoder_entry, reinterpret_cast<void *>(EN_ARGS), ln ? EN_LN_OCC : EN_OCC);
  return 0;
}
#endif  // EN_LN_TU
