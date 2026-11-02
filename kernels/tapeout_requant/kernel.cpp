// kernel.cpp: probe of the tapeout MX requantizer's GPU-input path (GEMMINI_REQUANT window).
//
// The path takes one 32-byte beat per full-warp halfword store: lane i stores bf16 element i of
// the beat to window + 2 * (element index).  The 16 lanes are merged by lane index, not by
// address, and two consecutive beats form one 32-element MX block.  The e4m3 codes land in SMEM
// at (window offset) / 2.  Only one warp may feed the window, and no other SMEM traffic may run
// meanwhile (collector and D-channel defects of the tapeout RTL).
//
// Core 0 warp 0 does everything: generate RQ_NB blocks of bf16 in SMEM, feed them, poll for the
// output, then compare the codes with an integer model of the hardware (block scale from the
// max's exponent minus 8, i.e. block max in [256, 448]) and with the SIMT quantizer of the FA
// kernel after an exponent shift by -8 (see fix_code).

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <rad_tapeout.h>
#include <radiance.h>
#include <stdint.h>

#include "include/gemmini.h"
#include <mxgemmini_mmio.h>

#include "rq.h"

static inline uint32_t mcycle() {
  uint32_t c; asm volatile("csrr %0, mcycle" : "=r"(c)); return c;
}
static inline uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}
static inline float u2f(uint32_t u) { return __builtin_bit_cast(float, u); }
static inline uint32_t f2u(float f) { return __builtin_bit_cast(uint32_t, f); }

// bf16 test input: block magnitude 2^(blk % 24 - 12), elements spread over 10 binades below it,
// random sign, a few zeros.
static inline uint32_t src_value(uint32_t e) {
  const uint32_t h = hash32(e * 4u + 7u);
  const uint32_t exp = 115u + (e / 32u) % 24u - (h >> 8) % 10u;
  if ((h >> 4) % 64u == 0) return 0;
  return ((h >> 31) << 15) | (exp << 7) | (h & 0x7Fu);
}

// e4m3 code of |y|: round to nearest even or truncate, subnormals kept, saturation at 448 (0x7E).
static inline uint32_t e4m3_mag(float y, bool trunc) {
  const uint32_t b = f2u(y) & 0x7FFFFFFFu;
  if (b == 0) return 0;
  int e = (int)(b >> 23) - 127;
  const uint32_t m = (b & 0x7FFFFFu) | 0x800000u;
  const bool normal = e >= -6;
  uint32_t shift = normal ? 20u : (uint32_t)(14 - e);
  if (shift > 25u) shift = 25u;
  uint32_t q = m >> shift;
  if (!trunc) {
    const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (q & 1u))) q++;
  }
  if (!normal) return q >= 8u ? 0x08u : q;
  if (q == 16u) { q = 8u; e++; }
  if (e > 8 || (e == 8 && (q & 7u) > 6u)) return 0x7Eu;
  return ((uint32_t)(e + 7) << 3) | (q & 7u);
}
static inline uint32_t e4m3_code(float y, bool trunc) {
  const uint32_t m = e4m3_mag(y, trunc);
  return m ? (m | ((f2u(y) >> 24) & 0x80u)) : 0u;
}
// The SIMT quantizer's rule: subnormals flushed to zero.
static inline uint32_t e4m3_ftz(float y) {
  const uint32_t m = e4m3_mag(y, false);
  return m >= 8u ? (m | ((f2u(y) >> 24) & 0x80u)) : 0u;
}
// Hardware code (block max in [256, 448]) -> code for block max in [1, 2): exponent field - 8.
// Fields 9..15 drop bit 6 (no borrow); fields 0..8 would become zero or subnormal: flush.
static inline uint32_t fix_code(uint32_t c) { return ((c >> 3) & 15u) >= 9u ? (c & 0xBFu) : 0u; }

static inline float bf16f(uint32_t h) { return u2f(h << 16); }

static inline void red_sink(uint32_t v) {
  reinterpret_cast<volatile __shared uint32_t *>(RQ_RED)[MU_NUM_THREADS * 8u] = v;
}

static void requant_probe(uint32_t lane, uint32_t mode) {
  volatile __shared uint32_t *src32 = reinterpret_cast<volatile __shared uint32_t *>(RQ_SRC);
  volatile __shared uint16_t *src16 = reinterpret_cast<volatile __shared uint16_t *>(RQ_SRC);
  volatile __shared uint32_t *dst32 = reinterpret_cast<volatile __shared uint32_t *>(RQ_DST);
  volatile __shared uint8_t *dst8 = reinterpret_cast<volatile __shared uint8_t *>(RQ_DST);
  volatile __shared uint16_t *win =
      reinterpret_cast<volatile __shared uint16_t *>(GEMMINI_REQUANT + 2u * RQ_DST);
  constexpr uint32_t NE = RQ_NB * 32u;

  for (uint32_t w = lane; w < NE / 2u; w += MU_NUM_THREADS)
    src32[w] = src_value(2u * w) | (src_value(2u * w + 1u) << 16);
  for (uint32_t w = lane; w < NE / 4u; w += MU_NUM_THREADS) dst32[w] = 0xFFFFFFFFu;
  if (lane == 0) {
    // The requantizer writes its (truncated) scales to this address; keep them out of the way.
    gemmini_mxquant_config_mvout(rad_device_to_host_address(RQ_SCRATCH), 1, 1, 1, 0, 0, 1);
    gemmini_fence();
  } else {
    asm volatile("nop");
  }
  reinterpret_cast<volatile __shared uint32_t *>(RQ_FLAG_SMEM)[0] = 0u;
  mu_fence_smem();

  // Feed: 16 beats per batch, loads first so no SMEM load overlaps the window stores.
  const uint32_t t0 = mcycle();
  if (mode == RQ_MODE_FEED || mode == RQ_MODE_CONTEND) {
    for (uint32_t base = 0; base < NE; base += 16u * MU_NUM_THREADS) {
      uint32_t v[16];
#pragma unroll
      for (uint32_t i = 0; i < 16u; i++) v[i] = src16[base + i * MU_NUM_THREADS + lane];
#pragma unroll
      for (uint32_t i = 0; i < 16u; i++) win[base + i * MU_NUM_THREADS + lane] = (uint16_t)v[i];
    }
  } else if (mode == RQ_MODE_FENCE) {
    for (uint32_t base = 0; base < NE; base += 16u * MU_NUM_THREADS) {
      uint32_t v[16];
#pragma unroll
      for (uint32_t i = 0; i < 16u; i++) v[i] = src16[base + i * MU_NUM_THREADS + lane];
      mu_fence_smem();
#pragma unroll
      for (uint32_t i = 0; i < 16u; i++) win[base + i * MU_NUM_THREADS + lane] = (uint16_t)v[i];
    }
  } else if (mode == RQ_MODE_FEEDW || mode == RQ_MODE_CSAME || mode == RQ_MODE_COTHER) {
    for (uint32_t base = 0; base < NE / 2u; base += 8u * MU_NUM_THREADS) {   // 8 blocks per batch
      uint32_t v[8];
#pragma unroll
      for (uint32_t b = 0; b < 8u; b++) v[b] = src32[base + b * MU_NUM_THREADS + lane];
      mu_fence_smem();
#pragma unroll
      for (uint32_t b = 0; b < 8u; b++) {
        win[2u * (base + b * MU_NUM_THREADS) + lane] = (uint16_t)v[b];
        win[2u * (base + b * MU_NUM_THREADS) + MU_NUM_THREADS + lane] = (uint16_t)(v[b] >> 16);
      }
    }
  } else if (mode == RQ_MODE_GEN) {
    for (uint32_t base = 0; base < NE; base += 16u * MU_NUM_THREADS) {
      uint32_t v[16];
#pragma unroll
      for (uint32_t i = 0; i < 16u; i++) v[i] = src_value(base + i * MU_NUM_THREADS + lane);
      asm volatile("" ::: "memory");
#pragma unroll
      for (uint32_t i = 0; i < 16u; i++) win[base + i * MU_NUM_THREADS + lane] = (uint16_t)v[i];
    }
  } else if (mode == RQ_MODE_STORES) {     // timing only: data from a register
    const uint16_t v = (uint16_t)(0x3F80u + lane);
    for (uint32_t base = 0; base < NE; base += 16u * MU_NUM_THREADS) {
#pragma unroll
      for (uint32_t i = 0; i < 16u; i++) win[base + i * MU_NUM_THREADS + lane] = v;
    }
  } else {                                 // RQ_MODE_LOADS, timing only
    uint32_t acc = 0;
    for (uint32_t base = 0; base < NE; base += 16u * MU_NUM_THREADS) {
#pragma unroll
      for (uint32_t i = 0; i < 16u; i++) acc += src16[base + i * MU_NUM_THREADS + lane];
    }
    red_sink(acc);
  }
  mu_fence_smem();
  const uint32_t t1 = mcycle();
  uint32_t polls = 0;
  while (dst32[NE / 4u - 1u] == 0xFFFFFFFFu && polls < 200000u) polls++;
  const uint32_t t2 = mcycle();
  reinterpret_cast<volatile __shared uint32_t *>(RQ_FLAG_SMEM)[0] = 1u;

  // Check: lane handles blocks lane, lane + 16, ...
  uint32_t raw_rne = 0, raw_trunc = 0, raw_rev = 0, fix_ref = 0, fix_far = 0;
  uint32_t first_bad = 0xFFFFFFFFu, first_info = 0;
  for (uint32_t blk = lane; blk < RQ_NB; blk += MU_NUM_THREADS) {
    uint32_t amax = 0;
    for (uint32_t i = 0; i < 32u; i++) {
      const uint32_t a = src16[blk * 32u + i] & 0x7FFFu;
      if (a > amax) amax = a;
    }
    const uint32_t em = amax >> 7;                       // bf16 exponent field of the block max
    const float hw_mul = u2f((262u - em) << 23);          // 2^(135 - em): block max -> [256, 512)
    const float ref_mul = u2f((254u - em) << 23);         // 2^(127 - em): block max -> [1, 2)
    const bool evenodd = mode == RQ_MODE_FEEDW || mode == RQ_MODE_CSAME || mode == RQ_MODE_COTHER;
    for (uint32_t i = 0; i < 32u; i++) {
      // output byte i holds beat element c = 31 - i (the requantizer reverses the 32 lanes);
      // beat element c is input element c, or 2c / 2(c - 16) + 1 for the even-then-odd feed
      const uint32_t c = 31u - i;
      const uint32_t e = evenodd ? (c < 16u ? 2u * c : 2u * (c - 16u) + 1u) : c;
      const uint32_t xb = src16[blk * 32u + e];
      const float x = bf16f(xb);
      const uint32_t got = dst8[blk * 32u + i];
      const uint32_t got_rev = dst8[blk * 32u + c];   // natural-order reading, for comparison
      const uint32_t m_rne = e4m3_code(x * hw_mul, false);
      raw_rne += got != m_rne;
      raw_trunc += got != e4m3_code(x * hw_mul, true);
      raw_rev += got_rev != m_rne;
      const uint32_t fixed = fix_code(got), ref = e4m3_ftz(x * ref_mul);
      fix_ref += fixed != ref;
      const int d = (int)(fixed & 0x7Fu) - (int)(ref & 0x7Fu);
      fix_far += (fixed ^ ref) & 0x80u ? (ref & 0x7Fu) != 0 : (d > 1 || d < -1);
      if (got != m_rne && blk * 32u + i < first_bad) {
        first_bad = blk * 32u + i;
        first_info = got | (m_rne << 8) | (xb << 16);
      }
    }
  }
  // Copy SRC and DST to GMEM for the host (the epilogue drains L0d).
  volatile uint32_t *dump = reinterpret_cast<volatile uint32_t *>(RQ_DUMP);
  for (uint32_t w = lane; w < NE / 2u; w += MU_NUM_THREADS) dump[w] = src32[w];
  for (uint32_t w = lane; w < NE / 4u; w += MU_NUM_THREADS) dump[NE / 2u + w] = dst32[w];
  volatile __shared uint32_t *red = reinterpret_cast<volatile __shared uint32_t *>(RQ_RED);
  red[lane * 8u + 0] = raw_rne;
  red[lane * 8u + 1] = raw_trunc;
  red[lane * 8u + 2] = raw_rev;
  red[lane * 8u + 3] = fix_ref;
  red[lane * 8u + 4] = fix_far;
  red[lane * 8u + 5] = first_bad;
  red[lane * 8u + 6] = first_info;
  mu_fence_smem();
  if (lane == 0) {
    uint32_t s[5] = {0, 0, 0, 0, 0}, fb = 0xFFFFFFFFu, fi = 0;
    for (uint32_t l = 0; l < MU_NUM_THREADS; l++) {
      for (uint32_t k = 0; k < 5u; k++) s[k] += red[l * 8u + k];
      if (red[l * 8u + 5] < fb) { fb = red[l * 8u + 5]; fi = red[l * 8u + 6]; }
    }
    rad_pb_put(RQ_PB_STATUS, polls < 200000u ? 1u : 2u);
    rad_pb_put(RQ_PB_T_FEED, t1 - t0);
    rad_pb_put(RQ_PB_T_DONE, t2 - t0);
    rad_pb_put(RQ_PB_POLLS, polls);
    for (uint32_t k = 0; k < 5u; k++) rad_pb_put(RQ_PB_RAW_RNE + k, s[k]);
    rad_pb_put(RQ_PB_FIRST_BAD, fb);
    rad_pb_put(RQ_PB_FIRST_GOT, fi);
    for (uint32_t w = 0; w < 8u; w++) {
      rad_pb_put(RQ_PB_DUMP + w, dst32[w]);
      uint32_t mw = 0;
      for (uint32_t j = 0; j < 4u; j++) {
        uint32_t amax = 0;
        for (uint32_t i = 0; i < 32u; i++) {
          const uint32_t a = src16[i] & 0x7FFFu;
          if (a > amax) amax = a;
        }
        mw |= e4m3_code(bf16f(src16[4u * w + j]) * u2f((262u - (amax >> 7)) << 23), false) << (8u * j);
      }
      rad_pb_put(RQ_PB_DUMP + 8u + w, mw);
    }
  } else {
    asm volatile("nop");
  }
}

// CONTEND mode, core 1's warps: SMEM loads and stores (with bank conflicts) until warp 0 has
// seen its output.  Warp 0 sets the SMEM flag to 0 before feeding and to 1 after; a worker stops
// only on a 1 that follows a 0 it saw (SMEM may hold a 1 from the previous run).
static void contend(uint32_t lane) {
  volatile __shared uint32_t *flag = reinterpret_cast<volatile __shared uint32_t *>(RQ_FLAG_SMEM);
  volatile __shared uint32_t *buf = reinterpret_cast<volatile __shared uint32_t *>(0x18000u);
  uint32_t acc = lane, it = 0, seen0 = 0;
  for (;;) {
    const uint32_t f = flag[0];
    if ((seen0 && f == 1u) || it >= 400000u) break;
    seen0 |= (uint32_t)(f == 0u);
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) {
      acc += buf[(lane * 16u + i * 64u + it) & 0xFFFu];   // 16-word lane stride: subbank conflicts
      buf[(lane * 17u + i * 33u + it) & 0xFFFu] = acc;
    }
    it++;
  }
}

static inline uint32_t dma_word(uint32_t i) { return hash32(i * 4u + 11u); }

// DMA probe (core 0 warp 0): host-written bytes DRAM -> SMEM 0 (mvin) -> DRAM (mvout from the
// scratchpad); SIMT compares both copies word for word.  Scratchpad row r is SMEM byte 16 r.
static void dma_probe(uint32_t lane, uint32_t rows) {
  uint32_t t0 = 0, t1 = 0, t2 = 0;
  if (lane == 0) {
    gemmini_flush(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                                0, 0, 3, false);
    gemmini_extended3_config_ld(DIM, MVIN_SCALE_IDENTITY, false, 0);
    gemmini_config_st(DIM);
    gemmini_fence();
    t0 = mcycle();
    for (uint32_t off = 0; off < RQ_DMA_BYTES; off += rows * DIM)
      gemmini_extended_mvin(rad_device_to_host_address(RQ_DMA_SRC + off), off / DIM, DIM, rows);
    gemmini_fence_waitcount(0);
    gemmini_fence();
    t1 = mcycle();
  } else {
    asm volatile("nop");
  }
  mu_fence_smem();
  const volatile __shared uint32_t *sm = reinterpret_cast<const volatile __shared uint32_t *>(0u);
  uint32_t bad_in = 0;
  for (uint32_t w = lane; w < RQ_DMA_BYTES / 4u; w += MU_NUM_THREADS) bad_in += sm[w] != dma_word(w);
  mu_fence_smem();
  if (lane == 0) {
    const uint32_t ts = mcycle();
    for (uint32_t off = 0; off < RQ_DMA_BYTES; off += rows * DIM)
      gemmini_extended_mvout(rad_device_to_host_address(RQ_DMA_DST + off), off / DIM, DIM, rows);
    gemmini_fence_waitcount(0);
    gemmini_fence();
    t2 = mcycle() - ts;
  } else {
    asm volatile("nop");
  }
  const volatile uint32_t *dd = reinterpret_cast<const volatile uint32_t *>(RQ_DMA_DST);
  uint32_t bad_out = 0;
  for (uint32_t w = lane; w < RQ_DMA_BYTES / 4u; w += MU_NUM_THREADS) bad_out += dd[w] != dma_word(w);
  volatile __shared uint32_t *red = reinterpret_cast<volatile __shared uint32_t *>(RQ_RED);
  red[lane] = bad_in;
  red[16u + lane] = bad_out;
  mu_fence_smem();
  if (lane == 0) {
    uint32_t a = 0, b = 0;
    for (uint32_t l = 0; l < MU_NUM_THREADS; l++) { a += red[l]; b += red[16u + l]; }
    rad_pb_put(RQ_PB_STATUS, 1u);
    rad_pb_put(RQ_PB_T_FEED, t1 - t0);   // mvin time
    rad_pb_put(RQ_PB_T_DONE, t2);        // mvout time
    rad_pb_put(RQ_PB_RAW_RNE, a);        // SMEM words != source
    rad_pb_put(RQ_PB_RAW_TRUNC, b);      // DRAM copy words != source
  } else {
    asm volatile("nop");
  }
}

// Issue-rate probe (warp 0 only): 8 independent chains x 64 iterations of one op type; reports
// cycles per warp instruction x 100 for each type in RQ_PB_DUMP + k.
#define RQ_OP8(op)                                                                                \
  asm volatile(op " %0, %0, %8\n\t" op " %1, %1, %8\n\t" op " %2, %2, %8\n\t" op " %3, %3, %8\n\t" \
               op " %4, %4, %8\n\t" op " %5, %5, %8\n\t" op " %6, %6, %8\n\t" op " %7, %7, %8"     \
               : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3), "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7)   \
               : "r"(k))
template <int OP>
__attribute__((noinline)) static uint32_t issue_rate(uint32_t seed) {
  uint32_t r0 = seed, r1 = seed + 1, r2 = seed + 2, r3 = seed + 3, r4 = seed + 4, r5 = seed + 5,
           r6 = seed + 6, r7 = seed + 7, k = 0x3F800001u;
  const uint32_t t0 = mcycle();
  for (uint32_t it = 0; it < 64u; it++) {
    if (OP == 0) RQ_OP8("add");
    if (OP == 1) RQ_OP8("fadd.s");
    if (OP == 2) RQ_OP8("fmul.s");
    if (OP == 3) RQ_OP8("fmul.h");
    if (OP == 4) RQ_OP8("fsgnjx.s");
    if (OP == 5) RQ_OP8("fmax.s");
    if (OP == 6) RQ_OP8("fdiv.s");
    if (OP == 7) RQ_OP8("fdiv.h");
    if (OP == 8) {
      asm volatile("fexp.h %0, %0\n\tfexp.h %1, %1\n\tfexp.h %2, %2\n\tfexp.h %3, %3\n\t"
                   "fexp.h %4, %4\n\tfexp.h %5, %5\n\tfexp.h %6, %6\n\tfexp.h %7, %7"
                   : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3), "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7));
    }
    if (OP == 9) {
      asm volatile("fmadd.s %0, %0, %8, %8\n\tfmadd.s %1, %1, %8, %8\n\tfmadd.s %2, %2, %8, %8\n\tfmadd.s %3, %3, %8, %8\n\t"
                   "fmadd.s %4, %4, %8, %8\n\tfmadd.s %5, %5, %8, %8\n\tfmadd.s %6, %6, %8, %8\n\tfmadd.s %7, %7, %8, %8"
                   : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3), "+r"(r4), "+r"(r5), "+r"(r6), "+r"(r7) : "r"(k));
    }
  }
  const uint32_t t1 = mcycle();
  reinterpret_cast<volatile __shared uint32_t *>(RQ_RED)[0] = r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
  return (t1 - t0) * 100u / 512u;
}
__attribute__((noinline)) static uint32_t issue_lds(uint32_t lane) {
  const volatile __shared uint32_t *p = reinterpret_cast<const volatile __shared uint32_t *>(0x8000u);
  uint32_t acc = 0;
  const uint32_t t0 = mcycle();
  for (uint32_t it = 0; it < 64u; it++) {
    uint32_t v[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) v[i] = p[(it * 8u + i) * 16u + lane];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) acc += v[i];
  }
  const uint32_t t1 = mcycle();
  reinterpret_cast<volatile __shared uint32_t *>(RQ_RED)[1] = acc;
  return (t1 - t0) * 100u / 512u;
}
static void issue_probe(uint32_t lane) {
  const uint32_t a = issue_rate<0>(lane), b = issue_rate<1>(lane), c = issue_rate<2>(lane),
                 d = issue_rate<3>(lane), e = issue_rate<4>(lane), f = issue_rate<5>(lane),
                 g = issue_lds(lane), h = issue_rate<6>(lane), i = issue_rate<7>(lane),
                 j = issue_rate<8>(lane), m = issue_rate<9>(lane);
  if (lane == 0) {
    rad_pb_put(RQ_PB_STATUS, 1u);
    rad_pb_put(RQ_PB_DUMP + 0u, a); rad_pb_put(RQ_PB_DUMP + 1u, b); rad_pb_put(RQ_PB_DUMP + 2u, c);
    rad_pb_put(RQ_PB_DUMP + 3u, d); rad_pb_put(RQ_PB_DUMP + 4u, e); rad_pb_put(RQ_PB_DUMP + 5u, f);
    rad_pb_put(RQ_PB_DUMP + 6u, g);
    rad_pb_put(RQ_PB_DUMP + 7u, h); rad_pb_put(RQ_PB_DUMP + 8u, i); rad_pb_put(RQ_PB_DUMP + 9u, j);
    rad_pb_put(RQ_PB_DUMP + 10u, m);
  } else {
    asm volatile("nop");
  }
}

// GELU throughput probe: every warp runs 64 x 8 independent GELU evaluations on registers.
static inline uint32_t mask_lt(uint32_t a, uint32_t b) {
  uint32_t m;
  asm("sltu %0, %1, %2\n\tneg %0, %0" : "=r"(m) : "r"(a), "r"(b));
  return m;
}
static inline uint32_t sel(uint32_t m, uint32_t x, uint32_t y) { return (x & ~m) | (y & m); }
static inline float fma_f(float a, float b, float c) {
  float r;
  asm("fmadd.s %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
template <bool UNITS>
static inline float gelu_probe_f(float x) {
  float u = x * fma_f(0.07135481627f, x * x, 1.5957691216057308f);
  const uint32_t ub = f2u(u);
  const uint32_t low = mask_lt(0xC2A00000u, ub);
  const uint32_t uh = (sel(low, ub, 0xC2A00000u) + 0x7FFFu + ((ub >> 16) & 1u)) >> 16;
  uint32_t eh = uh, rh;
  if (UNITS) asm volatile("fnexp.h %0, %1" : "=r"(eh) : "r"(uh));
  const float d = 1.0f + u2f(eh << 16);
  if (UNITS) asm volatile("fdiv.h %0, %1, %2" : "=r"(rh) : "r"(0x3F80u), "r"(f2u(d) >> 16));
  else rh = f2u(d * 0.5f) >> 16;
  const float r0 = u2f(rh << 16);
  float t;
  asm("fnmsub.s %0, %1, %2, %3" : "=r"(t) : "r"(d), "r"(r0), "r"(2.0f));
  return x * (r0 * t);
}
template <bool UNITS>
__attribute__((noinline)) static uint32_t gelu_rate(uint32_t lane) {
  float v[8];
#pragma unroll
  for (uint32_t i = 0; i < 8u; i++) v[i] = u2f(0x3F800000u + ((lane * 8u + i) << 16));
  const uint32_t t0 = mcycle();
  for (uint32_t it = 0; it < 64u; it++) {
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) v[i] = gelu_probe_f<UNITS>(v[i]) + 1.0f;
  }
  const uint32_t t1 = mcycle();
  uint32_t acc = 0;
#pragma unroll
  for (uint32_t i = 0; i < 8u; i++) acc ^= f2u(v[i]);
  reinterpret_cast<volatile __shared uint32_t *>(RQ_RED)[lane] = acc;
  return (t1 - t0) * 100u / 512u;   // cycles per element per warp x 100
}

static void rq_entry(void *raw_args, uint32_t tid, uint32_t nthreads, uint32_t) {
  const uint32_t warp = tid / MU_NUM_THREADS;   // warp w runs on core w % MU_NUM_CORES
  const uint32_t mode = reinterpret_cast<const volatile RQArgs *>(raw_args)->mode;
  if (((mode == RQ_MODE_CONTEND || mode == RQ_MODE_COTHER) && warp % MU_NUM_CORES == 1u) ||
      (mode == RQ_MODE_CSAME && warp % MU_NUM_CORES == 0u && warp != 0u))
    contend(tid % MU_NUM_THREADS);
  else asm volatile("nop");
  if (mode == RQ_MODE_GELU) {
    const uint32_t a = gelu_rate<true>(tid % MU_NUM_THREADS);
    const uint32_t b = gelu_rate<false>(tid % MU_NUM_THREADS);
    if (tid == 0) {
      rad_pb_put(RQ_PB_STATUS, 1u);
      rad_pb_put(RQ_PB_DUMP + 0u, a);
      rad_pb_put(RQ_PB_DUMP + 1u, b);
    } else {
      asm volatile("nop");
    }
  } else if (warp == 0 && mode == RQ_MODE_ISSUE)
    issue_probe(tid % MU_NUM_THREADS);
  else if (warp == 0 && (mode == RQ_MODE_DMA16 || mode == RQ_MODE_DMA64))
    dma_probe(tid % MU_NUM_THREADS, mode == RQ_MODE_DMA16 ? 16u : 64u);
  else if (warp == 0)
    requant_probe(tid % MU_NUM_THREADS, reinterpret_cast<const volatile RQArgs *>(raw_args)->mode);
  else asm volatile("nop");
  rad_tapeout_epilogue(tid, nthreads / MU_NUM_THREADS);
}

int main() {
  rad_tapeout_begin();
  mu_schedule(rq_entry, reinterpret_cast<void *>(RQ_ARGS), RQ_OCC);
  return 0;
}
