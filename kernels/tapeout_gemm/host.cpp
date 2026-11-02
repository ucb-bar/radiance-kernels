// host.cpp: Rocket side of the tapeout MX-FP8 GEMM.
//
// Generates the operands with the integer hash shared with gen_data.py, writes them to GPU DRAM,
// runs the kernel with rad_host_run(), and reports cycles and mesh utilization.  In TG_GEMM mode
// it also checks C against exact arithmetic (gemm_data.h), see the gate below.
//
// Build-time knobs: TG_MODE (TGMode, default TG_GEMM), TG_REPS (benchmark repetitions).
// The host has no FPU: integer arithmetic only.

#include <rad_host.h>
#include <stdint.h>

#include "gemm.h"
#include "gemm_data.h"

#ifndef TG_MODE
#define TG_MODE TG_GEMM
#endif
#ifndef TG_REPS
#define TG_REPS 8
#endif

static_assert(TG_M % TG_TM == 0 && TG_N % TG_TN == 0 && TG_K % TG_TK == 0, "shape");
static_assert(TG_M * TG_K <= TG_MAX_A_BYTES && TG_K * TG_N <= TG_MAX_B_BYTES &&
              TG_M * TG_N * 2 <= TG_MAX_C_BYTES, "operand window");

// Keep identical to gen_data.py.
static inline uint32_t tg_hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}
#ifndef TG_PATTERN
#define TG_PATTERN 0   // diagnostic operand patterns, see gen_data.py PATTERN
#endif
static inline uint8_t fp8_operand(uint32_t idx, uint32_t s) {
  if (TG_PATTERN == 5) return (s == 1 || (idx % TG_K) == (idx / TG_K) % TG_K) ? 0x38 : 0x00;
  if (TG_PATTERN == 1 && s == 0) return (idx % TG_K) == (idx / TG_K) % TG_K ? 0x38 : 0x00;
  if (TG_PATTERN == 2) return 0x38;
  const uint32_t h = tg_hash32(idx * 4u + s);
  uint32_t exp = (h >> 3) & 7u;
  if (TG_PATTERN == 3) exp = 1u + exp % 7u;
  uint32_t mant = h & 7u;
  if (TG_PATTERN == 6 && s == 0) {   // gen_data.py PATTERN 6
    exp += 8u;
    if (exp == 15u && mant == 7u) mant = 6u;
  }
  if (exp == 0) mant = 0;     // subnormals flushed to zero: they corrupt the mesh (gen_data.py)
  return (uint8_t)((((h >> 6) & 1u) << 7) | (exp << 3) | mant);
}
static inline uint8_t scale_code(uint32_t idx, uint32_t s) {
  if (TG_PATTERN == 5) {   // idx = g * M + m (A) or g * N + n (B)
    if (s == 2) return (uint8_t)(127u + (idx % TG_M + 5u * (idx / TG_M)) % 4u);
    return (uint8_t)(127u + 4u * ((idx % TG_N + 3u * (idx / TG_N)) % 4u));
  }
#ifdef TG_UNIT_SCALES
  (void)idx; (void)s;
  return 0x7Fu;   // gen_data.py --unit-scales
#else
  return (uint8_t)(0x7Bu + (tg_hash32(idx * 4u + s) & 7u));
#endif
}

// Byte streams are written 8 bytes per store: a byte store per element is ~8x slower, which in
// the metasim alone costs millions of cycles for a 128x128x256 problem.
template <typename F>
static void put_bytes(uint32_t device_addr, uint32_t n, F byte_at) {
  volatile uint64_t *dst = (volatile uint64_t *)rad_gpu_ptr(device_addr);
  for (uint32_t i = 0; i < n; i += 8) {
    uint64_t w = 0;
    for (uint32_t j = 0; j < 8; j++) w |= (uint64_t)byte_at(i + j) << (8 * j);
    dst[i / 8] = w;
  }
}

static void write_inputs(void) {
  static_assert((TG_M * TG_K) % 8 == 0 && (TG_K * TG_N) % 8 == 0 && TG_TM % 8 == 0 && TG_TN % 8 == 0,
                "8-byte packed writes");
#ifndef TG_NO_OPERANDS   // timing benchmarks on slow targets: A / B keep whatever DRAM holds
  put_bytes(TG_A_ADDR, TG_M * TG_K, [](uint32_t i) { return fp8_operand(i, 0); });
  put_bytes(TG_B_ADDR, TG_K * TG_N, [](uint32_t i) { return fp8_operand(i, 1); });
#endif

  // Scales: logical SA[K/32][M], SB[K/32][N]; stored per (output tile, k-tile) as [TK/32][TM]:
  // byte o of the stream is (tile, k-tile, group, column) = o / (KT*G*T), ..., o % T.
  constexpr uint32_t KT = TG_K / TG_TK, G = TG_TK / 32;
  put_bytes(TG_SA_ADDR, TG_M * TG_K / 32, [](uint32_t o) {
    const uint32_t m = o % TG_TM, gi = (o / TG_TM) % G, k = (o / (TG_TM * G)) % KT;
    const uint32_t mi = o / (TG_TM * G * KT);
    return scale_code((k * G + gi) * TG_M + mi * TG_TM + m, 2);
  });
  put_bytes(TG_SB_ADDR, TG_N * TG_K / 32, [](uint32_t o) {
    const uint32_t n = o % TG_TN, gi = (o / TG_TN) % G, k = (o / (TG_TN * G)) % KT;
    const uint32_t nj = o / (TG_TN * G * KT);
    return scale_code((k * G + gi) * TG_N + nj * TG_TN + n, 3);
  });

  volatile uint64_t *C = (volatile uint64_t *)rad_gpu_ptr(TG_C_ADDR);
  for (uint32_t i = 0; i < TG_M * TG_N / 4; i++) C[i] = 0xDEADBEEFDEADBEEFull;

  volatile TGArgs *args = (volatile TGArgs *)rad_gpu_ptr(TG_ARGS_ADDR);
  args->a = TG_A_ADDR; args->b = TG_B_ADDR; args->c = TG_C_ADDR;
  args->sa = TG_SA_ADDR; args->sb = TG_SB_ADDR;
  args->m = TG_M; args->n = TG_N; args->k = TG_K;
  args->tm = TG_TM; args->tn = TG_TN; args->tk = TG_TK;
  args->mode = TG_MODE; args->reps = TG_REPS;
}

// bf16 -> fixed point with 4 fractional bits (truncating toward zero); |v| < 2^27 assumed.
static inline int64_t bf16_fixed(uint16_t h) {
  const int e = (h >> 7) & 0xff;
  if (e == 0) return 0;
  const int64_t m = 128 + (h & 0x7f);            // value = m * 2^(e - 134)
  const int sh = e - 130;                        // times 2^4
  const int64_t v = sh >= 0 ? (sh > 40 ? (int64_t)1 << 50 : m << sh) : (sh < -8 ? 0 : m >> -sh);
  return (h & 0x8000u) ? -v : v;
}
static inline uint64_t isqrt64(uint64_t x) {
  uint64_t r = 0, bit = 1ull << 62;
  while (bit > x) bit >>= 2;
  while (bit) { if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; } else r >>= 1; bit >>= 2; }
  return r;
}

// Mesh cycles for `macs` multiply-accumulates at 256 MAC/cycle, as a percentage of `cycles`
// with two decimals (integer only).
static void put_util(uint64_t macs, uint64_t cycles) {
  const uint64_t bp = cycles ? (macs * 10000u / 256u) / cycles : 0;   // basis points
  rad_putu(bp / 100); rad_puts("."); rad_putu((bp / 10) % 10); rad_putu(bp % 10); rad_puts("%");
}

int main() {
  rad_puts("tapeout_gemm: M="); rad_putu(TG_M); rad_puts(" N="); rad_putu(TG_N);
  rad_puts(" K="); rad_putu(TG_K); rad_puts(" tile="); rad_putu(TG_TM); rad_puts("x");
  rad_putu(TG_TN); rad_puts("x"); rad_putu(TG_TK); rad_puts(" mode="); rad_putu(TG_MODE);
  rad_puts(" reps="); rad_putu(TG_REPS); rad_puts("\n");
  rad_flush();

  rad_host_gpu_soft_reset();
  write_inputs();
#ifdef TG_HOST_EVICT_BYTES
  // Experiment: force the host's dirty lines out to DRAM by streaming through a region larger
  // than the host caches, in case the mesh DMA reads DRAM without seeing host-cached writes.
  {
    volatile uint64_t *ev = (volatile uint64_t *)rad_gpu_ptr(0x1C000000u);
    for (uint32_t i = 0; i < TG_HOST_EVICT_BYTES / 8; i++) ev[i] = i;
    uint64_t sum = 0;
    for (uint32_t i = 0; i < TG_HOST_EVICT_BYTES / 8; i++) sum += ev[i];
    rad_puts("evict: bytes="); rad_putu(TG_HOST_EVICT_BYTES); rad_puts(" sum="); rad_putu(sum);
    rad_puts("\n");
  }
#endif
#ifdef MXG_PROF
  for (uint32_t j = 0; j < 4u; j++) *(volatile uint64_t *)RAD_PB_HOST(0, TG_PB_PROF + j) = 0;   // printBuf persists
#endif
  const int ran = rad_host_run();

  // Per cluster: config, work, first tile; the slowest cluster's times are the GEMM's.
  uint32_t t0 = 0, tc = 0, te = 0, tm = 0;
  for (uint32_t cl = 0; cl < RAD_HOST_NUM_CLUSTERS; cl++) {
    const uint32_t a0 = rad_host_postbox_cl(cl, TG_PB_T_START), ac = rad_host_postbox_cl(cl, TG_PB_T_CFG);
    const uint32_t ae = rad_host_postbox_cl(cl, TG_PB_T_END), am = rad_host_postbox_cl(cl, TG_PB_T_MARK);
    rad_puts("cycles: cluster="); rad_putu(cl); rad_puts(" config="); rad_putu(ac - a0);
    rad_puts(" work="); rad_putu(ae - ac); rad_puts(" first_tile="); rad_putu(am - ac); rad_puts("\n");
    if (cl == 0 || ae - ac > te - tc) { t0 = a0; tc = ac; te = ae; tm = am; }
  }

#ifdef MXG_PROF
  rad_puts("mxg_prof: issue="); rad_putu(rad_host_postbox(TG_PB_PROF + 0u));
  rad_puts(" scales="); rad_putu(rad_host_postbox(TG_PB_PROF + 1u));
  rad_puts(" drain_wait="); rad_putu(rad_host_postbox(TG_PB_PROF + 2u));
  rad_puts(" c_out="); rad_putu(rad_host_postbox(TG_PB_PROF + 3u)); rad_puts("\n");
#endif
  int pass = ran;
  const uint64_t tile_macs = (uint64_t)TG_TM * TG_TN * TG_TK;
  if (TG_MODE == TG_GEMM) {
    const uint64_t macs = (uint64_t)TG_M * TG_N * TG_K;
    rad_puts("util: work="); put_util(macs, te - tc); rad_puts(" with_config=");
    put_util(macs, te - t0); rad_puts("\n");
    const volatile uint16_t *C = (const volatile uint16_t *)rad_gpu_ptr(TG_C_ADDR);
    // Gate: no non-finite values, and Frobenius error vs exact math within 1.25x of what the
    // mesh's accumulation model (mx_golden) predicts.  Fixed point at 2^-4 keeps every squared
    // sum inside 64 bits for |C| < 2^27.  The checksum catches bit-level regressions between
    // kernel versions with the same tile shape (same accumulation order).
    uint64_t d2 = 0, g2 = 0, sum = 0;
    uint32_t nonfinite = 0, poison = 0;
    for (uint32_t i = 0; i < TG_M * TG_N; i++) {
      const uint16_t got = C[i];
      sum = (sum ^ got) * 0x100000001B3ull;
      if (got == 0xBEEF || got == 0xDEAD) poison++;
      if ((got & 0x7F80u) == 0x7F80u) { nonfinite++; continue; }
      const uint32_t row = i / TG_N;
      if (row % TG_CHECK_ROW_STRIDE) continue;   // only every stride-th row has a reference
      const int64_t o = bf16_fixed(got);
      const int64_t g = bf16_fixed(tg_exact_c[(row / TG_CHECK_ROW_STRIDE) * TG_N + i % TG_N]);
      d2 += (uint64_t)((o - g) * (o - g));
      g2 += (uint64_t)(g * g);
    }
    const uint64_t ppm = g2 ? isqrt64(d2) * 1000000ull / isqrt64(g2) : 0;
    rad_puts("check: frob_ppm="); rad_putu(ppm); rad_puts(" ref_ppm="); rad_putu(TG_REF_FROB_PPM);
    rad_puts(" nonfinite="); rad_putu(nonfinite); rad_puts(" poison="); rad_putu(poison);
    rad_puts(" checksum="); rad_putx((uint32_t)(sum ^ (sum >> 32))); rad_puts("\n");
    // poison is reported, not gated: a correct bf16 output can equal 0xDEAD or 0xBEEF, while an
    // unwritten element is non-finite or huge and already fails the Frobenius bound.
    pass = pass && nonfinite == 0 && ppm * 4 <= (uint64_t)TG_REF_FROB_PPM * 5;
#ifdef TG_DUMP
    // Whole C as hex, 16 values per line, for offline comparison (dump_compare.py).
    rad_puts("C_DUMP_BEGIN\n"); rad_flush();
    for (uint32_t i = 0; i < TG_M * TG_N; i++) {
      static const char hex[] = "0123456789abcdef";
      const uint16_t v = C[i];
      for (int sh = 12; sh >= 0; sh -= 4) rad_putc(hex[(v >> sh) & 0xf]);
      rad_putc((i & 15) == 15 ? '\n' : ' ');
      if ((i & 255) == 255) rad_flush();
    }
    rad_puts("C_DUMP_END\n"); rad_flush();
#endif
  } else if (TG_MODE == TG_BENCH_MVIN) {
    const uint64_t bytes = (uint64_t)TG_REPS * (TG_TM * TG_TK + TG_TK * TG_TN);
    rad_puts("mvin: bytes="); rad_putu(bytes); rad_puts(" cycles="); rad_putu(te - tc);
    rad_puts(" milli_bytes_per_cycle="); rad_putu(bytes * 1000u / (te - tc)); rad_puts("\n");
  } else if (TG_MODE == TG_BENCH_COMPUTE) {
    rad_puts("compute: util="); put_util(tile_macs * TG_REPS, te - tc);
    rad_puts(" ideal_cycles="); rad_putu(tile_macs * TG_REPS / 256u); rad_puts("\n");
  } else if (TG_MODE == TG_BENCH_SCALES) {
    const uint64_t words = (uint64_t)TG_REPS * (TG_TM + TG_TN) * TG_TK / 32u / 4u;
    rad_puts("scales: words="); rad_putu(words); rad_puts(" cycles="); rad_putu(te - tc);
    rad_puts(" milli_cycles_per_word="); rad_putu((te - tc) * 1000u / words); rad_puts("\n");
  }
  rad_puts(pass ? "PASS\n" : "FAIL\n");
  rad_flush();
  return pass ? 0 : 1;
}
