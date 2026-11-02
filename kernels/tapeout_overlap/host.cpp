// host.cpp: Rocket side of the mesh / SIMT overlap microbenchmark (kernel.cpp).
//
// Writes A, B and their scales (the integer hash shared with gen_data.py), poisons C, runs the
// kernel, and prints: timestamps, the workers' checksums, a hash over all of C, and C's
// Frobenius error against OV_REPS x (exact A B).  Arms of an experiment are compared by the C hash
// (must be bit-identical to the idle-worker arm) and by the worker checksums (must equal the
// serialized arm).  The host has no FPU: integer arithmetic only.
//
// OV_HOST_REPS must equal the kernel's OV_REPS.  OV_HOST_MM2 also checks C2 (C moved out of SMEM).

// A hung kernel must report within a metasim budget (the default allows 600M host cycles).
#ifndef RAD_HOST_RUN_MAX_CYCLES
#define RAD_HOST_RUN_MAX_CYCLES 6000000ull
#endif
#include <rad_host.h>
#include <stdint.h>

#include "ov.h"
#include "ov_data.h"

#ifndef OV_HOST_REPS
#define OV_HOST_REPS 8
#endif

#define TG_PATTERN 0
// Keep identical to gen_data.py.
static inline uint32_t tg_hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}
static inline uint8_t fp8_operand(uint32_t idx, uint32_t s) {
  const uint32_t h = tg_hash32(idx * 4u + s);
  const uint32_t exp = (h >> 3) & 7u;
  uint32_t mant = h & 7u;
  if (exp == 0) mant = 0;     // subnormals flushed to zero: they corrupt the mesh
  return (uint8_t)((((h >> 6) & 1u) << 7) | (exp << 3) | mant);
}
static inline uint8_t scale_code(uint32_t idx, uint32_t s) {
  return (uint8_t)(0x7Bu + (tg_hash32(idx * 4u + s) & 7u));
}

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
  put_bytes(OV_A_ADDR, TG_M * TG_K, [](uint32_t i) { return fp8_operand(i, 0); });
  put_bytes(OV_B_ADDR, TG_K * TG_N, [](uint32_t i) { return fp8_operand(i, 1); });
  constexpr uint32_t KT = TG_K / TG_TK, G = TG_TK / 32;
  put_bytes(OV_SA_ADDR, TG_M * TG_K / 32, [](uint32_t o) {
    const uint32_t m = o % TG_TM, gi = (o / TG_TM) % G, k = (o / (TG_TM * G)) % KT;
    const uint32_t mi = o / (TG_TM * G * KT);
    return scale_code((k * G + gi) * TG_M + mi * TG_TM + m, 2);
  });
  put_bytes(OV_SB_ADDR, TG_N * TG_K / 32, [](uint32_t o) {
    const uint32_t n = o % TG_TN, gi = (o / TG_TN) % G, k = (o / (TG_TN * G)) % KT;
    const uint32_t nj = o / (TG_TN * G * KT);
    return scale_code((k * G + gi) * TG_N + nj * TG_TN + n, 3);
  });
  volatile uint64_t *C = (volatile uint64_t *)rad_gpu_ptr(OV_C_ADDR);
  volatile uint64_t *C2 = (volatile uint64_t *)rad_gpu_ptr(OV_C2_ADDR);
  for (uint32_t i = 0; i < TG_M * TG_N / 4; i++) { C[i] = 0xDEADBEEFDEADBEEFull; C2[i] = 0xDEADBEEFDEADBEEFull; }
}

static inline int64_t bf16_fixed(uint16_t h) {
  const int e = (h >> 7) & 0xff;
  if (e == 0) return 0;
  const int64_t m = 128 + (h & 0x7f);
  const int sh = e - 130;
  const int64_t v = sh >= 0 ? (sh > 40 ? (int64_t)1 << 50 : m << sh) : (sh < -8 ? 0 : m >> -sh);
  return (h & 0x8000u) ? -v : v;
}
static inline uint64_t isqrt64(uint64_t x) {
  uint64_t r = 0, bit = 1ull << 62;
  while (bit > x) bit >>= 2;
  while (bit) { if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; } else r >>= 1; bit >>= 2; }
  return r;
}

// Hash of all of C, poison count, non-finite count and Frobenius ppm vs OV_HOST_REPS x exact.
static int check_c(const char *name, uint32_t addr) {
  const volatile uint64_t *C = (const volatile uint64_t *)rad_gpu_ptr(addr);
  uint64_t d2 = 0, g2 = 0, sum = 0xCBF29CE484222325ull, word = 0;
  int64_t og = 0;   // <C, reference>: C / reference best-fit scale = og / g2
  uint32_t nonfinite = 0, poison = 0;
  for (uint32_t i = 0; i < TG_M * TG_N; i++) {
    if ((i & 3u) == 0) word = C[i / 4];   // 64-bit loads: host reads are slow in the metasim
    const uint16_t got = (uint16_t)(word >> (16 * (i & 3u)));
    sum = (sum ^ got) * 0x100000001B3ull;
    if (got == 0xBEEF || got == 0xDEAD) poison++;
    if ((got & 0x7F80u) == 0x7F80u) { nonfinite++; continue; }
    const int64_t o = bf16_fixed(got);
    const int64_t g = bf16_fixed(tg_exact_c[i]) * OV_HOST_REPS;
    d2 += (uint64_t)((o - g) * (o - g));
    g2 += (uint64_t)(g * g);
    og += o * g;
  }
  rad_puts("check "); rad_puts(name); rad_puts(": fit_x8000=");
  if (og < 0) { rad_puts("-"); rad_putu(g2 ? (uint64_t)(-og) * 8000u / g2 : 0); }
  else rad_putu(g2 ? (uint64_t)og * 8000u / g2 : 0);
  rad_puts(" (8000 = exact, k * 1000 = C looks like k/8 of the reference)\n");
  const uint64_t ppm = g2 ? isqrt64(d2) * 1000000ull / isqrt64(g2) : 0;
#ifdef OV_HOST_TILEMAP
  // per 16x16 output tile: relative error in percent (capped at 99), 8 x 8 map, row = tile i
  {
    static uint64_t td2[8][8], tg2[8][8];
    for (uint32_t a = 0; a < 8; a++) for (uint32_t b = 0; b < 8; b++) { td2[a][b] = 0; tg2[a][b] = 0; }
    for (uint32_t i = 0; i < TG_M * TG_N; i += 4) {
      const uint64_t w4 = C[i / 4];
      for (uint32_t q = 0; q < 4; q++) {
        const uint32_t e = i + q, r = e / TG_N, c = e % TG_N;
        const uint16_t got = (uint16_t)(w4 >> (16 * q));
        if ((got & 0x7F80u) == 0x7F80u) { td2[r / 16][c / 16] += 1ull << 40; continue; }
        const int64_t o = bf16_fixed(got), g = bf16_fixed(tg_exact_c[e]) * OV_HOST_REPS;
        td2[r / 16][c / 16] += (uint64_t)((o - g) * (o - g));
        tg2[r / 16][c / 16] += (uint64_t)(g * g);
      }
    }
    for (uint32_t a = 0; a < 8; a++) {
      rad_puts("tilemap "); rad_putu(a); rad_puts(":");
      for (uint32_t b = 0; b < 8; b++) {
        uint64_t pct = tg2[a][b] ? isqrt64(td2[a][b]) * 100ull / isqrt64(tg2[a][b]) : 0;
        if (pct > 99) pct = 99;
        rad_puts(" "); if (pct < 10) rad_puts("0"); rad_putu(pct);
      }
      rad_puts("\n");
    }
  }
#endif
  rad_puts("check "); rad_puts(name); rad_puts(": hash="); rad_putx((uint32_t)(sum ^ (sum >> 32)));
  rad_puts(" frob_ppm="); rad_putu(ppm); rad_puts(" ref_ppm="); rad_putu(TG_REF_FROB_PPM);
  rad_puts(" nonfinite="); rad_putu(nonfinite); rad_puts(" poison="); rad_putu(poison); rad_puts("\n");
  return nonfinite == 0 && poison < 64 && ppm * 4 <= (uint64_t)TG_REF_FROB_PPM * 5;
}

// Expected worker checksums (kernel.cpp worker_pattern / worker_hash), computed from per-line
// access counts so the host loop stays short in simulation.
static inline uint32_t init_word(uint32_t addr) { return (addr * 2654435761u) ^ 0x5BD1E995u; }
static void expected_workers(uint32_t *rsum, uint32_t *hsum) {
  constexpr uint32_t L = OV_WBYTES / 64u;
  static uint32_t cnt[L][8], touched[L], last[L];
  for (uint32_t l = 0; l < L; l++) { touched[l] = 0; last[l] = 0; for (uint32_t k = 0; k < 8; k++) cnt[l][k] = 0; }
  for (uint32_t i = 0; i < OV_ITERS; i++)
    for (uint32_t k = 0; k < 8; k++) {
      const uint32_t l = ((8u * i + k) * OV_STRIDE) & (L - 1u);
      cnt[l][k]++; touched[l]++; last[l] = i + 1u;   // last = 1 + last iteration that wrote l
    }
  uint32_t rs = 0, hs = 0;
  for (uint32_t core = 0; core < 2; core++)
    for (uint32_t wic = 0; wic < OV_OCC; wic++) {
      if (OV_PAT == 0 || (core == 0 && wic == 0) || !((OV_CORES >> core) & 1u) ||
          !((OV_WMASK >> (core * OV_OCC + wic)) & 1u)) continue;
      const uint32_t base = OV_RB + (core * OV_OCC + wic) * OV_WBYTES;
      for (uint32_t lane = 0; lane < 16; lane++) {
        uint32_t acc = 0, h = 0;
        for (uint32_t l = 0; l < L; l++) {
          const uint32_t a = base + 64u * l + 4u * lane;
          uint32_t v = init_word(a);
          if (OV_PAT == 5)
            for (uint32_t k = 0; k < 8; k++) acc += cnt[l][k] * (a ^ (k << 3));
          if (OV_PAT == 1 || (OV_PAT == 4 && lane == 0))
            for (uint32_t k = 0; k < 8; k++) acc += cnt[l][k] * (v ^ (k << 3));
          if (OV_PAT == 2 && last[l]) v = a ^ ((last[l] - 1u) << 8);
          if (OV_PAT == 3 || OV_PAT == 10 || (OV_PAT == 8 && lane == 0)) v += touched[l];
          h += (v ^ a) * 0x85EBCA6Bu;
        }
        rs += acc; hs += h;
      }
    }
  *rsum = rs; *hsum = hs;
}

static void clear_slots(void) {
  for (uint32_t s = OV_PB_T0; s <= 50u; s++)
    *(volatile uint64_t *)RAD_PB_HOST(0, s) = 0ull;
}

int main() {
  rad_puts("tapeout_overlap: reps="); rad_putu(OV_HOST_REPS); rad_puts("\n");
  rad_flush();
  rad_host_gpu_soft_reset();
#ifdef OV_HOST_WRITES
  write_inputs();   // slow in the metasim (minutes of host stores); the GPU generates by default
#else
  *(volatile uint32_t *)rad_gpu_ptr(OV_ARGS_ADDR) = 0u;   // launch 0: GPU generates A, B, scales
  if (!rad_host_run()) { rad_puts("generate launch FAILED\n"); rad_flush(); }
#endif
  *(volatile uint32_t *)rad_gpu_ptr(OV_ARGS_ADDR) = 1u;
  clear_slots();
  const int ran = rad_host_run();

  const uint32_t t0 = rad_host_postbox(OV_PB_T0), ta = rad_host_postbox(OV_PB_T_A);
  const uint32_t ti = rad_host_postbox(OV_PB_T_ISS), tm = rad_host_postbox(OV_PB_T_MM);
  const uint32_t tb = rad_host_postbox(OV_PB_T_B), we = rad_host_postbox(OV_PB_W_END);
  const uint32_t wm = rad_host_postbox(OV_PB_W_MIN);
  rad_puts("cycles: setup="); rad_putu(ta - t0); rad_puts(" issue="); rad_putu(ti - ta);
  rad_puts(" mesh="); rad_putu(tm - ta); rad_puts(" window="); rad_putu(tb - ta);
  rad_puts(" workers_min="); rad_putu(wm - ta); rad_puts(" workers_max="); rad_putu(we - ta);
  rad_puts("\n");
  rad_puts("workers: rsum="); rad_putx(rad_host_postbox(OV_PB_RSUM));
  rad_puts(" hsum="); rad_putx(rad_host_postbox(OV_PB_HSUM));
  rad_puts(" iters="); rad_putu(rad_host_postbox(OV_PB_ITERS));
  rad_puts(" magic="); rad_putx(rad_host_postbox(OV_PB_MAGIC));
  rad_puts(" stage="); rad_putu(rad_host_postbox(OV_PB_STAGE)); rad_puts("\n");
#ifdef OV_PROGRESS
  rad_puts("progress: mm_issued="); rad_putu(rad_host_postbox(50));
  for (uint32_t s = 0; s < 6; s++) { rad_puts(" w"); rad_putu(s); rad_puts("="); rad_putu(rad_host_postbox(44 + s)); }
  rad_puts("\n");
#endif
  uint32_t ers, ehs;
  expected_workers(&ers, &ehs);
  // OV_PAT 6 and 16 read uninitialized DRAM: their read sums are not checked
  // OV_PAT 7's values depend on load/store interleaving and OV_PAT 9 hashes only the source
  // region (unchanged; its read sum is not modelled): checked only for the hash
  const int wok = OV_STAGE || OV_PAT == 14 || OV_PAT == 15 || OV_PAT == 7 || OV_PAT == 11 || ((OV_PAT == 9 || OV_PAT == 12 || OV_PAT == 13) && rad_host_postbox(OV_PB_HSUM) == ehs) || (OV_PAT == 6 || OV_PAT == 16 || rad_host_postbox(OV_PB_RSUM) == ers) && rad_host_postbox(OV_PB_HSUM) == ehs;
  rad_puts("workers: expect rsum="); rad_putx(ers); rad_puts(" hsum="); rad_putx(ehs);
  rad_puts(wok ? " MATCH\n" : " MISMATCH\n");
  rad_flush();
  int pass = wok && ran && rad_host_postbox(OV_PB_MAGIC) == (0x0E0F0000u | OV_HOST_REPS);
#ifdef OV_SMEMHASH
  {
    const volatile uint32_t *hb = rad_gpu_ptr(OV_HASH_ADDR);
    rad_puts("smemhash:");
    for (uint32_t b = 0; b < 32; b++) {
      uint32_t x = 0;
      for (uint32_t l = 0; l < 16; l++) x = x * 31u + hb[16 * b + l];
      rad_puts(" "); rad_putx(x);
    }
    rad_puts("\n"); rad_flush();
  }
#endif
  pass = check_c("C", OV_C_ADDR) && pass;
#ifdef OV_HOST_MM2
  pass = check_c("C2", OV_C2_ADDR) && pass;
#endif
  rad_puts(pass ? "PASS\n" : "FAIL\n");
  rad_flush();
  return pass ? 0 : 1;
}
