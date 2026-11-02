// host.cpp: Rocket side of the tapeout encoder layer.
//
// Generates the input and parameters with the integer hash shared with gen_ref.py, writes them to
// GPU DRAM, launches the stages, and checks sampled rows of each stage against ref_data.h.
// Build-time knob: EN_TEST (default 1: LNQ1 -> QKV chain; 2: split stage on synthetic qkv;
// 3: split + attention on synthetic qkv; 4: the whole layer, timed per stage).
// The host has no FPU: integer arithmetic only.

#include "enc.h"   // EN_CLUSTERS sets RAD_HOST_NUM_CLUSTERS (before rad_host.h)
#define RAD_HOST_NUM_CLUSTERS EN_CLUSTERS
#include <rad_host.h>
#include <stdint.h>

#include "enc.h"
#include "ref_data.h"

// Sample rows the checks read.  Each host read of GPU DRAM costs ~1000 cycles in the metasim, so
// metasim builds (EN_GEN_X) check the first 8 of the REF_NROWS sample rows.
#ifndef EN_CHECK_NROWS
#ifdef EN_GEN_X
#define EN_CHECK_NROWS 8u
#else
#define EN_CHECK_NROWS REF_NROWS
#endif
#endif

#ifndef EN_TEST
#define EN_TEST 1
#endif

static inline uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}
static inline uint32_t hs(uint32_t idx, uint32_t stream) { return hash32(idx * 8u + stream); }

static void write_inputs(void) {
#ifndef EN_GEN_X
  // x: bf16, sign random, exponent 124..127, mantissa random  (gen_ref.gen_x)
  volatile uint64_t *x = (volatile uint64_t *)rad_gpu_ptr(EN_X);
  for (uint32_t i = 0; i < EN_S * EN_D; i += 4) {
    uint64_t w = 0;
    for (uint32_t j = 0; j < 4; j++) {
      const uint32_t h = hs(i + j, EN_ST_X);
      const uint64_t b = ((h >> 31) << 15) | ((124u + ((h >> 8) & 3u)) << 7) | (h & 0x7Fu);
      w |= b << (16 * j);
    }
    x[i / 4] = w;
  }
#endif
  // LayerNorm params as fp32 holding bf16 values  (gen_ref.gen_ln)
  for (uint32_t which = 1; which <= 2; which++) {
    volatile uint64_t *g = (volatile uint64_t *)rad_gpu_ptr(which == 1 ? EN_LN1_G : EN_LN2_G);
    volatile uint64_t *b = (volatile uint64_t *)rad_gpu_ptr(which == 1 ? EN_LN1_B : EN_LN2_B);
    for (uint32_t d = 0; d < EN_D; d += 2) {
      uint64_t gw = 0, bw = 0;
      for (uint32_t j = 0; j < 2; j++) {
        const uint32_t hg = hs((d + j) * 4u + 2u * (which - 1), EN_ST_LN);
        const uint32_t hb = hs((d + j) * 4u + 2u * (which - 1) + 1u, EN_ST_LN);
        const uint64_t gb = 0x3F80u | (hg & 0x7Fu);
        const uint64_t bb = ((hb >> 31) << 15) | ((120u + ((hb >> 8) & 3u)) << 7) | (hb & 0x7Fu);
        gw |= (gb << 16) << (32 * j);
        bw |= (bb << 16) << (32 * j);
      }
      g[d / 2] = gw;
      b[d / 2] = bw;
    }
  }
}

// B-operand weight [K][N]: codes (exponent field 1..7) and logical scales [K/32][N] 0x79..0x7B
// stored tile-blocked [N/TN][K/TK][TK/32][TN]  (gen_ref.gen_weight)
static void write_weight(uint32_t stream, uint32_t wsc_base, uint32_t k, uint32_t n,
                         uint32_t codes_addr, uint32_t scales_addr) {
  volatile uint64_t *c = (volatile uint64_t *)rad_gpu_ptr(codes_addr);
  for (uint32_t i = 0; i < k * n; i += 8) {
    uint64_t w = 0;
    for (uint32_t j = 0; j < 8; j++) {
      const uint32_t h = hs(i + j, stream);
      const uint64_t code = (((h >> 6) & 1u) << 7) | ((1u + ((h >> 3) & 0xFFu) % 7u) << 3) | (h & 7u);
      w |= code << (8 * j);
    }
    c[i / 8] = w;
  }
  volatile uint64_t *sc = (volatile uint64_t *)rad_gpu_ptr(scales_addr);
  const uint32_t kt_n = k / EN_TK, g = EN_TK / 32;
  for (uint32_t o = 0; o < k * n / 32; o += 8) {
    uint64_t w = 0;
    for (uint32_t j = 0; j < 8; j++) {
      const uint32_t oo = o + j;
      const uint32_t col = oo % EN_TN, gi = (oo / EN_TN) % g, kt = (oo / (EN_TN * g)) % kt_n;
      const uint32_t nj = oo / (EN_TN * g * kt_n);
      const uint32_t logical = (kt * g + gi) * n + nj * EN_TN + col;
      w |= (uint64_t)(0x79u + hs(logical + wsc_base, EN_ST_WSC) % 3u) << (8 * j);
    }
    sc[o / 8] = w;
  }
}

// Synthetic qkv for the split test (gen_ref.gen_qkv_synthetic).
static void write_qkv_synthetic(void) {
  volatile uint64_t *q = (volatile uint64_t *)rad_gpu_ptr(EN_QKV);
  for (uint32_t i = 0; i < EN_S * 3 * EN_D; i += 4) {
    uint64_t w = 0;
    for (uint32_t j = 0; j < 4; j++) {
      const uint32_t h = hs(i + j, 7u);
      w |= (uint64_t)(((h >> 31) << 15) | ((124u + ((h >> 8) % 5u)) << 7) | (h & 0x7Fu)) << (16 * j);
    }
    q[i / 4] = w;
  }
}

// bf16 -> fixed point with `frac` fractional bits (default 4: |v| < 2^27 keeps sums of squares in
// 64 bits for the GEMM outputs; use more for small-valued tensors).
static inline int64_t bf16_fixed(uint16_t h, int frac = 4) {
  const int e = (h >> 7) & 0xff;
  if (e == 0) return 0;
  const int64_t m = 128 + (h & 0x7f);            // value = m * 2^(e - 134)
  const int sh = e - 134 + frac;
  const int64_t v = sh >= 0 ? (sh > 40 ? (int64_t)1 << 50 : m << sh) : (sh < -8 ? 0 : m >> -sh);
  return (h & 0x8000u) ? -v : v;
}
static inline uint64_t isqrt64(uint64_t x) {
  uint64_t r = 0, bit = 1ull << 62;
  while (bit > x) bit >>= 2;
  while (bit) { if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; } else r >>= 1; bit >>= 2; }
  return r;
}

// Sampled rows of a bf16 matrix [rows][cols] vs ref [REF_NROWS][cols]: Frobenius within 1.25x of
// the mesh model's own error, no non-finite values.
static int check_bf16_once(const char *name, uint32_t addr, uint32_t cols, const uint16_t *ref, uint32_t ref_ppm) {
  const volatile uint16_t *m = (const volatile uint16_t *)rad_gpu_ptr(addr);
  uint64_t d2 = 0, g2 = 0;
  uint32_t nonfinite = 0;
  for (uint32_t s = 0; s < EN_CHECK_NROWS; s++)
    for (uint32_t c = 0; c < cols; c++) {
      const uint16_t got = m[ref_rows[s] * cols + c];
      if ((got & 0x7F80u) == 0x7F80u) { nonfinite++; continue; }
      const int64_t o = bf16_fixed(got, 12), g = bf16_fixed(ref[s * cols + c], 12);
      d2 += (uint64_t)((o - g) * (o - g)); g2 += (uint64_t)(g * g);
    }
  const uint64_t ppm = g2 ? isqrt64(d2) * 1000000ull / isqrt64(g2) : 0;
  rad_puts("check "); rad_puts(name); rad_puts(": frob_ppm="); rad_putu(ppm);
  rad_puts(" model_ppm="); rad_putu(ref_ppm); rad_puts(" nonfinite="); rad_putu(nonfinite); rad_puts("\n");
  return nonfinite == 0 && ppm * 4 <= (uint64_t)ref_ppm * 5;
}

// Sampled rows of a bf16 matrix vs ref, absolute Frobenius bound in ppm.
static int check_bf16_frob(const char *name, uint32_t addr, uint32_t cols, const uint16_t *ref,
                           uint32_t max_ppm) {
  const volatile uint16_t *m = (const volatile uint16_t *)rad_gpu_ptr(addr);
  uint64_t d2 = 0, g2 = 0;
  uint32_t nonfinite = 0;
  for (uint32_t s = 0; s < EN_CHECK_NROWS; s++)
    for (uint32_t c = 0; c < cols; c++) {
      const uint16_t got = m[ref_rows[s] * cols + c];
      if ((got & 0x7F80u) == 0x7F80u) { nonfinite++; continue; }
      // 16 fractional bits: attention outputs are O(0.1), |O| < 2^8 keeps the sums in 64 bits
      const int64_t o = bf16_fixed(got, 16), g = bf16_fixed(ref[s * cols + c], 16);
      d2 += (uint64_t)((o - g) * (o - g)); g2 += (uint64_t)(g * g);
    }
  const uint64_t ppm = g2 ? isqrt64(d2) * 1000000ull / isqrt64(g2) : 0;
  rad_puts("check "); rad_puts(name); rad_puts(": frob_ppm="); rad_putu(ppm);
  rad_puts(" limit_ppm="); rad_putu(max_ppm); rad_puts(" nonfinite="); rad_putu(nonfinite); rad_puts("\n");
  return nonfinite == 0 && ppm <= max_ppm;
}

// Like check_bf16 with an explicit limit (ppm).
static int check_bf16_limit_once(const char *name, uint32_t addr, uint32_t cols, const uint16_t *ref,
                                 uint32_t limit_ppm) {
  const volatile uint16_t *m = (const volatile uint16_t *)rad_gpu_ptr(addr);
  uint64_t d2 = 0, g2 = 0;
  uint32_t nonfinite = 0;
#ifdef EN_CHECK_DEBUG
  uint32_t dbg = 0;
#endif
  for (uint32_t s = 0; s < EN_CHECK_NROWS; s++)
    for (uint32_t c = 0; c < cols; c++) {
      const uint16_t got = m[ref_rows[s] * cols + c];
#ifdef EN_CHECK_DEBUG   // first mismatches, and the same element through a 64-bit read
      if (got != ref[s * cols + c] && dbg < 4) {
        dbg++;
        const uint32_t e = ref_rows[s] * cols + c;
        const uint64_t w64 = ((const volatile uint64_t *)rad_gpu_ptr(addr))[e / 4];
        rad_puts("  mismatch s="); rad_putu(s); rad_puts(" row="); rad_putu(ref_rows[s]); rad_puts(" c="); rad_putu(c);
        rad_puts(" got16="); rad_putx(got); rad_puts(" via64="); rad_putx((uint32_t)((w64 >> (16 * (e % 4))) & 0xFFFF));
        rad_puts(" ref="); rad_putx(ref[s * cols + c]); rad_puts(" refaddr="); rad_putx((uint32_t)(uintptr_t)&ref[s * cols + c]);
        rad_puts("\n");
      }
#endif
      if ((got & 0x7F80u) == 0x7F80u) { nonfinite++; continue; }
      const int64_t o = bf16_fixed(got, 12), g = bf16_fixed(ref[s * cols + c], 12);
      d2 += (uint64_t)((o - g) * (o - g)); g2 += (uint64_t)(g * g);
    }
  const uint64_t ppm = g2 ? isqrt64(d2) * 1000000ull / isqrt64(g2) : 0;
  rad_puts("check "); rad_puts(name); rad_puts(": frob_ppm="); rad_putu(ppm);
  rad_puts(" limit_ppm="); rad_putu(limit_ppm); rad_puts(" nonfinite="); rad_putu(nonfinite); rad_puts("\n");
  return nonfinite == 0 && ppm <= limit_ppm;
}

// The host's reads of GPU DRAM have returned wrong data intermittently (qkv after the QKV GEMM,
// y in the stage test) while everything the GPU computed from the same buffers was right.  On a
// failure, wait ~2 s of host time and read again; both results are printed and the second decides.
#define EN_RECHECK_WAIT 20000000ull
static int check_bf16(const char *name, uint32_t addr, uint32_t cols, const uint16_t *ref, uint32_t ref_ppm) {
  if (check_bf16_once(name, addr, cols, ref, ref_ppm)) return 1;
  rad_wait_cycles(EN_RECHECK_WAIT);
  rad_puts("  re-read after wait: ");
  return check_bf16_once(name, addr, cols, ref, ref_ppm);
}
static int check_bf16_limit(const char *name, uint32_t addr, uint32_t cols, const uint16_t *ref,
                            uint32_t limit_ppm) {
  if (check_bf16_limit_once(name, addr, cols, ref, limit_ppm)) return 1;
  rad_wait_cycles(EN_RECHECK_WAIT);
  rad_puts("  re-read after wait: ");
  return check_bf16_limit_once(name, addr, cols, ref, limit_ppm);
}

// Attention output in the FA layout [H][S][DH] vs the reference o [rows][D] (heads side by side).
static void check_o_full(void) {
  const volatile uint16_t *m = (const volatile uint16_t *)rad_gpu_ptr(EN_FA_O);
  uint64_t d2 = 0, g2 = 0;
  uint32_t nonfinite = 0;
  for (uint32_t s = 0; s < EN_CHECK_NROWS; s++)
    for (uint32_t c = 0; c < EN_D; c++) {
      const uint16_t got = m[((c / EN_DH) * EN_S + ref_rows[s]) * EN_DH + c % EN_DH];
      if ((got & 0x7F80u) == 0x7F80u) { nonfinite++; continue; }
      const int64_t o = bf16_fixed(got, 16), g = bf16_fixed(ref_full_o[s * EN_D + c], 16);
      d2 += (uint64_t)((o - g) * (o - g)); g2 += (uint64_t)(g * g);
    }
  const uint64_t ppm = g2 ? isqrt64(d2) * 1000000ull / isqrt64(g2) : 0;
  rad_puts("check o: frob_ppm="); rad_putu(ppm); rad_puts(" model_ppm="); rad_putu(REF_O_MODEL_PPM);
  rad_puts(" nonfinite="); rad_putu(nonfinite); rad_puts("\n");
}

// Split check: heads 0 and 3; Q / K^T on the sample rows, V on columns 0..15.
static int check_split(void) {
  const volatile uint8_t *Q = (const volatile uint8_t *)rad_gpu_ptr(EN_FA_Q);
  const volatile uint8_t *QS = (const volatile uint8_t *)rad_gpu_ptr(EN_FA_QS);
  const volatile uint8_t *KT = (const volatile uint8_t *)rad_gpu_ptr(EN_FA_KT);
  const volatile uint8_t *KTS = (const volatile uint8_t *)rad_gpu_ptr(EN_FA_KTS);
  const volatile uint8_t *V = (const volatile uint8_t *)rad_gpu_ptr(EN_FA_V);
  const volatile uint8_t *VS = (const volatile uint8_t *)rad_gpu_ptr(EN_FA_VS);
  const uint32_t heads[2] = {0, 3}, G = EN_DH / 32;
  uint32_t bad[6] = {0, 0, 0, 0, 0, 0};
  for (uint32_t hi = 0; hi < 2; hi++) {
    const uint32_t h = heads[hi];
    for (uint32_t s = 0; s < EN_CHECK_NROWS; s++) {
      const uint32_t r = ref_rows[s];
      for (uint32_t d = 0; d < EN_DH; d++) {
        bad[0] += Q[h * EN_S * EN_DH + r * EN_DH + d] != ref_split_q[(hi * REF_NROWS + s) * EN_DH + d];
        bad[2] += KT[h * EN_DH * EN_S + d * EN_S + r] != ref_split_k[(hi * REF_NROWS + s) * EN_DH + d];
      }
      for (uint32_t g = 0; g < G; g++) {
        bad[1] += QS[h * EN_S * G + (r / 64) * G * 64 + g * 64 + r % 64] != ref_split_qs[(hi * REF_NROWS + s) * G + g];
        bad[3] += KTS[h * G * EN_S + g * EN_S + r] != ref_split_ks[(hi * REF_NROWS + s) * G + g];
      }
    }
    for (uint32_t d = 0; d < 16; d++) {
      for (uint32_t r = 0; r < EN_S; r++)
        bad[4] += V[h * EN_S * EN_DH + r * EN_DH + d] != ref_split_v[(hi * 16 + d) * EN_S + r];
      for (uint32_t b = 0; b < EN_S / 32; b++)
        bad[5] += VS[h * (EN_S / 32) * EN_DH + b * EN_DH + d] != ref_split_vs[(hi * 16 + d) * (EN_S / 32) + b];
    }
  }
  static const char *nm[6] = {"q", "qs", "kt", "kts", "v", "vs"};
  rad_puts("check split:");
  uint32_t total = 0;
  for (int i = 0; i < 6; i++) { rad_puts(" "); rad_puts(nm[i]); rad_puts("="); rad_putu(bad[i]); total += bad[i]; }
  rad_puts(" mismatches\n");
  return total == 0;
}

// Byte offset of (row, 32-group g) in the tile-blocked A-operand scale layout (as in kernel.cpp).
static inline uint32_t a_off(uint32_t r, uint32_t g, uint32_t cols) {
  const uint32_t kt = g / (EN_TK / 32), gi = g % (EN_TK / 32);
  return (((r / EN_TM) * (cols / EN_TK) + kt) * (EN_TK / 32) + gi) * EN_TM + r % EN_TM;
}
// Compare sampled rows of an A-operand (codes [rows][cols], tile-blocked scales) with the reference.
// max_permille: allowed code mismatches; 1 for bit-identical inputs, more when the quantized values
// themselves differ by bf16 ulps from the reference (GELU: polynomial-free bf16 exp vs numpy tanh).
static int check_a_operand(const char *name, uint32_t codes_addr, uint32_t scales_addr, uint32_t cols,
                           const uint8_t *ref_codes, const uint8_t *ref_scales, uint32_t max_permille = 1) {
  const volatile uint8_t *codes = (const volatile uint8_t *)rad_gpu_ptr(codes_addr);
  const volatile uint8_t *scales = (const volatile uint8_t *)rad_gpu_ptr(scales_addr);
  uint32_t code_bad = 0, scale_bad = 0, first_row = ~0u;
  for (uint32_t s = 0; s < EN_CHECK_NROWS; s++) {
    const uint32_t r = ref_rows[s];
    for (uint32_t c = 0; c < cols; c++) {
      if (codes[r * cols + c] != ref_codes[s * cols + c]) { code_bad++; if (first_row == ~0u) first_row = r; }
    }
    for (uint32_t g = 0; g < cols / 32; g++) {
      const uint32_t kt = g / (EN_TK / 32), gi = g % (EN_TK / 32);
      const uint32_t off = (((r / EN_TM) * (cols / EN_TK) + kt) * (EN_TK / 32) + gi) * EN_TM + r % EN_TM;
      if (scales[off] != ref_scales[s * (cols / 32) + g]) scale_bad++;
    }
  }
  rad_puts("check "); rad_puts(name); rad_puts(": code_mismatch="); rad_putu(code_bad);
  rad_puts("/"); rad_putu(EN_CHECK_NROWS * cols); rad_puts(" scale_mismatch="); rad_putu(scale_bad);
  rad_puts("/"); rad_putu(EN_CHECK_NROWS * cols / 32);
  if (first_row != ~0u) { rad_puts(" first_bad_row="); rad_putu(first_row); }
  rad_puts("\n");
  // a few last-bit differences from float summation order are expected; scales must match
  return scale_bad == 0 && code_bad * 1000 <= max_permille * EN_CHECK_NROWS * cols;
}

// e4m3 code times 2^(scale - 127), as a fixed-point integer with `frac` fractional bits.
static int64_t mx_fixed(uint8_t code, uint8_t scale, int frac) {
  const int E = (code >> 3) & 15, m = code & 7;
  const int64_t mant = E ? (8 | m) : m;
  const int sh = (E ? E : 1) - 10 + (int)scale - 127 + frac;
  const int64_t v = sh >= 0 ? mant << sh : (sh > -63 ? mant >> -sh : 0);
  return (code & 0x80) ? -v : v;
}
// Dequantized A operand (codes + tile-blocked scales) vs exact bf16 values, Frobenius in ppm.
// The reference's own codes go through the same sum, so the device may differ from the
// reference codes as long as its quantization error is no worse (limit: 1.05x the reference's).
static int check_a_frob(const char *name, uint32_t codes_addr, uint32_t scales_addr, uint32_t cols,
                        const uint16_t *exact, const uint8_t *ref_codes, const uint8_t *ref_scales) {
  const volatile uint8_t *codes = (const volatile uint8_t *)rad_gpu_ptr(codes_addr);
  const volatile uint8_t *scales = (const volatile uint8_t *)rad_gpu_ptr(scales_addr);
  uint64_t d2 = 0, r2 = 0, g2 = 0;
  for (uint32_t s = 0; s < EN_CHECK_NROWS; s++) {
    const uint32_t r = ref_rows[s];
    for (uint32_t c = 0; c < cols; c++) {
      const uint32_t g = c / 32, kt = g / (EN_TK / 32), gi = g % (EN_TK / 32);
      const uint32_t off = (((r / EN_TM) * (cols / EN_TK) + kt) * (EN_TK / 32) + gi) * EN_TM + r % EN_TM;
      const int64_t x = bf16_fixed(exact[s * cols + c], 12);
      const int64_t o = mx_fixed(codes[r * cols + c], scales[off], 12) - x;
      const int64_t q = mx_fixed(ref_codes[s * cols + c], ref_scales[s * (cols / 32) + g], 12) - x;
      d2 += (uint64_t)(o * o); r2 += (uint64_t)(q * q); g2 += (uint64_t)(x * x);
    }
  }
  const uint64_t ppm = g2 ? isqrt64(d2) * 1000000ull / isqrt64(g2) : 0;
  const uint64_t ref_ppm = g2 ? isqrt64(r2) * 1000000ull / isqrt64(g2) : 0;
  rad_puts("check "); rad_puts(name); rad_puts(": dequant_frob_ppm="); rad_putu(ppm);
  rad_puts(" ref_codes_ppm="); rad_putu(ref_ppm); rad_puts("\n");
  return ppm * 100 <= ref_ppm * 105;
}

// ideal mesh cycles of the layer at 256 MAC/cycle: four GEMMs + 16 attention tiles (QK + PV)
#define EN_IDEAL_MESH_CYCLES ((EN_S * 3 * EN_D * EN_D + EN_S * EN_D * EN_D + 2 * EN_S * EN_D * EN_F) / 256 + \
                              EN_H * (EN_S / 64) * 16384)

static void put_pct(uint64_t num, uint64_t den) {
  const uint64_t bp = den ? num * 10000u / den : 0;
  rad_putu(bp / 100); rad_puts("."); rad_putu((bp / 10) % 10); rad_putu(bp % 10); rad_puts("%");
}

// One args block per cluster (cluster c at EN_FA_ARGS + 0x100 c): with EN_CLUSTERS = 2, cluster c
// runs heads h + 2c and h + 2c + 1 (fa_stage.cpp: 8 tiles per launch per cluster).
static void set_fa_args_one(uint32_t addr, uint32_t h);
static void set_fa_args(uint32_t h) {
  for (uint32_t cl = 0; cl < EN_CLUSTERS; cl++) set_fa_args_one(EN_FA_ARGS + 0x100u * cl, h + cl * (EN_H / EN_CLUSTERS));
}
static void set_fa_args_one(uint32_t addr, uint32_t h) {
  volatile ENFaArgs *fa = (volatile ENFaArgs *)rad_gpu_ptr(addr);
  fa->q = EN_FA_Q + h * EN_S * EN_DH;
  fa->qs = EN_FA_QS + h * EN_S * (EN_DH / 32);
  fa->kt = EN_FA_KT + h * EN_DH * EN_S;
  fa->kts = EN_FA_KTS + h * (EN_DH / 32) * EN_S;
  fa->v = EN_FA_V + h * EN_S * EN_DH;
  fa->vs = EN_FA_VS + h * (EN_S / 32) * EN_DH;
  fa->o = EN_FA_O + h * EN_S * EN_DH * 2;
}

static int run_stage(uint32_t stage, const char *name);
// A stage's GPU time: the slowest cluster's start-to-end (each cluster stamps its own postbox).
static uint32_t stage_cycles() {
  uint32_t m = 0;
  for (uint32_t cl = 0; cl < EN_CLUSTERS; cl++) {
    const uint32_t d = rad_host_postbox_cl(cl, EN_PB_T_END) - rad_host_postbox_cl(cl, EN_PB_T_START);
    if (d > m) m = d;
  }
  return m;
}
static uint64_t run_stage_timed(uint32_t stage, const char *name, int *pass) {
  *pass = run_stage(stage, name) && *pass;
  return stage_cycles();
}

static int run_stage(uint32_t stage, const char *name) {
  volatile ENArgs *args = (volatile ENArgs *)rad_gpu_ptr(EN_ARGS);
  args->stage = stage;
  rad_puts("stage "); rad_puts(name); rad_puts(": ");
  rad_flush();
  // the postbox keeps last run's values: clear every slot read below
  const uint32_t slots[7] = {EN_PB_T_START, EN_PB_T_END, EN_PB_PROF, EN_PB_PROF + 1, EN_PB_PROF + 2, EN_PB_PROF + 3, EN_PB_PROF + 4};
  for (uint32_t cl = 0; cl < EN_CLUSTERS; cl++)
    for (uint32_t i = 0; i < 7; i++) *(volatile uint64_t *)RAD_PB_HOST(cl, slots[i]) = 0;
  const int ran = rad_host_run();
  rad_puts("stage "); rad_puts(name); rad_puts(" gpu_cycles="); rad_putu(stage_cycles());
#if EN_CLUSTERS > 1
  for (uint32_t cl = 0; cl < EN_CLUSTERS; cl++) {
    rad_puts(" c"); rad_putu(cl); rad_puts("=");
    rad_putu(rad_host_postbox_cl(cl, EN_PB_T_END) - rad_host_postbox_cl(cl, EN_PB_T_START));
  }
#endif
#ifdef EN_PRINT_T0   // GPU mcycle at each cluster's stage start (RTL simulation: tick ~ 2 x mcycle)
  for (uint32_t cl = 0; cl < EN_CLUSTERS; cl++) {
    rad_puts(" t0_c"); rad_putu(cl); rad_puts("="); rad_putu(rad_host_postbox_cl(cl, EN_PB_T_START));
  }
#endif
  if (rad_host_postbox(EN_PB_PROF + 2)) {
    rad_puts(" prof in="); rad_putu(rad_host_postbox(EN_PB_PROF));
    rad_puts(" bar="); rad_putu(rad_host_postbox(EN_PB_PROF + 1));
    rad_puts(" work="); rad_putu(rad_host_postbox(EN_PB_PROF + 2));
    rad_puts(" out="); rad_putu(rad_host_postbox(EN_PB_PROF + 3));
    rad_puts(" iss="); rad_putu(rad_host_postbox(EN_PB_PROF + 4));
  }
  rad_puts("\n");
  rad_flush();
  return ran;
}

// EN_HASH_OUT: every output buffer of the layer is poisoned before it and hashed whole after it, so
// a stage that skips rows cannot pass on stale DRAM (memory note stale-dram-masks-missing-writes).
struct HashRegion { const char *name; uint32_t addr, bytes; };
static const HashRegion kHashRegions[] = {
  {"h1", EN_H1, EN_S * EN_D}, {"h1s", EN_H1_S, EN_S * EN_D / 32},
  {"q", EN_FA_Q, EN_H * EN_S * EN_DH}, {"qs", EN_FA_QS, EN_H * EN_S * EN_DH / 32},
  {"kt", EN_FA_KT, EN_H * EN_S * EN_DH}, {"kts", EN_FA_KTS, EN_H * EN_S * EN_DH / 32},
  {"v", EN_FA_V, EN_H * EN_S * EN_DH}, {"vs", EN_FA_VS, EN_H * EN_S * EN_DH / 32},
  {"o", EN_FA_O, EN_H * EN_S * EN_DH * 2}, {"oa", EN_OA, EN_S * EN_D}, {"oas", EN_OA_S, EN_S * EN_D / 32},
  {"att", EN_ATT, EN_S * EN_D * 2}, {"x1", EN_X1, EN_S * EN_D * 2},
  {"h2", EN_H2, EN_S * EN_D}, {"h2s", EN_H2_S, EN_S * EN_D / 32},
  {"g", EN_G, EN_S * EN_F}, {"gs", EN_G_S, EN_S * EN_F / 32},
  {"ff2", EN_FF2, EN_S * EN_D * 2}, {"y", EN_Y, EN_S * EN_D * 2},
};
static void fill_outputs() {
  volatile ENArgs *a = (volatile ENArgs *)rad_gpu_ptr(EN_ARGS);
  for (uint32_t r = 0; r < sizeof(kHashRegions) / sizeof(kHashRegions[0]); r++) {
    a->gen_addr = kHashRegions[r].addr; a->gen_elems = kHashRegions[r].bytes; a->gen_stream = 0x51u + r;
    run_stage(EN_STAGE_FILL, "fill");
  }
}
static void hash_outputs() {
  volatile ENArgs *a = (volatile ENArgs *)rad_gpu_ptr(EN_ARGS);
  for (uint32_t r = 0; r < sizeof(kHashRegions) / sizeof(kHashRegions[0]); r++) {
    a->gen_addr = kHashRegions[r].addr; a->gen_elems = kHashRegions[r].bytes;
    run_stage(EN_STAGE_HASH, "hash");
    const volatile uint32_t *hw = (const volatile uint32_t *)rad_gpu_ptr(EN_HASH_OUT);
    uint32_t h = 0;
    for (uint32_t i = 0; i < 16u * 3u * 2u * EN_CLUSTERS; i++) h += hw[i];   // 6 warps per cluster
    rad_puts("hash "); rad_puts(kHashRegions[r].name); rad_puts("="); rad_putx(h); rad_puts("\n");
  }
}

// metasim: weights generated on the GPU (EN_STAGE_GENW) instead of host writes
static void gen_weight(uint32_t stream, uint32_t bytes, uint32_t codes_addr, uint32_t scales_addr) {
  volatile ENArgs *a = (volatile ENArgs *)rad_gpu_ptr(EN_ARGS);
  a->gen_addr = codes_addr; a->gen_elems = bytes; a->gen_stream = stream; a->gen_saddr = scales_addr;
  run_stage(EN_STAGE_GENW, "genw");
}
int main() {
  rad_puts("tapeout_encoder: S="); rad_putu(EN_S); rad_puts(" D="); rad_putu(EN_D);
  rad_puts(" H="); rad_putu(EN_H); rad_puts(" F="); rad_putu(EN_F); rad_puts("\n");
  rad_flush();
  rad_host_gpu_soft_reset();
  write_inputs();
  {
    volatile ENArgs *a = (volatile ENArgs *)rad_gpu_ptr(EN_ARGS);
    a->clear_lo = 0;
    a->clear_hi = 0;
  }
  int pass = 1;
#if EN_TEST == 5
  // LayerNorm 1 alone, minimal host writes (for the metasim)
#ifdef EN_GEN_X
  run_stage(EN_STAGE_GENX, "genx");
#endif
  pass = run_stage(EN_STAGE_LNQ1, "lnq1") && pass;
  check_a_operand("h1", EN_H1, EN_H1_S, EN_D, ref_h1_codes, ref_h1_scales, 1000);   // info: bf16 y changes some codes
  pass = check_a_frob("h1", EN_H1, EN_H1_S, EN_D, ref_h1_y, ref_h1_codes, ref_h1_scales) && pass;
#elif !defined(EN_GEN_X)   // metasim timing builds skip the large host writes
  write_weight(EN_ST_WQKV, 0u, EN_D, 3 * EN_D, EN_WQKV, EN_WQKV_S);
#endif
#if EN_TEST == 4
  // EN_QKV_TRACK: check the qkv buffer after every stage (it was overwritten in board run tp_enc_h4)
#ifdef EN_QKV_TRACK
  // snapshot of the sampled qkv rows right after the QKV GEMM; later stages must not change them
  static uint64_t qkv_snap[EN_CHECK_NROWS * 3 * EN_D / 4];
  auto qkv_track = [&](const char *st) {
    const volatile uint64_t *m = (const volatile uint64_t *)rad_gpu_ptr(EN_QKV);
    const uint32_t wpr = 3 * EN_D / 4;   // 8-byte words per row
    const bool take = st[0] == 'q' && st[1] == 'k' && st[2] == 'v' && st[3] == '_';
    uint32_t diff = 0;
    for (uint32_t s = 0; s < EN_CHECK_NROWS; s++)
      for (uint32_t i = 0; i < wpr; i++) {
        const uint64_t v = m[ref_rows[s] * wpr + i];
        if (take) qkv_snap[s * wpr + i] = v; else diff += v != qkv_snap[s * wpr + i];
      }
    rad_puts("qkv_track "); rad_puts(st); rad_puts(take ? " snapshot" : " changed_words="); if (!take) rad_putu(diff);
    rad_puts("\n");
  };
#define QKV_TRACK(st) qkv_track(st)
#else
#define QKV_TRACK(st) ((void)0)
#endif
  // the whole layer: 11 stages, attention as one launch
#ifdef EN_GEN_X   // metasim: inputs and weights from the GPU; the checks against ref then fail
  run_stage(EN_STAGE_GENX, "genx");
  gen_weight(EN_ST_WQKV, EN_D * 3 * EN_D, EN_WQKV, EN_WQKV_S);
  gen_weight(EN_ST_WO, EN_D * EN_D, EN_WO, EN_WO_S);
  gen_weight(EN_ST_W1, EN_D * EN_F, EN_W1, EN_W1_S);
  gen_weight(EN_ST_W2, EN_F * EN_D, EN_W2, EN_W2_S);
#else
  write_weight(EN_ST_WO, 1u << 22, EN_D, EN_D, EN_WO, EN_WO_S);
  write_weight(EN_ST_W1, 2u << 22, EN_D, EN_F, EN_W1, EN_W1_S);
  write_weight(EN_ST_W2, 3u << 22, EN_F, EN_D, EN_W2, EN_W2_S);
#endif
#ifdef EN_HASH_OUT
  fill_outputs();
#endif
  uint64_t total = 0;
  total += run_stage_timed(EN_STAGE_LNQ1, "lnq1", &pass);
  QKV_TRACK("lnq1");
#ifdef EN_FUSED_QKVSPLIT   // qkv never reaches DRAM: the "qkv" check below reads stale data
  total += run_stage_timed(EN_STAGE_QKVSPLIT, "qkv_split_fused", &pass);
#else
  total += run_stage_timed(EN_STAGE_QKV, "qkv_gemm", &pass);
  QKV_TRACK("qkv_gemm");
  total += run_stage_timed(EN_STAGE_SPLIT, "split", &pass);
  QKV_TRACK("split");
#endif
  set_fa_args(0);   // one launch covers all heads (fa_stage.cpp FA_ENC_TPH)
  total += run_stage_timed(EN_STAGE_FA, "attention", &pass);
  QKV_TRACK("attention");
#ifndef EN_FA_OQ   // EN_FA_OQ: attention already wrote the out-projection's A operand
  total += run_stage_timed(EN_STAGE_OQ, "o_quant", &pass);
  QKV_TRACK("o_quant");
#endif
#ifdef EN_FUSED_OPROJRES   // att never reaches DRAM: the "att" check below reads stale data
  total += run_stage_timed(EN_STAGE_OPROJRES, "oproj_res_fused", &pass);
  total += run_stage_timed(EN_STAGE_LNQ2, "lnq2", &pass);
#else
  total += run_stage_timed(EN_STAGE_OPROJ, "oproj_gemm", &pass);
  QKV_TRACK("oproj_gemm");
#ifdef EN_OPROJ_RESACC   // the out-projection wrote x1 = x + att itself; att is not written
  total += run_stage_timed(EN_STAGE_LNQ2, "lnq2", &pass);
  QKV_TRACK("lnq2");
#else
  total += run_stage_timed(EN_STAGE_RES1LN, "res1_lnq2", &pass);
  QKV_TRACK("res1_lnq2");
#endif
#endif
#ifdef EN_FUSED_UPGELU   // ff never reaches DRAM: the "gelu" check below reads stale data
#if defined(EN_FX_PROF) && defined(EN_FX_OVL2)
  for (uint32_t j = 0; j < 12u; j++) *(volatile uint64_t *)RAD_PB_HOST(0, EN_PB_UPROBE + j) = 0;
#endif
  total += run_stage_timed(EN_STAGE_UPGELU, "up_gelu_fused", &pass);
#if defined(EN_FX_PROF) && defined(EN_FX_OVL2)
  { const char *nm[12] = {"t0_issue", "t0_wait_b5", "t0_drain_scl", "t0_wait_b4", "w0_wait_b4", "w0_rounds", "w0_scale_fix", "w0_wait_b5",
                           "w0_rounds_only", "unused", "unused", "w0_scale_fix"};
    rad_puts("ovl2_prof:"); for (uint32_t j = 0; j < 12u; j++) { rad_puts(" "); rad_puts(nm[j]); rad_puts("="); rad_putu(rad_host_postbox(EN_PB_UPROBE + j)); } rad_puts("\n"); }
#endif
  QKV_TRACK("up_gelu_fused");
#else
  total += run_stage_timed(EN_STAGE_UP, "up_gemm", &pass);
  QKV_TRACK("up_gemm");
  total += run_stage_timed(EN_STAGE_GELUQ, "gelu_quant", &pass);
  QKV_TRACK("gelu_quant");
#endif
#ifdef EN_FUSED_DOWNRES   // ff2 never reaches DRAM: the "ff2" check below reads stale data
  total += run_stage_timed(EN_STAGE_DOWNRES, "down_res_fused", &pass);
#else
  total += run_stage_timed(EN_STAGE_DOWN, "down_gemm", &pass);
  QKV_TRACK("down_gemm");
#ifndef EN_DOWN_RESACC   // EN_DOWN_RESACC: the down GEMM writes y = x1 + ff2 itself; ff2 is not written
  total += run_stage_timed(EN_STAGE_RES2, "res2", &pass);
  QKV_TRACK("res2");
#endif
#endif
  rad_puts("layer gpu_cycles="); rad_putu(total);
  rad_puts(" ideal_mesh_cycles="); rad_putu(EN_IDEAL_MESH_CYCLES);
  // utilization against every mesh that runs: EN_CLUSTERS meshes x elapsed cycles
  rad_puts(" meshes="); rad_putu(EN_CLUSTERS);
  rad_puts(" layer_mesh_util="); put_pct(EN_IDEAL_MESH_CYCLES, total * EN_CLUSTERS); rad_puts("\n");
#ifdef EN_HASH_OUT
  hash_outputs();
#endif
  check_bf16("qkv", EN_QKV, 3 * EN_D, ref_full_qkv, REF_QKV_MODEL_PPM);
  check_o_full();
  check_bf16("att", EN_ATT, EN_D, ref_full_att, REF_ATT_MODEL_PPM);
  check_bf16("x1", EN_X1, EN_D, ref_full_x1, REF_X1_MODEL_PPM);
  check_bf16("gelu", EN_FF, EN_F, ref_full_g, REF_G_MODEL_PPM);
  check_bf16("ff2", EN_FF2, EN_D, ref_full_ff2, REF_FF2_MODEL_PPM);
  pass = check_bf16_limit("y", EN_Y, EN_D, ref_full_y, REF_Y_MODEL_PPM * 3 / 2) && pass;
#elif EN_TEST == 25
  // metasim reproducer for the attention output that depends on the attention code's address
  // (memory note fa-output-depends-on-placement; tp_enc_hzm2 wrong vs tp_enc_hzm2a right on the
  // board): x and W_qkv generated on the GPU, LN1, fused QKV split, attention, then oa's hash
  run_stage(EN_STAGE_GENX, "genx");
  gen_weight(EN_ST_WQKV, EN_D * 3 * EN_D, EN_WQKV, EN_WQKV_S);
#ifdef EN_T25_FILLALL   // poison every layer output buffer, as test 4 does
  fill_outputs();
#else
  {
    volatile ENArgs *a = (volatile ENArgs *)rad_gpu_ptr(EN_ARGS);
    a->gen_addr = EN_OA; a->gen_elems = EN_S * EN_D; a->gen_stream = 0x5Au;
    run_stage(EN_STAGE_FILL, "fill");
  }
#endif
  run_stage(EN_STAGE_LNQ1, "lnq1");
  run_stage(EN_STAGE_QKVSPLIT, "qkv_split_fused");
  set_fa_args(0);
  run_stage(EN_STAGE_FA, "attention");
  {
    volatile ENArgs *a = (volatile ENArgs *)rad_gpu_ptr(EN_ARGS);
    for (uint32_t r = 0; r < 2u; r++) {   // oa, then its scales
      a->gen_addr = r ? EN_OA_S : EN_OA; a->gen_elems = r ? EN_S * EN_D / 32 : EN_S * EN_D;
      run_stage(EN_STAGE_HASH, "hash");
      const volatile uint32_t *hw = (const volatile uint32_t *)rad_gpu_ptr(EN_HASH_OUT);
      uint32_t h = 0;
      for (uint32_t i = 0; i < 16u * 3u * 2u * EN_CLUSTERS; i++) h += hw[i];
      rad_puts(r ? "hash oas=" : "hash oa="); rad_putx(h); rad_puts("\n");
    }
  }
#elif EN_TEST == 18
  // metasim: a fused stage alone on whatever DRAM holds (hang hunting with the Muon trace)
#ifndef EN_T18_STAGE
#define EN_T18_STAGE EN_STAGE_QKVSPLIT
#endif
#ifdef EN_FX_PROF
  for (uint32_t j = 0; j < 5u; j++) *(volatile uint64_t *)RAD_PB_HOST(0, EN_PB_UPROBE + j) = 0;
#endif
#if defined(EN_FX_HPM) || defined(EN_STAGE_HPM)
  for (uint32_t i = 0; i < 2u * 2u * 2u * 32u; i++) ((volatile uint32_t *)rad_gpu_ptr(EN_HPM))[i] = 0u;
#endif
  if (EN_T18_STAGE == EN_STAGE_FA) set_fa_args(0);   // attention reads its args block
  run_stage(EN_T18_STAGE, "fused_alone");
#if defined(EN_FX_HPM) || defined(EN_STAGE_HPM)
  {   // per cluster / core: counter deltas over the first warp's epilogue window of one tile
    static const char *nm[EN_HPM_N] = {"cyc", "inst", "decoded", "dispatched", "eligible", "issued",
      "w0_waw", "w0_war", "w0_sb", "w0_rsfull", "w0_busy", "w0_busylsu", "w1_waw", "w1_war", "w1_sb",
      "w1_rsfull", "w1_busy", "w1_busylsu", "w2_waw", "w2_war", "w2_sb", "w2_rsfull", "w2_busy", "w2_busylsu"};
    const volatile uint32_t *h = (const volatile uint32_t *)rad_gpu_ptr(EN_HPM);
    for (uint32_t cc = 0; cc < 2u * 2u; cc++) {
      rad_puts("hpm cl"); rad_putu(cc / 2u); rad_puts(" core"); rad_putu(cc % 2u); rad_puts(":");
      for (uint32_t j = 0; j < EN_HPM_N; j++) {
        rad_puts(" "); rad_puts(nm[j]); rad_puts("="); rad_putu(h[(cc * 2u + 1u) * 32u + j] - h[(cc * 2u) * 32u + j]);
      }
      rad_puts("\n");
    }
    rad_flush();
  }
#endif
#ifdef EN_FX_PROF
  { const char *nm[5] = {"ksteps", "wait_b4", "work0", "wait_b5", "out"}; rad_puts("fx_prof warp0:"); for (uint32_t j = 0; j < 5u; j++) { rad_puts(" "); rad_puts(nm[j]); rad_puts("="); rad_putu(rad_host_postbox(EN_PB_UPROBE + j)); } rad_puts("\n"); }
#endif
#elif EN_TEST == 17
  // fused QKV GEMM + split (EN_STAGE_QKVSPLIT) vs the two-stage path on real layer data: every
  // 4 KB block of the six FA operand buffers must hash the same
  {
    static const uint32_t buf[6] = {EN_FA_Q, EN_FA_QS, EN_FA_KT, EN_FA_KTS, EN_FA_V, EN_FA_VS};
    static const uint32_t len[6] = {EN_H * EN_S * EN_DH, EN_H * EN_S * EN_DH / 32, EN_H * EN_S * EN_DH,
                                    EN_H * EN_S * EN_DH / 32, EN_H * EN_S * EN_DH, EN_H * EN_S * EN_DH / 32};
    static uint64_t hs[6][32];
#ifdef EN_GEN_X
    run_stage(EN_STAGE_GENX, "genx");
    gen_weight(EN_ST_WQKV, EN_D * 3 * EN_D, EN_WQKV, EN_WQKV_S);
#endif
    uint64_t total = 0;
    run_stage(EN_STAGE_LNQ1, "lnq1");
    total += run_stage_timed(EN_STAGE_QKV, "qkv_gemm", &pass);
    total += run_stage_timed(EN_STAGE_SPLIT, "split", &pass);
    rad_puts("two-stage gpu_cycles="); rad_putu(total); rad_puts("\n");
    for (int b = 0; b < 6; b++) {
      const volatile uint64_t *m = (const volatile uint64_t *)rad_gpu_ptr(buf[b]);
      for (uint32_t k = 0; k < len[b] / 4096; k++) {
        uint64_t x = 0x9E3779B97F4A7C15ull;
        for (uint32_t i = 0; i < 512; i++) x = (x ^ m[k * 512 + i]) * 0x100000001B3ull;
        hs[b][k] = x;
      }
      volatile uint64_t *mw = (volatile uint64_t *)rad_gpu_ptr(buf[b]);
      for (uint32_t i = 0; i < len[b] / 8; i++) mw[i] = 0x5555555555555555ull;   // poison
    }
    const uint64_t fused = run_stage_timed(EN_STAGE_QKVSPLIT, "qkv_split_fused", &pass);
    uint32_t bad = 0, nblk = 0;
    for (int b = 0; b < 6; b++) {
      const volatile uint64_t *m = (const volatile uint64_t *)rad_gpu_ptr(buf[b]);
      for (uint32_t k = 0; k < len[b] / 4096; k++, nblk++) {
        uint64_t x = 0x9E3779B97F4A7C15ull;
        for (uint32_t i = 0; i < 512; i++) x = (x ^ m[k * 512 + i]) * 0x100000001B3ull;
        if (x != hs[b][k]) { bad++; if (bad <= 4) { rad_puts(" bad buf="); rad_putu(b); rad_puts(" blk="); rad_putu(k); } }
      }
    }
    rad_puts("\ncheck fused_split_vs_two_stage: bad_blocks="); rad_putu(bad); rad_puts("/"); rad_putu(nblk);
    rad_puts(" fused_cycles="); rad_putu(fused); rad_puts("\n");
    pass = bad == 0 && pass;
  }
#elif EN_TEST == 16
  // fused FFN-up + GELU (EN_STAGE_UPGELU) vs the two-stage path on real layer data: G codes and
  // scales of the sampled rows must be identical (same bf16 C, same GELU, same quantizer)
  {
    static uint8_t g_ref[EN_CHECK_NROWS * EN_F], s_ref[EN_CHECK_NROWS * (EN_F / 32)];
#ifdef EN_GEN_X
    run_stage(EN_STAGE_GENX, "genx");
    gen_weight(EN_ST_WQKV, EN_D * 3 * EN_D, EN_WQKV, EN_WQKV_S);
    gen_weight(EN_ST_WO, EN_D * EN_D, EN_WO, EN_WO_S);
    gen_weight(EN_ST_W1, EN_D * EN_F, EN_W1, EN_W1_S);
#else
    write_weight(EN_ST_WO, 1u << 22, EN_D, EN_D, EN_WO, EN_WO_S);
    write_weight(EN_ST_W1, 2u << 22, EN_D, EN_F, EN_W1, EN_W1_S);
#endif
    uint64_t total = 0;
    run_stage(EN_STAGE_LNQ1, "lnq1");
    run_stage(EN_STAGE_QKV, "qkv_gemm");
    run_stage(EN_STAGE_SPLIT, "split");
    set_fa_args(0);
    run_stage(EN_STAGE_FA, "attention");
    run_stage(EN_STAGE_OQ, "o_quant");
    run_stage(EN_STAGE_OPROJ, "oproj_gemm");
    run_stage(EN_STAGE_RES1LN, "res1_lnq2");
    total += run_stage_timed(EN_STAGE_UP, "up_gemm", &pass);
    total += run_stage_timed(EN_STAGE_GELUQ, "gelu_quant", &pass);
    rad_puts("two-stage gpu_cycles="); rad_putu(total); rad_puts("\n");
    const volatile uint8_t *G = (const volatile uint8_t *)rad_gpu_ptr(EN_G);
    const volatile uint8_t *GS = (const volatile uint8_t *)rad_gpu_ptr(EN_G_S);
    for (uint32_t s = 0; s < EN_CHECK_NROWS; s++) {
      const uint32_t r = ref_rows[s];
      for (uint32_t c = 0; c < EN_F; c++) g_ref[s * EN_F + c] = G[r * EN_F + c];
      for (uint32_t g = 0; g < EN_F / 32; g++) s_ref[s * (EN_F / 32) + g] = GS[a_off(r, g, EN_F)];
    }
    // poison the sampled rows so a fused stage that writes nothing cannot pass
    volatile uint8_t *Gw = (volatile uint8_t *)rad_gpu_ptr(EN_G);
    volatile uint8_t *GSw = (volatile uint8_t *)rad_gpu_ptr(EN_G_S);
    for (uint32_t s = 0; s < EN_CHECK_NROWS; s++) {
      const uint32_t r = ref_rows[s];
      for (uint32_t c = 0; c < EN_F; c += 8) *(volatile uint64_t *)(Gw + r * EN_F + c) = 0x7F7F7F7F7F7F7F7Full;
      for (uint32_t g = 0; g < EN_F / 32; g++) GSw[a_off(r, g, EN_F)] = 0x55;
    }
    #ifdef EN_FX_PROF
    for (uint32_t j = 0; j < 5u; j++) *(volatile uint64_t *)RAD_PB_HOST(0, EN_PB_UPROBE + j) = 0;
    #endif
    const uint64_t fused = run_stage_timed(EN_STAGE_UPGELU, "up_gelu_fused", &pass);
    #ifdef EN_FX_PROF
    { const char *nm[5] = {"ksteps", "wait_b4", "work0", "wait_b5", "out"}; rad_puts("fx_prof warp0:"); for (uint32_t j = 0; j < 5u; j++) { rad_puts(" "); rad_puts(nm[j]); rad_puts("="); rad_putu(rad_host_postbox(EN_PB_UPROBE + j)); } rad_puts("\n"); }
    #endif
    uint32_t cbad = 0, sbad = 0;
    for (uint32_t s = 0; s < EN_CHECK_NROWS; s++) {
      const uint32_t r = ref_rows[s];
      for (uint32_t c = 0; c < EN_F; c++) cbad += G[r * EN_F + c] != g_ref[s * EN_F + c];
      for (uint32_t g = 0; g < EN_F / 32; g++) sbad += GS[a_off(r, g, EN_F)] != s_ref[s * (EN_F / 32) + g];
    }
    rad_puts("check fused_vs_two_stage: code_mismatch="); rad_putu(cbad); rad_puts("/"); rad_putu(EN_CHECK_NROWS * EN_F);
    rad_puts(" scale_mismatch="); rad_putu(sbad); rad_puts("/"); rad_putu(EN_CHECK_NROWS * EN_F / 32);
    rad_puts(" fused_cycles="); rad_putu(fused); rad_puts("\n");
    pass = cbad == 0 && sbad == 0 && pass;
  }
#elif EN_TEST == 3
  // split + attention (one launch, all heads) on a synthetic qkv
  write_qkv_synthetic();
  pass = run_stage(EN_STAGE_SPLIT, "split") && pass;
  pass = check_split() && pass;
  set_fa_args(0);   // one launch covers all heads
  pass = run_stage(EN_STAGE_FA, "fa") && pass;
  for (uint32_t h = 0; h < EN_H; h++)
    pass = check_bf16_frob("fa_o", EN_FA_O + h * EN_S * EN_DH * 2, EN_DH, ref_fa_o + h * REF_NROWS * EN_DH,
                           120000u) && pass;   // board = metasim enc_m14 per head (67k-107k)
#elif EN_TEST == 6
  // FA state dependence: per head, SMEM cleared then FA, then FA again without clearing
  write_qkv_synthetic();
  pass = run_stage(EN_STAGE_SPLIT, "split") && pass;
  pass = check_split() && pass;
  for (uint32_t h = 0; h < EN_H; h++) {
    set_fa_args(h);
    run_stage(EN_STAGE_CLEAR, "clear");
    run_stage(EN_STAGE_FA, "fa_after_clear");
    pass = check_bf16_frob("fa_o", EN_FA_O + h * EN_S * EN_DH * 2, EN_DH, ref_fa_o + h * REF_NROWS * EN_DH,
                           120000u) && pass;
    run_stage(EN_STAGE_FA, "fa_again");
    pass = check_bf16_frob("fa_o", EN_FA_O + h * EN_S * EN_DH * 2, EN_DH, ref_fa_o + h * REF_NROWS * EN_DH,
                           120000u) && pass;
  }
#elif EN_TEST == 7
  // metasim: FA state dependence with GPU-generated qkv; head 0 and head 2, clear then twice
  run_stage(EN_STAGE_GENQKV, "genqkv");
  pass = run_stage(EN_STAGE_SPLIT, "split") && pass;
  pass = check_split() && pass;
  for (uint32_t h = 0; h < EN_H; h += 2) {
    set_fa_args(h);
    run_stage(EN_STAGE_CLEAR, "clear");
    run_stage(EN_STAGE_FA, "fa_after_clear");
    pass = check_bf16_frob("fa_o", EN_FA_O + h * EN_S * EN_DH * 2, EN_DH, ref_fa_o + h * REF_NROWS * EN_DH,
                           120000u) && pass;
    run_stage(EN_STAGE_FA, "fa_again");
    pass = check_bf16_frob("fa_o", EN_FA_O + h * EN_S * EN_DH * 2, EN_DH, ref_fa_o + h * REF_NROWS * EN_DH,
                           120000u) && pass;
  }
#elif EN_TEST == 24
  // SIMT throughput probe: cycles for 2048 instructions per warp, all warps at once
  for (uint32_t s = 0; s < 13u; s++) *(volatile uint64_t *)RAD_PB_HOST(0, EN_PB_UPROBE + s) = 0;
  pass = run_stage(EN_STAGE_UPROBE, "uprobe") && pass;
  {
    const char *names[12] = {"fmadd.s", "fmadd.h", "xor", "smem_lw", "fmadd.s_chain x2048",
                             "fmadd.h_chain x2048", "smem_lw_chain x512", "global_lw_chain x512",
                             "fnexp.h x2048", "fdiv.h x2048", "fnexp.h_chain x512", "fdiv.h_chain x512"};
    for (uint32_t s = 0; s < 12u; s++) {
      rad_puts("uprobe "); rad_puts(names[s]); rad_puts(" cycles="); rad_putu(rad_host_postbox(EN_PB_UPROBE + s)); rad_puts("\n");
    }
  }
#elif EN_TEST == 23
  // metasim: the attention launch alone on whatever DRAM holds (hang hunting with the Muon trace)
  set_fa_args(0);
#ifdef EN_FA_MARKS   // FA's mcycle stamps, per cluster at 0x1F100000 + 0x800 c, cleared first
  for (uint32_t cl = 0; cl < EN_CLUSTERS; cl++) {
    volatile uint32_t *mk = (volatile uint32_t *)rad_gpu_ptr(0x1F100000u + 0x800u * cl);
    for (uint32_t i = 0; i < 512; i++) mk[i] = 0;
  }
#endif
  run_stage(EN_STAGE_FA, "fa_alone");
#ifdef EN_FA_MARKS
  for (uint32_t cl = 0; cl < EN_CLUSTERS; cl++) {
    const volatile uint32_t *mk = (const volatile uint32_t *)rad_gpu_ptr(0x1F100000u + 0x800u * cl);
    const uint32_t t0 = rad_host_postbox_cl(cl, EN_PB_T_START), t1 = rad_host_postbox_cl(cl, EN_PB_T_END);
    rad_puts("fa_marks cl="); rad_putu(cl); rad_puts(": start->m0="); rad_putu(mk[0] - t0);
    uint32_t last = mk[0];
    for (uint32_t i = 1; i < 512 && mk[i]; i++) { rad_puts(" "); rad_putu(mk[i] - mk[i - 1]); last = mk[i]; }
    rad_puts(" last->end="); rad_putu(t1 - last); rad_puts("\n");
    rad_flush();
  }
#endif
#elif EN_TEST == 22
  // 4-block vs 1-block move-in layout probe (kernel dma_probe)
  {
    volatile uint64_t *src = (volatile uint64_t *)rad_gpu_ptr(EN_PROBE_SRC);
    volatile uint64_t *dw = (volatile uint64_t *)rad_gpu_ptr(EN_PROBE_W);
    volatile uint64_t *dn = (volatile uint64_t *)rad_gpu_ptr(EN_PROBE_N);
    volatile uint64_t *d2 = (volatile uint64_t *)rad_gpu_ptr(EN_PROBE_2);
    for (uint32_t i = 0; i < 2048u / 8u; i++) {
      uint64_t v = 0;
      for (uint32_t j = 0; j < 8u; j++) v |= (uint64_t)((((8u * i + j) * 37u) >> 3 ^ (8u * i + j)) & 0xFFu) << (8u * j);
      src[i] = v; dw[i] = 0; dn[i] = 0; d2[i] = 0;
    }
    for (uint32_t cl = 0; cl < EN_CLUSTERS; cl++) { volatile uint32_t *tp = (volatile uint32_t *)rad_gpu_ptr(EN_PROBE_T + 16u * cl); tp[0] = tp[1] = tp[2] = tp[3] = 0; }
    { volatile uint64_t *dm = (volatile uint64_t *)rad_gpu_ptr(EN_PROBE_M); for (uint32_t i = 0; i < 2048u / 8u; i++) dm[i] = 0; }
    pass = run_stage(EN_STAGE_DMAPROBE, "dmaprobe") && pass;
    const volatile uint8_t *s8 = (const volatile uint8_t *)rad_gpu_ptr(EN_PROBE_SRC);
    const volatile uint8_t *w8 = (const volatile uint8_t *)rad_gpu_ptr(EN_PROBE_W);
    const volatile uint8_t *n8 = (const volatile uint8_t *)rad_gpu_ptr(EN_PROBE_N);
    const volatile uint8_t *t8 = (const volatile uint8_t *)rad_gpu_ptr(EN_PROBE_2);
    uint32_t bad_w = 0, bad_n = 0, bad_2 = 0;
    for (uint32_t k = 0; k < 8u; k++)
      for (uint32_t r = 0; r < 16u; r++)
        for (uint32_t j = 0; j < 16u; j++) {
          const uint32_t off = (16u * k + r) * 16u + j;
          const uint8_t want = s8[128u * r + 16u * k + j];
          if (n8[off] != want) bad_n++;
          if (t8[off] != want) bad_2++;
          if (w8[off] != want) {
            if (bad_w < 8u) {
              rad_puts("  wide k="); rad_putu(k); rad_puts(" r="); rad_putu(r); rad_puts(" j="); rad_putu(j);
              rad_puts(" got="); rad_putu(w8[off]); rad_puts(" want="); rad_putu(want); rad_puts("\n");
            }
            bad_w++;
          }
        }
    rad_puts("dmaprobe: narrow_bad="); rad_putu(bad_n); rad_puts(" wide4_bad="); rad_putu(bad_w); rad_puts(" wide2_bad="); rad_putu(bad_2); rad_puts(" of 2048\n");
    // where did the wide rows land?  first 4 words of SMEM-image rows 0, 1, 16, 17
    const uint32_t rows[4] = {0u, 1u, 16u, 17u};
    for (uint32_t i = 0; i < 4u; i++) {
      rad_puts("  wide row "); rad_putu(rows[i]); rad_puts(":");
      for (uint32_t j = 0; j < 16u; j++) { rad_puts(" "); rad_putu(w8[16u * rows[i] + j]); }
      rad_puts("\n");
    }
    {
      for (uint32_t cl = 0; cl < EN_CLUSTERS; cl++) {
        const volatile uint32_t *tp = (const volatile uint32_t *)rad_gpu_ptr(EN_PROBE_T + 16u * cl);
        rad_puts("dmaprobe 64KB c"); rad_putu(cl); rad_puts(": mvin_1blk_cycles="); rad_putu(tp[0]); rad_puts(" mvin_2blk_cycles="); rad_putu(tp[1]);
        rad_puts(" mvout_cycles="); rad_putu(tp[2]); rad_puts(" mvout_2blk_cycles="); rad_putu(tp[3]); rad_puts("\n");
      }
      const volatile uint8_t *m8 = (const volatile uint8_t *)rad_gpu_ptr(EN_PROBE_M);
      uint32_t bad_m = 0;
      for (uint32_t i = 0; i < 2048u; i++) {
        if (m8[i] != s8[i]) {
          if (bad_m < 6u) { rad_puts("  mvout2 off="); rad_putu(i); rad_puts(" got="); rad_putu(m8[i]); rad_puts(" want="); rad_putu(s8[i]); rad_puts("\n"); }
          bad_m++;
        }
      }
      rad_puts("dmaprobe: mvout2_bad="); rad_putu(bad_m); rad_puts(" of 2048\n");
    }
    pass = pass && bad_2 == 0u && bad_n == 0u;   // 4-block is known to fail (cols field is 6 bits)
  }
#elif EN_TEST == 14
  // metasim: split, a QKV GEMM (on whatever h1 holds; it only precedes FA), then ONE FA launch for
  // all heads, checked per head against the synthetic-qkv reference
  run_stage(EN_STAGE_GENQKV, "genqkv");
  pass = run_stage(EN_STAGE_SPLIT, "split") && pass;
  pass = check_split() && pass;
  run_stage(EN_STAGE_QKV, "qkv_gemm_before_fa");
  set_fa_args(0);
#ifdef EN_FA_MARKS   // FA's per-tile mcycle stamps (FA_SP_MARK1: one per tile at 0x1F100000), cleared first
  {
    volatile uint32_t *mk = (volatile uint32_t *)rad_gpu_ptr(0x1F100000u);
    for (uint32_t i = 0; i < 160; i++) mk[i] = 0;
  }
#endif
  pass = run_stage(EN_STAGE_FA, "fa_all_heads") && pass;
#ifdef EN_FA_MARKS
  {
    const volatile uint32_t *mk = (const volatile uint32_t *)rad_gpu_ptr(0x1F100000u);
    const uint32_t t0 = rad_host_postbox(EN_PB_T_START), t1 = rad_host_postbox(EN_PB_T_END);
    rad_puts("fa_marks: start->m0="); rad_putu(mk[0] - t0);
    for (uint32_t i = 1; i < 160 && mk[i]; i++) { rad_puts(" d"); rad_putu(i); rad_puts("="); rad_putu(mk[i] - mk[i - 1]); }
    uint32_t last = 0;
    for (uint32_t i = 0; i < 160 && mk[i]; i++) last = mk[i];
    rad_puts(" last->end="); rad_putu(t1 - last); rad_puts("\n");
  }
#endif
  for (uint32_t h = 0; h < EN_H; h++)
    pass = check_bf16_frob("fa_o", EN_FA_O + h * EN_S * EN_DH * 2, EN_DH, ref_fa_o + h * REF_NROWS * EN_DH,
                           120000u) && pass;
#elif EN_TEST == 13
  // metasim: every DMA SIMT stage on GPU-generated inputs, against gen_ref.py
  {
    volatile ENArgs *a = (volatile ENArgs *)rad_gpu_ptr(EN_ARGS);
    a->gen_addr = EN_FF; a->gen_elems = EN_S * EN_F; a->gen_stream = EN_ST_GEN_FF;
    run_stage(EN_STAGE_GEN, "gen_ff");
    pass = run_stage(EN_STAGE_GELUQ, "gelu_quant") && pass;
    pass = check_bf16_limit("gelu_bf16", EN_FF, EN_F, ref_stg_g, 20000u) && pass;
    check_a_operand("g", EN_G, EN_G_S, EN_F, ref_stg_g_codes, ref_stg_g_scales, 20);   // info only
    pass = check_a_frob("g", EN_G, EN_G_S, EN_F, ref_stg_g, ref_stg_g_codes, ref_stg_g_scales) && pass;
    a->gen_addr = EN_FA_O; a->gen_elems = EN_H * EN_S * EN_DH; a->gen_stream = EN_ST_GEN_O;
    run_stage(EN_STAGE_GEN, "gen_o");
    pass = run_stage(EN_STAGE_OQ, "o_quant") && pass;
    pass = check_a_operand("oa", EN_OA, EN_OA_S, EN_D, ref_stg_oa_codes, ref_stg_oa_scales) && pass;
    run_stage(EN_STAGE_GENX, "genx");
    a->gen_addr = EN_ATT; a->gen_elems = EN_S * EN_D; a->gen_stream = EN_ST_GEN_ATT;
    run_stage(EN_STAGE_GEN, "gen_att");
#if defined(EN_LN_PROF) || defined(EN_RND_PROF)
    for (uint32_t j = 0; j < 6u; j++) *(volatile uint64_t *)RAD_PB_HOST(0, EN_PB_UPROBE + j) = 0;
#endif
    pass = run_stage(EN_STAGE_RES1LN, "res1_lnq2") && pass;
#ifdef EN_RND_PROF
    rad_puts("rnd_prof res1_lnq2 work+fence per warp gw0 0..5:");
    for (uint32_t j = 0; j < 6u; j++) { rad_puts(" "); rad_putu(rad_host_postbox(EN_PB_UPROBE + j)); }
    rad_puts("\n");
#endif
#ifdef EN_LN_PROF
    {
      const char *nm[5] = {"res_add", "stats", "reduce", "pass1", "pass2"};
      rad_puts("ln_prof worker0:");
      for (uint32_t j = 0; j < 5u; j++) { rad_puts(" "); rad_puts(nm[j]); rad_puts("="); rad_putu(rad_host_postbox(EN_PB_UPROBE + j)); }
      rad_puts("\n");
    }
#endif
    pass = check_bf16_limit("x1", EN_X1, EN_D, ref_stg_x1, 0u) && pass;
    check_a_operand("h2", EN_H2, EN_H2_S, EN_D, ref_stg_h2_codes, ref_stg_h2_scales, 1000);   // info
    pass = check_a_frob("h2", EN_H2, EN_H2_S, EN_D, ref_stg_h2_y, ref_stg_h2_codes, ref_stg_h2_scales) && pass;
    a->gen_addr = EN_FF2; a->gen_elems = EN_S * EN_D; a->gen_stream = EN_ST_GEN_FF2;
    run_stage(EN_STAGE_GEN, "gen_ff2");
    pass = run_stage(EN_STAGE_RES2, "res2") && pass;
#ifdef EN_Y_DEBUG   // y read 3 times ~2 s apart: first words vs ref, and a checksum of all of y
    for (int k = 0; k < 3; k++) {
      const volatile uint16_t *yy = (const volatile uint16_t *)rad_gpu_ptr(EN_Y);
      const volatile uint64_t *y8 = (const volatile uint64_t *)rad_gpu_ptr(EN_Y);
      uint64_t h = 0;
      for (uint32_t i = 0; i < EN_S * EN_D / 4; i++) h = (h ^ y8[i]) * 0x100000001B3ull;
      rad_puts("ydbg read="); rad_putu(k); rad_puts(" hash="); rad_putx((uint32_t)(h >> 32)); rad_putx((uint32_t)h);
      rad_puts(" y[0..7]=");
      for (int i = 0; i < 8; i++) { rad_putx(yy[i]); rad_puts(","); }
      rad_puts(" ref=");
      for (int i = 0; i < 8; i++) { rad_putx(ref_stg_y[i]); rad_puts(","); }
      rad_puts(" row64[0..3]=");
      for (int i = 0; i < 4; i++) { rad_putx(yy[64 * EN_D + i]); rad_puts(","); }
      rad_puts("\n");
      rad_flush();
      rad_wait_cycles(EN_RECHECK_WAIT);
    }
#endif
    pass = check_bf16_limit("y", EN_Y, EN_D, ref_stg_y, 0u) && pass;
  }
#elif EN_TEST == 12
  // metasim: which SMEM range must be cleared between two FA launches (assertion test, no data)
  set_fa_args(0);
  run_stage(EN_STAGE_FA, "fa_first");
  {
    volatile ENArgs *a = (volatile ENArgs *)rad_gpu_ptr(EN_ARGS);
    a->clear_lo = EN_CLEAR_LO;
    a->clear_hi = EN_CLEAR_HI;
  }
#ifdef EN_GEMM_BETWEEN
  run_stage(EN_STAGE_OPROJ, "gemm_between");
#else
  run_stage(EN_STAGE_CLEAR, "clear_part");
#endif
  run_stage(EN_STAGE_FA, "fa_second");
  rad_puts("second FA completed\n");
#elif EN_TEST == 11
  // metasim timing of the remaining SIMT stages on whatever DRAM holds (no checks)
  run_stage(EN_STAGE_OQ, "o_quant");
  run_stage(EN_STAGE_RES1LN, "res1_lnq2");
  run_stage(EN_STAGE_GELUQ, "gelu_quant");
  run_stage(EN_STAGE_RES2, "res2");
#elif EN_TEST == 10
  // metasim: FA after a no-op launch (no SMEM or Gemmini activity) vs after a clear
  run_stage(EN_STAGE_GENQKV, "genqkv");
  pass = run_stage(EN_STAGE_SPLIT, "split") && pass;
  set_fa_args(0);
  run_stage(EN_STAGE_CLEAR, "clear");
  run_stage(EN_STAGE_FA, "fa_after_clear");
  pass = check_bf16_frob("fa_o", EN_FA_O, EN_DH, ref_fa_o, 120000u) && pass;
  run_stage(0u, "noop");
  run_stage(EN_STAGE_FA, "fa_after_noop");
  pass = check_bf16_frob("fa_o", EN_FA_O, EN_DH, ref_fa_o, 120000u) && pass;
#elif EN_TEST == 9
  // metasim: split control flow only (no data generation, no check)
  pass = run_stage(EN_STAGE_SPLIT, "split") && pass;
#elif EN_TEST == 8
  // metasim: split on GPU-generated qkv, exact check
  run_stage(EN_STAGE_GENQKV, "genqkv");
  pass = run_stage(EN_STAGE_SPLIT, "split") && pass;
  pass = check_split() && pass;
#elif EN_TEST == 2
  // split stage alone, on a synthetic qkv (exact code comparison)
#ifndef EN_GEN_X   // metasim: skip the 768 KB host write (timing only)
  write_qkv_synthetic();
#endif
  pass = run_stage(EN_STAGE_SPLIT, "split") && pass;
  pass = check_split() && pass;
#elif EN_TEST == 1
  pass = run_stage(EN_STAGE_LNQ1, "lnq1") && pass;
  check_a_operand("h1", EN_H1, EN_H1_S, EN_D, ref_h1_codes, ref_h1_scales, 1000);   // info: bf16 y changes some codes
  pass = check_a_frob("h1", EN_H1, EN_H1_S, EN_D, ref_h1_y, ref_h1_codes, ref_h1_scales) && pass;
  pass = run_stage(EN_STAGE_QKV, "qkv") && pass;
  pass = check_bf16("qkv", EN_QKV, 3 * EN_D, ref_qkv, REF_QKV_PPM) && pass;
#endif
  rad_puts(pass ? "PASS\n" : "FAIL\n");
  rad_flush();
  return pass ? 0 : 1;
}
