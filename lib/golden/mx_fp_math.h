// MX floating-point primitives reused from Spike/riscv-isa-sim
// (github.com/riscv-software-src/riscv-isa-sim), BSD-3-Clause. See NOTICE.
#ifndef _GEMMINI_MX_FP_MATH_H
#define _GEMMINI_MX_FP_MATH_H

#include <cstdint>
#include <cmath>

namespace mx {

inline uint16_t f32_to_bf16_rne(float x) {
  // Canonicalize -0 to +0 to match the Python golden, which never emits BF16
  // 0x8000 (Python's matmul output uses unsigned 0).
  if (x == 0.0f) return 0;
  union { float f; uint32_t u; } v; v.f = x;
  uint32_t bits = v.u;
  if (((bits >> 23) & 0xFF) == 0xFF) {
    uint32_t bf = bits >> 16;
    if ((bits & 0x7FFFFFu) && !(bf & 0x40)) bf |= 0x40;
    return (uint16_t)(bf & 0xFFFF);
  }
  uint32_t lsb = (bits >> 16) & 1;
  uint32_t rounded = bits + 0x7FFF + lsb;
  uint16_t out = (uint16_t)((rounded >> 16) & 0xFFFF);
  // Tiny inputs that round down to zero magnitude: also canonicalize.
  if ((out & 0x7FFF) == 0) return 0;
  return out;
}

inline float bf16_to_f32(uint16_t bf) {
  union { uint32_t u; float f; } v; v.u = ((uint32_t)bf) << 16;
  return v.f;
}

inline float bf16_round(float x) { return bf16_to_f32(f32_to_bf16_rne(x)); }

inline int fp_emax(int e_bits, int m_bits) {
  int bias = (1 << (e_bits - 1)) - 1;
  if (e_bits == 4 && m_bits == 3) return bias + 1;
  if (e_bits == 2 && m_bits == 1) return (1 << e_bits) - 1 - bias;
  return bias;
}

inline float mx_product_saturate(float x, int e_bits, int m_bits) {
  int bias = (1 << (e_bits - 1)) - 1;
  bool is_mx_fp8 = (e_bits == 4 && m_bits == 3);
  int emax = is_mx_fp8 ? bias + 1 : bias;
  float scale = (float)(1 << m_bits);
  int max_mant = is_mx_fp8 ? ((1 << m_bits) - 2) : ((1 << m_bits) - 1);
  float max_normal = ldexpf(1.0f + max_mant / scale, emax);
  int sat_man = (1 << m_bits) - 2;
  float sat_val = ldexpf(1.0f + sat_man / scale, bias + 1);
  float ax = fabsf(x);
  // Mirror Python's torch.sign(-0.0) == 0.0 — canonicalize signed zero to +0.
  if (ax == 0.0f) return 0.0f;
  if (ax > max_normal) return std::signbit(x) ? -sat_val : sat_val;
  return x;
}

inline float mx_product_quantize_trunc(float x, int e_bits, int frac_bits) {
  if (!std::isfinite(x) || x == 0.0f) return mx_product_saturate(x, e_bits, frac_bits);
  int e;
  float m = frexpf(fabsf(x), &e);
  int E = e - 1;
  float scale = (float)(1 << frac_bits);
  float frac = 2.0f * m - 1.0f;
  float frac_q = floorf(frac * scale) / scale;
  float val = ldexpf(1.0f + frac_q, E);
  float out = std::signbit(x) ? -val : val;
  return mx_product_saturate(out, e_bits, frac_bits);
}

inline float fp_quantize_trunc(float x, int e_bits, int m_bits) {
  if (x == 0.0f || !std::isfinite(x)) return 0.0f;
  bool neg = std::signbit(x);
  float ax = fabsf(x);
  int E;
  frexpf(ax, &E); E -= 1;
  int bias = (1 << (e_bits - 1)) - 1;
  int emin = 1 - bias;
  bool is_mx_fp8 = (e_bits == 4 && m_bits == 3);
  bool is_fp4 = (e_bits == 2 && m_bits == 1);
  int emax = fp_emax(e_bits, m_bits);
  float subnorm_delta = ldexpf(1.0f, emin - m_bits);
  float scale = (float)(1 << m_bits);
  float v;
  if (E < emin) {
    int k = (int)floorf(ax / subnorm_delta);
    if (k > (1 << m_bits) - 1) k = (1 << m_bits) - 1;
    v = k * subnorm_delta;
  } else if (E > emax) {
    float base_max = ldexpf(1.0f, emax);
    float delta_max = base_max / scale;
    int max_mant = is_mx_fp8 ? ((1 << m_bits) - 2) : ((1 << m_bits) - 1);
    v = base_max + max_mant * delta_max;
  } else {
    float base = ldexpf(1.0f, E);
    float delta = base / scale;
    float t = (ax - base) / delta;
    int k = (int)floorf(t);
    int hi = (1 << m_bits) - 1;
    if (is_mx_fp8 && E == emax) hi = (1 << m_bits) - 2;
    if (k > hi) k = hi;
    if (k < 0) k = 0;
    v = base + k * delta;
  }
  (void)is_fp4;
  return neg ? -v : v;
}

inline uint64_t _round_div_pow2_rne_u64(uint64_t n, int shift) {
  if (shift <= 0) return n << (-shift);
  uint64_t q = n >> shift;
  uint64_t rem = n & ((1ull << shift) - 1);
  uint64_t half = 1ull << (shift - 1);
  if (rem > half || (rem == half && (q & 1))) q += 1;
  return q;
}

inline float fp_quantize_rne_scalar(float x, int e_bits, int m_bits) {
  if (std::isnan(x)) return std::nanf("");
  if (std::isinf(x)) return x;
  if (x == 0.0f) return 0.0f;
  bool neg = std::signbit(x);
  float ax = fabsf(x);

  union { float f; uint32_t u; } v32; v32.f = ax;
  uint32_t bits = v32.u;
  uint32_t mant = bits & 0x7FFFFF;
  int e_raw    = (bits >> 23) & 0xFF;
  uint64_t num;
  int exp2;
  if (e_raw == 0) {
    if (mant == 0) return neg ? -0.0f : 0.0f;
    num  = mant;
    exp2 = 1 - 127 - 23;
  } else {
    num  = (1ull << 23) | mant;
    exp2 = e_raw - 127 - 23;
  }
  int p = 63 - __builtin_clzll(num);
  int E = exp2 + p;

  int bias = (1 << (e_bits - 1)) - 1;
  int emin = 1 - bias;
  int emax = bias;

  if (E < emin) {
    int shift = (emin - m_bits) - exp2;
    uint64_t sub_sig = _round_div_pow2_rne_u64(num, shift);
    // Match Python _round_dyadic_to_scalar: underflow returns unsigned +0,
    // dropping the sign (so signed-zero never leaks into the accumulator and
    // ultimately into bf16 0x8000).
    if (sub_sig == 0) return 0.0f;
    if (sub_sig >= (1ull << m_bits)) {
      uint64_t total_sig = sub_sig;
      int E_fin = emin;
      if (total_sig >= (1ull << (m_bits + 1))) { total_sig >>= 1; E_fin += 1; }
      if (E_fin > emax) return neg ? -INFINITY : INFINITY;
      float r = ldexpf((float)total_sig, E_fin - m_bits);
      return neg ? -r : r;
    }
    float r = ldexpf((float)sub_sig, emin - m_bits);
    return neg ? -r : r;
  }

  uint64_t total_sig = _round_div_pow2_rne_u64(num, p - m_bits);
  if (total_sig >= (1ull << (m_bits + 1))) { total_sig >>= 1; E += 1; }
  if (E > emax) return neg ? -INFINITY : INFINITY;
  float r = ldexpf((float)total_sig, E - m_bits);
  return neg ? -r : r;
}

inline float fp_quantize_rne(float x, int e_bits, int m_bits) {
  if (e_bits == 8 && m_bits == 7) return bf16_round(x);
  return fp_quantize_rne_scalar(x, e_bits, m_bits);
}

inline float fp_add_exact(float x, float y, int e_bits, int m_bits) {
  if (std::isnan(x) || std::isnan(y)) return std::nanf("");
  if (std::isinf(x) || std::isinf(y)) {
    if (std::isinf(x) && std::isinf(y) && std::signbit(x) != std::signbit(y)) return std::nanf("");
    return std::isinf(x) ? x : y;
  }
  if (x == 0.0f) return y;
  if (y == 0.0f) return x;
  float s = x + y;
  return fp_quantize_rne_scalar(s, e_bits, m_bits);
}

inline float bf16_accum_add(float x, float y) {
  return fp_add_exact(bf16_round(x), bf16_round(y), 8, 7);
}

// Banker's rounding (round-half-to-even) matching Python 3's int(round(x)).
inline int round_half_to_even(float x) {
  float fl = floorf(x);
  float frac = x - fl;
  int fli = (int)fl;
  if (frac < 0.5f) return fli;
  if (frac > 0.5f) return fli + 1;
  return (fli & 1) ? fli + 1 : fli;
}

inline uint8_t fp8_e4m3_to_code(float v) {
  if (v == 0.0f || !std::isfinite(v)) return 0;
  int s = std::signbit(v) ? 1 : 0;
  float av = fabsf(v);
  int E = (int)floorf(log2f(av));
  int bias = 7;
  int emin = -6;
  int emax = 8;
  if (E < emin) return 0;
  int E_used = E, mant;
  if (E > emax) { E_used = emax; mant = 6; }
  else {
    float base = ldexpf(1.0f, E_used);
    float delta = base / 8.0f;
    int k = round_half_to_even((av - base) / delta);
    if (k >= 8) { E_used += 1; k = 0; if (E_used > emax) { E_used = emax; k = 6; } }
    else { int hi = (E_used == emax) ? 6 : 7; if (k > hi) k = hi; if (k < 0) k = 0; }
    mant = k;
  }
  return (uint8_t)((s << 7) | (((E_used + bias) & 0xF) << 3) | (mant & 0x7));
}

inline float fp8_e4m3_decode(uint8_t code) {
  int s = (code >> 7) & 1;
  int e = (code >> 3) & 0xF;
  int m = code & 0x7;
  int bias = 7;
  float val;
  if (e == 0) val = (m / 8.0f) * ldexpf(1.0f, 1 - bias);
  else val = (1.0f + m / 8.0f) * ldexpf(1.0f, e - bias);
  return s ? -val : val;
}

inline float fp4_e2m1_decode(uint8_t code) {
  int s = (code >> 3) & 1;
  int e = (code >> 1) & 0x3;
  int m = code & 0x1;
  float val;
  if (e == 0) val = (m / 2.0f);
  else val = (1.0f + m / 2.0f) * ldexpf(1.0f, e - 1);
  return s ? -val : val;
}

inline uint8_t fpe8m0_decode_exp(uint8_t code) { return code; }
inline float fpe8m0_decode(uint8_t code) {
  if (code == 0xFF) return std::nanf("");
  return ldexpf(1.0f, (int)code - 127);
}

// Decode a 6-bit FP6 E3M2 code (1 sign | 3 exp | 2 mant, bias=3) to float.
// Subnormal (exp=0): value = mant * 2^(emin-mant_bits) = mant * 2^-4.
inline float fp6_e3m2_decode(uint8_t code) {
  int s = (code >> 5) & 1;
  int e = (code >> 2) & 0x7;
  int m = code & 0x3;
  float val;
  if (e == 0) val = m * 0.0625f;
  else        val = (1.0f + m * 0.25f) * ldexpf(1.0f, e - 3);
  return s ? -val : val;
}

// BF16 bits -> E4M2 float (RNE), matches lut_mapping_demo._bf16_to_e4m2_rne.
inline float bf16_bits_to_e4m2_rne(uint16_t bits) {
  int sign_bit = (bits >> 15) & 1;
  int E = (bits >> 7) & 0xFF;
  int M = bits & 0x7F;
  if (E == 0) return 0.0f;
  if (E == 255) return sign_bit ? -INFINITY : INFINITY;
  int e = E - 127;
  float sign_f = sign_bit ? -1.0f : 1.0f;
  if (e >= -6 && e <= 7) {
    int q = (M >> 5) & 3;
    int r = (M >> 4) & 1;
    int sticky = (M & 0xF) != 0;
    int lsb = (M >> 5) & 1;
    int round_up = r & (sticky | lsb);
    int sig_rounded = q + round_up;
    int carry = sig_rounded >= 4;
    int mant_out = carry ? 0 : sig_rounded;
    int exp_out  = carry ? e + 1 : e;
    if (exp_out > 7) return sign_f * INFINITY;
    return sign_f * (1.0f + mant_out * 0.25f) * ldexpf(1.0f, exp_out);
  }
  if (e == -7) { int k = (M <= 32) ? 2 : ((M <= 95) ? 3 : 4); return sign_f * (float)k * (1.0f / 256.0f); }
  if (e == -8) { int k = (M < 64) ? 1 : 2; return sign_f * (float)k * (1.0f / 256.0f); }
  if (e == -9) { int k = (M == 0) ? 0 : 1; return sign_f * (float)k * (1.0f / 256.0f); }
  return 0.0f;
}

// Deterministic E4M2 float -> FP6 float (matches _e4m2_to_fp6).
inline float e4m2_to_fp6_float(float x) {
  float sign = (x < 0.0f) ? -1.0f : 1.0f;
  float ax = fabsf(x);
  if (!std::isfinite(ax) || ax >= 32.0f) return sign * 28.0f;
  if (ax <= 0.0546875f) return 0.0f;
  if (ax >= 0.0625f && ax <= 0.21875f) {
    float sub = (ax <= 0.078125f) ? 0.0625f
              : (ax <= 0.15625f)  ? 0.125f
                                  : 0.1875f;
    return sign * sub;
  }
  return x;
}

// FP6 grid float -> 6-bit FP6 E3M2 code (matches _fp6_value_to_code).
inline uint8_t fp6_value_to_code(float v) {
  if (v == 0.0f) return 0;
  int s = (v < 0.0f) ? 1 : 0;
  float av = fabsf(v);
  const int e_bits = 3, m_bits = 2, bias = 3;
  if (av < 0.25f) {  // 2^(1-bias) = 2^-2
    float quantum = 0.0625f;
    int mant = (int)lroundf(av / quantum);
    if (mant < 0) mant = 0;
    if (mant > (1 << m_bits) - 1) mant = (1 << m_bits) - 1;
    return (uint8_t)((s << (e_bits + m_bits)) | mant);
  }
  int E = (int)floorf(log2f(av));
  float base = ldexpf(1.0f, E);
  int mant = (int)lroundf((av - base) / (base / 4.0f));
  if (mant >= 4) { mant = 0; E += 1; }
  int biased = E + bias;
  if (biased > (1 << e_bits) - 1) biased = (1 << e_bits) - 1;
  if (mant > (1 << m_bits) - 1) mant = (1 << m_bits) - 1;
  return (uint8_t)((s << (e_bits + m_bits)) | (biased << m_bits) | mant);
}

// BF16 bits -> 6-bit FP6 E3M2 code (full HW pipeline).
inline uint8_t bf16_bits_to_fp6_e3m2_code(uint16_t bits) {
  return fp6_value_to_code(e4m2_to_fp6_float(bf16_bits_to_e4m2_rne(bits)));
}

// FP6 6-bit code -> 9-bit signed fixed-point (mirrors FP6E3M2NearestFinder.scala).
inline int fp6_to_fixed_point(uint8_t val) {
  val &= 0x3F;
  int sign = (val >> 5) & 1;
  int exp  = (val >> 2) & 0x7;
  int mant = val & 0x3;
  int is_zero = (exp == 0) && (mant == 0);
  int s_exp = (exp == 0) ? -2 : (exp - 3);
  int implicit = (exp == 0) ? 0 : 1;
  int sig = (implicit << 2) | mant;
  int shift_amt = (s_exp + 2) & 0x7;
  int shifted = (sig << shift_amt) & 0x1FF;
  int magnitude = shifted & 0xFF;
  int signed_val = sign ? -magnitude : magnitude;
  return is_zero ? 0 : signed_val;
}

// Nearest LUT-entry finder. Tie-breaking: lower index wins.
inline int fp6e3m2_nearest_finder(uint8_t in_code, const uint8_t lut[16]) {
  int fixed_in = fp6_to_fixed_point(in_code);
  int best = 0;
  int best_d = 0;
  for (int i = 0; i < 16; i++) {
    int f = fp6_to_fixed_point(lut[i]);
    int d = fixed_in - f;
    if (d < 0) d = -d;
    d &= 0x1FF;
    if (i == 0 || d < best_d) { best_d = d; best = i; }
  }
  return best;
}

// Unpack 16 × 6-bit FP6 codes from 3 little-endian uint32 words (96 bits).
inline void unpack_lut_96bit(const uint32_t dwords[3], uint8_t codes[16]) {
  uint64_t lo = (uint64_t)dwords[0] | ((uint64_t)dwords[1] << 32);
  uint32_t hi = dwords[2];
  for (int i = 0; i < 16; i++) {
    int bit = i * 6;
    if (bit + 6 <= 64)      codes[i] = (uint8_t)((lo >> bit) & 0x3F);
    else if (bit >= 64)     codes[i] = (uint8_t)((hi >> (bit - 64)) & 0x3F);
    else {
      uint64_t merged = (lo >> bit) | ((uint64_t)hi << (64 - bit));
      codes[i] = (uint8_t)(merged & 0x3F);
    }
  }
}

// BF16 -> E3M1 (RNE) -> E2M1 (deterministic map) -> 4-bit FP4 code, mirroring
// fp4_matmul_model.py::hw_bf16_to_e2m1.
inline uint8_t bf16_bits_to_fp4_e2m1_code(uint16_t bf16) {
  int sign_bit = (bf16 >> 15) & 1;
  int E = (bf16 >> 7) & 0xFF;
  int M = bf16 & 0x7F;
  if (E == 0) return 0;
  if (E == 255) return (uint8_t)((sign_bit << 3) | 0x7);
  int e = E - 127;

  // BF16 -> E3M1 magnitude (the float value on the E3M1 grid).
  // Grid: 0, 0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, ...
  float e3m1_mag;
  if (e >= -2 && e <= 3) {
    int S = 128 + M;
    int q = (S >> 6) & 1;
    int r = (S >> 5) & 1;
    int sticky = (S & 0x1F) != 0;
    int round_up = r & (sticky | q);
    int sig2 = q + round_up;
    int mant_out = (sig2 >= 2) ? 0 : sig2;
    int exp_out  = (sig2 >= 2) ? e + 1 : e;
    if (exp_out > 3) return (uint8_t)((sign_bit << 3) | 0x7);
    e3m1_mag = (1.0f + mant_out * 0.5f) * ldexpf(1.0f, exp_out);
  } else if (e == -3) {
    e3m1_mag = (M >= 64) ? 0.25f : 0.125f;
  } else if (e == -4) {
    e3m1_mag = (M > 0) ? 0.125f : 0.0f;
  } else {
    e3m1_mag = 0.0f;
  }

  // E3M1 magnitude -> E2M1 (FP4) code (sign-magnitude).
  uint8_t mag_code;
  if (e3m1_mag == 0.0f || e3m1_mag == 0.125f || e3m1_mag == 0.25f) return 0;
  if (e3m1_mag == 0.375f || e3m1_mag == 0.5f)                       mag_code = 0x1;
  else if (e3m1_mag == 0.75f || e3m1_mag == 1.0f)                   mag_code = 0x2;
  else if (e3m1_mag == 1.5f)                                        mag_code = 0x3;
  else if (e3m1_mag == 2.0f)                                        mag_code = 0x4;
  else if (e3m1_mag == 3.0f)                                        mag_code = 0x5;
  else if (e3m1_mag == 4.0f)                                        mag_code = 0x6;
  else                                                              mag_code = 0x7;  // 6.0 or larger
  return (uint8_t)((sign_bit << 3) | mag_code);
}

}

#endif
