// GELU, tanh approximation, bf16 in/out, fp32 math.
//   gelu(x) = 0.5 x (1 + tanh(z)) = x * sigmoid(2 z) = x / (1 + 2^(-2 z log2 e)),
//   z = sqrt(2/pi) (x + 0.044715 x^3)
// Every thread of every cluster handles 32-bit words (two bf16) in a grid-stride loop; UNROLL
// independent words per iteration keep that many loads in flight per thread.
#include <nightly/device.h>
#include <nightly/verify.h>
#include "gelu_data.h"

#ifndef GELU_UNROLL
#define GELU_UNROLL 4
#endif

static inline float bf16_lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
static inline float bf16_hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xFFFF0000u); }
static inline uint32_t f32_to_bf16(float f) {   // round to nearest even; inputs are finite
  const uint32_t u = __builtin_bit_cast(uint32_t, f);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

// 2^t for t in [-126, 126]: 2^n * p(f), n = floor(t), f in [0,1), degree-5 minimax (rel err 2e-7)
static inline float exp2_poly(float t) {
  t = __builtin_fminf(__builtin_fmaxf(t, -126.0f), 126.0f);
  int32_t n;   // floor(t): t is clamped, so a round-down convert needs no range branch
  asm("fcvt.w.s %0, %1, rdn" : "=r"(n) : "r"(t));
  const float f = t - (float)n;
  float p = 1.8775767e-3f;
  p = p * f + 8.9893397e-3f;
  p = p * f + 5.5826318e-2f;
  p = p * f + 2.4015361e-1f;
  p = p * f + 6.9315308e-1f;
  p = p * f + 9.9999994e-1f;
  return __builtin_bit_cast(float, __builtin_bit_cast(int32_t, p) + (n << 23));
}

static inline float gelu(float x) {
  constexpr float K0 = 0.7978845608f;             // sqrt(2/pi)
  constexpr float K1 = 0.044715f;
  constexpr float M2LOG2E = -2.0f * 1.4426950409f; // -2 log2(e)
  const float z = K0 * (x + K1 * x * x * x);
  return x / (1.0f + exp2_poly(M2LOG2E * z));
}

static inline uint32_t gelu2(uint32_t w) {
  return f32_to_bf16(gelu(bf16_lo(w))) | (f32_to_bf16(gelu(bf16_hi(w))) << 16);
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t *X = (const uint32_t *)GELU_X_ADDR;
  uint32_t *Y = (uint32_t *)GELU_Y_ADDR;
  constexpr uint32_t WORDS = GELU_N / 2;
  const uint32_t nthr = tpb * NIGHTLY_CLUSTERS;
  const uint32_t step = nthr * GELU_UNROLL;
  uint32_t i = tb * tpb + tid;
  for (; i + (GELU_UNROLL - 1) * nthr < WORDS; i += step) {
#if defined(GELU_HB) && GELU_HB
    if (tid == 0 && tb == 0) nightly_heartbeat(i); else asm volatile("nop");   // diagnosis only
#endif
    uint32_t w[GELU_UNROLL];
#pragma unroll
    for (int u = 0; u < GELU_UNROLL; u++) w[u] = X[i + u * nthr];
#pragma unroll
    for (int u = 0; u < GELU_UNROLL; u++) Y[i + u * nthr] = gelu2(w[u]);
  }
  for (; i < WORDS; i += nthr) Y[i] = gelu2(X[i]);
  nightly_kernel_end(tid, tpb);
  nightly_verify_bf16(GELU_Y_ADDR, GELU_G_ADDR, GELU_N, tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
