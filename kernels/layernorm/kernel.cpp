// SigLIP vision LayerNorm (SmolVLA / SmolVLM2-500M vision tower).
//
//   mean[t]  = (1/D) sum_j x[t,j]
//   var[t]   = (1/D) sum_j (x[t,j]-mean[t])^2          (biased, torch LayerNorm)
//   out[t,j] = (x[t,j]-mean[t]) / sqrt(var[t]+eps) * gamma[j] + beta[j]
//
// The op the RMSNorm-complete kernel set lacks: MEAN-SUBTRACTION + BETA bias.
// All fp32 (norms are precision-light). One token-row per thread, grid-stride.
// Self-verifying against a numpy fp32 golden; verdict via tohost ECALL (0 = pass).

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"

#ifndef NUM_WARPS
#define NUM_WARPS 4          // occupancy per core; threadblock spans MU_NUM_CORES
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-3f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 2.0e-4f
#endif

#include "data"

// fp32 reciprocal-sqrt (no libm on this toolchain): fast-inv-sqrt seed + 3 Newton.
static inline float mu_rsqrt(float x) {
  union { float f; uint32_t i; } u; u.f = x;
  u.i = 0x5f3759dfu - (u.i >> 1);
  float y = u.f;
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

struct KernelArgs {
  __global float *in, *gamma, *beta, *out;
  uint32_t rows, cols;
};

static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t cols = a->cols;
  const float invD = 1.0f / (float)cols;
  const float eps = LN_EPS;
  for (uint32_t row = tid_in_threadblock; row < a->rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    // pass 1: single-pass stats -- Sx = sum x, Sxx = sum x^2 (one load per elem).
    float sx = 0.0f, sxx = 0.0f;
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < cols; j++) { const float v = a->in[base + j]; sx += v; sxx += v * v; }
    const float mean = sx * invD;
    const float var = sxx * invD - mean * mean;        // biased variance
    const float inv = mu_rsqrt(var + eps);
    // pass 2: normalize (mean-sub) + affine (gamma scale, beta shift).
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < cols; j++)
      a->out[base + j] = (a->in[base + j] - mean) * inv * a->gamma[j] + a->beta[j];
  }
}

static KernelArgs kernel_args;


int main() {
  kernel_args = {x_raw, gamma_raw, beta_raw, out_raw, LN_M, LN_D};
  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);

  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  mu_verify_f32(out_raw, gold_raw, VERIFY_COUNT, TOLERANCE_REL, TOLERANCE_ABS);
  return 0;
}
