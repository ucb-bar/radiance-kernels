// Gemma-2 RMSNorm  OUT[r,:] = x/sqrt(mean(x^2)+eps) * (1.0 + gamma)
// (fp32, 16x2304 = real Gemma-2-2b hidden). Gemma stores the norm weight zero-centered and
// applies it as (1.0 + weight); eps = 1e-6 (config rms_norm_eps). This differs from the
// Llama/TinyLlama RMSNorm (test27) which applies `gamma` directly with eps=1e-5.
//

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"

#ifndef NUM_WARPS
#define NUM_WARPS 4 // occupancy per core; threadblock spans MU_NUM_CORES cores
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

// Relative FP tolerance: accommodates reassociated accumulation and the RTL's
// ~1-ULP-approximate divider.
#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-3f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 2.0e-4f
#endif

#include "data"

// fp32 reciprocal-sqrt (no libm/sqrtf on this toolchain): fast-inverse-sqrt seed + 3 Newton iters.
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
  __global float *in, *gamma, *out;
  uint32_t rows, cols;
};

// Gemma-2 RMSNorm(1+w): one row per thread, grid-stride. Sum of squares -> rsqrt(ms+eps)
// -> scale by (1.0 + gamma). eps=1e-6, gamma stored zero-centered (Gemma applies 1+weight).
// Only differences vs the Llama RMSNorm (sol27): eps 1e-6 and the (1.0f + gamma) factor.
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t cols = a->cols;
  const float eps = 1.0e-6f;
  for (uint32_t row = tid_in_threadblock; row < a->rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    float ss = 0.0f;
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < cols; j++) { const float v = a->in[base + j]; ss += v * v; }
    const float inv = mu_rsqrt(ss / (float)cols + eps);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < cols; j++)
      a->out[base + j] = a->in[base + j] * inv * (1.0f + a->gamma[j]);
  }
}


static KernelArgs kernel_args;


int main() {
  kernel_args = {x_raw, gamma_raw, out_raw, ROWS, COLS};

  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);

  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  mu_verify_f32(out_raw, gold_raw, VERIFY_COUNT, TOLERANCE_REL, TOLERANCE_ABS);
  return 0;
}
