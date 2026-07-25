// Gemma-2 FINAL-logit soft-capping: OUT[i] = cap * tanh(logits[i] / cap), cap = 30.0.
//
// Confirmed from google/gemma-2-2b-it config.json: "final_logit_softcapping": 30.0.
// This is a plain SIMT elementwise epilogue on the LM-head output (no reductions, no
// cross-lane comms). It is the simplest of the Gemma soft-cap ops -- and it validates the
// tanh primitive that the attention-score soft-cap (cap=50, folded into flash online
// softmax) reuses. Self-checking: verdict via tohost ECALL (0 = pass).
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-3f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 2.0e-4f
#endif

#include "data"

// ---- fp32 exp (range reduction + 5th-order poly; no libm on this toolchain) ----
// noinline: keep the poly's temporaries out of the caller's register set (the 8-warp
// phys-reg file is only 256 regs; inlining the fp32 tanh here overflows it -- same reason
// the flash softmax helpers are all noinline).
static __attribute__((noinline)) float expf_(float x) {
  if (x > 88.0f) return 3.0e38f;
  if (x < -88.0f) return 0.0f;
  const float LOG2E = 1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;
  float t = x * LOG2E;
  int k = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)k * LN2;               // r in [-ln2/2, ln2/2]
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f
              + r * (0.041666668f + r * 0.008333334f))));
  union { float f; uint32_t i; } u;
  int e = k + 127;
  if (e <= 0) return 0.0f;
  if (e >= 255) return 3.0e38f;
  u.i = (uint32_t)e << 23;                     // 2^k
  return p * u.f;
}

// tanh(x) = sign(x) * (1 - e)/(1 + e), e = exp(-2|x|) in (0,1]  (numerically stable).
static __attribute__((noinline)) float tanhf_(float x) {
  float ax = x < 0.0f ? -x : x;
  float e = expf_(-2.0f * ax);
  float t = (1.0f - e) / (1.0f + e);
  return x < 0.0f ? -t : t;
}

// cap * tanh(x / cap) -- the Gemma soft-cap primitive (shared by attn cap=50 & final cap=30).
static inline float softcap(float x, float cap) {
  return cap * tanhf_(x / cap);
}

struct KernelArgs {
  __global float *in, *out;
  uint32_t n;
  float cap;
};

static void kernel_body(void *raw_arg, uint32_t tid_in_threadblock,
                        uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  KernelArgs *a = reinterpret_cast<KernelArgs *>(raw_arg);
  const float cap = a->cap;
  // Grid-stride over the flat tile; both cores redundantly cover the whole tile (idempotent
  // elementwise op), matching the baseline idiom -- avoids a core-partition bug.
  for (uint32_t i = tid_in_threadblock; i < a->n; i += threads_per_threadblock)
    a->out[i] = softcap(a->in[i], cap);
}

static KernelArgs kernel_args;


int main() {
  kernel_args = {x_raw, out_raw, VERIFY_COUNT, FINAL_SOFTCAP};

  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);

  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  mu_verify_f32(out_raw, gold_raw, VERIFY_COUNT, TOLERANCE_REL, TOLERANCE_ABS);
  return 0;
}
