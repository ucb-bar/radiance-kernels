// GeGLU coverage kernel: C[m,n] = gelu(A) * B  (gelu_pytorch_tanh) — Gemma2 FFN + SmolVLA vision.
// A = gate_proj activations, B = up_proj output; gelu = the tanh-approx GELU (real Gemma-2 HF act).
// Plain GELU (SmolVLA vision MLP) is the B=1 subset. Golden (gen_data.py) is the exact numpy-tanh
// GELU; the kernel uses the algebraically-identical sigmoid form (see gelu() below), agreeing to
// ~1e-7 within TOLERANCE_REL/ABS. Verified at Gemma-2-2B FFN dims (N=9216) — cyclotron tohost=0.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
// KERNEL_OCCUPANCY = warps SPAWNED per core (passed to mu_schedule -> vx_wspawn),
// deliberately FEWER than the header NUM_WARPS (hardware warp slots). The body streams
// two __global loads + one store per element over M*N; every spawned warp x 16 lanes
// puts a coalesced miss stream through the 4KB per-tile l0d, which has NO landing pads
// (makeLandingPads=false). Too many concurrent miss streams overrun the l0d response
// path and trip the TLNBDCache backpressure assert ("response must be ready!",
// TLNBDCache.scala:201). Lowering occupancy bounds the concurrent l0d misses; the
// grid-stride loop (i += tpb) still covers every element. Override with -DKERNEL_OCCUPANCY=N.
#ifndef KERNEL_OCCUPANCY
#define KERNEL_OCCUPANCY 2
#endif
extern "C" uint32_t __mu_num_warps = KERNEL_OCCUPANCY;
#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-3f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 2.0e-4f
#endif
#include "data"

static inline float mu_exp(float x) {
  const float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
  float t = x * LOG2E;
  int k = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)k * LN2;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * 0.008333334f))));
  union { uint32_t i; float f; } s; s.i = (uint32_t)((k + 127) << 23);
  return p * s.f;
}
// gelu_pytorch_tanh, 0.5*x*(1+tanh(C*(x+0.044715 x^3))) with C=sqrt(2/pi),
// rearranged algebraically into a single sigmoid: 0.5*x*(1+tanh(z)) = x/(1+exp(-2z)).
// Same register footprint as silu (one mu_exp, one reciprocal) — keeps the SIMT
// epilogue under the 256 physical-register Rename limit. Matches the numpy tanh
// golden to ~1e-7 (within TOLERANCE_REL/ABS).
static inline float gelu(float x) {
  const float C2 = 1.5957691216057308f; // 2*sqrt(2/pi)
  float u = C2 * (x + 0.044715f * x * x * x);
  return x / (1.0f + mu_exp(-u));
}

struct KernelArgs { __global float *A, *B, *C; uint32_t M, N; };

void kernel_body(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  auto* a = reinterpret_cast<KernelArgs*>(raw);
  const uint32_t total = a->M * a->N;
  #pragma clang loop unroll(disable)
  for (uint32_t i = tid; i < total; i += tpb) a->C[i] = gelu(a->A[i]) * a->B[i];
}

static KernelArgs kernel_args;
__global float C_raw[VERIFY_COUNT] = {0};
int main() {
  kernel_args = {A_raw, B_raw, C_raw, M, N};
  mu_schedule(kernel_body, &kernel_args, KERNEL_OCCUPANCY);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  mu_verify_f32(C_raw, gold_raw, VERIFY_COUNT, TOLERANCE_REL, TOLERANCE_ABS);
  return 0;
}
