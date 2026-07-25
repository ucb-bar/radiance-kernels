// Gemma-2 decoder-layer NORM/RESIDUAL WIRING (the "sandwich" 4-norm topology).
// fp32, R=8 rows (tokens), D=2304 (real Gemma-2-2b hidden). Validates the exact ordering from HF
// transformers Gemma2DecoderLayer.forward:
//
//   residual = h
//   h = input_layernorm(h)              # NORM 1  (pre-attention)
//   h = self_attn(h)
//   h = post_attention_layernorm(h)     # NORM 2  (post-attention)
//   h = residual + h                    # RESIDUAL 1  (bypasses BOTH norm 1 and norm 2)
//   residual = h
//   h = pre_feedforward_layernorm(h)    # NORM 3  (pre-FFN)
//   h = mlp(h)
//   h = post_feedforward_layernorm(h)   # NORM 4  (post-FFN)
//   h = residual + h                    # RESIDUAL 2  (bypasses BOTH norm 3 and norm 4)
//
// Each norm is Gemma2RMSNorm: rms(x) * (1.0 + weight), eps=1e-6. The self_attn and mlp sublayers
// are STUBBED as deterministic per-channel transforms (h[j] *= wa[j] / wm[j]) because those ops
// are covered by separate kernels; what is under test here is purely the norm/residual TOPOLOGY
// (which weight applies where, and that each residual carries the pre-norm value across the sublayer).
//

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
  __global float *x;      // [R, D] input residual stream
  __global float *w_in;   // [D] input_layernorm weight (zero-centered)
  __global float *w_pa;   // [D] post_attention_layernorm weight
  __global float *w_pf;   // [D] pre_feedforward_layernorm weight
  __global float *w_pff;  // [D] post_feedforward_layernorm weight
  __global float *wa;     // [D] self_attn stub (per-channel scale)
  __global float *wm;     // [D] mlp stub (per-channel scale)
  __global float *tmp;    // [R, D] scratch
  __global float *out;    // [R, D] output
  uint32_t R, D;
};

// Gemma-2 decoder-layer norm/residual wiring: one token-row per thread, grid-stride.
// Executes the exact 4-norm "sandwich" topology from Gemma2DecoderLayer.forward, using a global
// scratch row (tmp) so each RMSNorm can reduce over the full row before scaling. Sublayers are
// per-channel scales (wa/wm). eps=1e-6, norm applies (1.0 + weight).
static inline float row_inv_rms(__global const float* r, uint32_t D) {
  float ss = 0.0f;
  #pragma clang loop unroll(disable)
  for (uint32_t j = 0; j < D; j++) { const float v = r[j]; ss += v * v; }
  return mu_rsqrt(ss / (float)D + 1.0e-6f);
}
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t D = a->D;
  for (uint32_t r = tid_in_threadblock; r < a->R; r += threads_per_threadblock) {
    const uint32_t o = r * D;
    __global float* x   = a->x   + o;   // input residual stream (preserved)
    __global float* tmp = a->tmp + o;   // scratch
    __global float* out = a->out + o;

    // NORM 1 (input_layernorm) + self_attn stub -> tmp
    float inv = row_inv_rms(x, D);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++)
      tmp[j] = (x[j] * inv * (1.0f + a->w_in[j])) * a->wa[j];

    // NORM 2 (post_attention_layernorm) on tmp, then RESIDUAL 1 (+ x) -> out (= x1)
    inv = row_inv_rms(tmp, D);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++)
      out[j] = x[j] + (tmp[j] * inv * (1.0f + a->w_pa[j]));

    // NORM 3 (pre_feedforward_layernorm) on x1 (=out) + mlp stub -> tmp
    inv = row_inv_rms(out, D);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++)
      tmp[j] = (out[j] * inv * (1.0f + a->w_pf[j])) * a->wm[j];

    // NORM 4 (post_feedforward_layernorm) on tmp, then RESIDUAL 2 (+ x1=out) -> out
    inv = row_inv_rms(tmp, D);
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++)
      out[j] = out[j] + (tmp[j] * inv * (1.0f + a->w_pff[j]));
  }
}


static KernelArgs kernel_args;


int main() {
  kernel_args = {x_raw, w_in_raw, w_pa_raw, w_pf_raw, w_pff_raw, wa_raw, wm_raw,
                 tmp_raw, out_raw, ROWS, COLS};

  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);

  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  mu_verify_f32(out_raw, gold_raw, VERIFY_COUNT, TOLERANCE_REL, TOLERANCE_ABS);
  return 0;
}
