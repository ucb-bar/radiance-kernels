// Qwen2 QKV bias-add EPILOGUE on the fp8 QKV projection move-out.
//
//   out[token, c] (bf16) = proj[token, c] (bf16, fp8-GEMM move-out) + bias[c] (fp32)
//
// This is the one op DeepSeek-R1-Distill-Qwen-1.5B (Qwen2) has that TinyLlama/Llama-2 lack:
// a learned bias on the Q/K/V projection outputs, broadcast over tokens. Real dims: hidden
// K=1536, head_dim=128, GQA 12 Q-heads : 2 KV-heads -> QKV width N = 1536 + 256 + 256 = 2048;
// bias = concat(q_bias[1536], k_bias[256], v_bias[256]), one value per (head, head_dim) channel.
//
// The GEMM stays fp8 (proj is the genuine mx_golden fp8 move-out, stored bf16); the bias is
// precision-light (fp32). The SIMT epilogue reads proj[bf16], adds bias[col], truncates to bf16.
// Verified tolerance-exact against bf16_trunc( bf16_to_f(proj) + bias ) (tohost=0).
//
// The kernel must define:
//   void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
//                    uint32_t threads_per_threadblock, uint32_t threadblock_id);

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-3f    // bf16 has ~8 mantissa bits: ~4e-3 ULP; be generous but tight.
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 3.0e-2f
#endif

#include "data"

static inline uint32_t f2b(float f){ union{float f; uint32_t u;} v; v.f=f; return v.u; }
static inline float    b2f(uint32_t u){ union{float f; uint32_t u;} v; v.u=u; return v.f; }
static inline float    bf16_to_f(uint16_t h){ return b2f((uint32_t)h << 16); }
static inline uint16_t f_to_bf16(float f){ return (uint16_t)(f2b(f) >> 16); }  // truncate

// Baseline QKV bias-add epilogue: one row (token) per thread, grid-stride over tokens; for each
// column add the broadcast per-channel bias. col c belongs to head c/head_dim (structure implicit).
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)raw_arg; (void)threadblock_id;
  const uint32_t rows = QKV_M, cols = QKV_N;
  for (uint32_t row = tid_in_threadblock; row < rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    #pragma clang loop unroll(disable)
    for (uint32_t c = 0; c < cols; c++) {
      const float p = bf16_to_f(proj_bf16[base + c]);
      out_raw[base + c] = f_to_bf16(p + bias_f32[c]);
    }
  }
}

static inline float fabsf_(float x) { return x < 0.0f ? -x : x; }
static inline bool close_enough(float c, float g) {
  return fabsf_(c - g) <= TOLERANCE_REL * fabsf_(g) + TOLERANCE_ABS;
}

int main() {
  mu_schedule(kernel_body, nullptr, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);

  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (mu_hart_id() != 0) { for (;;) {} }

  uint32_t errors = 0;
  for (uint32_t i = 0; i < VERIFY_COUNT; i++) {
    if (!close_enough(bf16_to_f(out_raw[i]), bf16_to_f(gold_raw[i]))) errors++;
  }
  uint32_t code = errors ? ((errors << 1) | 1u) : 0u;
  mu_tohost(code);
  return 0;
}
