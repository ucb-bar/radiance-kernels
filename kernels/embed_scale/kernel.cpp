// Gemma-2 scaled embedding gather  OUT[t,:] = TABLE[ids[t],:] * sqrt(D)
// (fp32, T=16 tokens, D=2304 = real Gemma-2-2b hidden_size; scale = sqrt(2304) = 48.0 exact).
//
// Confirmed from HF transformers Gemma2TextScaledWordEmbedding:
//   embed_scale = config.hidden_size ** 0.5   (== 48.0 for hidden=2304, exact in bf16)
//   out = Embedding(ids) * embed_scale
// Table is tied to the LM head (_tied_weights_keys), but that does not affect this op.
//

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

// Gather + exact-power-of-two scale: results are bit-exact, use a tight tolerance.
#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-6f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 1.0e-6f
#endif

#include "data"

struct KernelArgs {
  __global float *table;   // [V, D]
  __global uint32_t *ids;  // [T]
  __global float *out;     // [T, D]
  uint32_t T, D;
};

// Gemma-2 scaled embedding gather: OUT[t,:] = TABLE[ids[t],:] * sqrt(D).
// One token-row per thread, grid-stride; gather the row TABLE[id] and scale by 48.0 (=sqrt(2304)).
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t D = a->D;
  const float scale = 48.0f; // sqrt(2304), exact
  for (uint32_t t = tid_in_threadblock; t < a->T; t += threads_per_threadblock) {
    const uint32_t src = a->ids[t] * D;
    const uint32_t dst = t * D;
    #pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < D; j++) a->out[dst + j] = a->table[src + j] * scale;
  }
}


static KernelArgs kernel_args;


int main() {
  kernel_args = {table_raw, ids_raw, out_raw, TT, DD};

  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);

  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  mu_verify_f32(out_raw, gold_raw, VERIFY_COUNT, TOLERANCE_REL, TOLERANCE_ABS);
  return 0;
}
