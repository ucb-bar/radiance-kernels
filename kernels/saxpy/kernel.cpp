#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <stdint.h>

#define SAXPY_NUM_WARPS 4

struct SaxpyArgs {
  __global float* src;
  __global float* dst;
  float factor;
  uint32_t n;
};

static inline void saxpy(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<SaxpyArgs*>(arg);
  (void)threadblock_id;

  for (uint32_t i = tid_in_threadblock; i < args->n; i += threads_per_threadblock) {
    args->dst[i] = args->dst[i] + args->src[i] * args->factor;
  }
}

SaxpyArgs saxpy_args = {
  .src = nullptr,
  .dst = nullptr,
  .factor = 0.0f,
  .n = 0,
};

#include "data"

int main() {
  saxpy_args.src = reinterpret_cast<__global float*>(src_raw);
  saxpy_args.dst = reinterpret_cast<__global float*>(dst_raw);
  saxpy_args.factor = __builtin_bit_cast(float, factor_raw);
  saxpy_args.n = n;
  mu_schedule(saxpy, &saxpy_args, SAXPY_NUM_WARPS);
  return 0;
}
