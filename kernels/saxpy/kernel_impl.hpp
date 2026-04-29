#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <stdint.h>

#ifndef SAXPY_NUM_WARPS
#define SAXPY_NUM_WARPS 4
#endif

#ifndef SAXPY_ILP
#define SAXPY_ILP 1
#endif

extern "C" uint32_t __mu_num_warps = SAXPY_NUM_WARPS;

struct SaxpyArgs {
  __global float* src;
  __global float* dst;
  float factor;
  uint32_t n;
};

template <uint32_t ILP>
static inline void saxpy_impl(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  static_assert(ILP > 0);
  auto* args = reinterpret_cast<SaxpyArgs*>(arg);
  (void)threadblock_id;

  const uint32_t ilp_stride = threads_per_threadblock * ILP;
  const uint32_t ilp_span = threads_per_threadblock * (ILP - 1);
  uint32_t base = tid_in_threadblock;

  for (; base + ilp_span < args->n; base += ilp_stride) {
    float src_vals[ILP];
    float dst_vals[ILP];
    float out_vals[ILP];

    #pragma unroll
    for (uint32_t u = 0; u < ILP; ++u) {
      const uint32_t i = base + u * threads_per_threadblock;
      dst_vals[u] = args->dst[i];
      src_vals[u] = args->src[i];
    }

    #pragma unroll
    for (uint32_t u = 0; u < ILP; ++u) {
      out_vals[u] = dst_vals[u] + src_vals[u] * args->factor;
    }

    #pragma unroll
    for (uint32_t u = 0; u < ILP; ++u) {
      const uint32_t i = base + u * threads_per_threadblock;
      args->dst[i] = out_vals[u];
    }
  }

  if constexpr (ILP > 1) {
    for (; base < args->n; base += threads_per_threadblock) {
      args->dst[base] = args->dst[base] + args->src[base] * args->factor;
    }
  }
}

static inline void saxpy(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  saxpy_impl<SAXPY_ILP>(
    arg, tid_in_threadblock, threads_per_threadblock, threadblock_id);
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
