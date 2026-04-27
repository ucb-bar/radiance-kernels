#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#ifndef VECADD_ILP
#define VECADD_ILP 1
#endif

#ifndef VECADD_NUM_WARPS
#define VECADD_NUM_WARPS 4
#endif

extern "C" uint32_t __mu_num_warps = VECADD_NUM_WARPS;

struct VecAddArgs {
  __global float* A;
  __global float* B;
  __global float* C;
  uint32_t n;
};

#ifndef VECADD_OUTER_UNROLL
#define VECADD_OUTER_UNROLL 1
#endif

template <uint32_t ILP>
static inline void vecadd_impl(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<VecAddArgs*>(raw_arg);

  constexpr uint32_t kIlp = ILP;
  constexpr uint32_t kWarpWidth = MU_NUM_THREADS;
  constexpr uint32_t kOuter = VECADD_OUTER_UNROLL;
  const uint32_t tid_in_warp = tid_in_threadblock % kWarpWidth;
  const uint32_t warp_id = tid_in_threadblock / kWarpWidth;
  const uint32_t warps_per_threadblock = threads_per_threadblock / kWarpWidth;
  const uint32_t elems_per_chunk = kWarpWidth * kIlp;
  uint32_t chunk;
  uint32_t chunk_stride;

#if MU_NUM_CLUSTERS == 1
  (void)threadblock_id;
  chunk = warp_id * elems_per_chunk;
  chunk_stride = warps_per_threadblock * elems_per_chunk;
#else
  const uint32_t global_warp_id = threadblock_id * warps_per_threadblock + warp_id;
  const uint32_t global_warp_stride = MU_NUM_CLUSTERS * warps_per_threadblock;
  chunk = global_warp_id * elems_per_chunk;
  chunk_stride = global_warp_stride * elems_per_chunk;
#endif

  const uint32_t outer_stride = chunk_stride * kOuter;
  float a_vals[kIlp];
  float b_vals[kIlp];
  float c_vals[kIlp];

  // Benchmark assumption: n is a multiple of the per-warp chunk times the
  // outer-loop unroll factor.
  #pragma unroll 1
  for (; chunk < args->n; chunk += outer_stride) {
    #pragma unroll VECADD_OUTER_UNROLL
    for (uint32_t outer = 0; outer < kOuter; ++outer) {
      const uint32_t lane_base = chunk + outer * chunk_stride + tid_in_warp;

      #pragma unroll
      for (uint32_t i = 0; i < kIlp; ++i) {
        const uint32_t idx = lane_base + i * kWarpWidth;
        a_vals[i] = args->A[idx];
        b_vals[i] = args->B[idx];
      }

      #pragma unroll
      for (uint32_t i = 0; i < kIlp; ++i) {
        c_vals[i] = a_vals[i] + b_vals[i];
      }

      #pragma unroll
      for (uint32_t i = 0; i < kIlp; ++i) {
        const uint32_t idx = lane_base + i * kWarpWidth;
        args->C[idx] = c_vals[i];
      }
    }
  }
}

static inline void vecadd(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  vecadd_impl<VECADD_ILP>(
    raw_arg, tid_in_threadblock, threads_per_threadblock, threadblock_id);
}

VecAddArgs vecadd_args = {
  .A = nullptr,
  .B = nullptr,
  .C = nullptr,
  .n = 0,
};

#include "data"

int main() {
  vecadd_args.A = reinterpret_cast<__global float*>(A_raw);
  vecadd_args.B = reinterpret_cast<__global float*>(B_raw);
  vecadd_args.C = reinterpret_cast<__global float*>(C_raw);
  vecadd_args.n = n;
  mu_schedule(vecadd, &vecadd_args, VECADD_NUM_WARPS);
  return 0;
}
