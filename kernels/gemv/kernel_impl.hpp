#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <stdint.h>

#ifndef GEMV_NUM_WARPS
#define GEMV_NUM_WARPS 8
#endif

#ifndef GEMV_ILP
#define GEMV_ILP 1
#endif

extern "C" uint32_t __mu_num_warps = GEMV_NUM_WARPS;

struct GEMVArgs {
  __global _Float16* A;
  __global _Float16* x;
  __global _Float16* y;
  uint32_t m;
  uint32_t n;
};

__shared _Float16* const sdata = reinterpret_cast<__shared _Float16*>(0x0);

template <uint32_t MAX_STRIDE>
static inline void reduce(uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MAX_STRIDE; stride *= 2) {
    if (lane_id % stride == 0)
      sdata[tid] = sdata[tid] + sdata[tid + (stride >> 1)];
  }
}

template <uint32_t ILP>
static inline void gemv_impl(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  static_assert(ILP > 0);
  static_assert((MU_NUM_THREADS & (MU_NUM_THREADS - 1)) == 0);
  auto* args = reinterpret_cast<GEMVArgs*>(arg);
  constexpr uint32_t kWarpWidth = MU_NUM_THREADS;
  uint32_t lane_id = tid_in_threadblock % kWarpWidth;
  uint32_t warp_id = tid_in_threadblock / kWarpWidth;
  uint32_t warps_per_threadblock = threads_per_threadblock / kWarpWidth;
  uint32_t global_warp_id = threadblock_id * warps_per_threadblock + warp_id;
  uint32_t global_warp_stride = MU_NUM_CLUSTERS * warps_per_threadblock;

  const uint32_t row_elems = args->n;
  __global _Float16* x = args->x;
  __global _Float16* y = args->y;

  for (uint32_t row = global_warp_id; row < args->m; row += global_warp_stride) {
    __global _Float16* A = args->A + row * row_elems;
    _Float16 accum[ILP] = {};
    const uint32_t ilp_stride = kWarpWidth * ILP;
    const uint32_t ilp_span = kWarpWidth * (ILP - 1);

    // Benchmark assumption: each row slice is divisible into full ILP groups.
    for (uint32_t base = lane_id; base + ilp_span < row_elems; base += ilp_stride) {
      _Float16 a_vals[ILP];
      _Float16 x_vals[ILP];

      #pragma unroll
      for (uint32_t u = 0; u < ILP; ++u) {
        const uint32_t idx = base + u * kWarpWidth;
        a_vals[u] = A[idx];
        x_vals[u] = x[idx];
      }

      #pragma unroll
      for (uint32_t u = 0; u < ILP; ++u)
        accum[u] += a_vals[u] * x_vals[u];
    }

    _Float16 sum = accum[0];
    #pragma unroll
    for (uint32_t u = 1; u < ILP; ++u)
      sum += accum[u];

    sdata[tid_in_threadblock] = sum;
    reduce<kWarpWidth>(tid_in_threadblock, lane_id);

    if (lane_id == 0)
      y[row] = sdata[tid_in_threadblock];
  }
}

static inline void gemv(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  gemv_impl<GEMV_ILP>(
    arg, tid_in_threadblock, threads_per_threadblock, threadblock_id);
}

GEMVArgs gemv_args = {
  .A = nullptr,
  .x = nullptr,
  .y = nullptr,
  .m = 0,
  .n = 0,
};

#include "data"

int main() {
  gemv_args.A = reinterpret_cast<__global _Float16*>(A_raw);
  gemv_args.x = reinterpret_cast<__global _Float16*>(x_raw);
  gemv_args.y = reinterpret_cast<__global _Float16*>(y_raw);
  gemv_args.m = m;
  gemv_args.n = n;
  mu_schedule(gemv, &gemv_args, GEMV_NUM_WARPS);
  return 0;
}
