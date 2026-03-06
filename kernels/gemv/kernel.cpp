#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

struct GEMVArgs {
  __global _Float16* A;
  __global _Float16* x;
  __global _Float16* y;
  uint32_t m;
  uint32_t n;
};

__shared _Float16* const sdata = reinterpret_cast<__shared _Float16*>(0x0);

static inline void reduce(uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MU_NUM_THREADS; stride *= 2) {
    if (lane_id % stride == 0)
      sdata[tid] = sdata[tid] + sdata[tid + (stride >> 1)];
  }
}

// compute Ax where A is MxN and x is Nx1 (stored transposed, so 1xN storage layout) and stores back in x
// requires that the row is at least BLOCK_SIZE large and no more threadblocks are launched.
// need at least BLOCK_SIZE shared mem space
void gemv(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<GEMVArgs*>(arg);
  uint32_t lane_id = tid_in_threadblock % 16;
  uint32_t warp_id = tid_in_threadblock / 16;
  uint32_t tid = tid_in_threadblock;

  uint32_t row_elems = args->n;
  uint32_t block_elem_idx = threadblock_id * row_elems;
  uint32_t chunks_per_block = (row_elems + MU_BLOCK_SIZE - 1) / MU_BLOCK_SIZE;

  __global _Float16 *A = args->A + block_elem_idx;
  __global _Float16 *x = args->x;
  __global _Float16 *y = args->y;


  _Float16 accum = A[tid] * x[tid];

  for (uint32_t chunk = 1; chunk < chunks_per_block; chunk++) {
    uint32_t idx = chunk * MU_BLOCK_SIZE + tid;
    if (idx >= row_elems) break;
    accum += A[idx] * x[idx];
  }

  sdata[tid] = accum;

  // warp reduce
  reduce(tid, lane_id);

  // block reduce = warp reduce only works when num_warps = num_lanes
  if (lane_id == 0)
    sdata[warp_id] = sdata[tid];

  mu_barrier(0, MU_BLOCK_NUM_WARPS);

  // block reduce
  if (warp_id == 0) {
    reduce(tid, lane_id);
    if (lane_id == 0)
      y[threadblock_id] = sdata[0];
  }
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
  mu_schedule(gemv, &gemv_args);
  return 0;
}
