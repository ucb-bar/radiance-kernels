#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

struct SoftmaxArgs {
  __global _Float16* x;
  uint32_t rows;
  uint32_t cols;
};

__shared _Float16* const sdata = reinterpret_cast<__shared _Float16*>(0x0);

static inline void reduce(__shared _Float16 *max_sdata, __shared _Float16 *denom_sdata, uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MU_NUM_THREADS; stride *= 2) {
    if (lane_id % stride == 0) {
      uint32_t idx_a = tid, idx_b = (tid + (stride >> 1));
      _Float16 max_a = max_sdata[idx_a], max_b = max_sdata[idx_b];
      _Float16 next_max = fmaxf(max_a, max_b);
      denom_sdata[tid] = denom_sdata[idx_a] * mu_fexp(next_max - max_a) + denom_sdata[idx_b] * mu_fexp(next_max - max_b);
      max_sdata[tid] = next_max;
    }
  }
}

// requires that cols + BLOCK_SIZE * 2 fits in smem (one row + max of one row + denom of one row)
// requires that you do NOT spawn a threadblock for a non existent row
void softmax(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<SoftmaxArgs*>(arg);
  uint32_t lane_id = tid_in_threadblock % 16;
  uint32_t warp_id = tid_in_threadblock / 16;
  uint32_t tid = tid_in_threadblock;
  

  uint32_t row_elems = args->cols;
  uint32_t block_elem_idx = threadblock_id * row_elems;
  uint32_t chunks_per_block = (row_elems + MU_BLOCK_SIZE - 1) / MU_BLOCK_SIZE;

  __global _Float16 *x = args->x + block_elem_idx;
  __shared _Float16 *x_sdata = sdata;
  __shared _Float16 *max_sdata = sdata + row_elems;
  __shared _Float16 *denom_sdata = max_sdata + MU_BLOCK_SIZE;
  
  // init from first valid element to avoid 0 * exp(+inf) on first step
  // require that each row is a multiple of block size
  x_sdata[tid] = x[tid];
  max_sdata[tid] = x_sdata[tid];
  denom_sdata[tid] = (_Float16)0x1.0000000000000p+0;

  // repeat for remaining chunks
  for (uint32_t chunk = 1; chunk < chunks_per_block; chunk++) {
    uint32_t idx = chunk * MU_BLOCK_SIZE + tid;
    if (idx >= row_elems) break;
    x_sdata[idx] = x[idx];
    _Float16 next_max = fmaxf(x_sdata[idx], max_sdata[tid]);
    denom_sdata[tid] = denom_sdata[tid] * mu_fexp(next_max - max_sdata[tid]) + mu_fexp(x_sdata[idx] - next_max);
    max_sdata[tid] = next_max;
  }

  // warp reduce
  reduce(max_sdata, denom_sdata, tid, lane_id);

  // only works because num_warps = num_lanes
  if (lane_id == 0) {
    denom_sdata[warp_id] = denom_sdata[tid];
    max_sdata[warp_id] = max_sdata[tid];
  }

  vx_barrier(0, MU_BLOCK_NUM_WARPS);

  // block reduce
  if (warp_id == 0) {
    reduce(max_sdata, denom_sdata, tid, lane_id);
    if (lane_id == 0)
      denom_sdata[0] = 1 / denom_sdata[0]; 
  }

  vx_barrier(0, MU_BLOCK_NUM_WARPS);

  // max and denom in tid 0
  _Float16 m = max_sdata[0], inv_d = denom_sdata[0];

  // compute softmax for each element chunk by chunk
  for (uint32_t chunk = 0; chunk < chunks_per_block; chunk++) {
    uint32_t idx = chunk * MU_BLOCK_SIZE + tid;
    if (idx >= row_elems) break;
    x[idx] = mu_fexp(x_sdata[idx] - m) * inv_d;
  }
}

SoftmaxArgs softmax_args = {
  .x = nullptr,
  .rows = 0,
  .cols = 0,
};

#include "data"

int main() {
  softmax_args.x = x;
  softmax_args.rows = rows;
  softmax_args.cols = cols;
  mu_schedule(softmax, &softmax_args);
  return 0;
}
