#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

struct SoftmaxArgs {
  __global float* x;
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
      denom_sdata[tid] = denom_sdata[idx_a] * mu_fexp(max_a - next_max) + denom_sdata[idx_b] * mu_fexp(max_b - next_max);
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
  uint32_t block_elem_idx = threadblock_id * row_elems / 2;
  uint32_t chunks_per_block = (row_elems + MU_DOUBLE_BLOCK_SIZE - 1) / MU_DOUBLE_BLOCK_SIZE;

  __global float *x = args->x + block_elem_idx;
  __shared _Float16 *x_sdata = sdata;
  __shared _Float16 *max_sdata = sdata + row_elems;
  __shared _Float16 *denom_sdata = max_sdata + MU_DOUBLE_BLOCK_SIZE;
  
  // pass 1 max
  ((float*)x_sdata)[tid] = x[tid];
  _Float16 max = fmaxf(x_sdata[2*tid], x_sdata[2*tid + 1]);
  for (uint32_t chunk = 1; chunk < chunks_per_block; chunk++) {
    uint32_t idx = chunk * MU_BLOCK_SIZE + tid;
    if (2*idx >= row_elems) break;
    ((float*)x_sdata)[idx] = x[idx];
    max = fmaxf(fmaxf(x_sdata[2*idx], x_sdata[2*idx + 1]), max);
  }

  // pass 2 denom
  _Float16 denom = (_Float16)0;
  for (uint32_t chunk = 0; chunk < chunks_per_block; chunk++) {
    uint32_t idx = chunk * MU_BLOCK_SIZE + tid;
    if (2 * idx >= row_elems) break;
    denom += mu_fexp(x_sdata[2*idx] - max) + mu_fexp(x_sdata[2*idx + 1] - max);
  }

  max_sdata[tid] = max;
  denom_sdata[tid] = denom;

  // warp reduce
  reduce(max_sdata, denom_sdata, tid, lane_id);

  // only works because num_warps = num_lanes
  if (lane_id == 0) {
    denom_sdata[warp_id] = denom_sdata[tid];
    max_sdata[warp_id] = max_sdata[tid];
  }

  mu_barrier(0, MU_BLOCK_NUM_WARPS);

  // block reduce
  if (warp_id == 0) {
    reduce(max_sdata, denom_sdata, tid, lane_id);
    if (lane_id == 0)
      denom_sdata[0] = __builtin_bit_cast(_Float16, ONE_BF16_BITS) / denom_sdata[0];
  }

  mu_barrier(0, MU_BLOCK_NUM_WARPS);

  // max and denom in tid 0
  _Float16 m = max_sdata[0], inv_d = denom_sdata[0];

  // pass 3 compute softmax for each element chunk by chunk
  for (uint32_t chunk = 0; chunk < chunks_per_block; chunk++) {
    uint32_t idx = chunk * MU_BLOCK_SIZE + tid;
    if (2*idx >= row_elems) break;
    x_sdata[2*idx] = mu_fexp(x_sdata[2*idx] - m) * inv_d;
    x_sdata[2*idx + 1] = mu_fexp(x_sdata[2*idx + 1] - m) * inv_d;
    x[idx] = ((float *)x_sdata)[idx];
  }
}

SoftmaxArgs softmax_args = {
  .x = nullptr,
  .rows = 0,
  .cols = 0,
};

#include "data"

int main() {
  softmax_args.x = reinterpret_cast<__global float*>(x_raw);
  softmax_args.rows = rows;
  softmax_args.cols = cols;
  mu_schedule(softmax, &softmax_args, vx_num_warps());
  return 0;
}
