#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

struct SoftmaxArgs {
  float* x;
  uint32_t rows;
  uint32_t cols;
};

__shared float* const sdata = reinterpret_cast<__shared float*>(0x0);

template <uint32_t skip>
inline void reduce(__shared float *max_sdata, __shared float *denom_sdata, uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MU_NUM_THREADS; stride *= 2) {
    if (lane_id % stride) {
      uint32_t idx_a = tid * skip, idx_b = (tid + stride >> 1) * skip;
      float max_a = max_sdata[tid], max_b = max_sdata[idx_b];
      float next_max = fmaxf(max_a, max_b);
      denom_sdata[tid] = denom_sdata[tid] * (next_max - max_a) + denom_sdata[idx_b] * (next_max - max_b);
      max_sdata[tid] = next_max;
    }
  }
}

// requires that cols + BLOCK_SIZE * 2 fits in smem (one row + max of one row + denom of one row)
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
  uint32_t rows_per_threadblock = (MU_BLOCK_SIZE + row_elems - 1 / row_elems) ;
  uint32_t block_row_idx = threadblock_id * rows_per_threadblock;
  uint32_t block_elem_idx = block_row_idx * row_elems;
  uint32_t chunks_per_block = (row_elems + MU_BLOCK_SIZE - 1) / MU_BLOCK_SIZE;

  float *x = args->x + block_elem_idx;
  __shared float *x_sdata = sdata;
  __shared float *max_sdata = sdata + row_elems;
  __shared float *denom_sdata = max_sdata + MU_BLOCK_SIZE;
  
  float max = -INFINITY;
  float denom = 0; 

  // load chunk and compute max and denom
  for (uint32_t chunk = 0; chunk < chunks_per_block; chunk++) {
    // load row to smem, fexp in smem
    uint32_t idx = chunk * MU_BLOCK_SIZE + tid;
    x_sdata[idx] = x[idx + block_elem_idx];
    float next_max = fmaxf(x_sdata[idx], max);
    denom = denom * (next_max - max) + mu_fexp(x_sdata[idx] - next_max);
    max = next_max;

    // next chunk
    x += MU_BLOCK_SIZE;
    x_sdata += MU_BLOCK_SIZE;
  }

  denom_sdata[tid] = denom;
  max_sdata[tid] = max;

  // warp reduce
  reduce<0>(max_sdata, denom_sdata, tid, lane_id);

  vx_barrier(0, MU_BLOCK_NUM_WARPS);

  // block reduce
  if (warp_id == 0)
    reduce<MU_NUM_THREADS>(max_sdata, denom_sdata, tid, lane_id);

  // max and denom in tid 0
  float m = max_sdata[0], d = denom_sdata[0];

  x -= row_elems;
  x_sdata -= row_elems;
  // compute softmax for each element chunk by chunk
  for (uint32_t chunk = 0; chunk < chunks_per_block; chunk++) {
    uint32_t idx = chunk * MU_BLOCK_SIZE + tid;
    x[idx] = mu_fexp(x_sdata[idx + block_elem_idx] - m) / d;
    x += MU_BLOCK_SIZE;
    x_sdata += MU_BLOCK_SIZE;
  }
}

SoftmaxArgs softmax_args = {
  .x = nullptr,
  .rows = 0,
  .cols = 0,
};

int main() {
  mu_schedule(softmax, &softmax_args);
  return 0;
}
