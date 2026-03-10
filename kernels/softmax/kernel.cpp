#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#define NUM_WARPS 4
#define DOUBLE_BLOCK_SIZE MU_DOUBLE_BLOCK_SIZE(NUM_WARPS)
#define BLOCK_SIZE MU_BLOCK_SIZE(NUM_WARPS)
#define BLOCK_NUM_WARPS MU_BLOCK_NUM_WARPS(NUM_WARPS)
// only works for powers of 2 reduction, otherwise you have to manually write your own reduction
#define THREAD_DIV (MU_NUM_MAX_WARPS / NUM_WARPS)

struct SoftmaxArgs {
  __global uint32_t* x;
  uint32_t rows;
  uint32_t cols;
};

__shared uint32_t* const sdata = reinterpret_cast<__shared uint32_t*>(0x0);

template <uint32_t MAX_STRIDE>
static inline void reduce(__shared uint32_t *max_sdata, __shared uint32_t *denom_sdata, uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MAX_STRIDE; stride *= 2) {
    if (lane_id % stride == 0) {
      uint32_t idx_a = tid, idx_b = (tid + (stride >> 1));
      _Float16 max_a = __builtin_bit_cast(_Float16, (uint16_t) max_sdata[idx_a]), max_b = __builtin_bit_cast(_Float16, (uint16_t) max_sdata[idx_b]);
      _Float16 next_max = fmaxf(max_a, max_b);
      _Float16 denom = denom_sdata[idx_a] * mu_fexp(max_a - next_max) + denom_sdata[idx_b] * mu_fexp(max_b - next_max);
      denom_sdata[tid] = (uint32_t) __builtin_bit_cast(uint16_t, denom);
      max_sdata[tid] = (uint32_t) __builtin_bit_cast(uint16_t, next_max);
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
  uint32_t row_elems_fp32 = args->cols / 2;
  uint32_t block_elem_idx = threadblock_id * row_elems_fp32;
  uint32_t chunks_per_block = (row_elems + DOUBLE_BLOCK_SIZE - 1) / DOUBLE_BLOCK_SIZE;

  __global uint32_t *x = args->x + block_elem_idx;
  __shared uint32_t *x_sdata = sdata;
  __shared uint32_t *max_sdata = sdata + row_elems;
  __shared uint32_t *denom_sdata = max_sdata + DOUBLE_BLOCK_SIZE;
  
  // pass 1 max + denom
  uint32_t x_fp32 = x[tid];
  x_sdata[tid] = x_fp32;
  _Float16 x0 = __builtin_bit_cast(_Float16, (uint16_t) (x_fp32 >> 16));
  _Float16 x1 = __builtin_bit_cast(_Float16, (uint16_t) (x_fp32 & 0xFFFF));
  _Float16 max = fmaxf(x0, x1);
  _Float16 denom = mu_fexp(x0 - max) + mu_fexp(x1 - max);

  for (uint32_t chunk = 1; chunk < chunks_per_block; chunk++) {
    uint32_t idx = chunk * BLOCK_SIZE + tid;
    if (idx >= row_elems_fp32) break;
    x_fp32 = x[idx];
    x_sdata[idx] = x_fp32;
    x0 = __builtin_bit_cast(_Float16, (uint16_t) (x_fp32 >> 16));
    x1 = __builtin_bit_cast(_Float16, (uint16_t) (x_fp32 & 0xFFFF));
    _Float16 next_max = fmaxf(fmaxf(x0, x1), max);
    denom = denom * mu_fexp(max - next_max) + mu_fexp(x0 - next_max) + mu_fexp(x1 - next_max);
    max = next_max;
  }

  max_sdata[tid] = (uint32_t) __builtin_bit_cast(uint16_t, max);
  denom_sdata[tid] = (uint32_t) __builtin_bit_cast(uint16_t, denom);

  // warp reduce
  reduce<MU_NUM_THREADS>(max_sdata, denom_sdata, tid, lane_id);

  // only works because num_warps = num_lanes
  if (lane_id == 0) {
    denom_sdata[warp_id] = denom_sdata[tid];
    max_sdata[warp_id] = max_sdata[tid];
  }

  mu_barrier(0, BLOCK_NUM_WARPS);

  // block reduce
  if (warp_id == 0) {
    reduce<MU_NUM_THREADS / THREAD_DIV>(max_sdata, denom_sdata, tid, lane_id);
    if (lane_id == 0)
      denom_sdata[0] = (uint32_t) __builtin_bit_cast(uint16_t, __builtin_bit_cast(_Float16, ONE_BF16_BITS) / __builtin_bit_cast(_Float16, (uint16_t) denom_sdata[0]));
  }

  mu_barrier(0, BLOCK_NUM_WARPS);

  // max and denom in tid 0
  _Float16 m = __builtin_bit_cast(_Float16, (uint16_t) max_sdata[0]), inv_d = __builtin_bit_cast(_Float16, (uint16_t) denom_sdata[0]);

  // pass 2 compute softmax for each element chunk by chunk
  for (uint32_t chunk = 0; chunk < chunks_per_block; chunk++) {
    uint32_t idx = chunk * BLOCK_SIZE + tid;
    if (idx >= row_elems_fp32) break;
    x_fp32 = x_sdata[idx];
    x0 = __builtin_bit_cast(_Float16, (uint16_t) (x_fp32 >> 16));
    x1 = __builtin_bit_cast(_Float16, (uint16_t) (x_fp32 & 0xFFFF));
    uint32_t x0_new = (uint32_t) __builtin_bit_cast(uint16_t, mu_fexp(x0 - m) * inv_d);
    uint32_t x1_new = (uint32_t) __builtin_bit_cast(uint16_t, mu_fexp(x1 - m) * inv_d);
    uint32_t x_new = x0_new | (x1_new << 16);
    x[idx] = x_new;
  }
}

SoftmaxArgs softmax_args = {
  .x = nullptr,
  .rows = 0,
  .cols = 0,
};

#include "data"

int main() {
  softmax_args.x = reinterpret_cast<__global uint32_t*>(x_raw);
  softmax_args.rows = rows;
  softmax_args.cols = cols;
  mu_schedule(softmax, &softmax_args, NUM_WARPS);
  return 0;
}
