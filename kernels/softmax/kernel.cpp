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
static inline void reduce(__shared uint32_t *buf_sdata, uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MAX_STRIDE; stride *= 2) {
    if (lane_id % stride == 0) {
      uint32_t idx_a = tid, idx_b = (tid + (stride >> 1));
      auto [max_a, denom_a] = unpack_bf16x2(buf_sdata[idx_a]);
      auto [max_b, denom_b] = unpack_bf16x2(buf_sdata[idx_b]);
      _Float16 max = fmaxf(max_a, max_b);
      _Float16 denom = denom_a * mu_fexp(max_a - max) + denom_b * mu_fexp(max_b - max);
      buf_sdata[tid] = pack_bf16x2(max, denom);
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
  
  uint32_t rows_per_block = args->rows / MU_NUM_CLUSTERS;
  uint32_t row_elems = args->cols;
  uint32_t row_elems_fp32 = args->cols / 2;

  #pragma unroll 8
  for (uint32_t row = 0; row < rows_per_block; row++) {
    uint32_t block_elem_idx = (threadblock_id * rows_per_block + row) * row_elems_fp32;
    uint32_t chunks_per_block = (row_elems + DOUBLE_BLOCK_SIZE - 1) / DOUBLE_BLOCK_SIZE;

    __global uint32_t *x = args->x + block_elem_idx;
    __shared uint32_t *x_sdata = sdata;
    __shared uint32_t *buf_sdata = sdata + row_elems;
    
    // pass 1 max + denom
    uint32_t x_fp32 = x[tid];
    x_sdata[tid] = x_fp32;
    auto [x1, x0] = unpack_bf16x2(x_fp32);
    _Float16 max = fmaxf(x0, x1);
    _Float16 denom = mu_fexp(x0 - max) + mu_fexp(x1 - max);

    for (uint32_t chunk = 1; chunk < chunks_per_block; chunk++) {
      uint32_t idx = chunk * BLOCK_SIZE + tid;
      if (idx >= row_elems_fp32) break;
      x_fp32 = x[idx];
      x_sdata[idx] = x_fp32;
      auto [x1_next, x0_next] = unpack_bf16x2(x_fp32);
      _Float16 next_max = fmaxf(fmaxf(x0_next, x1_next), max);
      denom = denom * mu_fexp(max - next_max) + mu_fexp(x0_next - next_max) + mu_fexp(x1_next - next_max);
      max = next_max;
    }

    buf_sdata[tid] = pack_bf16x2(max, denom);

    // warp reduce
    reduce<MU_NUM_THREADS>(buf_sdata, tid, lane_id);

    // only works because num_warps = num_lanes
    if (lane_id == 0) {
      buf_sdata[warp_id] = buf_sdata[tid];
    }

    mu_barrier(0, BLOCK_NUM_WARPS);

    // block reduce
    if (warp_id == 0) {
      reduce<MU_NUM_THREADS / THREAD_DIV>(buf_sdata, tid, lane_id);
      if (lane_id == 0) {
        auto [max_final, denom_final] = unpack_bf16x2(buf_sdata[0]);
        _Float16 inv_denom = as_bf16(ONE_BF16_BITS) / denom_final;
        buf_sdata[0] = pack_bf16x2(max_final, inv_denom);
      }
    }

    mu_barrier(0, BLOCK_NUM_WARPS);

    // max and inv_denom in tid 0
    auto [m, inv_d] = unpack_bf16x2(buf_sdata[0]);

    // pass 2 compute softmax for each element chunk by chunk
    for (uint32_t chunk = 0; chunk < chunks_per_block; chunk++) {
      uint32_t idx = chunk * BLOCK_SIZE + tid;
      if (idx >= row_elems_fp32) break;
      auto [lo, hi] = unpack_bf16x2(x_sdata[idx]);
      x[idx] = pack_bf16x2(mu_fexp(lo - m) * inv_d, mu_fexp(hi - m) * inv_d);
    }

    mu_barrier(0, BLOCK_NUM_WARPS);
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
