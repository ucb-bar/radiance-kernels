#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#define NUM_WARPS 4
#define ILP 4
#define DOUBLE_BLOCK_SIZE MU_DOUBLE_BLOCK_SIZE(NUM_WARPS)
#define BLOCK_SIZE MU_BLOCK_SIZE(NUM_WARPS)
#define BLOCK_NUM_WARPS MU_BLOCK_NUM_WARPS(NUM_WARPS)
// only works for powers of 2 reduction, otherwise you have to manually write your own reduction
#define THREAD_DIV (MU_NUM_MAX_WARPS / NUM_WARPS)

extern "C" uint32_t __mu_num_warps = NUM_WARPS;

struct SoftmaxArgs {
  __global uint32_t* x;
  uint32_t rows;
  uint32_t cols;
};

__shared uint32_t* const sdata = reinterpret_cast<__shared uint32_t*>(0x0);

template <uint32_t MAX_STRIDE>
static inline void reduce_max(__shared uint32_t *buf_sdata, uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MAX_STRIDE; stride *= 2) {
    if (lane_id % stride == 0) {
      _Float16 a = as_bf16((uint16_t)buf_sdata[tid]);
      _Float16 b = as_bf16((uint16_t)buf_sdata[tid + (stride >> 1)]);
      buf_sdata[tid] = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)fmaxf(a, b));
    }
  }
}

template <uint32_t MAX_STRIDE>
static inline void reduce_sum(__shared uint32_t *buf_sdata, uint32_t tid, uint32_t lane_id) {
  for (uint32_t stride = 2; stride <= MAX_STRIDE; stride *= 2) {
    if (lane_id % stride == 0) {
      _Float16 a = as_bf16((uint16_t)buf_sdata[tid]);
      _Float16 b = as_bf16((uint16_t)buf_sdata[tid + (stride >> 1)]);
      buf_sdata[tid] = (uint32_t)__builtin_bit_cast(uint16_t, a + b);
    }
  }
}

// requires that cols + BLOCK_SIZE * 2 fits in smem (one row + max of one row + denom of one row)
// requires that you do NOT spawn a threadblock for a non existent row
static inline void softmax(
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

  #pragma unroll
  for (uint32_t row = 0; row < rows_per_block; row++) {
    uint32_t block_elem_idx = (threadblock_id * rows_per_block + row) * row_elems_fp32;
    uint32_t chunks_per_block = (row_elems + DOUBLE_BLOCK_SIZE - 1) / DOUBLE_BLOCK_SIZE;

    __global uint32_t *x = args->x + block_elem_idx;
    __shared uint32_t *x_sdata = sdata;
    __shared uint32_t *buf_sdata = sdata + row_elems;
    
    // pass 1: find max
    _Float16 max_acc[ILP];
    uint32_t x_fp32[ILP];
    #pragma unroll ILP
    for (int i = 0; i < ILP; i++)
      max_acc[i] = as_bf16(NEG_INF_BF16_BITS);

    #pragma unroll
    for (uint32_t chunk = 0; chunk < chunks_per_block; chunk += ILP) {
      #pragma unroll ILP
      for (int i = 0; i < ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        x_fp32[i] = x[idx];
      }
      for (int i = 0; i < ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        x_sdata[idx] = x_fp32[i];
      }
      for (int i = 0; i < ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        auto [x1, x0] = unpack_bf16x2(x_fp32[i]);
        max_acc[i] = fmaxf(fmaxf(x0, x1), max_acc[i]);
      }
    }

    _Float16 max = max_acc[0];
    #pragma unroll ILP
    for (int i = 1; i < ILP; i++)
      max = fmaxf(max, max_acc[i]);

    buf_sdata[tid] = (uint32_t)__builtin_bit_cast(uint16_t, max);

    // warp reduce max
    reduce_max<MU_NUM_THREADS>(buf_sdata, tid, lane_id);
    if (lane_id == 0)
      buf_sdata[warp_id] = buf_sdata[tid];
    mu_fence_smem();
    mu_barrier(0, BLOCK_NUM_WARPS);

    // block reduce max
    if (warp_id == 0)
      reduce_max<MU_NUM_THREADS / THREAD_DIV>(buf_sdata, tid, lane_id);
    mu_fence_smem();
    mu_barrier(0, BLOCK_NUM_WARPS);

    _Float16 m = as_bf16((uint16_t)buf_sdata[0]);

    // pass 2: compute denom with known max
    _Float16 denom_acc[ILP];
    #pragma unroll ILP
    for (int i = 0; i < ILP; i++)
      denom_acc[i] = 0;

    uint32_t xss[ILP];
    _Float16 x0s[ILP], x1s[ILP];
    #pragma unroll
    for (uint32_t chunk = 0; chunk < chunks_per_block; chunk += ILP) {
      #pragma unroll ILP
      for (int i = 0; i < ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        xss[i] = x_sdata[idx];
      }
      for (int i = 0; i < ILP; i++) {
        auto [x1, x0] = unpack_bf16x2(xss[i]);
        x0s[i] = x0; x1s[i] = x1;
      }
      for (int i = 0; i < ILP; i++) {
        denom_acc[i] += mu_fexp(x0s[i] - m) + mu_fexp(x1s[i] - m);
      }
    }

    _Float16 denom = denom_acc[0];
    #pragma unroll ILP
    for (int i = 1; i < ILP; i++)
      denom += denom_acc[i];

    buf_sdata[tid] = (uint32_t)__builtin_bit_cast(uint16_t, denom);

    // warp reduce denom
    reduce_sum<MU_NUM_THREADS>(buf_sdata, tid, lane_id);
    if (lane_id == 0)
      buf_sdata[warp_id] = buf_sdata[tid];
    mu_fence_smem();
    mu_barrier(0, BLOCK_NUM_WARPS);

    // block reduce denom
    if (warp_id == 0) {
      reduce_sum<MU_NUM_THREADS / THREAD_DIV>(buf_sdata, tid, lane_id);
      if (lane_id == 0) {
        _Float16 denom_final = as_bf16((uint16_t)buf_sdata[0]);
        _Float16 inv_denom = as_bf16(ONE_BF16_BITS) / denom_final;
        buf_sdata[0] = (uint32_t)__builtin_bit_cast(uint16_t, inv_denom);
      }
    }
    mu_fence_smem();
    mu_barrier(0, BLOCK_NUM_WARPS);

    _Float16 inv_d = as_bf16((uint16_t)buf_sdata[0]);

    // pass 3: compute softmax output
    uint32_t sh_ld[ILP], exps[ILP];
    _Float16 lows[ILP], his[ILP];
    #pragma unroll
    for (uint32_t chunk = 0; chunk < chunks_per_block; chunk += ILP) {
      #pragma unroll ILP
      for (uint32_t i = 0; i < ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        sh_ld[i] = x_sdata[idx];
      }
      #pragma unroll ILP
      for (uint32_t i = 0; i < ILP; i++) {
        auto [lo, hi] = unpack_bf16x2(sh_ld[i]);
        lows[i] = lo; his[i] = hi;
      }
      #pragma unroll ILP
      for (uint32_t i = 0; i < ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        exps[i] = pack_bf16x2(mu_fexp(lows[i] - m) * inv_d, mu_fexp(his[i] - m) * inv_d);
      }
      #pragma unroll ILP
      for (uint32_t i = 0; i < ILP; i++) {
        uint32_t idx = (chunk + i) * BLOCK_SIZE + tid;
        x[idx] = exps[i];
      }
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
