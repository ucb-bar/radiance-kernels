#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

// Constraints on parameters:
// * Memory:
//   (BM + BN) * BK * sizeof(float) <= sharedmem size.
//   BM * BK == BN * BK >= threadblock size >= NT * CORES_PER_CLUSTER
//     When larger, the kernel runs a sequential loop to read into sharedmem;
//     but smaller case is not handled.
// * Compute:
//   ( M* N) / (TM*TN) == grid size >= NC*NW*NT
//   (BM*BN) / (TM*TN) == threadblock size < NT * NW * CORES_PER_CLUSTER
//   (BM*BN) / (TM*TN) == threadblock size >= NT * CORES_PER_CLUSTER
// * Combining BM * BK >= (BM*BN) / (TM*TN) == threadblock yields
//   BM <= BK*TM*TN
#define BM 32
#define BN BM
#define BK 8
#define TM 4
#define TN 4

void threadblock_barrier(unsigned int tid_in_threadblock, unsigned int barrier_id, unsigned int count) {
    vx_fence();
    vx_barrier(barrier_id, count);
}

void thread_block_gemm(kernel_arg_t *__UNIFORM__ arg,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threadblock_dim_x,
                              const uint32_t threadblock_dim_y,
                              const uint32_t threadblock_id_x,
                              const uint32_t threadblock_id_y,
                              const uint32_t threadblock_id_in_cluster,
                              float *sharedmem_per_threadblock) {
  const float *A = (const float *)arg->addr_a;
  const float *B = (const float *)arg->addr_b;
  float *C = (float *)arg->addr_c;

  // assumes NT == NW == matrix_dim
  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;

  // FIXME: Output block size is assumed to be square, i.e. BM == BN
  // const uint32_t BM = threadblock_dim_y;
  // const uint32_t BN = threadblock_dim_y;
  // const uint32_t BK = threadblock_dim_x;
  // constexpr uint32_t BM = 8;
  // constexpr uint32_t BN = 8;
  // constexpr uint32_t BK = 2;

  const uint32_t local_a_row = tid_in_threadblock / BK;
  const uint32_t local_a_col = tid_in_threadblock % BK;
  const uint32_t local_b_row = tid_in_threadblock / BN;
  const uint32_t local_b_col = tid_in_threadblock % BN;
  const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
  const uint32_t global_b_col = BN * threadblock_id_x + local_b_col;

  const uint32_t local_c_row = tid_in_threadblock / (BN / TN);
  const uint32_t local_c_col = tid_in_threadblock % (BN / TN);

  // each thread generates TM output element
  float reg_c[TM * TN] = { 0.0f };
  float reg_a[TM] = { 0.0f };
  float reg_b[TN] = { 0.0f };

  volatile float *local_a = sharedmem_per_threadblock;
  // const size_t local_a_elems = threadblock_dim_x * threadblock_dim_y;
  const size_t local_a_elems = (BM * BK);
  volatile float *local_b = sharedmem_per_threadblock + local_a_elems;

  constexpr uint32_t stride_a = (BM * BN) / BK / (TM * TN);
  constexpr uint32_t stride_b = (BM * BN) / BN / (TM * TN);

  for (uint32_t k = 0; k < dim_k; k += BK) {
    // Data move from GMEM to SMEM
    //
    // Make sure global offset values for A and B are contiguous between
    // neighboring threads to ensure GMEM coalescing.
#pragma GCC unroll 2
    for (uint32_t load_offset = 0; load_offset < BM; load_offset += stride_a) {
      const uint32_t global_a_offset =
          dim_k * (global_a_row + load_offset) + (k + local_a_col);
      local_a[BK * (local_a_row + load_offset) + local_a_col] =
          A[global_a_offset];
    }
#pragma GCC unroll 2
    for (uint32_t load_offset = 0; load_offset < BK; load_offset += stride_b) {
      const uint32_t global_b_offset =
          dim_n * (k + local_b_row + load_offset) + global_b_col;
      local_b[BN * (local_b_row + load_offset) + local_b_col] =
          B[global_b_offset];
    }

    threadblock_barrier(tid_in_threadblock, threadblock_id_in_cluster,
                        threadblock_dim_y);

    // Compute single tile*tile matmul
#pragma GCC unroll 4
    for (uint32_t local_k = 0; local_k < BK; local_k++) {
      // First, pump data from SMEM->RF
#pragma GCC unroll TM
      for (uint32_t res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
        reg_a[res_idx_m] =
            local_a[BK * (TM * local_c_row + res_idx_m) + local_k];
      }
#pragma GCC unroll TN
      for (uint32_t res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
        reg_b[res_idx_n] =
            local_b[BN * local_k + (TN * local_c_col + res_idx_n)];
      }

      // Next, compute multiple result elements (TM*TN) by reusing data in RF
#pragma GCC unroll TM
      for (uint32_t res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
#pragma GCC unroll TN
        for (uint32_t res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
          // NOTE use of local_b_row
          reg_c[TN * res_idx_m + res_idx_n] +=
              reg_a[res_idx_m] * reg_b[res_idx_n];
          // reg_c[TN * res_idx_m + res_idx_n] +=
          //     local_a[BK * (TM * local_c_row + res_idx_m) + local_k] *
          //     local_b[BN * local_k + (TN * local_c_col + res_idx_n)];
        }
      }
    }

    threadblock_barrier(tid_in_threadblock, threadblock_id_in_cluster,
                        threadblock_dim_y);
  }

  // Store result data from RF to GMEM
#pragma GCC unroll TM
  for (uint32_t res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
#pragma GCC unroll TN
    for (uint32_t res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
      C[dim_n * (BM * threadblock_id_y + TM * local_c_row + res_idx_m) +
        (BN * threadblock_id_x + TN * local_c_col + res_idx_n)] =
          reg_c[TN * res_idx_m + res_idx_n];
    }
  }
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

  const uint32_t threads_per_threadblock = (BM * BN) / (TM * TN);
#ifdef RADIANCE
  const uint32_t threadblocks_per_core = vx_num_threads() * vx_num_warps() /
                                         threads_per_threadblock *
                                         CORES_PER_CLUSTER;
#else
  const uint32_t threadblocks_per_core =
      vx_num_threads() * vx_num_warps() / threads_per_threadblock;
#endif
  const uint32_t threadblock_dim_x = vx_num_threads();
  const uint32_t threadblock_dim_y = vx_num_warps() / threadblocks_per_core;
  const int threadblock_id = task_id / threads_per_threadblock;
  const int threadblock_id_in_cluster = threadblock_id % threadblocks_per_core;
  const int tid_in_threadblock = task_id % threads_per_threadblock;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_n_in_blocks = dim_n / BN;
  const int threadblock_id_x = threadblock_id % dim_n_in_blocks;
  const int threadblock_id_y = threadblock_id / dim_n_in_blocks;

  // "static" shared memory allocation.  This would determine threadblock
  // occupancy of a single cluster
  float *sharedmem_per_threadblock =
      (float *)DEV_SMEM_START_ADDR + (2 * BM * BK) * threadblock_id_in_cluster;
  thread_block_gemm(arg, tid_in_threadblock, threadblock_dim_x,
                    threadblock_dim_y, threadblock_id_x, threadblock_id_y,
                    threadblock_id_in_cluster, sharedmem_per_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;
  const uint32_t grid_size = arg->dim_m * arg->dim_n / (TM * TN);
#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
