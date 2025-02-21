#define RISCV_CUSTOM3   0x7B

#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "util.hpp"

#define DOUBLE_BUFFER 1
#undef ELEM_PER_THREAD
#define ELEM_PER_THREAD (WMITER * WNITER * ((TCM * TCN) / NUM_THREADS) / (DOUBLE_BUFFER ? 2 : 1))

// FIXME: NUM_THREADS and NUM_WARPS hardcoded
#if ((BM * BN / ELEM_PER_THREAD) > (CORES_PER_CLUSTER * 8 * 8))
#error "threadblock size too big for cluster"
#endif

inline void global_dmem_load(const uint32_t dim_n, const uint32_t dim_k,
                             const uint32_t k, const float *A, const float *B,
                             volatile float *local_a, volatile float *local_b,
                             const uint32_t tid_in_threadblock,
                             const uint32_t threadblock_id_x,
                             const uint32_t threadblock_id_y) {
  const uint32_t local_a_row = tid_in_threadblock / BK;
  const uint32_t local_a_col = tid_in_threadblock % BK;
  const uint32_t local_as_row = tid_in_threadblock / BM;
  const uint32_t local_as_col = tid_in_threadblock % BM;
  const uint32_t local_b_row = tid_in_threadblock / BN;
  const uint32_t local_b_col = tid_in_threadblock % BN;

  constexpr uint32_t threads_in_warpgroup =
      (BM * BN) / ELEM_PER_THREAD / (DOUBLE_BUFFER ? 2 : 1); // FIXME

  // Data move from GMEM to SMEM
  //
  // Make sure global offset values for A and B are contiguous between
  // neighboring threads to ensure GMEM coalescing.
  //
  // TODO: Sharedmem swizzling is important here
  if constexpr (!TRANSPOSE_AT_PRODUCE) {
    // if !TRANSPOSE_AT_PRODUCE, we only support coalesced GMEM loads
    static_assert(TRANSPOSE_AT_PRODUCE || GMEM_COALESCED_A);

    const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
    // number of rows a full TB can read at a time
    constexpr uint32_t row_stride_a = threads_in_warpgroup / BK;
    const float *global_a = A + dim_k * global_a_row + (k + local_a_col);
    volatile float *local_a_tmp = local_a + BK * local_a_row + local_a_col;

    static_assert(
        row_stride_a * 8 <= BM,
        "manual loop unrolling condition not met; consider increasing BM");
    static_assert(
        (BM % (row_stride_a * 8)) == 0,
        "manual loop unrolling condition not met; BM should be power-of-two");

#pragma GCC unroll 1
    for (uint32_t local_row_offset = 0; local_row_offset < BM;
         local_row_offset += row_stride_a * 8) {
      // const uint32_t global_a_offset =
      //     dim_k * (global_a_row + local_row_offset) + (k + local_a_col);
      // local_a[BK * (local_a_row + local_row_offset) + local_a_col] =
      //     A[global_a_offset];
      //
      // *local_a_tmp = *global_a;
      // global_a += dim_k * row_stride_a;
      // local_a_tmp += BK * row_stride_a;

      asm volatile ("flw ft0, (%0)"   :: "r"(global_a));
      global_a += dim_k * row_stride_a;
      asm volatile ("flw ft1, (%0)"   :: "r"(global_a));
      global_a += dim_k * row_stride_a;
      asm volatile ("flw ft2, (%0)"   :: "r"(global_a));
      global_a += dim_k * row_stride_a;
      asm volatile ("flw ft3, (%0)"   :: "r"(global_a));
      global_a += dim_k * row_stride_a;
      asm volatile ("flw ft4, (%0)"   :: "r"(global_a));
      global_a += dim_k * row_stride_a;
      asm volatile ("flw ft5, (%0)"   :: "r"(global_a));
      global_a += dim_k * row_stride_a;
      asm volatile ("flw ft6, (%0)"   :: "r"(global_a));
      global_a += dim_k * row_stride_a;
      asm volatile ("flw ft7, (%0)"   :: "r"(global_a));
      global_a += dim_k * row_stride_a;

      // stride along columns
      // bank conflicts
      asm volatile ("fsw ft0, %0(%1)" :: "i"(BK * row_stride_a * 0 * sizeof(float)), "r"(local_a_tmp));
      asm volatile ("fsw ft1, %0(%1)" :: "i"(BK * row_stride_a * 1 * sizeof(float)), "r"(local_a_tmp));
      asm volatile ("fsw ft2, %0(%1)" :: "i"(BK * row_stride_a * 2 * sizeof(float)), "r"(local_a_tmp));
      asm volatile ("fsw ft3, %0(%1)" :: "i"(BK * row_stride_a * 3 * sizeof(float)), "r"(local_a_tmp));
      local_a_tmp += BK * row_stride_a * 4;
      asm volatile ("fsw ft4, %0(%1)" :: "i"(BK * row_stride_a * 0 * sizeof(float)), "r"(local_a_tmp));
      asm volatile ("fsw ft5, %0(%1)" :: "i"(BK * row_stride_a * 1 * sizeof(float)), "r"(local_a_tmp));
      asm volatile ("fsw ft6, %0(%1)" :: "i"(BK * row_stride_a * 2 * sizeof(float)), "r"(local_a_tmp));
      asm volatile ("fsw ft7, %0(%1)" :: "i"(BK * row_stride_a * 3 * sizeof(float)), "r"(local_a_tmp));
      local_a_tmp += BK * row_stride_a * 4;
    }
  } else {
    if constexpr (!GMEM_COALESCED_A) {
      constexpr uint32_t row_stride_as = threads_in_warpgroup / BM;
      const uint32_t global_a_row = BM * threadblock_id_y + local_as_col;
      // NOTE that GMEM reads are transposed
      const float *global_a = A + dim_k * global_a_row + (k + local_as_row);
      volatile float *local_a_tmp = local_a + BM * local_as_row + local_as_col;

      static_assert(
          row_stride_as * 8 <= BK,
          "manual loop unrolling condition not met; consider increasing BK");
      static_assert(
          (BK % (row_stride_as * 8)) == 0,
          "manual loop unrolling condition not met; BK should be power-of-two");

#pragma GCC unroll 1
      for (uint32_t local_row_offset = 0; local_row_offset < BK;
           local_row_offset += row_stride_as * 8) {
        // @perf: bank conflicts here
        // const uint32_t global_a_offset =
        //     dim_k * (global_a_row) + (k + local_as_row + local_row_offset);
        // FIXME experimenting with global coalescing
        // const uint32_t global_a_offset =
        //     dim_k * (global_a_row + local_row_offset) + (k + local_as_col);
        // local_a[BM * (local_as_row + local_row_offset) + local_as_col] =
        //     A[global_a_offset];

        // *local_a_tmp = *global_a;
        asm volatile ("flw ft0, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft1, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft2, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft3, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft4, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft5, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft6, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;
        asm volatile ("flw ft7, (%0)"   :: "r"(global_a));
        global_a += row_stride_as;

        asm volatile ("fsw ft0, %0(%1)" :: "i"(BM * row_stride_as * 0 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft1, %0(%1)" :: "i"(BM * row_stride_as * 1 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft2, %0(%1)" :: "i"(BM * row_stride_as * 2 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft3, %0(%1)" :: "i"(BM * row_stride_as * 3 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft4, %0(%1)" :: "i"(BM * row_stride_as * 4 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft5, %0(%1)" :: "i"(BM * row_stride_as * 5 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft6, %0(%1)" :: "i"(BM * row_stride_as * 6 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft7, %0(%1)" :: "i"(BM * row_stride_as * 7 * sizeof(float)), "r"(local_a_tmp));
        local_a_tmp += BM * row_stride_as * 8;
      }
    } else {
      constexpr uint32_t row_stride_a = threads_in_warpgroup / BK;
      const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
      const float *global_a = A + dim_k * global_a_row + (k + local_a_col);
      // NOTE that SMEM writes are transposed
      volatile float *local_a_tmp = local_a + BM * local_a_col + local_a_row;

      static_assert(
          row_stride_a * 8 <= BM,
          "manual loop unrolling condition not met; consider increasing BM");
      static_assert(
          (BM % (row_stride_a * 8)) == 0,
          "manual loop unrolling condition not met; BM should be power-of-two");

#pragma GCC unroll 1
      for (uint32_t local_row_offset = 0; local_row_offset < BM;
           local_row_offset += row_stride_a * 8) {
        // const uint32_t global_a_offset =
        //     dim_k * (global_a_row + local_row_offset) + (k + local_a_col);
        // NOTE that SMEM writes are transposed
        // local_a[BM * (local_a_col) + local_a_row + local_row_offset] =
        //     A[global_a_offset];

        asm volatile ("flw ft0, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft1, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft2, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft3, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft4, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft5, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft6, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;
        asm volatile ("flw ft7, (%0)"   :: "r"(global_a));
        global_a += dim_k * row_stride_a;

        // stride along columns
        // bank conflicts
        asm volatile ("fsw ft0, %0(%1)" :: "i"(row_stride_a * 0 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft1, %0(%1)" :: "i"(row_stride_a * 1 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft2, %0(%1)" :: "i"(row_stride_a * 2 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft3, %0(%1)" :: "i"(row_stride_a * 3 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft4, %0(%1)" :: "i"(row_stride_a * 4 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft5, %0(%1)" :: "i"(row_stride_a * 5 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft6, %0(%1)" :: "i"(row_stride_a * 6 * sizeof(float)), "r"(local_a_tmp));
        asm volatile ("fsw ft7, %0(%1)" :: "i"(row_stride_a * 7 * sizeof(float)), "r"(local_a_tmp));
        local_a_tmp += row_stride_a * 8;
      }
    }
  }

  constexpr uint32_t row_stride_b = threads_in_warpgroup / BN;
  const uint32_t global_b_col = BN * threadblock_id_x + local_b_col;
  const float *global_b = B + dim_n * (k + local_b_row) + global_b_col;
  volatile float *local_b_tmp = local_b + BN * local_b_row + local_b_col;

  static_assert(
      row_stride_b * 8 <= BK,
      "manual loop unrolling condition not met; consider increasing BK");
  static_assert(
      (BK % (row_stride_b * 8)) == 0,
      "manual loop unrolling condition not met; BK should be power-of-two");

#pragma GCC unroll 1
  for (uint32_t load_offset = 0; load_offset < BK;
       load_offset += row_stride_b * 8) {
    // const uint32_t global_b_offset =
    //     dim_n * (k + local_b_row + load_offset) + global_b_col;
    // local_b[BN * (local_b_row + load_offset) + local_b_col] =
    //     B[global_b_offset];

    // *local_b_tmp = *global_b;

    // global_b += dim_n * row_stride_b;
    // local_b_tmp += BN * row_stride_b;

    asm volatile ("flw ft0, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft1, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft2, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft3, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft4, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft5, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft6, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;
    asm volatile ("flw ft7, (%0)"   :: "r"(global_b));
    global_b += dim_n * row_stride_b;

    asm volatile ("fsw ft0, %0(%1)" :: "i"(BN * row_stride_b * 0 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft1, %0(%1)" :: "i"(BN * row_stride_b * 1 * sizeof(float)), "r"(local_b_tmp));
    local_b_tmp += BN * row_stride_b * 2;
    asm volatile ("fsw ft2, %0(%1)" :: "i"(BN * row_stride_b * 0 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft3, %0(%1)" :: "i"(BN * row_stride_b * 1 * sizeof(float)), "r"(local_b_tmp));
    local_b_tmp += BN * row_stride_b * 2;
    asm volatile ("fsw ft4, %0(%1)" :: "i"(BN * row_stride_b * 0 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft5, %0(%1)" :: "i"(BN * row_stride_b * 1 * sizeof(float)), "r"(local_b_tmp));
    local_b_tmp += BN * row_stride_b * 2;
    asm volatile ("fsw ft6, %0(%1)" :: "i"(BN * row_stride_b * 0 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft7, %0(%1)" :: "i"(BN * row_stride_b * 1 * sizeof(float)), "r"(local_b_tmp));
    local_b_tmp += BN * row_stride_b * 2;
  }
}

inline void thread_block_gemm(kernel_arg_t *__UNIFORM__ arg,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threads_per_threadblock,
                              const uint32_t threadblock_dim_y,
                              /*const uint32_t threadblock_id_x,
                              const uint32_t threadblock_id_y,*/
                              const uint32_t threadblocks_per_cluster,
                              const uint32_t threadblock_id_in_cluster,
                              float *sharedmem_per_threadblock) {
  const float *A = (const float *)arg->addr_a;
  const float *B = (const float *)arg->addr_b;
  float *C = (float *)arg->addr_c;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;

  const uint32_t local_a_row = tid_in_threadblock / BK;
  const uint32_t local_a_col = tid_in_threadblock % BK;
  const uint32_t local_as_row = tid_in_threadblock / BM;
  const uint32_t local_as_col = tid_in_threadblock % BM;
  const uint32_t local_b_row = tid_in_threadblock / BN;
  const uint32_t local_b_col = tid_in_threadblock % BN;

  const uint32_t threads_per_warpgroup = threads_per_threadblock / (DOUBLE_BUFFER ? 2 : 1);
  const uint32_t warpgroup_id = tid_in_threadblock / threads_per_warpgroup;
  const uint32_t tid_in_warpgroup = tid_in_threadblock % threads_per_warpgroup; // FIXME
  const uint32_t warp_in_warpgroup = tid_in_warpgroup / NUM_THREADS;
  // FIXME: warp_row / BN should be warp-specialized?
  const uint32_t warp_row = warp_in_warpgroup / (BN / WN);
  const uint32_t warp_col = warp_in_warpgroup % (BN / WN);
  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;

  volatile float *local_a = sharedmem_per_threadblock;
  // const size_t local_a_elems = threadblock_dim_x * threadblock_dim_y;
  constexpr size_t local_a_elems = (BM * BK);
  volatile float *local_b = sharedmem_per_threadblock + local_a_elems;
  constexpr size_t local_b_elems = (BK * BN);

  volatile float *local_a_buf = local_b + local_b_elems;
  volatile float *local_b_buf = local_a_buf + local_a_elems;

  // divide rows (M) by the number of threadblocks
  // FIXME: doesn't work with multiple clusters
  const uint32_t dim_m_range = (dim_m / threadblocks_per_cluster);
  const uint32_t dim_m_start = dim_m_range * threadblock_id_in_cluster;
  const uint32_t block_m_start = dim_m_start / BM;
  const uint32_t block_m_end = (dim_m_start + dim_m_range) / BM;

  if (warpgroup_id == 0) {
    // producer code: GMEM->SMEM data movement
#pragma GCC unroll 1
    for (uint32_t block_m = block_m_start; block_m < block_m_end; block_m++) {
#pragma GCC unroll 1
      for (uint32_t block_n = 0; (block_n * BN) < dim_n; block_n++) {
        if constexpr (DOUBLE_BUFFER) {
          // initiate software pipeline
          global_dmem_load(dim_n, dim_k, 0 /*k*/, A, B, local_a, local_b,
                           tid_in_warpgroup, block_n, block_m);

          threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
        }

        // NOTE: this *should* be signed integer to trigger arithmetic
        // right-shift
        int32_t k_index = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < (dim_k) - BK; k += BK) {
          volatile float *local_a_produce;
          volatile float *local_b_produce;
          if constexpr (DOUBLE_BUFFER) {
            const uint32_t mask_odd = (k_index & 1) << 31 >> 31;
            const uint32_t mask_even = ((k_index & 1) ^ 1) << 31 >> 31;
            // local_a_produce = (k_index % 2) ? local_a : local_a_buf;
            // local_b_produce = (k_index % 2) ? local_b : local_b_buf;
            local_a_produce = reinterpret_cast<volatile float *>(
                (mask_odd & reinterpret_cast<uintmax_t>(local_a)) |
                (mask_even & reinterpret_cast<uintmax_t>(local_a_buf)));
            local_b_produce = reinterpret_cast<volatile float *>(
                (mask_odd & reinterpret_cast<uintmax_t>(local_b)) |
                (mask_even & reinterpret_cast<uintmax_t>(local_b_buf)));
          } else {
            local_a_produce = local_a;
            local_b_produce = local_b;
          }
          k_index++;

          global_dmem_load(dim_n, dim_k, k + BK /*runahead*/, A, B,
                           local_a_produce, local_b_produce, tid_in_warpgroup,
                           block_n, block_m);

          threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
        }

        // sync with final consumer stage in the k-loop
        threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
      }
    }
  } else {
    // consumer code: SMEM->RF and compute

#pragma GCC unroll 1
    for (uint32_t block_m = block_m_start; block_m < block_m_end; block_m++) {
#pragma GCC unroll 1
      for (uint32_t block_n = 0; (block_n * BN) < dim_n; block_n++) {
        // clear out C
        initialize_C(0);
        initialize_C(1);

        // sync with initial producer stage in the k-loop
        threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);

        // NOTE: this *should* be signed integer to trigger arithmetic
        // right-shift
        int32_t k_index = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < (dim_k); k += BK) {
          const volatile float *local_a_consume;
          const volatile float *local_b_consume;
          if constexpr (DOUBLE_BUFFER) {
            // local_a_consume = (k_index % 2) ? local_a_buf : local_a;
            // local_b_consume = (k_index % 2) ? local_b_buf : local_b;
            // FIXME: swap multiply with bitshifts
            const uint32_t mask_odd = (k_index & 1) << 31 >> 31;
            const uint32_t mask_even = ((k_index & 1) ^ 1) << 31 >> 31;
            local_a_consume = reinterpret_cast<volatile float *>(
                (mask_odd & reinterpret_cast<uintmax_t>(local_a_buf)) |
                (mask_even & reinterpret_cast<uintmax_t>(local_a)));
            local_b_consume = reinterpret_cast<volatile float *>(
                (mask_odd & reinterpret_cast<uintmax_t>(local_b_buf)) |
                (mask_even & reinterpret_cast<uintmax_t>(local_b)));
          } else {
            local_a_consume = local_a;
            local_b_consume = local_b;
          }
          k_index++;

          // @perf: this loop spills to stack a lot because of all the flws in
#pragma GCC unroll 1
          for (int i = 0; i < BK_LOOP; i++) {
#pragma GCC unroll 2
            for (uint32_t local_k = 0; local_k < BK; local_k += TCK) {
#pragma GCC unroll 2
              for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
                // SMEM -> RF
                vx_wmma_load_b(local_b_consume, local_k, warp_col, wn_iter,
                               tid_in_warp);
#pragma GCC unroll 2
                for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
                  // SMEM -> RF
                  vx_wmma_load_a(local_a_consume, local_k, warp_row, wm_iter,
                                 tid_in_warp);
                  // perform mma
                  vx_wmma(wm_iter);
                }
              }
            }
          }

          threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
        }

#pragma GCC unroll 1
        for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll 1
          for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
            if (warpgroup_id == 1) {
              write_results(tid_in_warp, warp_col, warp_row, wn_iter, wm_iter,
                            dim_n, C, block_n, block_m);
            }
          }
        }
      }
    }
  }
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

#ifdef RADIANCE
  constexpr uint32_t cores_per_cluster = CORES_PER_CLUSTER;
#else
  constexpr uint32_t cores_per_cluster = 1;
#endif

  uint32_t threads_per_threadblock = (BM * BN) / (ELEM_PER_THREAD);
  const uint32_t hw_threads_per_cluster =
      cores_per_cluster * vx_num_threads() * vx_num_warps();
  // cap maximum threadblock size to # of HW threads in cluster, to prevent
  // multiple "wave" invocations which slows down the kernel
  if (threads_per_threadblock > hw_threads_per_cluster) {
    threads_per_threadblock = hw_threads_per_cluster;
  }
  const uint32_t threadblocks_per_cluster =
      hw_threads_per_cluster / threads_per_threadblock;

  const uint32_t threadblock_dim_y = vx_num_warps() / threadblocks_per_cluster;
  const int threadblock_id = task_id / threads_per_threadblock;
  const int threadblock_id_in_cluster =
      threadblock_id % threadblocks_per_cluster;
  const int tid_in_threadblock = task_id % threads_per_threadblock;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_n_in_blocks = dim_n / BN;
  const int threadblock_id_x = threadblock_id % dim_n_in_blocks;
  const int threadblock_id_y = threadblock_id / dim_n_in_blocks;
  const uint32_t problem_size = (dim_m * dim_n) / (ELEM_PER_THREAD);
  const uint32_t num_threadblocks = problem_size / threads_per_threadblock;

  // "static" shared memory allocation.  This would determine threadblock
  // occupancy of a single cluster
  float *sharedmem_per_threadblock =
      (float *)DEV_SMEM_START_ADDR +
      2 /*double-buffering*/ * (2 * BM * BK) * threadblock_id_in_cluster;

  thread_block_gemm(arg, tid_in_threadblock, threads_per_threadblock,
                    threadblock_dim_y,
                    /*threadblock_id_x, threadblock_id_y,*/
                    threadblocks_per_cluster,
                    // threadblock_id,
                    threadblock_id_in_cluster,
                    sharedmem_per_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  const uint32_t problem_size = (arg->dim_m * arg->dim_n) / (ELEM_PER_THREAD);
  const uint32_t hw_threads_per_cluster =
      CORES_PER_CLUSTER * vx_num_threads() * vx_num_warps();
  // prevent launching more threads than the necessary problem size
  // TODO: this does not take into account multiple clusters
  const uint32_t grid_size = (problem_size > hw_threads_per_cluster)
                                 ? hw_threads_per_cluster
                                 : problem_size;

#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
