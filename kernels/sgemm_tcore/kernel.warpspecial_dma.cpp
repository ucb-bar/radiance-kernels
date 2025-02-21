#define RISCV_CUSTOM3   0x7B

#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define SMEM_ADDR_Q0 ((float * const) 0xff000000)
#define SMEM_ADDR_Q1 ((float * const) 0xff001000)
#define SMEM_ADDR_Q2 ((float * const) 0xff002000)
#define SMEM_ADDR_Q3 ((float * const) 0xff003000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x80
#define SPAD_ADDR_Q2 0x100
#define SPAD_ADDR_Q3 0x180

// number of loop around the inner 0..TCK..BK loop to simulate perfect-DRAM
// scenario
#define BK_LOOP 1
#define TRANSPOSE_AS 1
// GMEM_COALESCED sets bank conflict-free accesses for
// 1: GMEM loads of A matrix
// 0: SMEM stores of A matrix
#define GMEM_COALESCED_A 1

#define DOUBLE_BUFFER 1

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
#define BN 32
#define BK 32
#define WM 16
#define WN 8
#define TCM 8
#define TCN 8
#define TCK 8
#define WMITER (WM / TCM)
#define WNITER (WN / TCN)
#define ELEM_PER_THREAD (WMITER * WNITER * ((TCM * TCN) / NUM_THREADS) / (DOUBLE_BUFFER ? 2 : 1))

// FIXME: NUM_THREADS and NUM_WARPS hardcoded
#if ((BM * BN / ELEM_PER_THREAD) > (CORES_PER_CLUSTER * 8 * 8))
#error "threadblock size too big for cluster"
#endif

inline constexpr void map_operand_32lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // A (row major)
  // Figure 7(a) in paper
  // row  0~ 3: threadgroups 0 and 2
  // row  4~ 7: threadgroups 4 and 6
  // row  8~11: threadgroups 1 and 3
  // row 12~15: threadgroups 5 and 7
  row = tid % 4;
  row += (tg * 8) % 16;
  row += (tg / 4) * 4;

  // B (column major)
  // NOTE: Matrix B mapping in Figure 7(a) is incorrect; below is the
  // corrected mapping:
  // col  0~ 3: threadgroups 0 and 1
  // col  4~ 7: threadgroups 4 and 5
  // col  8~11: threadgroups 2 and 3
  // col 12~15: threadgroups 6 and 7
  col = tid % 4;
  col += ((tg % 4) / 2) * 8;
  col += (tg / 4) * 4;
}

inline constexpr void map_operand_8lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // A (row major)
  // row  0~ 3: threadgroup 0
  // row  4~ 7: threadgroup 1
  row = tid % 4;
  row += tg * 4;

  // B (column major)
  // col  0~ 3: threadgroup 0
  // col  4~ 7: threadgroup 1
  col = tid % 4;
  col += tg * 4;
}

inline constexpr void map_operand(const int tid, int &row, int &col) {
  if constexpr (NUM_THREADS == 32) {
    map_operand_32lanes(tid, row, col);
  } else if constexpr (NUM_THREADS == 8) {
    map_operand_8lanes(tid, row, col);
  } else {
    // FIXME: not allowed
  }
}

inline constexpr void map_c_32lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // C
  // Figure 7(b), left
  col = ((tg % 4) / 2) * 8;
  row = (tg * 8) % 16;
  row += (tg / 4) * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

inline constexpr void map_c_8lanes(const int tid, int &row, int &col) {
  const int tg = tid / 4;

  // C
  col = 0;
  row = tg * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

inline constexpr void map_c(const int tid, int &row, int &col) {
  if constexpr (NUM_THREADS == 32) {
    map_c_32lanes(tid, row, col);
  } else if constexpr (NUM_THREADS == 8) {
    map_c_8lanes(tid, row, col);
  } else {
    // FIXME: not allowed
  }
}

inline void vx_wmma(const int dest_reg) {
  if (dest_reg == 0) {
    asm volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM3));
  } else {
    asm volatile (".insn r %0, 0, 0, x1, x0, x0" :: "i"(RISCV_CUSTOM3));
  }
}

// `local_k` is assumed to be multiple of TCK
inline void vx_wmma_load_a(volatile float *smem_A, const int local_k,
                  const int warp_row, const int wm_iter, const int thread_in_warp) {
  const int tid = thread_in_warp;
  const int tg = tid / 4;

  // TODO: this is duplicately computed between vx_wmma_load_a and vx_wmma_load_b
  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  constexpr int smem_A_rows = BM;
  constexpr int smem_A_cols = BK;
  constexpr int smem_AS_rows = BK;
  constexpr int smem_AS_cols = BM;

  if constexpr (!TRANSPOSE_AS) {
    int A_offset = (WM * warp_row + TCM * wm_iter + row) * smem_A_cols;

    // @perf: bank conflicts
    // f8-f15 stores a single row of A
    asm volatile("flw  f0, %0" ::"m"(smem_A[A_offset + (local_k + 0)]));
    asm volatile("flw  f1, %0" ::"m"(smem_A[A_offset + (local_k + 1)]));
    asm volatile("flw  f2, %0" ::"m"(smem_A[A_offset + (local_k + 2)]));
    asm volatile("flw  f3, %0" ::"m"(smem_A[A_offset + (local_k + 3)]));
    asm volatile("flw  f4, %0" ::"m"(smem_A[A_offset + (local_k + 4)]));
    asm volatile("flw  f5, %0" ::"m"(smem_A[A_offset + (local_k + 5)]));
    asm volatile("flw  f6, %0" ::"m"(smem_A[A_offset + (local_k + 6)]));
    asm volatile("flw  f7, %0" ::"m"(smem_A[A_offset + (local_k + 7)]));
  } else {
    // transposed A
    // f8-f15 stores a single row of A
    volatile float *smem_addr;
    smem_addr = &smem_A[((local_k + 0) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row];
    asm volatile("flw  f0, %0(%1)" :: "i"(smem_AS_cols * 0 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f1, %0(%1)" :: "i"(smem_AS_cols * 1 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f2, %0(%1)" :: "i"(smem_AS_cols * 2 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f3, %0(%1)" :: "i"(smem_AS_cols * 3 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f4, %0(%1)" :: "i"(smem_AS_cols * 4 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f5, %0(%1)" :: "i"(smem_AS_cols * 5 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f6, %0(%1)" :: "i"(smem_AS_cols * 6 * sizeof(float)), "r"(smem_addr));
    asm volatile("flw  f7, %0(%1)" :: "i"(smem_AS_cols * 7 * sizeof(float)), "r"(smem_addr));

    // asm volatile("flw  f0, %0" ::"m"(smem_A[((local_k + 0) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f1, %0" ::"m"(smem_A[((local_k + 1) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f2, %0" ::"m"(smem_A[((local_k + 2) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f3, %0" ::"m"(smem_A[((local_k + 3) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f4, %0" ::"m"(smem_A[((local_k + 4) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f5, %0" ::"m"(smem_A[((local_k + 5) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f6, %0" ::"m"(smem_A[((local_k + 6) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
    // asm volatile("flw  f7, %0" ::"m"(smem_A[((local_k + 7) * smem_AS_cols) + (WM * warp_row + TCM * wm_iter) + row]));
  }
}

// `local_k` is assumed to be multiple of TCK
inline void vx_wmma_load_b(volatile float *smem_B, const int local_k,
                           const int warp_col, const int wn_iter,
                           const int thread_in_warp) {
  const int tid = thread_in_warp;
  const int tg = tid / 4;

  int row = 0;
  int col = 0;
  map_operand(tid, row, col);

  constexpr int smem_B_rows = BK;
  constexpr int smem_B_cols = BN;

  // f8-f15 stores a single column of B
  volatile float *smem_addr;
  smem_addr = &smem_B[((local_k + 0) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col];
  asm volatile("flw  f8, %0(%1)" :: "i"(smem_B_cols * 0 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw  f9, %0(%1)" :: "i"(smem_B_cols * 1 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f10, %0(%1)" :: "i"(smem_B_cols * 2 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f11, %0(%1)" :: "i"(smem_B_cols * 3 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f12, %0(%1)" :: "i"(smem_B_cols * 4 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f13, %0(%1)" :: "i"(smem_B_cols * 5 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f14, %0(%1)" :: "i"(smem_B_cols * 6 * sizeof(float)), "r"(smem_addr));
  asm volatile("flw f15, %0(%1)" :: "i"(smem_B_cols * 7 * sizeof(float)), "r"(smem_addr));

  // asm volatile("flw  f8, %0" ::"m"(smem_B[((local_k + 0) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw  f9, %0" ::"m"(smem_B[((local_k + 1) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f10, %0" ::"m"(smem_B[((local_k + 2) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f11, %0" ::"m"(smem_B[((local_k + 3) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f12, %0" ::"m"(smem_B[((local_k + 4) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f13, %0" ::"m"(smem_B[((local_k + 5) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f14, %0" ::"m"(smem_B[((local_k + 6) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
  // asm volatile("flw f15, %0" ::"m"(smem_B[((local_k + 7) * smem_B_cols) + (WN * warp_col + TCN * wn_iter) + col]));
}

inline void initialize_C(const int dest_reg) {
  // initialize C to zeros
  if (dest_reg == 0) {
    asm volatile("fmv.w.x f16, x0");
    asm volatile("fmv.w.x f17, x0");
    asm volatile("fmv.w.x f18, x0");
    asm volatile("fmv.w.x f19, x0");
    asm volatile("fmv.w.x f20, x0");
    asm volatile("fmv.w.x f21, x0");
    asm volatile("fmv.w.x f22, x0");
    asm volatile("fmv.w.x f23, x0");
  } else {
    asm volatile("fmv.w.x f24, x0");
    asm volatile("fmv.w.x f25, x0");
    asm volatile("fmv.w.x f26, x0");
    asm volatile("fmv.w.x f27, x0");
    asm volatile("fmv.w.x f28, x0");
    asm volatile("fmv.w.x f29, x0");
    asm volatile("fmv.w.x f30, x0");
    asm volatile("fmv.w.x f31, x0");
  }
}

inline void write_results(const int thread_in_warp, const int warp_col,
                          const int warp_row, const int wn_iter,
                          const int wm_iter, const int dim_n,
                          float *C, const int threadblock_id_x,
                          const int threadblock_id_y) {
  int tid = thread_in_warp;
  int tg = tid / 4;

  // these are [0, TCM/TCN)
  int tid_row = 0;
  int tid_col = 0;
  map_c(tid, tid_row, tid_col);

  int local_row = (WM * warp_row + TCM * wm_iter) + tid_row;
  int local_col = (WN * warp_col + TCN * wn_iter) + tid_col;

  float *global_offset_C = C +
                           (BM * threadblock_id_y) * dim_n +
                           BN * threadblock_id_x;

  // @perf: this likely causes a lot of gmem bank conflicts
  if (wm_iter == 0) {
    volatile float *gmem_addr;
    volatile float *gmem_addr_tmp;
    gmem_addr = &global_offset_C[dim_n * (local_row + 0) + (local_col + 0)];
    asm volatile ("fsw f16, %0" :: "m"(*(gmem_addr + 0)));
    asm volatile ("fsw f17, %0" :: "m"(*(gmem_addr + 1)));
    gmem_addr_tmp = gmem_addr + (2 * dim_n);
    asm volatile ("fsw f18, %0" :: "m"(*(gmem_addr_tmp + 0)));
    asm volatile ("fsw f19, %0" :: "m"(*(gmem_addr_tmp + 1)));
    gmem_addr += 4;
    asm volatile ("fsw f20, %0" :: "m"(*(gmem_addr + 0)));
    asm volatile ("fsw f21, %0" :: "m"(*(gmem_addr + 1)));
    gmem_addr_tmp = gmem_addr + (2 * dim_n);
    asm volatile ("fsw f22, %0" :: "m"(*(gmem_addr_tmp + 0)));
    asm volatile ("fsw f23, %0" :: "m"(*(gmem_addr_tmp + 1)));
    // asm volatile ("fsw f16, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 0)]));
    // asm volatile ("fsw f17, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 1)]));
    // asm volatile ("fsw f18, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 0)]));
    // asm volatile ("fsw f19, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 1)]));
    // asm volatile ("fsw f20, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 4)]));
    // asm volatile ("fsw f21, %0" :: "m"(global_offset_C[dim_n * (local_row + 0) + (local_col + 5)]));
    // asm volatile ("fsw f22, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 4)]));
    // asm volatile ("fsw f23, %0" :: "m"(global_offset_C[dim_n * (local_row + 2) + (local_col + 5)]));
  } else {
    volatile float *gmem_addr;
    volatile float *gmem_addr_tmp;
    gmem_addr = &global_offset_C[dim_n * (local_row + 0) + (local_col + 0)];
    gmem_addr_tmp = gmem_addr + (2 * dim_n);
    asm volatile ("fsw f24, %0" :: "m"(*(gmem_addr + 0)));
    asm volatile ("fsw f25, %0" :: "m"(*(gmem_addr + 1)));
    asm volatile ("fsw f26, %0" :: "m"(*(gmem_addr_tmp + 0)));
    asm volatile ("fsw f27, %0" :: "m"(*(gmem_addr_tmp + 1)));
    gmem_addr += 4;
    gmem_addr_tmp = gmem_addr + (2 * dim_n);
    asm volatile ("fsw f28, %0" :: "m"(*(gmem_addr + 0)));
    asm volatile ("fsw f29, %0" :: "m"(*(gmem_addr + 1)));
    asm volatile ("fsw f30, %0" :: "m"(*(gmem_addr_tmp + 0)));
    asm volatile ("fsw f31, %0" :: "m"(*(gmem_addr_tmp + 1)));
  }
}

inline void threadblock_barrier(const uint32_t barrier_id, const uint32_t count) {
  vx_fence();
  vx_barrier(barrier_id, count);
  // vx_barrier(0, count);
}

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
  if constexpr (!TRANSPOSE_AS) {
    // FIXME: !TRANSPOSE_AS code is old

    const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
    // number of rows a full TB can read at a time
    constexpr uint32_t row_stride_a = threads_in_warpgroup / BK;
    const float *global_a = A + dim_k * global_a_row + (k + local_a_col);
    volatile float *local_a_tmp = local_a + BK * local_a_row + local_a_col;

#pragma GCC unroll 1
    for (uint32_t local_row_offset = 0; local_row_offset < BM;
         local_row_offset += row_stride_a) {
      // const uint32_t global_a_offset =
      //     dim_k * (global_a_row + local_row_offset) + (k + local_a_col);
      // local_a[BK * (local_a_row + local_row_offset) + local_a_col] =
      //     A[global_a_offset];
      *local_a_tmp = *global_a;

      global_a += dim_k * row_stride_a;
      local_a_tmp += BK * row_stride_a;
    }
  } else {
    if constexpr (!GMEM_COALESCED_A) {
      constexpr uint32_t row_stride_as = threads_in_warpgroup / BM;
      const uint32_t global_a_row = BM * threadblock_id_y + local_as_col;
      const float *global_a = A + dim_k * global_a_row + (k + local_as_row);
      // FIXME experimenting with global coalescing
      // const uint32_t global_a_row = BM * threadblock_id_y + local_as_row;
      // const float *global_a = A + dim_k * global_a_row + (k + local_as_col);
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
    asm volatile ("fsw ft2, %0(%1)" :: "i"(BN * row_stride_b * 2 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft3, %0(%1)" :: "i"(BN * row_stride_b * 3 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft4, %0(%1)" :: "i"(BN * row_stride_b * 4 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft5, %0(%1)" :: "i"(BN * row_stride_b * 5 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft6, %0(%1)" :: "i"(BN * row_stride_b * 6 * sizeof(float)), "r"(local_b_tmp));
    asm volatile ("fsw ft7, %0(%1)" :: "i"(BN * row_stride_b * 7 * sizeof(float)), "r"(local_b_tmp));
    local_b_tmp += BN * row_stride_b * 8;
  }
}

inline void thread_block_gemm(kernel_arg_t *__UNIFORM__ arg,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threads_per_threadblock,
                              const uint32_t threadblock_dim_x,
                              const uint32_t threadblock_dim_y,
                              /*const uint32_t threadblock_id_x,
                              const uint32_t threadblock_id_y,*/
                              // const uint32_t threadblock_id_in_cluster,
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

  // const uint32_t threads_per_warpgroup = threads_per_threadblock / (DOUBLE_BUFFER ? 2 : 1);
  const uint32_t threads_per_warpgroup = threads_per_threadblock / 1;
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

  constexpr uint32_t skips =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);

  if (tid_in_warpgroup == 0) {
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    // gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose,
    // b_transpose);

    gemmini_extended3_config_ld(dim_k * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 0);
    gemmini_extended3_config_ld(dim_n * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 1);
    gemmini_extended_config_st(dim_n * sizeof(elem_t), 0, MVIN_SCALE_IDENTITY);

    gemmini_fence();
  }

  if (false && warpgroup_id == 0) {
    // producer code: GMEM->SMEM data movement

#pragma GCC unroll 1
    for (uint32_t block_m = 0; (block_m * BM) < dim_m; block_m++) {
#pragma GCC unroll 1
      for (uint32_t block_n = 0; (block_n * BN) < dim_n; block_n++) {

#if 1
        if (tid_in_warpgroup == 0) {
          // configure dma gmem address to load from
          ROCC_INSTRUCTION_RS1_RS2(
              XCUSTOM_ACC,
              (uint64_t)(A + block_m * BM * dim_k + 0/*block_k*/ * BK),
              (uint64_t)(B + 0/*block_k*/ * BK * dim_n + block_n * BN),
              k_LOOP_WS_CONFIG_ADDRS_AB)
          // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
          GEMMINI_CISC_CMD_R((dim_n << 16) | (dim_k << 8) | 8);

          // GEMMINI_CISC_CMD_I(10);
          // gemmini_fence();

          // configure loop iteration bounds
          // FIXME: shouldn't be necessary
// #define BOUND_INST 0x400040004ULL
          // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, BOUND_INST, k_LOOP_WS_CONFIG_BOUNDS)
          // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, SPAD_ADDR_Q0, SPAD_ADDR_Q1, k_LOOP_WS_CONFIG_SPAD_AB)
          // ROCC_INSTRUCTION_RS1_RS2(
          //     XCUSTOM_ACC,
          //     ((uint64_t)(/*a_spad_id:*/ 0) << 18) |
          //         ((uint64_t)(/*b_spad_id:*/ 0) << 16) |
          //         ((uint64_t)(/*act:0*/ 0) << 8) | ((/*low_D:*/ 0) << 2) |
          //         ((/*full_C:*/ 0) << 1) | (/*ex_accumulate:*/ 0),
          //     ((uint64_t)(/*C_spad_addr:*/ A) << 32) | 0x200U | (skips) |
          //         ((/*is_resadd*/ 0) << 2) | ((/*B_transpose:*/ 0) << 1) |
          //         (/*A_transpose:*/ 1),
          //     k_LOOP_WS)
          // gemmini_fence();

          // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
          sp_tiled_matmul_full_spad_ws(
              (/*block_k:*/0 & 1) ? SPAD_ADDR_Q2 : SPAD_ADDR_Q0,
              (/*block_k:*/0 & 1) ? SPAD_ADDR_Q3 : SPAD_ADDR_Q1,
              /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q1,
              /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
              /*pad_J=*/0, /*pad_K=*/0,
              /*a_transpose=*/1, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
              /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
          gemmini_fence();
        }

        threadblock_barrier(0 /*threadblock_id_in_cluster*/, threadblock_dim_y);

#pragma GCC unroll 1
        for (uint32_t block_k = 1; block_k * BK < (8 * dim_k); block_k++) {
          if (tid_in_warpgroup == 0) {
            ROCC_INSTRUCTION_RS1_RS2(
                XCUSTOM_ACC,
                (uint64_t)(A + block_m * BM * dim_k + block_k * BK),
                (uint64_t)(B + block_k * BK * dim_n + block_n * BN),
                k_LOOP_WS_CONFIG_ADDRS_AB)
            // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
            GEMMINI_CISC_CMD_R((dim_n << 16) | (dim_k << 8) | 8);

            // GEMMINI_CISC_CMD_I(10);
            // gemmini_fence();

            // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
            sp_tiled_matmul_full_spad_ws(
                (block_k & 1) ? SPAD_ADDR_Q2 : SPAD_ADDR_Q0,
                (block_k & 1) ? SPAD_ADDR_Q3 : SPAD_ADDR_Q1,
                /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q1,
                /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
                /*pad_J=*/0, /*pad_K=*/0,
                /*a_transpose=*/1, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
                /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
            gemmini_fence();
          }

          threadblock_barrier(0/*threadblock_id_in_cluster*/, threadblock_dim_y);
        }

        // sync with final consumer stage in the k-loop
        threadblock_barrier(0/*threadblock_id_in_cluster*/, threadblock_dim_y);

#else
        if constexpr (DOUBLE_BUFFER) {
          // initiate software pipeline
          global_dmem_load(dim_n, dim_k, 0 /*k*/, A, B, local_a, local_b,
                           tid_in_warpgroup, block_n, block_m);

          threadblock_barrier(0/*threadblock_id_in_cluster*/, threadblock_dim_y);
        }

        // NOTE: this *should* be signed integer to trigger arithmetic
        // right-shift
        int32_t k_index = 0;
#pragma GCC unroll 1
        for (uint32_t k = 0; k < dim_k - BK; k += BK) {
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

          threadblock_barrier(0/*threadblock_id_in_cluster*/, threadblock_dim_y);
        }

        // sync with final consumer stage in the k-loop
        threadblock_barrier(0/*threadblock_id_in_cluster*/, threadblock_dim_y);
#endif
      }
    }
  } else {
    // consumer code: SMEM->RF and compute

#pragma GCC unroll 1
    for (uint32_t block_m = 0; (block_m * BM) < dim_m; block_m++) {
#pragma GCC unroll 1
      for (uint32_t block_n = 0; (block_n * BN) < dim_n; block_n++) {
        // clear out C
        initialize_C(0);
        initialize_C(1);

        if (tid_in_warpgroup == 0) {
          // configure dma gmem address to load from
          ROCC_INSTRUCTION_RS1_RS2(
              XCUSTOM_ACC,
              (uint64_t)(A + block_m * BM * dim_k + 0/*block_k*/ * BK),
              (uint64_t)(B + 0/*block_k*/ * BK * dim_n + block_n * BN),
              k_LOOP_WS_CONFIG_ADDRS_AB)
          // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
          GEMMINI_CISC_CMD_R((dim_n << 16) | (dim_k << 8) | 8);

          // GEMMINI_CISC_CMD_I(10);
          // gemmini_fence();

          // configure loop iteration bounds
          // FIXME: shouldn't be necessary
// #define BOUND_INST 0x400040004ULL
          // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, BOUND_INST, k_LOOP_WS_CONFIG_BOUNDS)
          // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, SPAD_ADDR_Q0, SPAD_ADDR_Q1, k_LOOP_WS_CONFIG_SPAD_AB)
          // ROCC_INSTRUCTION_RS1_RS2(
          //     XCUSTOM_ACC,
          //     ((uint64_t)(/*a_spad_id:*/ 0) << 18) |
          //         ((uint64_t)(/*b_spad_id:*/ 0) << 16) |
          //         ((uint64_t)(/*act:0*/ 0) << 8) | ((/*low_D:*/ 0) << 2) |
          //         ((/*full_C:*/ 0) << 1) | (/*ex_accumulate:*/ 0),
          //     ((uint64_t)(/*C_spad_addr:*/ A) << 32) | 0x200U | (skips) |
          //         ((/*is_resadd*/ 0) << 2) | ((/*B_transpose:*/ 0) << 1) |
          //         (/*A_transpose:*/ 1),
          //     k_LOOP_WS)
          // gemmini_fence();

          // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
          // FIXME: block_k is 0 for two times
          sp_tiled_matmul_full_spad_ws(
              (/*block_k:*/0 & 1) ? SPAD_ADDR_Q2 : SPAD_ADDR_Q0,
              (/*block_k:*/0 & 1) ? SPAD_ADDR_Q3 : SPAD_ADDR_Q1,
              /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q1,
              /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
              /*pad_J=*/0, /*pad_K=*/0,
              /*a_transpose=*/1, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
              /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
          gemmini_fence();
        }

        // sync with initial producer stage in the k-loop
        threadblock_barrier(0/*threadblock_id_in_cluster*/, threadblock_dim_y);

        // NOTE: this *should* be signed integer to trigger arithmetic
        // right-shift
        int32_t k_index = 0;
#pragma GCC unroll 1
        for (uint32_t block_k = 0; block_k * BK < (8 * dim_k); block_k++) {
          if (tid_in_warpgroup == 0) {
            ROCC_INSTRUCTION_RS1_RS2(
                XCUSTOM_ACC,
                (uint64_t)(A + block_m * BM * dim_k + block_k * BK),
                (uint64_t)(B + block_k * BK * dim_n + block_n * BN),
                k_LOOP_WS_CONFIG_ADDRS_AB)
            // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
            GEMMINI_CISC_CMD_R((dim_n << 16) | (dim_k << 8) | 8);

            // GEMMINI_CISC_CMD_I(10);
            // gemmini_fence();

            // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
            sp_tiled_matmul_full_spad_ws(
                (block_k & 1) ? SPAD_ADDR_Q0 : SPAD_ADDR_Q2,
                (block_k & 1) ? SPAD_ADDR_Q1 : SPAD_ADDR_Q3,
                /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q1,
                /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
                /*pad_J=*/0, /*pad_K=*/0,
                /*a_transpose=*/1, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
                /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
            gemmini_fence();
          }

          volatile float *local_a_consume;
          volatile float *local_b_consume;
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

          threadblock_barrier(0/*threadblock_id_in_cluster*/, threadblock_dim_y);
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

  const uint32_t threads_per_threadblock = (BM * BN) / (ELEM_PER_THREAD);
#ifdef RADIANCE
  const uint32_t threadblocks_per_core = CORES_PER_CLUSTER * vx_num_threads() *
                                         vx_num_warps() /
                                         threads_per_threadblock;
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

  const int warp_id = vx_warp_id();
  thread_block_gemm(arg, tid_in_threadblock, threads_per_threadblock,
                    threadblock_dim_x, threadblock_dim_y, /*threadblock_id_x,
                    threadblock_id_y,*/ /*threadblock_id_in_cluster, */
                    sharedmem_per_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  const uint32_t threads_per_cluster =
      CORES_PER_CLUSTER * vx_num_threads() * vx_num_warps();
  // const uint32_t grid_size = arg->dim_m * arg->dim_n / ELEM_PER_THREAD;
  const uint32_t grid_size = threads_per_cluster;

#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
