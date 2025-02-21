#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "util.hpp"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define GEMMINI_DMA 1
#if SMEM_SIZE == 0x4000
#define SMEM_ADDR_Q0 ((float * const) 0xff000000)
#define SMEM_ADDR_Q1 ((float * const) 0xff001000)
#define SMEM_ADDR_Q2 ((float * const) 0xff002000)
#define SMEM_ADDR_Q3 ((float * const) 0xff003000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x80
#define SPAD_ADDR_Q2 0x100
#define SPAD_ADDR_Q3 0x180
#define BOUND_INST 0x400040004ULL
#elif SMEM_SIZE == 0x10000
#define SMEM_ADDR_Q0 ((float * const) 0xff000000)
#define SMEM_ADDR_Q1 ((float * const) 0xff004000)
#define SMEM_ADDR_Q2 ((float * const) 0xff008000)
#define SMEM_ADDR_Q3 ((float * const) 0xff00c000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x200
#define SPAD_ADDR_Q2 0x400
#define SPAD_ADDR_Q3 0x600
#define BOUND_INST 0x800080008ULL
#else
#error Unsupported smem size
#endif

#define WARP_SPECIALIZED 1
#define SWISH(beta, x) ((x) / (1 + exp(-(beta) * (x))))

static_assert(
    !WARP_SPECIALIZED || GEMMINI_DMA,
    "warp specialization is currently only supported with GEMMINI_DMA");

static_assert((BM * BN / ELEM_PER_THREAD) * (WARP_SPECIALIZED ? 2 : 1) ==
                  (CORES_PER_CLUSTER * NUM_WARPS * NUM_THREADS),
              "threadblock size does not match hw threads in cluster");

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

  constexpr uint32_t threads_in_threadblock = (BM * BN) / ELEM_PER_THREAD;

  // Data move from GMEM to SMEM
  //
  // Make sure global offset values for A and B are contiguous between
  // neighboring threads to ensure GMEM coalescing.
  //
  // TODO: Sharedmem swizzling is important here
  if constexpr (!TRANSPOSE_AT_PRODUCE) {
    // FIXME: !TRANSPOSE_AS code is old

    const uint32_t global_a_row = BM * threadblock_id_y + local_a_row;
    // number of rows a full TB can read at a time
    constexpr uint32_t row_stride_a = threads_in_threadblock / BK;
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
      constexpr uint32_t row_stride_as = threads_in_threadblock / BM;
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
      constexpr uint32_t row_stride_a = threads_in_threadblock / BK;
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

  constexpr uint32_t row_stride_b = threads_in_threadblock / BN;
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

inline void activate_block(const uint32_t dim_n, float *const C,
                           const uint32_t tile_i, const uint32_t tile_j,
                           const uint32_t warp_row, const uint32_t warp_col,
                           const uint32_t tid_in_threadblock) {
  // activation code currently assumes that the column-width of a warp
  // tile exactly matches SIMD width
  static_assert(WN == NUM_THREADS);

  const uint32_t col_in_warptile = tid_in_threadblock % WN;
  // const uint32_t row_in_warptile = elem_i; // FIXME: doesn't work with WN !=
  // NUM_THREADS
  const uint32_t row_in_warptile = 0;
  const uint32_t C_row = (tile_i * BM) + (warp_row * WM) + row_in_warptile;
  const uint32_t C_col = (tile_j * BN) + (warp_col * WN) + col_in_warptile;
  float *const global_C = C + dim_n * C_row + C_col;
  const float *global_C_curr = global_C;

  // ELEM_PER_THREAD macro does not take into account warp-specialization
  constexpr uint32_t elem_per_thread =
      ELEM_PER_THREAD * (WARP_SPECIALIZED ? 2 : 1);
  constexpr uint32_t asm_unrolled = 8; // working with f0~f7 at a time
  // each thread works on ELEM_PER_THREAD elements, which can be larger than 1
  static_assert((elem_per_thread % asm_unrolled) == 0,
                "unmet manual unroll condition for elem_per_thread");

#if 1
  float elems[elem_per_thread];

#pragma GCC unroll asm_unrolled
  for (int elem_i = 0; elem_i < elem_per_thread; elem_i++) {
    elems[elem_i] = global_C[dim_n * elem_i];
    elems[elem_i] = SWISH(1.0f, elems[elem_i]);
    global_C[dim_n * elem_i] = elems[elem_i];
  }
#else
  for (int i = 0; i < elem_per_thread; i += asm_unrolled) {
    // read in elements from GMEM to RF
    asm volatile("mv   t6, %0" ::"r"(global_C_curr));
    asm volatile("flw  f0, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("flw  f1, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("flw  f2, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("flw  f3, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("flw  f4, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("flw  f5, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("flw  f6, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("flw  f7, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));

    if constexpr (true) {
      // FIXME: this is likely incorrect; f0~f7 regs get overwritten by
      // the compiler
      register float x0 asm("f0");
      register float x1 asm("f1");
      register float x2 asm("f2");
      register float x3 asm("f3");
      register float x4 asm("f4");
      register float x5 asm("f5");
      register float x6 asm("f6");
      register float x7 asm("f7");
      asm volatile("fmv.s  %0, f0" :"=f"(x0));
      x0 = SWISH(1.0f, x0);
      asm volatile("fmv.s  f0, %0" ::"f"(x0));
      asm volatile("fmv.s  %0, f1" :"=f"(x1));
      x1 = SWISH(1.0f, x1);
      asm volatile("fmv.s  f1, %0" ::"f"(x1));
      asm volatile("fmv.s  %0, f1" :"=f"(x2));
      x2 = SWISH(1.0f, x2);
      asm volatile("fmv.s  f1, %0" ::"f"(x2));
      asm volatile("fmv.s  %0, f1" :"=f"(x3));
      x3 = SWISH(1.0f, x3);
      asm volatile("fmv.s  f1, %0" ::"f"(x3));
      asm volatile("fmv.s  %0, f1" :"=f"(x4));
      x4 = SWISH(1.0f, x4);
      asm volatile("fmv.s  f1, %0" ::"f"(x4));
      asm volatile("fmv.s  %0, f1" :"=f"(x5));
      x5 = SWISH(1.0f, x5);
      asm volatile("fmv.s  f1, %0" ::"f"(x5));
      asm volatile("fmv.s  %0, f1" :"=f"(x6));
      x6 = SWISH(1.0f, x6);
      asm volatile("fmv.s  f1, %0" ::"f"(x6));
      asm volatile("fmv.s  %0, f1" :"=f"(x7));
      x7 = SWISH(1.0f, x7);
      asm volatile("fmv.s  f1, %0" ::"f"(x7));
    } else {
      // do elem-wise e^x
      // each register has 3 temporary registers:
      // f0 has f8, f9, f10
      // f1 has f11, f12, f13
      asm volatile("fcvt.s.w f9, %0" ::"r"(1));
      asm volatile("fadd.s f8, f9, f0"); // acc = 1 + x
      asm volatile("fcvt.s.w f9, %0" ::"r"(2));
      asm volatile("fdiv.s f10, f0, f9");      // x / 2
      asm volatile("fmadd.s f8, f10, f0, f8"); // acc += (x / 2) * x
      asm volatile("fcvt.s.w f9, %0" ::"r"(3));
      asm volatile("fmul.s f10, f10, f0");     // (x * x) / 2
      asm volatile("fdiv.s f10, f10, f9");     // (x * x) / (2 * 3)
      asm volatile("fmadd.s f0, f10, f0, f8"); // acc += (x * x) / (2 * 3) * x

      asm volatile("fcvt.s.w f12, %0" ::"r"(1));
      asm volatile("fadd.s f11, f12, f1");
      asm volatile("fcvt.s.w f12, %0" ::"r"(2));
      asm volatile("fdiv.s f13, f1, f12");
      asm volatile("fmadd.s f11, f13, f1, f11");
      asm volatile("fcvt.s.w f12, %0" ::"r"(3));
      asm volatile("fmul.s f13, f13, f1");
      asm volatile("fdiv.s f13, f13, f12");
      asm volatile("fmadd.s f1, f13, f1, f11");

      asm volatile("fcvt.s.w f15, %0" ::"r"(1));
      asm volatile("fadd.s f14, f15, f2");
      asm volatile("fcvt.s.w f15, %0" ::"r"(2));
      asm volatile("fdiv.s f16, f2, f15");
      asm volatile("fmadd.s f14, f16, f2, f14");
      asm volatile("fcvt.s.w f15, %0" ::"r"(3));
      asm volatile("fmul.s f16, f16, f2");
      asm volatile("fdiv.s f16, f16, f15");
      asm volatile("fmadd.s f2, f16, f2, f14");

      asm volatile("fcvt.s.w f18, %0" ::"r"(1));
      asm volatile("fadd.s f17, f18, f3");
      asm volatile("fcvt.s.w f18, %0" ::"r"(2));
      asm volatile("fdiv.s f19, f3, f18");
      asm volatile("fmadd.s f17, f19, f3, f17");
      asm volatile("fcvt.s.w f18, %0" ::"r"(3));
      asm volatile("fmul.s f19, f19, f3");
      asm volatile("fdiv.s f19, f19, f18");
      asm volatile("fmadd.s f3, f19, f3, f17");

      asm volatile("fcvt.s.w f21, %0" ::"r"(1));
      asm volatile("fadd.s f20, f21, f4");
      asm volatile("fcvt.s.w f21, %0" ::"r"(2));
      asm volatile("fdiv.s f22, f4, f21");
      asm volatile("fmadd.s f20, f22, f4, f20");
      asm volatile("fcvt.s.w f21, %0" ::"r"(3));
      asm volatile("fmul.s f22, f22, f4");
      asm volatile("fdiv.s f22, f22, f21");
      asm volatile("fmadd.s f4, f22, f4, f20");

      asm volatile("fcvt.s.w f24, %0" ::"r"(1));
      asm volatile("fadd.s f23, f24, f5");
      asm volatile("fcvt.s.w f24, %0" ::"r"(2));
      asm volatile("fdiv.s f25, f5, f24");
      asm volatile("fmadd.s f23, f25, f5, f23");
      asm volatile("fcvt.s.w f24, %0" ::"r"(3));
      asm volatile("fmul.s f25, f25, f5");
      asm volatile("fdiv.s f25, f25, f24");
      asm volatile("fmadd.s f5, f25, f5, f23");

      asm volatile("fcvt.s.w f27, %0" ::"r"(1));
      asm volatile("fadd.s f26, f27, f6");
      asm volatile("fcvt.s.w f27, %0" ::"r"(2));
      asm volatile("fdiv.s f28, f6, f27");
      asm volatile("fmadd.s f26, f28, f6, f26");
      asm volatile("fcvt.s.w f27, %0" ::"r"(3));
      asm volatile("fmul.s f28, f28, f6");
      asm volatile("fdiv.s f28, f28, f27");
      asm volatile("fmadd.s f6, f28, f6, f26");

      asm volatile("fcvt.s.w f30, %0" ::"r"(1));
      asm volatile("fadd.s f29, f30, f7");
      asm volatile("fcvt.s.w f30, %0" ::"r"(2));
      asm volatile("fdiv.s f31, f7, f30");
      asm volatile("fmadd.s f29, f31, f7, f29");
      asm volatile("fcvt.s.w f30, %0" ::"r"(3));
      asm volatile("fmul.s f31, f31, f7");
      asm volatile("fdiv.s f31, f31, f30");
      asm volatile("fmadd.s f7, f31, f7, f29");
    }

    // move back from RF to gmem
    asm volatile("mv   t6, %0" ::"r"(global_C_curr));
    asm volatile("fsw  f0, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("fsw  f1, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("fsw  f2, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("fsw  f3, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("fsw  f4, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("fsw  f5, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("fsw  f6, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("fsw  f7, (t6)");
    asm volatile("add  t6, t6, %0" ::"r"(dim_n * sizeof(float)));
    asm volatile("mv   %0, t6" :"=r"(global_C_curr));
  }
#endif
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

  const uint32_t threads_per_warpgroup =
      threads_per_threadblock / (WARP_SPECIALIZED ? 2 : 1);
  const uint32_t warpgroup_id = tid_in_threadblock / threads_per_warpgroup; // NOCHECKIN
  const uint32_t tid_in_warpgroup = tid_in_threadblock % threads_per_warpgroup;
  const uint32_t warp_id_in_warpgroup = tid_in_warpgroup / NUM_THREADS;
  const uint32_t warp_row = warp_id_in_warpgroup / (BN / WN);
  const uint32_t warp_col = warp_id_in_warpgroup % (BN / WN);
  const uint32_t tid_in_warp = tid_in_threadblock % NUM_THREADS;

  // layout: local_a -- local_a_buf -- local_b -- local_b_buf
  volatile float *local_a = sharedmem_per_threadblock;
  constexpr size_t local_a_elems = (BM * BK);
  volatile float *local_a_buf = local_a + local_a_elems;

  volatile float *local_b = local_a_buf + local_a_elems;
  // set local B tile size to be the same as local A size (BM * BK), since DMA
  // is currently only configured for square-shape tiles.  FIXME.
  constexpr size_t local_b_elems = (BK * BM);
  volatile float *local_b_buf = local_a_buf + local_b_elems;

  constexpr uint32_t skips =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);

#if (GEMMINI_DMA == 1)
  // regardless of WARP_SPECIALIZED, this has to be done by one thread
  if (tid_in_threadblock == 0) {
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
#endif

  // divide rows (M) by the number of threadblocks
  const uint32_t dim_m_range = (dim_m / threadblocks_per_cluster);
  const uint32_t dim_m_start = dim_m_range * threadblock_id_in_cluster;
  const uint32_t block_m_start = dim_m_start / BM;
  const uint32_t block_m_end = (dim_m_start + dim_m_range) / BM;

#pragma GCC unroll 1
  for (uint32_t block_m = block_m_start; block_m < block_m_end; block_m++) {
#pragma GCC unroll 1
    for (uint32_t block_n = 0; (block_n * BN) < dim_n; block_n++) {
      // clear out C
      initialize_C(0);
      initialize_C(1);

      if constexpr (GEMMINI_DMA) {
        if (!WARP_SPECIALIZED || warpgroup_id == 0) {
          // pipeline initiation
          if (tid_in_threadblock == 0) {
            // configure dma gmem address to load from
            // FIXME: block_k is wrong
            ROCC_INSTRUCTION_RS1_RS2(
                XCUSTOM_ACC,
                (uint64_t)(A + block_m * BM * dim_k + /*block_k:*/0 * BK),
                (uint64_t)(B + /*block_k:*/0 * BK * dim_n + block_n * BN),
                k_LOOP_WS_CONFIG_ADDRS_AB)
            // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
            GEMMINI_CISC_CMD_R((dim_n << 16) | (dim_k << 8) | 8);
            gemmini_fence();

            GEMMINI_CISC_CMD_I(10);
            gemmini_fence();

#if 0
            // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
            // FIXME: block_k is 0 for two times
            sp_tiled_matmul_full_spad_ws(
#if 1
                SPAD_ADDR_Q0, SPAD_ADDR_Q1,
#else
                (/*block_k:*/ 0 & 1) ? SPAD_ADDR_Q2 : SPAD_ADDR_Q0,
                (/*block_k:*/ 0 & 1) ? SPAD_ADDR_Q3 : SPAD_ADDR_Q1,
#endif
                /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q3,
                /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
                /*pad_J=*/0, /*pad_K=*/0,
                /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
                /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
            gemmini_fence();
#endif
          }

          // threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
          threadblock_barrier(1 /*gemm worpgroup*/,
                              threadblock_dim_y / (WARP_SPECIALIZED ? 2 : 1));
        }
      }

#pragma GCC unroll 1
      for (uint32_t block_k = 0; (block_k * BK) < (dim_k); block_k++) {

        // producer code: GMEM->SMEM memory movement
        // ---------------------------------------------------------------------
        //
        // this is either done using DMA or SIMT cores depending on GEMMINI_DMA

#if (GEMMINI_DMA == 1)
        if (!WARP_SPECIALIZED || warpgroup_id == 0) {
          if (tid_in_threadblock == 0) {
            // configure dma gmem address to load from
            // FIXME: block_k is wrong
            ROCC_INSTRUCTION_RS1_RS2(
                XCUSTOM_ACC,
                (uint64_t)(A + block_m * BM * dim_k + (block_k + 1/*runahead*/) * BK),
                (uint64_t)(B + (block_k + 1/*runahead*/) * BK * dim_n + block_n * BN),
                k_LOOP_WS_CONFIG_ADDRS_AB)
            // GEMMINI_CISC(8) does k_LOOP_WS_CONFIG_STRIDES_AB
            GEMMINI_CISC_CMD_R((dim_n << 16) | (dim_k << 8) | 8);
            // gemmini_fence();

            // block_k is even: opcode 11 (write to local_a_buf)
            // block_k is odd:  opcode 10 (write to local_a)
            const uint32_t opcode = 11 - (block_k & 1);
            GEMMINI_CISC_CMD_R(opcode);
            // // TODO: branch is probably slow
            // if (block_k & 1) {
            //   GEMMINI_CISC_CMD_I(12);
            // } else { // block_k == 0 is here
            //   GEMMINI_CISC_CMD_I(13);
            // }

            // configure loop iteration bounds
            // FIXME: shouldn't be necessary
            // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, BOUND_INST,
            // k_LOOP_WS_CONFIG_BOUNDS) ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
            // SPAD_ADDR_Q0, SPAD_ADDR_Q1, k_LOOP_WS_CONFIG_SPAD_AB)
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

#if 0
            uint32_t spad_a_produce;
            uint32_t spad_b_produce;
            const uint32_t mask_odd = (block_k & 1) << 31 >> 31;
            const uint32_t mask_even = ((block_k & 1) ^ 1) << 31 >> 31;
            spad_a_produce =
                ((mask_odd & (SPAD_ADDR_Q0)) | (mask_even & (SPAD_ADDR_Q2)));
            spad_b_produce =
                ((mask_odd & (SPAD_ADDR_Q1)) | (mask_even & (SPAD_ADDR_Q3)));
            // sp_tiled_matmul_full_spad_ws includes CONFIG_BOUNDS
            // FIXME: block_k is 0 for two times
            sp_tiled_matmul_full_spad_ws(
                spad_a_produce,
                spad_b_produce,
                /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q1,
                /*I=*/BM / DIM, /*J=*/BN / DIM, /*K=*/BK / DIM, /*pad_I=*/0,
                /*pad_J=*/0, /*pad_K=*/0,
                /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
                /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips)
#endif
          }
        }
#else
        global_dmem_load(dim_n, dim_k, block_k * BK, A, B, local_a, local_b,
                         tid_in_threadblock, block_n, block_m);

        threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
#endif

        if (!WARP_SPECIALIZED || warpgroup_id == 0) {
          // consumer code: SMEM->RF and compute
          // ----------------------------------------------------------------------
          // @perf: this loop spills to stack a lot because of all the flws in
          const volatile float *local_a_consume;
          const volatile float *local_b_consume;
          if constexpr (GEMMINI_DMA) {
            // local_a_consume = (k_index % 2) ? local_a_buf : local_a;
            // local_b_consume = (k_index % 2) ? local_b_buf : local_b;
            // FIXME: swap multiply with bitshifts
            // const uint32_t mask_odd = (block_k & 1) << 31 >> 31;
            // const uint32_t mask_even = ((block_k & 1) ^ 1) << 31 >> 31;
            // local_a_consume = reinterpret_cast<volatile float *>(
            //     (mask_odd & reinterpret_cast<uintmax_t>(local_a_buf)) |
            //     (mask_even & reinterpret_cast<uintmax_t>(local_a)));
            // local_b_consume = reinterpret_cast<volatile float *>(
            //     (mask_odd & reinterpret_cast<uintmax_t>(local_b_buf)) |
            //     (mask_even & reinterpret_cast<uintmax_t>(local_b)));
            local_a_consume = local_a + (block_k & 1) * (local_a_elems);
            local_b_consume = local_b + (block_k & 1) * (local_b_elems);
          } else {
            local_a_consume = local_a;
            local_b_consume = local_b;
          }

#pragma GCC unroll 1
          for (int i = 0; i < BK_LOOP; i++) {
#pragma GCC unroll 4
            for (uint32_t local_k = 0; local_k < BK; local_k += TCK) {
#pragma GCC unroll 2
              for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
                // SMEM -> RF
                vx_wmma_load_b(local_b_consume, local_k, warp_col, wn_iter, tid_in_warp);
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

          if constexpr (GEMMINI_DMA) {
            // Call gemmini fence at the end of the loop to overlap dma & wmma.
            // Hopefully by this time, dma would have finished so that this is a
            // no-op
            if (tid_in_threadblock == 0) {
              gemmini_fence();
            }
          }

          // threadblock_barrier(threadblock_id_in_cluster, threadblock_dim_y);
          threadblock_barrier(1 /*gemm warpgroup*/,
                              threadblock_dim_y / (WARP_SPECIALIZED ? 2 : 1));
        }
      }

      // warp specialization: activation on the previous (M,N) tile in the 2nd
      // warpgroup
      if (WARP_SPECIALIZED && warpgroup_id == 1) {
        const uint32_t warp_id_in_threadblock =
            tid_in_threadblock / NUM_THREADS;
        const uint32_t warp_row = warp_id_in_threadblock / (BN / WN);
        const uint32_t warp_col = warp_id_in_threadblock % (BN / WN);
        activate_block(dim_n, C, block_m, block_n, warp_row, warp_col,
                       tid_in_threadblock);
      }

      // global barrier that synchronizes both warpgroups at every M-N
      // iteration
      threadblock_barrier(0/*all warpgroups*/, threadblock_dim_y);

#pragma GCC unroll 2
      for (int wm_iter = 0; wm_iter < WMITER; wm_iter++) {
#pragma GCC unroll 2
        for (int wn_iter = 0; wn_iter < WNITER; wn_iter++) {
          write_results(tid_in_warp, warp_col, warp_row, wn_iter, wm_iter,
                        dim_n, C, block_n, block_m);
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

  uint32_t threads_per_threadblock =
      (BM * BN) / (ELEM_PER_THREAD) * (WARP_SPECIALIZED ? 2 : 1);
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

  // "static" shared memory allocation.  This would determine threadblock
  // occupancy of a single cluster
  float *sharedmem_per_threadblock =
      (float *)DEV_SMEM_START_ADDR + 2/*overkill for non-dma*/ * ((BM + BN) * BK) *
                                         threadblock_id_in_cluster;

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

  const uint32_t problem_size = (arg->dim_m * arg->dim_n) / (ELEM_PER_THREAD) *
                                (WARP_SPECIALIZED ? 2 : 1);
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
