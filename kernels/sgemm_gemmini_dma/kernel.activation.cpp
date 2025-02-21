#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define TILE_M 64
#define TILE_N 64
#define TILE_K 64
#define SMEM_ADDR_Q0 ((float * const) 0xff000000)
#define SMEM_ADDR_Q1 ((float * const) 0xff004000)
#define SMEM_ADDR_Q2 ((float * const) 0xff008000)
#define SMEM_ADDR_Q3 ((float * const) 0xff00c000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x200
#define SPAD_ADDR_Q2 0x400
#define SPAD_ADDR_Q3 0x600
#define BOUND_INST 0x800080008ULL

// #define TILE_M 32
// #define TILE_N 32
// #define TILE_K 32
// #define SMEM_ADDR_Q0 ((float * const) 0xff000000)
// #define SMEM_ADDR_Q1 ((float * const) 0xff001000)
// #define SMEM_ADDR_Q2 ((float * const) 0xff002000)
// #define SMEM_ADDR_Q3 ((float * const) 0xff003000)
// #define SPAD_ADDR_Q0 0x0
// #define SPAD_ADDR_Q1 0x80
// #define SPAD_ADDR_Q2 0x100
// #define SPAD_ADDR_Q3 0x180
// #define BOUND_INST 0x400040004ULL

#define WM 16
#define WN 8
#define ELEM_PER_THREAD ((WM * WN) / NUM_THREADS)

// FIXME: NUM_THREADS and NUM_WARPS hardcoded
#if ((TILE_M * TILE_N / ELEM_PER_THREAD) > (CORES_PER_CLUSTER * 8 * 8))
#error "threadblock size too big for cluster"
#endif

#define NUM_THREADS_IN_CLUSTER 256 \
// (NUM_CORES * NUM_WARPS * NUM_THREADS)

#define rd_cycles_force(x) asm volatile ("csrr %0, mcycle" : "=r" (x))
#define rd_cycles(x) rd_cycles_force(x)
#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})
#define PRINTF(...) sprintf(PRINT_BUF, __VA_ARGS__)
// #define PRINTF(...) vx_printf(__VA_ARGS__)
#define SWISH(beta, x) ((x) / (1 + exp(-(beta) * (x))))
//#define POWER

inline void threadblock_barrier(unsigned int barrier_id, unsigned int count) {
  vx_fence();
  vx_barrier(barrier_id, count);
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
  const uint32_t C_row = (tile_i * TILE_M) + (warp_row * WM) + row_in_warptile;
  const uint32_t C_col = (tile_j * TILE_N) + (warp_col * WN) + col_in_warptile;
  float *const global_C = C + dim_n * C_row + C_col;
  const float *global_C_curr = global_C;

  // ELEM_PER_THREAD macro does not take into account warp-specialization
  constexpr uint32_t elem_per_thread = ELEM_PER_THREAD;
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

void thread_block_matmul_gemmini(kernel_arg_t *__UNIFORM__ arg,
                                 const uint32_t threadblock_id,
                                 const uint32_t tid_in_threadblock) {
  asm volatile ("matmul_start_%=:" :: );
  const float * const A = (const float * const) arg->addr_a;
  const float * const B = (const float * const) arg->addr_b;
  float * const C = (float * const) arg->addr_c;

  if (HW_TID() == 0) {
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    // gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
    PRINTF("start\n");
  }

  vx_fence();

  uint32_t marker0, marker1;
  rd_cycles_force(marker0);

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;
  const uint32_t num_tiles_m = dim_m / TILE_M;
  const uint32_t num_tiles_n = dim_n / TILE_N;
  const uint32_t num_tiles_k = dim_k / TILE_K;
  constexpr uint32_t num_threads_in_cluster = NUM_THREADS_IN_CLUSTER;

  const uint32_t local_c_row = tid_in_threadblock / TILE_N;
  const uint32_t local_c_col = tid_in_threadblock % TILE_N;

  const uint32_t num_tile_rows_per_tb = num_tiles_m / NUM_CLUSTERS;

  if (HW_TID() == 0) {
    gemmini_extended3_config_ld(dim_k * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(dim_n * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
    // gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
    gemmini_extended_config_st(dim_n * sizeof(elem_t), 0, MVIN_SCALE_IDENTITY);
    // gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
  }

  uint32_t tile_i = 0;
  uint32_t tile_j = 0;
  for (tile_i = num_tile_rows_per_tb * threadblock_id;
       tile_i < num_tile_rows_per_tb * (threadblock_id + 1);
       tile_i += 1) {
    for (tile_j = 0; tile_j < num_tiles_n; tile_j += 1) {
      for (int tile_k = 0; tile_k < num_tiles_k; tile_k += 1) {
        if (HW_TID() == 0) {
          ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
                                   (uint64_t) (A + tile_i * TILE_M * dim_k + tile_k * TILE_K),
                                   (uint64_t) (B + tile_k * TILE_K * dim_n + tile_j * TILE_N), k_LOOP_WS_CONFIG_ADDRS_AB)
          GEMMINI_CISC_CMD_R((dim_n) << 16 | (dim_k << 8) | 8);
          // DMA move in GMEM->SMEM
          if (tile_k & 1) {
            GEMMINI_CISC_CMD_I(11);
          } else {
            GEMMINI_CISC_CMD_I(10);
          }

          // compute
          if (tile_k == 0) {
            gemmini_fence();
            GEMMINI_CISC_CMD_I(0);
          } else if (tile_k & 1) {
            gemmini_fence();
            GEMMINI_CISC_CMD_I(2);
          } else {
            gemmini_fence();
            GEMMINI_CISC_CMD_I(1);
          }
        }
      }

      // while Gemmini is computing, software-pipeline with activation on the
      // previous (M,N) tile
      if ((tid_in_threadblock >= NUM_THREADS /*excludes warp 0*/)) {
        const uint32_t warp_id_in_threadblock = tid_in_threadblock / NUM_THREADS;
        const uint32_t warp_row = warp_id_in_threadblock / (TILE_N / WN);
        const uint32_t warp_col = warp_id_in_threadblock % (TILE_N / WN);
        activate_block(dim_n, C, tile_i, tile_j, warp_row, warp_col,
                       tid_in_threadblock);

        // // for warp 1, do warp 0's worth of work as well
        // if (vx_warp_id() == 1) {
        //   const uint32_t warp_row = (warp_id_in_threadblock - 1) / (TILE_N / WN);
        //   const uint32_t warp_col = (warp_id_in_threadblock - 1) % (TILE_N / WN);
        //   activate_block(dim_n, C, tile_i, tile_j, warp_row, warp_col,
        //                  tid_in_threadblock);
        // }
      }

      if (HW_TID() == 0) {
        // fence the tile computation after activation; depending on the
        // duration of activation, this can be no-op
        gemmini_fence();
        gemmini_fence();
        gemmini_fence();
        gemmini_fence();
        // // mvout to scratchpad for activation
        // GEMMINI_CISC_CMD_I(9);
        // gemmini_fence();
      }

      // synchronize activation and GEMM on (M,N) tile
      threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS);

      // move out to dram
      if (HW_TID() == 0) {
        float * const dram_c_tile_start = C + tile_i * TILE_M * dim_n + tile_j * TILE_N;
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, BOUND_INST, k_LOOP_WS_CONFIG_BOUNDS)
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, (uint64_t) dram_c_tile_start, k_LOOP_WS_CONFIG_ADDRS_DC)
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, dim_n, k_LOOP_WS_CONFIG_STRIDES_DC)
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0,
                                 loop_matmul_skips(1, 1, 1, 1, 0 /*store C*/),
                                 k_LOOP_WS)
      }
    }
  }

  // last (M,N) block activation
  if ((tid_in_threadblock >= NUM_THREADS /*excludes warp 0*/)) {
    const uint32_t warp_id_in_threadblock = tid_in_threadblock / NUM_THREADS;
    const uint32_t warp_row = warp_id_in_threadblock / (TILE_N / WN);
    const uint32_t warp_col = warp_id_in_threadblock % (TILE_N / WN);
    activate_block(dim_n, C, tile_i, tile_j, warp_row, warp_col,
                   tid_in_threadblock);

    // for warp 1, do warp 0's worth of work as well
    if (vx_warp_id() == 1) {
      const uint32_t warp_row = (warp_id_in_threadblock - 1) / (TILE_N / WN);
      const uint32_t warp_col = (warp_id_in_threadblock - 1) % (TILE_N / WN);
      activate_block(dim_n, C, tile_i, tile_j, warp_row, warp_col,
                     tid_in_threadblock);
    }
  }


  // last thread block complete
  if (threadblock_id == NUM_CLUSTERS - 1) {
    threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS);
    rd_cycles_force(marker1);
    if (HW_TID() == 0) {
      #ifdef POWER
        PRINTF("\nstart %d end %d\n", marker0, marker1);
      #else
        PRINTF("\ncomplete\n");
        PRINTF("total cycles:         %d\n", marker1 - marker0);
        for (int i = 0; i < dim_m; i += 8) {
          for (int j = 0; j < dim_n; j += 8) {
            PRINTF("%d %d ", (int) (C[i * dim_n + j]), (int) (C[i * dim_n + j + 4]));
          }
          PRINTF("\n");
        }
      #endif
    }
  }
  threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS);
  vx_tmc(0);
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  const int threadblock_id = task_id / NUM_THREADS_IN_CLUSTER;
  const int tid_in_threadblock = task_id % NUM_THREADS_IN_CLUSTER;

  thread_block_matmul_gemmini(arg, threadblock_id, tid_in_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  const uint32_t num_threads_in_cluster = NUM_THREADS_IN_CLUSTER;
  const uint32_t grid_size = num_threads_in_cluster * NUM_CLUSTERS;
#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
