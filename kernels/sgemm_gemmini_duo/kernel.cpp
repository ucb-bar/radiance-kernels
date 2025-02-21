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

#define NUM_THREADS_IN_CLUSTER 256
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

inline void gemm_in_gemmini(const uint32_t gemmini_i, const uint32_t dim_m,
                            const uint32_t dim_n, const uint32_t dim_k,
                            const float *A, const float *B, float *C,
                            const uint32_t threadblock_id) {
  uint32_t marker0, marker1;
  rd_cycles_force(marker0);

  const uint32_t num_tiles_m = dim_m / TILE_M;
  const uint32_t num_tiles_n = dim_n / TILE_N;
  const uint32_t num_tiles_k = dim_k / TILE_K;

  const uint32_t num_tile_rows_per_tb = num_tiles_m / NUM_CLUSTERS;

  use_gemmini(gemmini_i);
  if (HW_TID() == 0) {
    gemmini_extended3_config_ld(dim_k * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(dim_n * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
    gemmini_extended_config_st(dim_n * sizeof(elem_t), 0, MVIN_SCALE_IDENTITY);
  }
  for (uint32_t tile_i = num_tile_rows_per_tb * threadblock_id;
       tile_i < num_tile_rows_per_tb * (threadblock_id + 1);
       tile_i += 1) {
    for (int tile_j = 0; tile_j < num_tiles_n; tile_j += 1) {
      if (HW_TID() == 0) {
        for (int tile_k = 0; tile_k < num_tiles_k; tile_k += 1) {
          ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
                                   (uint64_t) (A + tile_i * TILE_M * dim_k + tile_k * TILE_K),
                                   (uint64_t) (B + tile_k * TILE_K * dim_n + tile_j * TILE_N),
                                   k_LOOP_WS_CONFIG_ADDRS_AB)
          GEMMINI_CISC_CMD_R((dim_n) << 16 | (dim_k << 8) | GEMMINI_CISC_IMM(8, gemmini_i));
          /* DMA move-in */
          if (tile_k & 1) {
            GEMMINI_CISC_CMD_I(GEMMINI_CISC_IMM(11, gemmini_i));
          } else {
            GEMMINI_CISC_CMD_I(GEMMINI_CISC_IMM(10, gemmini_i));
          }
          /* compute in-SMEM */
          if (tile_k == 0) {
            gemmini_fence();
            GEMMINI_CISC_CMD_I(GEMMINI_CISC_IMM(0, gemmini_i));
          } else if (tile_k & 1) {
            gemmini_fence();
            GEMMINI_CISC_CMD_I(GEMMINI_CISC_IMM(2, gemmini_i));
          } else {
            gemmini_fence();
            GEMMINI_CISC_CMD_I(GEMMINI_CISC_IMM(1, gemmini_i));
          }
        }
        gemmini_fence();
        gemmini_fence();
        gemmini_fence();
        gemmini_fence();
        GEMMINI_CISC_CMD_I(GEMMINI_CISC_IMM(9, gemmini_i));
        gemmini_fence();
      }
      threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS);
      if (HW_TID() == 0) {
        float *const dram_c_tile_start = C + tile_i * TILE_M * dim_n + tile_j * TILE_N;
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, BOUND_INST, k_LOOP_WS_CONFIG_BOUNDS)
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, (uint64_t) dram_c_tile_start, k_LOOP_WS_CONFIG_ADDRS_DC)
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, dim_n, k_LOOP_WS_CONFIG_STRIDES_DC)
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, loop_matmul_skips(1, 1, 1, 1, 0), k_LOOP_WS)
      }
    }
  }
  if (gemmini_i != 0 && threadblock_id == NUM_CLUSTERS - 1) {
    threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS);
    rd_cycles_force(marker1);
    if (HW_TID() == 0) {
      PRINTF("\ncomplete on core %d\n", gemmini_i);
      PRINTF("total cycles:         %d\n", marker1 - marker0);
      for (int i = 0; i < 1 /*dim_m*/; i += 8) { /* print one line only for quick test running */
        for (int j = 0; j < dim_n; j += 8) {
          PRINTF("%d %d ", (int) (C[i * dim_n + j]), (int) (C[i * dim_n + j + 4]));
        }
        PRINTF("\n");
      }
    }
  }
  threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS);
}

void thread_block_matmul_gemmini(kernel_arg_t *__UNIFORM__ arg,
                                 const uint32_t threadblock_id,
                                 const uint32_t tid_in_threadblock) {
  asm volatile ("matmul_start_%=:"::);
  const float *const A = (const float *const) arg->addr_a;
  const float *const B = (const float *const) arg->addr_b;
  float *const C = (float *const) arg->addr_c;

  if (HW_TID() == 0) {
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    use_gemmini(1);
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    use_gemmini(0);
    // gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);
    PRINTF("start\n");
  }

  vx_fence();

  uint32_t marker0, marker1;
  rd_cycles_force(marker0);

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;

  gemm_in_gemmini(0, dim_m, dim_n, dim_k, A, B, C, threadblock_id);
  gemm_in_gemmini(1, dim_m, dim_n, dim_k, A, B, C, threadblock_id);

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
