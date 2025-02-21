#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define NUM_CLUSTERS 1
// #define FP32
#define HOPPER

#ifdef FP32
// fp32
#define TILE_M 64
#define TILE_N 64
#define TILE_K 64
#define TILE_MN 4096
#define TILE_MK 4096
#define TILE_NK 4096
#define NUM_THREADS_IN_CLUSTER 256

#define SMEM_ADDR_Q0 ((mem_elem_t * const) 0xff000000)
#define SMEM_ADDR_Q1 ((mem_elem_t * const) 0xff004000)
#define SMEM_ADDR_Q2 ((mem_elem_t * const) 0xff008000)
#define SMEM_ADDR_Q3 ((mem_elem_t * const) 0xff00c000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x200
#define SPAD_ADDR_Q2 0x400
#define SPAD_ADDR_Q3 0x600
#define SPAD_ADDR_Q4 0x800
typedef float smem_elem_t;
typedef float mem_elem_t;

#else
// fp16
#define TILE_M 128
#define TILE_N 64
#define TILE_K 128
#define TILE_MN 8192
#define TILE_MK 16384
#define TILE_NK 8192
#define NUM_WARPS_ 8
#define NUM_THREADS_ 8

#ifdef HOPPER
#define NUM_THREADS_IN_CLUSTER 256
#define CORES_PER_CLUSTER_ 4
#else
#define NUM_THREADS_IN_CLUSTER 512
#define CORES_PER_CLUSTER_ 8
#endif

#define SMEM_ADDR_Q0 ((mem_elem_t * const) 0xff000000)
#define SMEM_ADDR_Q1 ((mem_elem_t * const) 0xff008000)
#define SMEM_ADDR_Q2 ((mem_elem_t * const) 0xff001000)
#define SMEM_ADDR_Q3 ((mem_elem_t * const) 0xff018000)
#define SPAD_ADDR_Q0 0x0
#define SPAD_ADDR_Q1 0x400
#define SPAD_ADDR_Q2 0x800
#define SPAD_ADDR_Q3 0xc00
#define SPAD_ADDR_Q4 0x1000
typedef uint16_t smem_elem_t;
typedef uint32_t mem_elem_t;
#endif

#define HARDCODE
#define REGBLOCK
#define OFFLOAD_ACCUMULATE
#define REMATERIALIZE
#define DBUF
//#define CISC
#define POWER

//#define DEBUG_PRINT
//#define DETAILED_PERF
//#define ACTIVATE

#define rd_cycles_force(x) asm volatile ("csrr %0, mcycle" : "=r" (x))
#ifdef DETAILED_PERF
  #define rd_cycles(x) rd_cycles_force(x)
#else
  #define rd_cycles(x)
#endif
#ifdef REMATERIALIZE
  #define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})
#else
  #define HW_TID() hw_tid
#endif
#define PRINTF(...) sprintf(PRINT_BUF, __VA_ARGS__)
// #define PRINTF(...) vx_printf(__VA_ARGS__)
#define SWISH(beta, x) ((x) / (1 + exp(-(beta) * (x))))

inline void threadblock_barrier(unsigned int barrier_id, unsigned int count) {
  vx_fence();
  vx_barrier(barrier_id, count);
}

void thread_block_matmul_gemmini(kernel_arg_t *__UNIFORM__ arg,
                                 const uint32_t threadblock_id,
                                 const uint32_t tid_in_threadblock) {
  const smem_elem_t * const A = (const smem_elem_t * const) arg->addr_a;
  const smem_elem_t * const B = (const smem_elem_t * const) arg->addr_b;
  smem_elem_t * const C = (smem_elem_t * const) arg->addr_c;

  if (tid_in_threadblock % NUM_THREADS_IN_CLUSTER == 0) {
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);
    // gemmini_extended_config_ex(dataflow, act & 3, 0, 1, a_transpose, b_transpose);

    // gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
    #ifndef POWER
    PRINTF("start\n");
    #endif
  }

  vx_fence();

  uint32_t marker0, marker1, marker2, marker3, marker4;
  uint32_t marker5, marker6, marker7, marker8, marker9;
  #ifdef ACTIVATE
  uint32_t swish_dur = 0;
  #endif
  rd_cycles_force(marker0);
  MARK_BEG();

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_k = arg->dim_k;
  const uint32_t num_tiles_m = dim_m / TILE_M;
  const uint32_t num_tiles_n = dim_n / TILE_N;
  const uint32_t num_tiles_k = dim_k / TILE_K;
  constexpr uint32_t num_threads_in_cluster = NUM_THREADS_IN_CLUSTER;
  constexpr uint32_t a_elems_per_thread = TILE_MK / num_threads_in_cluster;
  constexpr uint32_t b_elems_per_thread = TILE_NK / num_threads_in_cluster;
  constexpr uint32_t c_elems_per_thread = TILE_MN / num_threads_in_cluster;
  constexpr uint32_t e_mult = sizeof(uint32_t) / sizeof(smem_elem_t);
  const uint32_t hw_tid = tid_in_threadblock % num_threads_in_cluster;

  // the dram coordinates are (i1 + i0, j1 + j0). i0 and j0 are both spatially mapped only.
  const uint32_t j0 = HW_TID() * e_mult % DIM;
  const uint32_t i0 = (HW_TID() * e_mult / DIM) % DIM;

  // j1 is both spatially and temporally mapped. j1 increases every iteration.
  const uint32_t j1_idx = (HW_TID() * e_mult / DIM / DIM) * DIM; // A: % TILE_K, B: % TILE_N, C: % TILE_N
  // every iteratioon, j1 increases by j1_stride
  constexpr uint32_t j1_stride = (num_threads_in_cluster * e_mult / DIM / DIM) * DIM; // mod TILE_W after stride

  // i1 is only temporally mapped. i1 increments every one or more iterations
  constexpr uint32_t i1_stride = DIM; // step per increment (increment doesnt happen every iteration)
  constexpr uint32_t i1_iters_a = (DIM * DIM * (TILE_K / DIM)) / num_threads_in_cluster / e_mult; // num of iters before striding
  constexpr uint32_t i1_iters_b = (DIM * DIM * (TILE_N / DIM)) / num_threads_in_cluster / e_mult; // num of iters before striding

  const uint32_t num_tile_rows_per_tb = num_tiles_m / NUM_CLUSTERS;

  if (HW_TID() == 0) {
    gemmini_extended3_config_ld(dim_k * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(dim_n * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
    // gemmini_extended3_config_ld(repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
    gemmini_extended_config_st(dim_n * sizeof(elem_t), 0, MVIN_SCALE_IDENTITY);
    // gemmini_extended_config_st(stride_C * sizeof_C, act & 3, scale);
  }

  for (uint32_t tile_i = num_tile_rows_per_tb * threadblock_id;
                tile_i < num_tile_rows_per_tb * (threadblock_id + 1);
                tile_i += 1) {
    for (int tile_j = 0; tile_j < num_tiles_n; tile_j += 1) {
      mem_elem_t * const smem_c_tile_start = SMEM_ADDR_Q1;
      #ifdef OFFLOAD_ACCUMULATE
      mem_elem_t * const smem_acc_tile_start = SMEM_ADDR_Q0 + HW_TID();
      #else
      mem_elem_t * const smem_acc_tile_start = SMEM_ADDR_Q2 + hw_tid;
      #endif

#ifndef FP32
#ifdef HOPPER
      constexpr uint32_t every_iter = j1_stride;
      const uint32_t every_4iters_a = i1_stride * dim_k;
      const uint32_t runtime_const_a = i0 * dim_k + j1_idx + j0;
      const uint32_t every_2iters_b = i1_stride * dim_n;
      const uint32_t runtime_const_b = i0 * dim_n + j1_idx + j0;
#else
      constexpr uint32_t every_iter = j1_stride;
      const uint32_t every_2iters_a = i1_stride * dim_k;
      const uint32_t runtime_const_a = i0 * dim_k + j1_idx + j0;
      const uint32_t every_iter_b = i1_stride * dim_n;
      const uint32_t runtime_const_b = i0 * dim_n + j1_idx + j0;
#endif
#else
      constexpr uint32_t every_iter = j1_stride;
      const uint32_t every_2iters_a = i1_stride * dim_k;
      const uint32_t runtime_const_a = i0 * dim_k + j1_idx + j0;
      const uint32_t every_2iters_b = i1_stride * dim_n;
      const uint32_t runtime_const_b = i0 * dim_n + j1_idx + j0;
#endif

      for (int tile_k = 0; tile_k < num_tiles_k; tile_k += 1) {
        // TODO: double buffer
        rd_cycles(marker1);

        #ifdef HARDCODE
          // #if (TILE_MK / NUM_THREADS / NUM_WARPS_ / CORES_PER_CLUSTER) != 8
          //   #error CANNOT UNROLL
          // #endif

        const mem_elem_t * const dram_a_tile_start = (const mem_elem_t * const) (A + tile_i * TILE_M * dim_k + tile_k * TILE_K + runtime_const_a);
        const mem_elem_t * const dram_b_tile_start = (const mem_elem_t * const) (B + tile_k * TILE_K * dim_n + tile_j * TILE_N + runtime_const_b);
        #ifdef DBUF
        mem_elem_t * const smem_a_tile_start = (mem_elem_t * const) (((tile_k & 1) ? SMEM_ADDR_Q1 : SMEM_ADDR_Q0) + HW_TID() * e_mult);
        mem_elem_t * const smem_b_tile_start = (mem_elem_t * const) (((tile_k & 1) ? SMEM_ADDR_Q3 : SMEM_ADDR_Q2) + HW_TID() * e_mult);
        #else
        mem_elem_t * const smem_a_tile_start = (mem_elem_t * const) (SMEM_ADDR_Q0 + HW_TID() * e_mult);
        mem_elem_t * const smem_b_tile_start = (mem_elem_t * const) (SMEM_ADDR_Q3 + HW_TID() * e_mult);
        #endif

        {
        #ifndef REGBLOCK
            /*
          smem_a_tile_start[0 * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[every_iter * 0 + every_2iters_a * 0];
          smem_a_tile_start[1 * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[every_iter * 1 + every_2iters_a * 0];
          smem_a_tile_start[2 * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[every_iter * 0 + every_2iters_a * 1];
          smem_a_tile_start[3 * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[every_iter * 1 + every_2iters_a * 1];
          smem_a_tile_start[4 * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[every_iter * 0 + every_2iters_a * 2];
          smem_a_tile_start[5 * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[every_iter * 1 + every_2iters_a * 2];
          smem_a_tile_start[6 * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[every_iter * 0 + every_2iters_a * 3];
          smem_a_tile_start[7 * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[every_iter * 1 + every_2iters_a * 3];

          smem_b_tile_start[0 * num_threads_in_cluster + hw_tid] = \
            dram_b_tile_start[every_iter * 0 + every_2iters_b * 0];
          smem_b_tile_start[1 * num_threads_in_cluster + hw_tid] = \
            dram_b_tile_start[every_iter * 1 + every_2iters_b * 0];
          smem_b_tile_start[2 * num_threads_in_cluster + hw_tid] = \
            dram_b_tile_start[every_iter * 0 + every_2iters_b * 1];
          smem_b_tile_start[3 * num_threads_in_cluster + hw_tid] = \
            dram_b_tile_start[every_iter * 1 + every_2iters_b * 1];
          smem_b_tile_start[4 * num_threads_in_cluster + hw_tid] = \
            dram_b_tile_start[every_iter * 0 + every_2iters_b * 2];
          smem_b_tile_start[5 * num_threads_in_cluster + hw_tid] = \
            dram_b_tile_start[every_iter * 1 + every_2iters_b * 2];
          smem_b_tile_start[6 * num_threads_in_cluster + hw_tid] = \
            dram_b_tile_start[every_iter * 0 + every_2iters_b * 3];
          smem_b_tile_start[7 * num_threads_in_cluster + hw_tid] = \
            dram_b_tile_start[every_iter * 1 + every_2iters_b * 3];
            */
        #else
#ifndef FP32
#ifdef HOPPER
          mem_elem_t v0 = dram_a_tile_start[every_iter * 0 + every_4iters_a * 0];
          mem_elem_t v1 = dram_a_tile_start[every_iter * 1 + every_4iters_a * 0];
          mem_elem_t v2 = dram_a_tile_start[every_iter * 2 + every_4iters_a * 0];
          mem_elem_t v3 = dram_a_tile_start[every_iter * 3 + every_4iters_a * 0];
          smem_a_tile_start[0 * num_threads_in_cluster] = v0;
          smem_a_tile_start[1 * num_threads_in_cluster] = v1;
          smem_a_tile_start[2 * num_threads_in_cluster] = v2;
          smem_a_tile_start[3 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_4iters_a * 1];
          v1 = dram_a_tile_start[every_iter * 1 + every_4iters_a * 1];
          v2 = dram_a_tile_start[every_iter * 2 + every_4iters_a * 1];
          v3 = dram_a_tile_start[every_iter * 3 + every_4iters_a * 1];
          smem_a_tile_start[4 * num_threads_in_cluster] = v0;
          smem_a_tile_start[5 * num_threads_in_cluster] = v1;
          smem_a_tile_start[6 * num_threads_in_cluster] = v2;
          smem_a_tile_start[7 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_4iters_a * 2];
          v1 = dram_a_tile_start[every_iter * 1 + every_4iters_a * 2];
          v2 = dram_a_tile_start[every_iter * 2 + every_4iters_a * 2];
          v3 = dram_a_tile_start[every_iter * 3 + every_4iters_a * 2];
          smem_a_tile_start[8  * num_threads_in_cluster] = v0;
          smem_a_tile_start[9  * num_threads_in_cluster] = v1;
          smem_a_tile_start[10 * num_threads_in_cluster] = v2;
          smem_a_tile_start[11 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_4iters_a * 3];
          v1 = dram_a_tile_start[every_iter * 1 + every_4iters_a * 3];
          v2 = dram_a_tile_start[every_iter * 2 + every_4iters_a * 3];
          v3 = dram_a_tile_start[every_iter * 3 + every_4iters_a * 3];
          smem_a_tile_start[12 * num_threads_in_cluster] = v0;
          smem_a_tile_start[13 * num_threads_in_cluster] = v1;
          smem_a_tile_start[14 * num_threads_in_cluster] = v2;
          smem_a_tile_start[15 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_4iters_a * 4];
          v1 = dram_a_tile_start[every_iter * 1 + every_4iters_a * 4];
          v2 = dram_a_tile_start[every_iter * 2 + every_4iters_a * 4];
          v3 = dram_a_tile_start[every_iter * 3 + every_4iters_a * 4];
          smem_a_tile_start[16 * num_threads_in_cluster] = v0;
          smem_a_tile_start[17 * num_threads_in_cluster] = v1;
          smem_a_tile_start[18 * num_threads_in_cluster] = v2;
          smem_a_tile_start[19 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_4iters_a * 5];
          v1 = dram_a_tile_start[every_iter * 1 + every_4iters_a * 5];
          v2 = dram_a_tile_start[every_iter * 2 + every_4iters_a * 5];
          v3 = dram_a_tile_start[every_iter * 3 + every_4iters_a * 5];
          smem_a_tile_start[20 * num_threads_in_cluster] = v0;
          smem_a_tile_start[21 * num_threads_in_cluster] = v1;
          smem_a_tile_start[22 * num_threads_in_cluster] = v2;
          smem_a_tile_start[23 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_4iters_a * 6];
          v1 = dram_a_tile_start[every_iter * 1 + every_4iters_a * 6];
          v2 = dram_a_tile_start[every_iter * 2 + every_4iters_a * 6];
          v3 = dram_a_tile_start[every_iter * 3 + every_4iters_a * 6];
          smem_a_tile_start[24 * num_threads_in_cluster] = v0;
          smem_a_tile_start[25 * num_threads_in_cluster] = v1;
          smem_a_tile_start[26 * num_threads_in_cluster] = v2;
          smem_a_tile_start[27 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_4iters_a * 7];
          v1 = dram_a_tile_start[every_iter * 1 + every_4iters_a * 7];
          v2 = dram_a_tile_start[every_iter * 2 + every_4iters_a * 7];
          v3 = dram_a_tile_start[every_iter * 3 + every_4iters_a * 7];
          smem_a_tile_start[28 * num_threads_in_cluster] = v0;
          smem_a_tile_start[29 * num_threads_in_cluster] = v1;
          smem_a_tile_start[30 * num_threads_in_cluster] = v2;
          smem_a_tile_start[31 * num_threads_in_cluster] = v3;

          // --------------------

          v0 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 0];
          v1 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 0];
          v2 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 1];
          v3 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 1];
          smem_b_tile_start[0 * num_threads_in_cluster] = v0;
          smem_b_tile_start[1 * num_threads_in_cluster] = v1;
          smem_b_tile_start[2 * num_threads_in_cluster] = v2;
          smem_b_tile_start[3 * num_threads_in_cluster] = v3;

          v0 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 2];
          v1 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 2];
          v2 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 3];
          v3 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 3];
          smem_b_tile_start[4 * num_threads_in_cluster] = v0;
          smem_b_tile_start[5 * num_threads_in_cluster] = v1;
          smem_b_tile_start[6 * num_threads_in_cluster] = v2;
          smem_b_tile_start[7 * num_threads_in_cluster] = v3;

          v0 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 4];
          v1 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 4];
          v2 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 5];
          v3 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 5];
          smem_b_tile_start[8 * num_threads_in_cluster] = v0;
          smem_b_tile_start[9 * num_threads_in_cluster] = v1;
          smem_b_tile_start[10 * num_threads_in_cluster] = v2;
          smem_b_tile_start[11 * num_threads_in_cluster] = v3;

          v0 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 6];
          v1 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 6];
          v2 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 7];
          v3 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 7];
          smem_b_tile_start[12 * num_threads_in_cluster] = v0;
          smem_b_tile_start[13 * num_threads_in_cluster] = v1;
          smem_b_tile_start[14 * num_threads_in_cluster] = v2;
          smem_b_tile_start[15 * num_threads_in_cluster] = v3;
#else
          mem_elem_t v0 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 0];
          mem_elem_t v1 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 0];
          mem_elem_t v2 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 1];
          mem_elem_t v3 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 1];
          smem_a_tile_start[0 * num_threads_in_cluster] = v0;
          smem_a_tile_start[1 * num_threads_in_cluster] = v1;
          smem_a_tile_start[2 * num_threads_in_cluster] = v2;
          smem_a_tile_start[3 * num_threads_in_cluster] = v3;

          v0 = dram_b_tile_start[every_iter * 0 + every_iter_b * 0];
          v1 = dram_b_tile_start[every_iter * 0 + every_iter_b * 1];
          v2 = dram_b_tile_start[every_iter * 0 + every_iter_b * 2];
          v3 = dram_b_tile_start[every_iter * 0 + every_iter_b * 3];
          smem_b_tile_start[0 * num_threads_in_cluster] = v0;
          smem_b_tile_start[1 * num_threads_in_cluster] = v1;
          smem_b_tile_start[2 * num_threads_in_cluster] = v2;
          smem_b_tile_start[3 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 2];
          v1 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 2];
          v2 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 3];
          v3 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 3];
          smem_a_tile_start[4 * num_threads_in_cluster] = v0;
          smem_a_tile_start[5 * num_threads_in_cluster] = v1;
          smem_a_tile_start[6 * num_threads_in_cluster] = v2;
          smem_a_tile_start[7 * num_threads_in_cluster] = v3;

          v0 = dram_b_tile_start[every_iter * 0 + every_iter_b * 4];
          v1 = dram_b_tile_start[every_iter * 0 + every_iter_b * 5];
          v2 = dram_b_tile_start[every_iter * 0 + every_iter_b * 6];
          v3 = dram_b_tile_start[every_iter * 0 + every_iter_b * 7];
          smem_b_tile_start[4 * num_threads_in_cluster] = v0;
          smem_b_tile_start[5 * num_threads_in_cluster] = v1;
          smem_b_tile_start[6 * num_threads_in_cluster] = v2;
          smem_b_tile_start[7 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 4];
          v1 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 4];
          v2 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 5];
          v3 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 5];
          smem_a_tile_start[8  * num_threads_in_cluster] = v0;
          smem_a_tile_start[9  * num_threads_in_cluster] = v1;
          smem_a_tile_start[10 * num_threads_in_cluster] = v2;
          smem_a_tile_start[11 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 6];
          v1 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 6];
          v2 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 7];
          v3 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 7];
          smem_a_tile_start[12 * num_threads_in_cluster] = v0;
          smem_a_tile_start[13 * num_threads_in_cluster] = v1;
          smem_a_tile_start[14 * num_threads_in_cluster] = v2;
          smem_a_tile_start[15 * num_threads_in_cluster] = v3;
#endif
#else
          mem_elem_t v0 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 0];
          mem_elem_t v1 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 0];
          mem_elem_t v2 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 1];
          mem_elem_t v3 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 1];
          smem_a_tile_start[0 * num_threads_in_cluster] = v0;
          smem_a_tile_start[1 * num_threads_in_cluster] = v1;
          smem_a_tile_start[2 * num_threads_in_cluster] = v2;
          smem_a_tile_start[3 * num_threads_in_cluster] = v3;

          v0 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 0];
          v1 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 0];
          v2 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 1];
          v3 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 1];
          smem_b_tile_start[0 * num_threads_in_cluster] = v0;
          smem_b_tile_start[1 * num_threads_in_cluster] = v1;
          smem_b_tile_start[2 * num_threads_in_cluster] = v2;
          smem_b_tile_start[3 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 2];
          v1 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 2];
          v2 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 3];
          v3 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 3];
          smem_a_tile_start[4 * num_threads_in_cluster] = v0;
          smem_a_tile_start[5 * num_threads_in_cluster] = v1;
          smem_a_tile_start[6 * num_threads_in_cluster] = v2;
          smem_a_tile_start[7 * num_threads_in_cluster] = v3;

          v0 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 2];
          v1 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 2];
          v2 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 3];
          v3 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 3];
          smem_b_tile_start[4 * num_threads_in_cluster] = v0;
          smem_b_tile_start[5 * num_threads_in_cluster] = v1;
          smem_b_tile_start[6 * num_threads_in_cluster] = v2;
          smem_b_tile_start[7 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 4];
          v1 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 4];
          v2 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 5];
          v3 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 5];
          smem_a_tile_start[8  * num_threads_in_cluster] = v0;
          smem_a_tile_start[9  * num_threads_in_cluster] = v1;
          smem_a_tile_start[10 * num_threads_in_cluster] = v2;
          smem_a_tile_start[11 * num_threads_in_cluster] = v3;

          v0 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 4];
          v1 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 4];
          v2 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 5];
          v3 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 5];
          smem_b_tile_start[8  * num_threads_in_cluster] = v0;
          smem_b_tile_start[9  * num_threads_in_cluster] = v1;
          smem_b_tile_start[10 * num_threads_in_cluster] = v2;
          smem_b_tile_start[11 * num_threads_in_cluster] = v3;

          v0 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 6];
          v1 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 6];
          v2 = dram_a_tile_start[every_iter * 0 + every_2iters_a * 7];
          v3 = dram_a_tile_start[every_iter * 1 + every_2iters_a * 7];
          smem_a_tile_start[12 * num_threads_in_cluster] = v0;
          smem_a_tile_start[13 * num_threads_in_cluster] = v1;
          smem_a_tile_start[14 * num_threads_in_cluster] = v2;
          smem_a_tile_start[15 * num_threads_in_cluster] = v3;

          v0 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 6];
          v1 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 6];
          v2 = dram_b_tile_start[every_iter * 0 + every_2iters_b * 7];
          v3 = dram_b_tile_start[every_iter * 1 + every_2iters_b * 7];
          smem_b_tile_start[12 * num_threads_in_cluster] = v0;
          smem_b_tile_start[13 * num_threads_in_cluster] = v1;
          smem_b_tile_start[14 * num_threads_in_cluster] = v2;
          smem_b_tile_start[15 * num_threads_in_cluster] = v3;
#endif
        #endif
        }
        #else
        __asm__("loop_load_ab:");

        const float * const dram_a_tile_start = A + tile_i * TILE_M * dim_k + tile_k * TILE_K;
        const float * const dram_b_tile_start = B + tile_k * TILE_K * dim_n + tile_j * TILE_N;
        float * const smem_a_tile_start = SMEM_ADDR_Q0;
        float * const smem_b_tile_start = SMEM_ADDR_Q3;

        /* for (uint32_t thread_i = 0, j1 = 0, i1 = 0;
          thread_i < a_elems_per_thread;
          thread_i += 1,
          j1 = (j1 + j1_stride) % TILE_K,
          i1 = (thread_i % i1_iters == 0) ? i1 + i1_stride : i1) {
          smem_a_tile_start[thread_i * num_threads_in_cluster + hw_tid] = \
            dram_a_tile_start[(0 + i0) * dim_k + j1 + j1_idx + j0];
        } */
        #pragma clang loop unroll(disable)
        for (int thread_i = 0; thread_i < a_elems_per_thread; thread_i++) {
          uint32_t elem_offset = hw_tid + num_threads_in_cluster * thread_i;
          smem_a_tile_start[SMEM_MAT_OFFSET(elem_offset / TILE_K, elem_offset % TILE_K, TILE_K)] = \
            dram_a_tile_start[elem_offset / TILE_K * dim_k + elem_offset % TILE_K];
        }
        __asm__("loop_load_a_end:");
        #pragma clang loop unroll(disable)
        for (int thread_i = 0; thread_i < b_elems_per_thread; thread_i++) {
          uint32_t elem_offset = hw_tid + num_threads_in_cluster * thread_i;
          smem_b_tile_start[SMEM_MAT_OFFSET(elem_offset / TILE_N, elem_offset % TILE_N, TILE_N)] = \
            dram_b_tile_start[elem_offset / TILE_N * dim_n + elem_offset % TILE_N];
        }
        #endif

        #ifdef DEBUG_PRINT
        if (hw_tid == 0) {
          PRINTF("\nA %d %d\n", tile_i, tile_k);
          for (int i = 0; i < TILE_M; i += 8) {
            for (int j = 0; j < TILE_K; j += 8) {
              uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_K);
              PRINTF("%x %x ",
                (int) (smem_a_tile_start[mat_offset]),
                (int) (smem_a_tile_start[mat_offset + 4])
              );
            }
            PRINTF("\n");
          }
          PRINTF("\nB %d %d\n", tile_k, tile_j);
          for (int i = 0; i < TILE_K; i += 8) {
            for (int j = 0; j < TILE_N; j += 8) {
              uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_N);
              PRINTF("%x %x ",
                (int) (smem_b_tile_start[mat_offset]),
                (int) (smem_b_tile_start[mat_offset + 4])
              );
            }
            PRINTF("\n");
          }
        }
        #endif


        rd_cycles(marker2);
        // cluster wide barrier to wait for A and B loads to complete
        threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS_);
        rd_cycles(marker3);
        if (HW_TID() == 0) {
          #ifdef DBUF
            gemmini_fence();
          #endif
          #ifdef CISC
          #ifndef DBUF
          #error MUST ENABLE DBUF
          #endif
          #ifndef OFFLOAD_ACCUMULATE
          #error MUST OFFLOAD ACCUMULATE
          #endif
          if (tile_k == 0) {
            GEMMINI_CISC_CMD_I(0);
          } else if (tile_k & 1) {
            GEMMINI_CISC_CMD_I(2);
          } else {
            GEMMINI_CISC_CMD_I(1);
          }
          #else
          sp_tiled_matmul_full_spad_ws(
            #ifdef DBUF
              (tile_k & 1) ? SPAD_ADDR_Q1 : SPAD_ADDR_Q0, (tile_k & 1) ? SPAD_ADDR_Q3 : SPAD_ADDR_Q2,
            #else
              SPAD_ADDR_Q0, SPAD_ADDR_Q3,
            #endif
            /*spad_D=*/0, /*spad_C=*/SPAD_ADDR_Q1,
            /*I=*/TILE_M / DIM, /*J=*/TILE_N / DIM, /*K=*/TILE_K / DIM, /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
            /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
            #ifndef OFFLOAD_ACCUMULATE
            /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/0x38U)
            #else
            /*acc=*/tile_k != 0, /*act=*/NO_ACTIVATION, /*skips=*/0xB8U)
            #endif

          #endif
          #ifndef DBUF
          gemmini_fence();
          #endif
        }
        rd_cycles(marker4);
        #ifndef DBUF
        threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS_);
        #endif
        rd_cycles(marker5);

        // accumulate C matrix
        #ifndef OFFLOAD_ACCUMULATE
        __asm__("accumulate:");
        if (tile_k == 0) {
          #pragma GCC ivdep
          #pragma GCC unroll 8
          for (int thread_i = 0; thread_i < c_elems_per_thread; thread_i++) {
            constexpr uint32_t s = num_threads_in_cluster;
            smem_acc_tile_start[thread_i * s] = smem_c_tile_start[hw_tid + s * thread_i];
          }
        } else {
          #if (TILE_NK / NUM_THREADS_ / NUM_WARPS_ / CORES_PER_CLUSTER_) != 8
          #error CANNOT UNROLL
          #endif
          for (int thread_i = 0; thread_i < c_elems_per_thread; thread_i += 8) {
            constexpr uint32_t s = num_threads_in_cluster;
            smem_acc_tile_start[s * 0] += smem_c_tile_start[hw_tid + s * 0];
            smem_acc_tile_start[s * 1] += smem_c_tile_start[hw_tid + s * 1];
            smem_acc_tile_start[s * 2] += smem_c_tile_start[hw_tid + s * 2];
            smem_acc_tile_start[s * 3] += smem_c_tile_start[hw_tid + s * 3];
            smem_acc_tile_start[s * 4] += smem_c_tile_start[hw_tid + s * 4];
            smem_acc_tile_start[s * 5] += smem_c_tile_start[hw_tid + s * 5];
            smem_acc_tile_start[s * 6] += smem_c_tile_start[hw_tid + s * 6];
            smem_acc_tile_start[s * 7] += smem_c_tile_start[hw_tid + s * 7];
          }
        }
        __asm__("end_accumulate:");
        #endif

        #ifdef DEBUG_PRINT
        if (hw_tid == 0) {
          PRINTF("\nC %d %d %d\n", tile_i, tile_j, tile_k);
          for (int i = 0; i < TILE_M; i += 8) {
            for (int j = 0; j < TILE_N; j += 8) {
              uint32_t mat_offset = SMEM_MAT_OFFSET(i, j, TILE_N);
              PRINTF("%d %d ",
                (int) (smem_c_tile_start[mat_offset]),
                (int) (smem_c_tile_start[mat_offset + 4])
              );
            }
            PRINTF("\n");
          }
        }
        #endif
        rd_cycles(marker6);

        /* if (HW_TID() == 0) {
          PRINTF("\ntile start:           %d\n", marker1);
          PRINTF("single tile cycles:   %d\n", marker6 - marker1);
          PRINTF("A/B tile load cycles: %d\n", marker2 - marker1);
          PRINTF("first barrier:        %d\n", marker3 - marker2);
          PRINTF("gemmini cycles:       %d\n", marker4 - marker3);
          PRINTF("second barrier:       %d\n", marker5 - marker4);
        } */

      }

      #ifdef OFFLOAD_ACCUMULATE
      threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS_);
      rd_cycles(marker6);
      // mvout to scratchpad for activation
      if (HW_TID() == 0) {
//        #ifdef DBUF
//        gemmini_fence();
//        #endif
      #ifdef CISC
        GEMMINI_CISC_CMD_I(9);
      #else
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, (((uint64_t) TILE_K / DIM) << 32) |
          (((uint64_t) TILE_N / DIM) << 16) | ((uint64_t) TILE_M / DIM), k_LOOP_WS_CONFIG_BOUNDS)
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, 0x278U, k_LOOP_WS)
      #endif
        gemmini_fence();
      }
      threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS_);
      #endif
      rd_cycles(marker7);

      // move out to dram
      #ifdef HARDCODE
      // #if (TILE_MN / NUM_THREADS / NUM_WARPS_ / CORES_PER_CLUSTER) != 8
      //   #error CANNOT UNROLL
      // #endif
      mem_elem_t * const dram_c_tile_start = (mem_elem_t * const) (C + tile_i * TILE_M * dim_n + tile_j * TILE_N + runtime_const_b);

      #ifdef REGBLOCK
#ifndef FP32
#ifdef HOPPER
      mem_elem_t v0 = smem_acc_tile_start[0 * num_threads_in_cluster];
      mem_elem_t v1 = smem_acc_tile_start[1 * num_threads_in_cluster];
      mem_elem_t v2 = smem_acc_tile_start[2 * num_threads_in_cluster];
      mem_elem_t v3 = smem_acc_tile_start[3 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 0] = v0;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 0] = v1;
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 1] = v2;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 1] = v3;

      v0 = smem_acc_tile_start[4 * num_threads_in_cluster];
      v1 = smem_acc_tile_start[5 * num_threads_in_cluster];
      v2 = smem_acc_tile_start[6 * num_threads_in_cluster];
      v3 = smem_acc_tile_start[7 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 2] = v0;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 2] = v1;
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 3] = v2;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 3] = v3;

      v0 = smem_acc_tile_start[8 * num_threads_in_cluster];
      v1 = smem_acc_tile_start[9 * num_threads_in_cluster];
      v2 = smem_acc_tile_start[10 * num_threads_in_cluster];
      v3 = smem_acc_tile_start[11 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 4] = v0;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 4] = v1;
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 5] = v2;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 5] = v3;

      v0 = smem_acc_tile_start[12 * num_threads_in_cluster];
      v1 = smem_acc_tile_start[13 * num_threads_in_cluster];
      v2 = smem_acc_tile_start[14 * num_threads_in_cluster];
      v3 = smem_acc_tile_start[15 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 6] = v0;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 6] = v1;
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 7] = v2;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 7] = v3;
#else // not HOPPER
      mem_elem_t v0 = smem_acc_tile_start[0 * num_threads_in_cluster];
      mem_elem_t v1 = smem_acc_tile_start[1 * num_threads_in_cluster];
      mem_elem_t v2 = smem_acc_tile_start[2 * num_threads_in_cluster];
      mem_elem_t v3 = smem_acc_tile_start[3 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 0 + every_iter_b * 0] = v0;
      dram_c_tile_start[every_iter * 0 + every_iter_b * 1] = v1;
      dram_c_tile_start[every_iter * 0 + every_iter_b * 2] = v2;
      dram_c_tile_start[every_iter * 0 + every_iter_b * 3] = v3;

      v0 = smem_acc_tile_start[4 * num_threads_in_cluster];
      v1 = smem_acc_tile_start[5 * num_threads_in_cluster];
      v2 = smem_acc_tile_start[6 * num_threads_in_cluster];
      v3 = smem_acc_tile_start[7 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 0 + every_iter_b * 4] = v0;
      dram_c_tile_start[every_iter * 0 + every_iter_b * 5] = v1;
      dram_c_tile_start[every_iter * 0 + every_iter_b * 6] = v2;
      dram_c_tile_start[every_iter * 0 + every_iter_b * 7] = v3;
#endif // HOPPER

#else // FP32
      mem_elem_t v0 = smem_acc_tile_start[0 * num_threads_in_cluster];
      mem_elem_t v1 = smem_acc_tile_start[1 * num_threads_in_cluster];
      mem_elem_t v2 = smem_acc_tile_start[2 * num_threads_in_cluster];
      mem_elem_t v3 = smem_acc_tile_start[3 * num_threads_in_cluster];
      #ifdef ACTIVATE
      uint32_t swish_start, swish_end;
      rd_cycles_force(swish_start);
      v0 = SWISH(1, v0);
      v1 = SWISH(1, v1);
      v2 = SWISH(1, v2);
      v3 = SWISH(1, v3);
      rd_cycles_force(swish_end);
      swish_dur += swish_end - swish_start;
      #endif
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 0] = v0;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 0] = v1;
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 1] = v2;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 1] = v3;

      v0 = smem_acc_tile_start[4 * num_threads_in_cluster];
      v1 = smem_acc_tile_start[5 * num_threads_in_cluster];
      v2 = smem_acc_tile_start[6 * num_threads_in_cluster];
      v3 = smem_acc_tile_start[7 * num_threads_in_cluster];
      #ifdef ACTIVATE
      rd_cycles_force(swish_start);
      v0 = SWISH(1, v0);
      v1 = SWISH(1, v1);
      v2 = SWISH(1, v2);
      v3 = SWISH(1, v3);
      rd_cycles_force(swish_end);
      swish_dur += swish_end - swish_start;
      #endif
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 2] = v0;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 2] = v1;
      dram_c_tile_start[every_iter * 0 + every_2iters_b * 3] = v2;
      dram_c_tile_start[every_iter * 1 + every_2iters_b * 3] = v3;

      // v0 = smem_acc_tile_start[8  * num_threads_in_cluster];
      // v1 = smem_acc_tile_start[9  * num_threads_in_cluster];
      // v2 = smem_acc_tile_start[10 * num_threads_in_cluster];
      // v3 = smem_acc_tile_start[11 * num_threads_in_cluster];
      // dram_c_tile_start[every_iter * 0 + every_2iters * 4] = v0;
      // dram_c_tile_start[every_iter * 1 + every_2iters * 4] = v1;
      // dram_c_tile_start[every_iter * 0 + every_2iters * 5] = v2;
      // dram_c_tile_start[every_iter * 1 + every_2iters * 5] = v3;

      // v0 = smem_acc_tile_start[12 * num_threads_in_cluster];
      // v1 = smem_acc_tile_start[13 * num_threads_in_cluster];
      // v2 = smem_acc_tile_start[14 * num_threads_in_cluster];
      // v3 = smem_acc_tile_start[15 * num_threads_in_cluster];
      // dram_c_tile_start[every_iter * 0 + every_2iters * 6] = v0;
      // dram_c_tile_start[every_iter * 1 + every_2iters * 6] = v1;
      // dram_c_tile_start[every_iter * 0 + every_2iters * 7] = v2;
      // dram_c_tile_start[every_iter * 1 + every_2iters * 7] = v3;
#endif // FP16/FP32
      #else
      dram_c_tile_start[every_iter * 0 + every_2iters * 0] = \
        smem_acc_tile_start[0 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 1 + every_2iters * 0] = \
        smem_acc_tile_start[1 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 0 + every_2iters * 1] = \
        smem_acc_tile_start[2 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 1 + every_2iters * 1] = \
        smem_acc_tile_start[3 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 0 + every_2iters * 2] = \
        smem_acc_tile_start[4 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 1 + every_2iters * 2] = \
        smem_acc_tile_start[5 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 0 + every_2iters * 3] = \
        smem_acc_tile_start[6 * num_threads_in_cluster];
      dram_c_tile_start[every_iter * 1 + every_2iters * 3] = \
        smem_acc_tile_start[7 * num_threads_in_cluster];
      #endif

      #else
      float * const dram_c_tile_start = C + tile_i * TILE_M * dim_n + tile_j * TILE_N;
      #pragma clang loop unroll(disable)
      for (int thread_i = 0; thread_i < c_elems_per_thread; thread_i++) {
        uint32_t elem_offset = hw_tid + num_threads_in_cluster * thread_i;
        dram_c_tile_start[elem_offset / TILE_N * dim_n + elem_offset % TILE_N] = \
          *(SMEM_ADDR_Q2 + SMEM_MAT_OFFSET(elem_offset / TILE_N, elem_offset % TILE_N, TILE_N));
      }
      #endif

      // rd_cycles_force(marker8);
    }
  }
  // last thread block complete
  if (threadblock_id == NUM_CLUSTERS - 1) {
    threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS_);
    MARK_END();
    rd_cycles_force(marker9);
    #ifdef POWER
    if (HW_TID() == 0) {
        PRINTF("%d\n", marker9 - marker0);
    }
    #else
    if (HW_TID() == 0) {
      PRINTF("\ncomplete\n");
      PRINTF("total cycles:         %d\n", marker9 - marker0);
    }
    #ifdef ACTIVATE
    if (HW_TID() == 0) {
      PRINTF("swish cycles:         %d\n", swish_dur);
    }
    #endif
    #ifdef DETAILED_PERF
      vx_tmc(0x81);
      for (int x = 0; x < num_threads_in_cluster; x += num_threads_in_cluster - 1) {
        if (HW_TID() == x) {
          PRINTF("\ntile start:           %d\n", marker1);
          PRINTF("single tile cycles:   %d\n", marker6 - marker1);
          PRINTF("A/B tile load cycles: %d\n", marker2 - marker1);
          PRINTF("first barrier:        %d\n", marker3 - marker2);
          PRINTF("gemmini cycles:       %d\n", marker4 - marker3);
          PRINTF("second barrier:       %d\n", marker5 - marker4);
          #ifndef OFFLOAD_ACCUMULATE
          PRINTF("accumulation cycles:  %d\n", marker6 - marker5);
          #else
          PRINTF("smem mvout cycles:    %d %d-%d\n", marker7 - marker6, marker7, marker6);
          #endif
          PRINTF("dram mvout cycles:    %d\n", marker8 - marker7);
        }
        threadblock_barrier(/*barrier_id=*/1, /*count=*/NUM_WARPS_);
      }
    #endif
    if (HW_TID() == 0) {
      for (int i = 0; i < dim_m; i += 8) {
        for (int j = 0; j < dim_n; j += 8) {
          PRINTF("%d %d ", (int) (C[i * dim_n + j]), (int) (C[i * dim_n + j + 4]));
        }
        PRINTF("\n");
      }
    }
    #endif
  }
  threadblock_barrier(/*barrier_id=*/0, /*count=*/NUM_WARPS_);
  vx_tmc(0);
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

  const int threadblock_id = task_id / NUM_THREADS_IN_CLUSTER;
  const int tid_in_threadblock = task_id % NUM_THREADS_IN_CLUSTER;

  thread_block_matmul_gemmini(arg, threadblock_id, tid_in_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  const uint32_t num_threads_in_cluster = NUM_THREADS_IN_CLUSTER; // vx_num_threads() * vx_num_warps() * CORES_PER_CLUSTER;
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
