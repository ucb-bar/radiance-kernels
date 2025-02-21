#define RISCV_CUSTOM3   0x7B

#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

#define BM 16
#define BN 16
#define BK 8

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

inline void vx_wmma() {
  asm volatile (".insn r %0, 0, 0, x0, x0, x0" :: "i"(RISCV_CUSTOM3));
}

void vx_wmma_load(volatile float *smem_A, volatile float *smem_B, int warp_x,
                  int warp_y, int thread_in_warp) {
  int tid = thread_in_warp;
  int tg = tid / 4;

  int row = 0;
  int col = 0;
  map_operand_32lanes(tid, row, col);

  int smem_A_m = 32;
  int smem_A_n = 8;
  int smem_B_m = 8;
  int smem_B_n = 32;

  int A_offset = (row + BM * warp_y) * smem_A_n;

  asm volatile("flw f0, %0" ::"m"(smem_A[A_offset + 0]));
  asm volatile("flw f1, %0" ::"m"(smem_A[A_offset + 1]));
  asm volatile("flw f2, %0" ::"m"(smem_A[A_offset + 2]));
  asm volatile("flw f3, %0" ::"m"(smem_A[A_offset + 3]));
  asm volatile("flw f4, %0" ::"m"(smem_A[A_offset + 4]));
  asm volatile("flw f5, %0" ::"m"(smem_A[A_offset + 5]));
  asm volatile("flw f6, %0" ::"m"(smem_A[A_offset + 6]));
  asm volatile("flw f7, %0" ::"m"(smem_A[A_offset + 7]));

  asm volatile("flw f8 , %0" ::"m"(smem_B[(0 * smem_B_n) + warp_x * BN + col]));
  asm volatile("flw f9 , %0" ::"m"(smem_B[(1 * smem_B_n) + warp_x * BN + col]));
  asm volatile("flw f10, %0" ::"m"(smem_B[(2 * smem_B_n) + warp_x * BN + col]));
  asm volatile("flw f11, %0" ::"m"(smem_B[(3 * smem_B_n) + warp_x * BN + col]));
  asm volatile("flw f12, %0" ::"m"(smem_B[(4 * smem_B_n) + warp_x * BN + col]));
  asm volatile("flw f13, %0" ::"m"(smem_B[(5 * smem_B_n) + warp_x * BN + col]));
  asm volatile("flw f14, %0" ::"m"(smem_B[(6 * smem_B_n) + warp_x * BN + col]));
  asm volatile("flw f15, %0" ::"m"(smem_B[(7 * smem_B_n) + warp_x * BN + col]));
}

inline void initialize_C() {
  // initialize C to zeros
  asm volatile("fmv.w.x f16, x0");
  asm volatile("fmv.w.x f17, x0");
  asm volatile("fmv.w.x f18, x0");
  asm volatile("fmv.w.x f19, x0");
  asm volatile("fmv.w.x f20, x0");
  asm volatile("fmv.w.x f21, x0");
  asm volatile("fmv.w.x f22, x0");
  asm volatile("fmv.w.x f23, x0");
}

inline void write_results(volatile float *local_warp_results,
                          int thread_in_warp, int warp_x, int warp_y, int dim_m,
                          int dim_n, float *C, int threadblock_id_x,
                          int threadblock_id_y) {
  int tid = thread_in_warp;
  int tg = tid / 4;

  asm volatile("fsw f16, %0" ::"m"(local_warp_results[tid * 8 + 0]));
  asm volatile("fsw f17, %0" ::"m"(local_warp_results[tid * 8 + 1]));
  asm volatile("fsw f18, %0" ::"m"(local_warp_results[tid * 8 + 2]));
  asm volatile("fsw f19, %0" ::"m"(local_warp_results[tid * 8 + 3]));
  asm volatile("fsw f20, %0" ::"m"(local_warp_results[tid * 8 + 4]));
  asm volatile("fsw f21, %0" ::"m"(local_warp_results[tid * 8 + 5]));
  asm volatile("fsw f22, %0" ::"m"(local_warp_results[tid * 8 + 6]));
  asm volatile("fsw f23, %0" ::"m"(local_warp_results[tid * 8 + 7]));

  /*
     col = ((threadgroup % 4) // 2) * 8
     row = (threadgroup * 8) % 16
     row += (threadgroup // 4) * 4
     offsets = [(0, 0), (0, 1), (2, 0), (2, 1), (0, 4), (0, 5), (2, 4), (2, 5)]
     offset = offsets[register-16]
     row += offset[0]
     col += offset[1]
     thread_offsets = [(0, 0), (1, 0), (0, 2), (1, 2)]
     thread_offset = thread_offsets[thread % 4]
     row += thread_offset[0]
     col += thread_offset[1]
     return (row, col)
     */

  int local_row = 0;
  int local_col = 0;
  map_c_32lanes(tid, local_row, local_col);

  float *global_offset_C = C +
                           (threadblock_id_y * BM * 2 + warp_y * BM) * dim_n +
                           threadblock_id_x * BN * 2 + warp_x * BM;
  for (int i = 0; i < 8; i += 1) {
    int row_offset = ((i / 2) % 2) * 2;
    int col_offset = (i / 4) * 4 + i % 2;

    int adjusted_local_row = local_row + row_offset;
    int adjusted_local_col = local_col + col_offset;

    float v = local_warp_results[tid * 8 + i];
    global_offset_C[adjusted_local_row * dim_n + adjusted_local_col] = v;
  }
}

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
                       const uint32_t threadblock_id,
                       float *sharedmem_per_threadblock) {
  const float *A = (const float *)arg->addr_a;
  const float *B = (const float *)arg->addr_b;
  float *C = (float *)arg->addr_c;

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

  const uint32_t warp_in_threadblock = tid_in_threadblock / 32;
  const uint32_t tid_in_warp = tid_in_threadblock % 32;
  const uint32_t warp_x = warp_in_threadblock % 2;
  const uint32_t warp_y = warp_in_threadblock / 2;

  const uint32_t global_a_row = threadblock_dim_y * threadblock_id_y;

  // 32 * 8 block of A, being loaded by 4 warps
  const uint32_t local_a_row = warp_y * BM + warp_x * (BM / 2) + (tid_in_warp / BK);
  const uint32_t local_a_col = tid_in_warp % BK;

  // 8 * 32 block of B, being loaded by 4 warps
  // do a fat coalesced load
  const uint32_t global_b_col = threadblock_dim_x * threadblock_id_x;
  const uint32_t local_b_row = warp_in_threadblock;
  const uint32_t local_b_col = tid_in_warp;
  
  volatile float *local_a = sharedmem_per_threadblock;
  const size_t local_a_elems = (threadblock_dim_y * BK);
  volatile float *local_b = sharedmem_per_threadblock + local_a_elems;
  const size_t local_b_elems = (threadblock_dim_x * BK);
  volatile float *local_warp_results = local_b + local_b_elems + (warp_in_threadblock * BM * BN);

  // clear out C
  initialize_C();

  for (uint32_t k = 0; k < dim_k; k += BK) {
    // Data move from GMEM to SMEM
    //
    // Make sure global offset values for A and B are contiguous between
    // neighboring threads to ensure GMEM coalescing. (not possible for A here, but for B it's doable)
        
    // coalesced load from global memory -> unit-stride store into shared memory
    uint32_t global_a_offset =
          dim_k * (global_a_row + local_a_row) + (k + local_a_col);
    uint32_t shared_a_offset =
          BK * local_a_row + local_a_col;
    
    local_a[shared_a_offset] = A[global_a_offset];
    
    global_a_offset += dim_k * (BM / 4);
    shared_a_offset += BK * (BM / 4);
    
    local_a[shared_a_offset] = A[global_a_offset];

    uint32_t global_b_offset =
          dim_n * (k + local_b_row) + (global_b_col + local_b_col);
    uint32_t shared_b_offset =
          (BN * 2) * (local_b_row) + local_b_col;
    
    local_b[shared_b_offset] = B[global_b_offset];
    
    global_b_offset += dim_n * (BK / 2);
    shared_b_offset += (BN * 2) * (BK / 2);

    local_b[shared_b_offset] = B[global_b_offset];

    // want all 4 warps to reach barrier before moving on (just use barrier 0 for now)
    threadblock_barrier(tid_in_threadblock, 0, 4);

    // perform wmma
    vx_wmma_load(local_a, local_b, warp_x, warp_y, tid_in_warp);
    vx_wmma();
    
    // same as above
    threadblock_barrier(tid_in_threadblock, 0, 4);
  }

  write_results(
    local_warp_results, 
    tid_in_warp, 
    warp_x,
    warp_y,
    dim_m,
    dim_n,
    C,
    threadblock_id_x,
    threadblock_id_y
  );
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock
  const int NT = 32; // vx_num_threads();
  const int NW = 4;  // vx_num_warps();
  const uint32_t threads_per_threadblock = NT * NW;

  // matches 4 warp capacity
  const uint32_t threadblock_dim_x = 2 * BN;
  const uint32_t threadblock_dim_y = 2 * BM;
  const int threadblock_id = task_id / threads_per_threadblock;
  const int tid_in_threadblock = task_id % threads_per_threadblock;

  const uint32_t dim_m = arg->dim_m;
  const uint32_t dim_n = arg->dim_n;
  const uint32_t dim_n_in_blocks = dim_n / threadblock_dim_x;
  const int threadblock_id_x = threadblock_id % dim_n_in_blocks;
  const int threadblock_id_y = threadblock_id / dim_n_in_blocks;

  // "static" shared memory allocation.  This would determine threadblock
  // occupancy of a single cluster
  // only 1 threadblock running at a time, so this is ok
  float *sharedmem_per_threadblock =
      (float *)DEV_SMEM_START_ADDR; // + (2 * BM * BK) + (2 * BN * BK) * threadblock_id;
  
  thread_block_gemm(arg, tid_in_threadblock, threadblock_dim_x,
                    threadblock_dim_y, threadblock_id_x, threadblock_id_y,
                    threadblock_id, sharedmem_per_threadblock);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;
  int NT = vx_num_threads();

  // TODO: add support for edge-case (m, n not divisible by 16) 
  const uint32_t grid_size = arg->dim_m * arg->dim_n * NT / (BM * BN);
  
  // for now, simplifying assumption of just 1 core
  // vx_spawn_tasks_contiguous first runs warps 1 through NW, then NW+1 through 2*NW, etc.
  // we can thus treat 1 through NW as a single threadblock for the purposes of the optimization.
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
