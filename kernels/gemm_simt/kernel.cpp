#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#define NUM_WARPS 4

// all numbers below in number of BF16 elements
#define BK 32
#define TM 2
#define TN 4
#define TB_X 16
#define TB_Y 8
#define BLOCK_X TB_X * TM
#define BLOCK_Y TB_Y * TN
#define BLOCK_SIZE BLOCK_X * BLOCK_Y

#define THREADBLOCK_SIZE MU_BLOCK_SIZE(NUM_WARPS)
#define NUM_A_CHUNKS BLOCK_X * BK / THREADBLOCK_SIZE // ENSURE BLOCK_X * BK >= THREADBLOCK_SIZE and is a multiple of THREADBLOCK_SIZE
#define NUM_B_CHUNKS BK * BLOCK_Y / THREADBLOCK_SIZE // same as above

extern "C" uint32_t __mu_num_warps = NUM_WARPS;

struct GEMMArgs {
  __global uint32_t* A;
  __global uint32_t* B;
  __global uint32_t* C;
  uint32_t M;
  uint32_t K;
  uint32_t N;
};

__shared uint32_t* const sdata = reinterpret_cast<__shared uint32_t*>(0x0);

// C = A * B where A is MxK, B is KxN, C is MxN (all bf16)
static inline void gemm(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<GEMMArgs*>(arg);
  uint32_t lane_id = tid_in_threadblock % 16;
  uint32_t warp_id = tid_in_threadblock / 16;
  uint32_t tid = tid_in_threadblock;
  uint32_t total_blocks = args->M * args->N / BLOCK_SIZE;
  uint32_t blocks_per_cluster = total_blocks / MU_NUM_CLUSTERS;
  uint32_t block_N = args->N / BLOCK_Y;

  uint32_t M = args->M;
  uint32_t N = args->N;
  uint32_t K = args->K;
  __global uint32_t *A = args->A;
  __global uint32_t *B = args->B;
  __shared uint32_t *As = sdata;
  __shared uint32_t *Bs = sdata + BLOCK_X * BK / 2;

  for (uint32_t c_block = 0; c_block < blocks_per_cluster; c_block++) {
    uint32_t block_idx = threadblock_id * blocks_per_cluster + c_block;
    uint32_t block_x_idx = block_idx / block_N;
    uint32_t block_y_idx = block_idx % block_N;

    //grab x and y for each thread's TB_X x TB_Y C subblock 
    uint32_t thread_x = tid_in_threadblock / TB_Y;
    uint32_t thread_y = tid_in_threadblock % TB_Y;

    //accum
    _Float16 acc[TM * TN];
    for (uint32_t i = 0; i < TM*TN; i++) acc[i] = 0;

    //stream across K
    for (uint32_t k = 0; k < args->K; k += BK) {
      //load A block to smem
      for (uint32_t a_block = 0; a_block < NUM_A_CHUNKS; a_block++) {
        uint32_t elem_idx = tid_in_threadblock + a_block * THREADBLOCK_SIZE;
        uint32_t A_x = block_x_idx + (elem_idx / (BK / 2)); // BK bf16 elements = BK/2 uint32_t elements
        uint32_t A_y = k + (elem_idx % (BK / 2));
        As[elem_idx] = A[A_x * K + A_y];
      }
      //load B block to smem
      for (uint32_t b_block = 0; b_block < NUM_B_CHUNKS; b_block++) {
        uint32_t elem_idx = tid_in_threadblock + b_block * THREADBLOCK_SIZE;
        uint32_t B_y = block_y_idx + (elem_idx % (BLOCK_Y / 2)); // BLOCK_Y bf16 elements
        uint32_t B_x = k + (elem_idx % (BLOCK_Y / 2));
        Bs[elem_idx] = B[B_x * N + B_y];
      }
    }

  }
}

GEMMArgs gemm_args = {
  .A = nullptr,
  .B = nullptr,
  .C = nullptr,
  .M = 0,
  .K = 0,
  .N = 0,
};

#include "data"

int main() {
  gemm_args.A = reinterpret_cast<__global uint32_t*>(A_raw);
  gemm_args.B = reinterpret_cast<__global uint32_t*>(B_raw);
  gemm_args.C = reinterpret_cast<__global uint32_t*>(C_raw);
  gemm_args.M = M_val;
  gemm_args.K = K_val;
  gemm_args.N = N_val;
  mu_schedule(gemm, &gemm_args, NUM_WARPS);
  return 0;
}
