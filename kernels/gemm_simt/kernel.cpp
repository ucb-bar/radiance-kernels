#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#define NUM_WARPS 4

#define BM 16
#define BK 32
#define BN 16
#define TM 2
#define TN 2
#define TB_X 16
#define TB_Y 8
#define BLOCK_X TB_X * TM
#define BLOCK_Y TB_Y * TN
#define BLOCK_SIZE BLOCK_X * BLOCK_Y

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
  uint32_t block_M = args->M / BLOCK_X;
  uint32_t block_N = args->N / BLOCK_Y;

  for (uint32_t block = 0; block < blocks_per_cluster; block++) {
    uint32_t block_idx = threadblock_id * blocks_per_cluster + block;
    uint32_t block_x_idx = block_idx / block_N;
    uint32_t block_y_idx = block_idx % block_N;

    // grab x and y for each thread's TB_X x TB_Y C subblock 
    uint32_t thread_x = tid_in_threadblock / TB_Y;
    uint32_t thread_y = tid_in_threadblock % TB_Y;

    

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
