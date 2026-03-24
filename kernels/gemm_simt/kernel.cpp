#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#define NUM_WARPS 2

// all numbers below in number of BF16 elements
#define BK 32
#define TM 2
#define TN 2
#define TB_X 8
#define TB_Y 8
#define BLOCK_X (TB_X * TM)
#define BLOCK_Y (TB_Y * TN)
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

#define BLOCK_NUM_WARPS MU_BLOCK_NUM_WARPS(NUM_WARPS)
#define THREADBLOCK_SIZE MU_BLOCK_SIZE(NUM_WARPS)
#define NUM_A_CHUNKS (BLOCK_X * BK / 2) / THREADBLOCK_SIZE // ENSURE BLOCK_X * BK / 2 >= THREADBLOCK_SIZE and is a multiple of THREADBLOCK_SIZE
#define NUM_B_CHUNKS (BK * BLOCK_Y / 2) / THREADBLOCK_SIZE // same as above

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
  __global uint32_t *C = args->C;
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
    for (uint32_t k_block = 0; k_block < K; k_block += BK) {
      //load A block to smem
      #pragma unroll
      for (uint32_t a_block = 0; a_block < NUM_A_CHUNKS; a_block++) {
        uint32_t elem_idx = tid_in_threadblock + a_block * THREADBLOCK_SIZE;
        uint32_t A_x = block_x_idx * BLOCK_X + (elem_idx / (BK / 2)); // BK bf16 elements = BK/2 uint32_t elements
        uint32_t A_y = k_block / 2 + (elem_idx % (BK / 2));
        As[elem_idx] = A[A_x * (K / 2) + A_y];
      }
      //load B block to smem
      #pragma unroll
      for (uint32_t b_block = 0; b_block < NUM_B_CHUNKS; b_block++) {
        uint32_t elem_idx = tid_in_threadblock + b_block * THREADBLOCK_SIZE;
        uint32_t B_y = block_y_idx * BLOCK_Y / 2 + (elem_idx % (BLOCK_Y / 2)); // BLOCK_Y bf16 elements
        uint32_t B_x = k_block + (elem_idx / (BLOCK_Y / 2));
        Bs[elem_idx] = B[B_x * (N / 2) + B_y];
      }

      //hold up
      mu_fence_smem();
      mu_barrier(0, BLOCK_NUM_WARPS);

      //compute
      //j and k can vector load 2 BF16
      for (uint32_t k = 0; k < BK / 2; k++) {
        for (uint32_t i = 0; i < TM; i++) {
          uint32_t a_idx = thread_x * TM + i;
          auto [a0, a1] = unpack_bf16x2(As[a_idx * BK / 2 + k]);
          for (uint32_t j = 0; j < TN / 2; j++) {
            uint32_t b_idx = thread_y * TN / 2 + j;
            auto [b00, b10] = unpack_bf16x2(Bs[2*k * (BLOCK_Y / 2) + b_idx]);
            auto [b01, b11] = unpack_bf16x2(Bs[(2*k + 1) * (BLOCK_Y / 2) + b_idx]);
            acc[i * TN + 2 * j] += a0 * b00 + a1 * b01;
            acc[i * TN + 2 * j + 1] += a0 * b10 + a1 * b11;
          }
        }
      }

      mu_barrier(0, BLOCK_NUM_WARPS);
    }

    //store C
    uint32_t c_row = block_x_idx * BLOCK_X + thread_x * TM;
    uint32_t c_col = block_y_idx * BLOCK_Y / 2 + thread_y * TN / 2;
    #pragma unroll
    for (uint32_t i = 0; i < TM; i++) {
      uint32_t c_x = c_row + i;
      for (uint32_t j = 0; j < TN / 2; j++) {
        uint32_t c_y = c_col + j;
        C[c_x * (N / 2) + c_y] = pack_bf16x2(acc[i * TN + 2*j], acc[i * TN + 2*j + 1]);
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
