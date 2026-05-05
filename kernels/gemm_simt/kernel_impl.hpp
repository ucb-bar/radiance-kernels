#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#define ILP_MEM 2

// all numbers below in number of BF16 elements
#define BK 32
#define BM 16
#define BN 32
#ifndef TM
#define TM 1
#endif
#ifndef TN
#define TN 2
#endif
#define TBM (BM / TM)
#define TBN (BN / TN)
#define BLOCK_SIZE (BM * BN)

#define A_WORDS (BM * BK / 2)
#define B_WORDS (BK * BN / 2)
#define C_TILES (TBM * TBN)
#define GEMM_SIMT_NUM_WARPS (C_TILES / (MU_NUM_CORES * MU_NUM_THREADS))
#define BLOCK_NUM_WARPS MU_BLOCK_NUM_WARPS(GEMM_SIMT_NUM_WARPS)
#define THREADBLOCK_SIZE MU_BLOCK_SIZE(GEMM_SIMT_NUM_WARPS)
#define A_ITERS (A_WORDS / THREADBLOCK_SIZE)
#define B_ITERS (B_WORDS / THREADBLOCK_SIZE)
#define A_FULL_ITERS ((A_ITERS / ILP_MEM) * ILP_MEM)
#define B_FULL_ITERS ((B_ITERS / ILP_MEM) * ILP_MEM)

static_assert(BM % TM == 0, "Thread tile M must evenly divide the CTA tile M");
static_assert(BN % TN == 0, "Thread tile N must evenly divide the CTA tile N");
static_assert(TN % 2 == 0, "Thread tile N must contain packed BF16 pairs");
static_assert(C_TILES % (MU_NUM_CORES * MU_NUM_THREADS) == 0, "CTA thread tiles must map to full resident Muon warps");
static_assert(GEMM_SIMT_NUM_WARPS >= 1, "Derived warp occupancy must be at least 1");
static_assert(GEMM_SIMT_NUM_WARPS <= MU_NUM_MAX_WARPS, "Derived warp occupancy exceeds Muon maximum");
static_assert((GEMM_SIMT_NUM_WARPS & (GEMM_SIMT_NUM_WARPS - 1)) == 0, "Derived warp occupancy must be a power of two");
static_assert(C_TILES == THREADBLOCK_SIZE, "Each resident thread must own exactly one C microtile");
static_assert(A_WORDS % THREADBLOCK_SIZE == 0, "A tile load must evenly divide across the threadblock");
static_assert(B_WORDS % THREADBLOCK_SIZE == 0, "B tile load must evenly divide across the threadblock");

extern "C" uint32_t __mu_num_warps = GEMM_SIMT_NUM_WARPS;

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
  uint32_t tid = tid_in_threadblock;
  uint32_t total_blocks = args->M * args->N / BLOCK_SIZE;
  uint32_t blocks_per_cluster = total_blocks / MU_NUM_CLUSTERS;
  uint32_t block_N = args->N / BN;

  uint32_t M = args->M;
  uint32_t N = args->N;
  uint32_t K = args->K;
  __global uint32_t *A = args->A;
  __global uint32_t *B = args->B;
  __global uint32_t *C = args->C;
  __shared uint32_t *As = sdata;
  __shared uint32_t *Bs = sdata + BM * BK / 2;

  for (uint32_t c_block = 0; c_block < blocks_per_cluster; c_block++) {
    uint32_t block_idx = threadblock_id * blocks_per_cluster + c_block;
    uint32_t block_x_idx = block_idx / block_N;
    uint32_t block_y_idx = block_idx % block_N;

    // clear out accum
    _Float16 acc[TM * TN];
    #pragma unroll
    for (uint32_t i = 0; i < TM*TN; i++) acc[i] = 0;

    // stream across K
    for (uint32_t k_block = 0; k_block < K; k_block += BK) {
      //load A & B block to smem
      #pragma unroll
      for (uint32_t base = 0; base < A_FULL_ITERS; base += ILP_MEM) {
        uint32_t a_val[ILP_MEM];
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          uint32_t A_x = block_x_idx * BM + (elem_idx / (BK / 2)); // BK bf16 elements = BK/2 uint32_t elements
          uint32_t A_y = k_block / 2 + (elem_idx % (BK / 2));
          a_val[u] = A[A_x * (K / 2) + A_y];
        }
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          As[elem_idx] = a_val[u];
        }
      }
      #if (A_ITERS % ILP_MEM) != 0
      #pragma unroll
      for (uint32_t block = A_FULL_ITERS; block < A_ITERS; block++) {
        uint32_t elem_idx = tid + block * THREADBLOCK_SIZE;
        uint32_t A_x = block_x_idx * BM + (elem_idx / (BK / 2)); // BK bf16 elements = BK/2 uint32_t elements
        uint32_t A_y = k_block / 2 + (elem_idx % (BK / 2));
        As[elem_idx] = A[A_x * (K / 2) + A_y];
      }
      #endif
      #pragma unroll
      for (uint32_t base = 0; base < B_FULL_ITERS; base += ILP_MEM) {
        uint32_t b_val[ILP_MEM];
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          uint32_t B_y = block_y_idx * BN / 2 + (elem_idx % (BN / 2)); // BN bf16 elements
          uint32_t B_x = k_block + (elem_idx / (BN / 2));
          b_val[u] = B[B_x * (N / 2) + B_y];
        }
        #pragma unroll
        for (uint32_t u = 0; u < ILP_MEM; u++) {
          uint32_t elem_idx = tid + (base + u) * THREADBLOCK_SIZE;
          Bs[elem_idx] = b_val[u];
        }
      }
      #if (B_ITERS % ILP_MEM) != 0
      #pragma unroll
      for (uint32_t block = B_FULL_ITERS; block < B_ITERS; block++) {
        uint32_t elem_idx = tid + block * THREADBLOCK_SIZE;
        uint32_t B_y = block_y_idx * BN / 2 + (elem_idx % (BN / 2)); // BN bf16 elements
        uint32_t B_x = k_block + (elem_idx / (BN / 2));
        Bs[elem_idx] = B[B_x * (N / 2) + B_y];
      }
      #endif

      // hold up
      mu_fence_smem();
      mu_barrier(0, BLOCK_NUM_WARPS);

      // compute
      // j and k can vector load 2 BF16
      uint32_t thread_x = tid / TBN;
      uint32_t thread_y = tid % TBN;
      for (uint32_t k = 0; k < BK / 2; k++) {
        for (uint32_t i = 0; i < TM; i++) {
          uint32_t a_idx = thread_x * TM + i;
          auto [a0, a1] = unpack_bf16x2(As[a_idx * BK / 2 + k]);
          for (uint32_t j = 0; j < TN / 2; j++) {
            uint32_t b_idx = thread_y * TN / 2 + j;
            auto [b00, b10] = unpack_bf16x2(Bs[2*k * (BN / 2) + b_idx]);
            auto [b01, b11] = unpack_bf16x2(Bs[(2*k + 1) * (BN / 2) + b_idx]);
            acc[i * TN + 2 * j] += a0 * b00 + a1 * b01;
            acc[i * TN + 2 * j + 1] += a0 * b10 + a1 * b11;
          }
        }
      }

      mu_barrier(0, BLOCK_NUM_WARPS);
    }

    // store C
    uint32_t thread_x = tid / TBN;
    uint32_t thread_y = tid % TBN;
    uint32_t c_row = block_x_idx * BM + thread_x * TM;
    uint32_t c_col = block_y_idx * BN / 2 + thread_y * TN / 2;
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
  mu_schedule(gemm, &gemm_args, GEMM_SIMT_NUM_WARPS);
  return 0;
}
