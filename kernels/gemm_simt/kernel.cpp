#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#define NUM_WARPS 4

extern "C" uint32_t __mu_num_warps = NUM_WARPS;

struct GEMMArgs {
  __global uint16_t* A;
  __global uint16_t* B;
  __global uint16_t* C;
  uint32_t M;
  uint32_t K;
  uint32_t N;
};

// C = A * B where A is MxK, B is KxN, C is MxN (all bf16)
void gemm(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<GEMMArgs*>(arg);
  uint32_t lane_id = tid_in_threadblock % 16;
  uint32_t warp_id = tid_in_threadblock / 16;
  uint32_t tid = tid_in_threadblock;
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
  gemm_args.A = reinterpret_cast<__global uint16_t*>(A_raw);
  gemm_args.B = reinterpret_cast<__global uint16_t*>(B_raw);
  gemm_args.C = reinterpret_cast<__global uint16_t*>(C_raw);
  gemm_args.M = M_val;
  gemm_args.K = K_val;
  gemm_args.N = N_val;
  mu_schedule(gemm, &gemm_args, NUM_WARPS);
  return 0;
}
