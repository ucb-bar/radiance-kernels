#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <stdint.h>

#define VECADD_NUM_WARPS 4

struct VecAddArgs {
  __global float* A;
  __global float* B;
  __global float* C;
  uint32_t n;
};

static inline void vecadd(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<VecAddArgs*>(arg);
  (void)threadblock_id;

  for (uint32_t i = tid_in_threadblock; i < args->n; i += threads_per_threadblock) {
    args->C[i] = args->A[i] + args->B[i];
  }
}

VecAddArgs vecadd_args = {
  .A = nullptr,
  .B = nullptr,
  .C = nullptr,
  .n = 0,
};

#include "data"

int main() {
  vecadd_args.A = reinterpret_cast<__global float*>(A_raw);
  vecadd_args.B = reinterpret_cast<__global float*>(B_raw);
  vecadd_args.C = reinterpret_cast<__global float*>(C_raw);
  vecadd_args.n = n;
  mu_schedule(vecadd, &vecadd_args, VECADD_NUM_WARPS);
  return 0;
}
