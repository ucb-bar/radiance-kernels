#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#define NUM_WARPS 4
#define BLOCK_SIZE MU_BLOCK_SIZE(NUM_WARPS)

struct SWIGLUArgs {
  __global _Float16* A;
  __global _Float16* B;
  __global _Float16* C;
  uint32_t m;
  uint32_t n;
};

static inline _Float16 sigmoid(_Float16 x) {
    return 1 / (1 + mu_fnexp(x));
}

// C[i,j] = swish(A[i,j]) * B[i,j] where swish(x) = x * sigmoid(x)
// A, B, C are all MxN, rows split across clusters
void swiglu(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<SWIGLUArgs*>(arg);
  uint32_t tid = tid_in_threadblock;

  uint32_t row_elems = args->n;
  uint32_t rows_per_block = args->m / MU_NUM_CLUSTERS;

  #pragma unroll 4
  for (uint32_t row = 0; row < rows_per_block; row++) {
    uint32_t block_elem_idx = (threadblock_id * rows_per_block + row) * row_elems;
    uint32_t chunks_per_block = (row_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __global _Float16 *A = args->A + block_elem_idx;
    __global _Float16 *B = args->B + block_elem_idx;
    __global _Float16 *C = args->C + block_elem_idx;

    #pragma unroll 4
    for (uint32_t chunk = 0; chunk < chunks_per_block; chunk++) {
        uint32_t idx = chunk * BLOCK_SIZE + tid;
        if (idx >= row_elems) break;
        C[idx] = A[idx] * sigmoid(A[idx]) * B[idx];
    }
  }
}

SWIGLUArgs swiglu_args = {
  .A = nullptr,
  .B = nullptr,
  .C = nullptr,
  .m = 0,
  .n = 0,
};

#include "data"

int main() {
  swiglu_args.A = reinterpret_cast<__global _Float16*>(A_raw);
  swiglu_args.B = reinterpret_cast<__global _Float16*>(B_raw);
  swiglu_args.C = reinterpret_cast<__global _Float16*>(C_raw);
  swiglu_args.m = m;
  swiglu_args.n = n;
  mu_schedule(swiglu, &swiglu_args, NUM_WARPS);
  return 0;
}
