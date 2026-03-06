#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

struct SoftmaxArgs {
  __global _Float16* A;
  __global _Float16* x;
  uint32_t m;
  uint32_t n;
};

__shared _Float16* const sdata = reinterpret_cast<__shared _Float16*>(0x0);

// compute Ax where A is MxN and x is Nx1
void gemv(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<SoftmaxArgs*>(arg);
  uint32_t lane_id = tid_in_threadblock % 16;
  uint32_t warp_id = tid_in_threadblock / 16;
  uint32_t tid = tid_in_threadblock;
}

SoftmaxArgs gemv_args = {
  .A = nullptr,
  .x = nullptr,
  .m = 0,
  .n = 0,
};

#include "data"

int main() {
  mu_schedule(gemv, &gemv_args);
  return 0;
}
