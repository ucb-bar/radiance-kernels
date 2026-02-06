#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 16;
constexpr uint32_t kInnerBarriers = 4;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t num_threads = memstress::resolve_threads(arg);
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  const uint32_t iterations =
      memstress::resolve_iterations(arg, kDefaultIterations);
  uint32_t acc = static_cast<uint32_t>(task_id);

  // Nested barrier structure:
  // outer barrier id=0 wraps repeated inner barrier id=1 rendezvous.
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    vx_barrier(0, vx_num_warps());
    for (uint32_t inner = 0; inner < kInnerBarriers; ++inner) {
      vx_barrier(1, vx_num_warps());
      acc += (iter + inner);
      vx_barrier(1, vx_num_warps());
      acc ^= ((iter << 2) ^ inner);
    }
    vx_barrier(0, vx_num_warps());
  }

  volatile uint32_t* dst = memstress::dst_ptr(arg);
  dst[task_id] = acc;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = memstress::resolve_threads(arg);
  // Use per-core task spawn so 1-core microbench configs can still utilize multiple warps.
  vx_spawn_tasks(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
