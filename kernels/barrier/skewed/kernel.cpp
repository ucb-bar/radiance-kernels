#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 64;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t num_threads = memstress::resolve_threads(arg);
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  const uint32_t iterations =
      memstress::resolve_iterations(arg, kDefaultIterations);
  const uint32_t warp = static_cast<uint32_t>(vx_warp_id());
  uint32_t acc = static_cast<uint32_t>(task_id);

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    // Skew arrivals by warp so one participant consistently reaches barrier later
    const uint32_t spin_bound = warp * 64 + (iter & 0x7);
    for (uint32_t spin = 0; spin < spin_bound; ++spin) {
      acc += spin + iter;
    }
    vx_barrier(0, vx_num_warps());
  }

  volatile uint32_t* dst = memstress::dst_ptr(arg);
  dst[task_id] = acc;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = memstress::resolve_threads(arg);
  vx_spawn_tasks(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
