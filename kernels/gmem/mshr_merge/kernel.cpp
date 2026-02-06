#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 1;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t num_threads = memstress::resolve_threads(arg);
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  volatile uint32_t* src = memstress::src_ptr(arg);
  volatile uint32_t* dst = memstress::dst_ptr(arg);

  uint32_t acc = 0;
  const uint32_t iterations = memstress::resolve_iterations(arg, kDefaultIterations);
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    acc ^= src[0];  // all threads hit the same line for merge
  }
  if (task_id == 0) {
    dst[0] = static_cast<uint32_t>(vx_num_warps());
    dst[1] = static_cast<uint32_t>(vx_num_threads());
  }
  if (task_id < 8) {
    dst[2 + task_id] = static_cast<uint32_t>(task_id);
  }
  volatile uint32_t sink = acc;
  (void)sink;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = memstress::resolve_threads(arg);
  vx_spawn_tasks(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
