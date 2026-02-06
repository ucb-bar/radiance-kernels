#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 64;
constexpr uint32_t kDefaultStride = 8;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t num_threads = memstress::resolve_threads(arg);
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  const uint32_t iterations =
      memstress::resolve_iterations(arg, kDefaultIterations);
  const uint32_t stride = memstress::resolve_stride(arg, kDefaultStride);
  volatile uint32_t* src = memstress::src_ptr(arg);
  volatile uint32_t* dst = memstress::dst_ptr(arg);

  const size_t thread_stride = static_cast<size_t>(stride) * num_threads;
  const size_t base = static_cast<size_t>(task_id) * stride;

  uint32_t acc = 0;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const size_t idx = base + static_cast<size_t>(iter) * thread_stride;
    acc += src[idx];
  }
  dst[task_id] = acc;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = memstress::resolve_threads(arg);
  vx_spawn_tasks_cluster(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
