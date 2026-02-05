#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 1;
constexpr uint32_t kDefaultStride = 1;
constexpr uint32_t kLineBytes = 32;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t num_threads = memstress::resolve_threads(arg);
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  const uint32_t iterations = memstress::resolve_iterations(arg, kDefaultIterations);
  const uint32_t stride = memstress::resolve_stride(arg, kDefaultStride);
  volatile uint32_t* src = memstress::src_ptr(arg);
  volatile uint32_t* dst = memstress::dst_ptr(arg);

  const uint32_t lane = static_cast<uint32_t>(vx_thread_id());
  const uint32_t warp = static_cast<uint32_t>(vx_warp_id());
  const uint32_t core = static_cast<uint32_t>(vx_core_id());
  const uint32_t line_elems = kLineBytes / sizeof(uint32_t);
  const uint32_t base = (core * line_elems * 2u) + (warp * line_elems);

  uint32_t acc = 0;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const size_t idx = static_cast<size_t>(base) + lane * stride;
    acc ^= src[idx];
  }
  dst[task_id] = acc;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = memstress::resolve_threads(arg);
  vx_spawn_tasks_cluster(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
