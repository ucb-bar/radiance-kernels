#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultStride = 32;  // 128B stride at 4B/elem
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t num_threads = memstress::resolve_threads(arg);
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  const uint32_t stride = memstress::resolve_stride(arg, kDefaultStride);
  volatile uint32_t* src = memstress::src_ptr(arg);

  const size_t idx = static_cast<size_t>(task_id) * stride;
  uint32_t acc = 0;
  acc ^= src[idx];

  volatile uint32_t sink = acc;
  (void)sink;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = memstress::resolve_threads(arg);
  vx_spawn_tasks(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
