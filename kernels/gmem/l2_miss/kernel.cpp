#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kFirstIndex = 0;
constexpr uint32_t kSecondIndex = 32;  // 128 bytes apart
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  vx_tmc(1);
  if (task_id != 0) {
    return;
  }

  volatile uint32_t* src = memstress::src_ptr(arg);

  uint32_t acc = 0;
  acc ^= src[kFirstIndex];
  acc ^= src[kSecondIndex];

  volatile uint32_t sink = acc;
  (void)sink;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = 1;
  vx_spawn_tasks_cluster(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
