#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kWarmIndex = 0;
constexpr uint32_t kMeasureIndex = 0;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  (void)arg;
  vx_tmc(1);
  if (task_id != 0) {
    return;
  }

  volatile uint32_t* src = reinterpret_cast<volatile uint32_t*>(0x90000000UL);

  uint32_t acc = 0;
  // warmup: populate L2 with the target cache line
  acc ^= src[kWarmIndex];

  // flush L0 (fence.i) and L1 (fence) so the next access hits L2 only
  asm volatile("fence.i");
  asm volatile("fence iorw, iorw");

  // measurement: L0 miss, L1 miss, L2 hit expected
  acc ^= src[kMeasureIndex];

  volatile uint32_t sink = acc;
  (void)sink;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  (void)arg;
  const uint32_t grid = 1;
  vx_spawn_tasks_cluster(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
