#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 1;
constexpr uint32_t kWarmIndex = 0;
constexpr uint32_t kHitIndex = 0;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  // Force a single active lane to avoid extra memory ops from other lanes
  vx_tmc(1);
  // Force a single executing thread to ensure deterministic single-task behavior
  const uint32_t num_threads = 1;
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  // one measurement load
  const uint32_t iterations = 1;
  volatile uint32_t* src = reinterpret_cast<volatile uint32_t*>(0x90000000UL);

  uint32_t acc = 0;
  // perform a single scalar load to populate the L0 line before the measured dependent load
  uintptr_t base = reinterpret_cast<uintptr_t>(src);
  volatile uint32_t* warm_ptr = reinterpret_cast<volatile uint32_t*>(base + (uintptr_t)kWarmIndex * sizeof(uint32_t));
  uint32_t tmp = *warm_ptr;
  (void)tmp;

  // perform a single scalar load from the target word so we issue one measured GMEM access
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    volatile uint32_t* hit_ptr = reinterpret_cast<volatile uint32_t*>(base + (uintptr_t)kHitIndex * sizeof(uint32_t));
    acc ^= *hit_ptr;
  }
  volatile uint32_t sink = acc;
  (void)sink;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  // Force single task to ensure the microbenchmark issues exactly two GMEM loads
  const uint32_t grid = 1;
  vx_spawn_tasks_cluster(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
