#include <shared_mem.h>
#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 64;
constexpr uint32_t kSharedSize = 256;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t num_threads = memstress::resolve_threads(arg);
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  const uint32_t iterations =
      memstress::resolve_iterations(arg, kDefaultIterations);
  const uint32_t lane = static_cast<uint32_t>(vx_thread_id());
  const uint32_t warp = static_cast<uint32_t>(vx_warp_id());
  const uint32_t idx = ((warp << 4) + lane) % kSharedSize;
  volatile uint32_t* shared_data =
      reinterpret_cast<volatile uint32_t*>(DEV_SMEM_START_ADDR);

  uint32_t acc = lane + 11;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    vx_smem_store_u32(shared_data + idx, acc + iter);
  }
  // final read ensures stores remain data-dependent
  acc ^= vx_smem_load_u32(shared_data + idx);
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = memstress::resolve_threads(arg);
  vx_spawn_tasks_cluster(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
