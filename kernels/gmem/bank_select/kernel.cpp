#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 64;
constexpr uint32_t kDefaultBanks = 2;
constexpr uint32_t kLineBytes = 32;

// line addresses for L1 bank mapping with seed 0x1111222233334444 and banks=2
constexpr uint64_t kBankLineAddr[2] = {0, 1};
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t num_threads = memstress::resolve_threads(arg);
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  const uint32_t iterations = memstress::resolve_iterations(arg, kDefaultIterations);
  const uint32_t banks = memstress::resolve_smem_banks(arg, kDefaultBanks);
  const uint32_t target_bank = (banks == 0 ? 0u : (vx_warp_id() % banks));

  volatile uint32_t* src = memstress::src_ptr(arg);
  volatile uint32_t* dst = memstress::dst_ptr(arg);

  const uint64_t line_addr =
      (banks == 2) ? kBankLineAddr[target_bank] : static_cast<uint64_t>(task_id);

  const size_t elem_index =
      static_cast<size_t>(line_addr) * (kLineBytes / sizeof(uint32_t));

  uint32_t acc = 0;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    acc ^= src[elem_index];
    acc ^= src[elem_index + 1];
  }

  dst[task_id] = acc;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = memstress::resolve_threads(arg);
  vx_spawn_tasks_cluster(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
