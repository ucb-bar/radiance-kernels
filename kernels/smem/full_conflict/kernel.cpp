#include <shared_mem.h>
#include <stdint.h>
#include <vx_intrinsics.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 128;
}

int main() {
  const uint32_t num_lanes = static_cast<uint32_t>(vx_num_threads());
  const uint32_t full_mask =
      (num_lanes >= 32) ? 0xFFFF'FFFFu : ((1u << num_lanes) - 1u);
  vx_tmc(full_mask);

  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t iterations =
      memstress::resolve_iterations(arg, kDefaultIterations);
  const uint32_t lane = static_cast<uint32_t>(vx_thread_id());
  volatile uint32_t* shared_word =
      reinterpret_cast<volatile uint32_t*>(DEV_SMEM_START_ADDR);

  uint32_t acc = lane + 1u;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    // Every lane reads the same shared word each iteration -> full bank conflict
    acc ^= vx_smem_load_u32(shared_word);
  }

  asm volatile("" : "+r"(acc) :: "memory");
  return 0;
}
