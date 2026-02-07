#include <shared_mem.h>
#include <stdint.h>
#include <vx_intrinsics.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 256;
constexpr uint32_t kBankCount = 8;
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
  const uint32_t warp = static_cast<uint32_t>(vx_warp_id());
  volatile uint32_t* shared_words =
      reinterpret_cast<volatile uint32_t*>(DEV_SMEM_START_ADDR);

  // Map both warps to the same bank set, but different subbanks:
  // warp 0 -> subbank 0 (idx 0..7), warp 1 -> subbank 1 (idx 8..15).
  const uint32_t idx = (lane % kBankCount) + (((warp & 1u) == 0u) ? 0u : kBankCount);

  uint32_t acc = lane + 1u;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    // Use warp-role split so load/store can be in flight concurrently:
    // warp 0 issues stores while warp 1 issues loads to the same banks.
    if ((warp & 1u) == 0u) {
      vx_smem_store_u32(shared_words + idx, acc + iter);
    } else {
      acc ^= vx_smem_load_u32(shared_words + idx);
    }
  }

  asm volatile("" : "+r"(acc) :: "memory");
  return 0;
}
