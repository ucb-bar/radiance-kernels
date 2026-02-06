#include <stdint.h>
#include <vx_intrinsics.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 128;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t iterations =
      memstress::resolve_iterations(arg, kDefaultIterations);

  uint32_t acc = 0u;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    vx_tmc_one();
    acc ^= iter;
  }

  volatile uint32_t* dst = memstress::dst_ptr(arg);
  dst[0] = acc;
  return 0;
}
