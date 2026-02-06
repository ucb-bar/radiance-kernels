#include <stdint.h>
#include <vx_intrinsics.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 256;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t iterations =
      memstress::resolve_iterations(arg, kDefaultIterations);

  float acc = 1.25f;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    acc = acc * 1.0001f + 0.25f;
  }

  volatile uint32_t* dst = memstress::dst_ptr(arg);
  dst[0] = static_cast<uint32_t>(acc);
  return 0;
}
