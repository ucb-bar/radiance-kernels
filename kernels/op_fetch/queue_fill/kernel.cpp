#include <stdint.h>
#include <vx_intrinsics.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 128;
}

int main() {
  vx_tmc(1);
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t iterations =
      memstress::resolve_iterations(arg, kDefaultIterations);
  volatile uint32_t* src = memstress::src_ptr(arg);

  uint32_t acc = 0;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    acc ^= src[0];
  }
  asm volatile("" : "+r"(acc) :: "memory");
  return 0;
}
