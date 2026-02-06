#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>

#include "../../memstress_common/helpers.h"

namespace {
constexpr uint32_t kDefaultIterations = 256;
}

static inline uint32_t path0(uint32_t v) { return v * 3u + 1u; }
static inline uint32_t path1(uint32_t v) { return v ^ 0x55aa55aau; }
static inline uint32_t path2(uint32_t v) { return (v << 1) | (v >> 31); }
static inline uint32_t path3(uint32_t v) { return v + 0x1234u; }
static inline uint32_t path4(uint32_t v) { return v * 7u; }
static inline uint32_t path5(uint32_t v) { return v ^ (v >> 3); }
static inline uint32_t path6(uint32_t v) { return v + (v << 2); }
static inline uint32_t path7(uint32_t v) { return v ^ 0xdeadbeefu; }

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t num_threads = memstress::resolve_threads(arg);
  if (static_cast<uint32_t>(task_id) >= num_threads) {
    return;
  }

  const uint32_t iterations =
      memstress::resolve_iterations(arg, kDefaultIterations);
  uint32_t acc = static_cast<uint32_t>(task_id);
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    switch ((acc + iter) & 7u) {
      case 0: acc = path0(acc); break;
      case 1: acc = path1(acc); break;
      case 2: acc = path2(acc); break;
      case 3: acc = path3(acc); break;
      case 4: acc = path4(acc); break;
      case 5: acc = path5(acc); break;
      case 6: acc = path6(acc); break;
      default: acc = path7(acc); break;
    }
  }

  volatile uint32_t* dst = memstress::dst_ptr(arg);
  dst[task_id] = acc;
}

int main() {
  auto* arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
  const uint32_t grid = memstress::resolve_threads(arg);
  vx_spawn_tasks_cluster(grid, (vx_spawn_tasks_cb)kernel_body, arg);
  return 0;
}
