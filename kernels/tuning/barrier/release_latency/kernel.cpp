#include <stdint.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 256;
constexpr uint32_t kWarmupIterations = 16;
}

int main() {
  const uint32_t lanes = static_cast<uint32_t>(vx_num_threads());
  const uint32_t active_lanes = (lanes == 0) ? 1u : lanes;
  const uint32_t mask =
      (active_lanes >= 32) ? 0xFFFF'FFFFu : ((1u << active_lanes) - 1u);

  // Keep all lanes active in every warp to avoid divergent barrier behavior
  vx_tmc(mask);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);
  uint32_t warps = static_cast<uint32_t>(vx_num_warps());
  if (warps == 0) {
    warps = 1;
  }

  uint32_t acc = static_cast<uint32_t>(vx_thread_id()) ^
                 (static_cast<uint32_t>(vx_warp_id()) << 8);
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    vx_barrier(0, warps);
    acc += i + 1u;
  }

  const uint32_t start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    // Measure steady-state barrier release timing across all participating warps
    vx_barrier(0, warps);
    acc ^= (i + 0x9e3779b9u);
  }
  const uint32_t end = tune_read_cycle();

  const uint32_t cycles = end - start;
  const uint32_t base_latency = (iterations == 0) ? 0 : (cycles / iterations);

  TuneResult result{};
  result.base_latency = base_latency;
  result.bytes_per_cycle = 0;
  result.queue_capacity = 0;
  result.completions_per_cycle = 0;
  result.cycles = cycles;
  result.ops = iterations;

  volatile uint32_t* out = tune::out_u32(arg);
  out[6] = acc;
  tune_store_result(out, result);
  tune_print_result("barrier_release_latency", result);
  return 0;
}
