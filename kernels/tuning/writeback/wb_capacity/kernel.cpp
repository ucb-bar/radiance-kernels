#include <stdint.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 512;
constexpr uint32_t kWarmupIterations = 32;
constexpr uint32_t kMaxProbeDepth = 32;
}

int main() {
  vx_tmc(1);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);

  // probe sustained number of independent produced results before retire pressure
  uint32_t acc = 0xA1B2C3D4u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    acc = (acc + i) ^ (acc >> 2);
  }

  uint32_t baseline_cycles = 0u;
  {
    const uint32_t start = tune_read_cycle();
    for (uint32_t i = 0; i < iterations; ++i) {
      uint32_t t = i + 1u;
      acc ^= (t * 3u) + 1u;
    }
    const uint32_t end = tune_read_cycle();
    baseline_cycles = end - start;
  }

  const uint32_t baseline_ops = (iterations == 0) ? 1u : iterations;
  const uint32_t baseline_cpi =
      (baseline_cycles == 0) ? 1u : (baseline_cycles / baseline_ops);
  const uint32_t threshold_cpi = baseline_cpi + (baseline_cpi >> 1);

  uint32_t best_depth = 1u;
  uint32_t best_cycles = baseline_cycles;
  uint32_t best_ops = baseline_ops;

  for (uint32_t depth = 1; depth <= kMaxProbeDepth; ++depth) {
    uint32_t regs[kMaxProbeDepth];
    for (uint32_t k = 0; k < depth; ++k) {
      regs[k] = (k + 1u) * 0x11111111u;
    }

    const uint32_t start = tune_read_cycle();
    for (uint32_t i = 0; i < iterations; ++i) {
      for (uint32_t k = 0; k < depth; ++k) {
        regs[k] = (regs[k] + i + (k * 7u)) ^ (regs[k] >> 1);
      }
    }
    const uint32_t end = tune_read_cycle();

    for (uint32_t k = 0; k < depth; ++k) {
      acc ^= regs[k];
    }

    const uint32_t cycles = end - start;
    const uint32_t ops = (iterations == 0) ? 1u : (iterations * depth);
    const uint32_t cpi = (cycles == 0) ? 1u : (cycles / ops);
    if (cpi <= threshold_cpi) {
      best_depth = depth;
      best_cycles = cycles;
      best_ops = ops;
    }
  }

  TuneResult result{};
  result.base_latency = 0;
  result.bytes_per_cycle = 0;
  result.queue_capacity = best_depth;
  result.completions_per_cycle = 0;
  result.cycles = best_cycles;
  result.ops = best_ops;

  volatile uint32_t* out = tune::out_u32(arg);
  out[6] = acc;
  tune_store_result(out, result);
  tune_print_result("writeback_wb_capacity", result);
  return 0;
}
