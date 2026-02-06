#include <stdint.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 256;
constexpr uint32_t kWarmupIterations = 16;
constexpr uint32_t kProbeMaxDepth = 32;
constexpr uint32_t kFootprintWords = 65536;
constexpr uint32_t kStrideWords = 64;
}

int main() {
  vx_tmc(1);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);
  volatile uint32_t* src = memstress::src_ptr(arg);
  const uint32_t mask = kFootprintWords - 1u;

  uint32_t acc = 0x31415926u;
  uint32_t idx = 0u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    idx = (idx + kStrideWords) & mask;
    acc ^= src[idx];
  }

  uint32_t baseline_cycles = 0u;
  {
    const uint32_t start = tune_read_cycle();
    for (uint32_t i = 0; i < iterations; ++i) {
      const uint32_t a0 = (i * 131u) & mask;
      acc ^= src[a0];
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
  uint32_t vals[kProbeMaxDepth];

  for (uint32_t depth = 1; depth <= kProbeMaxDepth; ++depth) {
    const uint32_t start = tune_read_cycle();
    for (uint32_t i = 0; i < iterations; ++i) {
      const uint32_t base = (i * 131u) & mask;
      for (uint32_t j = 0; j < depth; ++j) {
        const uint32_t addr = (base + (j * kStrideWords)) & mask;
        vals[j] = src[addr];
      }
      for (uint32_t j = 0; j < depth; ++j) {
        acc ^= vals[j];
      }
    }
    const uint32_t end = tune_read_cycle();
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
  tune_print_result("gmem_mshr_capacity", result);
  return 0;
}
