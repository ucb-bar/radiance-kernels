#include <stdint.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 512;
constexpr uint32_t kWarmupIterations = 32;
}

int main() {
  vx_tmc(1);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);

  // Isolate FP dependency latency with a dependent FMA-like chain
  float acc = 1.25f;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    acc = acc * 1.0009765625f + 0.125f;
  }

  const uint32_t start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    // Each FP op depends on the prior FP result
    acc = acc * 1.0009765625f + 0.125f;
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
  out[6] = static_cast<uint32_t>(acc);
  tune_store_result(out, result);
  tune_print_result("execute_fp_latency", result);
  return 0;
}
