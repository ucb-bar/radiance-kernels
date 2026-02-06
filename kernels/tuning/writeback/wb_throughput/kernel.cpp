#include <stdint.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 1024;
constexpr uint32_t kWarmupIterations = 64;
constexpr uint32_t kOpsPerIter = 8;
}

int main() {
  vx_tmc(1);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);

  // independent chains estimate writeback/retire completion width
  uint32_t a0 = 1u, a1 = 3u, a2 = 5u, a3 = 7u;
  uint32_t a4 = 9u, a5 = 11u, a6 = 13u, a7 = 15u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    a0 += 3u; a1 += 5u; a2 += 7u; a3 += 11u;
    a4 += 13u; a5 += 17u; a6 += 19u; a7 += 23u;
  }

  const uint32_t start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    a0 += 3u; a1 += 5u; a2 += 7u; a3 += 11u;
    a4 += 13u; a5 += 17u; a6 += 19u; a7 += 23u;
  }
  const uint32_t end = tune_read_cycle();

  const uint32_t cycles = end - start;
  const uint32_t ops = iterations * kOpsPerIter;
  const uint32_t completions_per_cycle = (cycles == 0) ? 0 : (ops / cycles);

  TuneResult result{};
  result.base_latency = 0;
  result.bytes_per_cycle = 0;
  result.queue_capacity = 0;
  result.completions_per_cycle = completions_per_cycle;
  result.cycles = cycles;
  result.ops = ops;

  volatile uint32_t* out = tune::out_u32(arg);
  out[6] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
  tune_store_result(out, result);
  tune_print_result("writeback_wb_throughput", result);
  return 0;
}
