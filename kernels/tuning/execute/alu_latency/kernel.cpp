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

  // Isolate ALU dependency latency with a single dependent chain
  uint32_t acc = 0x12345678u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    acc = (acc + i) ^ (acc >> 1);
  }

  const uint32_t start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    // every iteration depends on prior acc value
    acc = (acc + i) ^ (acc >> 1);
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
  tune_print_result("execute_alu_latency", result);
  return 0;
}
