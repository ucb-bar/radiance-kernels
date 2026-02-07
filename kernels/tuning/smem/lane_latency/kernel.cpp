#include <shared_mem.h>
#include <stdint.h>
#include <vx_intrinsics.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 512;
constexpr uint32_t kWarmupIterations = 32;
}

int main() {
  // Isolate per-lane SMEM load latency with a single active lane
  vx_tmc_one();
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);
  volatile uint32_t* shared_words =
      reinterpret_cast<volatile uint32_t*>(DEV_SMEM_START_ADDR);

  vx_smem_store_u32(shared_words, 0x13579BDFu);

  uint32_t acc = 0u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    acc ^= vx_smem_load_u32(shared_words);
  }

  const uint32_t start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    // dependent accumulation from a fixed shared-memory location
    acc ^= vx_smem_load_u32(shared_words);
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
  tune_print_result("smem_lane_latency", result);
  return 0;
}
