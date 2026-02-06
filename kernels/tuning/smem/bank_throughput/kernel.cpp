#include <shared_mem.h>
#include <stdint.h>
#include <vx_intrinsics.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 1024;
constexpr uint32_t kWarmupIterations = 64;
constexpr uint32_t kBytesPerLoad = 4;
}

/*
  Estimate shared-memory throughput (bytes_per_cycle)
  - Method: all lanes repeatedly load own shared slot with minimal conflicts
  - Extract: bytes_per_cycle = (ops * 4) / cycles

*/
int main() {
  const uint32_t lanes = static_cast<uint32_t>(vx_num_threads());
  const uint32_t active_lanes = (lanes == 0) ? 1u : lanes;
  const uint32_t mask = (active_lanes >= 32) ? 0xFFFF'FFFFu : ((1u << active_lanes) - 1u);

  // Estimate SMEM bytes/cycle with conflict-minimized bank mapping
  vx_tmc(mask);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);
  volatile uint32_t* shared_words =
      reinterpret_cast<volatile uint32_t*>(vx_shared_ptr(0));

  const uint32_t lane = static_cast<uint32_t>(vx_thread_id());
  shared_words[lane] = lane + 1u;

  uint32_t acc = lane + 7u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    acc ^= shared_words[lane];
  }

  const uint32_t start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    // Each lane repeatedly reads its own slot to maximize throughput
    acc ^= shared_words[lane];
  }
  const uint32_t end = tune_read_cycle();

  const uint32_t cycles = end - start;
  const uint32_t ops = iterations * active_lanes;
  const uint32_t total_bytes = ops * kBytesPerLoad;
  const uint32_t bytes_per_cycle = (cycles == 0) ? 0 : (total_bytes / cycles);

  TuneResult result{};
  result.base_latency = 0;
  result.bytes_per_cycle = bytes_per_cycle;
  result.queue_capacity = 0;
  result.completions_per_cycle = 0;
  result.cycles = cycles;
  result.ops = ops;

  volatile uint32_t* out = tune::out_u32(arg);
  out[6] = acc;
  tune_store_result(out, result);
  tune_print_result("smem_bank_throughput", result);
  return 0;
}
