#include <shared_mem.h>
#include <stdint.h>
#include <vx_intrinsics.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 256;
constexpr uint32_t kWarmupIterations = 32;
}

/*
  We estimate shared-bank service latency under full same-bank contention
    - Method: all active lanes repeatedly load shared_words[0]
    - Extract: base_latency = cycles / ops 
*/

int main() {
  const uint32_t lanes = static_cast<uint32_t>(vx_num_threads());
  const uint32_t mask = (lanes >= 32) ? 0xFFFF'FFFFu : ((1u << lanes) - 1u);

  // Isolate SMEM bank service latency under maximum same-bank conflict
  vx_tmc(mask);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);
  volatile uint32_t* shared_words =
      reinterpret_cast<volatile uint32_t*>(DEV_SMEM_START_ADDR);
  vx_smem_store_u32(shared_words, 0x2468ACE1u);

  uint32_t acc = static_cast<uint32_t>(vx_thread_id()) + 1u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    acc ^= vx_smem_load_u32(shared_words);
  }

  const uint32_t start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    // All lanes load shared_words[0], forcing one bank hot spot
    acc ^= vx_smem_load_u32(shared_words);
  }
  const uint32_t end = tune_read_cycle();

  const uint32_t cycles = end - start;
  const uint32_t ops = iterations * (lanes == 0 ? 1u : lanes);
  const uint32_t base_latency = (ops == 0) ? 0 : (cycles / ops);

  TuneResult result{};
  result.base_latency = base_latency;
  result.bytes_per_cycle = 0;
  result.queue_capacity = 0;
  result.completions_per_cycle = 0;
  result.cycles = cycles;
  result.ops = ops;

  volatile uint32_t* out = tune::out_u32(arg);
  out[6] = acc;
  tune_store_result(out, result);
  tune_print_result("smem_bank_latency", result);
  return 0;
}
