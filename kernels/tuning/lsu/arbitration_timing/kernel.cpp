#include <stdint.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 512;
constexpr uint32_t kWarmupIterations = 32;
constexpr uint32_t kWorkingSetMask = 1023;
}

int main() {
  vx_tmc(1);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);
  volatile uint32_t* src = memstress::src_ptr(arg);
  volatile uint32_t* dst = memstress::dst_ptr(arg);

  uint32_t acc = 0u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    const uint32_t idx = i & kWorkingSetMask;
    acc ^= src[idx];
    dst[idx] = acc + i;
  }
  vx_fence();

  uint32_t load_cycles = 0u;
  {
    const uint32_t start = tune_read_cycle();
    for (uint32_t i = 0; i < iterations; ++i) {
      acc ^= src[(i * 11u) & kWorkingSetMask];
    }
    const uint32_t end = tune_read_cycle();
    load_cycles = end - start;
  }

  uint32_t store_cycles = 0u;
  {
    const uint32_t start = tune_read_cycle();
    for (uint32_t i = 0; i < iterations; ++i) {
      dst[(i * 7u) & kWorkingSetMask] = acc + i;
    }
    vx_fence();
    const uint32_t end = tune_read_cycle();
    store_cycles = end - start;
  }

  uint32_t mixed_cycles = 0u;
  {
    const uint32_t start = tune_read_cycle();
    for (uint32_t i = 0; i < iterations; ++i) {
      const uint32_t lidx = (i * 11u) & kWorkingSetMask;
      const uint32_t sidx = (i * 7u) & kWorkingSetMask;
      acc ^= src[lidx];
      dst[sidx] = acc + i;
    }
    vx_fence();
    const uint32_t end = tune_read_cycle();
    mixed_cycles = end - start;
  }

  const uint32_t iter_nonzero = (iterations == 0) ? 1u : iterations;
  const uint32_t load_cpi = (load_cycles == 0) ? 0u : (load_cycles / iter_nonzero);
  const uint32_t store_cpi = (store_cycles == 0) ? 0u : (store_cycles / iter_nonzero);
  const uint32_t mixed_cpi = (mixed_cycles == 0) ? 0u : (mixed_cycles / iter_nonzero);
  const uint32_t baseline_cpi = (load_cpi > store_cpi) ? load_cpi : store_cpi;
  const uint32_t arbitration_penalty =
      (mixed_cpi > baseline_cpi) ? (mixed_cpi - baseline_cpi) : 0u;
  const uint32_t mixed_ops = iterations * 2u;
  const uint32_t completions_per_cycle =
      (mixed_cycles == 0) ? 0u : (mixed_ops / mixed_cycles);

  TuneResult result{};
  result.base_latency = arbitration_penalty;
  result.bytes_per_cycle = 0;
  result.queue_capacity = 0;
  result.completions_per_cycle = completions_per_cycle;
  result.cycles = mixed_cycles;
  result.ops = mixed_ops;

  volatile uint32_t* out = tune::out_u32(arg);
  out[6] = acc ^ load_cycles ^ store_cycles ^ mixed_cycles;
  tune_store_result(out, result);
  tune_print_result("lsu_arbitration_timing", result);
  return 0;
}
