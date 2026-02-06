#include <stdint.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 1024;
constexpr uint32_t kWarmupIterations = 32;
constexpr uint32_t kFootprintWords = 32768;
constexpr uint32_t kStrideWords = 8;
constexpr uint32_t kLoadsPerIter = 8;
}

int main() {
  vx_tmc(1);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);
  volatile uint32_t* src = memstress::src_ptr(arg);
  const uint32_t mask = kFootprintWords - 1u;

  uint32_t acc = 0x89ABCDEFu;
  uint32_t p0 = 0u, p1 = 512u, p2 = 1024u, p3 = 1536u;
  uint32_t p4 = 2048u, p5 = 2560u, p6 = 3072u, p7 = 3584u;

  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    acc ^= src[p0]; p0 = (p0 + kStrideWords) & mask;
    acc ^= src[p1]; p1 = (p1 + kStrideWords) & mask;
    acc ^= src[p2]; p2 = (p2 + kStrideWords) & mask;
    acc ^= src[p3]; p3 = (p3 + kStrideWords) & mask;
    acc ^= src[p4]; p4 = (p4 + kStrideWords) & mask;
    acc ^= src[p5]; p5 = (p5 + kStrideWords) & mask;
    acc ^= src[p6]; p6 = (p6 + kStrideWords) & mask;
    acc ^= src[p7]; p7 = (p7 + kStrideWords) & mask;
  }

  const uint32_t start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    acc ^= src[p0]; p0 = (p0 + kStrideWords) & mask;
    acc ^= src[p1]; p1 = (p1 + kStrideWords) & mask;
    acc ^= src[p2]; p2 = (p2 + kStrideWords) & mask;
    acc ^= src[p3]; p3 = (p3 + kStrideWords) & mask;
    acc ^= src[p4]; p4 = (p4 + kStrideWords) & mask;
    acc ^= src[p5]; p5 = (p5 + kStrideWords) & mask;
    acc ^= src[p6]; p6 = (p6 + kStrideWords) & mask;
    acc ^= src[p7]; p7 = (p7 + kStrideWords) & mask;
  }
  const uint32_t end = tune_read_cycle();
  const uint32_t cycles = end - start;
  const uint32_t ops = iterations * kLoadsPerIter;
  const uint32_t completions_per_cycle = (cycles == 0) ? 0 : (ops / cycles);

  TuneResult result{};
  result.base_latency = 0;
  result.bytes_per_cycle = 0;
  result.queue_capacity = 0;
  result.completions_per_cycle = completions_per_cycle;
  result.cycles = cycles;
  result.ops = ops;

  volatile uint32_t* out = tune::out_u32(arg);
  out[6] = acc;
  tune_store_result(out, result);
  tune_print_result("gmem_completions_per_cycle", result);
  return 0;
}
