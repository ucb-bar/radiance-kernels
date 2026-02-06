#include <stdint.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 512;
constexpr uint32_t kWarmupIterations = 32;
constexpr uint32_t kLatencyFootprintWords = 65536;   // 256KB
constexpr uint32_t kBandwidthFootprintWords = 65536; // 256KB
constexpr uint32_t kStrideWords = 64;
constexpr uint32_t kLoadsPerIter = 8;
constexpr uint32_t kBytesPerLoad = 4;
}

int main() {
  vx_tmc(1);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);
  volatile uint32_t* src = memstress::src_ptr(arg);

  uint32_t acc = 0xCAFEBABEu;
  uint32_t idx = 0u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    idx = (idx + kStrideWords + (acc & 31u)) & (kLatencyFootprintWords - 1u);
    acc ^= src[idx];
  }

  const uint32_t lat_start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    idx = (idx + kStrideWords + (acc & 31u)) & (kLatencyFootprintWords - 1u);
    acc ^= src[idx];
  }
  const uint32_t lat_end = tune_read_cycle();
  const uint32_t latency_cycles = lat_end - lat_start;
  const uint32_t base_latency = (iterations == 0) ? 0 : (latency_cycles / iterations);

  uint32_t p0 = 0u, p1 = 1024u, p2 = 2048u, p3 = 3072u;
  uint32_t p4 = 4096u, p5 = 5120u, p6 = 6144u, p7 = 7168u;
  const uint32_t bw_mask = kBandwidthFootprintWords - 1u;
  const uint32_t bw_start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    acc ^= src[p0]; p0 = (p0 + kStrideWords) & bw_mask;
    acc ^= src[p1]; p1 = (p1 + kStrideWords) & bw_mask;
    acc ^= src[p2]; p2 = (p2 + kStrideWords) & bw_mask;
    acc ^= src[p3]; p3 = (p3 + kStrideWords) & bw_mask;
    acc ^= src[p4]; p4 = (p4 + kStrideWords) & bw_mask;
    acc ^= src[p5]; p5 = (p5 + kStrideWords) & bw_mask;
    acc ^= src[p6]; p6 = (p6 + kStrideWords) & bw_mask;
    acc ^= src[p7]; p7 = (p7 + kStrideWords) & bw_mask;
  }
  const uint32_t bw_end = tune_read_cycle();
  const uint32_t bandwidth_cycles = bw_end - bw_start;
  const uint32_t bw_ops = iterations * kLoadsPerIter;
  const uint32_t total_bytes = bw_ops * kBytesPerLoad;
  const uint32_t bytes_per_cycle = (bandwidth_cycles == 0) ? 0 : (total_bytes / bandwidth_cycles);

  TuneResult result{};
  result.base_latency = base_latency;
  result.bytes_per_cycle = bytes_per_cycle;
  result.queue_capacity = 0;
  result.completions_per_cycle = 0;
  result.cycles = bandwidth_cycles;
  result.ops = bw_ops;

  volatile uint32_t* out = tune::out_u32(arg);
  out[6] = acc;
  tune_store_result(out, result);
  tune_print_result("gmem_dram_latency_bw", result);
  return 0;
}
