#include <stdint.h>

#include "../../common/tune_args.h"
#include "../../common/tune_report.h"
#include "../../common/tune_timer.h"

namespace {
constexpr uint32_t kDefaultIterations = 2048;
constexpr uint32_t kWarmupIterations = 128;
constexpr uint32_t kNumBlocks = 16;
}

__attribute__((noinline)) uint32_t ic_blk0(uint32_t x) { return (x ^ 0x9e3779b9u) + 0x11u; }
__attribute__((noinline)) uint32_t ic_blk1(uint32_t x) { return (x + 0x7f4a7c15u) ^ 0x21u; }
__attribute__((noinline)) uint32_t ic_blk2(uint32_t x) { return (x << 1) ^ 0x31u; }
__attribute__((noinline)) uint32_t ic_blk3(uint32_t x) { return (x >> 1) + 0x41u; }
__attribute__((noinline)) uint32_t ic_blk4(uint32_t x) { return (x ^ (x << 5)) + 0x51u; }
__attribute__((noinline)) uint32_t ic_blk5(uint32_t x) { return (x + (x >> 2)) ^ 0x61u; }
__attribute__((noinline)) uint32_t ic_blk6(uint32_t x) { return (x * 3u) + 0x71u; }
__attribute__((noinline)) uint32_t ic_blk7(uint32_t x) { return (x * 5u) ^ 0x81u; }
__attribute__((noinline)) uint32_t ic_blk8(uint32_t x) { return (x + 0x1234u) ^ 0x91u; }
__attribute__((noinline)) uint32_t ic_blk9(uint32_t x) { return (x ^ 0x8765u) + 0xA1u; }
__attribute__((noinline)) uint32_t ic_blk10(uint32_t x) { return (x << 2) + 0xB1u; }
__attribute__((noinline)) uint32_t ic_blk11(uint32_t x) { return (x >> 2) ^ 0xC1u; }
__attribute__((noinline)) uint32_t ic_blk12(uint32_t x) { return (x + (x << 3)) + 0xD1u; }
__attribute__((noinline)) uint32_t ic_blk13(uint32_t x) { return (x ^ (x >> 3)) ^ 0xE1u; }
__attribute__((noinline)) uint32_t ic_blk14(uint32_t x) { return (x * 9u) + 0xF1u; }
__attribute__((noinline)) uint32_t ic_blk15(uint32_t x) { return (x * 7u) ^ 0x101u; }

int main() {
  vx_tmc(1);
  kernel_arg_t* arg = tune::args();
  const uint32_t iterations = tune::iterations(arg, kDefaultIterations);
  uint32_t (*const blocks[kNumBlocks])(uint32_t) = {
      ic_blk0,  ic_blk1,  ic_blk2,  ic_blk3,
      ic_blk4,  ic_blk5,  ic_blk6,  ic_blk7,
      ic_blk8,  ic_blk9,  ic_blk10, ic_blk11,
      ic_blk12, ic_blk13, ic_blk14, ic_blk15};

  uint32_t hit_acc = 0x2468ACEu;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    hit_acc = ic_blk0(hit_acc + i);
  }
  const uint32_t hit_start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    hit_acc = ic_blk0(hit_acc + i);
  }
  const uint32_t hit_end = tune_read_cycle();
  const uint32_t hit_cycles = hit_end - hit_start;
  const uint32_t hit_cpi = (iterations == 0) ? 0 : (hit_cycles / iterations);

  uint32_t miss_acc = 0x13579BDu;
  uint32_t selector = 0u;
  for (uint32_t i = 0; i < kWarmupIterations; ++i) {
    selector = (selector * 5u + 3u) & (kNumBlocks - 1u);
    miss_acc = blocks[selector](miss_acc + i);
  }
  const uint32_t miss_start = tune_read_cycle();
  for (uint32_t i = 0; i < iterations; ++i) {
    // randomized block selection to force frequent I-cache line changes
    selector = (selector * 5u + 3u) & (kNumBlocks - 1u);
    miss_acc = blocks[selector](miss_acc + i);
  }
  const uint32_t miss_end = tune_read_cycle();
  const uint32_t miss_cycles = miss_end - miss_start;
  const uint32_t miss_cpi = (iterations == 0) ? 0 : (miss_cycles / iterations);
  const uint32_t miss_penalty = (miss_cpi > hit_cpi) ? (miss_cpi - hit_cpi) : 0u;

  TuneResult result{};
  result.base_latency = miss_penalty;
  result.bytes_per_cycle = 0;
  result.queue_capacity = 0;
  result.completions_per_cycle = 0;
  result.cycles = miss_cycles;
  result.ops = iterations;

  volatile uint32_t* out = tune::out_u32(arg);
  out[6] = hit_acc ^ miss_acc;
  tune_store_result(out, result);
  tune_print_result("icache_miss_latency", result);
  return 0;
}
