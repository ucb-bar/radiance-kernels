#ifndef RADIANCE_KERNELS_TUNING_TUNE_REPORT_H_
#define RADIANCE_KERNELS_TUNING_TUNE_REPORT_H_

#include <stdint.h>
// #include <stdio.h>

struct TuneResult {
  uint32_t base_latency;
  uint32_t bytes_per_cycle;
  uint32_t queue_capacity;
  uint32_t completions_per_cycle;
  uint32_t cycles;
  uint32_t ops;
};

inline void tune_store_result(volatile uint32_t* dst, const TuneResult& r) {
  if (!dst) return;
  dst[0] = r.base_latency;
  dst[1] = r.bytes_per_cycle;
  dst[2] = r.queue_capacity;
  dst[3] = r.completions_per_cycle;
  dst[4] = r.cycles;
  dst[5] = r.ops;
}

inline void tune_print_result(const char* name, const TuneResult& r) {
  (void)name;
  (void)r;
  // printf(
  //     "TUNE_RESULT test=%s base_latency=%u bytes_per_cycle=%u "
  //     "queue_capacity=%u completions_per_cycle=%u cycles=%u ops=%u\n",
  //     name,
  //     r.base_latency,
  //     r.bytes_per_cycle,
  //     r.queue_capacity,
  //     r.completions_per_cycle,
  //     r.cycles,
  //     r.ops);
}

#endif
