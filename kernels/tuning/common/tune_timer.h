#ifndef RADIANCE_KERNELS_TUNING_TUNE_TIMER_H_
#define RADIANCE_KERNELS_TUNING_TUNE_TIMER_H_

#include <stdint.h>

inline uint32_t tune_read_cycle() {
  uint32_t x;
  asm volatile("csrr %0, mcycle" : "=r"(x));
  return x;
}

#endif
