#include <inttypes.h>
#include <stdio.h>

#include "include/matmul_data.h"

#define GPU_RESET 0x41000000ULL
#define GPU_ALL_FINISHED 0x41000008ULL
#define GPU_CORES 0x41000010ULL

#define READ_MMIO_32(addr)                                                     \
  ({                                                                           \
    uint32_t result = (*(volatile uint32_t *)(addr));                          \
    result;                                                                    \
  })

#define WRITE_MMIO_32(addr, data)                                              \
  (*(volatile uint32_t *)(addr)) = (uint32_t)(data)

extern volatile uint64_t tohost;
volatile uint64_t *tocpu = (volatile uint64_t *)0x100010000ULL;

inline static void SYNC_GPU() {
  volatile uint64_t _fromcpu = *tocpu;
  if (_fromcpu > 0) {
    volatile uint64_t _fromhost = tohost;
    tohost = _fromcpu;
    *tocpu = _fromhost;
  }
}

void load_scale_factors(volatile uint64_t *sf_mem, const uint8_t *scale_factors,
                        int n) {
  uint64_t *dword_scale_factors = (uint64_t *)scale_factors;
  for (size_t i = 0; i < n / 8; i++) {
    sf_mem[i] = dword_scale_factors[i];
  }
}

int main() {
  WRITE_MMIO_32(GPU_RESET, 1);
  // tohost = 0;
  *tocpu = tohost;

  load_scale_factors((volatile uint64_t *)0x4008a000, &A_scales_row[0][0], 32);
  load_scale_factors((volatile uint64_t *)0x40088000, &B_scales_col[0][0], 32);

  printf("start GPU\n");
  WRITE_MMIO_32(GPU_RESET, 0);

  uint32_t finished = 0;
  while (!finished) {
    SYNC_GPU();
    finished = READ_MMIO_32(GPU_ALL_FINISHED);
    // uint32_t core0 = READ_MMIO_32(GPU_CORES);
    // uint32_t core1 = READ_MMIO_32(GPU_CORES + 4);
    // uint32_t core2 = READ_MMIO_32(GPU_CORES + 8);
    // uint32_t core3 = READ_MMIO_32(GPU_CORES + 12);
    // printf("%d %d %d %d\n", core0, core1, core2, core3);
  }
  printf("finished\n");

  return 0;
}
