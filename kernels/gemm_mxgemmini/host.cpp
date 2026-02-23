#include <inttypes.h>
#include <stdio.h>
#include <radiance.h>

#include "include/matmul_data.h"

void load_scale_factors(volatile uint64_t *sf_mem, const uint8_t *scale_factors,
                        int n) {
  uint64_t *dword_scale_factors = (uint64_t *)scale_factors;
  for (size_t i = 0; i < n / 8; i++) {
    sf_mem[i] = dword_scale_factors[i];
  }
}

int main() {
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
  // tohost = 0;
  *tocpu = tohost;

  load_scale_factors((volatile uint64_t *)0x4008a000, &A_scales_row[0][0], 32);
  load_scale_factors((volatile uint64_t *)0x40088000, &B_scales_col[0][0], 32);

  printf("start GPU\n");
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

  uint32_t finished = 0;
  while (!finished) {
    SYNC_GPU();
    finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
    // uint32_t core0 = READ_MMIO_32(RAD_HOST_GPU_CORES);
    // uint32_t core1 = READ_MMIO_32(RAD_HOST_GPU_CORES + 4);
    // uint32_t core2 = READ_MMIO_32(RAD_HOST_GPU_CORES + 8);
    // uint32_t core3 = READ_MMIO_32(RAD_HOST_GPU_CORES + 12);
    // printf("%d %d %d %d\n", core0, core1, core2, core3);
  }
  printf("finished\n");

  return 0;
}
