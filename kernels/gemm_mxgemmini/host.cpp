#include <inttypes.h>
#include <stdio.h>
#include <radiance.h>

int main() {
#if 1
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
  // tohost = 0;
  *tocpu = tohost;

  // printf("start GPU\n");
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
  // printf("finished\n");
#endif

  return 0;
}
