#include <stdio.h>
#include <radiance.h>

int main(void) {
    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
    *tocpu = tohost;

    // populate kernel argument
    auto arg = reinterpret_cast<uint32_t *>(RAD_HOST_ARG_BASE);
    *arg = 42;

    // flush kernel argument writes
    asm volatile("fence rw, rw" ::: "memory");

    printf("start GPU\n");
    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

    uint32_t finished = 0;
    while (!finished) {
        SYNC_GPU();
        finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
    }
    printf("finished\n");

    return 0;
}
