#include <radiance.h>

int main() {
#if 0
    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
    tohost = 0;
    *tocpu = tohost;

    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

    uint32_t finished = 0;
    while (!finished) {
        SYNC_GPU();
        finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
    }
#endif

    return 0;
}
