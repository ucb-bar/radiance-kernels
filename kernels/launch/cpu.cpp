#include <stdio.h>
#include <inttypes.h>

#define GPU_RESET 0x41000000ULL
#define GPU_ALL_FINISHED 0x41000008ULL
#define GPU_CORES 0x41000010ULL

#define READ_MMIO_32(addr) ({ \
    uint32_t result = (*(volatile uint32_t *) (addr)); \
    result; \
})

#define WRITE_MMIO_32(addr, data) \
    (*(volatile uint32_t *) (addr)) = (uint32_t) (data)

extern volatile uint64_t tohost;
volatile uint64_t *tocpu = (volatile uint64_t *) 0x100010000ULL;

// #define SYNC_GPU() ({ \
//     uint64_t fromgpu = *tocpu; \
//     *tocpu = tohost; \
//     tohost = fromgpu; \
// })

inline static void SYNC_GPU() {
    uint64_t old;

    // atomically: old = *tocpu; *tocpu = *tohost;
    __asm__ __volatile__(
        "amoswap.d.aqrl %0, %2, (%1)"
        : "=r"(old)
        : "r"(tocpu), "r"(tohost)
        : "memory"
    );

    tohost = old;
}

int main() {
    *tocpu = 0;
    printf("reset\n");
    WRITE_MMIO_32(GPU_RESET, 1);
    printf("wait\n");
    WRITE_MMIO_32(GPU_RESET, 0);

    uint32_t finished = 0;
    while (!finished) {
        SYNC_GPU();
        finished = READ_MMIO_32(GPU_ALL_FINISHED);
        uint32_t core0 = READ_MMIO_32(GPU_CORES);
        uint32_t core1 = READ_MMIO_32(GPU_CORES + 4);
        uint32_t core2 = READ_MMIO_32(GPU_CORES + 8);
        uint32_t core3 = READ_MMIO_32(GPU_CORES + 12);
        // printf("%d %d %d %d\n", core0, core1, core2, core3);
    }

    printf("done!\n");
    return 0;
}
