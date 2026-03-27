#include <radiance.h>

/* Data-region offsets (device address - DEVICE_KERNEL_BASE) */
#define A_ELF_OFF    0x1000u   /* A in ELF / DRAM                       */
#define B_ELF_OFF    0x3000u   /* B in ELF / DRAM                       */
#define A_DEV_OFF    0x1000u   /* GPU reads A from dev 0x10001000       */
#define B_DEV_OFF    0x5000u   /* GPU reads B from dev 0x10005000 (+16K)*/
#define VEC_SIZE     0x2000u   /* 8 KiB each                            */
#define PHASE_OFF    0x0F00u   /* phase variable in scratchpad page 0   */

/* --- helpers ------------------------------------------------------------ */

/* Copy from DRAM (fused ELF) to scratchpad, same src and dst offset */
static void copy_section(uint32_t offset, uint32_t size_bytes) {
    volatile uint64_t *src = (volatile uint64_t *)(DRAM_KERNEL_BASE + offset);
    volatile uint64_t *dst = (volatile uint64_t *)(POST_OR_BASE + offset);
    for (uint32_t i = 0; i < (size_bytes + 7) / 8; i++)
        dst[i] = src[i];
}

/* Copy from DRAM at src_offset to scratchpad at dst_offset (different) */
static void copy_data(uint32_t src_offset, uint32_t dst_offset, uint32_t size) {
    volatile uint64_t *src = (volatile uint64_t *)(DRAM_KERNEL_BASE + src_offset);
    volatile uint64_t *dst = (volatile uint64_t *)(POST_OR_BASE + dst_offset);
    for (uint32_t i = 0; i < (size + 7) / 8; i++)
        dst[i] = src[i];
}

static void write_phase(uint32_t phase) {
    *(volatile uint32_t *)(POST_OR_BASE + PHASE_OFF) = phase;
}

static void gpu_run(void) {
    asm volatile("fence" ::: "memory");
    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);          /* deassert reset */
    uint32_t done = 0;
    while (!done) {
        SYNC_GPU();
        done = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
    }
}

static void gpu_reset(void) {
    WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
    /* Wait for L0I cache flush to complete before next deassert */
    for (volatile int i = 100; i > 0; i--);
}

/* --- main --------------------------------------------------------------- */

int main() {
    /*
     * ELF LOAD segment sizes — update after building kernel.radiance.elf
     * with: readelf -lW kernel.radiance.elf
     */

    /* Phase 0 — stream A into L2 through scratchpad */
    copy_section(0x0000, 0x1000);       /* kernel code (first 4K page)     */
    copy_section(A_ELF_OFF, VEC_SIZE);  /* A data to scratchpad 0x1000     */
    write_phase(0);

    *GPU_ADDR_OR_MMIO = SPAD_BASE;
    gpu_run();
    gpu_reset();

    /* Phase 1 — stream B into L2 at aliased address
     *   B's ELF/DRAM location is B_ELF_OFF (0x3000), but we write it to
     *   POST_OR + B_DEV_OFF (0x5000) so the GPU reads a different full
     *   address that aliases to the same scratchpad region as A. */
    copy_data(B_ELF_OFF, B_DEV_OFF, VEC_SIZE);
    write_phase(1);
    gpu_run();
    gpu_reset();

    /* Phase 2 — compute C = A * B, both served from L2 */
    write_phase(2);
    gpu_run();

    return 0;
}
