#include <radiance.h>

static void copy_section(uint32_t offset, uint32_t size_bytes) {
  volatile uint64_t *src = (volatile uint64_t *)(DRAM_KERNEL_BASE + offset);
  volatile uint64_t *dst = (volatile uint64_t *)(POST_OR_BASE + offset);
  for (uint32_t i = 0; i < (size_bytes + 7) / 8; i++) {
    dst[i] = src[i];
  }
}

int main() {
  copy_section(0x0000, 0x68);
  copy_section(0x1000, 0x48);
  copy_section(0x2000, 0x108);
  copy_section(0x3000, 0x08);

  asm volatile ("fence" ::: "memory");
  // 3. Switch GPU address base from DRAM to scratchpad
  *GPU_ADDR_OR_MMIO = SPAD_BASE;

  // 4. Deassert GPU reset
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

  // 5. Wait for GPU to finish
  uint32_t finished = 0;
  while (!finished) {
    SYNC_GPU();
    finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
  }

  return 0;
}
