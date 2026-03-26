#include <radiance.h>
// MMIOAddressOrNode register for cluster 0
// cluster baseAddr (0x4000_0000) + peripheralAddrOffset (0x80000) + 0x1000
#define GPU_ADDR_OR_MMIO ((volatile uint64_t *)0x40081000ull)

// OR-node base for scratchpad: gmemAddr + gmemSize = 0x1_8000_0000
#define SPAD_BASE 0x180000000ull

// Device kernel base address (GPU address space)
#define DEVICE_KERNEL_BASE 0x10000000ull

// Post-OR address: device_addr | SPAD_BASE
// This is the address the GPU will read after the OR node remaps it,
// so the host must write here for LLC coherence probes to work.
#define POST_OR_BASE (SPAD_BASE | DEVICE_KERNEL_BASE)  // 0x1_9000_0000

// Kernel code in DRAM (host view): 0x1_0000_0000 + 0x10000000 = 0x1_1000_0000
#define DRAM_KERNEL_BASE 0x110000000ull

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

  // // 2. Write test value into scratchpad
  // *TEST_ADDR_SPAD = 1;

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
