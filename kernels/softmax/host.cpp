#include <radiance.h>

// MMIOAddressOrNode register for cluster 0
// cluster baseAddr (0x4000_0000) + peripheralAddrOffset (0x80000) + 0x1000
#define GPU_ADDR_OR_MMIO ((volatile uint64_t *)0x40081000ull)

// OR-node base for scratchpad
#define SPAD_BASE 0x180000000ull

// Device kernel base address (GPU address space)
#define DEVICE_KERNEL_BASE 0x10000000ull

// Post-OR address: where the GPU reads after OR-node remapping.
// Host writes here so LLC coherence probes serve the data to the GPU.
#define POST_OR_BASE (SPAD_BASE | DEVICE_KERNEL_BASE)

// Kernel code in DRAM (host view): 0x1_0000_0000 + 0x10000000
#define DRAM_KERNEL_BASE 0x110000000ull

static void copy_section(uint32_t offset, uint32_t size_bytes) {
  volatile uint64_t *src = (volatile uint64_t *)(DRAM_KERNEL_BASE + offset);
  volatile uint64_t *dst = (volatile uint64_t *)(POST_OR_BASE + offset);
  for (uint32_t i = 0; i < (size_bytes + 7) / 8; i++) {
    dst[i] = src[i];
  }
}

int main() {
  // Copy kernel LOAD segments from DRAM to post-OR addresses.
  // Sizes from: readelf -l kernel.radiance.elf (2 rows x 512 cols)
  copy_section(0x0000, 0x68);   // .init
  copy_section(0x1000, 0x4c);   // .tohost + .sdata
  copy_section(0x2000, 0xe40);  // .text
  copy_section(0x3000, 0x808);  // .data (input tensors)

  asm volatile("fence" ::: "memory");

  // Switch GPU address base to scratchpad
  *GPU_ADDR_OR_MMIO = SPAD_BASE;

  // Deassert GPU reset
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

  // Wait for GPU to finish
  uint32_t finished = 0;
  while (!finished) {
    SYNC_GPU();
    finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
  }

  return 0;
}
