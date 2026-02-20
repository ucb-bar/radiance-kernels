#ifndef __MU_INTRINSICS_H__
#define __MU_INTRINSICS_H__

#include <stdint.h>
#include <vx_intrinsics.h>

// Base of GPU GMEM(DRAM) address space in host CPU's global address space
#define GPU_DRAM_ADDR_BASE 0x100000000ul

inline void store_shared(uint32_t base, uint32_t offset, uint32_t data) {
  asm volatile("sw.shared %2, %1(%0)" :: "r"(base), "I"(offset), "r"(data)
               : "memory");
}

// This compiles to CSR reads which stalls the pipeline. Use sparingly & cache.
inline int mu_num_threads() {
    return vx_num_threads();
}

inline void store64_shared(uint32_t base, uint32_t offset, uint64_t data) {
  uint32_t lo = static_cast<uint32_t>(data);
  uint32_t hi = static_cast<uint32_t>(data >> 32);
  store_shared(base, offset,     lo);
  store_shared(base, offset + 4, hi);
}

inline float mu_fexp(float arg) {
  float output;
  asm volatile("fexp.s %0, %1" ::"r"(output), "r"(arg));
  return output;
}

// TODO: half?

#endif // __MU_INTRINSICS_H__
