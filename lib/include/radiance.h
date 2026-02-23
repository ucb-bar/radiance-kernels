/*
 * radiance.h: Radiance host/device API.
 *
 * This header implements interface for control GPU kernel launch, access
 * kernel argument space, etc to both host and device.  Host-only interfaces
 * such as kernel launch are not exposed to the device kernel.
 */

#ifndef __RADIANCE_H__
#define __RADIANCE_H__

#include <stdint.h>

// Host/device base address of argument buffer space
#define RAD_HOST_ARG_BASE   0x10FFF0000ull
#define RAD_DEVICE_ARG_BASE  0x0FFF0000ul
// Host base address of GPU GMEM(DRAM) address space
#define RAD_HOST_GPU_DRAM_BASE 0x100000000ul

#define RAD_HOST_GPU_RESET 0x41000000ull
#define RAD_HOST_GPU_ALL_FINISHED 0x41000008ull
#define RAD_HOST_GPU_CORES 0x41000010ull

#ifndef RADIANCE_DEVICE

// Host-only interfaces
// --------------------

#define READ_MMIO_32(addr)                                                     \
  ({                                                                           \
    uint32_t result = (*(volatile uint32_t *)(addr));                          \
    result;                                                                    \
  })

#define WRITE_MMIO_32(addr, data)                                              \
  (*(volatile uint32_t *)(addr)) = (uint32_t)(data)

extern volatile uint64_t tohost;
volatile uint64_t *tocpu = (volatile uint64_t *)0x100010000ULL;

inline static void SYNC_GPU() {
  volatile uint64_t _fromcpu = *tocpu;
  if (_fromcpu > 0) {
    volatile uint64_t _fromhost = tohost;
    tohost = _fromcpu;
    *tocpu = _fromhost;
  }
}

#endif

#endif // __RADIANCE_H__
