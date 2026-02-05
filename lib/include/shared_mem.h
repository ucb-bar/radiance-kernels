// Copyright © 2019-2023
//
// Shared-memory helper definitions used by the SMEM microbenchmarks.

#pragma once

#include <VX_config.h>
#include <stdint.h>

// Scratchpad is aliased with the architectural stack space in the Vortex
// memory map.  SMEM_LOG_SIZE / SMEM_BASE_ADDR come from VX_config.h.
#ifndef DEV_SMEM_START_ADDR
#define DEV_SMEM_START_ADDR SMEM_BASE_ADDR
#endif

#ifndef SMEM_SIZE
#define SMEM_SIZE (1u << SMEM_LOG_SIZE)
#endif

#if defined(__clang__)
#define __shared__ __attribute__((annotate("vortex.shared")))
#else
#define __shared__
#endif

inline volatile void* vx_shared_ptr(uint32_t byte_offset) {
  return reinterpret_cast<volatile void*>(DEV_SMEM_START_ADDR + byte_offset);
}
