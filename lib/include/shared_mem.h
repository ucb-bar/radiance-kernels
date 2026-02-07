#pragma once

#include <VX_config.h>
#include <stdint.h>

#ifndef DEV_SMEM_START_ADDR
#define DEV_SMEM_START_ADDR SMEM_BASE_ADDR
#endif

#ifndef SMEM_SIZE
#define SMEM_SIZE (1u << SMEM_LOG_SIZE)
#endif

#define __global __attribute__((address_space(0)))
#define __shared __attribute__((address_space(1)))

static inline uint32_t vx_smem_load_u32(const volatile uint32_t* addr) {
  // convert full pointer into a byte offset
  uintptr_t byte_offset = ((uintptr_t)addr) - ((uintptr_t)DEV_SMEM_START_ADDR);
  // put operands into registers
  register const volatile uint32_t* rs1 asm("a1") =
      (const volatile uint32_t*)byte_offset;
  register uint32_t rd asm("a0");
  // load opcode
  asm volatile(".8byte 0x0000000000b41483" : "=r"(rd) : "r"(rs1) : "memory");
  return rd;
}

static inline void vx_smem_store_u32(volatile uint32_t* addr, uint32_t value) {
  uintptr_t byte_offset = ((uintptr_t)addr) - ((uintptr_t)DEV_SMEM_START_ADDR);
  register volatile uint32_t* rs1 asm("a1") = (volatile uint32_t*)byte_offset;
  register uint32_t rs2 asm("a0") = value;
  // store opcode
  asm volatile(".8byte 0x00000000a0b400a3" :: "r"(rs2), "r"(rs1) : "memory");
}
