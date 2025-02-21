#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7fff0000
#define DEV_SMEM_START_ADDR 0xff000000

typedef struct {
  uint32_t dim_m;
  uint32_t dim_n;
  uint32_t dim_k;
  uint64_t addr_a;
  uint64_t addr_b;
  uint64_t addr_c;
} kernel_arg_t;

#endif
