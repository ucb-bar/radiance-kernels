#ifndef RADIANCE_KERNELS_MEMSTRESS_COMMON_H_
#define RADIANCE_KERNELS_MEMSTRESS_COMMON_H_

#include <stdint.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7fff0000UL

typedef struct {
  uint32_t num_threads;         // override for vx_spawn grid size
  uint32_t iterations;          // per-test loop iterations
  uint32_t elements_per_thread; // work per logical thread
  uint32_t stride_elems;        // used by strided/indirect patterns
  uint32_t tile_elems;          // shared-memory tile size hints
  uint32_t smem_banks;          // expected number of SMEM banks
  uint64_t gmem_src;            // base pointer for read streams
  uint64_t gmem_dst;            // base pointer for write streams
  uint64_t gmem_aux;            // optional third region
} kernel_arg_t;

#endif
