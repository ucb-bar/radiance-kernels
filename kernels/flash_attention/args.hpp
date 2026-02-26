#ifndef _ARGS_H_
#define _ARGS_H_

#include <cstdint>

struct FlashAttentionKernelArgs {
  uint32_t dim_seqlen;
  uint32_t dim_headdim;
  _Float16 *addr_q;
  _Float16 *addr_k;
  _Float16 *addr_v;
  _Float16 *addr_o;
};

#endif
