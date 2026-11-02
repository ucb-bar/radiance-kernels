// rq.h: shared constants for the tapeout MX requantizer probe (kernel.cpp, host.cpp).
#ifndef TAPEOUT_RQ_H
#define TAPEOUT_RQ_H

#include <stdint.h>

#ifndef RQ_NB
#define RQ_NB 512u          // 32-element MX blocks per run: 512 = one 64 x 256 FA P tile
#endif
#define RQ_SRC     0x0000u   // SMEM: bf16 input, RQ_NB * 64 bytes
#define RQ_DST     0x10000u  // SMEM: e4m3 output, RQ_NB * 32 bytes (window offset 2 * RQ_DST)
#define RQ_RED     0x1C000u  // SMEM: cross-lane reduction scratch
#define RQ_FLAG_SMEM 0x1C400u // SMEM: CONTEND-mode flag
#define RQ_SCRATCH 0x1CA00000u   // GMEM: target of the requantizer's own scale writes
#define RQ_DUMP    0x1CB00000u   // GMEM: copy of SRC (bf16) then DST (codes), for offline analysis
#ifndef RQ_DUMP_BLOCKS
#define RQ_DUMP_BLOCKS 64u     // blocks the host prints
#endif
#define RQ_DMA_SRC 0x1CC00000u   // DMA probe: host-written bytes
#define RQ_DMA_DST 0x1CC10000u   // DMA probe: move-out target
#define RQ_DMA_BYTES 0x8000u
#define RQ_ARGS    0x1F0F0000u
#define RQ_OCC     3u   // warps per core

// printBuf slots (32..63 are free for kernels)
#define RQ_PB_STATUS     32u   // 1 = output arrived, 2 = poll timed out
#define RQ_PB_T_FEED     33u   // cycles to issue all requantizer stores
#define RQ_PB_T_DONE     34u   // cycles from the first store until the last block's output is seen
#define RQ_PB_POLLS      35u
#define RQ_PB_RAW_RNE    36u   // raw output codes != model (round to nearest even), natural order
#define RQ_PB_RAW_TRUNC  37u   // raw output codes != model (truncation), natural order
#define RQ_PB_RAW_REV    38u   // raw output codes != model (RNE), order reversed within each block
#define RQ_PB_FIX_REF    39u   // exponent-shifted output codes != the SIMT quantizer (FA rule)
#define RQ_PB_FIX_FAR    40u   // ... of those, differing by more than one code step
#define RQ_PB_FIRST_BAD  41u   // first element where raw != RNE model: index
#define RQ_PB_FIRST_GOT  42u   // ... code read, model code, input bf16 (packed got | model << 8 | x << 16)
#define RQ_PB_FLAG       43u   // FEED done flag for the CONTEND workers
#define RQ_PB_DUMP       48u   // 48..55: first 32 output bytes; 56..63: first 32 model bytes

#define RQ_MODE_FEED   0u   // feed SRC through the requantizer, then check
#define RQ_MODE_STORES 1u   // window stores of a register value only (timing)
#define RQ_MODE_LOADS  2u   // SRC loads only (timing)
#define RQ_MODE_FENCE  4u   // as FEED, with fence.s between each batch's loads and its stores
#define RQ_MODE_GEN    5u   // as FEED, values computed in registers (no SMEM loads)
#define RQ_MODE_FEEDW  6u   // word loads, fence.s, then per block: low halves (beat 1), high (beat 2):
                            //   block element order even-then-odd
#define RQ_MODE_CSAME  7u   // FEEDW while core 0's other warps load and store SMEM
#define RQ_MODE_COTHER 8u
#define RQ_MODE_DMA16  9u   // DMA probe: DRAM -> SMEM -> DRAM, 16-row (256 B) commands
#define RQ_MODE_DMA64  10u  // same with 64-row (1 KB) commands
#define RQ_MODE_ISSUE  11u  // issue-rate probe: cycles per warp instruction for several op types
#define RQ_MODE_GELU   12u  // GELU element throughput, all warps at once (with and without the bf16 units)   // FEEDW while core 1's warps load and store SMEM
#define RQ_MODE_CONTEND 3u  // as FEED, while core 1's warps load and store SMEM (0x18000..) throughout
#ifndef RQ_MODE
#define RQ_MODE RQ_MODE_FEED
#endif

struct RQArgs {
  uint32_t mode;
};

#endif  // TAPEOUT_RQ_H
