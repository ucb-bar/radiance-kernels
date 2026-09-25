// template.h: layout shared by the GPU kernel (kernel.cpp) and the Rocket host (host.cpp).
//
// Addresses are GPU (device) addresses.  The host sees device address X at
// RAD_HOST_GPU_DRAM_BASE | X (0x1_0000_0000 | X).  Everything lives in the 0x1F00_0000 window,
// which is known to work for both host and GPU accesses on the tapeout FPGA; low device
// addresses (for example 0x1_0000, the old tocpu proxy) have faulted there.
//
// Keep clear of the drain scratch the epilogue uses: RAD_DRAIN_SCRATCH (0x1F20_0000) and the L1
// drain window above it (0x1F30_0000 .. 0x1F34_0000), see lib/include/rad_tapeout.h.

#ifndef TAPEOUT_TEMPLATE_H
#define TAPEOUT_TEMPLATE_H

#include <stdint.h>

#define TT_N          4096u          // elements per vector
#define TT_A_ADDR     0x1F000000u    // uint32_t A[TT_N]
#define TT_B_ADDR     0x1F010000u    // uint32_t B[TT_N]
#define TT_C_ADDR     0x1F020000u    // uint32_t C[TT_N], written by the GPU
#define TT_ARGS_ADDR  0x1F0F0000u    // struct TTArgs, written by the host before launch

// Warps per core.  The epilogue is verified on the board at 1, 2 and 3.
#define TT_OCCUPANCY  3u

#define TT_POISON     0xDEADBEEFu

// Kernel arguments.  The host writes this struct to TT_ARGS_ADDR before it releases the GPU.
// Every field is 32 bits so the rv64 host and the rv32 GPU agree on the layout.
struct TTArgs {
  uint32_t a;        // device address of A
  uint32_t b;        // device address of B
  uint32_t c;        // device address of C
  uint32_t n;        // number of elements
};

#endif  // TAPEOUT_TEMPLATE_H
