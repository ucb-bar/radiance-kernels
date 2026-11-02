// gemm.h: layout shared by the GPU kernel (kernel.cpp), the Rocket host (host.cpp) and the data
// generator (gen_data.py).  MX-FP8 GEMM C[M][N] (bf16) = A[M][K] (e4m3) x B[K][N] (e4m3) with
// one e8m0 scale per 32 elements along K.
//
// Device addresses; the host sees device address X at 0x1_0000_0000 | X.  Operands live at
// 0x1C00_0000 (verified writable by host and GPU on the U250); the argument block stays in the
// 0x1F00_0000 window, below the epilogue drain scratch at 0x1F20_0000.

#ifndef TAPEOUT_GEMM_H
#define TAPEOUT_GEMM_H

#include <stdint.h>

#define TG_A_ADDR      0x1C000000u   // A  [M][K]  e4m3, row-major             (2 MB)
#define TG_B_ADDR      0x1C200000u   // B  [K][N]  e4m3, row-major             (2 MB)
#define TG_C_ADDR      0x1C400000u   // C  [M][N]  bf16, row-major, written by the mesh (4 MB)
#define TG_SA_ADDR     0x1C800000u   // A scales, tile-blocked: [M/TM][K/TK][TK/32][TM]
#define TG_SB_ADDR     0x1C900000u   // B scales, tile-blocked: [N/TN][K/TK][TK/32][TN]
#define TG_SCRATCH     0x1CA00000u   // C-scale move-out target required by the config; unused
#define TG_ARGS_ADDR   0x1F0F0000u   // struct TGArgs

#define TG_MAX_A_BYTES (TG_B_ADDR - TG_A_ADDR)
#define TG_MAX_B_BYTES (TG_C_ADDR - TG_B_ADDR)
#define TG_MAX_C_BYTES (TG_SA_ADDR - TG_C_ADDR)

// What the kernel runs.  The benchmarks repeat one operation `reps` times on output tile (0,0).
enum TGMode : uint32_t {
  TG_GEMM = 0,          // the full GEMM, verified against the golden
  TG_BENCH_MVIN = 1,    // move in one A tile and one B tile, fence; repeated
  TG_BENCH_COMPUTE = 2, // one k-tile matmul on resident operands, no fence between; repeated
  TG_BENCH_SCALES = 3,  // write one k-tile of A and B scales to the scale SRAM; repeated
};

struct TGArgs {
  uint32_t a, b, c, sa, sb;      // device addresses
  uint32_t m, n, k;              // GEMM shape
  uint32_t tm, tn, tk;           // tile shape (multiples of 16; tk a multiple of 32)
  uint32_t mode;                 // TGMode
  uint32_t reps;                 // benchmark repetitions
};

// printBuf slots for timestamps (mcycle of GPU thread 0).  32..47 are free in the slot map in
// lib/include/rad_tapeout.h.
#define TG_PB_T_START  32u   // entry
#define TG_PB_T_CFG    33u   // after the mesh configuration
#define TG_PB_T_END    34u   // all work issued and fenced
#define TG_PB_T_MARK   35u   // TG_GEMM: first output tile done
#define TG_PB_PROF     36u   // 36..39: MXG_PROF run_pipe phases (issue, scales, drain wait, C out)

#endif  // TAPEOUT_GEMM_H
