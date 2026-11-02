// ov.h: layout shared by the GPU kernel (kernel.cpp) and the Rocket host (host.cpp) of the
// mesh/SIMT overlap microbenchmark.
//
// Thread 0 runs OV_REPS matmuls (128 x 128 x 128 MX-FP8, operands resident in SMEM) while worker
// warps run a parameterized SMEM access pattern.  The host checks C (moved from the accumulator to
// DRAM) and prints the workers' checksums; arms are compared against a serialized arm.

#ifndef TAPEOUT_OVERLAP_H
#define TAPEOUT_OVERLAP_H

#include <stdint.h>

#define OV_A_ADDR      0x1C000000u   // A  [128][128] e4m3
#define OV_B_ADDR      0x1C200000u   // B  [128][128] e4m3
#define OV_C_ADDR      0x1C400000u   // C  [128][128] bf16, moved out of the accumulator
#define OV_SA_ADDR     0x1C800000u   // A scales [K/32][128]
#define OV_SB_ADDR     0x1C900000u   // B scales [K/32][128]
#define OV_SCRATCH     0x1CA00000u   // C-scale move-out target required by the config; unused
#define OV_C2_ADDR     0x1CB00000u   // OV_MM 2: C moved out of SMEM (C_SPAD_ROW) by DMA reads
#define OV_GREG        0x1CC00000u   // OV_PAT 6: DRAM region read by the workers (+ SMEM-style offsets)
#define OV_GRD         0x1D000000u   // OV_PAT 16: per-warp DRAM read regions (OV_GBYTES each)
#define OV_GWR         0x1D800000u   // OV_PAT 16: per-warp DRAM write regions (OV_GBYTES each)
#define OV_GBYTES      0x10000u      // 64 KB = 1024 lines per warp: 3 warps per core exceed the L0d
#define OV_HASH_ADDR   0x1CD00000u   // OV_SMEMHASH: 32 blocks x 16 lane hashes
#define OV_ARGS_ADDR   0x1F0F0000u   // word 0: launch mode (0 generate operands in DRAM, 1 benchmark)

// printBuf postbox slots (32..63 are free for kernels, rad_tapeout.h slot map)
#define OV_PB_T0       32u   // thread 0: entry
#define OV_PB_T_A      33u   // thread 0: after barrier A (overlap window opens)
#define OV_PB_T_ISS    34u   // thread 0: all matmuls issued
#define OV_PB_T_MM     35u   // thread 0: matmuls drained
#define OV_PB_T_B      36u   // thread 0: after barrier B (all workers done)
#define OV_PB_W_END    37u   // latest worker end time (mcycle, core-local counter of each core)
#define OV_PB_RSUM     38u   // sum over worker lanes of the read accumulator
#define OV_PB_HSUM     39u   // sum over worker lanes of the region hash after barrier B
#define OV_PB_W_MIN    40u   // earliest worker end time
#define OV_PB_MAGIC    41u   // 0x0E0Fxxxx: kernel reached its final post (xxxx = reps)
#define OV_PB_ITERS    42u   // worker iterations (echo of the build flag)
#define OV_PB_STAGE    43u   // thread 0 progress beacon (1..9, see kernel.cpp)

// SMEM byte offset of the per-lane result area (bank 3, below B; used only when the mesh is idle)
#define OV_RES_SMEM    0x1B000u

// Experiment knobs (documented in kernel.cpp); the host reads them too, to compute the workers'
// expected checksums.  Pass the same -D flags to both sides (build_ov.sh does).
#ifndef OV_PAT
#define OV_PAT 0
#endif
#ifndef OV_RB
#define OV_RB 0x8000u
#endif
#ifndef OV_WBYTES
#define OV_WBYTES 2048u
#endif
#ifndef OV_ITERS
#define OV_ITERS 1024u
#endif
#ifndef OV_GAP
#define OV_GAP 0u
#endif
#ifndef OV_CORES
#define OV_CORES 3u
#endif
#ifndef OV_SER
#define OV_SER 0
#endif
#ifndef OV_MM
#define OV_MM 0
#endif
#ifndef OV_REPS
#define OV_REPS 8u
#endif
#ifndef OV_STRIDE
#define OV_STRIDE 1u   // line stride between consecutive accesses of a warp
#endif

#ifndef OV_RB2
#define OV_RB2 0x10000u   // OV_PAT 9: store region
#endif
#ifndef OV_WMASK
#define OV_WMASK 0x3Fu   // worker slots (slot = core * 3 + warp_in_core; slot 0 is the mesh driver)
#endif
#ifndef OV_NOPOLL
#define OV_NOPOLL 0
#endif
#ifndef OV_POLLGAP
#define OV_POLLGAP 0
#endif
#ifndef OV_DUMMYPOLL
#define OV_DUMMYPOLL 0
#endif
#ifndef OV_STAGE
#define OV_STAGE 0     // 1: fx_stage_ovl2-like per-k-step structure (kernel.cpp OV_STAGE section)
#endif
#ifndef OV_SMVIN
#define OV_SMVIN 0
#endif
#ifndef OV_SSCL
#define OV_SSCL 0
#endif
#ifndef OV_SCST
#define OV_SCST 0
#endif
#ifndef OV_SMVOUT
#define OV_SMVOUT 0
#endif
#ifndef OV_SW
#define OV_SW 0
#endif
#ifndef OV_STAGE_POLLGAP
#define OV_STAGE_POLLGAP 0
#endif
#ifndef OV_S4LD
#define OV_S4LD 1   // OV_SW 4: C loads
#endif
#ifndef OV_S4ST
#define OV_S4ST 1   // OV_SW 4: code word stores
#endif
#ifndef OV_BREAK
#define OV_BREAK 0   // 1: corrupt A, 2: workers skip an iteration (falsify the checks)
#endif

#define OV_OCC 3u
// number of worker warps (for the worker-only barriers of OV_PAT 13)
#define OV_WK(c, w) ((((OV_CORES >> (c)) & 1u) && ((OV_WMASK >> ((c) * 3u + (w))) & 1u)) ? 1u : 0u)
#define OV_NWORKERS (OV_WK(0, 1) + OV_WK(0, 2) + OV_WK(1, 0) + OV_WK(1, 1) + OV_WK(1, 2))

#endif  // TAPEOUT_OVERLAP_H
