// Rocket (rv64 host) side of the FA-MX kernel.
//
// HOST-ASSISTED MX SCALE LOADING
// ==============================
// The GPU used to burn ~32.3k cycles in `load_scale_factors()` (576 single-thread 4-byte GMEM
// reads + SF-SRAM writes).  The gemmini scale SRAM (ScalingFactorMem) is a plain TL slave:
//   cluster_base + 0x88000 -> weight/B scales      (GEMMINI_SF_MEM_B, GPU-local 0x88000)
//   cluster_base + 0x8a000 -> activation/A scales  (GEMMINI_SF_MEM_A, GPU-local 0x8a000)
// The RV32 Muon lanes can only emit 4-byte stores, which have to traverse FlitMergeNode's
// pair-merge FSM (radiance/memory/FlitMergeNode.scala:36-37); that FSM asserts on any
// non-ascending / unaligned pair, which is why the GPU had to write them single-threaded and
// strictly in order.  An 8-BYTE store has size==3 and BYPASSES the merge node entirely
// (`shouldMerge` requires size==2) while still satisfying GemminiTile.scala:286.  The rv64 host
// can issue those, so it writes the whole scale set itself and the GPU is built -DFA_NOSCALES.
//
// Addressing: RadianceCluster baseAddr = 0x4000_0000 + 0x10_0000*clusterId
// (subsystem/Configs.scala:456), peripheralAddrOffset = 0x8_0000, SF mem = +0x8000
// (Configs.scala:379).  The kernel runs REDUNDANTLY on both clusters, so BOTH gemminis' scale
// SRAMs have to be filled.  This is the same system-physical space the host already uses for
// GPU_ADDR_OR_MMIO (0x40081000) -- NOT the 0x1_xxxx_xxxx GPU-DRAM alias.
//
// Double buffering: SF_MEM_B is needed twice per FA pass (K scales for QK^T, then V scales for
// PV) and both are loaded up front, so they must land in different halves of the weight-scale
// double buffer (ScaleFactorMem.scala:195-196,230-237; SW offset GEMMINI_SF_MEM_BUFFER_OFFSET
// = 0x800 -> SRAM banks 0/1 vs 2/3).  K -> buffer 0, V -> buffer 1; the GPU flips
// CONFIG_SCALE_MEM's scale_w_sel bit (rs1[61]) for the PV matmul -- see mxgemm_core.hpp.
//
// TIMING / HANDSHAKE (this is the load-bearing part).  RadianceTapeoutSimConfig uses
// WithGPUResetAggregator(defaultReset=false): the Muon cores are NOT held in reset, they start
// at t=0 together with Rocket.  There is no "pre-launch" window at all -- this is a genuinely
// CONCURRENT host write that races the GPU.  MEASURED (-DFA_HOST_TIMING build, stamps echoed
// back through the mailbox as marks):
//     Rocket enters main()            cycle  2,286
//     all 4,608 scale bytes written   cycle 13,895   (11,609 cyc / 576 8B stores = ~20 cyc each,
//                                                     i.e. ~0.40 B/cyc of host->cluster MMIO)
//     GPU reaches kernel entry        cycle 14,175
// So the host actually finishes BEFORE the GPU starts, and the flag waits below cost only
// 262/611 cyc (QK) and 383/233 cyc (V).  For reference the GPU's own load_scale_factors moves
// 1,280 B in 20,470 cyc = 0.0625 B/cyc, so the host is ~6.4x faster per byte AND concurrent.
// The handshake is still mandatory, not decorative: an earlier no-handshake version had the
// second cluster's K scales land after the mesh had consumed them -- cluster 0 came out at
// Frobenius 4.60% (correct) and cluster 1 at 78.7% (garbage).

#pragma GCC optimize("O2")

#include <inttypes.h>
#include <stdio.h>
#include <radiance.h>

// ============================================================================================
// MATCHED PAIR: the prefill below is compiled ONLY when the GPU side is built -DFA_NOSCALES.
// /tmp/fa_build.sh forwards its -D list to the host compile via FA_HOST_DEFS (see the kernel
// Makefile).  This gating is NOT cosmetic: the scale SRAM's weight write port has a SINGLE
// shared 2-beat pairing counter (ScaleFactorMem.scala:135-148 -- two consecutive 8B beats are
// concatenated into one 16B SRAM row and the row address is taken from the SECOND beat).  If
// the host and the GPU both write weight scales at the same time their beats interleave and
// pair up wrongly, silently corrupting the SRAM.  So: host writes them, or the GPU does --
// never both.
// ============================================================================================
#ifdef FA_NOSCALES
#define FA_HOST_SCALE_PREFILL 1
#include "include/fa_data.h"
#endif

#ifdef FA_HOST_SCALE_PREFILL

// ---- gemmini scale-SRAM addresses, host (system-physical) address space --------------
#define CLUSTER_BASE(cl)  (0x40000000ull + 0x100000ull * (cl))
#define SF_MEM_B(cl)      (CLUSTER_BASE(cl) + 0x88000ull)   // weight / B scales
#define SF_MEM_A(cl)      (CLUSTER_BASE(cl) + 0x8a000ull)   // activation / A scales
#define SF_BUFFER_OFFSET  0x800ull                          // GEMMINI_SF_MEM_BUFFER_OFFSET
#define NUM_CLUSTERS      2

// Host<->GPU mailbox in an unused 256B window of cluster SMEM (device 0x17F00; the kernel's SMEM
// map ends at REDBUF_SMEM=0x15000 and the gemmini B-spad starts at 0x18000).  Two hard-won rules
// for this channel, both measured:
//   * writes MUST be 8 bytes.  A 4-byte (sub-beat) write from the host's TL port is silently
//     dropped -- the GPU spun 195k iterations on a flag "written" with sw, while the same address
//     written with sd reads back correctly.
//   * it must be cluster SMEM, not the cluster print buffer (TLRAM at +0x80000): an 8B host store
//     there never became visible to the Muon either.
// Layout:
//   +0x00  QK scales ready (magic)     +0x10  V scales ready (magic)
//   +0x80  (t_enter_main, t_prefill_done) rdcycle stamps, echoed back as marks under
//          -DFA_HOST_TIMING purely for measurement.
// The HOST clears both flags as the very first thing in main() (~2.3k cyc); the GPU only ever
// reads them (it first looks at ~27k).  Clearing them GPU-side instead deadlocks -- Rocket
// finishes the QK subset before the GPU reaches its kernel entry at ~15k, so the GPU would wipe
// a signal that had already been raised.  (Observed, twice.)
#define HOST_MBOX(cl)     (CLUSTER_BASE(cl) + 0x17F00ull)
#define MBOX_QK_READY     0x00
#define MBOX_V_READY      0x10
#define MBOX_STAMPS       0x80
#define FA_HOST_MAGIC     0x5CA1E5u

// Dense, strictly-ascending 8-byte stores: the SF write port pairs two consecutive 8B beats
// into one 16B SRAM row, so they must be dense, in order, and an even multiple of 8 bytes.
static void sf_write(uint64_t dst, const void *src, unsigned nbytes) {
  volatile uint64_t *d = (volatile uint64_t *)dst;
  const unsigned n = nbytes / 8;
  if (((uintptr_t)src & 7u) == 0) {
    const uint64_t *s = (const uint64_t *)src;   // fa_data.h arrays are 8B aligned in practice
    for (unsigned i = 0; i < n; i++) d[i] = s[i];
  } else {
    const uint8_t *s = (const uint8_t *)src;
    for (unsigned i = 0; i < n; i++) {
      uint64_t v;
      __builtin_memcpy(&v, s + 8u * i, 8);
      d[i] = v;
    }
  }
}

// QKF: TILE_M=Sq=64, TILE_N=Sk=256, TILE_K=d=128
//   A scales = TILE_M*TILE_K/32 =  256 B   (QK_A_scales_row[FA_GK][FA_SQ])
//   B scales = TILE_N*TILE_K/32 = 1024 B   (QK_B_scales_col[FA_GK][FA_SK])
// PVF: TILE_M=Sq=64, TILE_N=d=128, TILE_K=Sk=256
//   B scales = TILE_N*TILE_K/32 = 1024 B   (V_scales[FA_GKV][FA_D]);  A scales are the runtime
//   P scales, produced on-GPU by pack_scales_to_sfmem into SF_MEM_A buffer 0.
#define QK_A_SCALE_BYTES  (FA_GK * FA_SQ)     // 256
#define QK_B_SCALE_BYTES  (FA_GK * FA_SK)     // 1024
#define V_SCALE_BYTES     (FA_GKV * FA_D)     // 1024

// MUST be an 8-byte store: a 4-byte (sub-beat, PutPartial) write from the host's TL port into
// cluster SMEM is silently DROPPED -- measured, the GPU spun forever on a 4B-written flag while
// the same mailbox written with `sd` was read back correctly.
static inline void mbox_set(unsigned off, uint32_t val) {
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    *(volatile uint64_t *)(HOST_MBOX(cl) + off) = (uint64_t)val;
}

static inline uint64_t rdcycle(void) {
  uint64_t c;
  asm volatile("rdcycle %0" : "=r"(c));
  return c;
}
#endif // FA_HOST_SCALE_PREFILL

int main() {
#ifdef FA_HOST_SCALE_PREFILL
  const uint64_t t_enter = rdcycle();
  mbox_set(MBOX_QK_READY, 0);   // own the mailbox before the GPU ever looks at it
  mbox_set(MBOX_V_READY, 0);

  // Phase 1: QK^T scales (deadline = the QK matmul, ~34k cyc) for BOTH clusters, then publish.
  for (int cl = 0; cl < NUM_CLUSTERS; cl++) {
    sf_write(SF_MEM_A(cl), &QK_A_scales_row[0][0], QK_A_SCALE_BYTES);
    sf_write(SF_MEM_B(cl), &QK_B_scales_col[0][0], QK_B_SCALE_BYTES);
  }
  asm volatile("fence" ::: "memory");
  mbox_set(MBOX_QK_READY, FA_HOST_MAGIC);

  // Phase 2: PV V scales into weight-scale buffer 1 (deadline ~140k cyc -- huge slack).
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    sf_write(SF_MEM_B(cl) + SF_BUFFER_OFFSET, &V_scales[0][0], V_SCALE_BYTES);
  asm volatile("fence" ::: "memory");
  mbox_set(MBOX_V_READY, FA_HOST_MAGIC);

  const uint64_t t_done = rdcycle();
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    *(volatile uint64_t *)(HOST_MBOX(cl) + MBOX_STAMPS) = (t_done << 32) | (uint32_t)t_enter;
#endif

  tohost = 0;
  *tocpu = tohost;

  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

  uint32_t finished = 0;
  while (!finished) {
    SYNC_GPU();
    finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
  }

  return 0;
}
