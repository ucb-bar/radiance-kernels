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
// SF_MEM_A IS ALSO SPLIT (2026-07-25).  Within one FA tile the activation scale SRAM has two
// producers: QK's A operand is Q (host-written QK_A_scales_row) and PV's A operand is the
// requantized P (GPU-written by pack_scales_to_sfmem, which hardcodes half 0).  Single-shot they
// can share half 0 -- QK consumes it before pack overwrites it -- but under the FA_STEADY
// multi-tile loop tile t's P scales are still sitting in half 0 when tile t+1's QK reads it, so
// every tile after the first computes Q * (softmax scales).  So: QK_A -> activation half **1**
// (SF_MEM_A + 0x800, act SRAM banks 2/3) and the GPU keeps half 0.  The GPU sets
// CONFIG_SCALE_MEM's scale_mem_read_act_sel (rs1[60]) to 1 for QK and 0 for PV.
// As a bonus this makes the host's half PRIVATE to the host, which is what makes the per-tile
// refill (-DFA_HOSTHS below) possible at all.
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
#define MBOX_GPU_QKDONE   0x20   // GPU -> host: #tiles whose QK matmul has drained
#define MBOX_GPU_PVDONE   0x30   // GPU -> host: #tiles whose PV matmul has drained
#define MBOX_STAMPS       0x80   // diagnostic block, echoed to MARK_GMEM+0x300 by the GPU
#define FA_HOST_MAGIC     0x5CA1E5u

// ---- FA_STEADY tile count (must MATCH the GPU build; /tmp/fa_build.sh forwards the -D list) ----
#ifdef FA_STEADY
#  if   defined(FA_NT1)
#    define HOST_NTILES 1
#  elif defined(FA_NT2)
#    define HOST_NTILES 2
#  elif defined(FA_NT3)
#    define HOST_NTILES 3
#  elif defined(FA_NT4)
#    define HOST_NTILES 4
#  elif defined(FA_NT6)
#    define HOST_NTILES 6
#  elif defined(FA_NT8)
#    define HOST_NTILES 8
#  else
#    define HOST_NTILES 4
#  endif
#else
#  define HOST_NTILES 1
#endif

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

// ---- one FA tile's worth of scale traffic ---------------------------------------------------
// 2,304 B per cluster (QK_A 256 + QK_B 1024 + V 1024), 4,608 B for the pair.  These are the three
// arrays a REAL streaming FA would have to re-fetch every tile (K and V change per KV block).
//
// THE QK PAIR IS SPLIT INTO TWO FUNCTIONS BECAUSE THE TWO PORTS HAVE DIFFERENT HAZARDS.
// ScaleFactorMem has TWO independent write ports, each with its OWN 2-beat pairing register
// (ScaleFactorMem.scala:119-147 `write_weight_counter`, :150-176 `write_act_counter`): two
// consecutive 8B TL beats are concatenated into one 16B SRAM row and the row address is taken
// from the SECOND beat.  The port is selected by the TOP address bit (GemminiTile.scala:265;
// 0x8a000 -> act, 0x88000 -> weight), NOT by the double-buffer half.  Consequences:
//   * WEIGHT port (QK_B = K scales, V scales): in an -DFA_NOSCALES build the GPU never writes
//     weight scales at all, so this port is HOST-PRIVATE.  The only constraint is the mesh's
//     READ -- a half may be rewritten once the gemm that reads it has drained.
//   * ACT port (QK_A = Q scales): SHARED.  pack_scales_to_sfmem writes the runtime P scales into
//     act half 0 every tile, and its 4B stores are merged into 8B beats by FlitMergeNode.  If a
//     host `sd` lands between two of those beats the pair is formed from one GPU beat + one host
//     beat and the resulting 16B row is written to the wrong address with half-wrong data --
//     SILENT corruption.  The double-buffer split does NOT protect against this: the pairing
//     register is shared across all four banks of the port.
// So the host's ACT write is only legal in a window where the GPU provably touches no act scale.
// A sequential FULL_ATTN2 tile is
//     QK prefetch | QK matmul | softmax | PV prefetch | requant | PACK(act half 0) | PV matmul
// so the safe window is [tile t-1's PV matmul drained .. tile t's PACK], which the GPU signals by
// publishing MBOX_GPU_PVDONE >= t.  That window contains softmax+requant (~40k cycles of slack);
// the act write itself is 2 x 256 B = 64 `sd` ~= 1.3k cycles.  See the refill loop in main().
static inline void write_qk_a_scales(void) {   // ACT port  -- hazard above, PVDONE-gated
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    // QK activation (Q) scales -> activation half 1; the GPU owns half 0 for the P scales.
    sf_write(SF_MEM_A(cl) + SF_BUFFER_OFFSET, &QK_A_scales_row[0][0], QK_A_SCALE_BYTES);
}
static inline void write_qk_b_scales(void) {   // WEIGHT port -- host-private under FA_NOSCALES
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    // QK weight (K) scales -> weight half 0.
    sf_write(SF_MEM_B(cl), &QK_B_scales_col[0][0], QK_B_SCALE_BYTES);
}
static inline void write_qk_scales(void) { write_qk_a_scales(); write_qk_b_scales(); }
static inline void write_v_scales(void) {
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    sf_write(SF_MEM_B(cl) + SF_BUFFER_OFFSET, &V_scales[0][0], V_SCALE_BYTES);
}

// Read a GPU->host progress word out of cluster SMEM.  8-byte load to mirror the write path (the
// cluster SMEM slave is word-strided behind a min-8B fragmenter; a 4B host access is the one that
// was measured to be silently dropped on the write side).
static inline uint32_t mbox_get(unsigned cl, unsigned off) {
  return (uint32_t)*(volatile uint64_t *)(HOST_MBOX(cl) + off);
}

// Diagnostics the GPU echoes to MARK_GMEM+0x300 (mxgemm_core.hpp, -DFA_HOST_TIMING).
static uint32_t g_diag[16] = {0};

// ---- MMIO cost microbenchmark (-DFA_HOSTPROBE) ----------------------------------------------
// Everything the host can usefully offload reduces to "how many cycles does one 8-byte host
// access to the cluster cost", and READS are a different question from WRITES: the scale prefill
// is write-only (fire-and-forget into a TL slave) but a `pack_scales` offload would have to READ
// 512 words of SCALE_SMEM per cluster first, and Rocket cannot pipeline uncached loads at all.
// Measured once at entry, before the prefill, so it never perturbs the handshake.  Timed with
// rdcycle around a fixed count; the loads are volatile and their results are summed into a value
// that is published, so nothing is optimizable away.
//   diag[10] = cycles for 256 8B reads of cluster-0 SMEM   (-> 512 words, one cluster's P scales)
//   diag[11] = cycles for  64 8B writes to SF_MEM_A half 1 (-> 512 B, one cluster's QK_A scales)
//   diag[12] = cycles for 128 8B writes to SF_MEM_B half 0 (-> 1 KB, one cluster's K scales)
//   diag[13] = read checksum (proves the loads happened)
#define SCALE_SMEM_HOST(cl) (CLUSTER_BASE(cl) + 0x14000ull)   // kernel SCALE_SMEM, device 0x14000

static void host_mmio_probe(void) {
  volatile uint64_t *rd = (volatile uint64_t *)SCALE_SMEM_HOST(0);
  uint64_t sum = 0;
  uint64_t t0 = rdcycle();
  for (int i = 0; i < 256; i++) sum += rd[i];
  uint64_t t1 = rdcycle();
  sf_write(SF_MEM_A(0) + SF_BUFFER_OFFSET, &QK_A_scales_row[0][0], QK_A_SCALE_BYTES);
  uint64_t t2 = rdcycle();
  sf_write(SF_MEM_B(0), &QK_B_scales_col[0][0], QK_B_SCALE_BYTES);
  uint64_t t3 = rdcycle();
  g_diag[10] = (uint32_t)(t1 - t0);
  g_diag[11] = (uint32_t)(t2 - t1);
  g_diag[12] = (uint32_t)(t3 - t2);
  g_diag[13] = (uint32_t)sum;
}

#ifdef FA_HOSTCFG
// ---- HOST-ISSUED GEMMINI CONFIG + MOVE-IN (capture / replay) ---------------------------------
// See the long FA_HOSTCFG comment in mxgemm_core.hpp for the why and the mutual exclusion rules.
// The GPU records its tile-0 12-command stream (5 words per command: rs1 lo/hi, rs2 lo/hi, inst)
// into cluster SMEM; the host caches it in its own DRAM and replays it for every later tile.
#define HOST_CFGREC(cl, pv) (CLUSTER_BASE(cl) + ((pv) ? 0x17A00ull : 0x17800ull))
#define MBOX_CFGREC         0x40   // GPU -> host: 1 = QK recorded, 2 = QK and PV recorded
#define GEMMINI_CTRL_HOST(cl) (CLUSTER_BASE(cl) + 0x84000ull)   // GemminiTile.scala:419 regmap
#define CFG_MAX_CMDS 16
static uint32_t g_rec[2][CFG_MAX_CMDS][5];
static uint32_t g_rec_n[2] = {0, 0};

// Cluster 0's recording is valid for both clusters (rad_device_to_host_address carries no cluster
// id).  Read with 8-byte loads: a 4-byte host access into the cluster is the one that was measured
// to be silently dropped.
static void cfg_capture(void) {
  for (int pv = 0; pv < 2; pv++) {
    const uint64_t base = HOST_CFGREC(0, pv);
    uint32_t n = (uint32_t)*(volatile uint64_t *)base;      // rec[0] = command count
    if (n > CFG_MAX_CMDS) n = CFG_MAX_CMDS;                 // never trust SMEM blindly
    g_rec_n[pv] = n;
    for (uint32_t c = 0; c < n; c++)
      for (int w = 0; w < 5; w++) {
        const uint64_t a = base + 4ull * (1u + 5u * c + (unsigned)w);
        const uint64_t v = *(volatile uint64_t *)(a & ~7ull);
        g_rec[pv][c][w] = (a & 4u) ? (uint32_t)(v >> 32) : (uint32_t)v;
      }
  }
}

// Replay one gemm's command stream into BOTH clusters' gemmini command ports.  rs1 (+0x10) and rs2
// (+0x18) are each a pair of 32-bit RegFields inside one 8-byte word, so a single `sd` sets both
// halves; the write to +0x00 latches the instruction and FIRES the command, and it backpressures
// on gemminiIO.ready, so no polling is needed.
static void cfg_replay(int pv) {
  for (int cl = 0; cl < NUM_CLUSTERS; cl++) {
    volatile uint64_t *ctl = (volatile uint64_t *)GEMMINI_CTRL_HOST(cl);
    const uint32_t n = g_rec_n[pv];
    for (uint32_t c = 0; c < n; c++) {
      const uint32_t *r = g_rec[pv][c];
      if (r[4] == 0u) {                                   // pseudo-command: gemmini_fence()
        for (uint32_t s = 0; s < 100000u; s++)
          if ((uint32_t)ctl[4] == 0u) break;              // +0x20 busy
#ifdef FA_HOSTCFG_NOMVIN
        // TIMING-ONLY PROBE (-DFA_HOSTCFG_NOMVIN): replay the CONFIG group but drop the GMEM->spad
        // MOVE-IN group that follows the fence, so the matmuls run on whatever is already in the
        // operand spads.  The OUTPUT IS GARBAGE (Frobenius 116% -- the SIMT requant overwrites the
        // A spad with P every tile, so Q does not survive); the only thing this build measures is
        // what the operand move-in costs in the steady-state slope.
        // RESULT: 80,137 cyc/tile vs 80,054 with the move-in -- i.e. ZERO.  Once the host issues
        // the mvin at the PVDONE/QKDONE handshake points, its DMA is already fully hidden behind
        // finalize + softmax + requant + pack, so a spad-double-buffered host prefetch (issue tile
        // t+1's K/V a whole tile early) has nothing left to win.  That is why it was not built.
        break;
#endif
        continue;
      }
      ctl[2] = ((uint64_t)r[1] << 32) | (uint64_t)r[0];   // +0x10 rs1 {lo, hi}
      ctl[3] = ((uint64_t)r[3] << 32) | (uint64_t)r[2];   // +0x18 rs2 {lo, hi}
      ctl[0] = (uint64_t)r[4];                            // +0x00 inst -> issue
    }
  }
}
#endif // FA_HOSTCFG
static inline void diag_publish(void) {
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    for (int i = 0; i < 8; i++)
      *(volatile uint64_t *)(HOST_MBOX(cl) + MBOX_STAMPS + 8 * i) =
          ((uint64_t)g_diag[2 * i + 1] << 32) | (uint64_t)g_diag[2 * i];
}
#endif // FA_HOST_SCALE_PREFILL

int main() {
#ifdef FA_HOST_SCALE_PREFILL
  const uint64_t t_enter = rdcycle();
  mbox_set(MBOX_QK_READY, 0);   // own the mailbox before the GPU ever looks at it
  mbox_set(MBOX_V_READY, 0);
  mbox_set(MBOX_GPU_QKDONE, 0); // ...and pre-zero the GPU->host progress words so that a broken
  mbox_set(MBOX_GPU_PVDONE, 0); //    host read path shows up as "stuck at 0" rather than garbage.
#ifdef FA_HOSTCFG
  mbox_set(MBOX_CFGREC, 0);     // same: SMEM contents at reset are undefined
#endif
#ifdef FA_HOSTPROBE
  host_mmio_probe();            // MMIO read/write cost microbenchmark (diag[10..13])
#endif

  // Phase 1: QK^T scales (deadline = the QK matmul, ~34k cyc) for BOTH clusters, then publish.
  write_qk_scales();
  asm volatile("fence" ::: "memory");
#ifdef FA_HOSTHS
  mbox_set(MBOX_QK_READY, 1);   // sequence number = #tiles resident
#else
  mbox_set(MBOX_QK_READY, FA_HOST_MAGIC);
#endif

  // Phase 2: PV V scales into weight-scale buffer 1 (deadline ~140k cyc -- huge slack).
  write_v_scales();
  asm volatile("fence" ::: "memory");
#ifdef FA_HOSTHS
  mbox_set(MBOX_V_READY, 1);
#else
  mbox_set(MBOX_V_READY, FA_HOST_MAGIC);
#endif

  const uint64_t t_done = rdcycle();
  g_diag[0] = (uint32_t)t_enter;
  g_diag[1] = (uint32_t)t_done;
  g_diag[4] = (uint32_t)(t_done - t_enter);   // cycles spent actually pushing scale bytes
  diag_publish();

#ifdef FA_HOSTHS
  // ============================================================================================
  // PER-TILE SCALE REFILL (the thing that makes host offload work at STEADY STATE).
  //
  // A real streaming FA re-reads K_j / V_j (and therefore their MX scales) for every KV block, so
  // the honest steady-state experiment has the host re-push all 4,608 B every tile rather than
  // relying on FA_STEADY re-using one resident copy.  There are only TWO halves per scale SRAM and
  // BOTH are live in every tile (act: QK_A | P ; weight: K | V), so there is no spare buffer to
  // ping-pong into -- the host has to be told when a half has been drained.  That is what the
  // GPU->host progress words are for (written by mxgemm_compute_tile after its trailing
  // gemmini_fence, i.e. strictly after the mesh's last scale read of that gemm):
  //     GPU_QKDONE >= t  =>  tile t-1's QK is drained  =>  act half 1 + weight half 0 are dead
  //     GPU_PVDONE >= t  =>  tile t-1's PV is drained  =>  weight half 1 is dead
  // Every wait is bounded in host cycles so that a broken read path degrades to "runs anyway with
  // possibly-corrupt output + a nonzero timeout counter in the diagnostics", never a hung sim.
  //
  // ------------------------------------------------------------------------------------------
  // *** SECOND, HARDER CONSTRAINT (2026-07-26): THE MERGE NODE, NOT JUST THE SCALE SRAM. ***
  // Data liveness is not the only thing that serializes host and GPU here.  EVERY write into the
  // gemmini tile -- the scale SRAMs at +0x88000/+0x8a000 AND the ROCC command port at +0x84000 --
  // funnels through GemminiTile's single FlitMergeNode, which pair-merges consecutive 4-byte
  // beats (radiance/memory/FlitMergeNode.scala:36-37, `shouldMerge` requires size==2).  The GPU's
  // ROCC issue macro is 4-byte `sw.shared`es, so a host 8-byte `sd` (which bypasses the merge)
  // landing between the two halves of a GPU pair makes the node emit a MALFORMED request.  The
  // symptom is not silent corruption, it is a dead simulation:
  //     TLMonitor xbar_3 (RadianceCluster.scala:112, the host->cluster extReqXbar):
  //     "'D' channel contains improper response size", and the GPU's store never completes.
  // MEASURED: build `FULL_ATTN2 FA_STEADY FA_NT4 FA_NOSCALES FA_HOSTHS` (this refill loop, GPU
  // still issuing its own gemmini commands) asserts at exactly time 330,243,000 ps -- twice,
  // bit-identically, on two separately built images (hsh4, st2) -- after hanging at the end of
  // tile 1.  Adding FA_HOSTCFG makes it disappear, because then the GPU issues NO ROCC command
  // at all after tile 0.  That is the real reason the two features belong together.
  //
  // So the schedule below is built around GPU ROCC-QUIET WINDOWS, not around data liveness:
  //     [ PVDONE >= t ................. QK_READY = t+1 ]   GPU is in bar4/finalize/tile-t entry
  //                                                        and then BLOCKS on QK_READY -> quiet
  //     [ QKDONE >= t+1 ............... V_READY  = t+1 ]   GPU is in bar2/softmax, its QK
  //                                                        compute ROCC burst has drained -> quiet
  // Everything the host pushes must sit inside one of those two windows.  The old schedule
  // violated both: it started the K refill at QKDONE>=t (which is *during* tile t's own QK issue)
  // and pushed the V scales right after publishing QK_READY (i.e. straight into the GPU's QK
  // compute_tile ROCC burst).  Splitting the payload 1,280 B / 1,024 B across the two windows
  // keeps each burst comfortably inside it: window 1 is ~16k cycles wide (bar4 8.7k + finalize
  // 7.0k) against ~7.9k of stores, window 2 is ~13k wide (bar2 + softmax) against ~5.1k.
  // ------------------------------------------------------------------------------------------
  const uint64_t WAIT_LIMIT = 200000ull;   // ~1.7 tile periods; generous, still finite
  uint64_t wait_cycles = 0;
  uint32_t to_qk = 0, to_pv = 0, last_qk = 0, last_pv = 0;
#ifdef FA_HOSTCFG
  // Grab the GPU's tile-0 recording of the two 12-command gemmini streams.  Published at tile 0's
  // PV prefetch, which is well before the first thing the loop below needs (tile 1's QK).
  {
    const uint64_t t0 = rdcycle();
    for (;;) {
      const uint32_t a = mbox_get(0, MBOX_CFGREC), b = mbox_get(1, MBOX_CFGREC);
      if (((a < b) ? a : b) >= 2u) break;
      if (rdcycle() - t0 > WAIT_LIMIT) { g_diag[14] = 1; break; }   // capture timed out
    }
    wait_cycles += rdcycle() - t0;
    cfg_capture();
    g_diag[15] = (g_rec_n[0] << 8) | g_rec_n[1];   // expect 0x0c0c (12 commands each)
  }
#endif
  for (uint32_t t = 1; t < (uint32_t)HOST_NTILES; t++) {
    // ==== WINDOW 1: [ PVDONE >= t .. QK_READY = t+1 ] ==========================================
    // PVDONE >= t means tile t-1's PV matmul has drained, which is the GPU's LAST gemmini command
    // of tile t-1.  From here until the host publishes QK_READY = t+1 the GPU issues no ROCC at
    // all (it runs bar4 -> finalize_O -> loop -> tile t's prefetch, which blocks on QK_READY), so
    // the merge node is host-private and BOTH of these are legal:
    //   * WEIGHT half 0 (K scales): dead, tile t-1's QK drained long ago (PVDONE >= t => QKDONE >= t)
    //   * ACT half 1 (Q scales): dead, and the shared act port is quiet -- tile t-1's
    //     pack_scales_to_sfmem ran BEFORE its PV matmul, tile t's is a whole softmax+requant away.
    // The old code split these across the QKDONE>=t and PVDONE>=t gates to shorten the critical
    // section; that put the K refill inside tile t's own QK issue and deadlocked the fabric.  The
    // window is ~16k cycles wide against ~7.9k of stores, so nothing is lost by merging them.
    {
      const uint64_t t0 = rdcycle();
      for (;;) {
        const uint32_t a = mbox_get(0, MBOX_GPU_PVDONE), b = mbox_get(1, MBOX_GPU_PVDONE);
        last_pv = (a < b) ? a : b;
        if (last_pv >= t) break;
        if (rdcycle() - t0 > WAIT_LIMIT) { to_pv++; break; }
      }
      wait_cycles += rdcycle() - t0;
    }
    write_qk_a_scales();        // ACT half 1  (2 x 256 B)
    write_qk_b_scales();        // WEIGHT half 0 (2 x 1,024 B)
#ifdef FA_HOSTCFG
    cfg_replay(0);              // QK config + Q/K move-in: 12 ROCC commands, 36 `sd` per cluster
#endif
    asm volatile("fence" ::: "memory");
    mbox_set(MBOX_QK_READY, t + 1);
    // ---- BISECTION PROBE for the fabric deadlock (-DFA_HOSTHS_VEARLY, measurement only) --------
    // Puts exactly ONE of the three scale bursts back where the old (asserting) schedule had it --
    // the 2 x 1,024 B V-scale push, issued immediately after QK_READY, i.e. straight into tile t's
    // gemmini command + SF traffic -- and leaves the other two in their safe window.  If this
    // brings the assertion back while the fixed schedule is clean, the failure is attributable to
    // that single burst overlapping GPU traffic, not to "the schedule" in general.
    // NEVER build a reported result with this define.
#ifdef FA_HOSTHS_VEARLY
    write_v_scales();
#endif

    // ==== WINDOW 2: [ QKDONE >= t+1 .. V_READY = t+1 ] =========================================
    // QKDONE >= t+1 means tile t's OWN QK matmul has drained, i.e. the GPU's QK compute_tile ROCC
    // burst (CONFIG_SCALE_MEM + 3 x loop_ws) is over and the mesh has stopped reading the K
    // scales.  The GPU then runs bar2 + softmax (~13k cycles, no gemmini command at all) and only
    // blocks again on V_READY, so this is the second host-private window.
    //   * WEIGHT half 1 (V scales): dead since PVDONE >= t.
    //   * the PV command stream additionally MUST NOT be replayed before this point -- its move-in
    //     overwrites the B spad that tile t's QK matmul reads K out of.
    // Waiting here is not optional even without FA_HOSTCFG: writing the V scales right after
    // publishing QK_READY (what the old code did) drops them straight into the GPU's QK
    // compute_tile ROCC burst.
    {
      const uint64_t t0 = rdcycle();
      for (;;) {
        const uint32_t a = mbox_get(0, MBOX_GPU_QKDONE), b = mbox_get(1, MBOX_GPU_QKDONE);
        last_qk = (a < b) ? a : b;
        if (last_qk >= t + 1u) break;
        if (rdcycle() - t0 > WAIT_LIMIT) { to_qk++; break; }
      }
      wait_cycles += rdcycle() - t0;
    }
#ifndef FA_HOSTHS_VEARLY
    write_v_scales();           // WEIGHT half 1 (2 x 1,024 B)
#endif
#ifdef FA_HOSTCFG
    cfg_replay(1);              // PV config + V move-in
#endif
    asm volatile("fence" ::: "memory");
    mbox_set(MBOX_V_READY, t + 1);
  }
  {
    const uint64_t t_end = rdcycle();
    g_diag[2] = (uint32_t)HOST_NTILES;
    g_diag[3] = (uint32_t)t_end;
    g_diag[4] = (uint32_t)((t_end - t_enter) - wait_cycles);  // host BUSY cycles (stores only)
    g_diag[5] = (uint32_t)wait_cycles;                        // host IDLE waiting on the GPU
    g_diag[6] = to_qk; g_diag[7] = to_pv;
    g_diag[8] = last_qk; g_diag[9] = last_pv;
    diag_publish();
  }
#endif // FA_HOSTHS
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
