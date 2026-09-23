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

#define RAD_TAPEOUT_HOST 1
#include <rad_tapeout.h>
#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>
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
#if defined(FA_NOSCALES) || defined(FA_SP_HSF)
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
#define MBOX_GPU_PACKREQ  0x50   // GPU -> host: #tiles whose requant is done (-DFA_HOSTPACK)
#define MBOX_HOST_PACKED  0x60   // host -> GPU: #tiles whose P scales are in SF_MEM_A
#define MBOX_STAMPS       0x80   // diagnostic block, echoed to MARK_GMEM+0x300 by the GPU
#define FA_HOST_MAGIC     0x5CA1E5u

// ---- FA_STEADY tile count (must MATCH the GPU build; /tmp/fa_build.sh forwards the -D list) ----
#if defined(FA_STEADY) || defined(FA_SP)
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
// -DFA_HOST_ASEL0 (DIAGNOSTIC): put the Q scales in activation half **0** and have QK read half 0
// (mxgemm_core.hpp matches on the same define), i.e. the pre-split arrangement where the GPU's
// pack_scales_to_sfmem overwrites them with the softmax P scales every tile.  Under FA_STEADY every
// tile is identical, so "tile t's QK multiplied Q by tile t-1's P scales" is a FIXED wrong answer --
// which makes its Frobenius value a fingerprint.  Build this to find out whether a mystery
// steady-state failure is that specific bug or something else.  Never build a result with it.
#ifdef FA_HOST_ASEL0
#define QK_A_HALF_OFFSET 0ull
#else
#define QK_A_HALF_OFFSET SF_BUFFER_OFFSET
#endif
static inline void write_qk_a_scales(void) {   // ACT port  -- hazard above, PVDONE-gated
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    // QK activation (Q) scales -> activation half 1; the GPU owns half 0 for the P scales.
    sf_write(SF_MEM_A(cl) + QK_A_HALF_OFFSET, &QK_A_scales_row[0][0], QK_A_SCALE_BYTES);
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
//
// *** WHAT THE FSDB ACTUALLY SHOWS (2026-07-26) -- read this before changing the access width. ***
// Waveform of the failing run (`FA_NOSCALES FA_HOSTHS FA_HOSTCFG`, assertion at 179,473,000 ps,
// captured with hs_fsdb.sh), on the monitored host->cluster edge
// element_reset_domain_element.xbar_3.auto_anon_in_*:
//   * a_bits_size is 3 for every request on this edge -- the host is the only master here.
//   * the host's 8-byte Gets of cluster SMEM are answered CORRECTLY (opcode 1 AccessAckData,
//     d_bits_size 3) 165+ times in a row while it polls the mailbox every ~184 cycles.
//   * then ONE response comes back opcode 1 with d_bits_size = 2 (four bytes) and the monitor
//     kills the run.  It arrives already-wrong on auto_anon_out_0_d_bits_size, i.e. it is produced
//     downstream on the clcbus leg, not by the xbar.
//   * the GPU had already wedged 93k cycles earlier (last MARK 85,859, and the FSDB stops growing
//     there -- the design is genuinely frozen, not merely slow).
// So the width of the host's reads is NOT the bug: the same 8-byte Get works hundreds of times.
// The bug is that the GPU wedges first, with a 4-byte SF write pair stuck in FlitMergeNode
// (GemminiTile.scala:186-189).  A stuck non-last merge beat leaves the node's early-acknowledge
// path (`in.d.valid := out.d.valid || (shouldMerge && in.a.valid && !isLastReq)`, with
// `in.d.bits.size := log2Ceil(from)` == 2, FlitMergeNode.scala:88-100) permanently asserted, and
// that size-2 beat is what lands on the host's outstanding 8-byte read.  The assertion is the
// SYMPTOM; the deadlock of the GPU's scale-SRAM write against host traffic on the shared port is
// the cause.  Narrowing the host's reads to 4 bytes only makes A size match the bogus D size, so
// the run HANGS instead of failing loudly -- strictly worse.  Hence 8-byte reads stay the default
// (-DFA_HOST_RD4 exists only to demonstrate that masking).  The real fix is -DFA_HOSTPACK: take
// the GPU off the scale-SRAM port entirely.
//
// *** THE `fence` IS ALSO LOAD-BEARING. ***  Rocket retires stores into its store buffer
// and does NOT wait for the TL response, but it blocks on an uncached load.  So without this fence
// a host `sd` into the cluster can still be in flight when the next host `ld` into the cluster is
// issued, and the two share the host->cluster control path
//     extReqXbar -> TLSourceShrinker(controlInFlights) -> TLFragmenter(8, 128, alwaysMin) -> clcbus
// (RadianceCluster.scala:109-112) whose source-id remapping and beat accounting are what decide
// the size the response is reported with.  Overlap them and a response comes back with the wrong
// size, which kills the run at the extReqXbar monitor:
//     "'D' channel contains improper response size" (RadianceCluster.scala:112) + a hung store.
// EVIDENCE: every configuration that reads the mailbox at all (FA_HOSTHS, FA_HOSTCFG) has hit that
// assertion in some build -- `FA_NOSCALES FA_HOSTHS` deterministically at 330,243,000 ps on two
// images x two seeds, and `+FA_HOSTCFG` at 179,473,000 ps -- while every read-free build
// (FA_NOSCALES alone, which only ever stores) has never once hit it in ~20 runs.  A store-to-load
// fence is the difference between those two groups, and it costs a couple of cycles on a path that
// is already 20 cycles per access.
static inline uint32_t mbox_get(unsigned cl, unsigned off) {
  asm volatile("fence" ::: "memory");
#ifdef FA_HOST_RD4
  return *(volatile uint32_t *)(HOST_MBOX(cl) + off);   // masks the assertion into a hang -- see above
#else
  return (uint32_t)*(volatile uint64_t *)(HOST_MBOX(cl) + off);
#endif
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

#ifdef FA_HOSTPACK
// ============================================================================================
// HOST-PACKED RUNTIME P SCALES  (-DFA_HOSTPACK; needs FA_HOSTHS, and the GPU must be built
// -DFA_NOPACK so pack_scales_to_sfmem is compiled out)
// ============================================================================================
// This is NOT primarily a performance item -- it is what makes a PER-TILE host scale refill
// structurally legal on this RTL.  The scale SRAM sits behind FlitMergeNode(from=4, to=8)
// (GemminiTile.scala:186-189) because the write manager only accepts size==3 requests
// (GemminiTile.scala:286) while an RV32 Muon lane can only emit 4-byte stores (store64_shared in
// lib/include/mu_intrinsics.h:27-32 is literally two 4-byte stores).  So the merge node pairs the
// GPU's 4-byte writes, and its pairing state -- the beat counter, the merged-request register, and
// the per-source `wasMerged` bit that decides the size a response is reported with
// (FlitMergeNode.scala:33-100) -- is GLOBAL to the port and shared with the host, whose 8-byte
// stores take the non-merging path.  Mixing the two requestors is what kills runs (see the
// FA_HOSTHS schedule comment).  Remove the GPU as an SF requestor entirely and every request into
// that node is an 8-byte host store that never merges, so there is no shared state left to corrupt.
//
// The pack itself is trivial: requant leaves 512 E8M0 scales in SCALE_SMEM as one scale per 32-bit
// word (flash_mx_impl.hpp:1074-1082), and SF_MEM_A wants them as 128 contiguous words, 4 scales per
// word, low byte first.  Both clusters run the same tile redundantly, so ONE cluster's SCALE_SMEM
// is read and the result is written to both.
//   read  : 256 8-byte loads  of cluster-0 SMEM   (~20.2 cyc each, measured -> ~5.2k)
//   write : 2 x 64 8-byte stores into SF_MEM_A     (~15 cyc each     -> ~1.9k)
// against the GPU's measured 8,234 cyc for pack+bar3, so it is roughly break-even on the critical
// path -- the point is legality, not speed.  Report it as such.
// SCALE_SMEM_HOST is defined above with the MMIO probe.
#define P_SCALE_WORDS       128                              // (FA_SK/32)*FA_SQ / 4 = 512/4

static void host_pack_p_scales(void) {
  static uint32_t buf[P_SCALE_WORDS];
  asm volatile("fence" ::: "memory");     // no host store in flight before these loads
  // 512 scale words in (one E8M0 byte in the low byte of each), 128 packed words out, 4 per word.
  // 8-byte loads: two source words per load, which halves the number of host accesses.
  const volatile uint64_t *src = (const volatile uint64_t *)SCALE_SMEM_HOST(0);
  for (unsigned w = 0; w < P_SCALE_WORDS; w++) {
    const uint64_t a = src[2u * w + 0u];      // scale words 4w+0 (low half), 4w+1 (high half)
    const uint64_t b = src[2u * w + 1u];      // scale words 4w+2, 4w+3
    buf[w] = ((uint32_t)(a & 0xffu))
           | ((uint32_t)((a >> 32) & 0xffu) << 8)
           | ((uint32_t)(b & 0xffu) << 16)
           | ((uint32_t)((b >> 32) & 0xffu) << 24);
  }
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    sf_write(SF_MEM_A(cl), buf, 4u * P_SCALE_WORDS);   // ACT half 0 -- exactly where pack wrote
}

// Wait for the GPU to say "tile `seq-1`'s requant is done, SCALE_SMEM is valid", pack, and answer.
// The GPU publishes the request at the top of its PV mxgemm_compute_tile and blocks on the reply,
// so this is strictly serial with the GPU by construction -- there is nothing to overlap.
static void host_service_pack(uint32_t seq, uint64_t wait_limit, uint64_t *wait_cycles,
                             uint32_t *timeouts) {
  const uint64_t t0 = rdcycle();
  for (;;) {
    const uint32_t a = mbox_get(0, MBOX_GPU_PACKREQ), b = mbox_get(1, MBOX_GPU_PACKREQ);
    if (((a < b) ? a : b) >= seq) break;
    if (rdcycle() - t0 > wait_limit) { (*timeouts)++; break; }
  }
  *wait_cycles += rdcycle() - t0;
  host_pack_p_scales();
  asm volatile("fence" ::: "memory");
  mbox_set(MBOX_HOST_PACKED, seq);
}
#endif // FA_HOSTPACK

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
  asm volatile("fence" ::: "memory");   // no host store may be in flight -- see mbox_get's note
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

// ============================================================================================
// FA_SP_HSF -- HOST-OWNED MX SCALE SRAM FOR THE *PIPELINED* (FA_SP) KERNEL
// ============================================================================================
// This is the composition the fourth pass never tried: the host scale offload on top of the
// FA_SP_OPV/OPVQ software pipeline rather than on the sequential FULL_ATTN2 body.  It is a
// different, and much safer, proposition than -DFA_HOSTHS/-DFA_HOSTPACK were:
//
// (1) THE GPU IS NO LONGER AN SF REQUESTOR AT ALL.  The FA_SP_HSF GPU build issues ZERO writes to
//     either scale port -- not the K/V prologue scales, not the per-tile Q scales, and not the
//     per-tile packed P scales.  That removes, by construction, the one hazard the earlier attempts
//     kept tripping over: ScaleFactorMem's per-port 2-beat pairing register is SHARED across the
//     double-buffer halves, so a host `sd` landing between two of the GPU's FlitMergeNode-merged
//     4-byte beats forms a pair from one GPU beat and one host beat and writes a 16 B SRAM row to
//     the wrong address.  With one requestor there is no interleaving to form.
// (2) THE HAND-OFF SITS IN A STAGE WITH NO MESH ACTIVITY.  Under FA_SP_OPVQ the iteration is
//         A: PV(t-1) mesh || softmax(t)   B: acc->O   C1: pass A   C2: finalize(t-1)   D: QK(t+1)
//         mesh || convert(t)   E: acc->S
//     and stage C2 is the one stage where the mesh is idle end to end (PV drained in B, QK not
//     issued until D).  The GPU publishes PACKREQ at the top of C2 and blocks on PACKED at the top
//     of D, so the host's ~5.4k cycles of scale traffic land entirely inside C2's ~9.6k of
//     finalize, with no mesh scale READ anywhere near them.  That is what the fourth pass's 7-of-12
//     failure was missing -- there the host wrote scales while the GPU's consuming matmul was live.
// (3) IT PAYS FOR ITSELF TWICE OVER, and not in the way the sequential measurement suggested.  On
//     FA_SP the per-tile GPU scale traffic is only 192 words (64 Q + 128 P), not 704, so the direct
//     saving is small.  The win is that those 192 words are the ONLY thing warp 0 does in stages C2
//     and D -- ~4.2k and ~8.3k of strictly serial, unparallelisable SF stores -- and removing them
//     lets warp 0 join the SIX-warp finalize and the SIX-warp convert.  Projected -1.6k on C2 and
//     -1.6k on D.
// (4) EVERY HOST POLL IS PRECEDED BY A store-to-load `fence`, which mbox_get already does.
//
// MAILBOX ADDRESS: the FA_SP SMEM map is completely different from the sequential one and the old
// HOST_MBOX at device 0x17F00 lands INSIDE FA_SP's Q scratchpad region (0x16000..0x18000), where
// the Q move-in DMA would overwrite it every tile.  FA_SP_HSF uses 0x14D00 instead -- see the long
// note at FA_SP_MBOX in flash_attention_mx.cpp for why 0x15F00, the obvious choice, is also taken.
// It must match FA_SP_MBOX in flash_attention_mx.cpp.
#ifdef FA_SP_HSF
// 0x14D00, NOT 0x15F00: FA_SM_2P's pb buffer (0x15680, stride 2*NT+1 = 33 halfwords) writes 0x15F10
// and 0x15F20 -- the PACKREQ and PACKED words.  See the long note at FA_SP_MBOX in the kernel.
#ifdef FA_SP_HSF_PB
// printBuf (RadianceCluster.scala:96-103): an unused 512 B TLRAM at baseAddr + peripheralAddrOffset,
// beatBytes 8, atomics = true, behind the SAME clcbus the SMEM leg hangs off.  See the long note at
// FA_SP_MBOX in flash_attention_mx.cpp for why this is the discriminator and not just a workaround.
#define SP_MBOX(cl)        (CLUSTER_BASE(cl) + 0x80000ull)
#else
#define SP_MBOX(cl)        (CLUSTER_BASE(cl) + 0x14D00ull)
#endif
#define SPM_READY          0x00     // host -> GPU: prologue K/V/Q scales are in the SRAM
#define SPM_PACKREQ        0x10     // GPU  -> host: #tiles whose pass A has published SCALE_SMEM
#define SPM_PACKED         0x20     // host -> GPU: #tiles whose Q + P scales are in the SRAM
#define SPM_DIAG           0x40     // host -> GPU: [0]=wait cycles, [1]=(timeouts<<32)|ntiles
#define SP_P_SCALE_WORDS   128      // (FA_SK/32)*FA_SQ / 4

static inline void sp_mbox_set(unsigned off, uint32_t v) {
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    *(volatile uint64_t *)(SP_MBOX(cl) + off) = (uint64_t)v;
}
static inline uint32_t sp_mbox_get(unsigned cl, unsigned off) {
  asm volatile("fence" ::: "memory");          // store-to-load: never poll behind our own stores
#ifdef FA_SP_HSF_RD4
  // ---- EXPERIMENT: 4-BYTE (size=2) Get instead of 8-byte (size=3). --------------------------
  // The host-read failure is TLMonitor xbar_3 "'D' channel contains improper response size", and
  // xbar_3 = extReqXbar is UPSTREAM of everything that could resize a beat.  Between it and the
  // SMEM word SRAMs the request passes TLSourceShrinker, TLFragmenter(8,128,alwaysMin=true)
  // (RadianceCluster.scala:107), TLFragmenter(wordSize=4,128) for extClients
  // (RadianceSharedMemComponents.scala:78) and RWSplitterNode -- whose D path does
  //     arb_in.bits.size := trim(r.bits.size, 1 << in_node.d.bits.size.getWidth)  // FIXME: check
  //                                                                               // truncation
  // and whose generated inbound size field really is [1:0] against a [3:0] outbound.
  // The SMEM subbanks are wordSize = 4 BYTES, so a 4-byte Get is the one width that needs no
  // fragmentation and no reassembly anywhere on this path.  If the mechanism is width-related this
  // just works; if it still fails, width is exonerated and the source/size TRIMMING is the suspect.
  // NOTE the asymmetry that makes this worth testing rather than assuming: host 4-byte WRITES into
  // the cluster are silently DROPPED (measured), but that is the write path -- PutPartial with a
  // sub-beat mask -- and says nothing about Gets.
  return *(volatile uint32_t *)(SP_MBOX(cl) + off);
#else
  return (uint32_t)*(volatile uint64_t *)(SP_MBOX(cl) + off);
#endif
}

// *** THROTTLE THE POLL.  A TIGHT HOST POLL LOOP KILLS THE FABRIC. ***  MEASURED: the maximal stack
// (FA_SP_HSF + FA_SM_2P + FA_SM_2PBM + FA_SP_SMBMAX + FA_SM_2PRAW) $finished at 164,937,000 ps
// (~82.5k cycles) with
//     TLMonitor xbar_3 (RadianceCluster.scala:112, the host->cluster extReqXbar):
//     "'D' channel contains improper response size"
// and the where matters: 82.5k is inside the SOFTMAX of iteration 0, i.e. a window in which the host
// has ALREADY finished every scale write and is doing nothing but reading the mailbox.  So this is
// not the host-vs-GPU scale-port hazard that killed FA_HOSTCFG -- with FA_SP_HSF the GPU writes no
// scale word at all -- it is the POLL ITSELF: an unthrottled stream of uncached 8-byte Gets into
// cluster SMEM, concurrent with a softmax that is saturating that SMEM, and the shared-SMEM slave
// path answers one of them with the wrong size.  The same build minus FA_SM_2PBM (a lighter softmax)
// got 40k cycles further without tripping it, which is what "timing window" looks like.
// THE FIX IS FREE: the host has nothing else to do, and the windows it is waiting for are ~9,500
// cycles wide, so polling every ~256 cycles instead of every ~20 cuts the request rate by an order
// of magnitude and costs nothing measurable.  Poll cluster 0 first and only look at cluster 1 once
// cluster 0 is satisfied, which halves the requests again.
static inline void sp_backoff(void) {
  for (int i = 0; i < 256; i++) asm volatile("nop");
}
// Both clusters run the same tile redundantly, so "both have reached seq" is the condition; testing
// them in sequence rather than every iteration keeps the Get rate down.
static inline int sp_both_reached(unsigned off, uint32_t seq) {
  if (sp_mbox_get(0, off) < seq) return 0;
  return sp_mbox_get(1, off) >= seq;
}

// Pack the 512 E8M0 scale words requant pass A left in SCALE_SMEM (one scale in the low byte of
// each 32-bit word) into the 128 contiguous words SF_MEM_A wants (4 scales per word, low byte
// first), and push them to ACT half 0 on both clusters.  Both clusters compute the same tile
// redundantly, so one cluster's SCALE_SMEM is authoritative.
static void sp_pack_p_scales(void) {
  static uint32_t buf[SP_P_SCALE_WORDS];
  asm volatile("fence" ::: "memory");
  const volatile uint64_t *src = (const volatile uint64_t *)SCALE_SMEM_HOST(0);
  for (unsigned w = 0; w < SP_P_SCALE_WORDS; w++) {
    const uint64_t a = src[2u * w + 0u];       // scale words 4w+0 (low half), 4w+1 (high half)
    const uint64_t b = src[2u * w + 1u];       // scale words 4w+2, 4w+3
    buf[w] = ((uint32_t)(a & 0xffu))
           | ((uint32_t)((a >> 32) & 0xffu) << 8)
           | ((uint32_t)(b & 0xffu) << 16)
           | ((uint32_t)((b >> 32) & 0xffu) << 24);
  }
  for (int cl = 0; cl < NUM_CLUSTERS; cl++)
    sf_write(SF_MEM_A(cl), buf, 4u * SP_P_SCALE_WORDS);            // ACT half 0
}
#endif // FA_SP_HSF


// ===========================================================================================
// FA_FPGA -- run this kernel on the U250 FireSim board and score it ON-CHIP.
//
// Three things force a separate path from the RTL-sim flow:
//  (1) fa_verify_tiles.py scores a SIM TRACE.  There is no trace on the FPGA, so the score has
//      to be computed by the rv64 host from the O window itself.
//  (2) RAD_HOST_GPU_ALL_FINISHED IS UNUSABLE on this board -- measured: it reads 1 before the
//      GPU has started and 0 after it has run (fpga-bringup hw_10..hw_14).  The stock
//      while(!finished) spin below would therefore never exit.  Completion is detected from the
//      DATA instead: poison the O window pre-launch and wait for it to be fully overwritten.
//  (3) HTIF costs ~40M target cycles PER write() syscall on this board, so the whole report is
//      formatted into one buffer and emitted with a single write.
//
// O is at 0x40040000..0x40043fff = 4096 words = 8192 bf16 halfwords, in cluster space that the
// host can read directly (the same system-physical space as GPU_ADDR_OR_MMIO), NOT the
// 0x1_xxxx_xxxx GPU-DRAM alias.  fa_verify_out.py compares the window in ADDRESS ORDER after a
// reshape, so a flat comparison here is exactly equivalent -- reshape does not repair pairing.
// ===========================================================================================
#ifdef FA_FPGA
#include "fa_golden_O.h"
#include "fa_golden_O_flash.h"   // the reference the kernel actually implements
#ifndef FAF_O_BASE
#define FAF_O_BASE   0x40040000ull
#endif
#define FAF_O_WORDS  4096u
#define FAF_POISON   0xDEADBEEFu
#ifndef FA_FPGA_SETTLE
#define FA_FPGA_SETTLE 20000000ull   /* cycles to let the L0d writeback drain after last store */
#endif
#ifndef FA_FPGA_WAIT
#define FA_FPGA_WAIT 1200000000ull   /* hard cap on the completion wait */
#endif

static char faf_buf[4096];
static unsigned faf_len = 0;
static void faf_s(const char *p) { while (*p && faf_len < sizeof(faf_buf) - 1) faf_buf[faf_len++] = *p++; }
static void faf_u(uint64_t v) {
  char t[24]; int n = 0; if (!v) t[n++] = '0';
  while (v) { t[n++] = (char)('0' + (v % 10)); v /= 10; }
  while (n-- && faf_len < sizeof(faf_buf) - 1) faf_buf[faf_len++] = t[n];
}
static void faf_x(uint32_t v) {
  static const char h[] = "0123456789abcdef";
  for (int i = 7; i >= 0; i--) if (faf_len < sizeof(faf_buf) - 1) faf_buf[faf_len++] = h[(v >> (i*4)) & 0xf];
}
static void faf_flush(void) { write(1, faf_buf, faf_len); faf_len = 0; }
static inline uint64_t faf_cyc(void) { uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
// NO FLOATING POINT IN THIS HOST.  WithNSmallCores gives the host Rocket
// RocketCoreParams(useVM = false, fpu = None)  (rocket-chip rocket/Configs.scala:140), so
// mstatus.FS is hardwired 0 and EVERY FP instruction is illegal -- including the fsd/fld spills
// the compiler emits in a function's PROLOGUE when the body touches a double.  Those trap, the
// bootrom handler returns to them, and they retry forever: a constant ~1790/1210 alternation
// between 0x80000xxx and 0x100xx that reads exactly like a hang.  No csrs mstatus can fix it --
// there is nothing to enable.  Score in INTEGER arithmetic and assert zero FP opcodes in the build.

// ---- MX C-SCALE OVERRUN DETECTOR (host half) ------------------------------------------------
// Pairs with FA_SCALE_GUARD in mxgemm_core.hpp, which places a 64-word 0x5A5AC0DE sentinel
// immediately after C_scale_factors inside one struct so the two are guaranteed adjacent.
// FA_GUARD_ADDR is the DEVICE address of that sentinel, resolved from the rv32 ELF at build
// time (build.sh) rather than assumed, and device addr X is visible to the host at X|0x1_00000000.
//
// Read twice.  BEFORE launch the sentinel must be intact: that proves it is inside a LOAD
// segment and actually reached device memory, which is not automatic here -- a zero-initialised
// static lands in an orphaned .bss that the loader never writes.  AFTER the run a damaged
// sentinel means Gemmini wrote past the end of C_scale_factors.
//
// ASYMMETRY, state it before reading any result: detection is conclusive, absence is not.
// The sentinel lives in DRAM and every C_scale_factors use hands its address to the Gemmini
// mvout DMA, so an overrun should land in the memory the host reads -- but a clean sentinel
// only shows nothing reached DRAM, not that nothing was written.
#if defined(FA_SCALE_GUARD) && defined(FA_GUARD_ADDR)
#ifndef FA_GUARD_WORD
#define FA_GUARD_WORD   0x5A5AC0DEu
#endif
#define FA_GUARD_WORDS  64u
static uint32_t fa_guard_check(const char *when) {
  volatile uint32_t *g = (volatile uint32_t *)(0x100000000ull | (uint64_t)(FA_GUARD_ADDR));
  uint32_t bad = 0; uint32_t first = 0xffffffffu;
  for (uint32_t i = 0; i < FA_GUARD_WORDS; i++) {
    if (g[i] != FA_GUARD_WORD) { bad++; if (first == 0xffffffffu) first = i; }
  }
  faf_s("FA_FPGA: scale_guard "); faf_s(when);
  faf_s(" addr=0x"); faf_x((uint32_t)(FA_GUARD_ADDR));
  faf_s(" bad="); faf_u(bad); faf_s("/"); faf_u(FA_GUARD_WORDS);
  faf_s(" first_bad="); faf_u(first);
  faf_s(" w[0,1,63]="); faf_x(g[0]); faf_s(" "); faf_x(g[1]); faf_s(" "); faf_x(g[FA_GUARD_WORDS-1]);
  faf_s("\n"); faf_flush();
  return bad;
}
#endif
#endif // FA_FPGA

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

#ifdef FA_SP_HSF
  // ---- FA_SP_HSF: the host owns the whole MX scale SRAM for the FA_SP pipeline. --------------
  sp_mbox_set(SPM_READY, 0);
  sp_mbox_set(SPM_PACKREQ, 0);
  sp_mbox_set(SPM_PACKED, 0);
  asm volatile("fence" ::: "memory");
  // Prologue: everything the GPU used to write in its own prologue plus tile 0's Q scales.
  // K -> weight half 0, V -> weight half 1, Q -> act half 1.  The GPU blocks on SPM_READY before
  // its first matmul, so there is no race with the mesh.
  write_qk_b_scales();                                     // K  -> WEIGHT half 0
  write_v_scales();                                        // V  -> WEIGHT half 1
  write_qk_a_scales();                                     // Q  -> ACT    half 1
  asm volatile("fence" ::: "memory");
  sp_mbox_set(SPM_READY, FA_HOST_MAGIC);
  {
    const uint64_t WAIT_LIMIT = 400000ull;                 // finite: a broken poll must not hang
    uint64_t waited = 0; uint32_t timeouts = 0;
    for (uint32_t t = 1; t <= (uint32_t)HOST_NTILES; t++) {
      const uint64_t t0 = rdcycle();
      for (;;) {
        if (sp_both_reached(SPM_PACKREQ, t)) break;
        if (rdcycle() - t0 > WAIT_LIMIT) { timeouts++; break; }
        sp_backoff();
      }
      waited += rdcycle() - t0;
      // Both halves of the act port, in the stage where the mesh is idle: the runtime P scales for
      // tile t-1 (act half 0, read by PV(t-1) in stage A of the next iteration) and the Q scales
      // for the next tile (act half 1, read by QK, issued in stage D right after the GPU sees
      // SPM_PACKED).  The Q push is redundant in this harness -- Q is loop-invariant -- but it is
      // kept so the per-tile cost accounting stays honest against the GPU build it replaces.
      sp_pack_p_scales();
      write_qk_a_scales();
      asm volatile("fence" ::: "memory");
      sp_mbox_set(SPM_PACKED, t);
    }
    // Report through the FA_SP mailbox, NOT diag_publish(): that writes HOST_MBOX+0x80 = device
    // 0x17F80, which is inside Q's scratchpad under the FA_SP map.
    for (int cl = 0; cl < NUM_CLUSTERS; cl++) {
      *(volatile uint64_t *)(SP_MBOX(cl) + SPM_DIAG + 0) = waited;
      *(volatile uint64_t *)(SP_MBOX(cl) + SPM_DIAG + 8) =
          ((uint64_t)timeouts << 32) | (uint64_t)(uint32_t)HOST_NTILES;
    }
  }
#endif

#ifndef FA_SP_HSF
  // ---- SEQUENTIAL-KERNEL PREFILL.  Skipped entirely under FA_SP_HSF: its mailbox lives at device
  // 0x17F00, which in the FA_SP SMEM map is INSIDE Q's scratchpad (0x16000..0x18000), so every
  // mbox_set/diag_publish here writes into Q.  The pre-zeroing at the top of main() is harmless
  // (~2.3k cyc, long before the GPU's Q move-in overwrites it) but this tail is not: it runs while
  // the GPU is still in its last tiles.  FA_SP_HSF has already done all of the scale work above and
  // reports through SP_MBOX + SPM_DIAG, so there is nothing here it needs.
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

#endif  // !FA_SP_HSF
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
#ifdef FA_HOSTPACK
  // TILE 0's P scales.  With the GPU built -DFA_NOPACK there is no GPU-side packer at all, so tile
  // 0 needs servicing too -- and it happens BEFORE the refill loop's first iteration, because the
  // loop's window-1 gate (PVDONE >= 1) is only reached after tile 0's PV, which needs these scales.
  host_service_pack(1, WAIT_LIMIT, &wait_cycles, &to_pv);
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
#ifdef FA_HOSTPACK
    // ==== WINDOW 3: [ PACKREQ >= t+1 .. PACKED = t+1 ] =========================================
    // The GPU asks for tile t's P scales at the top of its PV compute_tile and blocks on the reply,
    // so it is provably idle here (and, being built -DFA_NOPACK, it never writes SF at all).
    host_service_pack(t + 1, WAIT_LIMIT, &wait_cycles, &to_pv);
#endif
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

#ifndef FA_FPGA
  tohost = 0;
  *tocpu = tohost;
#else
  // DO NOT prime the tocpu char-proxy on the FPGA path.  tocpu is
  // (volatile uint64_t *)0x1_0001_0000 = GPU-DRAM-alias of DEVICE 0x1_0000, far below the
  // 0x1F00_0000 region measured good, and this store is the ONLY thing the FA host executes
  // before my block that the (working) bring-up host does not.  Localized by preprocessing
  // main() with the real flag set: everything else in the preamble is gated on
  // FA_HOST_SCALE_PREFILL and compiles out.  Symptom it produced: a constant ~1790/1210
  // alternation between 0x80000xxx and the bootrom trap vector -- a faulting store the handler
  // returns to, retried forever.  The proxy exists for SYNC_GPU, which the FPGA path never uses.
#endif

#ifdef FA_FPGA
  // (No FPU on this core -- see the note at the top of the FA_FPGA block.)
  faf_s("FA_FPGA: S0 entered main-tail\n"); faf_flush();
  volatile uint32_t *Ow = (volatile uint32_t *)FAF_O_BASE;
  // Poison the O window so "GPU never wrote" is distinguishable from "GPU wrote zeros".
  // O's value range is [-0.697,0.936]; neither 0xdead nor 0xbeef is a bf16 code in that range,
  // so a poison word can never be a legitimate result.
  faf_s("FA_FPGA: S1 about to poison O window\n"); faf_flush();
  for (uint32_t i = 0; i < FAF_O_WORDS; i++) Ow[i] = FAF_POISON;
  { volatile uint32_t *mk = (volatile uint32_t *)(FAF_O_BASE + 0x100000ull);
    for (int k = 0; k < 16; k++) mk[k] = FAF_POISON; }
  faf_s("FA_FPGA: S2 poison loop done\n"); faf_flush();
  asm volatile("fence" ::: "memory");
  const uint32_t poison_rb = Ow[0];
  faf_s("FA_FPGA: S3 readback ok rb="); faf_x(poison_rb); faf_s("\n"); faf_flush();          // proves the host->cluster write path before launch

  // The GPU needs a soft-reset EDGE to launch: measured both ways on this board -- with the
  // assert/release it runs, without it the cores are already latched finished and never execute.
#if defined(FA_SCALE_GUARD) && defined(FA_GUARD_ADDR)
  // Gate: if this reports bad != 0 the sentinel never loaded, and the post-run read means
  // nothing.  Do not interpret the post-run number unless this one is 0.
  fa_guard_check("pre_launch ");
#endif
#ifdef FA_DRAM_NOISE
  // ---- EMULATE STALE DRAM.  The metasim starts with ZEROED DRAM; the board does not, and FA's
  // .bss is NOBITS so nothing initialises it.  Readelf on this image:
  //     LOAD 0x10001000  FileSiz 0x048  MemSiz 0x8c0   RW
  // so bytes 0x10001048..0x100018c0 are never written by the loader, and that window contains
  // BOTH orphaned statics -- C_scale_factors (0x10001060) and schedule_context (0x10001880).
  // That 2168-byte window is the ONLY confirmed difference between the metasim and the board, so
  // filling it with a seeded LCG makes the two comparable in the one input that differs.
  // Written by the HOST before the GPU is released, so it works identically in both.
  {
    const uint64_t lo = 0x100000000ull + 0x10001048ull, hi = 0x100000000ull + 0x100018c0ull;
    uint32_t x = (uint32_t)(FA_DRAM_NOISE);
    for (uint64_t a = lo; a < hi; a += 4) {
      x = x * 1664525u + 1013904223u;          // Numerical Recipes LCG
      *(volatile uint32_t *)a = x;
    }
    asm volatile("fence" ::: "memory");
    faf_s("FA_FPGA: dram_noise seed="); faf_x((uint32_t)(FA_DRAM_NOISE));
    faf_s(" window=0x10001048..0x100018c0 readback=");
    faf_x(*(volatile uint32_t *)(0x100000000ull + 0x10001060ull));   // C_scale_factors[0]
    faf_s("\n"); faf_flush();
  }
#endif
  faf_s("FA_FPGA: S4 about to touch GPU_RESET\n"); faf_flush();
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
  { const uint64_t u = faf_cyc() + 200000ull; while (faf_cyc() < u) asm volatile("" ::: "memory"); }
  const uint64_t t_launch = faf_cyc();
#ifndef FA_FPGA_NOLAUNCH
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);
#else
  /* DELIBERATE GATE-BREAK: never release the GPU.  The scorer MUST report
     still_poison=4096 and VERDICT=INVALID.  If this build ever reports PASS, the gate is
     measuring nothing -- which is the failure mode this campaign keeps hitting. */
#endif

#ifdef FA_FPGA_IMGPROBE
  // Did FA's OWN image reach the device intact?  The padded bring-up kernel passed this check at
  // 128 KB depth, which killed the truncation hypothesis -- but that kernel is essentially ONE
  // contiguous region, so it could never have exposed a PER-SEGMENT loading bug.  FA has five
  // rv32 segments with gaps (0x10000000/1000/2000/5000/6000) and a .bss, so it needs its own
  // check.  Read BEFORE releasing the GPU so nothing the kernel does can affect the result.
  // Device addr X is visible to the host at X | 0x1_0000_0000.
  {
    const uint64_t probes[10] = {
      0x110000000ull,  // seg0 start (entry)
      0x110000040ull,  // seg0 interior
      0x110001000ull,  // seg1 start
      0x110002000ull,  // seg2 start (.text)
      0x110004d00ull,  // seg2 near end
      0x110005000ull,  // seg3 start
      0x110006000ull,  // seg4 start (big rodata)
      0x110010000ull,  // seg4 interior
      0x110018000ull,  // seg4 near end
      0x110018940ull   // seg4 last words (top = 0x110018960)
    };
    for (int i = 0; i < 10; i++) {
      volatile uint32_t *q = (volatile uint32_t *)probes[i];
      faf_s("IMG @0x"); faf_x((uint32_t)(probes[i] & 0xffffffffu)); faf_s(" =");
      for (int k = 0; k < 4; k++) { faf_s(" "); faf_x(q[k]); }
      faf_s("\n");
    }
    faf_flush();
  }
#endif
  // LAUNCH CONFIRMATION -- without this, every conclusion about where the kernel stalls rests on
  // an unverified precondition.  all_finished reads 1 while the GPU is idle/never-started and 0
  // once it is running, so a transition to 0 after the release is the only evidence the cores
  // actually left reset.  The bring-up host has had this since hw_12; the FA host did not, so no
  // FA run so far has shown that its GPU was ever released.
  uint32_t gpu_started = 0;
  for (uint64_t w = 0; w < 2000000ull; w++) {
    if (READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED) == 0) { gpu_started = 1; break; }
  }
  faf_s("FA_FPGA: gpu_started="); faf_u(gpu_started);
  faf_s("  cores_at_release =");
  for (int g = 0; g < 4; g++) { faf_s(" "); faf_u(READ_MMIO_32(RAD_HOST_GPU_CORES + 4 * g)); }
  faf_s("\n"); faf_flush();

  // Completion by DATA, not by all_finished: wait until four spread sentinels are all non-poison.
  faf_s("FA_FPGA: S5 launched, entering completion poll\n"); faf_flush();
  uint32_t done = 0; uint64_t waited = 0;
  while (!done && waited < FA_FPGA_WAIT) {
    done = (Ow[0] != FAF_POISON) && (Ow[1365] != FAF_POISON)
        && (Ow[2730] != FAF_POISON) && (Ow[FAF_O_WORDS-1] != FAF_POISON);
    waited = faf_cyc() - t_launch;
  }
  const uint64_t t_done = faf_cyc();
  // let the L0d writeback drain: on this board readback is bit-exact only after a long settle
  { const uint64_t u = faf_cyc() + FA_FPGA_SETTLE; while (faf_cyc() < u) asm volatile("" ::: "memory"); }
  asm volatile("fence" ::: "memory");

  // Frobenius over the whole window, exactly fa_verify_out.py's metric.
#ifdef FA_EPILOGUE
  // EPILOGUE HANDSHAKE.  The GPU never asserts finished by design -- do not wait on it.
  {
    const int nc = 2;
    const int dn = rad_host_wait_epilogue(nc, 20000000ull);
    faf_s("FA_FPGA: epilogue done_cores="); faf_u(dn); faf_s(" of "); faf_u(nc);
    for (int c = 0; c < nc; c++) {
      faf_s(" done["); faf_u(c); faf_s("]=");
      faf_x((uint32_t)*(volatile unsigned long long *)RAD_PB_HOST(0, RAD_PB_EPI_CORE + c));
      faf_s(" trace["); faf_u(c); faf_s("]=");
      faf_x((uint32_t)*(volatile unsigned long long *)RAD_PB_HOST(0, RAD_PB_EPI_TRACE + c));
    }
    faf_s(" preO=");
    for (int i = 32; i <= 55; i++) { faf_x((uint32_t)*(volatile unsigned long long *)RAD_PB_HOST(0,i)); faf_s(","); }
    faf_s(" postO=");
    faf_x(((const volatile uint32_t *)0x11F000000ull)[3104]); faf_s(",");
    faf_x(((const volatile uint32_t *)0x11F000000ull)[3160]);
    faf_s(" warpstop=");
    for (int w = 0; w < 8; w++)
      faf_x((uint32_t)*(volatile unsigned long long *)RAD_PB_HOST(0, RAD_PB_EPI_WARP + w) != 0 ? 1u : 0u);
    faf_s("\n"); faf_flush();
    if (dn == nc) { rad_host_gpu_soft_reset(); faf_s("FA_FPGA: soft reset asserted\n"); }
    else          { faf_s("FA_FPGA: EPILOGUE TIMEOUT\n"); }
    faf_flush();
  }
#endif
  faf_s("FA_FPGA: S6 poll exited, scoring\n"); faf_flush();
  uint32_t still_poison = 0, exact = 0;
  uint64_t sumabs = 0;            // sum over halfwords of |bf16 code difference|
  uint32_t worst = 0;             // largest single code difference
  for (uint32_t w = 0; w < FAF_O_WORDS; w++) {
    const uint32_t v = Ow[w];
    if (v == FAF_POISON) still_poison++;
    for (int h = 0; h < 2; h++) {
      const uint16_t got = (uint16_t)(h ? (v >> 16) : (v & 0xffffu));
      const uint16_t exp = FA_GOLDEN_O[w * 2 + h];
      if (got == exp) { exact++; continue; }
      const uint32_t d = (got > exp) ? (uint32_t)(got - exp) : (uint32_t)(exp - got);
      sumabs += d; if (d > worst) worst = d;
    }
  }
  // TERMINATION TEST.  If the FA warps never retire, core.io.finished never rises, the
  // core-finish L0i/L0d writeback (MuonTile.scala:371-374) never fires, and O stays resident in
  // L0d -- which is exactly "still_poison=4096 and no amount of host waiting helps".
  // GPUResetAggregator maps one read-only finished reg per elaborated tile at 0x10+gcid*4
  // (CanHaveGPUReset.scala:99-101); the bring-up kernel reads 1 1 here when it DOES terminate,
  // so 0 0 means the kernel is still running and the flush never happened.
  // Per-stage progress probe: the kernel's own MARK() array, redirected to GPU DRAM.
  {
    volatile uint32_t *mk = (volatile uint32_t *)(FAF_O_BASE + 0x100000ull);
    faf_s("FA_FPGA: marks ="); 
    /* 22 marks are emitted: 1 entry + 4 key-blocks x 5 + 1 final.  Printing only 16 cut the
     * run off inside key-block j=2 and made a 4x error in the cycles-per-tile arithmetic
     * possible -- one MARK period is one Bk=64 KEY BLOCK, not one attention tile. */
    for (int k = 0; k < 24; k++) { faf_s(" "); faf_x(mk[k]); }
    faf_s("\n  (slot15=e47e4e47 means fa_entry WAS entered; deadbeef there means it never was)\n");
  }
#if defined(FA_SCALE_GUARD) && defined(FA_GUARD_ADDR)
  fa_guard_check("post_run   ");
#endif
#ifdef FA_PB
  // ---- UNCACHED BEACON READ (printBuf, device 0x80000 -> host 0x40080000).
  // Read this BEFORE the cores_finished line: it is the only progress signal that does not depend
  // on the core-finish writeback, so on a hung run it is the only line in this report with content.
  {
    static const char *nm[8] = {"entry","stage","tile","preret","predrain","postdrain","retire","warps"};
    faf_s("FA_FPGA: pb_beacon\n");
    for (int i = 0; i < 8; i++) {
      const uint64_t v = *(volatile uint64_t *)(0x40080000ull + 8ull * i);
      faf_s("  pb["); faf_u(i); faf_s("] "); faf_s(nm[i]); faf_s(" = ");
      faf_x((uint32_t)v); faf_s(" / "); faf_x((uint32_t)(v >> 32)); faf_s("\n");
    }
    faf_flush();
  }
#endif
  faf_s("FA_FPGA: cores_finished_after_run =");
  for (int g = 0; g < 4; g++) { faf_s(" "); faf_u(READ_MMIO_32(RAD_HOST_GPU_CORES + 4 * g)); }
  faf_s("   all_finished="); faf_u(READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED)); faf_s("\n");
  faf_s("FA_FPGA: poison_rb="); faf_x(poison_rb);
  faf_s(" done="); faf_u(done);
  faf_s(" wait_cyc="); faf_u(t_done - t_launch);
  faf_s("\n");
  faf_s("FA_FPGA: O[0..3]="); for (int k = 0; k < 4; k++) { faf_s(" "); faf_x(Ow[k]); }
  faf_s("\n");
  faf_s("FA_FPGA: golden[0..3]=");
  for (int k = 0; k < 4; k++) { faf_s(" "); faf_x(((uint32_t)FA_GOLDEN_O[k*2+1] << 16) | FA_GOLDEN_O[k*2]); }
  faf_s("\n");
  // Compact poison LOCATION report.  The full O dump is truncated by the UART (~2592 of 4096
  // words), and the missing words turned out to be past the cut -- so print the indices directly.
  if (still_poison) {
    uint32_t shown = 0, firsts[8], lastp = 0;
    for (uint32_t w = 0; w < FAF_O_WORDS; w++) {
      if (Ow[w] == FAF_POISON) { if (shown < 8) firsts[shown++] = w; lastp = w; }
    }
    faf_s("FA_FPGA: poison_at first=");
    for (uint32_t i = 0; i < shown; i++) { faf_u(firsts[i]); faf_s(","); }
    faf_s(" last="); faf_u(lastp);
    faf_s(" (O words=4096, row=64 words)\n"); faf_flush();
  }
  faf_s("FA_FPGA: still_poison="); faf_u(still_poison); faf_s("/"); faf_u(FAF_O_WORDS);
  faf_s(" exact_halfwords="); faf_u(exact); faf_s("/"); faf_u(FAF_O_WORDS * 2); faf_s("\n");
  faf_s("FA_FPGA: code_sumabs="); faf_u(sumabs);
  faf_s(" worst_code_delta="); faf_u(worst); faf_s("\n");
  // Second score against the FLASH reference -- the algorithm the kernel implements.  Scoring
  // only against fa_golden_O.h (the plain fp32-reference output) confuses algorithm difference
  // with kernel error: a perfect flash kernel scores exact=1253/8192 sumabs=2508713 against it.
  {
    uint32_t fexact = 0; uint64_t fsum = 0; uint32_t fworst = 0;
    for (uint32_t w = 0; w < FAF_O_WORDS; w++) {
      const uint32_t v = Ow[w];
      for (int h = 0; h < 2; h++) {
        const uint16_t got = (uint16_t)(h ? (v >> 16) : (v & 0xffffu));
        const uint16_t exp = FA_GOLDEN_OF[w * 2 + h];
        if (got == exp) { fexact++; continue; }
        const uint32_t d = (got > exp) ? (uint32_t)(got - exp) : (uint32_t)(exp - got);
        fsum += d; if (d > fworst) fworst = d;
      }
    }
    faf_s("FA_FPGA: vs_flash exact="); faf_u(fexact); faf_s("/"); faf_u(FAF_O_WORDS * 2);
    faf_s(" sumabs="); faf_u(fsum);
    faf_s(" worst="); faf_u(fworst); faf_s("\n");
  }
#ifdef FA_DUMP_O
  // Dump the whole O window so the error can be computed properly offline.  The bf16-CODE
  // metrics above are a poor proxy: a near-zero element whose sign differs contributes ~32768
  // to code_sumabs, so the aggregate is dominated by elements that are numerically negligible.
  // Frobenius over the decoded values is the metric fa_verify_out.py uses and the one to quote.
  {
    faf_s("FA_FPGA: O_DUMP_BEGIN\n");
    for (uint32_t w = 0; w < FAF_O_WORDS; w++) {
      faf_x(Ow[w]);
      if ((w & 15u) == 15u) faf_s("\n"); else faf_s(" ");
      if ((w & 255u) == 255u) faf_flush();
    }
    faf_flush();
    faf_s("FA_FPGA: O_DUMP_END\n"); faf_flush();
  }
#endif
  faf_s(still_poison ? "FA_FPGA: VERDICT=INVALID (O window not fully written)\n"
                     : "FA_FPGA: VERDICT=SCORED (compare triple to the metasim reference)\n");
  faf_flush();
  return 0;
#else
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

  uint32_t finished = 0;
  while (!finished) {
    SYNC_GPU();
    finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
  }

  return 0;
#endif
}
