// fpga_bringup host (rv64 Rocket) side.
//
// Sequence:
//   1. poison the GMEM result window so a GPU that never runs is DISTINGUISHABLE
//      from a GPU that runs and writes zeros  (a check that cannot fail is worse
//      than no check),
//   2. release the GPU from soft reset,
//   3. poll RAD_HOST_GPU_ALL_FINISHED with a BOUNDED loop -- on timeout we still
//      print the per-core finished bits and whatever landed in GMEM,
//   4. read the GPU-written GMEM back through the host alias and verify it
//      against the same reference function the kernel used,
//   5. print rdcycle deltas.
//
// Deliberately does NOT use the tocpu/tohost char proxy (SYNC_GPU): that channel
// is a separate mechanism and both kernels that use it hang on hardware.  All
// output here is plain rv64 HTIF printf, which is already proven by hello.riscv.

#define RAD_TAPEOUT_HOST 1
#include <rad_tapeout.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <radiance.h>

// ---------------------------------------------------------------------------
// OUTPUT CHANNEL.  htif_nano's printf is UNBUFFERED: one HTIF syscall round trip
// per character.  MEASURED: ~11k target cycles/char in the XB-10 metasim, and
// ~1 s/char on the U250 at 30 MHz (80 chars in 90 s, run hw_03_bringup).  That
// makes printf useless for any harness that wants to emit real data.
// Everything here is therefore formatted into one buffer and emitted with a
// SINGLE write(1, ...) -- one HTIF round trip for the whole report.
// ---------------------------------------------------------------------------
static uint32_t g_core_present = 0;  /* bitmask of gcids that physically exist (set at reset) */
static char obuf[8192];
static unsigned olen = 0;
static void oputs(const char *s) { while (*s && olen < sizeof(obuf) - 1) obuf[olen++] = *s++; }
static void ohex(uint32_t v) {
  static const char h[] = "0123456789abcdef";
  for (int i = 7; i >= 0; i--) if (olen < sizeof(obuf) - 1) obuf[olen++] = h[(v >> (i * 4)) & 0xf];
}
static void odec(uint64_t v) {
  char t[24]; int n = 0;
  if (!v) t[n++] = '0';
  while (v) { t[n++] = (char)('0' + (v % 10)); v /= 10; }
  while (n-- && olen < sizeof(obuf) - 1) obuf[olen++] = t[n];
}
static void oflush(void) { write(1, obuf, olen); olen = 0; }

#ifndef FB_N
#define FB_N 4096
#endif

#define FB_OUT_DEV   0x1F000000u
#define FB_STAT_DEV  0x1F100000u
#define FB_POISON    0xDEADBEEFu

// bounded poll; 1 iteration is a couple of MMIO reads (~O(100) target cycles)
// cycles to wait between readback pass 1 and pass 2
// cycles to wait after all_finished before re-asserting the GPU soft reset
#ifndef FB_RESET_DELAY
#define FB_RESET_DELAY 900ull
#endif

#ifndef FB_DRAIN_CYCLES
#define FB_DRAIN_CYCLES 500000ull
#endif

#ifndef FB_HOSTFLUSH_WAIT
#define FB_HOSTFLUSH_WAIT 400000ull
#endif
#ifndef FB_HOSTFLUSH_DRAIN
#define FB_HOSTFLUSH_DRAIN 200000ull
#endif

#ifndef FB_POLL_LIMIT
#define FB_POLL_LIMIT 2000000ull
#endif

static inline uint64_t rdcycle64(void) {
  uint64_t c;
  asm volatile("rdcycle %0" : "=r"(c));
  return c;
}

static inline volatile uint32_t *dev32(uint32_t devaddr) {
  return (volatile uint32_t *)(RAD_HOST_GPU_DRAM_BASE | (uint64_t)devaddr);
}


// ---- HOST-SIDE L0i / L0d INVALIDATE BEFORE LAUNCH -------------------------------------------
// Measured 2026-09-15: the GPU executes the PREVIOUSLY loaded image.  fa_base2 was loaded (md5
// confirmed) and the GPU produced fb_ref(i) for all 4096 words plus fb_entry's 600d0000 sentinel;
// separately, a build compiled WITHOUT -DFB_DRAIN_TIMING reported a drain stamp its own GPU code
// cannot write.  L0i is 512 sets x 1 way x 32 B = 16 KB direct mapped (RadianceConfigs.scala:90),
// which comfortably holds a whole bring-up kernel, so a stale L0i re-executes the previous kernel
// end to end rather than crashing.  Nothing invalidates it across a driver invocation: the only
// writeback is the core-finish edge (MuonTile.scala:370-374), which a hung predecessor never takes.
//
// Both caches expose a write-only flush register, and the host can reach cluster peripheral space
// the same way it already reaches GPU_ADDR_OR_MMIO at 0x40081000:
//   RadianceCluster.scala:29        peripheralAddrOffset = 0x80000
//   subsystem/Configs.scala:130     per-tile peripheralAddr = base + 0x200 + tile*0x200
//   muon/MuonTile.scala:196 / :246  l0i at peripheralAddr, l0d at +0x100
//   memory/TLNBDCache.scala         regmap 0x0 -> RegField.w(32), any write triggers the flush
// giving tile t: l0i 0x40080200 + t*0x200, l0d 0x40080300 + t*0x200.
//
// NOT YET CONFIRMED: the l0d half of that arithmetic is the flush the GPU already uses via
// (cid<<9)+0x80300, so the mapping is one slot away from something known to work -- but the host
// path to these registers has never been exercised, and a store to an unmapped physical address
// on this VM-less host hangs rather than faults cleanly.  Keep it behind the define until a run
// shows it is harmless.
#ifndef FB_NUM_TILES
#define FB_NUM_TILES 2
#endif
static inline void fb_host_invalidate_caches(void) {
  for (int t = 0; t < FB_NUM_TILES; t++) {
    WRITE_MMIO_32(0x40080200ull + (uint64_t)t * 0x200ull, 1u);   // L0i
    WRITE_MMIO_32(0x40080300ull + (uint64_t)t * 0x200ull, 1u);   // L0d
  }
}

static inline uint32_t fb_ref(uint32_t i) { return (i * 3u + 1u) + (i * 7u + 2u); }

int main() {
  volatile uint32_t *out = dev32(FB_OUT_DEV);
  volatile uint32_t *stat = dev32(FB_STAT_DEV);

  const uint64_t t0 = rdcycle64();

  // 1. poison
  // FB_NO_POISON_OUT: skip poisoning OUT.  Rocket has a writeback L1 D$ and this
  // host's plain `fence` does NOT flush it, so poisoning OUT and then reading it
  // back can return the host's OWN stale lines rather than the GPU's stores --
  // which would look exactly like "the GPU never wrote".  STAT is still poisoned
  // (8 words, far from OUT) so "GPU never ran" is still detectable via its magic.
#ifndef FB_NO_POISON_OUT
  for (uint32_t i = 0; i < FB_N; i++) out[i] = FB_POISON;
#endif
  for (uint32_t i = 0; i < 32; i++) stat[i] = FB_POISON;   /* whole block: [8..17] stamps and
     [20..24] bss probe were previously UNpoisoned, so their contents could not be dated --
     0x000434af / 0x00027745 were almost certainly stale rdcycle stamps from the barrier sweep. */
  asm volatile("fence" ::: "memory");
  const uint64_t t_poison = rdcycle64();

  // read one back so a broken host->GMEM path shows up immediately
  /* Verify the poison LANDED before release: distinguishes "nothing wrote" from "something
     wrote garbage during the run". */
  oputs("host: poison landed? out[0]="); ohex(out[0]);
  oputs(" stat[8]="); ohex(stat[8]); oputs(" stat[20]="); ohex(stat[20]);
  oputs("  (all must be deadbeef here)\n");
  oputs("host: poison writeback out[0]="); ohex(out[0]);
  oputs(" stat[0]="); ohex(stat[0]); oputs("\n");

#ifdef FB_CTX_SEED
  // CONTROLLED EXPERIMENT -- CAUSE Mode A instead of waiting to sample it.
  // schedule_context is orphaned OUTSIDE every LOAD segment, so the loader never touches it and a
  // HOST write here PERSISTS into the run.  That makes the mechanism testable causally at n=1:
  //   Arm P: FB_CTX_SEED=0xDEAD0001   (misaligned, non-executable -> a jump to it faults)
  //          predict Mode A deterministically, and the in-kernel probe should read the sentinel
  //          back in STAT[4]/STAT[5] -- which is also the probe's own positive control.
  //   Arm C: FB_CTX_SEED=<&fb_entry>  predict clean deterministically.
  // Placed AFTER all image staging and poison, BEFORE the reset edge, so nothing overwrites it.
  // Read back BEFORE release: a failed host write must never masquerade as a result (that is the
  // failure mode every dead instrument today shared).
  {
    volatile uint32_t *ctxh = dev32(FB_CTX_SEED_DEV);
    ctxh[0] = (uint32_t)(FB_CTX_SEED);
    asm volatile("fence" ::: "memory");
    const uint32_t rb = ctxh[0];
    oputs("host: ctx_seed dev=0x"); ohex(FB_CTX_SEED_DEV);
    oputs(" wrote="); ohex((uint32_t)(FB_CTX_SEED));
    oputs(" readback="); ohex(rb);
    oputs(rb == (uint32_t)(FB_CTX_SEED) ? "  [SEED LANDED]" : "  [SEED WRITE FAILED -- run is void]");
    oputs("\n");
  }
#endif
  // RE-ARM THE GPU: assert soft reset, pause, then release.  This edge is REQUIRED --
  // measured both ways: WITH the toggle the GPU runs (barrier_sweep v1 N=0:
  // gpu_started=1, 0/4096 mismatches); WITHOUT it the GPU is already latched
  // "finished" at release and never executes (barrier_sweep v2 N=0: gpu_started=0,
  // all_finished=1 after 1 poll, 4096/4096 still poison).  My earlier reading -- that
  // the toggle did not re-arm -- was backwards.
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
  { const uint64_t u = rdcycle64() + 200000ull; while (rdcycle64() < u) asm volatile("" ::: "memory"); }
  // DEPLOYED TILE COUNT, over MMIO only -- immune to the GMEM readback defect.
  // GPUResetAggregator maps one read-only `finished` reg per elaborated Muon tile at
  // 0x10 + gcid*4 (CanHaveGPUReset.scala:99-101).  While the GPU is HELD IN RESET
  // every EXISTING core reports finished=1 (no valid warps); a gcid with no tile is
  // unmapped and reads 0.  So 1,1,0,0 => 2 tiles (1 cluster) and 1,1,1,1 => 4 tiles.
  // Record WHICH gcids physically exist: while the GPU is held in reset every EXISTING core
  // reports finished=1, and an unmapped gcid reads 0.  This mask is the denominator for the
  // completion test used by the verdict -- see the CHECK-THAT-CANNOT-PASS note below.
  g_core_present = 0;
  oputs("host: cores_finished_at_reset =");
  for (int g = 0; g < 4; g++) {
    const uint32_t v = READ_MMIO_32(RAD_HOST_GPU_CORES + 4 * g);
    if (v) g_core_present |= (1u << g);
    oputs(" "); odec(v);
  }
  oputs("\n");

#ifdef FB_PROBE_OWIN
  // Is the FA O window (0x40040000, cluster space) reachable from the rv64 HOST at all?
  // The FA kernel writes it with GPU global stores; host access is a separate question, and my
  // FA host traps the instant it touches it (metasim: Rocket loops at pc=0x10098, the trap
  // vector).  FA's own notes say host writes to cluster SMEM must be 8 BYTES -- a 4-byte
  // sub-beat write is silently dropped -- so probe both widths, reads first.
  {
    volatile uint32_t *o32 = (volatile uint32_t *)0x40040000ull;
    volatile uint64_t *o64 = (volatile uint64_t *)0x40040000ull;
    oputs("probe: read32[0]="); ohex(o32[0]);
    oputs(" read32[4095]="); ohex(o32[4095]); oputs("\n"); oflush();
    o64[0] = 0x1122334455667788ull; asm volatile("fence" ::: "memory");
    oputs("probe: after wr64 -> [0]="); ohex(o32[0]); oputs(" [1]="); ohex(o32[1]); oputs("\n");
    o32[2] = 0xA5A5A5A5u; asm volatile("fence" ::: "memory");
    oputs("probe: after wr32 -> [2]="); ohex(o32[2]); oputs("\n"); oflush();
  }
#endif
#ifdef FB_PROBE_IMAGE
  // Did the rv32 image actually REACH the device?  The GPU fetches device 0x1000_0000+, visible to
  // the host at 0x1_1000_0000+.  Read the loaded bytes back at the entry, at .text, and near the
  // TOP of the padded image.  Correct data at the top => the load is fine and the failure is in
  // GPU instruction fetch; garbage/zeros => the loader truncated and the GPU fetches nothing.
  {
    const uint64_t probes[4] = {0x110000000ull, 0x110002000ull, 0x110020000ull, 0x110036000ull};
    for (int i = 0; i < 4; i++) {
      volatile uint32_t *q = (volatile uint32_t *)probes[i];
      oputs("probe: @0x"); ohex((uint32_t)(probes[i] & 0xffffffffu)); oputs(" =");
      for (int k = 0; k < 4; k++) { oputs(" "); ohex(q[k]); }
      oputs("\n");
    }
    oflush();
  }
#endif
  // 2. release GPU
  oputs("host: releasing GPU\n");
  #ifdef FB_HOST_L0I_FLUSH
    fb_host_invalidate_caches();   // the image in DRAM is new; the caches are not
  #endif
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);
  const uint64_t t_release = rdcycle64();

#ifdef FB_HOSTFLUSH
  // Host-triggered flush.  Wait for the kernel to have issued its stores, assert the GPU soft
  // reset (which quiesces the GPU but does NOT reach the dcache -- CacheFlushUnit takes the
  // parent's implicit reset and softReset is not a port of MuonNonBlockingDCache), then poke each
  // core's flush register.  Device (cid<<9)+0x80300 is cluster peripheral space, visible to the
  // host at 0x40080300 (core 0) and 0x40080500 (core 1).
  { const uint64_t u = rdcycle64() + FB_HOSTFLUSH_WAIT; while (rdcycle64() < u) asm volatile("" ::: "memory"); }
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
  { const uint64_t u = rdcycle64() + 2000ull; while (rdcycle64() < u) asm volatile("" ::: "memory"); }
  *(volatile uint32_t *)0x40080300ull = 0;
  *(volatile uint32_t *)0x40080500ull = 0;
  asm volatile("fence" ::: "memory");
  { const uint64_t u = rdcycle64() + FB_HOSTFLUSH_DRAIN; while (rdcycle64() < u) asm volatile("" ::: "memory"); }
  oputs("host: host-triggered flush issued to 0x40080300/0x40080500\n");
#endif
  // 3a. HANDSHAKE: wait for all_finished to go 0, i.e. the GPU actually STARTED.
  // Without this the poll below is a check that cannot fail: consecutive FireSim
  // runs on the FPGA are NOT independent -- DRAM contents and the GPU
  // finished/soft-reset latch survive across driver invocations, so run N+1 can
  // read all_finished==1 left by run N, never execute the kernel at all, and still
  // "verify" run N's late-arriving data.  Observed exactly that in hw_11_flush
  // (gpu=79 cycles, all_finished=1 after 1 poll, out[0] already correct BEFORE the
  // release).  gpu_started + the STAT magic together make that state detectable.
  uint32_t gpu_started = 0;
  for (uint64_t w = 0; w < 200000ull; w++) {
    if (READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED) == 0) { gpu_started = 1; break; }
  }

  // 3b. bounded poll
  uint64_t iters = 0;
  uint32_t finished = 0;
  while (!finished && iters < FB_POLL_LIMIT) {
    finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED);
    iters++;
  }
  const uint64_t t_done = rdcycle64();
  // FB_RESET_DELAY: hold off on re-asserting the soft reset.  MuonTile.scala:370-374
  // starts the L0i/L0d writeback on the core.io.finished EDGE and only when
  // !softReset; RAD_HOST_GPU_ALL_FINISHED is that same signal
  // (CanHaveGPUReset.scala:92-97), so asserting reset the instant the host sees
  // all_finished can cut the writeback off.  Budget: GPUResetAggregator $finishes
  // the metasim after 1024 consecutive cycles of (allFinished && !coreReset)
  // (CanHaveGPUReset.scala:104-108), so this delay must stay under ~1000.
  {
    const uint64_t until = rdcycle64() + FB_RESET_DELAY;
    while (rdcycle64() < until) { asm volatile("" ::: "memory"); }
  }
  // Re-assert the GPU soft reset, before any printing.  In RTL sim
  // GPUResetAggregator does stop("no more active warps for 1k cycles") whenever
  // allFinished && !coreResets.head (CanHaveGPUReset.scala:104-108), which would
  // $finish the metasim before the verification output is flushed.  Same trick as
  // kernels/launch/host.cpp ("reset before print to prevent simulation end").
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 1);
  const uint32_t c0 = READ_MMIO_32(RAD_HOST_GPU_CORES + 0);
  const uint32_t c1 = READ_MMIO_32(RAD_HOST_GPU_CORES + 4);
  oputs("host: gpu_started="); odec(gpu_started); oputs("\n");
  oputs("host: all_finished="); odec(finished);
  oputs(" after "); odec(iters); oputs(" polls (core0="); odec(c0);
  oputs(" core1="); odec(c1); oputs(")\n");

  // 4. read back + verify
  asm volatile("fence" ::: "memory");
#ifdef FB_TAPEOUT_EPILOGUE
  // ---- EPILOGUE HANDSHAKE.  The GPU never asserts finished by design, so all_finished and the
  // per-core bits will NEVER come -- do not poll them.  Wait for the postbox, then soft-reset.
  {
    const int nc = 2;   // 1-cluster part: 2 muon tiles
    const int done = rad_host_wait_epilogue(nc, 20000000ull);
    oputs("host: epilogue done_cores="); odec(done); oputs(" of "); odec(nc);
    oputs("  pb_core0="); ohex((uint32_t)*(volatile uint64_t *)RAD_PB_HOST(0,0));
    oputs(" pb_core1="); ohex((uint32_t)*(volatile uint64_t *)RAD_PB_HOST(0,1));
    oputs("\n");
    if (done == nc) { rad_host_gpu_soft_reset(); oputs("host: soft reset asserted\n"); }
    else            { oputs("host: EPILOGUE TIMEOUT -- not resetting\n"); }
  }
#endif
#ifdef FB_PB_BEACON
  // ---- READ THE printBuf BEACON.  Uncached TLRAM at device 0x80000 -> host 0x40080000.
  // The only GPU->host channel that does not depend on the core-finish writeback, so the only
  // probe that can report progress from a kernel that never retires.  8-byte reads: that is the
  // TLRAM's native beat, and the GPU duplicated each 32-bit value into both halves.
  {
    oputs("host: pb_beacon =");
    for (int i = 0; i < 4; i++) {
      const uint64_t v = *(volatile uint64_t *)(0x40080000ull + 8ull * i);
      oputs(" "); ohex((uint32_t)v); oputs("/"); ohex((uint32_t)(v >> 32));
    }
    oputs("\n");
  }
#endif
  oputs("host: stat ="); for (int k = 0; k < 4; k++) { oputs(" "); ohex(stat[k]); } oputs("\n");
  /* STAT[9] = SMEM rung verify: 0x5EExxxx where xxxx is the mismatch count.  Printed explicitly --
     a slot written by the kernel but never displayed by the host is a silent null. */
  oputs("host: smem_verify="); ohex(stat[9]);
  if (stat[9] == FB_POISON)            oputs("  [SMEM-RUNG-ABSENT: not written]");
  else if ((stat[9] >> 16) != 0x5EE0u) oputs("  [UNEXPECTED SENTINEL]");
  else if ((stat[9] & 0xFFFFu) == 0)   oputs("  [SMEM OK: 0 mismatches]");
  else                                 oputs("  [SMEM MISMATCHES]");
  oputs("\n");
  /* CTX PROBE, slots [4..6] -- inside STAT line 0 so they are actually readable, unlike the old
     STAT[20] at offset 0x50 which sat in line 1 and never reported. */
  /* Drain cost, stamped by the GPU itself (STAT[12] magic distinguishes "not measured"
     from "measured zero").  [10] = manual flush window, [11] = capacity-eviction window;
     a build carries whichever drain it was compiled with, so the other reads poison. */
  if (stat[12] == 0xC0571234u) {
    oputs("host: drain_cycles manual_flush=");
    if (stat[10] == FB_POISON) oputs("absent"); else odec(stat[10]);
    oputs(" capacity_evict=");
    if (stat[11] == FB_POISON) oputs("absent"); else odec(stat[11]);
    oputs("\n");
  } else {
    oputs("host: drain_cycles NOT MEASURED (stat[12]="); ohex(stat[12]); oputs(")\n");
  }
  oputs("host: ctx_probe core0="); ohex(stat[4]);
  oputs(" core1="); ohex(stat[5]);
  oputs(" expected="); ohex(stat[6]);
  if (stat[4] == FB_POISON && stat[5] == FB_POISON) oputs("  [PROBE-ABSENT: not written]");
  else if (stat[4] != stat[5])                     oputs("  [CROSS-CORE MISMATCH]");
  else if (stat[6] != FB_POISON && stat[4] != stat[6]) oputs("  [BOTH WRONG vs expected]");
  else                                             oputs("  [both match expected]");
  oputs("\n");
  // Barrier sweep readback.  stat[8] is an instrument magic so "no stamps written" is
  // distinguishable from "stamps are legitimately zero" -- a zero-vs-absent confusion is how a
  // barrier cost of 0 would get reported as a result.
  if (stat[20] == 0xB55B55B5u) {
    oputs("host: .bss AT ENTRY (schedule_context) ="); for (int k=0;k<4;k++){oputs(" ");ohex(stat[21+k]);}
    oputs("   (all-zero => clean; nonzero => STALE DRAM from a previous run)\n");
  } else { oputs("host: .bss probe absent (stat[20]="); ohex(stat[20]); oputs(")\n"); }
  if (stat[8] == 0xBA44BA44u) {
    static const char *lbl[9] = {"overhead(0)", "cold1", "cold2", "cold4", "cold8",
                                 "warm1", "warm2", "warm4", "warm8"};
    static const unsigned nb[9] = {0, 1, 2, 4, 8, 1, 2, 4, 8};
    oputs("host: BARSWEEP raw cycles:\n");
    for (int k = 0; k < 9; k++) {
      const uint32_t d = stat[9 + k];
      oputs("  "); oputs(lbl[k]); oputs(" = "); odec(d);
      if (nb[k]) {
        const uint32_t ov = stat[9];
        const uint32_t net = (d > ov) ? (d - ov) : 0u;
        oputs("  net="); odec(net); oputs("  per_barrier="); odec(net / nb[k]);
      }
      oputs("\n");
    }
  } else {
    oputs("host: BARSWEEP absent (stat[8]="); ohex(stat[8]);
    oputs(") -- NOT a zero-cost result, the instrument did not report\n");
  }

  // ---- PASS 1: read back immediately after all_finished ----
  uint32_t bad = 0, still_poison = 0;
  uint32_t first_bad_i = 0xFFFFFFFFu, first_bad_v = 0, first_bad_e = 0;
  for (uint32_t i = 0; i < FB_N; i++) {
    const uint32_t v = out[i];
    const uint32_t e = fb_ref(i);
    if (v == FB_POISON) still_poison++;
    if (v != e) {
      if (bad == 0) { first_bad_i = i; first_bad_v = v; first_bad_e = e; }
      bad++;
    }
  }
  // ---- PASS 2: wait, then re-read.  THE DISCRIMINATING EXPERIMENT.
  // RAD_HOST_GPU_ALL_FINISHED is driven from the same core.io.finished that
  // *starts* the L0d writeback (MuonTile.scala:370-374 vs
  // CanHaveGPUReset.scala:92-97), so a nonzero pass-1 mismatch count that goes
  // to ZERO in pass 2 means "flush had not drained", whereas a mismatch that
  // SURVIVES the wait means stores were genuinely lost.  These have completely
  // different fixes, so the harness must be able to tell them apart.
  {
    const uint64_t spin_until = rdcycle64() + FB_DRAIN_CYCLES;
    while (rdcycle64() < spin_until) { asm volatile("" ::: "memory"); }
  }
  asm volatile("fence" ::: "memory");
  uint32_t bad2 = 0, poison2 = 0;
  for (uint32_t i = 0; i < FB_N; i++) {
    const uint32_t v = out[i];
    if (v == FB_POISON) poison2++;
    if (v != fb_ref(i)) bad2++;
  }
  oputs("host: out[0..3] ="); for (int k = 0; k < 4; k++) { oputs(" "); ohex(out[k]); }
  oputs("  (expect"); for (int k = 0; k < 4; k++) { oputs(" "); ohex(fb_ref(k)); } oputs(")\n");
  // WHICH lines escaped.  Every observed partial export is an exact multiple of 16 words = one
  // 64 B cache line (80/96/128/400/1200 -> 5/6/8/25/75 lines), which is the signature of a
  // writeback sweep that exported N whole lines and stopped -- not of a kernel that failed to run.
  // The flush walks sets sequentially, so the surviving line indices identify the wedge point in
  // the sweep order.  Same indices across trials => deterministic wedge; different => ack-ordering
  // nondeterminism, which is what a source-ID collision predicts.
  {
    uint32_t nlines = 0;
    oputs("host: written_lines(idx of 16-word lines with any non-poison) =");
    for (uint32_t L = 0; L < FB_N / 16u; L++) {
      uint32_t any = 0;
      for (uint32_t k = 0; k < 16u; k++) if (out[L * 16u + k] != FB_POISON) { any = 1; break; }
      if (any) { nlines++; if (nlines <= 256) { oputs(" "); odec(L); } }   /* 24->80->256: the 80 cap showed only 0..79, so it located where loss STARTS but not which lines are gone */
    }
    oputs("   total_lines="); odec(nlines);
    oputs(" of "); odec(FB_N / 16u); oputs("\n");
  }
  /* MISSING lines, printed directly.  The survivor list is 253 entries and the interesting set is
     the 3 that are absent; listing those answers "are 243/244/245 contiguous" with no arithmetic. */
  {
    uint32_t nmiss = 0;
    oputs("host: missing_lines(idx of 16-word lines that are ALL poison) =");
    for (uint32_t L = 0; L < FB_N / 16u; L++) {
      uint32_t any = 0;
      for (uint32_t k = 0; k < 16u; k++) if (out[L * 16u + k] != FB_POISON) { any = 1; break; }
      if (!any) { nmiss++; if (nmiss <= 256) { oputs(" "); odec(L); } }
    }
    oputs("   total_missing="); odec(nmiss); oputs("\n");
  }
  oputs("host: mismatches="); odec(bad); oputs("/"); odec(FB_N);
  oputs(" still_poison="); odec(still_poison); oputs("\n");
  if (bad) {
    oputs("host: first bad i="); odec(first_bad_i);
    oputs(" got="); ohex(first_bad_v); oputs(" want="); ohex(first_bad_e); oputs("\n");
  }

  // 5. cycles
  oputs("host: reset_delay="); odec(FB_RESET_DELAY); oputs("\n");
  /* Second read of the drain stamp, AFTER the pass-2 wait.  The first read sits right after
     the poll loop; if the stamp is absent there and present here, the host read too early
     rather than the GPU failing to write.  Two reads separate those; one cannot. */
  oputs("host: drain_cycles_pass2 stat[10]="); ohex(stat[10]);
  oputs(" stat[11]="); ohex(stat[11]);
  oputs(" stat[12]="); ohex(stat[12]); oputs("\n");
  oputs("host: pass2 (after "); odec(FB_DRAIN_CYCLES);
  oputs(" cyc) mismatches="); odec(bad2); oputs(" still_poison="); odec(poison2); oputs("\n");
  oputs("host: cycles poison="); odec(t_poison - t0);
  oputs(" gpu="); odec(t_done - t_release);
  oputs(" total="); odec(rdcycle64() - t0); oputs("\n");

  const uint32_t gpu_ran = gpu_started && (stat[0] == 0x600D0000u);  /* NB: STAT visibility, downstream of the flush */
  oputs("host: stat_visible="); odec(gpu_ran);   /* renamed: this is STAT VISIBILITY, not "the GPU ran" -- partial-export trials wrote words with stat_visible=0 */ oputs("\n");
  // CHECK THAT CANNOT PASS (fixed).  Both PASS branches were gated on `finished` =
  // RAD_HOST_GPU_ALL_FINISHED, which on this 1-cluster part cannot read 1 after the GPU runs:
  // the aggregate ANDs all 4 gcids but only 2 tiles are elaborated, so it stays 0 even when
  // every core that PHYSICALLY EXISTS reports finished=1.  A bit-exact run therefore printed
  // FAIL (observed 2026-08-24 01:53: mismatches=0/4096, still_poison=0, total_lines=256 of 256,
  // core0=1 core1=1, all_finished=0 -> "FAIL").  Use the per-core bits masked by cores present.
  uint32_t cores_done = 1;
  for (int g = 0; g < 4; g++)
    if (g_core_present & (1u << g))
      cores_done &= (READ_MMIO_32(RAD_HOST_GPU_CORES + 4 * g) != 0);
  if (!g_core_present) cores_done = 0;   /* no cores detected -> cannot claim completion */
  oputs("host: cores_done="); odec(cores_done);
  oputs(" present_mask="); odec(g_core_present);
  oputs(" all_finished="); odec(finished); oputs("\n");

  if (!gpu_ran)                      oputs("FPGA_BRINGUP: INVALID (STAT not visible -- flush may have partially exported; see written_lines)\n");
  else if (cores_done && bad == 0)   oputs("FPGA_BRINGUP: PASS (clean on first read)\n");
  else if (cores_done && bad2 == 0)  oputs("FPGA_BRINGUP: PASS_AFTER_DRAIN (readback race)\n");
  else if (bad == 0 || bad2 == 0)    oputs("FPGA_BRINGUP: PASS_DATA_ONLY (data bit-exact; per-core completion not confirmed)\n");
  else                               oputs("FPGA_BRINGUP: FAIL\n");

  // ONE HTIF round trip for the whole report.
  const uint64_t t_pre_out = rdcycle64();
  oflush();
  const uint64_t t_post_out = rdcycle64();
  // second, tiny write just to report what the first one cost
  oputs("host: report write cost "); odec(t_post_out - t_pre_out); oputs(" cycles\n");
  oflush();

  return 0;
}
