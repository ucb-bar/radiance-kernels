// fpga_bringup: minimal GPU-exercising kernel for U250 FireSim bring-up.
//
// Purpose: prove the three things that turn the board into a test environment
//   (a) an arbitrary ELF can be loaded and launched,
//   (b) GPU-written GMEM can be read back by the rv64 host for scoring,
//   (c) cycle counts are observable.
//
// The GPU writes OUT[i] = f(i) for FB_N elements, spread over every warp of both
// Muon cores of the (single) cluster.
//
// Memory map: device GMEM address X is visible to the host at X | 0x1_0000_0000
// (radiance.h rad_device_to_host_address).  The loaded kernel image occupies
// device 0x1000_0000 + a few 10s of KiB, so 0x1F00_0000 is well clear of it.
//
// NO MANUAL L0d FLUSH.  The hardware does it: MuonTile.scala:371-374
//     val justFinished = core.io.finished && !RegNext(core.io.finished)
//     when (justFinished && !softReset) {
//       outer.iFlushMaster.out.head._1.start := true.B
//       outer.dFlushMaster.out.head._1.start := true.B
//     }
// i.e. both L0 caches are flushed the cycle a core's last warp dies.  An earlier
// version of this kernel did an explicit (cid<<9)+0x80300 flush store behind a
// mu_barrier(0, MU_NUM_CORES*FB_NWARPS); that barrier DEADLOCKED (verified: the
// metasim's Rocket spun in the all_finished poll past 840k cycles and the GPU
// never reported finished), so it is gone.
//
// CAVEAT, still open: RAD_HOST_GPU_ALL_FINISHED is driven from the same
// core.io.finished that *starts* the flush (CanHaveGPUReset.scala:92-97), so the
// host can observe all_finished before the writeback has drained.  The host side
// re-reads after the poll to give it slack; if readback is ever flaky, this race
// is the first suspect.

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <rad_tapeout.h>
#include <stdint.h>




#ifdef FB_BSS_ZERO
// FIX (2): actually zero .bss.  mu_start.S:47-53 has the loop COMMENTED OUT, and .bss is NOBITS so
// the loader never writes it -- every zero-init static holds stale DRAM from the previous run.
//
// THE TWO DEFECTS COMPOSE, and neither fix works alone:
//   (1) mu_link.ld's `.bss : { *(.bss) }` misses -fdata-sections names (.bss.<sym>), so .bss was
//       ORPHAN-placed at ~0x10001060, BELOW .text and therefore below _edata.  Re-enabling the
//       zeroing loop verbatim would have memset _edata.._end and never touched it.
//   (2) With (1) fixed, .bss lands at 0x10031080 inside _edata(_end=0x100310c0) -- so the loop
//       finally covers it.  But placement alone leaves it full of stale DRAM.
// Done per-core (one thread each): the write is idempotent, and doing it on BOTH cores means each
// core's own L0d holds the zeros, so no cross-core visibility is required.
extern "C" char _edata[], _end[];
static inline void fb_zero_bss() {
  if (vx_thread_id() == 0) {
    volatile char *p = (volatile char *)_edata;
    volatile char *e = (volatile char *)_end;
    while (p < e) *p++ = 0;
    asm volatile("" ::: "memory");
  } else { asm volatile("nop"); }
}
#endif

#ifndef FB_N
#define FB_N 4096
#endif
#ifndef FB_PREFLUSH
#define FB_PREFLUSH 0u
#endif
#ifndef FB_NWARPS
#define FB_NWARPS 4
#endif

// FB_INIT_WARPS decouples the RESET warp count from the DISPATCH occupancy, which are otherwise
// both FB_NWARPS.  FA has them mismatched -- it dispatches occupancy 3 while leaving
// __mu_num_warps undefined, so mu_start.S falls back to `csrr nw` = 8 -- and FA's mu_schedule
// hangs where this kernel's completes.  Setting FB_NWARPS=3 FB_INIT_WARPS=8 reproduces FA's
// configuration exactly on a kernel that is known to work, which tests the mismatch on a clean
// substrate instead of on the kernel under suspicion.
#ifndef FB_INIT_WARPS
#define FB_INIT_WARPS FB_NWARPS
#endif
extern "C" uint32_t __mu_num_warps = FB_INIT_WARPS;

// ---- device-side GMEM layout (host reads these at | 0x1_0000_0000) ----
#define FB_OUT_DEV   0x1F000000u   // FB_N u32 results
#define FB_STAT_DEV  0x1F100000u   // status block

static volatile uint32_t *const OUT  = (volatile uint32_t *)FB_OUT_DEV;
static volatile uint32_t *const STAT = (volatile uint32_t *)FB_STAT_DEV;

// Reference value, duplicated verbatim on the host side.
// L0d -> L1 writeback for THIS core.  L1 is coherent, L0d is not, and a plain
// `fence` does not write L0d back (LSU.scala:520-525 -- a fence is a "fake entry"
// that only orders the store queue, it generates no downstream request).  The
// primitive is a SHARED-space store to (cid << 9) + 0x80300; the stored value is
// irrelevant.  Note the cid term: each core has its own flush port, so the
// executing thread must compute it from its own vx_core_id().
// MUST be issued by exactly ONE THREAD of ONE WARP per core -- an all-lanes or
// all-warps flush store can deadlock the flush port.
static inline void fb_flush_l0d() {
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *p = (volatile uint32_t *)((cid << 9) + 0x80300u);
  asm volatile("sw.shared x0, 0(%0)" :: "r"(p) : "memory");
}

// L0i INVALIDATE for THIS core.  Same mechanism as the L0d flush one function below, one
// peripheral slot lower.  Addresses, from the tapeout tree:
//   RadianceCluster.scala:29        peripheralAddrOffset = 0x80000
//   subsystem/Configs.scala:130     per-tile peripheralAddr = base + 0x200 + tile*0x200
//   muon/MuonTile.scala:196         l0i flushAddr = peripheralAddr
//   muon/MuonTile.scala:246         l0d flushAddr = peripheralAddr + 0x100
// so tile i sees l0i at (i<<9)+0x80200 and l0d at (i<<9)+0x80300.  The l0d half of that
// arithmetic is CONFIRMED: (cid<<9)+0x80300 is the flush this kernel already uses and it
// demonstrably works, which is what makes the l0i address a one-slot step rather than a guess.
// Still a guess until it is measured -- an unmapped store here would show as a hang.
//
// WHY A KERNEL WANTS THIS AT ITS END, not its start: by the time a kernel executes, whatever
// it is going to fetch has already been fetched.  Invalidating at the END is what lets the
// NEXT kernel's image be seen.  MuonTile.scala:370-374 does this on the core-finish edge --
// but only for a kernel that actually finishes, which is exactly what a hung predecessor
// does not do.
static inline void fb_flush_l0i() {
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *p = (volatile uint32_t *)((cid << 9) + 0x80200u);
  asm volatile("sw.shared x0, 0(%0)" :: "r"(p) : "memory");
}

// Capacity-eviction drain: walk `lines` 64B lines of per-core scratch so every dirty line in
// L0d is evicted through the MSHR writeback path, which uses source ids inside this client's
// declared IdRange and therefore works on bitstreams whose flush unit does not.  Caller supplies
// the barriers; this is the single-executor body only.
static inline void fb_scrub_body(uint32_t base, uint32_t lines) {
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *scrub = (volatile uint32_t *)(base + cid * (lines * CACHE_LINE_BYTES));
  for (uint32_t l = 0; l < lines; l++) scrub[l * (CACHE_LINE_BYTES / 4)] = l + 1u;
  mu_fence();
}

// ---- FB_PB_BEACON: a progress beacon that SURVIVES A HANG. -------------------------------
// Every existing probe in this project writes GMEM, and a GMEM store is invisible until the
// core-finish writeback (MuonTile.scala:370-374): L0d is writeback, and rad_l0d_drain only pushes
// it to L1, which is ALSO writeback.  So in any hang every mark reads poison no matter how far the
// kernel got -- the probe is blind in exactly the case it exists for.
//
// printBuf is the way out.  RadianceCluster.scala:96-103 instantiates
//     TLRAM(AddressSet(baseAddr + peripheralAddrOffset, 0x100*nTiles-1), cacheable = false,
//           atomics = true, beatBytes = 8)
// *** cacheable = false *** -- a store here bypasses L0d and L1 entirely and lands in the TLRAM,
// so it is host-visible IMMEDIATELY, with no writeback and no core-finish edge.  512 B at device
// 0x80000 (host 0x40080000), disjoint from the per-tile flush regs which start at 0x80200.
//
// UNPROVEN IN THIS DIRECTION.  host.cpp's mailbox note says an 8 B HOST STORE to printBuf never
// became visible to the Muon; that is host->GPU.  This is GPU->host, which nothing has tested.
// FB_PB_HANG deliberately wedges the kernel AFTER stamping, so the beacon is validated in the one
// condition it is meant to be read in -- rather than only on a run that completes anyway.
#ifndef FB_PB_BASE
#define FB_PB_BASE 0x80000u
#endif
static inline void fb_pb(uint32_t slot, uint32_t val) {
  // 4-byte shared stores, duplicated into both halves of the 8 B beat so an 8-byte host read sees
  // the value in either half regardless of how the beat is assembled.
  volatile uint32_t *p = (volatile uint32_t *)(FB_PB_BASE + slot * 8u);
  asm volatile("sw.shared %0, 0(%1)" :: "r"(val), "r"(p) : "memory");
  asm volatile("sw.shared %0, 0(%1)" :: "r"(val), "r"(p + 1) : "memory");
}

// register-only delay; never `volatile` on a local (the stack is in GMEM here)
static inline void fb_pause(uint32_t n) {
  asm volatile("1:\n\t addi %0, %0, -1\n\t bnez %0, 1b\n\t" : "+r"(n));
}

#ifdef FB_CTX_PERCORE
// PER-CORE schedule_context WRITE -- the test that breaks the circularity.
// mu_schedule.cpp:88 writes schedule_context from CORE 0 ONLY, then mu_fence + mu_barrier.
// mu_fence retires at the store-queue head WITHOUT an L0d writeback, and L0d is not coherent
// across cores, so CORE 1 reads the value from DRAM -- which is stale .bss (never zeroed,
// orphaned outside every LOAD segment).  Hence: run 1 fails on garbage; run 2+ of the SAME binary
// passes because run 1's end-of-kernel flush left the correct callback at that address.
//
// Zeroing .bss does NOT test this -- it guarantees core 1 reads callback==0, i.e. deterministic
// failure, which is exactly what `zero` produced (0/3).  Ambiguous, not a refutation.
// fb_bssfix2 does not test it either: it wrote from core 0 and FLUSHED, and the flush unit is the
// thing under suspicion, so a failed flush is indistinguishable from the hypothesis being wrong.
//
// Here EVERY core writes the context itself, into its OWN L0d.  No cross-core visibility, no
// flush, no dependence on .bss contents.
static void fb_entry(void *, uint32_t, uint32_t, uint32_t);
static inline void fb_ctx_percore() {
  if (vx_thread_id() == 0) {                       // one thread per core, on ALL cores
    volatile uint32_t *ctx = (volatile uint32_t *)FB_CTX_PERCORE;
    ctx[0] = (uint32_t)(uintptr_t)&fb_entry;       // callback
    ctx[1] = 0u;                                   // arg
    ctx[2] = FB_NWARPS;                            // occupancy
    asm volatile("" ::: "memory");
  } else { asm volatile("nop"); }
}
#endif

static inline uint32_t fb_ref(uint32_t i) { return (i * 3u + 1u) + (i * 7u + 2u); }

// GPU-side cycle counter, same CSR the FA MARK profiler uses (kernel.cpp:176 there).
static inline uint32_t fb_mcycle() { uint32_t c; asm volatile("csrr %0, mcycle" : "=r"(c)); return c; }
// N barriers, uniform control flow (constant trip count, every warp takes the same path), so this
// is NOT a divergent-region vx_bar; fa_baraudit-clean shape.
#define FB_BARN(n) do { _Pragma("clang loop unroll(full)") \
  for (uint32_t _k = 0; _k < (n); _k++) mu_barrier(1, MU_NUM_CORES * FB_NWARPS); } while (0)

// ---- FB_DRAIN_TIMING ------------------------------------------------------------------
// Cost of a writeback drain, measured on the GPU rather than inferred.  The host's `gpu=`
// number cannot be used for this: it is t_done - t_release, and t_done comes after a poll
// loop that always runs to FB_POLL_LIMIT because all_finished never re-asserts on hardware,
// so it is a constant and says nothing about kernel work.
// Window shape follows the barrier sweep's: read mcycle, do the thing, read mcycle, then
// store BETWEEN windows so the store's own memory traffic is never inside a measurement.
// STAT[12] is a magic, so "the instrument did not run" is distinguishable from "cost 0".
// Slots [10]/[11] collide with the barrier sweep's FB_STAMP range; the two are never built
// together (the sweep needs FB_NBAR), and STAT[12] disambiguates if they ever are.
#ifdef FB_DRAIN_TIMING
#define FB_DRAIN_T0(v)        const uint32_t v = fb_mcycle()
// THE STAMP MUST ITSELF BE DRAINED.  Written plainly because this kernel already records the
// same mistake once, in the barrier sweep: "v1 of this instrument stored all ten stamps at the
// END, i.e. BEHIND the very thing being measured ... and reported nothing at all."  I repeated
// it: the first version of FB_DRAIN_T1 stored after the drain, the store died in L0d, and all
// six board runs came back stat[12]=deadbeef.
// So: stamp, then run a SECOND short drain whose only job is to evict the stamp.  It is not
// measured and it is cheap, because almost nothing is dirty by then.  Both mechanisms are used,
// so the stamp survives on a bitstream whose flush unit is broken (the scrub) and the flush store
// costs nothing where it is not (the tapeout).
#define FB_DRAIN_STAMP_LINES 96u          /* > one L0d (64 lines) */
#define FB_DRAIN_STAMP_DEV   0x1F300000u  /* distinct from FB_SCRUB_DEV so the two never alias */
#define FB_DRAIN_T1(v, slot)  do {                                                       \
    const uint32_t _e = fb_mcycle();                                                     \
    if (tid_in_threadblock == 0) { STAT[slot] = _e - (v); STAT[12] = 0xC0571234u; }      \
    else { asm volatile("nop"); }                                                        \
    mu_barrier(1, MU_NUM_CORES * FB_NWARPS);                                             \
    {                                                                                    \
      const uint32_t _wr = tid_in_threadblock / MU_NUM_THREADS;                          \
      const uint32_t _ln = tid_in_threadblock % MU_NUM_THREADS;                          \
      if (_wr < MU_NUM_CORES && _ln == 0) {                                              \
        fb_scrub_body(FB_DRAIN_STAMP_DEV, FB_DRAIN_STAMP_LINES);                         \
        fb_flush_l0d();                                                                  \
        fb_pause(500);                                                                   \
      } else { asm volatile("nop"); }                                                    \
    }                                                                                    \
    mu_barrier(2, MU_NUM_CORES * FB_NWARPS);                                             \
  } while (0)
#else
#define FB_DRAIN_T0(v)        do { } while (0)
#define FB_DRAIN_T1(v, slot)  do { } while (0)
#endif

static void fb_entry(void * /*arg*/, uint32_t tid_in_threadblock,
                     uint32_t threads_per_threadblock,
                     uint32_t /*threadblock_id*/) {
#ifdef FB_PB_BEACON
  // Slot 0: "fb_entry reached", stamped by lane 0 of every warp that gets here.  Slot 1 counts
  // arrivals so a partial dispatch is distinguishable from none.  Written BEFORE any other work.
  if ((tid_in_threadblock % MU_NUM_THREADS) == 0u) {
    fb_pb(0, 0xE47E4E47u);
    fb_pb(1, 0x57A60000u | (tid_in_threadblock / MU_NUM_THREADS));
  } else { asm volatile("nop"); }
#ifdef FB_PB_HANG
  // DELIBERATE HANG, after the stamp.  This is the gate: if the host can still read slot 0 while
  // cores_finished stays 0 0 0 0, the beacon is a valid instrument for a hung kernel.  If it
  // cannot, the beacon is as blind as the GMEM marks and this whole approach is dead.
  for (;;) { asm volatile("" ::: "memory"); }
#endif
#endif
  // stamp the header from exactly one thread of the grid so the host can tell
  // "GPU never ran" from "GPU ran and produced wrong data"
  if (tid_in_threadblock == 0) {
    STAT[0] = 0x600D0000u;
    STAT[1] = FB_N;
    STAT[2] = threads_per_threadblock;
    STAT[3] = MU_NUM_CORES;
  }
  for (uint32_t i = tid_in_threadblock; i < FB_N; i += threads_per_threadblock) {
    OUT[i] = fb_ref(i);
  }
#ifdef FB_NBAR
  // BARRIER-COST SWEEP.  N extra cluster-wide barriers, otherwise identical kernel.
  // Uniform control flow (constant trip count, every warp takes the same path), so
  // this is NOT a divergent-region vx_bar and is not exposed to the llvm duplication
  // hazard.  Same barrier id reused sequentially = successive rendezvous, the normal
  // pattern.  Fully unrolled so the loop overhead cannot be mistaken for barrier cost.
#pragma clang loop unroll(full)
  for (uint32_t k = 0; k < FB_NBAR; k++) {
    mu_barrier(1, MU_NUM_CORES * FB_NWARPS);
  }
#endif

#ifdef FB_TAIL_SPIN
  // Diagnostic knob only (register-only, see fb_pause): buys the store queue time
  // before the warp retires.  Establishes the dose-response curve; NOT the fix.
  fb_pause(FB_TAIL_SPIN);
#endif

#ifdef FB_MANUAL_FLUSH
  FB_DRAIN_T0(_dtf0);
  // ---- THE CORRECT GPU->CPU COPY PATH ----------------------------------------
  // A fence must be executed by ONE WARP per core (and here, one thread of it).
  // Measured: with every warp fencing, 3 of 4 warps ISSUE the fence and never
  // WRITEBACK -- ms_05, pc=10002160.  That was software misuse of the primitive,
  // not an RTL bug.
  //
  // Sequence: all warps sync -> single executor per core fences (drains its store
  // queue) and kicks the L0d->L1 writeback -> all warps sync again so nobody
  // retires before the flush has been issued.
  //
  // BARRIER PLACEMENT: both vx_bar's are OUTSIDE the single-thread guard.  A
  // vx_bar inside an `if (tid == 0)`-style divergent region gets duplicated by
  // llvm into both paths and hangs the Synchronizer; the guard also carries an
  // explicit else-nop, which is the documented defence against that duplication.
  mu_barrier(1, MU_NUM_CORES * FB_NWARPS);
  {
    const uint32_t warp_rr = tid_in_threadblock / MU_NUM_THREADS; // warp_id_rr
    const uint32_t lane    = tid_in_threadblock % MU_NUM_THREADS;
    // warp_rr < MU_NUM_CORES picks warp 0 of each core exactly once
    // (mu_schedule.cpp: warp_id_rr = warp_id_in_core*cores + core_id)
    if (warp_rr < MU_NUM_CORES && lane == 0) {
      // FB_PREFLUSH: mu_barrier synchronizes WARPS, not MEMORY.  After barrier 1
      // every warp has issued its stores, but each warp's own LSU store queue may
      // still be draining (LSU.scala: retirement is per-warp, gated on that warp's
      // logicalHead).  mu_fence() below drains only THIS warp's queue, so without a
      // pause the flush can sweep before the other warps' stores reach L0d.
      // Cheap: only one thread per core spins, so it costs no SIMT-wide time.
      fb_pause(FB_PREFLUSH);
#ifndef FB_BARRIER_ONLY
      // FB_BARRIER_ONLY isolates the two barriers from the fence + flush store, so a
      // hardware-only hang can be attributed to one primitive rather than the trio.
      mu_fence();
      fb_flush_l0d();
#endif
      fb_pause(500);            // let the writeback drain before anyone retires
    } else {
      asm volatile("nop");      // defeat llvm branch/barrier duplication
    }
  }
  mu_barrier(2, MU_NUM_CORES * FB_NWARPS);
  FB_DRAIN_T1(_dtf0, 10);
#endif

#ifdef FB_EVICT_FLUSH
  FB_DRAIN_T0(_dte0);
  // ---- CAPACITY-EVICTION DRAIN: get L0d out to DRAM WITHOUT the flush unit --------------
  // Motivation, measured 2026-09-14 on the Fix 2b bitstream (BDF 09:00.0): the explicit
  // (cid<<9)+0x80300 flush writes back NOTHING there.  fb_spinflush returns total_lines=192,
  // first_bad at word 3072 = line 192, i.e. the last 64 lines -- exactly one L0d capacity --
  // never leave the cache, while the 192 lines that had already been pushed out by capacity
  // eviction arrive fine.  Same ELF gives 256 of 256 on the tapeout bitstream.
  // The two writeback paths are independent: MSHR evictions use source ids {0,1,2,3}, which
  // are inside this client's declared IdRange, whereas Fix 2 moved the flush unit's voluntary
  // releases to {5,6,7}, which are outside it (rocket HellaCache declares cache [0,nMSHRs)
  // and MMIO [nMSHRs,nMSHRs+nMMIOs), so [0,5) in this build).  Capacity eviction therefore
  // still works on a bitstream whose flush unit does not.
  //
  // So: after the real stores, walk a scratch region big enough to evict every line in L0d.
  // The dirty scrub lines left behind at the end do not matter -- nothing reads them.
  // This is a KERNEL-SIDE drain that is correct on BOTH bitstreams, and unlike the manual
  // flush it depends on no property of the flush unit at all.
#ifndef FB_EVICT_LINES
#define FB_EVICT_LINES 64u         // EXACT: L0dCacheConfig is nSets=64, nWays=1 (direct mapped,
                                   // 64 B lines = 4 KB), RadianceConfigs.scala:82-88.  64
                                   // consecutive lines therefore hit all 64 sets exactly once,
                                   // so this is sufficient, not a margin.  Measured cost of the
                                   // 256-line version was 31,208 cycles on the tapeout board.
#endif
#ifndef FB_SCRUB_DEV
#define FB_SCRUB_DEV 0x1F200000u   // per-core scratch, above OUT (0x1F000000) and STAT (0x1F100000)
#endif
  mu_barrier(1, MU_NUM_CORES * FB_NWARPS);
  {
    // Same executor selection as FB_MANUAL_FLUSH: warp 0 of each core, lane 0, with the
    // explicit else-nop that stops llvm duplicating the surrounding barriers.
    const uint32_t warp_rr = tid_in_threadblock / MU_NUM_THREADS;
    const uint32_t lane    = tid_in_threadblock % MU_NUM_THREADS;
    if (warp_rr < MU_NUM_CORES && lane == 0) {
      // One store per 64 B line, so every line is a distinct set-index step.
      fb_scrub_body(FB_SCRUB_DEV, FB_EVICT_LINES);
#ifdef FB_L0I_FLUSH
      fb_flush_l0i();   // leave L0i clean for whatever image runs next
#endif
      fb_pause(500);
    } else {
      asm volatile("nop");
    }
  }
  mu_barrier(2, MU_NUM_CORES * FB_NWARPS);
  FB_DRAIN_T1(_dte0, 11);
#endif

#ifdef FB_TAPEOUT_EPILOGUE
  // THE TAPEOUT EPILOGUE.  Replaces "return and let the core assert finished" -- which at
  // occupancy >= 2 leaves cores_finished at 0 and strands every output word in cache.
  // Never returns; the host soft-resets us after reading the postbox.
  rad_tapeout_epilogue(tid_in_threadblock, MU_NUM_CORES * FB_NWARPS);
#endif
#ifdef FB_GEOM_TEXT76
// RUNG 2b: isolate SIZE from SECTION/POSITION.  fb_padded passes with a 212 KB blob in .text
// (which pushes .data/.bss to 0x10036000); rung2 fails with a 76 KB blob in .rodata at the END
// (leaving .data/.bss at 0x10003000).  Two variables moved at once.  This rung uses FA's 76 KB
// size but keeps it in .text like the passing build, so .data/.bss are pushed high again.
//   passes  => the variable is .data/.bss placement, not blob size or section
//   fails   => the variable is blob SIZE (212 KB ok, 76 KB not), which would be very odd
asm(".section .text.fbgeom76,\"ax\",@progbits\n"
    ".balign 4096\n"
    ".rept 9516\n nop\n .endr\n"
    ".previous\n");   /* NOT ".text": with -ffunction-sections the enclosing function lives in
                          .text._ZL8fb_entry..., so ".text" switches into a DIFFERENT section that
                          lld places FIRST -- orphaning the function's `ret` and making it fall
                          through into the pad.  .previous returns to the actual prior section. */
#endif

#ifdef FB_FA_GEOM
// RUNG 2: give the WORKING bring-up kernel FA's IMAGE GEOMETRY and nothing else.
// FA:      0x10000000/0x68  0x10001000/0x8c0  0x10002000/0x2d90  0x10005000/0x8  0x10006000/0x12960
// fb_padded: big blob is 212 KB of never-executed .text; FA's is 76 KB of .rodata.
// This rung swaps the inert text pad for a rodata blob of FA's size, so the image SHAPE matches
// (5 segments, ~100 KB extent, large read-only data region) while the kernel still does only the
// trivial bring-up work.  If this breaks a kernel that otherwise passes 9/10, the mechanism is
// image shape and FA's code is irrelevant.
// Emitted via inline asm, the same mechanism that successfully produced the 212 KB nop pad.
// A C array with used/retain was dropped twice: `used` stops the compiler, `retain` was ignored
// by this clang, and --gc-sections removed it -- leaving the geometry silently UNCHANGED, which
// would have made this rung a null experiment that looked like a passing result.
asm(".section .rodata.fbgeom,\"a\",@progbits\n"
    ".balign 4096\n"
    ".globl _fbgeom_start\n_fbgeom_start:\n"
    ".rept 76128\n .byte 0xA5\n .endr\n"
    ".globl _fbgeom_end\n_fbgeom_end:\n"
    ".previous\n");   /* NOT ".text": with -ffunction-sections the enclosing function lives in
                          .text._ZL8fb_entry..., so ".text" switches into a DIFFERENT section that
                          lld places FIRST -- orphaning the function's `ret` and making it fall
                          through into the pad.  .previous returns to the actual prior section. */
#endif

#ifdef FB_SMEM_FA
// FA LADDER RUNG 2 -- FA'S SMEM FOOTPRINT, fb'S TRIVIAL WORKLOAD.
// Rung 1 (geometry) came back CLEAN, so FA's image layout is exonerated.  Next difference up the
// ladder: FA uses the cluster scratchpad heavily and fb does not (fb's only shared ops are the
// 0x80300 flush triggers).  FA's SMEM map spans 0x1000 (S_SMEM) .. 0x16000 (Q spad base), i.e.
// ~86 KB of the 128 KB cluster SMEM, addressed as volatile __shared uint32_t* -- the same
// convention used here so the rung exercises the same path.
// Stride 64 words touches one 256B region per step across the whole extent without spending the
// whole kernel on stores.  Readback is VERIFIED and reported, so a rung where SMEM silently does
// not work fails loudly instead of passing as a null.
// STAT[9] is free: [0..3] are fb_entry's sentinels, [7]/[8] belong to the barrier sweep, [4..6] to
// the ctx probe.  [9] is inside STAT line 0 (bytes 0x00-0x3f) which is demonstrably host-readable.
  if (vx_thread_id() == 0) {
    // Emitted via INLINE ASM, not C loads/stores: the first version used volatile __shared and
    // the compiler still elided it -- gate showed sw.shared=2 lw.shared=1 against a baseline of 4
    // flush triggers, i.e. the SMEM traffic was GONE while the rung looked built.  Same family as
    // used/retain being dropped by --gc-sections: layout present, work absent, vacuous pass.
    uint32_t bad = 0;
    for (uint32_t a = 0x1000u; a < 0x16000u; a += 256u) {
      uint32_t w = 0x5A000000u | a, r;
      asm volatile("sw.shared %0, 0(%1)" :: "r"(w), "r"(a) : "memory");
      asm volatile("lw.shared %0, 0(%1)" : "=r"(r) : "r"(a) : "memory");
      bad += (r != w);
    }
    asm volatile("" :: "r"(bad));   /* keep bad live */
    STAT[9] = 0x5EE00000u | (bad & 0xFFFFu);
  } else { asm volatile("nop"); }
#endif

#ifdef FB_GEOM_FA5
// FA LADDER RUNG 1 -- FA'S IMAGE GEOMETRY, fb'S TRIVIAL WORKLOAD.
// Measured on the CURRENT FA build (fa_goal2), rv32 LOAD segments:
//   0x110000000/0x68   0x110001000/0x48   0x110002000/0x2db8   0x110005000/0x8   0x110006000/0x12960
//   total 88,016 B across 5 segments
// fb_spinflush is 1,556 B across the same 5 segments -- same COUNT, 56x smaller.
// So the rung must grow TWO regions, not one:
//   (a) rodata 76,128 B  -> matches FA's 0x12960 tail segment exactly (this is FB_FA_GEOM's blob)
//   (b) text   10,432 B  -> grows fb's 0x4f8 (1,272) to FA's 0x2db8 (11,704)
// Emitted via inline asm with .previous, NOT ".text": with -ffunction-sections the enclosing
// function lives in .text._ZL8fb_entry..., so a bare ".text" switches into a DIFFERENT section
// that lld places FIRST, orphaning the function's `ret`.  That trap cost this project a whole
// build lineage.  A C array with used/retain was ALSO dropped twice by --gc-sections, silently
// leaving geometry UNCHANGED -- a null experiment that looks like a passing result.
asm(".section .rodata.fageom5,\"a\",@progbits\n"
    ".balign 4096\n"
    ".globl _fageom5_rodata\n_fageom5_rodata:\n"
    ".rept 76128\n .byte 0xA5\n .endr\n"
    ".previous\n");
asm(".section .text.fageom5,\"ax\",@progbits\n"
    ".balign 4096\n"
    ".globl _fageom5_text\n_fageom5_text:\n"
    ".rept 868\n nop\n .endr\n"   /* 1304-436: muon nops are 8B (verified 0x...000/008/010), so the
                                  3,488 B seg2 overshoot is 436 nops, NOT 872.  Trimming to 868
                                  puts seg2 at FA's 0x2db8 and every following segment on FA's own
                                  page addresses, removing the offset confound BEFORE the run. */
    ".previous\n");
#endif

#ifdef FB_PAD_IMAGE
// SIZE A/B: pad this (working) kernel's .text so its image spans the same range as FA's, whose
// rv32 top is 0x10018960 = 100,704 bytes past the 0x10000000 entry.  Nothing calls this; it exists
// only to move the top of the image.  If the padded kernel STOPS working, size/fetch is the cause
// and FA's code is exonerated entirely; if it still works, size is excluded.
asm(".section .text.fbpad,\"ax\",@progbits\n"
    ".balign 4096\n"
    ".globl _fb_pad_start\n_fb_pad_start:\n"
    ".rept 26000\n nop\n .endr\n"
    ".globl _fb_pad_end\n_fb_pad_end:\n"
    ".previous\n");   /* NOT ".text": with -ffunction-sections the enclosing function lives in
                          .text._ZL8fb_entry..., so ".text" switches into a DIFFERENT section that
                          lld places FIRST -- orphaning the function's `ret` and making it fall
                          through into the pad.  .previous returns to the actual prior section. */
#endif

#ifdef FB_ENDSPIN
  // Sanctioned end-of-kernel shape: finish stores, then SPIN so no warp retires.  With the cores
  // never finished the automatic (one-shot, drain-less) flush at MuonTile.scala:370-374 never
  // fires, so whatever the host reads back is attributable to the HOST-TRIGGERED flush alone.
  for (;;) { asm volatile("" ::: "memory"); }
#endif

#ifdef FB_ALLFENCE
  // ALL-WARPS FENCE AT EXIT.  A fence retires against ITS OWN warp's store queue (LSU.scala:520-525:
  // fences are "fake entries" that retire at the queue head once older entries are done), so a
  // single-warp fence drains one warp and leaves the rest uncovered.  Earlier I read "3 of 4 warps
  // never retired the fence" as a deadlock and removed the fence from three warps -- but that is
  // the fence WORKING and refusing to retire while those warps' stores were outstanding, i.e.
  // reporting the bug rather than causing it.
  // Barriers sit OUTSIDE any divergent region (vx_bar inside a guard gets duplicated by llvm and
  // hangs the Synchronizer); every warp executes the fence.
  mu_barrier(1, MU_NUM_CORES * FB_NWARPS);
  mu_fence();
  mu_barrier(2, MU_NUM_CORES * FB_NWARPS);
#endif

#ifdef FB_BARSWEEP
  // BARRIER-COST SWEEP, cold and warm in one run, with INCREMENTAL STAMPING.
  //
  // v1 of this instrument stored all ten stamps at the END, i.e. BEHIND the very thing being
  // measured: on hardware the run died partway (3856/4096 still poison, stat[] entirely poison)
  // and reported nothing at all.  Each window is now stamped the moment it closes, and STAT[7]
  // carries a progress counter, so a run that dies in a later group still yields the earlier ones.
  //
  // Window shape: t=mcycle; barriers; t'=mcycle; STORE.  The store sits BETWEEN windows, never
  // inside one, so the deltas stay free of memory traffic (verified in the disassembly).
  // The magic is written BEFORE the first barrier so "GPU never reached the sweep" is
  // distinguishable from "the sweep ran and the numbers are zero".
  if (tid_in_threadblock == 0) { STAT[8] = 0xBA44BA44u; STAT[7] = 0u; }
#define FB_STAMP(slot, n, prog) do {                                                    \
    const uint32_t _a = fb_mcycle();                                                    \
    FB_BARN(n);                                                                         \
    const uint32_t _b = fb_mcycle();                                                    \
    if (tid_in_threadblock == 0) { STAT[slot] = _b - _a; STAT[7] = (prog); }             \
    else { asm volatile("nop"); }                                                       \
  } while (0)
  FB_STAMP(9,  0, 1);   // instrument overhead: two adjacent csrr, zero barriers
  FB_STAMP(10, 1, 2);   // cold 1
  FB_STAMP(11, 2, 3);   // cold 2
  FB_STAMP(12, 4, 4);   // cold 4
  FB_STAMP(13, 8, 5);   // cold 8
  FB_STAMP(14, 1, 6);   // warm 1
  FB_STAMP(15, 2, 7);   // warm 2
  FB_STAMP(16, 4, 8);   // warm 4
  FB_STAMP(17, 8, 9);   // warm 8
#endif

#ifndef FB_NO_FENCE
  // LEGACY / BROKEN-ON-PURPOSE PATH, kept so the misuse stays reproducible:
  // every warp fences here, which wedges warps 0..N-2 forever.



#ifdef FB_SELF_EVICT
  // one thread per core; all-lanes would just repeat the same 128 stores 16x
  if ((tid_in_threadblock % MU_NUM_THREADS) == 0 &&
      (tid_in_threadblock / MU_NUM_THREADS) < MU_NUM_CORES) { fb_self_evict(); }
  else { asm volatile("nop"); }
#endif

  mu_fence();
#endif
}

#ifdef FB_SELF_EVICT
// SELF-EVICTION EPILOGUE -- export the output WITHOUT using the flush writeback at all.
// L0d is 64 sets x 1 way x 64 B = 4 KB, direct-mapped.  Writing one line per set to a private
// scratch region evicts whatever occupied that set, and a dirty eviction writes back through the
// normal path -- which is independent of CacheFlushUnit, its source ids, and the one-shot
// `finished` sweep.  Two passes over all 64 sets guarantees every set is displaced.
// Scratch is per-core so the two cores cannot evict each other's lines.
//   256/256 exported => the wedge is confined to the flush path
//   still ~202       => it is not, and the flush-unit theory narrows or dies
static inline void fb_self_evict() {
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *scratch = (volatile uint32_t *)(0x1F200000u + (cid << 16));
  for (uint32_t i = 0; i < 128u; i++) scratch[i * 16u] = i;   // 16 words = 64 B stride = 1 per set
  asm volatile("" ::: "memory");
}
#endif

#ifdef FB_BSS_FIX
// FIX CANDIDATE.  Two defects compound here:
//  (1) .bss is never zeroed (mu_start.S:47-53 commented out) and lies outside every LOAD segment,
//      so schedule_context holds stale DRAM from whatever ran before.
//  (2) mu_schedule writes schedule_context from CORE 0 ONLY, then does mu_fence() + mu_barrier.
//      mu_fence drains core 0's store QUEUE but does NOT write back L0d, and L0d is not coherent
//      between cores -- so CORE 1 reads schedule_context from DRAM, never seeing core 0's write.
// Whether it works therefore depends on what DRAM already held, which is why an identical ELF
// passes sometimes: a previous run of the SAME binary left the correct callback at the same
// address.  Here core 0 writes the context itself and FLUSHES L0d, so the value is in DRAM before
// any core reads it.  Layout mirrors mu_schedule.cpp's Context{callback, arg, occupancy}.
static inline void fb_fix_ctx() {
  if (vx_core_id() == 0 && vx_thread_id() == 0) {
    volatile uint32_t *ctx = (volatile uint32_t *)FB_BSS_FIX;
    ctx[0] = (uint32_t)(uintptr_t)&fb_entry;   // callback
    ctx[1] = 0u;                               // arg
    ctx[2] = FB_NWARPS;                        // occupancy
    asm volatile("fence" ::: "memory");
    volatile uint32_t *fp = (volatile uint32_t *)(0x80300u);   // core 0's L0d flush
    asm volatile("sw.shared x0, 0(%0)" :: "r"(fp) : "memory");
    { uint32_t n = 2000; asm volatile("1: addi %0,%0,-1\n\t bnez %0,1b\n\t" : "+r"(n)); }
  } else { asm volatile("nop"); }
}
#endif

#ifdef FB_BSS_PROBE
// EXPERIMENT: is .bss stale DRAM at kernel entry?
// mu_start.S:47-53 has the BSS clear COMMENTED OUT ("TODO: re-enable for global/statics"), and
// .bss is NOBITS lying OUTSIDE every LOAD segment (seg @0x10001000 has filesz==memsz==0x4c,
// ending 0x1000104c, while schedule_context sits at 0x10001080).  So nothing -- not the loader,
// not the runtime -- ever writes it, and it holds whatever the previous run left in that DRAM.
// schedule_context is mu_schedule's callback pointer / arg / occupancy, so garbage there sends
// the GPU to a garbage PC.  Read it BEFORE mu_schedule can overwrite it, and flush so the host
// can see what we read.
static inline void fb_probe_bss() {
  const volatile uint32_t *bss = (const volatile uint32_t *)FB_BSS_PROBE;   // schedule_context
  uint32_t w[4]; for (int i = 0; i < 4; i++) w[i] = bss[i];
  STAT[20] = 0xB55B55B5u; for (int i = 0; i < 4; i++) STAT[21 + i] = w[i];
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *fp = (volatile uint32_t *)((cid << 9) + 0x80300u);
  asm volatile("sw.shared x0, 0(%0)" :: "r"(fp) : "memory");
}
#endif

int main() {
#ifdef FB_CTX_PROBE
  // DIRECT MEASUREMENT OF MODE A, from the one place that still runs when dispatch fails.
  // You cannot instrument "the kernel never ran" from inside the kernel: in Mode A the callback
  // never executes, so nothing in fb_entry can ever report.  But Mode A is a DISPATCH failure, so
  // main() DID run -- it had to, in order to call mu_schedule.  Probe here, before mu_schedule.
  //
  // PLACEMENT.  The previous probe wrote STAT[20] = byte offset 0x50, which is in the SECOND 64B
  // line (0x40-0x7f) -- six words past the visibility boundary -- and was never readable on any
  // run.  STAT[0..15] is line 0 (0x00-0x3f) and is demonstrably visible in every log we have.
  // STAT[0..3] are fb_entry's sentinels and STAT[7]/STAT[8] belong to the barrier sweep
  // (STAT[8]=0xBA44BA44, STAT[7]=prog), so the only free slots inside line 0 are [4..6].
  //
  // FLUSH HERE, before mu_schedule: on a Mode A run the end-of-kernel flush never happens, so the
  // value only reaches DRAM if we push it now.
  //
  // PER-CORE slots make cross-core incoherence directly visible:
  //   STAT[4] = core 0's observed callback,  STAT[5] = core 1's observed callback,
  //   STAT[6] = the EXPECTED callback (&fb_entry), written by whichever core gets there.
  // Outcomes: [4]!=[5]            -> cross-core L0d incoherence, demonstrated on that run
  //           both != [6] on fail -> uninitialised schedule_context confirmed
  //           both == [6] on fail -> callback hypothesis dead; stop spending slots on ctxpc power
  if (vx_thread_id() == 0) {
    const uint32_t cid = (uint32_t)vx_core_id();
    const volatile uint32_t *ctx = (const volatile uint32_t *)FB_CTX_PROBE;
    const uint32_t cb = ctx[0];                 // observed callback, BEFORE any fix writes it
    STAT[4 + (cid & 1u)] = cb;
    STAT[6] = (uint32_t)(uintptr_t)&fb_entry;   // expected value, for comparison
    asm volatile("fence" ::: "memory");
    volatile uint32_t *fp = (volatile uint32_t *)((cid << 9) + 0x80300u);
    asm volatile("sw.shared x0, 0(%0)" :: "r"(fp) : "memory");
    { uint32_t n = 2000; asm volatile("1: addi %0,%0,-1\n\t bnez %0,1b\n\t" : "+r"(n)); }
  } else { asm volatile("nop"); }
#endif
#ifdef FB_CTX_PERCORE
  fb_ctx_percore();
#endif
#ifdef FB_BSS_ZERO
  fb_zero_bss();
#endif
#ifdef FB_BSS_PROBE
  fb_probe_bss();
#endif
#ifdef FB_BSS_FIX
  fb_fix_ctx();
#endif
  mu_schedule(fb_entry, nullptr, FB_NWARPS);
  return 0;
}
