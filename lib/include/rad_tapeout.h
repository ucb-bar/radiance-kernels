/*
 * rad_tapeout.h: cache-maintenance routines for the taped-out Radiance part.
 *
 * The taped-out RTL (radiance 00cb7d2, silicon tapeout-329a, and FPGA bitstream
 * fe4dc316 / FireSimRadianceSingleClusterConfig) has a defective L0d flush unit.
 * This header provides the drain that works on it, the cache geometry the drain
 * depends on, and the MMIO flush behind an explicit opt-in.
 *
 * Use rad_l0d_drain() -- or the RAD_L0D_DRAIN() wrapper -- at every kernel
 * boundary where host-visible results must be in DRAM.
 */

#ifndef __RAD_TAPEOUT_H__
#define __RAD_TAPEOUT_H__

#include <stdint.h>
/* The device half needs the Muon intrinsics; the rv64 HOST half must not pull them in -- they
 * drag in the libc++ type_traits set, which the host toolchain cannot parse.  Define
 * RAD_TAPEOUT_HOST before including this from host code: you then get the address map and the
 * host helpers, and none of the device code. */
#ifndef RAD_TAPEOUT_HOST
#include <mu_intrinsics.h>
#include <vx_intrinsics.h>
#endif

/* ---------------------------------------------------------------------------
 * Cache geometry.  Source: RadianceConfigs.scala:82-90 (L0dCacheConfig,
 * L0iCacheConfig, L1CacheConfig).  RAD_L0D_LINES is the number of distinct
 * lines the L0d can hold, and because the L0d is direct mapped it is also the
 * number of consecutive lines a drain must walk to touch every set once.
 * ------------------------------------------------------------------------- */
#define RAD_L0D_SETS        64u
#define RAD_L0D_WAYS        1u
#define RAD_L0D_LINE_BYTES  64u
#define RAD_L0D_LINES       (RAD_L0D_SETS * RAD_L0D_WAYS)          /* 64   */
#define RAD_L0D_BYTES       (RAD_L0D_LINES * RAD_L0D_LINE_BYTES)   /* 4 KB */

#define RAD_L0I_SETS        512u
#define RAD_L0I_WAYS        1u
#define RAD_L0I_LINE_BYTES  32u
#define RAD_L0I_BYTES       (RAD_L0I_SETS * RAD_L0I_WAYS * RAD_L0I_LINE_BYTES) /* 16 KB */

#define RAD_L1_SETS         256u
#define RAD_L1_WAYS         4u
#define RAD_L1_LINE_BYTES   32u
#define RAD_L1_BYTES        (RAD_L1_SETS * RAD_L1_WAYS * RAD_L1_LINE_BYTES)  /* 32 KB */

/* ---------------------------------------------------------------------------
 * Flush-unit MMIO.  MuonTile.scala:196 gives the L0i flush register at
 * muonParams.peripheralAddr and :246 the L0d one at peripheralAddr + 0x100, so
 * core i sees them at (i<<9)+0x80200 and (i<<9)+0x80300 in cluster space.
 * ------------------------------------------------------------------------- */
#define RAD_L0I_FLUSH_MMIO(cid)  ((((uint32_t)(cid)) << 9) + 0x80200u)
#define RAD_L0D_FLUSH_MMIO(cid)  ((((uint32_t)(cid)) << 9) + 0x80300u)

/* Per-core scratch for the capacity-eviction drain.  Must be device GMEM that
 * no one reads, and must not alias any live data: the drain leaves its own
 * dirty lines behind.  Each core takes RAD_L0D_BYTES starting at
 * RAD_DRAIN_SCRATCH + cid * RAD_L0D_BYTES.  Override per kernel if the default
 * collides with an output or stat window. */
#ifndef RAD_DRAIN_SCRATCH
#define RAD_DRAIN_SCRATCH  0x1F200000u
#endif

/* ---------------------------------------------------------------------------
 * CAPACITY-EVICTION DRAIN -- the supported way to get L0d out to DRAM.
 *
 * Walks RAD_L0D_LINES consecutive 64 B lines of per-core scratch.  On a
 * direct-mapped 64-set cache that hits every set exactly once, so every dirty
 * line is evicted through the MSHR writeback path.  MSHR evictions use source
 * ids inside the client's declared IdRange, so this path is independent of the
 * flush unit.
 *
 * Measured: 256 of 256 output lines bit-exact, 3 of 3 runs on each of the two
 * U250 boards.  Cost 15,727 cycles for 64 lines (31,391 for a 256-line version).
 *
 * Call from ONE lane of ONE warp per core, with the caller's barriers outside
 * the divergent guard; RAD_L0D_DRAIN() below does that.  The fence at the end
 * retires this warp's store queue only, so other warps must already have been
 * synchronised by the caller's first barrier.
 * ------------------------------------------------------------------------- */
#ifndef RAD_TAPEOUT_HOST
static inline void rad_l0d_drain(void) {
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *scrub =
      (volatile uint32_t *)(RAD_DRAIN_SCRATCH + cid * RAD_L0D_BYTES);
  for (uint32_t l = 0; l < RAD_L0D_LINES; l++) {
    scrub[l * (RAD_L0D_LINE_BYTES / 4u)] = l + 1u;
  }
  mu_fence();
}

/* Drain a caller-chosen window instead of RAD_DRAIN_SCRATCH.  `lines` must be
 * at least RAD_L0D_LINES to be a complete drain. */
static inline void rad_l0d_drain_at(uint32_t base, uint32_t lines) {
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *scrub =
      (volatile uint32_t *)(base + cid * (lines * RAD_L0D_LINE_BYTES));
  for (uint32_t l = 0; l < lines; l++) {
    scrub[l * (RAD_L0D_LINE_BYTES / 4u)] = l + 1u;
  }
  mu_fence();
}

/* register-only delay; never `volatile` on the counter, because the stack is
 * GMEM here and a spilled counter turns a 128-cycle pad into DRAM traffic. */
static inline void rad_pause(uint32_t n) {
  asm volatile("1:\n\t addi %0, %0, -1\n\t bnez %0, 1b\n\t" : "+r"(n));
}

/* ---------------------------------------------------------------------------
 * RAD_L0D_DRAIN(tid_in_threadblock, warps_per_block, bar_a, bar_b)
 *
 * Complete kernel-boundary drain: barrier, one executor per core, barrier.
 * `bar_a` and `bar_b` are barrier ids the kernel is not otherwise using.
 * Both barriers sit outside the divergent guard and the guard carries an
 * explicit else-nop, because llvm duplicates a vx_bar placed inside an
 * `if (lane == 0)` region into both paths and hangs the synchroniser.
 * ------------------------------------------------------------------------- */
#define RAD_L0D_DRAIN(tid_in_threadblock, warps_per_block, bar_a, bar_b)      \
  do {                                                                        \
    mu_barrier((bar_a), (warps_per_block));                                   \
    {                                                                         \
      const uint32_t _rd_warp = (tid_in_threadblock) / MU_NUM_THREADS;        \
      const uint32_t _rd_lane = (tid_in_threadblock) % MU_NUM_THREADS;        \
      /* warp_id_rr = warp_in_core*ncores + core, so _rd_warp < MU_NUM_CORES  \
       * selects warp 0 of each core. */                                      \
      if (_rd_warp < MU_NUM_CORES && _rd_lane == 0u) {                        \
        rad_l0d_drain();                                                      \
      } else {                                                                \
        asm volatile("nop");                                                  \
      }                                                                       \
    }                                                                         \
    mu_barrier((bar_b), (warps_per_block));                                   \
  } while (0)

/* ---------------------------------------------------------------------------
 * L0i invalidate.  The core-finish edge (MuonTile.scala:370-374) invalidates
 * L0i, but only for a kernel that finishes; a hung predecessor leaves its image
 * cached, and L0i is 16 KB direct mapped, which is large enough to hold a whole
 * bring-up kernel.  Invalidating at the END of a kernel is what lets the next
 * image be fetched.
 *
 * STATUS: unproven.  The 6-of-6 A/B that motivated it never ran its control, so
 * "the invalidate clears stale L0i" and "the invalidate wedges the kernel" fit
 * the same data.  Do not enable it in a measurement build without a control.
 * ------------------------------------------------------------------------- */
static inline void rad_l0i_invalidate(void) {
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *p = (volatile uint32_t *)RAD_L0I_FLUSH_MMIO(cid);
  asm volatile("sw.shared x0, 0(%0)" :: "r"(p) : "memory");
}

/* ---------------------------------------------------------------------------
 * L0d FLUSH-UNIT MMIO -- BROKEN ON THE TAPED-OUT PART.  DO NOT USE.
 *
 * A store to (cid<<9)+0x80300 wedges the L0d permanently, and the next memory
 * operation the core issues never completes.  Because mu_schedule() starts with
 * mu_fence(), a kernel that triggers this flush anywhere before or during
 * mu_schedule() hangs the whole GPU with cores_finished = 0.
 *
 * Mechanism (MuonDCache.scala:543-565): the flush raises `preflushing`, and
 * `flushing = preflushing || RegNext(preflushing) || flush.io.busy` holds
 * `io.cpu.req.ready` low for its duration.  `flush.io.busy` is
 * `flushing || (inFlights > 0)`, and `inFlights` only falls on
 * `wb_resp_fire`, so a flush whose voluntary releases are not accounted leaves
 * busy asserted forever and the L0d never accepts another request.
 *
 * Root-caused 2026-09-16 by FireSim metasim on the tapeout RTL:
 *   mark store + this flush   -> hangs, both cores stalled on mu_schedule's
 *                                fence at pc 0x10004c40
 *   no mark store, no flush   -> completes
 *   mark store, no flush      -> completes
 * and, isolating the mechanism, a build whose main is
 *   mark store, this flush, then a PLAIN LOAD, no fence anywhere
 * hangs on the load.  So the cache stops serving requests entirely; this is not
 * fence ordering.
 *
 * Kept here only so a kernel that deliberately tests the flush unit does not
 * re-derive the address.  Requires RAD_ALLOW_L0D_FLUSH_MMIO.
 * ------------------------------------------------------------------------- */
#ifdef RAD_ALLOW_L0D_FLUSH_MMIO
static inline void rad_l0d_flush_mmio_unsafe(void) {
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *p = (volatile uint32_t *)RAD_L0D_FLUSH_MMIO(cid);
  asm volatile("sw.shared x0, 0(%0)" :: "r"(p) : "memory");
}
#endif


/* ===========================================================================================
 * THE TAPEOUT EPILOGUE -- never let a core assert `finished`.
 *
 * WHY.  Measured 2026-09-21 with an uncached printBuf beacon, on both U250 boards: at occupancy
 * >= 2 the kernel runs to COMPLETION -- the final stage mark, every warp past the last barrier,
 * and mu_schedule() returning -- and then `cores_finished` stays 0 0 0 0 and not one output word
 * reaches DRAM.  It is not a hang.  WarpScheduler.scala:211 defines
 *     io.finished := VecInit(pcTracker.map(!_.valid)).asUInt.andR
 * so a core reports finished only when EVERY warp's pcTracker entry is invalid, and the only
 * thing that clears one is a commit with setTmask === 0 (WarpScheduler.scala:184-190).  At
 * occupancy 1 `vx_wspawn(1, ...)` spawns no worker at all, so that path is never taken and the
 * kernel works; at occupancy >= 2 it is taken and the core never reports finished.
 *
 * On top of that, MuonTile.scala:371-374 fires the L0d/L0i flush unit on the RISING EDGE of
 * finished -- the same flush unit that wedges the L0d (see rad_l0d_flush_mmio_unsafe below).  So
 * asserting finished is not merely unreliable, it puts the tile in a bad state.
 *
 * THE EPILOGUE SIDESTEPS BOTH.  No core ever asserts finished:
 *   1. every warp fences and rendezvous;
 *   2. warp 0 of each core drains L0d AND L1 by capacity -- no flush unit anywhere;
 *   3. warps other than warp 0 post "stopped" to the postbox and `vx_tmc 0`;
 *   4. warp 0 of each core waits for them, posts DONE for its core, and SPINS FOREVER;
 *   5. the host polls the postbox, sees DONE from every core, and asserts GPU soft reset.
 * Because warp 0 never retires, `finished` never rises, the finish-edge flush never fires, and
 * the data is already in DRAM when the host reads it.
 *
 * The postbox is printBuf (cacheable = false), so every step of this is visible to the host even
 * though the caches are in whatever state the kernel left them.
 *
 * Both barriers are outside the divergent guards and each guard carries an explicit else-nop:
 * llvm duplicates a vx_bar placed inside an `if (lane == 0)` region into both paths and hangs the
 * synchroniser.  And every barrier happens BEFORE any warp executes tmc 0 -- a barrier whose
 * participants have already exited never completes.
 * ========================================================================================= */

#endif /* !RAD_TAPEOUT_HOST -- the postbox map below is shared with the host */

/* postbox layout, in printBuf.  8-byte slots; each 32-bit value is written to both halves so an
 * 8-byte host read sees it however the beat is assembled. */
#define RAD_PB_BASE            0x80000u
#define RAD_PB_SLOT(i)         (RAD_PB_BASE + (i) * 8u)
/* SLOT MAP -- deliberately clear of 0..15, which kernels use for their own progress beacons.
 * Bug found 2026-09-21: the epilogue originally posted core DONE to slots 0/1, which FA's beacon
 * already used for entry/stage, so the host read a beacon value and reported a false timeout even
 * on runs where the epilogue had completed. Keep these two maps disjoint. */
/* *** THE COMPLETE printBuf SLOT MAP.  CHECK IT BEFORE ADDING A SLOT. ***  64 slots of 8 B.
 *   0..15  reserved for KERNEL beacons (FA uses 0..7)
 *   16,17  epilogue: per-core DONE
 *   20,21  epilogue: per-core trace
 *   24..31 epilogue: per-warp STOP  (24 + warp_rr, up to 8 warps)
 *   32..63 free for diagnostics
 * Collisions here have produced THREE false readings in this campaign: a working epilogue
 * reported as a timeout, and twice a diagnostic that read back another field's magic. */
#define RAD_PB_EPI_CORE        16u   /* + core id : warp 0 posts DONE when its core is quiesced */
#define RAD_PB_EPI_WARP        24u   /* + warp_rr : each stopping warp posts before tmc 0       */
#define RAD_PB_EPI_TRACE       20u   /* + core id : how far warp 0 got, for diagnosing timeouts */
#define RAD_EPI_T_BARA         0xA0u /* passed barrier A            */
#define RAD_EPI_T_DRAIN        0xA1u /* drains done                 */
#define RAD_EPI_T_BARB         0xA2u /* passed barrier B            */
#define RAD_EPI_T_POLLED       0xA3u /* finished waiting for warps  */
#define RAD_EPI_DONE_MAGIC     0xD01E0000u
#define RAD_EPI_STOP_MAGIC     0x570BBED0u
/* host-side addresses of the same slots (cluster base + device offset) */
#define RAD_PB_HOST(cl, i)     (0x40000000ull + 0x100000ull * (cl) + 0x80000ull + (i) * 8ull)

/* *** DO NOT ADD DELAYS BETWEEN THE SFU-CLASS OPS BELOW. ***  Tried 2026-09-21 on the theory that
 * SFUPipe.scala:68-71 `assert(!reqSent)` (simulation-only, so silicon drops the overlapping
 * request instead) meant barrier/barrier/tmc were colliding.  Inserting 2000-cycle gaps made it
 * WORSE: the kernel-source call site went from 0/4096 to 288/4096 poison, reproducibly.  The
 * hypothesis is refuted and the spacing is harmful. */
#ifndef RAD_EPI_SETTLE
#define RAD_EPI_SETTLE 20000u   /* cycles between the two drain passes */
#endif
#ifndef RAD_EPI_BAR_A
#define RAD_EPI_BAR_A 13u
#endif
#ifndef RAD_EPI_BAR_B
#define RAD_EPI_BAR_B 14u
#endif

#ifndef RAD_TAPEOUT_HOST
static inline void rad_pb_put(uint32_t slot, uint32_t v) {
  volatile uint32_t *p = (volatile uint32_t *)RAD_PB_SLOT(slot);
  asm volatile("sw.shared %0, 0(%1)" :: "r"(v), "r"(p) : "memory");
  asm volatile("sw.shared %0, 0(%1)" :: "r"(v), "r"(p + 1) : "memory");
}
static inline uint32_t rad_pb_get(uint32_t slot) {
  volatile uint32_t *p = (volatile uint32_t *)RAD_PB_SLOT(slot);
  uint32_t v; asm volatile("lw.shared %0, 0(%1)" : "=r"(v) : "r"(p) : "memory");
  return v;
}

/* Drain the CLUSTER L1 by capacity.  The L1 is instantiated with no flushAddr
 * (RadianceCluster.scala:137-142), so it has NO flush unit and capacity eviction is the only way
 * its dirty lines ever leave.  256 sets x 4 ways x 32 B: writing 4 distinct tags into every set
 * evicts the whole cache.  Walk 2x the L1 size at line stride to be certain of it. */
/* RAD_L1_DRAIN_MULT: how many L1-sizes to walk.  The L1 is shared by both cores and its
 * replacement policy is not LRU-guaranteed, so one pass over exactly L1_BYTES is not a proof of
 * full eviction -- walking a multiple of it is.  Raise this if output words go missing. */
#ifndef RAD_L1_DRAIN_MULT
#define RAD_L1_DRAIN_MULT 4u
#endif
static inline void rad_l1_drain(void) {
  const uint32_t cid = (uint32_t)vx_core_id();
  volatile uint32_t *p =
      (volatile uint32_t *)(RAD_DRAIN_SCRATCH + 0x00100000u +
                            cid * (RAD_L1_DRAIN_MULT * RAD_L1_BYTES));
  const uint32_t n = (RAD_L1_DRAIN_MULT * RAD_L1_BYTES) / RAD_L1_LINE_BYTES;
  for (uint32_t i = 0; i < n; i++) p[i * (RAD_L1_LINE_BYTES / 4u)] = i + 1u;
  mu_fence();
}

/* *** HOW TO USE THIS ***
 * Call rad_tapeout_epilogue() as the LAST statement of your kernel's entry function -- the
 * function you hand to mu_schedule() -- passing the tid_in_threadblock and warps-per-block it
 * was given:
 *     static void my_entry(void *arg, uint32_t tid, uint32_t nthreads, uint32_t blk) {
 *         ... kernel body ...
 *         rad_tapeout_epilogue(tid, nthreads / MU_NUM_THREADS);   // never returns
 *     }
 * and on the host, after releasing the GPU:
 *     if (rad_host_wait_epilogue(2, 20000000ull) == 2) rad_host_gpu_soft_reset();
 * Do NOT poll all_finished or the per-core finished bits -- by design they never assert.
 *
 * *** DO NOT use the libmuonrt-tapeout.a / RAD_TAPEOUT=1 variant for production. ***  It calls
 * this same function from mu_schedule_standalone() (i.e. after your entry function RETURNS),
 * which needs no kernel edit and is tempting -- but measured on hardware 2026-09-21 it
 * deterministically loses part of the output: 64-96 of 4096 words, bit-identical across repeats,
 * localised to O rows 48-51 (the start of the last worker warp's block) rather than to the end
 * of the buffer.  Widening the L1 sweep 2x -> 4x and adding a second drain pass after a settle
 * both changed NOTHING, so it is neither capacity nor in-flight stores; the count moves with code
 * layout, which says race.  Root cause unknown.  The in-entry call site above is 0/4096 on every
 * occupancy, repeatably, so use that until the library variant is understood.
 */
/* Epilogue trace stamp.  UNGUARDED ON PURPOSE: w, cid and the slot index are warp-uniform, so
 * every lane stores the same value to the same postbox address and control flow stays uniform.
 * A guarded form -- `if (l == 0u) { ... } else { nop; }` -- placed immediately before mu_fence()
 * or vx_bar leaves the warp unreconverged and hangs the fence; measured 2026-09-22, it broke the
 * known-good kernel-source call site (0/4096 -> 240/4096).  Define RAD_EPI_NOPROBE to remove. */
#ifdef RAD_EPI_PROBE
#define RAD_EPI_STAMP(slot, val) rad_pb_put((slot), (val))
#else
#define RAD_EPI_STAMP(slot, val) do { } while (0)
#endif

/* Spin cycles after the arrival barrier, before the single fence, so the OTHER warps' store
 * queues retire: mu_barrier syncs WARPS, not MEMORY, and mu_fence drains only the issuing warp's
 * queue (LSU retirement is per-warp).  2000/500 are the values the measured-good FA_FPGA_FLUSH
 * block uses. */
#ifndef RAD_EPI_QUEUE_SPIN
#define RAD_EPI_QUEUE_SPIN 2000u
#endif
#ifndef RAD_EPI_POST_SPIN
#define RAD_EPI_POST_SPIN 500u
#endif

/* rad_tapeout_epilogue -- call INSTEAD of returning from the kernel body.  NEVER RETURNS.
 * A function, not a macro: the barrier/tmc sequence is long enough that macro line-continuations
 * are a liability, and noinline keeps llvm from cloning the vx_bar into divergent paths. */
__attribute__((noinline, convergent))
static void rad_tapeout_epilogue(uint32_t tid_in_threadblock, uint32_t warps_per_block) {
  const uint32_t w = tid_in_threadblock / MU_NUM_THREADS;
  const uint32_t l = tid_in_threadblock % MU_NUM_THREADS;
  const uint32_t cid = (uint32_t)vx_core_id();

  /* *** NO ALL-WARP mu_fence() HERE. ***  On the taped-out part only ONE warp per core may fence.
   * With every warp fencing, warps lose the SFU request and never return -- SFUPipe's
   * assert(!reqSent) is simulation-only, so silicon drops an overlapping request silently.  This
   * epilogue used to open with an all-warp mu_fence() and that is what hung it: measured
   * 2026-09-22, the manager warps (w = warp_in_core 0) were the usual victims, intermittently,
   * and any timing shift -- returning from the kernel entry first, or one added store -- changed
   * which warps survived.  The structure below mirrors the measured-good FA_FPGA_FLUSH block:
   * arrive, let the other warps' queues retire, then fence ONCE per core inside the guard. */
  mu_barrier(RAD_EPI_BAR_A, warps_per_block);
  RAD_EPI_STAMP(52u + (w & 3u), 0xFA000000u | w);
  RAD_EPI_STAMP(48u, warps_per_block);
  if (w < MU_NUM_CORES && l == 0u) { rad_pb_put(RAD_PB_EPI_TRACE + cid, RAD_EPI_T_BARA); }
  else { asm volatile("nop"); }

  /* warp_id_rr = warp_in_core*ncores + core, so w < MU_NUM_CORES is warp 0 of each core */
  if (w < MU_NUM_CORES && l == 0u) {
#ifdef RAD_EPI_PROBE
    /* DIAGNOSTIC (2026-09-22).  Reads the first word of the line that the library call site
     * always loses (O word 3104 = line 194 = L0d set 2) and its predecessor line, BEFORE any
     * drain runs.  A poison value here means the dirty line had already left the L0d without
     * reaching DRAM -- i.e. the loss happened before the epilogue, on the way in.  A real value
     * means the line was still resident and the drain is what lost it.  Slots 32..39 are free
     * (see the slot map above).  Off by default; it perturbs the L0d by one allocation. */
    rad_pb_put(32u + cid, *(volatile uint32_t *)(uintptr_t)(0x1F003080u));
    rad_pb_put(34u + cid, *(volatile uint32_t *)(uintptr_t)(0x1F003080u - 0x80u));
    rad_pb_put(36u + cid, *(volatile uint32_t *)(uintptr_t)(0x1F003080u + 0x100u));
#endif
    /* TWO PASSES with a settle between them: mu_fence() retires at the warp's store-queue head
     * and does not wait for the L0d to accept the store, so a late store can land in a line the
     * first sweep has already passed.  (An earlier note here blamed the library call site's
     * missing words on in-flight stores.  That was wrong: the library epilogue was hanging at the
     * all-warp fence above and never ran these drains at all, so those words were residue, not
     * loss.  Widening the sweep 2x -> 4x changed nothing, consistent with that.) */
    rad_pause(RAD_EPI_QUEUE_SPIN);   /* other warps' store queues retire */
    mu_fence();                      /* the ONLY fence in this epilogue; one warp per core */
    rad_l0d_drain();
    rad_l1_drain();
    rad_pause(RAD_EPI_SETTLE);
    rad_l0d_drain();
    rad_l1_drain();
    rad_pause(RAD_EPI_POST_SPIN);
#ifdef RAD_EPI_PROBE
    rad_pb_put(38u + cid, *(volatile uint32_t *)(uintptr_t)(0x1F003080u));
#endif
    rad_pb_put(RAD_PB_EPI_TRACE + cid, RAD_EPI_T_DRAIN);
  } else {
    asm volatile("nop");   /* defeat llvm barrier duplication around the divergent guard */
  }
  mu_barrier(RAD_EPI_BAR_B, warps_per_block);
  if (w < MU_NUM_CORES && l == 0u) { rad_pb_put(RAD_PB_EPI_TRACE + cid, RAD_EPI_T_BARB); }
  else { asm volatile("nop"); }

  if (w >= MU_NUM_CORES) {
    /* every warp that is not its core's warp 0: announce, then clear our pcTracker entry */
    if (l == 0u) { rad_pb_put(RAD_PB_EPI_WARP + w, RAD_EPI_STOP_MAGIC); }
    else { asm volatile("nop"); }
    mu_fence_smem();
    vx_tmc_zero();                  /* WarpScheduler.scala:187-189 -- never returns */
  }

  if (l == 0u) {
    /* warp 0 of this core: wait for this core's other warps.  warp_rr k*ncores + cid belongs to
     * core cid, so those are the ones to wait on. */
    for (uint32_t k = 1u; k * MU_NUM_CORES + cid < warps_per_block; k++) {
      const uint32_t slot = RAD_PB_EPI_WARP + k * MU_NUM_CORES + cid;
      for (uint32_t i = 0; i < 2000000u; i++) {
        if (rad_pb_get(slot) == RAD_EPI_STOP_MAGIC) break;
      }
    }
    rad_pb_put(RAD_PB_EPI_TRACE + cid, RAD_EPI_T_POLLED);
    rad_pb_put(RAD_PB_EPI_CORE + cid, RAD_EPI_DONE_MAGIC);
    mu_fence_smem();
  } else {
    asm volatile("nop");
  }

  /* SPIN FOREVER.  Register-only, so a wedged L0d cannot matter; and because this warp never
   * retires, core.io.finished never rises, the finish-edge flush never fires, and the tile is
   * never put in the bad state.  The host soft-resets us after reading the postbox. */
  for (;;) { asm volatile("" ::: "memory"); }
}

#endif /* !RAD_TAPEOUT_HOST */

/* ===========================================================================================
 * HOST SIDE of the epilogue.  Include with RAD_TAPEOUT_HOST defined from an rv64 host .cpp.
 *
 * The GPU never asserts finished, so DO NOT poll all_finished or the per-core finished bits --
 * they will never come, by construction.  Poll the postbox instead, then soft-reset the GPU.
 * Soft reset is what tears the spinning warp 0 down; MuonTile.scala:371 suppresses the
 * finish-edge flush when the finish was caused by reset, which is exactly what we want because
 * the epilogue has already drained both cache levels by capacity.
 * ========================================================================================= */
#ifdef RAD_TAPEOUT_HOST
#ifndef RAD_HOST_GPU_RESET
#define RAD_HOST_GPU_RESET 0x41000000ull
#endif
/* Returns the number of cores that reported DONE.  `spins` bounds the wait so a broken epilogue
 * degrades to a reported timeout, never to a host hang. */
static inline int rad_host_wait_epilogue(int ncores, unsigned long long spins) {
  int done = 0;
  for (unsigned long long i = 0; i < spins && done < ncores; i++) {
    asm volatile("fence" ::: "memory");     /* never poll behind our own stores */
    done = 0;
    for (int c = 0; c < ncores; c++) {
      /* MUST be RAD_PB_EPI_CORE + c, not c: the core DONE slots moved off 0/1 when they were
       * found to collide with kernel beacons.  Reading slot c here made a WORKING epilogue report
       * a timeout while the postbox plainly held the DONE magic. */
      const unsigned int v =
          (unsigned int)*(volatile unsigned long long *)RAD_PB_HOST(0, RAD_PB_EPI_CORE + c);
      if (v == RAD_EPI_DONE_MAGIC) done++;
    }
  }
  return done;
}
/* Assert GPU soft reset.  CanHaveGPUReset.scala:85-89: any non-zero write latches reset on every
 * core, which retires the spinning warp 0 without a finish-edge flush. */
static inline void rad_host_gpu_soft_reset(void) {
  *(volatile unsigned int *)RAD_HOST_GPU_RESET = 1u;
  asm volatile("fence" ::: "memory");
}
#endif /* RAD_TAPEOUT_HOST */

#endif /* __RAD_TAPEOUT_H__ */
