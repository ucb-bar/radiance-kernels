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
#include <mu_intrinsics.h>
#include <vx_intrinsics.h>

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

#endif /* __RAD_TAPEOUT_H__ */
