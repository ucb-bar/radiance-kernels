/*
 * nightly/device.h: Muon-side launch, timing and identity helpers for the nightly RTL
 * (see lib/include/nightly/README.md for the RTL versions).
 *
 * Launch protocol, shared with nightly/host.h:
 *   1. every core's boot warp waits in nightly_wait_launch() until the host writes
 *      NIGHTLY_LAUNCH_MAGIC to its cluster's printBuf LAUNCH slot;
 *   2. the kernel runs under mu_schedule(); its entry ends with nightly_kernel_end(), which
 *      synchronises the threadblock, writes the core's L0d back with `fence`, and posts the
 *      core's end mcycle to printBuf;
 *   3. the host waits for every core's END slot, then reads the results from GPU DRAM.
 * Posting END after the fence is what makes the host read safe: the per-core `finished` signal
 * has no memory-drain term, so all_finished alone does not prove the stores left the L0d.
 */
#ifndef NIGHTLY_DEVICE_H
#define NIGHTLY_DEVICE_H

#include <stdint.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <nightly/pb.h>

/* Number of clusters (SMs) the kernel is built for: 1 = RadianceSingleSMHBMConfig,
 * 2 = RadianceHBMConfig.  Set by the kernel Makefile per ELF. */
#ifndef NIGHTLY_CLUSTERS
#define NIGHTLY_CLUSTERS 1
#endif

/* Warps per core the kernel runs (set by nightly.mk from OCC, which also caps the compiler's
 * allocatable GPRs to 256/OCC).  The boot code spawns exactly this many warps, so no idle warp
 * holds physical registers. */
#ifndef NIGHTLY_OCC
#define NIGHTLY_OCC 8
#endif
extern "C" uint32_t __mu_num_warps = NIGHTLY_OCC;

/* Barrier ids: mu_schedule uses 0.  nightly_kernel_end uses 15.  Kernels use 1..14. */
#define NIGHTLY_BAR_END 15u

static inline uint32_t nightly_mcycle(void) {
  uint32_t c;
  asm volatile("csrr %0, mcycle" : "=r"(c) :: "memory");
  return c;
}

static inline uint32_t nightly_cluster_id(void) {
  uint32_t c;
  asm volatile("csrr %0, %1" : "=r"(c) : "i"(MU_CSR_CLUSTER_ID));
  return c;
}

static inline void nightly_pb_put(uint32_t slot, uint32_t v) {
  const uint32_t a = NIGHTLY_PB_DEV_BASE + slot * 8u;
  asm volatile("sw.shared %0, 0(%1)" :: "r"(v), "r"(a) : "memory");
  asm volatile("sw.shared %0, 4(%1)" :: "r"(v), "r"(a) : "memory");
}

static inline uint32_t nightly_pb_get(uint32_t slot) {
  const uint32_t a = NIGHTLY_PB_DEV_BASE + slot * 8u;
  uint32_t v;
  asm volatile("lw.shared %0, 0(%1)" : "=r"(v) : "r"(a) : "memory");
  return v;
}

/* Called by the boot warp (one thread) of every core, before mu_schedule(). */
static inline void nightly_wait_launch(void) {
  while (nightly_pb_get(NIGHTLY_PB_LAUNCH) != NIGHTLY_LAUNCH_MAGIC) {
    asm volatile("nop");
  }
  nightly_pb_put(NIGHTLY_PB_START + (uint32_t)vx_core_id(), nightly_mcycle());
}

/* Kernel-defined phase stamp (cluster core 0, one thread). */
static inline void nightly_phase(uint32_t idx) {
  nightly_pb_put(NIGHTLY_PB_PHASE + (idx & 7u), nightly_mcycle());
}

/* Progress value for the host's wait loop (one thread; e.g. (tile << 16) | block).  RTL simulation
 * shows no output until the host finishes, so this is how a hang becomes visible while it runs. */
static inline void nightly_heartbeat(uint32_t v) { nightly_pb_put(NIGHTLY_PB_HEARTBEAT, v); }

/* End of a kernel entry: every thread of the threadblock must call this exactly once.
 * noinline + convergent: inlined, clang tail-duplicated its barrier into both arms of a preceding
 * `if (lead) ... else nop` (FA v2: `if (lead) mx::fence()`), so warp 0 ran barrier 15 twice and
 * hung.  A convergent call is not duplicated into divergent paths. */
__attribute__((noinline, convergent))
static void nightly_kernel_end(uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
  const uint32_t warps = threads_per_threadblock / MU_NUM_THREADS;
  mu_barrier(NIGHTLY_BAR_END, warps);
  /* threadblock warp w runs on core (w % MU_NUM_CORES): warps 0..MU_NUM_CORES-1 cover each core once */
  if (tid_in_threadblock < MU_NUM_CORES * MU_NUM_THREADS) {
    mu_fence();
    if ((tid_in_threadblock % MU_NUM_THREADS) == 0)
      nightly_pb_put(NIGHTLY_PB_END + (uint32_t)vx_core_id(), nightly_mcycle());
  } else {
    asm volatile("nop");
  }
}

/* Standard GPU main(): wait for the host, run `entry` at NIGHTLY_OCC warps per core. */
static inline int nightly_main(mu_schedule_callback entry, void *arg) {
  nightly_wait_launch();
  mu_schedule(entry, arg, NIGHTLY_OCC);
  return 0;
}

#endif
