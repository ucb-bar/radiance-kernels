/*
 * mu_schedule.cpp: Static scheduler/bootstrapping routine for Muon kernels.
 *
 * Spawns the correct number of warps and threads given the grid and
 * threadblock dimension, using a static mapping from local thread ID to
 * hardware thread ID.
 */

#include <mu_schedule.h>
#include <mu_intrinsics.h>
#ifdef MU_TAPEOUT_EPILOGUE
#include <rad_tapeout.h>
#endif

#define NUM_CORES_MAX 1024

extern "C" {

struct __attribute__((aligned(CACHE_LINE_BYTES))) Context {
    mu_schedule_callback callback;
    void *arg;
    uint32_t occupancy;
    uint8_t _pad[CACHE_LINE_BYTES - 3 * sizeof(uint32_t)];
};

/* Since vx_wspawn can only enter a function with no arguments, we need a
 * persistent state to restore contexts such as kernel arguments.
 * Must occupy its own cache line (64B) to avoid flush races with adjacent data. */
static volatile Context schedule_context;

static void __attribute__ ((noinline)) mu_schedule_standalone() {
    uint32_t cluster_id = 0;
    // TODO: clid not supported in assembler
    asm volatile("csrr %0, %1" : "=r"(cluster_id) : "i"(MU_CSR_CLUSTER_ID));
    const auto core_id_in_cluster = vx_core_id();
    const auto cores_per_cluster = MU_NUM_CORES;
    const auto global_core_id = cluster_id * cores_per_cluster + core_id_in_cluster;

    const auto &context = schedule_context;
    const auto occupancy = context.occupancy;

    // thread_id mapping is core-round-robin, i.e. warp 4k+0 maps to core 0,
    // warp 4k+1 maps to core 1, etc.
    const auto warp_id_in_core = vx_warp_id();
    const auto warp_id_rr_in_cluster = (warp_id_in_core * cores_per_cluster) + core_id_in_cluster;
    const auto tid_in_warp = vx_thread_id();
    const auto tid_in_cluster = warp_id_rr_in_cluster * MU_NUM_THREADS + tid_in_warp;
    const auto threads_per_cluster =
        cores_per_cluster * occupancy * MU_NUM_THREADS;
    const auto tid_global = threads_per_cluster * cluster_id + tid_in_cluster;

    // 1-threadblock-to-1-cluster
    const auto threads_per_threadblock = threads_per_cluster;
    const auto tid_in_threadblock = tid_global % threads_per_threadblock;
    const auto threadblock_id = tid_global / threads_per_threadblock;

    const auto callback = context.callback;
    auto arg = context.arg;

    callback(arg, tid_in_threadblock, threads_per_threadblock, threadblock_id);

#ifdef MU_TAPEOUT_EPILOGUE
    // TAPEOUT EPILOGUE.  On the taped-out part a core that asserts `finished` fires the L0d/L0i
    // flush unit (MuonTile.scala:371-374), which wedges the L0d; and at occupancy >= 2 the core
    // never asserts finished at all, so nothing is ever written back and the whole output stays
    // stranded in cache even though the kernel ran to completion.  Measured on both U250 boards
    // 2026-09-21.  The epilogue drains L0d and L1 by capacity, parks warp 0 of each core in a
    // spin so `finished` never rises, and hands off to the host through the printBuf postbox.
    // NEVER RETURNS -- the host soft-resets the GPU once it sees the postbox.
    rad_tapeout_epilogue(tid_in_threadblock, threads_per_threadblock / MU_NUM_THREADS);
#endif
}

static void mu_schedule_workers() {
    vx_tmc(-1);
    mu_schedule_standalone();
    vx_tmc_zero();
}

static void mu_schedule_manager() {
    vx_tmc(-1);
    mu_schedule_standalone();
    vx_tmc(1);
}

/** Entry point that "bootstraps" kernel via a single thread elected in every
 *  core.
 *
 *  * Schedules the kernel with persistent thread blocks, i.e. 1 thread block
 *    maximally occupying all cores in each cluster.
 *  * `occupancy` determines the number of warps spawned in each core for the
 *    kernel.
 *  * The kernel grid size is fixed to NUM_CLUSTERS * NUM_CORES *
 *    `occupancy` * NUM_THREADS.
 *
 *  TODO relax this. */
void mu_schedule(mu_schedule_callback callback, void *arg, const uint32_t occupancy) {
    const auto core_id = vx_core_id();
    const auto thread_id = vx_thread_id();
    // update kernel launch context
    // Elect one thread PER CORE, not one per cluster.  L0d is per-core and is not coherent
    // across cores, and mu_fence() only retires the issuing warp's store queue -- it does not
    // push the line to a shared level.  Writing from core 0 alone therefore left every other
    // core reading whatever memory happened to hold at `schedule_context`, which is a NOBITS
    // .bss symbol that nothing initialises: zeros under metasim (core 1 jalr'd to 0x0) and
    // stale DRAM on the FPGA.  Whether the old code worked was a race between core 0's L0d
    // eviction and core 1's read.
    // Every core writes identical values -- callback, arg and occupancy are the same arguments
    // on every core -- so there is no cross-core race, and each core ends up with a correct
    // copy in its own L0d without any coherency requirement.
    (void)core_id;
    if (thread_id == 0) {
        schedule_context.callback = callback;
        schedule_context.arg = arg;
        schedule_context.occupancy = occupancy;
    }

    // fence & barrier to ensure ordering on context
    mu_fence();
    // mu_schedule is entered from every core's warp 0
    mu_barrier(0, MU_NUM_CORES);

    // schedule worker threads & manager thread
    vx_wspawn(occupancy, mu_schedule_workers);
    mu_schedule_manager();

    // TODO: add threadblock barrier
}

} // extern "C"
