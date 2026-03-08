/*
 * mu_schedule.cpp: Static scheduler/bootstrapping routine for Muon kernels.
 *
 * Spawns the correct number of warps and threads given the grid and
 * threadblock dimension, using a static mapping from local thread ID to
 * hardware thread ID.
 */

#include <mu_schedule.h>
#include <mu_intrinsics.h>

#define NUM_CORES_MAX 1024

extern "C" {

struct Context {
    mu_schedule_callback callback;
    void *arg;
    uint32_t occupancy;
};

/* Since vx_wspawn can only enter a function with no arguments, we need a
 * persistent state to restore contexts such as kernel arguments. */
static Context schedule_context;

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
    // elect a single thread per cluster to prevent racy writes
    if (core_id == 0 && thread_id == 0) {
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
