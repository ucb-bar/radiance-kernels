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
};

/* Since vx_wspawn can only enter a function with no arguments, we need a
 * persistent state to restore contexts such as kernel arguments.
 * This is set to per-core to prevent racy writes. */
static Context schedule_context[NUM_CORES_MAX];

static void __attribute__ ((noinline)) mu_schedule_standalone() {
    uint32_t global_core_id = vx_core_id(); // TODO
    uint32_t global_tid = 0;
    asm volatile("csrr %0, mhartid" : "=r"(global_tid));

    const uint32_t threads_per_threadblock =
        vx_num_cores() * vx_num_warps() * vx_num_threads();
    const uint32_t tid_in_threadblock = global_tid % threads_per_threadblock;
    const uint32_t threadblock_id = global_tid / threads_per_threadblock;

    const auto &context = schedule_context[global_core_id];
    const auto callback = context.callback;
    auto arg = context.arg;

    callback(arg, tid_in_threadblock, threads_per_threadblock, threadblock_id);
}

static void __attribute__ ((noinline)) mu_schedule_workers() {
    vx_tmc(-1);
    mu_schedule_standalone();
    vx_tmc_zero();
}

/** Entry point that "bootstraps" kernel via a single thread elected in every
 *  core.
 *
 *  Schedules the kernel with persistent thread blocks, i.e. 1 thread block
 *  maximally occupying all cores and warps in each cluster. Therefore, the
 *  kernel grid size is fixed to the total thread slots in HW, i.e.
 *  NUM_CLUSTERS * NUM_CORES * NUM_WARPS * NUM_THREADS.
 *  TODO relax this. */
void mu_schedule(mu_schedule_callback callback, void *arg) {
    const auto num_warps = vx_num_warps();

    const auto core_id = vx_core_id();
    schedule_context[core_id].callback = callback;
    schedule_context[core_id].arg = arg;

    // TODO: fence here?

    // schedule worker threads & retire
    vx_wspawn(num_warps, mu_schedule_workers);

    // schedule main thread & return single-threaded
    {
        vx_tmc(-1);
        mu_schedule_standalone();
        vx_tmc(1);
    }

    // TODO: add threadblock barrier
}

} // extern "C"
