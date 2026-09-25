// kernel.cpp: GPU side of the tapeout template.  C[i] = A[i] + B[i] on uint32_t.
//
// Structure every tapeout kernel needs:
//   1. main() runs once per core (warp 0, lane 0).  rad_tapeout_begin() clears this core's
//      epilogue postbox slots and acknowledges the launch; then main() calls mu_schedule().
//   2. The entry function does the work, then calls rad_tapeout_epilogue() instead of returning.
//      The epilogue drains the caches, parks the cores, and tells the host through the postbox.
//      Never let a core assert `finished` on the taped-out part (see rad_tapeout.h).
//
// Rules this file follows, each of which has cost days on this part:
//   * Arguments come from memory the host wrote (TT_ARGS_ADDR), not from zero-initialized
//     statics: .bss is never loaded or zeroed on this platform.
//   * No mu_fence() in the kernel body.  Only one warp per core may fence; the epilogue does it.
//   * No barrier inside a divergent branch; if one is needed, put it outside and give the
//     branch an explicit else { asm volatile("nop"); }.

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <rad_tapeout.h>
#include <stdint.h>

#include "template.h"

static void vecadd_entry(void *raw_args, uint32_t tid_in_threadblock,
                         uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;   // one threadblock per cluster, and the tapeout part has one cluster
  const volatile TTArgs *args = reinterpret_cast<const volatile TTArgs *>(raw_args);
  const __global uint32_t *a = reinterpret_cast<const __global uint32_t *>(args->a);
  const __global uint32_t *b = reinterpret_cast<const __global uint32_t *>(args->b);
  __global uint32_t *c = reinterpret_cast<__global uint32_t *>(args->c);
  const uint32_t n = args->n;

  // Grid-stride loop: consecutive threads touch consecutive words, so each warp's 16 lanes
  // cover one 64 B line per iteration.
  for (uint32_t i = tid_in_threadblock; i < n; i += threads_per_threadblock) {
    c[i] = a[i] + b[i];
  }

  // Must be the last statement.  Never returns; the host soft-resets the GPU.
  rad_tapeout_epilogue(tid_in_threadblock, threads_per_threadblock / MU_NUM_THREADS);
}

int main() {
  rad_tapeout_begin();
  mu_schedule(vecadd_entry, reinterpret_cast<void *>(TT_ARGS_ADDR), TT_OCCUPANCY);
  return 0;   // not reached: the epilogue never returns
}
