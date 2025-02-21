#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "include/gemmini.h"
#include "gemmini_mmio.h"

#define NUM_CLUSTERS 1
#define NUM_THREADS_IN_CLUSTER 512

#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // constexpr uint32_t timer = 50000;
  // uint32_t counter = 0;
  // while ((counter++) < timer) {
  //   asm("");
  // }
  //
  // to prevent optimize-out
  // reinterpret_cast<uint32_t *>(arg->addr_c)[0] = counter;

  // call barrier in a divergent branch, which will hang the core
  asm volatile("li x1, 0xa0a0a0a0");
  asm volatile("li x2, 0xa0a0a0a0");
  asm volatile("li x3, 0xa0a0a0a0");
  asm volatile("li x4, 0xa0a0a0a0");
  asm volatile("li x5, 0xa0a0a0a0");
  asm volatile("li x6, 0xa0a0a0a0");
  asm volatile("li x7, 0xa0a0a0a0");
  asm volatile("li x8, 0xa0a0a0a0");
  asm volatile("li x9, 0xa0a0a0a0");
  asm volatile("li x10, 0xa0a0a0a0");
  asm volatile("li x11, 0xa0a0a0a0");
  asm volatile("li x12, 0xa0a0a0a0");
  asm volatile("li x13, 0xa0a0a0a0");
  asm volatile("li x14, 0xa0a0a0a0");
  asm volatile("li x15, 0xa0a0a0a0");
  asm volatile("li x16, 0xa0a0a0a0");
  asm volatile("li x17, 0xa0a0a0a0");
  asm volatile("li x18, 0xa0a0a0a0");
  asm volatile("li x19, 0xa0a0a0a0");
  asm volatile("li x20, 0xa0a0a0a0");
  asm volatile("li x21, 0xa0a0a0a0");
  asm volatile("li x22, 0xa0a0a0a0");
  asm volatile("li x23, 0xa0a0a0a0");
  asm volatile("li x24, 0xa0a0a0a0");
  asm volatile("li x25, 0xa0a0a0a0");
  asm volatile("li x26, 0xa0a0a0a0");
  asm volatile("li x27, 0xa0a0a0a0");
  asm volatile("li x28, 0xa0a0a0a0");
  asm volatile("li x29, 0xa0a0a0a0");
  asm volatile("li x30, 0xa0a0a0a0");
  asm volatile("li x31, 0xa0a0a0a0");
  MARK_BEG();
  asm volatile("csrr a0, 0xcc1");
  asm volatile("beqz a0, bar");
  asm volatile("vx_tmc zero");
  asm volatile("bar:");
  asm volatile("vx_bar zero, a0");
  // if ((vx_thread_id() % NUM_THREADS) == 0) {
  //   vx_barrier(0, NUM_WARPS);
  // }

  vx_tmc(0);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  // spawn a single warp in every core
  const uint32_t grid_size = NUM_THREADS * NUM_CORES;
#ifdef RADIANCE
  vx_spawn_tasks_cluster(NUM_THREADS_IN_CLUSTER, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
