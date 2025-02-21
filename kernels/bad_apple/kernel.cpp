#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

#define rd_cycles(x) asm volatile ("csrr %0, mcycle" : "=r" (x))
#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})
#define PRINTF(...) sprintf((char *) (0xff010000UL), __VA_ARGS__)

inline void threadblock_barrier(unsigned int barrier_id, unsigned int count) {
  vx_fence();
  vx_barrier(barrier_id, count);
}

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  vx_tmc(0xff);
  const volatile uint32_t *const A = (const volatile uint32_t *const) arg->addr_a;

  // vx_tmc(1);
  // for (int i = 0; i < 75; i++) {
  //   if (task_id == i) {
  //     PRINTF("%d", task_id);
  //   }
  // }
  // threadblock_barrier(2, 5);

  /*
  #define mark(x) \
    vx_tmc(0x80); \
    if (task_id == 79) *(((volatile uint32_t *) x)) = x; \
    vx_fence(); \
    vx_barrier(x & 1, 5); \
    vx_tmc(0xff)

  #define write_fb0(value) *(((volatile uint32_t *) 0xff010000UL) + task_id) = (value)
  #define write_fb1(value) *(((volatile uint32_t *) 0xff010200UL) + task_id) = (value)

  while (true) {
    uint32_t v0, v1, v2, v3;
    v0 = A[task_id];
    v1 = A[1 * 75 + task_id];
    v2 = A[2 * 75 + task_id];
    v3 = A[3 * 75 + task_id];
    for (int i = 30; i < 6569; i += 4) {
      write_fb1(v0);
      v0 = A[(i + 0) * 75 + task_id];
      mark(0xff010130);

      write_fb0(v1);
      v1 = A[(i + 1) * 75 + task_id];
      mark(0xff010330);

      write_fb1(v2);
      v2 = A[(i + 2) * 75 + task_id];
      mark(0xff010130);

      write_fb0(v3);
      v3 = A[(i + 3) * 75 + task_id];
      mark(0xff010330);
    }
  }
  */

  #define WORKERS 128

  // #define WORDS 1350
  // #define LINES 338
  // #define ITERS 11 // = 1350 / 128

  #define WORDS 5400
  #define LINES 1350
  #define T_ITERS 43

  #define mark_fb0() \
    vx_tmc(0x80); if (task_id == 127) *(((volatile uint32_t *) 0xff011000UL)) = LINES; \
    vx_fence(); vx_barrier(0, 8); vx_tmc(0xff)
  #define mark_fb1() \
    vx_tmc(0x80); if (task_id == 127) *(((volatile uint32_t *) 0xff011004UL)) = LINES; \
    vx_fence(); vx_barrier(1, 8); vx_tmc(0xff)
  #define write_fb0(addr, value) *(((volatile uint32_t *) 0xff018000UL) + addr) = (value)
  #define write_fb1(addr, value) *(((volatile uint32_t *) 0xff020000UL) + addr) = (value)

  #define CYCLES_TO_WAIT 240000

  uint64_t cycles0, cycles1;
  cycles0 = 0;

  while (true) {
    volatile uint32_t v0, v1;
    for (int i = 20; i < 6569; i += 1) {
      v0 = A[i * WORDS + task_id];
      v1 = A[i * WORDS + WORKERS + task_id];
      int offset0 = 0 * WORKERS + task_id;
      int offset1 = 1 * WORKERS + task_id;

      for (int j = 1; j < T_ITERS; j += 2) {
        write_fb0(offset0, v0);
        offset0 += 2 * WORKERS;
        v0 = A[(i + 0) * WORDS + offset0];
        write_fb0(offset1, v1);
        offset1 += 2 * WORKERS;
        v1 = A[(i + 0) * WORDS + offset1];
      }
      write_fb0(offset0, v0);
      write_fb0(offset1, v1);

      /*offset0 += 2 * WORKERS;
      v0 = A[(i + 0) * WORDS + offset0];
      write_fb0(offset0, v0);*/

      if (task_id == 0) {
        rd_cycles(cycles1);
        while (cycles1 - cycles0 < CYCLES_TO_WAIT) {
          rd_cycles(cycles1);
        }
        cycles0 = cycles1;
      }

      threadblock_barrier(0, 8);
      mark_fb0();
    }
  }
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

#ifdef RADIANCE
  vx_spawn_tasks_cluster(128, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(8, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}