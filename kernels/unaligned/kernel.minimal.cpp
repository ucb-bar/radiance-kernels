#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})

inline void threadblock_barrier(unsigned int barrier_id, unsigned int count) __attribute__((convergent)) {
    vx_fence();
    vx_barrier(barrier_id, count);
}

#define ADDR0 0xff008004UL
#define ADDR1 0xff009004UL

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
    // size_t t = (size_t) (task_id * 4) % 32;
    asm volatile("nop");
    for (int i = 0; i < 8; i++) {
        if (i == 0) {
            if ((HW_TID() & 0x7) < 2) {
                asm volatile("lower_block:");
                volatile uint32_t a = *((volatile uint32_t *) (ADDR0));
                // *((volatile uint32_t *) (ADDR2)) = a;
            volatile uint32_t b = a + 1;
            } else {
                asm volatile("upper_block:");
                volatile uint32_t a = *((volatile uint32_t *) (ADDR1));
                // *((volatile uint32_t *) (ADDR3)) = a;
            volatile uint32_t b = a + 1;
            }
        }
                volatile uint32_t a = *((volatile uint32_t *) (ADDR1));
    }
    threadblock_barrier(2, 2);
}

int main() { // __attribute__((convergent)) {
    kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

    vx_spawn_tasks_cluster(64, (vx_spawn_tasks_cb)kernel_body, arg);
    return 0;
}
