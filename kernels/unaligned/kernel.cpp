#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

#define NUM_THREADS_IN_CLUSTER 32
#define NUM_CLUSTERS 1

#define rd_cycles_force(x) asm volatile ("csrr %0, mcycle" : "=r" (x))
#define rd_cycles(x) rd_cycles_force(x)
#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})
#define PRINT_BUF ((char *) (0xff020000UL))
#define PRINTF(...) sprintf(PRINT_BUF, __VA_ARGS__)

inline void threadblock_barrier(unsigned int barrier_id, unsigned int count) __attribute__((convergent)) {
    vx_fence();
    vx_barrier(barrier_id, count);
}

#define ADDR0 0xff008004UL
#define ADDR1 0xff009004UL
#define ADDR2 0xff00a004UL
#define ADDR3 0xff00b004UL

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) __attribute__((convergent)) {
    size_t t = (size_t) (task_id * 4) % 32;
    if (t == 0) {
        for (int j = 0; j < 0x400; j += 0x100) {
            for (int i = 0; i < 8; i++) {
                *((volatile uint32_t *) (ADDR0 + j + i * 4)) = 0xbeef;
                *((volatile uint32_t *) (ADDR1 + j + i * 4)) = 0xbeef;
            }
        }
    }
    threadblock_barrier(0, 1);
    // for (int i = 0; i < 8; i++) {
        if (HW_TID() % 8 < 5) {
        // if (true) {
            asm volatile("lower_block:");
            volatile uint32_t a = *((volatile uint32_t *) (ADDR0 + 0x000 + t));
            volatile uint32_t b = *((volatile uint32_t *) (ADDR0 + 0x100 + t));
            volatile uint32_t c = *((volatile uint32_t *) (ADDR0 + 0x200 + t));
            volatile uint32_t d = *((volatile uint32_t *) (ADDR0 + 0x300 + t));

            volatile uint32_t u = *((volatile uint32_t *) (ADDR1 + 0x000 + t));
            volatile uint32_t v = *((volatile uint32_t *) (ADDR1 + 0x100 + t));
            volatile uint32_t w = *((volatile uint32_t *) (ADDR1 + 0x200 + t));
            volatile uint32_t x = *((volatile uint32_t *) (ADDR1 + 0x300 + t));

            *((volatile uint32_t *) (ADDR2 + 0x000 + t)) = a;
            *((volatile uint32_t *) (ADDR2 + 0x100 + t)) = b;
            *((volatile uint32_t *) (ADDR2 + 0x200 + t)) = c;
            *((volatile uint32_t *) (ADDR2 + 0x300 + t)) = d;

            *((volatile uint32_t *) (ADDR3 + 0x000 + t)) = u;
            *((volatile uint32_t *) (ADDR3 + 0x100 + t)) = v;
            *((volatile uint32_t *) (ADDR3 + 0x200 + t)) = w;
            *((volatile uint32_t *) (ADDR3 + 0x300 + t)) = x;
        } else {
            asm volatile("upper_block:");
            volatile uint32_t a = *((volatile uint32_t *) (ADDR1 + 0x000 + t));
            volatile uint32_t b = *((volatile uint32_t *) (ADDR1 + 0x100 + t));
            volatile uint32_t c = *((volatile uint32_t *) (ADDR1 + 0x200 + t));
            volatile uint32_t d = *((volatile uint32_t *) (ADDR1 + 0x300 + t));

            volatile uint32_t u = *((volatile uint32_t *) (ADDR0 + 0x000 + t));
            volatile uint32_t v = *((volatile uint32_t *) (ADDR0 + 0x100 + t));
            volatile uint32_t w = *((volatile uint32_t *) (ADDR0 + 0x200 + t));
            volatile uint32_t x = *((volatile uint32_t *) (ADDR0 + 0x300 + t));

            // for (int y = 4; y < 8; y++) {
            //     if (task_id == y) {
            //         PRINTF("Task ID: %d, a: %x, b: %x, c: %x, d: %x\n", task_id, a, b, c, d);
            //         PRINTF("Task ID: %d, u: %x, v: %x, w: %x, x: %x\n", task_id, u, v, w, x);
            //     }
            // }
            // threadblock_barrier(1, 1);

            *((volatile uint32_t *) (ADDR3 + 0x000 + t)) = a;
            *((volatile uint32_t *) (ADDR3 + 0x100 + t)) = b;
            *((volatile uint32_t *) (ADDR3 + 0x200 + t)) = c;
            *((volatile uint32_t *) (ADDR3 + 0x300 + t)) = d;

            *((volatile uint32_t *) (ADDR2 + 0x000 + t)) = u;
            *((volatile uint32_t *) (ADDR2 + 0x100 + t)) = v;
            *((volatile uint32_t *) (ADDR2 + 0x200 + t)) = w;
            *((volatile uint32_t *) (ADDR2 + 0x300 + t)) = x;
        }
    // }
    threadblock_barrier(2, 1);
    PRINTF(".");
    if (task_id == 0) {
        bool correct = true;
        PRINTF("\n");
        for (int j = 0; j < 0x400; j += 0x100) {
            for (int i = 0; i < 8; i++) {
                int v2 = *((volatile uint32_t *) (ADDR2 + i * 4 + j));
                if (v2 != 0xbeef) {
                    correct = false;
                    PRINTF("mismatch at %x, got %x\n", ADDR2 + i * 4 + j, v2);
                }
                int v3 = *((volatile uint32_t *) (ADDR3 + i * 4 + j));
                if (v3 != 0xbeef) {
                    correct = false;
                    PRINTF("mismatch at %x, got %x\n", ADDR3 + i * 4 + j, v3);
                }
            }
        }
        if (correct) {
            PRINTF("test passed\n");
        }
    }
}

int main() __attribute__((convergent)) {
    kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

    const uint32_t num_threads_in_cluster = NUM_THREADS_IN_CLUSTER;
    const uint32_t grid_size = num_threads_in_cluster * NUM_CLUSTERS;
    vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
    return 0;
}
