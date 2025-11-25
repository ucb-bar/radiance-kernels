#include <stdint.h>

#define HW_TID(gtid) asm volatile ("csrr %0, mhartid" : "=r" (gtid))


void gemm() {
    uint32_t tid;
    HW_TID(tid);
    volatile float y = 0;
    volatile uint32_t z = 0;
    if (tid < 8) {
        y += (float) tid;
    } else {
        z += tid;
    }
}

__attribute__((section(".text.startup")))
int main() {
    asm volatile (".insn r %0, 1, 0, x0, %1, %2" :: "i"(0x0B), "r"(4), "r"(&gemm));
    gemm();
}
