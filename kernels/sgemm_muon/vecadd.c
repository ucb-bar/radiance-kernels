#include <stdint.h>

#define HW_TID(gtid) asm volatile ("csrr %0, mhartid" : "=r" (gtid))
#define N 2048
#define THREADS 16
#define UNROLL 16

uint32_t *A = (uint32_t volatile *) 0xa0000000ULL;
uint32_t *B = (uint32_t volatile *) 0xa1000000ULL;
uint32_t *C = (uint32_t volatile *) 0xc0000000ULL;

void vecadd() {
    uint32_t tid;
    HW_TID(tid);
#pragma unroll(32)
    for (int i = 0; i < N; i += THREADS) {
        uint32_t a = A[i + tid];
        uint32_t b = B[i + tid];
        C[i + tid] = a + b;
    }
}

__attribute__((section(".text.startup")))
int main() {
    asm volatile (".insn r %0, 1, 0, x0, %1, %2" :: "i"(0x0B), "r"(4), "r"(&vecadd));
    vecadd();
}
