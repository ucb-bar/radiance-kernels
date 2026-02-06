#include <inttypes.h>

#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})
volatile uint64_t *tocpu = (volatile uint64_t *) 0x10000U;

int main() {
    const char message[] = "muon hello\n";
    for (int i = 0; i < 11; i++) {
        asm volatile("nop");
       
        if (HW_TID() == 0) {
            while (*tocpu != 0) {
                asm volatile("fence");
            }
            *tocpu = (1ULL << 56) | (1ULL << 48) | message[i];
        }
    }
    return 0;
}
