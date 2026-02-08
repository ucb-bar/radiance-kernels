#include <inttypes.h>
#include <cstring>

#define HW_TID() ({uint32_t gtid; asm volatile ("csrr %0, mhartid" : "=r" (gtid)); gtid;})
volatile uint64_t *tocpu = (volatile uint64_t *) 0x10000U;

int main() {
    // wait until prev flush ends
    for (int i = 0; i < 1000; i++) {
        asm volatile("nop");
    }
    asm volatile("fence");
    // for (int i = 0; i < 1000; i++) {
    //     asm volatile("nop");
    // }
    const char message[] = "muon says hello!\n";
    for (int i = 0; i < strlen(message); i++) {
        if (HW_TID() == 0) {
            do {
                asm volatile("fence");
            } while (*tocpu != 0);
            *tocpu = (1ULL << 56) | (1ULL << 48) | message[i];
        asm volatile("nop");
        asm volatile("nop");
        asm volatile("nop");
        asm volatile("nop");
        asm volatile("nop");
        asm volatile("nop");
            asm volatile("fence");
        }
    }
    return 0;
}
