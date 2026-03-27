#include <mu_intrinsics.h>
#include <stdint.h>

extern "C" uint32_t __mu_num_warps = 1;

/*
 * Memory layout (device addresses):
 *   0x10000000 .. ~0x10000FFF  kernel code  (scratchpad page 0)
 *   0x10001000 .. 0x10002FFF   A data       (scratchpad 0x1000-0x2FFF)
 *   0x10005000 .. 0x10006FFF   B data       (scratchpad 0x1000-0x2FFF, +16K alias)
 *   0x10009000 .. 0x1000AFFF   C output     (scratchpad 0x1000-0x2FFF, +32K alias)
 *
 * Phase variable at fixed address 0x10000F00 (scratchpad 0x0F00, in code page).
 * Host writes the phase before each GPU run.
 *
 * L2 is 128 KiB and retains cached data across GPU resets.
 * Scratchpad is 16 KiB and aliases on 14 bits (mod 0x4000).
 */

#define A_BASE  ((volatile __global uint32_t *)0x10001000u)
#define B_BASE  ((volatile __global uint32_t *)0x10005000u)
#define C_BASE  ((__global uint32_t *)0x10009000u)
#define N_U32   2048u   /* 8 KiB / 4 bytes */

#define PHASE   (*(volatile __global uint32_t *)0x10000F00u)

#include "a_data"
#include "b_data"

int main() {
    uint32_t phase = PHASE;

    if (phase == 0) {
        /* Phase 0: read every word of A so it fills the L2 */
        for (uint32_t i = 0; i < N_U32; i++)
            (void)A_BASE[i];
    } else if (phase == 1) {
        /* Phase 1: read every word of B (aliased address) into L2 */
        for (uint32_t i = 0; i < N_U32; i++)
            (void)B_BASE[i];
    } else {
        /* Phase 2: compute C = A * B, both served from L2 */
        for (uint32_t i = 0; i < N_U32; i++) {
            uint32_t av = A_BASE[i];
            uint32_t bv = B_BASE[i];
            auto [a_hi, a_lo] = unpack_bf16x2(av);
            auto [b_hi, b_lo] = unpack_bf16x2(bv);
            C_BASE[i] = pack_bf16x2(a_lo * b_lo, a_hi * b_hi);
        }
    }

    return 0;
}
