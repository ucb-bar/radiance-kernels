#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/matmul_fp8_128x128x256.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .GEMM_K = 256,
    .TILE_M = 128,
    .TILE_N = 128,
    .TILE_K = 256,
    .FP4FP6 = false,
};

void mxgemm_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock,
                  uint32_t threadblock_id) {
    if (tid_in_threadblock != 0) {
        return;
    }

    mxgemm<C>();
}

int main() {
    mu_schedule(mxgemm_entry, nullptr, 2);
    return 0;
}
