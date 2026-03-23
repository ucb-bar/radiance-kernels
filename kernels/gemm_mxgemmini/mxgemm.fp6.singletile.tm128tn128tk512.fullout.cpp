#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/matmul_data_mx_lut_hw.h"
#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .GEMM_K = 512,
    .TILE_M = 128,
    .TILE_N = 128,
    .TILE_K = 512,
    .FP4FP6 = true,
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
