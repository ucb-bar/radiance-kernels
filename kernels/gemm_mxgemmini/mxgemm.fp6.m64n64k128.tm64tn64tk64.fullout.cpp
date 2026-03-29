#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.fp6.m64n64k128.h"
// unify naming for A_in
static const uint8_t *A_in = &A_in_hw[0][0];
#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .TILE_M = 64,
    .TILE_N = 64,
    .TILE_K = 64,
    .FP4FP6 = true,
    .QUANT_OUTPUT = false,
};

void mxgemm_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock,
                  uint32_t threadblock_id) {
    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    mxgemm<C>(C.TILE_M, C.TILE_N, 128, C_gmem, tid_in_threadblock,
              threads_per_threadblock, threadblock_id);
}

int main() {
    mu_schedule(mxgemm_entry, nullptr, 2);
    return 0;
}
