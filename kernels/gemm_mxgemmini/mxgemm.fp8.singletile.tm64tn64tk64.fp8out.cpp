#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.fp8.m64n64k64.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .TILE_M = 64,
    .TILE_N = 64,
    .TILE_K = 64,
    .FP4FP6 = false,
    .QUANT_OUTPUT = true,
};

void mxgemm_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock,
                  uint32_t threadblock_id) {
    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    mxgemm<C>(C.TILE_M, C.TILE_N, C.TILE_K, C_gmem, tid_in_threadblock,
              threads_per_threadblock, threadblock_id);

    const auto warp_id = threads_per_threadblock % MU_NUM_THREADS;
    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    mu_barrier(4, warps_per_threadblock);

    // generate verifiable trace for the scale factor output
    if (warp_id == 0) {
        const auto tid_in_warp = tid_in_threadblock % MU_NUM_THREADS;
        auto C_scale_gmem = reinterpret_cast<uint8_t *>(0x50000000);
        copy_gmem_to_gmem_simt<C.TILE_M, C.TILE_N / 32, sizeof(uint8_t)>(
            reinterpret_cast<const uint8_t *>(&C_scale_factors[0]),
            C_scale_gmem, tid_in_warp, MU_NUM_THREADS);
    }
}

int main() {
    mu_schedule(mxgemm_entry, nullptr, 2);
    return 0;
}
