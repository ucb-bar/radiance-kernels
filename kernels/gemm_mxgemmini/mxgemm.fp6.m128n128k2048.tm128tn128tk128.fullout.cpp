// FP6 128x128 output tile, K=2048 over K-tiles of TILE_K=128 (fullout, no requant).
// 128x128 fullout requires TILE_K<=128 (bf16 C accumulator + double-buffered A/B fit the SPAD).
#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.fp6.m128n128k2048.h"
// unify naming for A_in (fp6 header exposes A_in_hw and defines the LUTs)
static const uint8_t *A_in = &A_in_hw[0][0];
#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .TILE_M = 128,
    .TILE_N = 128,
    .TILE_K = 128,
    .DATATYPE = GemmDatatype::FP6,
    .QUANT_OUTPUT = false,
};

void mxgemm_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock, uint32_t threadblock_id) {
    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    mxgemm<C>(C.TILE_M, C.TILE_N, MATMUL_K, C_gmem, tid_in_threadblock,
              threads_per_threadblock, threadblock_id);
}

int main() {
    mu_schedule(mxgemm_entry, nullptr, 2);
    return 0;
}
