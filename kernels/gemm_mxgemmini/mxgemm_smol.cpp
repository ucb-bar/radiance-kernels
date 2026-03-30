#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

// #include "mxgemm.data.fp6.m64n64k64.h"
// unify naming for A_in
static const uint8_t A_in_hw[64][64] = {0};
static const uint8_t *A_in = &A_in_hw[0][0];
static const uint8_t B_in[64][64] = {0};
static const uint32_t A_lut[32][3] = {0};
static const uint32_t B_lut[32][3] = {0};
static const uint32_t C_lut[32][3] = {0};
static const uint8_t A_scales_row[1][1] = {0};
static const uint8_t B_scales_col[1][1] = {0};
#include "mxgemm_lib.hpp"

// bogus
constexpr GemmConfig C{
    .TILE_M = 64,
    .TILE_N = 64,
    .TILE_K = 64,
    .FP4FP6 = true,
    .QUANT_OUTPUT = false,
};

void smol_entry(void *arg, uint32_t tid_in_threadblock,
                uint32_t threads_per_threadblock,
                uint32_t threadblock_id) {
    if (tid_in_threadblock == 0) {
        configure_mxgemmini<C>(C.TILE_M, C.TILE_N, C.TILE_K);
    }
}

int main() {
    mu_schedule(smol_entry, nullptr, 2);
    return 0;
}
