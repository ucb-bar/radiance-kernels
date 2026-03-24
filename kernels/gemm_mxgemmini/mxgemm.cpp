#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

// #include "include/matmul_data_mx_fp8.h"
// #include "include/matmul_fp8_64x64.h"
// #include "include/matmul_fp8_128x128.h"
// #include "include/matmul_fp8_128x128x256.h"
// static const uint8_t A_lut[64][16] = {0};
// static const uint8_t B_lut[64][16] = {0};
// static const uint8_t C_lut[64][16] = {0};
#include "include/matmul_data_mx_lut_hw.h"

#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .TILE_M = 128,
    .TILE_N = 128,
    .TILE_K = 128,
    .FP4FP6 = true,
    .QUANT_OUTPUT = true,
};

void mxgemm_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock, uint32_t threadblock_id) {
    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    mxgemm<C>(C.TILE_M, C.TILE_N, C.TILE_K, C_gmem, tid_in_threadblock,
              threads_per_threadblock, threadblock_id);
}

int main() {
    mu_schedule(mxgemm_entry, nullptr, 2);
    return 0;
}
