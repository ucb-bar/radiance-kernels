#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

// #include "include/matmul_data_mx_fp8.h"
#include "include/matmul_fp8_64x64.h"
// #include "include/matmul_fp8_128x128.h"
// #include "include/matmul_fp8_128x128x256.h"
// #include "include/matmul_data_mx_lut_hw.h"

static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};

#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .GEMM_K = 64,
    .TILE_M = 64,
    .TILE_N = 64,
    .TILE_K = 64,
    .FP4FP6 = false,
};

void mxgemm_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock,
                  uint32_t threadblock_id) {
    if (tid_in_threadblock == 0) {
        mxgemm<C>();
    }

    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    mu_barrier(0, warps_per_threadblock);

    auto C_smem = reinterpret_cast<const __shared _Float16 *>(SPAD_DEST * DIM);
    auto C_gmem = reinterpret_cast<_Float16 *>(0x40000000);
    copy_output_smem_to_gmem_simt<64, 64>(C_smem, C_gmem, tid_in_threadblock,
                                          threads_per_threadblock);
}

int main() {
    mu_schedule(mxgemm_entry, nullptr, 2);
    return 0;
}
