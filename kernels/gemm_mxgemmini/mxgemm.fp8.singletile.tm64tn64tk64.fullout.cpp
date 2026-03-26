#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/matmul_fp8_64x64.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .TILE_M = 64,
    .TILE_N = 64,
    .TILE_K = 64,
    .FP4FP6 = false,
    .QUANT_OUTPUT = false,
};

void mxgemm_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock,
                  uint32_t threadblock_id) {
    auto A_readout_gmem = reinterpret_cast<uint8_t *>(0x50000000);
    auto B_readout_gmem = reinterpret_cast<uint8_t *>(0x60000000);
    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    mxgemm<C>(C.TILE_M, C.TILE_N, C.TILE_K, C_gmem, tid_in_threadblock,
              threads_per_threadblock, threadblock_id);

    // Read out A/B from SMEM for verification
    //
    // const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    // mu_barrier(4, warps_per_threadblock);

    // auto a_spad_addr_tile0 = calculate_spad_addr<false>(0);
    // auto b_spad_addr_tile0 =
    //     calculate_spad_addr<true>(0) - C.PE_TILES_K() * C.PE_TILES_J() * DIM;
    // auto a_smem_tile0 =
    //     reinterpret_cast<const __shared uint8_t *>(a_spad_addr_tile0 * DIM);
    // auto b_smem_tile0 =
    //     reinterpret_cast<const __shared uint8_t *>(b_spad_addr_tile0 * DIM);
    // copy_smem_to_gmem_simt<C.TILE_M, C.TILE_K, sizeof(A_in[0][0])>(
    //     a_smem_tile0, A_readout_gmem, tid_in_threadblock,
    //     threads_per_threadblock);
    // copy_smem_to_gmem_simt<C.TILE_K, C.TILE_N, sizeof(B_in[0][0])>(
    //     b_smem_tile0, B_readout_gmem, tid_in_threadblock,
    //     threads_per_threadblock);
}

int main() {
    mu_schedule(mxgemm_entry, nullptr, 2);
    return 0;
}
