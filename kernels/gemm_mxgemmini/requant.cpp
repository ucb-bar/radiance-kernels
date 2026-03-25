#include <stdint.h>
#include <radiance.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>
#include <mu_lib.h>

#include "include/matmul_fp8_64x64.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"

// model flashattention tile size
constexpr GemmConfig C{
    .TILE_M = 64,
    .TILE_N = 64,
    .TILE_K = 64,
    .FP4FP6 = false,
    .QUANT_OUTPUT = true,
};

void requant_entry(void *arg, uint32_t tid_in_threadblock,
                   uint32_t threads_per_threadblock, uint32_t threadblock_id) {
    // do a dummy single-tile gemm to clear out uninitialized queues/X's in
    // Gemmini
    mxgemm_single_output_tile<C>(C.TILE_M, C.TILE_N, C.TILE_K,
                                 tid_in_threadblock);

    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);

    const auto warp_id = tid_in_threadblock / MU_NUM_THREADS;
    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    mu_barrier(3, warps_per_threadblock);

    constexpr auto N = C.TILE_M * C.TILE_N; // in bf16
    auto C_out_bf16_gmem = reinterpret_cast<const uint8_t *>(&C_out_bf16[0][0]);
    const auto smem_requant_base =
        reinterpret_cast<__shared uint8_t *>(GEMMINI_REQUANT);
    // store_smem<N, /*zero_stride=*/false>(smem_requant_base,
    //                                      static_cast<uint16_t>(0x3f80));
    copy_gmem_to_smem_simt_bf16<C.TILE_M, C.TILE_N, sizeof(C_out_bf16[0][0])>(
        C_out_bf16_gmem, smem_requant_base, tid_in_threadblock,
        threads_per_threadblock);
}

int main() {
    mu_schedule(requant_entry, nullptr, 2);
    return 0;
}
