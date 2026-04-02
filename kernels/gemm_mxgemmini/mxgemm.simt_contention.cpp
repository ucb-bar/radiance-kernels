#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.fp8.m64n64k512.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .TILE_M = 64,
    .TILE_N = 64,
    .TILE_K = 64,
    .DATATYPE = GemmDatatype::FP8,
    .QUANT_OUTPUT = false,
};

void simt_contention_entry(void *arg, uint32_t tid_in_threadblock,
                             uint32_t threads_per_threadblock,
                             uint32_t threadblock_id) {
    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    auto dummy_gmem = reinterpret_cast<uint8_t *>(0x60000000);

    // specialize warps to:
    // 0:   Gemmini manager
    // 1-4: SMEM read/write worker
    //
    // to introduce synthetic contention on SMEM banks across Gemmini<->SIMT.

    const auto warp_id = tid_in_threadblock / MU_NUM_THREADS;

    if (warp_id == 0) {
        const auto threads_in_warpgroup = MU_NUM_THREADS * 1;
        const auto tid_in_warpgroup = tid_in_threadblock % MU_NUM_THREADS;
        mxgemm<C>(C.TILE_M, C.TILE_N, 512, C_gmem, tid_in_warpgroup,
                  threads_in_warpgroup, threadblock_id);
    } else if (warp_id == 1) {
        const auto tid_in_warp = tid_in_threadblock % MU_NUM_THREADS;

        // read dummy data from SMEM->GMEM to introduce read contention
        // have warp 0 round-robin through bank 0~4
        constexpr auto SMEM_BANK_SIZE = MU_SMEM_SIZE_BYTES / 4;
#pragma unroll 32
        for (int i = 0; i < 1024 * 32 /*arbitrary*/; i++) {
            auto dummy_smem_base =
                reinterpret_cast<const volatile __shared uint8_t *>(
                    SMEM_BANK_SIZE * (i % 4));
            auto dummy_smem_addr =
                reinterpret_cast<const volatile __shared uint32_t *>(
                    dummy_smem_base) +
                tid_in_warp;
            auto dummy = *dummy_smem_addr;
        }
    }
}

int main() {
    mu_schedule(simt_contention_entry, nullptr, 1);
    return 0;
}
