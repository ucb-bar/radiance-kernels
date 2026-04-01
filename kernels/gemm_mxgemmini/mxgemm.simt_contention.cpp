#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.fp8.m128n128k256.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"

constexpr uint32_t CORE_WARP_OCCUPANCY = 3;

constexpr GemmConfig C{
    .TILE_M = 128,
    .TILE_N = 128,
    .TILE_K = 128,
    .DATATYPE = GemmDatatype::FP8,
    .QUANT_OUTPUT = false,
};

void mxgemm_simt_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock, uint32_t threadblock_id) {
    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    auto dummy_gmem = reinterpret_cast<uint8_t *>(0x60000000);

    // specialize warps to:
    // 0:   Gemmini manager
    // 1-4: SMEM read/write worker
    //
    // to introduce synthetic contention on SMEM banks across Gemmini<->SIMT.
    
    const auto warp_id = tid_in_threadblock / MU_NUM_THREADS;
    constexpr uint32_t num_worker_warps = 4;
    static_assert(num_worker_warps <= (CORE_WARP_OCCUPANCY * MU_NUM_CORES - 1));

    if (warp_id == 0) {
        const auto threads_in_warpgroup = MU_NUM_THREADS * 1;
        const auto tid_in_warpgroup = tid_in_threadblock % MU_NUM_THREADS;
        mxgemm<C>(C.TILE_M, C.TILE_N, 256, C_gmem, tid_in_warpgroup,
                  threads_in_warpgroup, threadblock_id);
    } else if (1 <= warp_id && warp_id < 1 + num_worker_warps) {
        const auto threads_in_warpgroup = MU_NUM_THREADS * num_worker_warps;
        const auto warp_id_in_warpgroup = warp_id - 1;
        const auto tid_in_warp = tid_in_threadblock % MU_NUM_THREADS;

        // read dummy data from SMEM->GMEM to introduce read contention
        // warp N reads from bank N. maximize confusion across all banks for
        // MxGemmini
        constexpr auto SMEM_BANK_SIZE = MU_SMEM_SIZE_BYTES / 4;
        auto dummy_smem = reinterpret_cast<const __shared uint8_t *>(
            SMEM_BANK_SIZE * (warp_id_in_warpgroup % 4));
        for (int i = 0; i < 4; i++) {
            // force each warp read full C from its own bank
            copy_smem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                                   C.OUT_ELEM_SIZE()>(
                dummy_smem, dummy_gmem, tid_in_warp, MU_NUM_THREADS);
        }
    }
}

int main() {
    mu_schedule(mxgemm_simt_entry, nullptr, 3);
    return 0;
}
