#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.fp6.m128n128k256.h"
// unify naming for A_in
static const uint8_t *A_in = &A_in_hw[0][0];
#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .TILE_M = 128,
    .TILE_N = 128,
    .TILE_K = 128,
    .DATATYPE = GemmDatatype::FP6,
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
    
    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    const auto warp_id = tid_in_threadblock / MU_NUM_THREADS;
    if (warp_id == 0) {
        const auto tid_in_warpgroup = tid_in_threadblock % MU_NUM_THREADS;
        const auto threads_in_warpgroup = MU_NUM_THREADS * 1;
        mxgemm<C>(C.TILE_M, C.TILE_N, 256, C_gmem, tid_in_warpgroup,
                  threads_in_warpgroup, threadblock_id);
    } else if (1 <= warp_id && warp_id < 3) {
        const auto tid_in_warpgroup = tid_in_threadblock - MU_NUM_THREADS;
        const auto threads_in_warpgroup = MU_NUM_THREADS * 3;

        // read dummy data from SMEM->GMEM to introduce read contention
        // rotate all banks 0~3
        auto dummy_smem = reinterpret_cast<const __shared uint8_t *>(0x0);
        for (int i = 0; i < 8; i++) {
            auto dummy_smem_rr = dummy_smem + (i % 4) * (MU_SMEM_SIZE_BYTES / 4);
            copy_smem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                                   C.OUT_ELEM_SIZE()>(
                dummy_smem, dummy_gmem, tid_in_warpgroup, threads_in_warpgroup);
        }
    }
}

int main() {
    mu_schedule(mxgemm_simt_entry, nullptr, 3);
    return 0;
}
