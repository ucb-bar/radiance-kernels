#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.fp8.m64n64k512.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
#include "mxgemm_lib.hpp"

constexpr uint32_t CORE_WARP_OCCUPANCY = 3;

constexpr GemmConfig C{
    .TILE_M = 64,
    .TILE_N = 64,
    .TILE_K = 64,
    .DATATYPE = GemmDatatype::FP8,
    .QUANT_OUTPUT = false,
};

void flash_contention_entry(void *arg, uint32_t tid_in_threadblock,
                            uint32_t threads_per_threadblock,
                            uint32_t threadblock_id) {
    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    const auto warp_id = tid_in_threadblock / MU_NUM_THREADS;
    const auto tid_in_warp = tid_in_threadblock % MU_NUM_THREADS;
    // constexpr uint32_t num_worker_warps = 4;
    // static_assert(num_worker_warps <= (CORE_WARP_OCCUPANCY * MU_NUM_CORES - 1));

    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    auto dummy_gmem = reinterpret_cast<uint8_t *>(0x60000000);

    const auto dim_m = C.TILE_M;
    const auto dim_n = C.TILE_N;
    const auto dim_k = 512;

    if (warp_id == 0) {
        mxgemm_single_output_tile<C, true>(
            dim_m, dim_n, dim_k, tid_in_threadblock, threads_per_threadblock);

        auto C_smem =
            reinterpret_cast<const __shared uint8_t *>(SPAD_DEST * DIM);
        copy_smem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                               C.OUT_ELEM_SIZE()>(
            C_smem, C_gmem, tid_in_warp,
            MU_NUM_THREADS /* single-warp moveout */);
    } else if (warp_id == 1) {
        // barrier after first tile move-in
        // FIXME: unify barrier-id with mxgemm
        constexpr auto barrier_id = 2;
        mu_barrier(barrier_id, warps_per_threadblock);

        asm volatile("simt_worker_loop_start_%=:" ::);

        uint32_t tile_k = 0;

#pragma unroll 1
        for (; (tile_k * C.TILE_K) < dim_k; tile_k++) {
            const uint32_t even_k = ((tile_k + 1) & 1);

            // read dummy data from SMEM->GMEM to introduce read contention
            constexpr auto SMEM_BANK_SIZE = MU_SMEM_SIZE_BYTES / 4;
            // A: bank 1->0->1->0->...
            // B: bank 2->3->2->3->...
            const auto A_smem_base =
                reinterpret_cast<const volatile __shared uint8_t *>(
                    SMEM_BANK_SIZE * even_k);
            const auto B_smem_base =
                reinterpret_cast<const volatile __shared uint8_t *>(
                    MU_SMEM_SIZE_BYTES - SMEM_BANK_SIZE * (even_k + 1));

#pragma unroll 32
            for (int i = 0; i < 1024 /*arbitrary*/; i++) {
                auto A_smem_addr =
                    reinterpret_cast<const volatile __shared uint32_t *>(
                        A_smem_base) +
                    tid_in_warp;
                auto B_smem_addr =
                    reinterpret_cast<const volatile __shared uint32_t *>(
                        B_smem_base) +
                    tid_in_warp;
                auto A_dummy = *A_smem_addr;
                auto B_dummy = *B_smem_addr;
            }

            mu_barrier(barrier_id, warps_per_threadblock);
        }

        asm volatile("simt_worker_loop_end_%=:" ::);
    }
}

// void simple_contention_entry(void *arg, uint32_t tid_in_threadblock,
//                              uint32_t threads_per_threadblock,
//                              uint32_t threadblock_id) {
//     auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);
//     auto dummy_gmem = reinterpret_cast<uint8_t *>(0x60000000);

//     // specialize warps to:
//     // 0:   Gemmini manager
//     // 1-4: SMEM read/write worker
//     //
//     // to introduce synthetic contention on SMEM banks across Gemmini<->SIMT.

//     const auto warp_id = tid_in_threadblock / MU_NUM_THREADS;
//     constexpr uint32_t num_worker_warps = 4;
//     static_assert(num_worker_warps <= (CORE_WARP_OCCUPANCY * MU_NUM_CORES - 1));

//     if (warp_id == 0) {
//         const auto threads_in_warpgroup = MU_NUM_THREADS * 1;
//         const auto tid_in_warpgroup = tid_in_threadblock % MU_NUM_THREADS;
//         mxgemm<C>(C.TILE_M, C.TILE_N, 256, C_gmem, tid_in_warpgroup,
//                   threads_in_warpgroup, threadblock_id);
//     } else if (warp_id == 1) {
//         const auto tid_in_warp = tid_in_threadblock % MU_NUM_THREADS;

//         // read dummy data from SMEM->GMEM to introduce read contention
//         // have warp 0 round-robin through bank 0~4
//         constexpr auto SMEM_BANK_SIZE = MU_SMEM_SIZE_BYTES / 4;
// #pragma unroll 32
//         for (int i = 0; i < 1024 * 32 /*arbitrary*/; i++) {
//             auto dummy_smem_base =
//                 reinterpret_cast<const volatile __shared uint8_t *>(
//                     SMEM_BANK_SIZE * (i % 4));
//             auto dummy_smem_addr =
//                 reinterpret_cast<const volatile __shared uint32_t *>(
//                     dummy_smem_base) +
//                 tid_in_warp;
//             auto dummy = *dummy_smem_addr;
//         }
//     }
// }

int main() {
    mu_schedule(flash_contention_entry, nullptr, 1);
    return 0;
}
