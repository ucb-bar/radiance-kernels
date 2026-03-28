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
    // NOTE: this also configures the datatype for requantizer
    mxgemm_single_output_tile<C>(C.TILE_M, C.TILE_N, C.TILE_K,
                                 tid_in_threadblock);

    const auto tid_in_warp = tid_in_threadblock % MU_NUM_THREADS;
    const auto warp_id = tid_in_threadblock / MU_NUM_THREADS;
    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    mu_barrier(3, warps_per_threadblock);

    // MxRequantizer expects pre-quant data to be written in sequentially
    // ascending addresses, effectively disallowing warp- and
    // core-interleaving. However, within a single warp, stores are guaranteed
    // to leave LSU in program-order.  Therefore we force single-warp
    // execution.
    if (warp_id == 0) {
        constexpr auto N = C.TILE_M * C.TILE_N; // in bf16
        auto C_out_bf16_gmem = reinterpret_cast<const uint8_t *>(&C_out_bf16[0][0]);
        const auto smem_requant_base =
            reinterpret_cast<__shared uint8_t *>(GEMMINI_REQUANT);
        copy_gmem_to_smem_simt_bf16<C.TILE_M, C.TILE_N, sizeof(C_out_bf16[0][0])>(
            C_out_bf16_gmem, smem_requant_base, tid_in_warp /* trick as one core */,
            MU_NUM_THREADS /* trick as one core */);
    }

    mu_fence_smem();
    mu_barrier(4, warps_per_threadblock);

    // generate verifiable trace
    auto C_quant_smem = reinterpret_cast<const __shared uint8_t *>(0x0);
    auto C_quant_gmem = reinterpret_cast<uint8_t *>(0x40000000);
    copy_smem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                           C.OUT_ELEM_SIZE()>(
        C_quant_smem, C_quant_gmem, tid_in_threadblock, threads_per_threadblock);
}

int main() {
    mu_schedule(requant_entry, nullptr, 1);
    return 0;
}
