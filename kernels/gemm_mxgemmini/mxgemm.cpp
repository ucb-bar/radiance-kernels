#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm_lib.hpp"

void mxgemm(void *arg, uint32_t tid_in_threadblock,
            uint32_t threads_per_threadblock,
            uint32_t threadblock_id) {
    if (tid_in_threadblock != 0) {
        return;
    }

    // TODO: dim_m / dim_n
    const uint32_t dim_k = GEMM_K;

    configure_mxgemmini();

    // -----------------
    // Initiate pipeline
    // -----------------
    //
    int tile_k = 0;
    // TODO: change 0's for multiple SMEM tiles
    copy_gmem_to_smem_async(0, 0, tile_k);

    // Load scaling factors from GMEM to the scale SRAM
    // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));
    load_scale_factors(calculate_scale_factor_addr<false>(tile_k), &A_scales_row[0][0], SCALE_FAC_DIM);
    load_scale_factors(calculate_scale_factor_addr<true> (tile_k), &B_scales_col[0][0], SCALE_FAC_DIM);

    asm volatile ("load_lut_start_%=:" :: );

#if USE_LUT
    if constexpr (USE_LUT) {
        // A_lut[M>>G][LUT_SIZE] and B_lut[N>>G][LUT_SIZE]: one unique LUT per slot
        for (size_t i = 0; i < (TILE_N >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            load_lut_row(reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT0_ADDR) + 3 * i,
                     (uint8_t *)B_lut[i]);
        }
        for (size_t i = 0; i < (TILE_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            load_lut_row(reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT1_ADDR) + 3 * i,
                     (uint8_t *)A_lut[i]);
        }
        for (size_t i = 0; i < (TILE_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            load_lut_row(reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT2_ADDR) + 3 * i,
                     (uint8_t *)A_lut[i]);
        }
    }
#endif

    asm volatile ("load_lut_end_%=:" :: );

    // fence scale factor and LUT writes
    mu_fence_smem();

    // wait for GMEM->SMEM copy
    gemmini_fence();

    // ------------------------------
    // Main software-pipelined K-loop
    // ------------------------------
    //
    asm volatile ("main_matmul_k_loop_start_%=:" :: );

    //                 ┌───┐   ┌───┐
    //             ┌───────────┐
    //         ┌───────┐
    // Loop 1: M0->M1->C0->M0->C1->M1->C0
    //         ┌───┐   ┌───┐   ┌───┐
    // Loop 2: M0->C0->M1->C1->M0->C0->...
    //
    // FIXME: below is not possible in current fence_ready():
    // Operations of the same type are serialized (e.g. M0->M1->M2); therefore
    // no need to fence against previous same-double-buffer op.
    for (; (tile_k * TILE_K) < dim_k; tile_k++) {
        const auto last_k = ((tile_k + 1) * TILE_K) >= dim_k;
        const auto odd_k = (tile_k & 1);

        // GMEM->SMEM DMA for the next tile_k
        // TODO: This results in an unnecessary move-in at the last K tile
        if constexpr (!DISABLE_DMA_MOVE_IN) {
            // gemmini_fence_ready();
            copy_gmem_to_smem_async(0, 0, tile_k + 1);
        }

        // asynchrously kick off matmul for this tile_k
        // gemmini_fence_ready();
        matmul_tile_async(tile_k, last_k);

        // update scale factors for the next tile_k
        // make sure to place this between tile_async and fence to hide latency
        if constexpr (!DISABLE_SCALE_FACTOR_UPDATE) {
            load_scale_factors(calculate_scale_factor_addr<false>(tile_k + 1),
                               &A_scales_row[0][0], SCALE_FAC_DIM);
            load_scale_factors(calculate_scale_factor_addr<true>(tile_k + 1),
                               &B_scales_col[0][0], SCALE_FAC_DIM);

            // fence scale factor and LUT writes before next Gemmini compute
            mu_fence_smem();

            // configure scalefac->PE double-buffer read; inst: 0x3420b07b
            gemmini_mxquant_config_mvout(
                rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])), // TODO: change for multiple SMEM tiles
                PE_TILES_I, PE_TILES_J, PE_TILES_K,
                odd_k, // A double-buffer toggle
                odd_k, // B double-buffer toggle
                QUANT_LUT_UPDATE_GRANULARITY);
        }

        // TODO: add LUT loading
        gemmini_fence();
    }

    gemmini_fence();

    asm volatile ("main_matmul_k_loop_end_%=:" :: );

#if 0
    // read back C_out_got from DMEM to generate some traces
    volatile uint64_t sum = 0;
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            uint64_t got = C_out_got[i][j];
            sum += got;
        }
    }
#endif
}

int main() {
    mu_schedule(mxgemm, nullptr, 2);
    return 0;
}
