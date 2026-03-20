#include <stdint.h>
#include <radiance.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
#include "mxgemmini_mmio.h"
#include "lib.h"

// model flashattention tile size
constexpr auto TILE_M = 128;
constexpr auto TILE_N = 128;
constexpr auto TILE_K = 128;
constexpr auto PE_M = 16;
constexpr auto PE_N = 16;
constexpr auto PE_K = 16;
constexpr auto PE_TILES_I = TILE_M / PE_M;
constexpr auto PE_TILES_J = TILE_N / PE_N;
constexpr auto PE_TILES_K = TILE_K / PE_K;
constexpr auto QUANT_LUT_UPDATE_GRANULARITY = 1;

static uint32_t C_scale_factors[TILE_M * TILE_N / 32] __attribute__((aligned(32))) = {0};

void store_requant(void *arg, uint32_t tid_in_threadblock,
                   uint32_t threads_per_threadblock,
                   uint32_t threadblock_id) {
    if (tid_in_threadblock == 0) {
        // configure scale factor move-out address for requantizer
        gemmini_mxquant_config_mvout(
            rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
            PE_TILES_I, PE_TILES_J, PE_TILES_K,
            0, // A double-buffer toggle
            0, // B double-buffer toggle
            QUANT_LUT_UPDATE_GRANULARITY);

        // wait for configuration finish
        gemmini_fence();
    }

    const auto warp_id = tid_in_threadblock / MU_NUM_THREADS;
    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    mu_barrier(0, warps_per_threadblock);

    if (vx_core_id() != 0) {
        return;
    }

    if (warp_id != 0) {
        return;
    }

    constexpr auto N = TILE_M * TILE_N; // in bf16
    // constexpr auto N = GEMMINI_REQUANT_SIZE / sizeof(uint16_t);
    const auto base = reinterpret_cast<volatile __shared uint16_t *>(GEMMINI_REQUANT);
    store_smem<N, /*zero_stride=*/false>(base, static_cast<uint16_t>(0x3f80));
}

int main() {
    mu_schedule(store_requant, nullptr, MU_NUM_WARPS);
    return 0;
}
