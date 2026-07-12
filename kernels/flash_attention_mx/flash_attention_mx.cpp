// MXFP8 flash-attention kernel -- incremental bring-up.
//
// FA-K3 milestone: drive ONE Gemmini MX GEMM to compute S = Q @ K^T (contract
// over headdim d) using self-generated data (include/fa_data.h), move S out to
// GMEM as bf16, and self-check against the embedded golden QK_S_bf16.
//
// This de-risks the data layout + the pointer-parameterized gemm core before
// adding softmax (FA-K4) and the PV gemm (FA-K5).
#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/fa_data.h"

// LUTs are only used for FP6; declared (not defined-used) so the FP8 path's
// `if constexpr (USE_LUT())` branch parses. Match mxgemm.cpp.
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};

#include "mxgemm_core.hpp"
#include "flash_mx_impl.hpp"

// QK^T: S = Q@K^T, M=Sq N=Sk K=d. PV: O = P@V, M=Sq N=d K=Sk.
constexpr GemmConfig QK{
    .TILE_M = FA_SQ, .TILE_N = FA_SK, .TILE_K = FA_D,
    .DATATYPE = GemmDatatype::FP8, .QUANT_OUTPUT = false,
};
constexpr GemmConfig PV{
    .TILE_M = FA_SQ, .TILE_N = FA_D, .TILE_K = FA_SK,
    .DATATYPE = GemmDatatype::FP8, .QUANT_OUTPUT = false,
};
// Requantizer config: requantize the softmax P tile [Sq][Sk] (bf16) to MX FP8.
constexpr GemmConfig RQ{
    .TILE_M = FA_SQ, .TILE_N = FA_SK, .TILE_K = FA_SK,
    .DATATYPE = GemmDatatype::FP8, .QUANT_OUTPUT = true,
};

// GMEM scratch addresses (device-side).
static constexpr uint32_t S_GMEM  = 0x40000000;     // QK^T output S (bf16) [Sq][Sk]
static constexpr uint32_t P_GMEM  = 0x40010000;     // softmax P (fp8 e4m3) [Sq][Sk]
static constexpr uint32_t PS_GMEM = 0x40020000;     // P scales (E8M0) [Sk/32][Sq]
static constexpr uint32_t L_GMEM  = 0x40030000;     // row denom l (fp32) [Sq]
static constexpr uint32_t O_GMEM  = 0x40040000;     // final attention output O (bf16) [Sq][d]
// softmax_scale = 1/sqrt(d), emitted as bf16 by the data generator (depends on d).
static constexpr uint16_t SOFTMAX_SCALE_BF16 = FA_SOFTMAX_SCALE_BF16;

// S (bf16 QK^T result) stays in SMEM at SPAD_DEST*DIM; softmax reads it there.
static constexpr uint32_t S_SMEM = SPAD_DEST * DIM;  // 0x1000
// l (row denom, bf16) kept in SMEM so normalize avoids a GMEM load (which stalls).
static constexpr uint32_t L_SMEM = 0xE000;
// P (softmax probabilities, bf16 packed) scratch in SMEM, then fed to the requantizer.
static constexpr uint32_t P_SMEM = 0x10000;
// Per-scale SMEM scratch (one 32-bit word per E8M0 code) -- avoids overlapping
// vectorized byte stores; packed to contiguous GMEM bytes by pack_scales_to_gmem.
static constexpr uint32_t SCALE_SMEM = 0xD000;

void fa_entry(void *arg, uint32_t tid_in_threadblock,
              uint32_t threads_per_threadblock, uint32_t threadblock_id) {
    const auto wpb = threads_per_threadblock / MU_NUM_THREADS;

    // ---- QK^T: S = Q @ K^T (Gemmini MX FP8 mesh) -> bf16 S in SMEM @ SPAD_DEST ----
    // single_output_tile leaves the bf16 result tile in SMEM (no GMEM move-out).
    mxgemm_single_output_tile<QK>(&QK_A_in[0][0], &QK_B_in[0][0],
                                  &QK_A_scales_row[0][0], &QK_B_scales_col[0][0],
                                  FA_SQ, FA_SK, FA_D,
                                  tid_in_threadblock, threads_per_threadblock);
    mu_barrier(3, wpb);  // all warps wait for thread-0's gemm (+ gemmini_fence)

    // ---- PREFETCH V for the PV gemm: thread-0 issues the V GMEM->SMEM move-in + config
    //      + V-scale load ASYNC (no gemmini_fence), then falls through to softmax. The V
    //      DMA overlaps the SIMT softmax+requant below (which produce P), hiding the PV
    //      move-in latency that was ~40 % of the run. B=V, A(=P) skipped (SKIP_A). ----
    mxgemm_prefetch_tile<PV, /*SKIP_A=*/true>(
        &V_in[0][0] /*A unused*/, &V_in[0][0],
        &V_scales[0][0] /*A scales unused*/, &V_scales[0][0],
        FA_SQ, FA_D, FA_SK, tid_in_threadblock);

    // ---- softmax(S * scale): activate in SIMT, write normalized P (bf16) to SMEM ----
    softmax_to_smem<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t *>(S_SMEM),
        reinterpret_cast<__shared uint32_t *>(P_SMEM),
        reinterpret_cast<float *>(L_GMEM),
        reinterpret_cast<__shared uint16_t *>(L_SMEM),
        SOFTMAX_SCALE_BF16, tid_in_threadblock, threads_per_threadblock);
    mu_fence_smem();
    mu_barrier(4, wpb);

    // ---- SIMT MX-FP8 requant of P, written ENTIRELY to SMEM (no GMEM round-trip / no
    //      unreliable global fence): e4m3 elements -> A scratchpad (spad 0) in the tiled
    //      layout the mesh expects; per-row E8M0 scales -> SMEM scratch (word per scale).
    //      P is 1/l-normalized so PV yields the final O. Uses the validated encoder.
    requant_P_to_spad_tiled<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint16_t *>(P_SMEM),
        reinterpret_cast<__shared uint32_t *>(0 /* A-spad base = SMEM byte 0 */),
        reinterpret_cast<__shared uint32_t *>(SCALE_SMEM),
        tid_in_threadblock, threads_per_threadblock);
    mu_fence_smem();
    mu_barrier(7, wpb);

    // ---- pack the SMEM scale scratch -> contiguous E8M0 bytes in the A scale SRAM
    //      (GEMMINI_SF_MEM_A), where the PV mesh reads A scales. Word stores (4/word),
    //      thread-0 -> no overlapping sub-word stores. fence.s makes it mesh-visible.
    pack_scales_to_sfmem<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t *>(SCALE_SMEM),
        reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_A), tid_in_threadblock);
    mu_fence_smem();
    mu_barrier(6, wpb);

    // ---- PV COMPUTE: O = P @ V. V was prefetched above (already in B-spad); A=P fp8 in
    //      spad 0 + A-scales in SF-SRAM (from requant/pack). Just drain the V DMA + matmul.
    mxgemm_compute_tile<PV>(tid_in_threadblock);
    mu_barrier(5, wpb);

    // ---- move final O (bf16) SPAD_DEST -> GMEM. SIMT copy (no /l: P was pre-normalized).
    //  KNOWN ISSUE (see FA_kernel.md): post-requant SIMT stores use a corrupted base
    //  register, AND the Gemmini-DMA move-out alternative asserts PutPartial on the SMEM
    //  xbar. O-output is the open blocker; requires the requantizer usage protocol. ----
    copy_smem_to_gmem_simt<FA_SQ, FA_D, sizeof(uint16_t)>(
        reinterpret_cast<const __shared uint8_t *>(S_SMEM),
        reinterpret_cast<uint8_t *>(O_GMEM),
        tid_in_threadblock, threads_per_threadblock);
}

int main() {
    mu_schedule(fa_entry, nullptr, 2);  // 2 warps (low occupancy: avoids L0d issue)
    return 0;
}
