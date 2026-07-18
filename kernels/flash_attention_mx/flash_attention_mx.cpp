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
// Streaming (flash) per-key-block gemms: QK_j = Q@K_j^T is [Sq][Bk] (N=Bk); PV_j = P_j@V_j
// contracts over Bk (K=Bk). Non-square tiles (Bk != Sq/d) -> square assert relaxed.
constexpr GemmConfig QK{
    .TILE_M = FA_SQ, .TILE_N = FA_BK, .TILE_K = FA_D,
    .DATATYPE = GemmDatatype::FP8, .QUANT_OUTPUT = false,
};
constexpr GemmConfig PV{
    .TILE_M = FA_SQ, .TILE_N = FA_D, .TILE_K = FA_BK,
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
// SMEM layout sized for FA tiles up to 128x128 (bytes). A-spad (Q/P-fp8) 0..0x4000 (16KB);
// C=SPAD_DEST (S/PV bf16) 0x4000..0xC000 (32KB); O_acc 0xC000..0x14000 (32KB); scratch
// 0x14000+; B-spad (K/V) at the top. All non-overlapping for Sq=Bk=d in {64,128}.
static constexpr uint32_t P_SMEM = 0x10000;     // (unused: fused_softmax_requant writes P-fp8 direct)
static constexpr uint32_t OACC_SMEM = 0xC000;   // running unnormalized O accumulator [Sq][d] bf16
static constexpr uint32_t SCALE_SMEM = 0x14000; // per-scale word scratch (packed -> SF-SRAM)
static constexpr uint32_t M_SMEM    = 0x14800;  // running row max
static constexpr uint32_t LS_SMEM   = 0x14A00;  // running row denom l
static constexpr uint32_t CORR_SMEM = 0x14C00;  // per-row rescale corr
static constexpr uint32_t REDBUF_SMEM = 0x15000; // per-warp tree-reduce scratch (was 0xC000)

// Lightweight phase profiler: thread-0 stores the mcycle counter to a GMEM marker array
// at each phase boundary. Parse stores to MARK_GMEM from the .out trace -> per-phase cycles.
static constexpr uint32_t MARK_GMEM = 0x40050000;
#define MARK() do { if (tid == 0) { uint32_t _c; asm volatile("csrr %0, mcycle" : "=r"(_c)); \
                                    ((volatile uint32_t *)MARK_GMEM)[mki++] = _c; } } while (0)

void fa_entry(void *arg, uint32_t tid_in_threadblock,
              uint32_t threads_per_threadblock, uint32_t threadblock_id) {
    const auto wpb = threads_per_threadblock / MU_NUM_THREADS;
    const auto tid = tid_in_threadblock;
    const auto thr = threads_per_threadblock;
    uint32_t mki = 0;
    // One-time gemmini setup (hoisted out of the per-block gemms). For square streaming
    // blocks (Bk=Sq=d) QK and PV share an identical config, so configure ONCE here and
    // pass DO_CONFIG=false below -> skips ~half the per-gemm ROCC command overhead.
    if (tid == 0) { gemmini_flush(0); configure_mxgemmini<QK>(FA_SQ, FA_BK, FA_D); }
    MARK();  // 0: entry

    // ===== Streaming (flash) attention: loop over FA_NBLK key blocks of Bk. Running
    // online-softmax state (m, l, O_acc) lives in SMEM across blocks; 1/l is deferred to
    // the finalize. Mirrors FA.mx_attention_flash -> validated against golden_O_flash. =====
    for (uint32_t j = 0; j < FA_NBLK; j++) {
        const uint32_t first = (j == 0);

        // QK_j: S_j = Q @ K_j^T  -> bf16 S_j @ SPAD_DEST. K_j^T = QK_B_blocks[j] [d][Bk].
        // FA QK is a SINGLE K-tile (dim_k=FA_D=TILE_K), so use the slim prefetch+compute
        // path (like PV) instead of mxgemm_single_output_tile's dead software-pipelined
        // K-loop -- drops the heavy gemm fn (33 arch regs) from the call graph, shrinking
        // the per-warp register footprint F toward the <=63 needed for 4 warps.
#ifdef USE_CISC_QK
        // CISC QK (option 2): issue loop_ws via csrw 0xacc, no muon fences.
        mxgemm_cisc_qk<QK>(
            &QK_A_in[0][0], &QK_B_blocks[j * FA_D][0],
            &QK_A_scales_row[0][0], &QK_B_scales_blocks[j * FA_GK][0],
            FA_SQ, FA_BK, FA_D, tid);
#else
        mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(
            &QK_A_in[0][0], &QK_B_blocks[j * FA_D][0],
            &QK_A_scales_row[0][0], &QK_B_scales_blocks[j * FA_GK][0],
            FA_SQ, FA_BK, FA_D, tid);
#ifdef RELOC_TEST
        mxgemm_compute_tile<QK>(tid, /*c_spad_dest=*/1536);   // S -> spad row 1536 (byte 0x6000)
#elif defined(QK_SPLIT)
        MARK();                          // after prefetch DMA issue
        mxgemm_compute_issue<QK>(tid);   // leading move-in drain + config + matmul ISSUE (no trailing fence)
        MARK();                          // issue side done
        if (tid == 0) gemmini_fence();   // trailing drain: matmul compute + mvout
#else
        mxgemm_compute_tile<QK>(tid);
#endif
#endif
        mu_barrier(2, wpb); MARK();      // (QK_SPLIT: this is the 3rd mark = drain done)


        // PREFETCH V_j for PV: async move-in (no fence) -> overlaps softmax+requant below,
        // hiding PV's DMA/config latency (was ~50 k cyc). SKIP_A (A=P comes from requant).
        mxgemm_prefetch_tile<PV, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(
            &V_in[j * FA_BK][0], &V_in[j * FA_BK][0],
            &V_scales[j * FA_GKB][0], &V_scales[j * FA_GKB][0],
            FA_SQ, FA_D, FA_BK, tid);

        // FUSED online-softmax + MX-FP8 requant: update running m/l, emit corr, and write
        // P_j fp8 directly to the A-spad (tiled) + E8M0 scales -> SCALE_SMEM. No P_SMEM
        // round-trip / double-read (thorough SIMT rewrite; contiguous 16-lane ownership).
        fused_softmax_requant<FA_SQ, FA_BK>(
#ifdef RELOC_TEST
            reinterpret_cast<const __shared uint16_t *>(0x6000),  // read relocated S
#else
            reinterpret_cast<const __shared uint16_t *>(S_SMEM),
#endif
            reinterpret_cast<__shared uint32_t *>(0 /* A-spad base */),
            reinterpret_cast<__shared uint32_t *>(SCALE_SMEM),
            reinterpret_cast<__shared uint16_t *>(M_SMEM),
            reinterpret_cast<__shared uint16_t *>(LS_SMEM),
            reinterpret_cast<__shared uint16_t *>(CORR_SMEM),
            SOFTMAX_SCALE_BF16, first, tid, thr);
        mu_fence_smem();
        mu_barrier(3, wpb); MARK();

        // pack scales -> A scale SRAM (SF_MEM_A) for the PV mesh.
        pack_scales_to_sfmem<FA_SQ, FA_BK>(
            reinterpret_cast<const __shared uint32_t *>(SCALE_SMEM),
            reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_A), tid, thr);
        mu_fence_smem();
        mu_barrier(4, wpb); MARK();

        // PV_j COMPUTE: V_j prefetched during the fused SIMT (async DMA hidden). Drain + matmul.
        mxgemm_compute_tile<PV>(tid);
        mu_barrier(5, wpb); MARK();

        // O_acc = (first ? 0 : O_acc*corr) + PV_j   (PV_j read from SPAD_DEST == S_SMEM).
        rescale_accumulate<FA_SQ, FA_D>(
            reinterpret_cast<__shared uint32_t *>(OACC_SMEM),
            reinterpret_cast<const __shared uint32_t *>(S_SMEM),
            reinterpret_cast<const __shared uint16_t *>(CORR_SMEM), first, tid, thr);
        mu_fence_smem();
        mu_barrier(6, wpb); MARK();
    }

    // ---- finalize: O = O_acc / l  -> GMEM (bf16). ----
    finalize_O<FA_SQ, FA_D>(
        reinterpret_cast<const __shared uint32_t *>(OACC_SMEM),
        reinterpret_cast<const __shared uint16_t *>(LS_SMEM),
        reinterpret_cast<uint32_t *>(O_GMEM), tid, thr);
    MARK();  // final
}

int main() {
    mu_schedule(fa_entry, nullptr, 3);  // 3 warps/core (occupancy): slim QK path dropped F to ~57;
                                        // occ=4 overflowed the 256 phys-reg file, occ=3 (3*57=171)
                                        // has margin. More warps hide the latency-bound SIMT softmax.
    return 0;
}
