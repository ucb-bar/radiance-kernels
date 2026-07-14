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
// ===== BANK-AWARE OVERLAP layout (Sq=64,Bk=64). 4 banks x 32KB; each bank's DMA read queue is
// depth-4 + un-backpressured, and a mesh read starves if a high-rate SIMT write shares its bank.
// During overlap: mesh reads Q(bank0)+K(bank3), writes S[nxt]; SIMT reads S[cur]+scratch, writes
// P. Key trick: P OVERWRITES S[cur] in-place (softmax loads S to regs first, then S[cur] is dead)
// so P always lands on bank_cur -- a bank with NO mesh access. scratch shares bank0 w/ Q (low rate).
// CRITICAL: mesh READ banks (Q=bank0, K=bank3) must be CLEAN of SIMT (SIMT scratch on K's bank
// starved the mesh K-read -> reservation-station stall). So scratch lives on bank1 (an S bank);
// it only shares with SIMT accesses (temporally separated from S-reads). K/V stay bank3 (mesh-only).
//   bank0(0x0)    : Q (mesh A) + PVout (post-drain only)
//   bank1(0x8000) : S0/P0 + O_acc + scratch (all SIMT)
//   bank2(0x10000): S1/P1
//   bank3(0x18000): K/V (mesh B) at top -- CLEAN of SIMT during overlap
static constexpr uint32_t PVOUT_SMEM= 0x2000;   // PV output (bank0; row 512); post-drain only
static constexpr uint32_t S0_SMEM   = 0x8000;   // S buffer 0 (bank1); row 2048
static constexpr uint32_t P0_SMEM   = 0xA000;   // P for cur=0 (bank1, co-located w/ S0); row 2560
static constexpr uint32_t OACC_SMEM = 0xB000;   // O accumulator [Sq][d] bf16 (bank1)
static constexpr uint32_t SCALE_SMEM = 0xF000;  // per-scale word scratch (bank1, packed -> SF-SRAM)
static constexpr uint32_t M_SMEM    = 0xF400;   // running row max (bank1)
static constexpr uint32_t LS_SMEM   = 0xF600;   // running row denom l (bank1)
static constexpr uint32_t CORR_SMEM = 0xF800;   // per-row rescale corr (bank1)
static constexpr uint32_t REDBUF_SMEM = 0xFA00; // per-warp tree-reduce scratch (bank1)
static constexpr uint32_t S1_SMEM   = 0x10000;  // S buffer 1 (bank2); row 4096
static constexpr uint32_t P1_SMEM   = 0x12000;  // P for cur=1 (bank2, co-located w/ S1); row 4608

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
    if (tid == 0) gemmini_flush(0);
    MARK();  // 0: entry

    // ===== ASYNC-OVERLAP streaming FA (softmax || QK on the mesh). Double-buffered S: QK_{j+1}
    // is issued ASYNC into S[nxt] (mesh computes it during softmax_j reading S[cur]); drained
    // before PV_j. Q resident @ row 0 (const); softmax writes P to PSPAD (row 2048) so it never
    // collides with Q. S=16KB double-buffers in-region. =====
    constexpr uint32_t S_ROW[2]  = {S0_SMEM / DIM, S1_SMEM / DIM};   // QK C-output spad rows
    constexpr uint32_t S_BYTE[2] = {S0_SMEM, S1_SMEM};              // softmax S read (byte)
    constexpr uint32_t P_BYTE[2] = {P0_SMEM, P1_SMEM};              // softmax P write (byte, bank_cur)
    constexpr uint32_t P_ROW[2]  = {P0_SMEM / DIM, P1_SMEM / DIM};  // PV A(=P) spad row
    constexpr uint32_t PVOUT_ROW = PVOUT_SMEM / DIM;               // PV C-output spad row

    // Prologue: QK_0 -> S0 (loads Q resident @0 + K_0, configs QK).
    mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(
        &QK_A_in[0][0], &QK_B_blocks[0][0], &QK_A_scales_row[0][0], &QK_B_scales_blocks[0][0],
        FA_SQ, FA_BK, FA_D, tid);
    mxgemm_compute_tile<QK>(tid, /*c_spad=*/S_ROW[0]);
    mu_barrier(1, wpb); MARK();

    for (uint32_t j = 0; j < FA_NBLK; j++) {
        const uint32_t first = (j == 0);
        const uint32_t cur = j & 1, nxt = (j + 1) & 1;

        // Issue QK_{j+1} ASYNC -> S[nxt] (overlaps softmax_j below). Reload Q(@0)+Qscales since
        // the prev block's PV overwrote SF_MEM_A with P scales.
        if (j + 1 < FA_NBLK) {
            mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(
                &QK_A_in[0][0], &QK_B_blocks[(j + 1) * FA_D][0], &QK_A_scales_row[0][0],
                &QK_B_scales_blocks[(j + 1) * FA_GK][0], FA_SQ, FA_BK, FA_D, tid);
            mxgemm_compute_issue<QK>(tid, /*c_spad=*/S_ROW[nxt]);   // async, no fence
        }

        // softmax_j reads S[cur], writes P[cur] -- both on bank_cur (NO mesh access), so no
        // read-queue overflow / mesh-starvation. mesh computes QK_{j+1} concurrently on banks 0/3/nxt.
        fused_softmax_requant<FA_SQ, FA_BK>(
            reinterpret_cast<const __shared uint16_t *>(S_BYTE[cur]),
            reinterpret_cast<__shared uint32_t *>(P_BYTE[cur]),
            reinterpret_cast<__shared uint32_t *>(SCALE_SMEM),
            reinterpret_cast<__shared uint16_t *>(M_SMEM),
            reinterpret_cast<__shared uint16_t *>(LS_SMEM),
            reinterpret_cast<__shared uint16_t *>(CORR_SMEM),
            SOFTMAX_SCALE_BF16, first, tid, thr);
        mu_fence_smem();
        if (j + 1 < FA_NBLK) mxgemm_drain(tid);   // QK_{j+1} done -> S[nxt] ready + SF_MEM free
        mu_barrier(2, wpb); MARK();

        // pack P scales -> SF_MEM_A (safe now: QK_{j+1} drained, no longer reading SF_MEM).
        pack_scales_to_sfmem<FA_SQ, FA_BK>(
            reinterpret_cast<const __shared uint32_t *>(SCALE_SMEM),
            reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_A), tid, thr);
        mu_fence_smem();
        mu_barrier(3, wpb);

        // PV_j: V_j move-in (B-even, K consumed) + config PV; matmul A=P@PSPAD -> C=PVOUT.
        mxgemm_prefetch_tile<PV, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(
            &V_in[j * FA_BK][0], &V_in[j * FA_BK][0],
            &V_scales[j * FA_GKB][0], &V_scales[j * FA_GKB][0], FA_SQ, FA_D, FA_BK, tid);
        mxgemm_compute_tile<PV>(tid, /*c_spad=*/PVOUT_ROW, /*a_spad=*/S_ROW[cur]);
        mu_barrier(4, wpb); MARK();

        // O_acc = (first ? 0 : O_acc*corr) + PV_j  (PV read from PVOUT_SMEM).
        rescale_accumulate<FA_SQ, FA_D>(
            reinterpret_cast<__shared uint32_t *>(OACC_SMEM),
            reinterpret_cast<const __shared uint32_t *>(PVOUT_SMEM),
            reinterpret_cast<const __shared uint16_t *>(CORR_SMEM), first, tid, thr);
        mu_fence_smem();
        mu_barrier(5, wpb); MARK();
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
