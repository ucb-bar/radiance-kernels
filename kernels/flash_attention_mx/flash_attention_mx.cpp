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
// retiring ALU pad to break >=3 back-to-back stalling ops (barrier/fence): the barrier RELEASE is a
// single-cycle unbuffered Valid pulse (Synchronizer.sv:87); a retiring commit between stalls restores slack.
#define BAR_PAD() do { volatile int _p=0; asm volatile("addi %0,%0,1" : "+r"(_p)); asm volatile("addi %0,%0,1" : "+r"(_p)); asm volatile("addi %0,%0,1" : "+r"(_p)); asm volatile("addi %0,%0,1" : "+r"(_p)); } while(0)
#define MARK() do { if (tid == 0) { uint32_t _c; asm volatile("csrr %0, mcycle" : "=r"(_c)); \
                                    ((volatile uint32_t *)MARK_GMEM)[mki++] = _c; } } while (0)

// Cooperative SMEM->SMEM copy (all threads), n uint32 words. Used to double-buffer S
// (mesh C-output can't relocate -> copy S off SPAD_DEST so QK_{j+1} can overwrite it).
static __attribute__((noinline)) void copy_smem_u32(__shared uint32_t *dst,
        const __shared uint32_t *src, uint32_t n, uint32_t tid, uint32_t thr) {
    for (uint32_t i = tid; i < n; i += thr) dst[i] = src[i];
}

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

#ifdef WARPSPEC
    // ===== Software-pipelined overlap: QK_{j+1} (async) runs on the mesh while the SIMT
    // does softmax_j. Q persistent @A_EVEN; P @A_ODD(0x8000, off Q); S double-buffered via
    // SIMT-copy (mesh C-output can't relocate). Tests whether the QK drain hides. =====
    {
    constexpr uint32_t SBUF[2] = {0x6000u, 0xA000u};      // S double-buffer (bytes)
    constexpr uint32_t P_LOC_ROW = 2048u;                 // A_ODD row (byte 0x8000)
    constexpr uint32_t SW = (FA_SQ * FA_BK) / 2;          // S words (bf16 packed 2/word)
    // prologue: load Q (persistent) + QK_0 -> SPAD_DEST -> Sbuf[0]
    mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(&QK_A_in[0][0],
        &QK_B_blocks[0][0], &QK_A_scales_row[0][0], &QK_B_scales_blocks[0][0],
        FA_SQ, FA_BK, FA_D, tid);
    mxgemm_compute_tile<QK>(tid);
    MARK();                        // dbg A: QK_0 done
    mu_barrier(2, wpb); MARK();    // dbg B: barrier2 passed
    copy_smem_u32(reinterpret_cast<__shared uint32_t*>(SBUF[0]),
                  reinterpret_cast<const __shared uint32_t*>(S_SMEM), SW, tid, thr);
    MARK();                        // dbg C: copy done
    mu_fence_smem(); mu_barrier(3, wpb); MARK();  // dbg D: prologue done
    for (uint32_t j = 0; j < FA_NBLK; j++) {
        const uint32_t first = (j == 0), cur = j & 1u, nxt = (j + 1u) & 1u;
        // async QK_{j+1} -> SPAD_DEST (SKIP_A: Q persists; K_{j+1}@B_EVEN); overlaps softmax_j
        if (j + 1 < FA_NBLK) {
#ifdef WS_RELOADQ
            mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(&QK_A_in[0][0],
                &QK_B_blocks[(j + 1) * FA_D][0], &QK_A_scales_row[0][0],
                &QK_B_scales_blocks[(j + 1) * FA_GK][0], FA_SQ, FA_BK, FA_D, tid);
#else
            mxgemm_prefetch_tile<QK, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(&QK_A_in[0][0],
                &QK_B_blocks[(j + 1) * FA_D][0], &QK_A_scales_row[0][0],
                &QK_B_scales_blocks[(j + 1) * FA_GK][0], FA_SQ, FA_BK, FA_D, tid);
#endif
#ifdef WSSYNC
            mxgemm_compute_tile<QK>(tid);                 // SYNC: isolate async-vs-structure
#else
            mxgemm_compute_issue<QK>(tid);                // async: no trailing fence
#endif
        }
        fused_softmax_requant<FA_SQ, FA_BK>(
            reinterpret_cast<const __shared uint16_t*>(SBUF[cur]),
            reinterpret_cast<__shared uint32_t*>(0x8000 /*P @ A_ODD, off Q*/),
            reinterpret_cast<__shared uint32_t*>(SCALE_SMEM),
            reinterpret_cast<__shared uint16_t*>(M_SMEM),
            reinterpret_cast<__shared uint16_t*>(LS_SMEM),
            reinterpret_cast<__shared uint16_t*>(CORR_SMEM),
            SOFTMAX_SCALE_BF16, first, tid, thr);
        mu_fence_smem();
        pack_scales_to_sfmem<FA_SQ, FA_BK>(
            reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
            reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A), tid, thr);
        mu_fence_smem(); mu_barrier(4, wpb); MARK();
        // drain QK_{j+1}, stash S_{j+1} -> Sbuf[nxt] (frees SPAD_DEST for PV)
        if (j + 1 < FA_NBLK) {
            if (tid == 0) gemmini_fence();
            mu_barrier(5, wpb);
            copy_smem_u32(reinterpret_cast<__shared uint32_t*>(SBUF[nxt]),
                          reinterpret_cast<const __shared uint32_t*>(S_SMEM), SW, tid, thr);
            mu_fence_smem(); mu_barrier(6, wpb);
        }
        // PV_j: V_j@B_EVEN (K consumed); read P@A_ODD via a_spad_override -> SPAD_DEST
        mxgemm_prefetch_tile<PV, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(&V_in[j * FA_BK][0],
            &V_in[j * FA_BK][0], &V_scales[j * FA_GKB][0], &V_scales[j * FA_GKB][0],
            FA_SQ, FA_D, FA_BK, tid);
        mxgemm_compute_tile<PV>(tid, /*c_spad_dest=*/SPAD_DEST, /*a_spad_override=*/P_LOC_ROW);
        mu_barrier(7, wpb); MARK();
        rescale_accumulate<FA_SQ, FA_D>(
            reinterpret_cast<__shared uint32_t*>(OACC_SMEM),
            reinterpret_cast<const __shared uint32_t*>(S_SMEM),
            reinterpret_cast<const __shared uint16_t*>(CORR_SMEM), first, tid, thr);
        mu_fence_smem(); mu_barrier(1, wpb); MARK();
    }
    }
#elif defined(WARPSPEC2)
    // ===== Overlap redesign v2 (root-cause-driven): ALL matmuls stay A_EVEN (spad0) — the only
    // pattern the gemmini loop double-buffer supports for independent overwrite matmuls. P is staged
    // in SCRATCH by softmax (so a concurrent QK_{j+1} can read Q@spad0), then SIMT-copied to spad0
    // right before PV. Q is reloaded each QK (the P copy clobbers spad0). QK_{j+1} runs on the mesh
    // during softmax_j (the overlap). WS2_SYNC drains QK before softmax (isolates the layout from the
    // async-overlap concurrency bug). =====
    {
    // WS4: a_spad_override is BROKEN (P512 -> 31.3% err), so P MUST end up at row0 for PV. softmax writes P to
    // SCRATCH (P_STAGE), then SIMT-copy P_STAGE->row0 after QK_{j+1} drains (Q consumed). Harden mesh<->SIMT
    // sync (the mark7->8 Heisenbug = a race): mu_fence_smem after every gemmini_fence before SIMT reads SPAD_DEST.
    constexpr uint32_t P_STAGE = 0x8000;               // P (fp8) scratch; copied to row0 pre-PV
    constexpr uint32_t SBUF    = 0xA000;               // S double-buffer (bf16); softmax reads here
    constexpr uint32_t PW = (FA_SQ * FA_BK) / 4;       // P fp8 words (4 B/word)
    constexpr uint32_t SW = (FA_SQ * FA_BK) / 2;       // S bf16 words (2 elems/word)
    // prologue: QK_0 -> S_0@SPAD_DEST -> copy to SBUF  (fine MARKs to localize any prologue hang)
    MARK();  // 1: prologue entry
    mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(&QK_A_in[0][0],
        &QK_B_blocks[0][0], &QK_A_scales_row[0][0], &QK_B_scales_blocks[0][0], FA_SQ, FA_BK, FA_D, tid);
    MARK();  // 2: prefetch QK_0 done
    mxgemm_compute_tile<QK>(tid);
    MARK();  // 3: compute QK_0 done
    mu_fence_smem(); BAR_PAD(); mu_barrier(2, wpb); BAR_PAD(); MARK();  // 4: barrier2 passed
    copy_smem_u32(reinterpret_cast<__shared uint32_t*>(SBUF),
                  reinterpret_cast<const __shared uint32_t*>(S_SMEM), SW, tid, thr);
    MARK();  // 5: copy done
    mu_fence_smem(); BAR_PAD(); mu_barrier(3, wpb); BAR_PAD(); MARK();  // 6: prologue done
    for (uint32_t j = 0; j < FA_NBLK; j++) {
        BAR_PAD();  // loop-top retiring pad
        const uint32_t first = (j == 0);
        // reload Q + issue QK_{j+1} (reads Q@spad0 -> S_{j+1}@SPAD_DEST); overlaps softmax_j
        if (j + 1 < FA_NBLK) {
            mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(&QK_A_in[0][0],
                &QK_B_blocks[(j + 1) * FA_D][0], &QK_A_scales_row[0][0],
                &QK_B_scales_blocks[(j + 1) * FA_GK][0], FA_SQ, FA_BK, FA_D, tid);
#ifdef WS2_SYNC
            mxgemm_compute_tile<QK>(tid);                 // drained before softmax (no overlap)
#else
            mxgemm_compute_issue<QK>(tid);                // async: overlaps softmax_j
#endif
        }
#ifdef WS_GATE
        // MECHANISM TEST: gate consumers so NO softmax SMEM traffic runs during the QK mvout.
        if (tid == 0) gemmini_fence();
        mu_fence_smem(); mu_barrier(3, wpb);
#endif
        // softmax_j: read SBUF (S_j), write P_j -> P_STAGE (NOT spad0 -> QK_{j+1} keeps Q@spad0)
        fused_softmax_requant<FA_SQ, FA_BK>(
            reinterpret_cast<const __shared uint16_t*>(SBUF),
            reinterpret_cast<__shared uint32_t*>(P_STAGE),
            reinterpret_cast<__shared uint32_t*>(SCALE_SMEM),
            reinterpret_cast<__shared uint16_t*>(M_SMEM),
            reinterpret_cast<__shared uint16_t*>(LS_SMEM),
            reinterpret_cast<__shared uint16_t*>(CORR_SMEM),
            SOFTMAX_SCALE_BF16, first, tid, thr);
        mu_fence_smem();
        pack_scales_to_sfmem<FA_SQ, FA_BK>(
            reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
            reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A), tid, thr);
        mu_fence_smem(); mu_barrier(4, wpb); MARK();  // 7(iter0): softmax+pack done
        BAR_PAD();  // break stall-run: bar4 -> [skip tid0 gemmini_fence] -> fence,bar5
        // drain QK_{j+1}, stash S_{j+1} -> SBUF (frees SPAD_DEST for PV)
        if (j + 1 < FA_NBLK) {
            if (tid == 0) gemmini_fence();
            mu_fence_smem(); mu_barrier(5, wpb);   // fence: mesh mvout(S_{j+1}) visible to SIMT before copy
            copy_smem_u32(reinterpret_cast<__shared uint32_t*>(SBUF),
                          reinterpret_cast<const __shared uint32_t*>(S_SMEM), SW, tid, thr);
            mu_fence_smem(); mu_barrier(6, wpb);
        }
        MARK();  // 8: drain+copy-S done
        // copy P_j: P_STAGE -> row0 (A_EVEN); clobbers Q (QK_{j+1} already read+drained it)
        copy_smem_u32(reinterpret_cast<__shared uint32_t*>(0),
                      reinterpret_cast<const __shared uint32_t*>(P_STAGE), PW, tid, thr);
        mu_fence_smem(); mu_barrier(3, wpb); MARK();  // 9: copy-P done (barrier renumbered 7->3: bar7 hangs!)
        BAR_PAD();  // break stall-run: bar3 -> [skip tid0 PV] -> bar1
        // PV_j: reads P@row0 (A_EVEN default), V@B_EVEN -> O_j@SPAD_DEST
        mxgemm_prefetch_tile<PV, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(&V_in[j * FA_BK][0],
            &V_in[j * FA_BK][0], &V_scales[j * FA_GKB][0], &V_scales[j * FA_GKB][0],
            FA_SQ, FA_D, FA_BK, tid);
        mxgemm_compute_tile<PV>(tid);
        mu_barrier(1, wpb); MARK();  // 10: PV done
        rescale_accumulate<FA_SQ, FA_D>(
            reinterpret_cast<__shared uint32_t*>(OACC_SMEM),
            reinterpret_cast<const __shared uint32_t*>(S_SMEM),
            reinterpret_cast<const __shared uint16_t*>(CORR_SMEM), first, tid, thr);
        mu_fence_smem(); mu_barrier(2, wpb); MARK();
    }
    }
#else
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
#ifdef ISO_TK1
            FA_SQ, FA_D, FA_BK, tid, /*tile_k=*/1);   // V -> B_ODD, V-scales -> odd SF (parity-consistent PV)
#else
            FA_SQ, FA_D, FA_BK, tid);
#endif

        // FUSED online-softmax + MX-FP8 requant: update running m/l, emit corr, and write
        // P_j fp8 directly to the A-spad (tiled) + E8M0 scales -> SCALE_SMEM. No P_SMEM
        // round-trip / double-read (thorough SIMT rewrite; contiguous 16-lane ownership).
        fused_softmax_requant<FA_SQ, FA_BK>(
#ifdef RELOC_TEST
            reinterpret_cast<const __shared uint16_t *>(0x6000),  // read relocated S
#else
            reinterpret_cast<const __shared uint16_t *>(S_SMEM),
#endif
#if defined(ISO_P512)
            reinterpret_cast<__shared uint32_t *>(0x2000 /* P @ row 512 (NON-A_ODD, neutral), isolation test */),
#elif defined(ISO_COPYP)
            reinterpret_cast<__shared uint32_t *>(0x8000 /* P @ scratch; SIMT-copy to row0 pre-PV (isolate copy-P) */),
#elif defined(ISO_PODD) || defined(ISO_TK1)
            reinterpret_cast<__shared uint32_t *>(0x8000 /* P @ A_ODD (row 2048), isolation test */),
#else
            reinterpret_cast<__shared uint32_t *>(0 /* A-spad base */),
#endif
            reinterpret_cast<__shared uint32_t *>(SCALE_SMEM),
            reinterpret_cast<__shared uint16_t *>(M_SMEM),
            reinterpret_cast<__shared uint16_t *>(LS_SMEM),
            reinterpret_cast<__shared uint16_t *>(CORR_SMEM),
            SOFTMAX_SCALE_BF16, first, tid, thr);
        mu_fence_smem();
#ifdef ISO_NONMONO
        mu_barrier(4, wpb); MARK();  // softmax (swapped 3<->4 -> seq 2,4,3,5,6 non-monotonic)
#else
        mu_barrier(3, wpb); MARK();
#endif

        // pack scales -> A scale SRAM (SF_MEM_A) for the PV mesh.
        pack_scales_to_sfmem<FA_SQ, FA_BK>(
            reinterpret_cast<const __shared uint32_t *>(SCALE_SMEM),
#ifdef ISO_TK1
            reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_A + GEMMINI_SF_MEM_BUFFER_OFFSET), tid, thr);  // odd SF (PV tile_k=1)
#else
            reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_A), tid, thr);
#endif
        mu_fence_smem();
#ifdef ISO_NONMONO
        mu_barrier(3, wpb); MARK();  // pack (swapped)
#else
        mu_barrier(4, wpb); MARK();
#endif

        // PV_j COMPUTE: V_j prefetched during the fused SIMT (async DMA hidden). Drain + matmul.
#if defined(ISO_BAR7FIX)
        // FIX TEST: pad (retiring ALU) breaks the bar4->inserted b2b run; then inserted->bar5 = 2 b2b (OK).
        BAR_PAD();
        mu_barrier(6, wpb);
        mxgemm_compute_tile<PV>(tid);
#elif defined(ISO_BARF)
        // added barrier WITH fences+nops around it (not instruction-adjacent to other barriers)
        mu_fence_smem();
        for (volatile int _k=0;_k<8;_k++) {}   // real intervening work
        mu_barrier(6, wpb);
        mu_fence_smem();
        for (volatile int _k=0;_k<8;_k++) {}
        mxgemm_compute_tile<PV>(tid);
#elif defined(ISO_BAR3)
        mu_barrier(3, wpb);              // control: reused LOW id 3 in PV pos (expect COMPLETE if bar7 is ID-specific)
        mxgemm_compute_tile<PV>(tid);
#elif defined(ISO_BAR7)
        mu_barrier(7, wpb);              // isolate mu_barrier(7) ALONE; data = baseline-correct
        mxgemm_compute_tile<PV>(tid);
#elif defined(ISO_SELFCOPY)
        // isolate copy_smem_u32 CALL (P stays @row0 baseline-correct; redundant self-copy row0->row0).
        copy_smem_u32(reinterpret_cast<__shared uint32_t*>(0),
                      reinterpret_cast<const __shared uint32_t*>(0), (FA_SQ*FA_BK)/4, tid, thr);
        mu_fence_smem(); mu_barrier(7, wpb);
        mxgemm_compute_tile<PV>(tid);
#elif defined(ISO_COPYP)
        // isolate copy-P: SIMT-copy P from scratch(0x8000) -> row0, then PV reads row0 (default). No pipeline/copy-S.
        copy_smem_u32(reinterpret_cast<__shared uint32_t*>(0),
                      reinterpret_cast<const __shared uint32_t*>(0x8000), (FA_SQ*FA_BK)/4, tid, thr);
        mu_fence_smem(); mu_barrier(7, wpb);
        mxgemm_compute_tile<PV>(tid);
#elif defined(ISO_P512)
        mxgemm_compute_tile<PV>(tid, /*c_spad_dest=*/SPAD_DEST, /*a_spad_override=*/512u);  // P @ row 512 (neutral, even config)
#elif defined(ISO_TK1)
        // parity-consistent PV: tile_k=1 => A_ODD(P@2048)+B_ODD(V)+odd config+odd scales, no address override
        mxgemm_compute_tile<PV>(tid, /*c_spad_dest=*/SPAD_DEST, /*a_ovr=*/0xffffffffu, /*b_ovr=*/0xffffffffu, /*tile_k=*/1);
#elif defined(ISO_PODD)
        mxgemm_compute_tile<PV>(tid, /*c_spad_dest=*/SPAD_DEST, /*a_spad_override=*/2048u);  // P @ A_ODD (tile_k=0 config: MISMATCH)
#else
        mxgemm_compute_tile<PV>(tid);
#endif
        mu_barrier(5, wpb); MARK();

        // O_acc = (first ? 0 : O_acc*corr) + PV_j   (PV_j read from SPAD_DEST == S_SMEM).
        rescale_accumulate<FA_SQ, FA_D>(
            reinterpret_cast<__shared uint32_t *>(OACC_SMEM),
            reinterpret_cast<const __shared uint32_t *>(S_SMEM),
            reinterpret_cast<const __shared uint16_t *>(CORR_SMEM), first, tid, thr);
        mu_fence_smem();
        mu_barrier(6, wpb); MARK();
    }
#endif  // WARPSPEC

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
