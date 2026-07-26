// MXFP8 flash-attention kernel: GQA + causal streaming online-softmax (QK^T ->
// softmax -> requant -> PV) over MX-Gemmini. See the "GQA + causal" comment below
// and README.md for the full description.
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

// ===== GQA + causal (TinyLlama prefill) flash attention =====
// FA_NQ query heads share FA_NKV KV heads in groups of FA_GRP (query head h reads KV head
// h/FA_GRP). Per query head we run a full streaming online-softmax pass over the KV head's
// key blocks, reusing the SAME KV head's K/V across the FA_GRP query heads that share it.
// Causal mask (prefill): query row i has global position FA_QPOS0+i; key col c of block j
// has global position j*FA_BK+c; entries with key_pos > query_pos are masked. Key blocks
// fully above the diagonal (min key > max query) are skipped entirely -> only FA_NBLK_USED
// leading blocks are processed (the data generator emits only those).
//
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
// ===== COLLISION-FREE OVERLAP layout (Sq=64,Bk=64). 4 banks x 32KB. The shared scratchpad
// arbitrates muon-BEFORE-gemmini (lowestIndexFirst), so ANY bank shared by the mesh and the muon
// starves the mesh. The mesh touches {Q,K,V @ bank0 (via B_SPAD_ADDR_EVEN=1024), S[nxt] @ bank1|2}.
// The muon touches {S[cur] @ bank2|1 (opposite of nxt), P[cur] co-located, scratch @ bank3}. Thus
// NO bank is ever shared: mesh S[nxt] (nxt) vs muon S[cur] (cur) differ every iter (nxt!=cur);
// bank0 (mesh Q/K/V) & bank3 (muon scratch) are disjoint. This is what lets QK_{j+1} overlap
// softmax_j without the mesh stalling.
//   bank0(0x0)    : Q (0..0x2000) + K/V (0x2000..0x4000, mesh B) + PVout (0x4000..0x8000, post-drain)
//   bank1(0x8000) : S0/P0 + O_acc
//   bank2(0x10000): S1/P1
//   bank3(0x18000): scratch (muon-only) -- disjoint from all mesh accesses
static constexpr uint32_t PVOUT_SMEM= 0x1A000;  // PV output (bank3-mid, avoids B_ODD/B_EVEN)
static constexpr uint32_t S0_SMEM   = 0x4000;   // S buffer 0 (SPAD_DEST, avoids A_ODD@0x8000)
static constexpr uint32_t P0_SMEM   = 0xA000;   // P for cur=0 (bank1)
static constexpr uint32_t OACC_SMEM = 0xB000;   // O accumulator [Sq][d] bf16 (bank1)
static constexpr uint32_t SCALE_SMEM = 0x14000; // scratch (bank2 = only gemmini-free bank)
static constexpr uint32_t M_SMEM    = 0x14800;  // running row max (bank2)
static constexpr uint32_t LS_SMEM   = 0x14A00;  // running row denom l (bank2)
static constexpr uint32_t CORR_SMEM = 0x14C00;  // per-row rescale corr (bank2)
static constexpr uint32_t REDBUF_SMEM = 0x15000;// per-warp tree-reduce scratch (bank2)
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
    if (tid == 0) gemmini_flush(0);

    // Double-buffered S regions (unchanged bank-partitioned layout; one query head's online
    // state is live at a time, so the layout is identical to the single-head baseline).
    constexpr uint32_t S_ROW[2]  = {S0_SMEM / DIM, S1_SMEM / DIM};   // QK C-output spad rows
    constexpr uint32_t S_BYTE[2] = {S0_SMEM, S1_SMEM};              // softmax S read (byte)
    constexpr uint32_t P_BYTE[2] = {P0_SMEM, P1_SMEM};              // softmax P write (byte, bank_cur)
    constexpr uint32_t P_ROW[2]  = {P0_SMEM / DIM, P1_SMEM / DIM};  // PV A(=P) spad row
    constexpr uint32_t PVOUT_ROW = PVOUT_SMEM / DIM;               // PV C-output spad row

    // ===== GQA outer loop: each query head h runs a full causal streaming FA against KV head
    // kv = h/FA_GRP. Consecutive query heads in a group share the SAME KV head's K/V blocks
    // (same QK_B_blocks/V_in base), so the KV data is reused across the group. =====
    for (uint32_t h = 0; h < FA_NQ; h++) {
        const uint32_t kv = h / FA_GRP;                        // shared KV head
        const uint32_t kvb = kv * FA_NBLK_USED;                // KV head's first key block
        const uint8_t *Qh        = &QK_A_in[h * FA_SQ][0];     // Q for this query head
        const uint8_t *Qh_scales = &QK_A_scales_row[h * FA_GK][0];
        // per-head bf16 output O[h] at O_GMEM + h*Sq*d*2
        uint32_t *Oh = reinterpret_cast<uint32_t *>(O_GMEM + h * (FA_SQ * FA_D * 2));

        // Prologue: QK_0 -> S0 (loads Q_h @0 + K_{kv,0}, configs QK).
        mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(
            Qh, &QK_B_blocks[kvb * FA_D][0], Qh_scales, &QK_B_scales_blocks[kvb * FA_GK][0],
            FA_SQ, FA_BK, FA_D, tid);
        mxgemm_compute_tile<QK>(tid, /*c_spad=*/S_ROW[0]);
        mu_barrier(1, wpb);

        for (uint32_t j = 0; j < FA_NBLK_USED; j++) {
            const uint32_t first = (j == 0);
            const uint32_t cur = j & 1, nxt = (j + 1) & 1;
            const uint32_t key_offset = j * FA_BK;             // this block's global key base
            // causal mask needed only for a block that straddles the diagonal (its max key
            // exceeds the earliest query position). Fully-below blocks -> apply_mask=0.
            const uint32_t apply_mask = (key_offset + FA_BK - 1) > FA_QPOS0;

            // QK_{j+1} of THIS head (serial: computed before softmax_j).
            if (j + 1 < FA_NBLK_USED) {
                mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(
                    Qh, &QK_B_blocks[(kvb + j + 1) * FA_D][0], Qh_scales,
                    &QK_B_scales_blocks[(kvb + j + 1) * FA_GK][0], FA_SQ, FA_BK, FA_D, tid);
                mxgemm_compute_tile<QK>(tid, /*c_spad=*/S_ROW[nxt]);
            }
            mu_barrier(1, wpb);   // QK_{j+1} fully done before softmax (serial)
            // THREAD-PER-ROW single-exp causal softmax: 0 per-row fences (vs 4 cooperative),
            // fewer exps, bit-identical masked P/scales. See flash_mx_impl.hpp.
            fused_softmax_requant_tpr<FA_SQ, FA_BK>(
                reinterpret_cast<const __shared uint16_t *>(S_BYTE[cur]),
                reinterpret_cast<__shared uint32_t *>(P_BYTE[cur]),
                reinterpret_cast<__shared uint32_t *>(SCALE_SMEM),
                reinterpret_cast<__shared uint16_t *>(M_SMEM),
                reinterpret_cast<__shared uint16_t *>(LS_SMEM),
                reinterpret_cast<__shared uint16_t *>(CORR_SMEM),
                SOFTMAX_SCALE_BF16, first, key_offset, FA_QPOS0, apply_mask, tid, thr);
            mu_fence_smem();
            mu_barrier(2, wpb);

            // pack P scales -> SF_MEM_A (safe now: QK_{j+1} drained, no longer reading SF_MEM).
            pack_scales_to_sfmem<FA_SQ, FA_BK>(
                reinterpret_cast<const __shared uint32_t *>(SCALE_SMEM),
                reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_A), tid, thr);
            mu_fence_smem();
            mu_barrier(3, wpb);

            // PV_j: V_j move-in (B-even, K consumed) + config PV; matmul A=P@PSPAD -> C=PVOUT.
            mxgemm_prefetch_tile<PV, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(
                &V_in[(kvb + j) * FA_BK][0], &V_in[(kvb + j) * FA_BK][0],
                &V_scales[(kvb + j) * FA_GKB][0], &V_scales[(kvb + j) * FA_GKB][0],
                FA_SQ, FA_D, FA_BK, tid);
            mxgemm_compute_tile<PV>(tid, /*c_spad=*/PVOUT_ROW, /*a_spad=*/P_ROW[cur]);
            mu_barrier(4, wpb);

            // O_acc = (first ? 0 : O_acc*corr) + PV_j  (PV read from PVOUT_SMEM).
            rescale_accumulate<FA_SQ, FA_D>(
                reinterpret_cast<__shared uint32_t *>(OACC_SMEM),
                reinterpret_cast<const __shared uint32_t *>(PVOUT_SMEM),
                reinterpret_cast<const __shared uint16_t *>(CORR_SMEM), first, tid, thr);
            mu_fence_smem();
            mu_barrier(5, wpb);
        }

        // ---- finalize: O[h] = O_acc / l  -> GMEM (bf16). ----
        finalize_O<FA_SQ, FA_D>(
            reinterpret_cast<const __shared uint32_t *>(OACC_SMEM),
            reinterpret_cast<const __shared uint16_t *>(LS_SMEM),
            Oh, tid, thr);
        mu_barrier(6, wpb);   // finish reading O_acc/l before the next head resets them
    }
}

int main() {
    mu_schedule(fa_entry, nullptr, 2);  // TEMP occ=2 to clear warpspec RF overflow + validate overlap correctness;
                                        // occ=4 overflowed the 256 phys-reg file, occ=3 (3*57=171)
                                        // has margin. More warps hide the latency-bound SIMT softmax.
    return 0;
}
