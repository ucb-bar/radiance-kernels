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
#include <vx_intrinsics.h>   // vx_core_id() — for the L0d->L1 dcache flush (H8 coherency fix)

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
// Full (non-streaming) attention configs: one big QK and PV matmul (amortize loop_ws FSM overhead,
// eliminate online rescale). QKF: S_full[Sq][Sk] = Q[Sq][d]@K^T[d][Sk] (N=Sk, K=d).
// PVF: O[Sq][d] = P_full[Sq][Sk]@V[Sk][d] (N=d, K=Sk).
constexpr GemmConfig QKF{
    .TILE_M = FA_SQ, .TILE_N = FA_SK, .TILE_K = FA_D,
    .DATATYPE = GemmDatatype::FP8, .QUANT_OUTPUT = false,
};
constexpr GemmConfig PVF{
    .TILE_M = FA_SQ, .TILE_N = FA_D, .TILE_K = FA_SK,
    .DATATYPE = GemmDatatype::FP8, .QUANT_OUTPUT = false,
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
static constexpr uint32_t CORR_SMEM = 0x14C00;
static constexpr uint32_t PACKED_SMEM = 0x14E00; // pre-packed SF-scale words (512B, free: <0x15000 reduce buf)  // per-row rescale corr
static constexpr uint32_t REDBUF_SMEM = 0x15000; // per-warp tree-reduce scratch (was 0xC000)

// Lightweight phase profiler: thread-0 stores the mcycle counter to a GMEM marker array
// at each phase boundary. Parse stores to MARK_GMEM from the .out trace -> per-phase cycles.
static constexpr uint32_t MARK_GMEM = 0x40050000;
// retiring ALU pad to break >=3 back-to-back stalling ops (barrier/fence): the barrier RELEASE is a
// single-cycle unbuffered Valid pulse (Synchronizer.sv:87); a retiring commit between stalls restores slack.
// BUG FIX (2026-07-25, FSDB-confirmed on flash_util2.fsdb): `_p` used to be declared `volatile int`, which
// forces a STACK slot -- and the stack lives in GMEM/DRAM. Each `"+r"(_p)` asm therefore compiled to a
// DRAM load + DRAM store around the addi, so the intended "4 retiring ALU ops" were really ~9-12 DRAM
// round-trips per BAR_PAD, x16 lanes x6 warps. Measured: the 10,443-cyc "bar2" phase was
// BAR_PAD#1 = 2,721 cyc | actual barrier handshake = 3 cyc (0.03%) | BAR_PAD#2 + MARK = 7,719 cyc, with
// lsu.io_globalQueuesEmpty low for 913/668/553/543 cyc across BAR_PAD#1's four addi's. Dropping `volatile`
// keeps _p in a register (still `asm volatile` so the addis are not optimized away) and restores the
// original intent. For the record: mu_barrier itself is 3 cyc and fence.s is 2 cyc -- barriers are FREE.
//
// *** WARNING -- THE FAST PAD IS NOT FREE. READ BEFORE CHANGING THE DEFAULT BELOW. ***
// Making ALL pads fast is worth -18.3% (188,533 -> 153,957 cyc single tile) but CORRUPTS THE OUTPUT
// of the first tile. Measured 2026-07-25, FULL_ATTN2, Frobenius vs golden_O_u16 (the CORRECT golden
// for this non-streaming path -- see the golden note on the FULL_ATTN2 block; 3.5666% == correct):
//
//   config                     total cyc   Frobenius   verdict
//   slow pad everywhere         188,533     3.5666%    correct (== the c3156c6 baseline)
//   all pads fast, FA_STEADY    153,957     6.1886%    CORRUPT
//   all pads fast, plain        154,225    18.1197%    CORRUPT (same code as above + no loop!)
//   FA_SLOWPAD2 (site 2 only)   173,657     3.5666%    correct
//   FA_SLOWPAD3 (site 3 only)   170,217     3.5666%    correct
//   FA_SLOWPAD4 (site 4 only)   168,646     3.5666%    correct  <-- cheapest correct, the DEFAULT
//
// Two facts pin down what this is. (1) A slow pad at ANY ONE of the three sites is sufficient, so it
// is not a single missing drain at a single point -- it is a marginal timing/layout-sensitive hazard
// that ~10k cycles of delay anywhere in the tile happens to close. (2) The two "all fast" rows are the
// same pads with only the FA_STEADY loop differing, yet give 6.19% vs 18.12% -- severity tracks
// unrelated code layout. Consistent with fence.s waiting ONLY on the Muon per-warp shared LSU queues:
// it does NOT wait on the Gemmini mvout, the V-mvin DMA, or the SF-SRAM scale writes.
// (3) In a 4-tile FA_STEADY run with ALL pads fast the FINAL tile's O is exactly correct (3.5666%),
// so the corruption is a COLD-START effect on tile 0 only; steady-state tiles are clean. That is why
// the steady-state slope below is still a valid measurement of a correct tile.
//
// THE PAD IS A TIMING MASK, NOT A FIX. The real fix is an explicit drain at the responsible point;
// that needs an FSDB root-cause. Until then site 4 stays slow so the DEFAULT BUILD IS CORRECT.
// FA_FASTPAD_ALL   = all three sites fast (fastest, CORRUPTS TILE 0 -- measurement use only).
// FA_SLOWPAD_ALL   = all three slow (the original verified 188,994-cyc behaviour).
// FA_SLOWPAD<n>    = force site n slow on top of whatever the default is.
#define BAR_PAD_FAST() do { int _p=0; asm volatile("addi %0,%0,1" : "+r"(_p)); asm volatile("addi %0,%0,1" : "+r"(_p)); asm volatile("addi %0,%0,1" : "+r"(_p)); asm volatile("addi %0,%0,1" : "+r"(_p)); } while(0)
#define BAR_PAD_SLOW() do { volatile int _p=0; asm volatile("addi %0,%0,1" : "+r"(_p)); asm volatile("addi %0,%0,1" : "+r"(_p)); asm volatile("addi %0,%0,1" : "+r"(_p)); asm volatile("addi %0,%0,1" : "+r"(_p)); } while(0)
// The GENERIC BAR_PAD() stays SLOW: it is used by FULL_ATTN, FULL_ATTN3, FA_HWREQ and the WARPSPEC*
// paths, none of which have been re-validated against the fast pad. Only the FULL_ATTN2 sites below
// opt in, and only where a correct O was actually measured. FA_FASTPAD_ALL makes everything fast.
#if defined(FA_FASTPAD_ALL)
#define BAR_PAD() BAR_PAD_FAST()
#else
#define BAR_PAD() BAR_PAD_SLOW()
#endif
// Per-barrier-site pads on the FULL_ATTN2 path: site 2 = post-QK / pre-softmax, site 3 = post-pack /
// pre-PV, site 4 = post-PV / pre-finalize. Sites 2 and 3 default FAST; site 4 defaults SLOW because
// that is the cheapest configuration measured to produce a CORRECT O (see the table above).
// VALIDATED 2026-07-25 at this default: FULL_ATTN2 single tile = 132,016 cyc, Frobenius 3.5666%.
#if defined(FA_SLOWPAD2) || defined(FA_SLOWPAD_ALL)
#define BAR_PAD2() BAR_PAD_SLOW()
#else
#define BAR_PAD2() BAR_PAD_FAST()
#endif
#if defined(FA_SLOWPAD3) || defined(FA_SLOWPAD_ALL)
#define BAR_PAD3() BAR_PAD_SLOW()
#else
#define BAR_PAD3() BAR_PAD_FAST()
#endif
#if defined(FA_SLOWPAD4) || defined(FA_SLOWPAD_ALL) || !defined(FA_FASTPAD_ALL)
#define BAR_PAD4() BAR_PAD_SLOW()
#else
#define BAR_PAD4() BAR_PAD_FAST()
#endif
#define MARK() do { if (tid == 0) { uint32_t _c; asm volatile("csrr %0, mcycle" : "=r"(_c)); \
                                    ((volatile uint32_t *)MARK_GMEM)[mki++] = _c; } } while (0)

// Cooperative SMEM->SMEM copy (all threads), n uint32 words. Used to double-buffer S
// (mesh C-output can't relocate -> copy S off SPAD_DEST so QK_{j+1} can overwrite it).
static __attribute__((noinline)) void copy_smem_u32(__shared uint32_t *dst,
        const __shared uint32_t *src, uint32_t n, uint32_t tid, uint32_t thr) {
    for (uint32_t i = tid; i < n; i += thr) dst[i] = src[i];
}

// GMEM(device addr) -> SMEM word copy (for reading a gemmini accmem->GMEM mvout back into the spad).
static __attribute__((noinline)) void copy_gmem_to_smem_u32(__shared uint32_t *dst,
        const volatile uint32_t *src, uint32_t n, uint32_t tid, uint32_t thr) {
    for (uint32_t i = tid; i < n; i += thr) dst[i] = src[i];
}

// ===========================================================================================
// FA_PIPE -- SOFTWARE-PIPELINED, RESOURCE-SPLIT steady-state MX-FP8 flash attention.
//
// THE IDEA.  A tile's cost is dominated by two DISJOINT resources that the sequential kernel
// runs one after the other:
//   (a) THREAD-0 SERIAL "gemmini agent" work: 704 4-byte stores into the MX scale SRAM
//       (Q 64 + K 256 + V 256 + packed-P 128 words) plus the ROCC config / mvin / matmul
//       issue.  The SF port is ~30-60 cyc/word of pure store latency on ONE thread and it
//       CANNOT be parallelised: FlitMergeNode.scala:62 asserts that consecutive 4-byte
//       flits are address-consecutive, so any lane-parallel pattern $finishes the sim
//       (measured: `lanesc` below dies at FlitMergeNode after ~45k cycles).
//   (b) ALL-THREAD SIMT work: softmax, requant, finalize.
//   (c) the MESH: QK + PV = 16,420 cycles of MACs, which reads its operands through the
//       gemmini's own spad read ports and needs ZERO muon SMEM line-slots.
// Because mu_barrier costs 3 cycles (Synchronizer.sv) these can be interleaved at fine
// grain.  FA_PIPE gives warp 0 the agent role and warps 1..5 the SIMT role, and slides the
// agent's work one stage AHEAD so that the 320 words of Q/K scales for tile i+1 are written
// while tile i's softmax runs.
//
// WHAT MAKES THE SLIDE LEGAL -- the MX scale double buffer.  ScalingFactorMem is split into
// two halves selected per-matmul by gemmini_mxquant_config_mvout(.., act_sel, w_sel, ..).
// FA_PIPE pins   QK -> half 0   and   PV -> half 1   (instead of the sequential kernel's
// "everything in half 0, overwrite as you go").  Then:
//   * V scales (B half 1) may be written while the QK matmul is reading B half 0;
//   * the packed P scales (A half 1) may be written while ... nothing, but they no longer
//     clobber the Q scales, so
//   * tile i+1's Q/K scales (half 0) may be written any time after tile i's QK has drained,
//     i.e. underneath tile i's softmax+requant.
// The spad addresses stay on the EVEN buffers for both gemms (a_spad/b_spad overrides), so
// this changes ONLY the scale-SRAM half, never the operand layout.
// ===========================================================================================
#if defined(FA_PIPE) || defined(FA_SFPAR)
static constexpr uint32_t FA_A_SPAD_EVEN = 0;                       // A operand spad row
static constexpr uint32_t FA_B_SPAD_EVEN = BANK_NUM * BANK_ROWS;    // B operand spad END row

// Register-only barrier pad (see the BAR_PAD note above: a stack-resident pad is DRAM).
#define FAP_PAD() do { int _q = 0;                       \
    asm volatile("addi %0,%0,1" : "+r"(_q));             \
    asm volatile("addi %0,%0,1" : "+r"(_q));             \
    asm volatile("addi %0,%0,1" : "+r"(_q));             \
    asm volatile("addi %0,%0,1" : "+r"(_q)); } while (0)
#define FAP_BAR(id) do { mu_fence_smem(); FAP_PAD(); mu_barrier((id), wpb); FAP_PAD(); } while (0)

// ---- agent primitives (thread 0 only; each is a separate noinline fn so the warp-uniform
// ---- `if (warp == 0)` branches stay compact and no barrier can be duplicated into them) ----

// config + issue the A/B GMEM->spad move-in for gemm C.  Async: no drain.
template <GemmConfig C, bool SKIP_A>
static __attribute__((noinline)) void fap_cfg_mvin(const uint8_t *A_in, const uint8_t *B_in,
        uint32_t m, uint32_t n, uint32_t k, uint32_t par, uint32_t tid) {
    if (tid != 0) return;
    configure_mxgemmini<C>(m, n, k, /*scale_w_sel=*/par, /*scale_act_sel=*/par);
    // tile_k = 0 => the DMA targets the EVEN spad buffers for BOTH gemms (the scale-SRAM
    // half is decoupled from the spad parity on purpose -- see the header comment).
    copy_gmem_to_smem_async<C, SKIP_A, /*EXPLICIT_MVIN=*/false>(A_in, B_in, m, n, k, 0, 0, 0);
}

// A-operand (activation) scales GMEM -> SF_MEM_A half `par`.
template <GemmConfig C>
static __attribute__((noinline)) void fap_scales_a(const uint8_t *As, uint32_t m, uint32_t n,
                                                   uint32_t par, uint32_t tid) {
    if (tid != 0) return;
    load_scale_factors(calculate_scale_factor_smem_addr<false>(par),
                       calculate_scale_factor_gmem_addr<C, false>(As, 0, m, n),
                       C.SCALE_FACTORS_PER_TILE());
}
// B-operand (weight) scales GMEM -> SF_MEM_B half `par`.
template <GemmConfig C>
static __attribute__((noinline)) void fap_scales_b(const uint8_t *Bs, uint32_t m, uint32_t n,
                                                   uint32_t par, uint32_t tid) {
    if (tid != 0) return;
    load_scale_factors(calculate_scale_factor_smem_addr<true>(par),
                       calculate_scale_factor_gmem_addr<C, true>(Bs, 0, m, n),
                       C.SCALE_FACTORS_PER_TILE_B());
}

// Issue the matmul (C -> c_dest) with an explicit scale-SRAM half.  NO trailing fence, so the
// caller can keep the agent busy while the mesh runs.
template <GemmConfig C>
static __attribute__((noinline)) void fap_mm_issue(uint32_t par, uint32_t c_dest, uint32_t tid) {
    if (tid != 0) return;
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        /*scale_act_sel=*/par, /*scale_w_sel=*/par, QUANT_LUT_UPDATE_GRANULARITY);
    matmul_tile_async<C>(/*tile_k=*/0, /*acc_move_out=*/true, /*accumulate=*/false,
                         /*b_spad_override=*/FA_B_SPAD_EVEN, /*c_spad_dest=*/c_dest,
                         /*a_spad_override=*/FA_A_SPAD_EVEN, /*force_first=*/1);
}

static __attribute__((noinline)) void fap_fence(uint32_t tid) {
    if (tid != 0) return;
    gemmini_fence();
}
#endif  // FA_PIPE || FA_SFPAR

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

#ifdef QKF_ONLY
    // QK-full correctness + timing: S_full[64][256] = Q@K^T in ONE loop_ws matmul. Verify vs golden_S.
    {
    mxgemm_prefetch_tile<QKF, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(
        &QK_A_in[0][0], &QK_B_in[0][0], &QK_A_scales_row[0][0], &QK_B_scales_col[0][0],
        FA_SQ, FA_SK, FA_D, tid);
    MARK();  // 1: prefetch issued
    mxgemm_compute_tile<QKF>(tid);           // S_full -> SPAD_DEST
    MARK();  // 2: QK_full done (mark1->2 = full QK cost)
    mu_fence_smem(); mu_barrier(2, wpb);
    // SIMT-copy S_full (bf16 [64][256] = 16384 u32 words) SPAD_DEST -> S_GMEM for verify
    for (uint32_t i = tid; i < (FA_SQ * FA_SK) / 2; i += thr)
        ((volatile uint32_t*)S_GMEM)[i] = ((const __shared uint32_t*)S_SMEM)[i];
    mu_fence_smem(); mu_barrier(3, wpb);
    MARK();  // 3: copy-out done
    }
#elif defined(FA_HWREQ)
    // ===== HW MxRequantizer ISOLATION TEST: QK_full -> online_softmax(bf16 P) -> feed P through the HW
    // requantizer (write bf16 -> GEMMINI_REQUANT, HW emits e4m3 @ spad0) -> dump spad0 fp8 -> verify vs
    // golden_Pnfp8 (row-major). Tests whether the HW requant produces correct fp8 VALUES. =====
    {
    constexpr uint32_t PBF = 0xC000;  // bf16 P scratch
    mxgemm_prefetch_tile<QKF, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(
        &QK_A_in[0][0], &QK_B_in[0][0], &QK_A_scales_row[0][0], &QK_B_scales_col[0][0],
        FA_SQ, FA_SK, FA_D, tid);
    MARK();  // 1
    mxgemm_compute_tile<QKF>(tid);
    MARK();  // 2
    mu_fence_smem(); BAR_PAD(); mu_barrier(2, wpb); BAR_PAD(); MARK();  // 3
    online_softmax_block<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(S_SMEM),
        reinterpret_cast<__shared uint32_t*>(PBF),
        reinterpret_cast<__shared uint16_t*>(M_SMEM),
        reinterpret_cast<__shared uint16_t*>(LS_SMEM),
        reinterpret_cast<__shared uint16_t*>(CORR_SMEM),
        SOFTMAX_SCALE_BF16, /*first=*/1, tid, thr);
    mu_fence_smem(); BAR_PAD(); mu_barrier(3, wpb); BAR_PAD(); MARK();  // 4: softmax done
    // latch the requantizer to FP8 output (RQ config: QUANT_OUTPUT=true -> GEMMINI_FORMAT_OUT=FP8)
    if (tid == 0) configure_mxgemmini<RQ>(FA_SQ, FA_SK, FA_SK);
    mu_fence_smem(); BAR_PAD(); mu_barrier(4, wpb); BAR_PAD(); MARK();  // 5: requant configured
    // warp0 streams bf16 P (PBF) -> GEMMINI_REQUANT (program-order); HW emits e4m3 @ spad0 (offset>>1)
    copy_P_to_requant<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(PBF),
        reinterpret_cast<__shared uint16_t*>(GEMMINI_REQUANT), tid);
    mu_fence_smem();
    if (tid == 0) gemmini_fence();     // wait for HW requantizer to finish emitting fp8 -> spad0
    mu_fence_smem(); BAR_PAD(); mu_barrier(5, wpb); BAR_PAD(); MARK();  // 6: P->requant done (HW drained)
    // dump fp8 P from spad0 (row0) -> S_GMEM for verify (HW emitted here). [64][256] fp8 = 4096 words
    for (uint32_t i = tid; i < (FA_SQ*FA_SK)/4; i += thr)
        ((volatile uint32_t*)S_GMEM)[i] = ((const __shared uint32_t*)0)[i];
    mu_fence_smem(); MARK();  // 7
    }
#elif defined(FULL_ATTN3)
    // ===== HW MxRequantizer in the REAL pipeline (feeds PVF, no debug dump). QK_full -> online_softmax(bf16 P)
    // -> SIMT scale compute (requant_P_to_spad_tiled: se->SCALE_SMEM, fp8->spad0) -> pack_scales->SF_MEM_A
    // -> HW requant (P->GEMMINI_REQUANT, HW OVERWRITES spad0 with e4m3) -> PVF(spad0 fp8 + SF_MEM_A scales)
    // -> finalize. Tests whether the HW fp8 is correct/usable in-pipeline (vs my earlier debug-dump hang). =====
    {
    constexpr uint32_t PBF = 0xC000;
    mxgemm_prefetch_tile<QKF, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(
        &QK_A_in[0][0], &QK_B_in[0][0], &QK_A_scales_row[0][0], &QK_B_scales_col[0][0], FA_SQ, FA_SK, FA_D, tid);
    MARK();  // 1
    mxgemm_compute_tile<QKF>(tid);
    MARK();  // 2
    mu_fence_smem(); BAR_PAD(); mu_barrier(2, wpb); BAR_PAD(); MARK();  // 3
    online_softmax_block<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(S_SMEM), reinterpret_cast<__shared uint32_t*>(PBF),
        reinterpret_cast<__shared uint16_t*>(M_SMEM), reinterpret_cast<__shared uint16_t*>(LS_SMEM),
        reinterpret_cast<__shared uint16_t*>(CORR_SMEM), SOFTMAX_SCALE_BF16, /*first=*/1, tid, thr);
    mu_fence_smem(); MARK();  // 4: softmax done
    // SIMT scale compute (+ fp8, which HW will overwrite): fills SCALE_SMEM
    requant_P_to_spad_tiled<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint16_t*>(PBF), reinterpret_cast<__shared uint32_t*>(0),
        reinterpret_cast<__shared uint32_t*>(SCALE_SMEM), tid, thr);
    mu_fence_smem(); MARK();  // 5: SIMT scales+fp8 done
    pack_scales_to_sfmem<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
        reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A), tid, thr);
    mu_fence_smem(); BAR_PAD(); mu_barrier(3, wpb); BAR_PAD(); MARK();  // 6: pack done
    // latch requantizer FP8, then warp0 streams bf16 P -> GEMMINI_REQUANT; HW emits e4m3 @ spad0 (OVERWRITES SIMT fp8)
    if (tid == 0) configure_mxgemmini<RQ>(FA_SQ, FA_SK, FA_SK);
    mu_fence_smem(); BAR_PAD(); mu_barrier(4, wpb); BAR_PAD(); MARK();  // 7: requant configured
    copy_P_to_requant<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(PBF), reinterpret_cast<__shared uint16_t*>(GEMMINI_REQUANT), tid);
    mu_fence_smem();
    if (tid == 0) gemmini_fence();
    mu_fence_smem(); BAR_PAD(); mu_barrier(5, wpb); BAR_PAD(); MARK();  // 8: HW requant done
    // PVF reads spad0 (HW fp8) + SF_MEM_A (SIMT scales)
    mxgemm_prefetch_tile<PVF, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(
        &V_in[0][0], &V_in[0][0], &V_scales[0][0], &V_scales[0][0], FA_SQ, FA_D, FA_SK, tid);
    MARK();  // 9
    mxgemm_compute_tile<PVF>(tid);
    mu_fence_smem(); BAR_PAD(); mu_barrier(6, wpb); BAR_PAD(); MARK();  // 10: PV done
    finalize_O<FA_SQ, FA_D>(
        reinterpret_cast<const __shared uint32_t*>(S_SMEM),
        reinterpret_cast<const __shared uint16_t*>(LS_SMEM), reinterpret_cast<uint32_t*>(O_GMEM), tid, thr);
    MARK();  // 11
    }
#elif defined(FULL_ATTN)
    // ===== Non-streaming (full) attention: ONE QK matmul over the whole Sk=256, ONE softmax (no online
    // correction => NO rescale), ONE PV matmul. Amortizes loop_ws FSM + config overhead over 4x the work
    // and eliminates the per-block rescale. No mesh<->SIMT overlap => no Hazard-2 race => completes. =====
    {
    // QK_full: S_full[Sq][Sk] = Q @ K^T  -> SPAD_DEST
    mxgemm_prefetch_tile<QKF, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(
        &QK_A_in[0][0], &QK_B_in[0][0], &QK_A_scales_row[0][0], &QK_B_scales_col[0][0],
        FA_SQ, FA_SK, FA_D, tid);
    MARK();  // 1: QK prefetch issued
    mxgemm_compute_tile<QKF>(tid);
    MARK();  // 2: QK_full done
    mu_fence_smem(); MARK();  // 3: fence after QK
    BAR_PAD(); mu_barrier(2, wpb); BAR_PAD(); MARK();  // 4: barrier2 passed
    // softmax_full over [Sq][Sk], single pass (first=1 => no online correction), P_full -> row0, l -> LS_SMEM
    fused_softmax_requant<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint16_t*>(S_SMEM),
        reinterpret_cast<__shared uint32_t*>(0),
        reinterpret_cast<__shared uint32_t*>(SCALE_SMEM),
        reinterpret_cast<__shared uint16_t*>(M_SMEM),
        reinterpret_cast<__shared uint16_t*>(LS_SMEM),
        reinterpret_cast<__shared uint16_t*>(CORR_SMEM),
        SOFTMAX_SCALE_BF16, /*first=*/1, tid, thr);
    mu_fence_smem(); MARK();  // 5: softmax done
    pack_scales_to_sfmem<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
        reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A), tid, thr);
    mu_fence_smem(); MARK();  // 6: pack done
#ifdef FA_DUMPP
    for (uint32_t i = tid; i < (FA_SQ*FA_SK)/4; i += thr)
        ((volatile uint32_t*)S_GMEM)[i] = ((const __shared uint32_t*)0)[i];        // fp8 P spad0->S_GMEM
    for (uint32_t i = tid; i < ((FA_SK/32)*FA_SQ)/4; i += thr)
        ((volatile uint32_t*)PS_GMEM)[i] = ((const __shared uint32_t*)GEMMINI_SF_MEM_A)[i]; // scales->PS_GMEM
    mu_fence_smem();
#endif
#ifdef FA_DUMPL
    // dump l (LS_SMEM bf16 [64]) -> L_GMEM for verify
    for (uint32_t i = tid; i < FA_SQ/2; i += thr)
        ((volatile uint32_t*)L_GMEM)[i] = ((const __shared uint32_t*)LS_SMEM)[i];
    mu_fence_smem();
#endif
    BAR_PAD(); mu_barrier(3, wpb); BAR_PAD(); MARK();  // 7: barrier3 passed
    // PV_full: O[Sq][d] = P_full @ V  -> SPAD_DEST (overwrites S_full, already consumed)
    mxgemm_prefetch_tile<PVF, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(
        &V_in[0][0], &V_in[0][0], &V_scales[0][0], &V_scales[0][0],
        FA_SQ, FA_D, FA_SK, tid);
    MARK();  // 8: PV prefetch issued
    mxgemm_compute_tile<PVF>(tid);
    MARK();  // 9: PV_full done
    mu_fence_smem(); BAR_PAD(); mu_barrier(4, wpb); BAR_PAD(); MARK();  // 10: barrier4 passed
#ifdef FA_DUMPL
    // dump O_unnorm (SPAD_DEST bf16 [64][128]) -> P_GMEM (reuse) for verify
    for (uint32_t i = tid; i < (FA_SQ*FA_D)/2; i += thr)
        ((volatile uint32_t*)P_GMEM)[i] = ((const __shared uint32_t*)S_SMEM)[i];
    mu_fence_smem();
#endif
    // finalize: O = O_unnorm / l -> O_GMEM
    finalize_O<FA_SQ, FA_D>(
        reinterpret_cast<const __shared uint32_t*>(S_SMEM),
        reinterpret_cast<const __shared uint16_t*>(LS_SMEM),
        reinterpret_cast<uint32_t*>(O_GMEM), tid, thr);
    MARK();  // 6: finalize done
    }
#elif defined(FULL_ATTN2)
    // *** WHICH GOLDEN (2026-07-25): verify FULL_ATTN2 against golden_O_u16.npy, NOT
    // golden_O_flash_u16.npy. fa_lanes_check.sh and most of the notes use the _flash_ one, which is
    // WRONG for this path: fa_gen_goldens.py builds golden_O_flash from mx_attention_flash() with
    // block_n=64, i.e. the STREAMING online-softmax reference, whereas FULL_ATTN2 is the
    // NON-streaming path (one QK over all Sk, one softmax, one PV) whose reference is the dense
    // mx_attention_dense() -> golden_O_u16.npy. The SAME correct output scores 3.5666% against
    // golden_O_u16 and 4.5954% against golden_O_flash, so "4.5954%" in the older notes means
    // "correct, measured against the wrong golden". Use 3.5666% / golden_O_u16.npy as the criterion.
    // ===== Full attention, SEPARATED softmax + requant to kill the width-256 subbank conflict.
    // online_softmax_block writes bf16 P row-major (NO tiled fp8 store -> no conflict); then
    // requant_P_to_spad_tiled (row-parallel: subbank=((row&3)*4+w) -> spreads across 4 subbanks,
    // 4-way conflict vs the fused per-lane version's 16-way) converts bf16 P -> fp8 tiled. =====
    {
    constexpr uint32_t PBF = 0xC000;  // bf16 P_full[64][256]=32KB scratch (OACC region, free pre-PV)
#ifdef FA_PIPE
    // ======================= SOFTWARE-PIPELINED BODY (see the header comment) ===============
    // NTILES selection (mirrors the FA_STEADY harness; FA_PIPE always loops).
#  if   defined(FA_NT1)
#    define FA_PTILES 1
#  elif defined(FA_NT2)
#    define FA_PTILES 2
#  elif defined(FA_NT3)
#    define FA_PTILES 3
#  elif defined(FA_NT4)
#    define FA_PTILES 4
#  elif defined(FA_NT6)
#    define FA_PTILES 6
#  elif defined(FA_NT8)
#    define FA_PTILES 8
#  else
#    define FA_PTILES 4
#  endif
    {
    const uint32_t warp  = tid / MU_NUM_THREADS;
    const uint32_t stid  = tid - MU_NUM_THREADS;          // tid within the warps-1..5 SIMT group
    const uint32_t sthr  = thr - MU_NUM_THREADS;          // 80 threads = 5 warps
    // Scale-SRAM halves: QK -> 0, PV -> 1 (see header).  A-spad/B-spad stay EVEN for both.
    constexpr uint32_t QKP = 0, PVP = 1;

    // ---- PROLOGUE: tile 0's Q/K scales.  Every later tile's are written one stage early,
    // ---- underneath the previous tile's softmax (agent stage S2 below).
    fap_scales_a<QKF>(&QK_A_scales_row[0][0], FA_SQ, FA_SK, QKP, tid);
    fap_scales_b<QKF>(&QK_B_scales_col[0][0], FA_SQ, FA_SK, QKP, tid);
    FAP_BAR(2);

    for (uint32_t ft = 0; ft < (uint32_t)FA_PTILES; ft++) {
    MARK();   // p0: TOP of the pipelined iteration (steady-state slope is measured here)

    // ================= S0: [agent] QK config + Q/K mvin issue  ||  [SIMT] finalize(i-1) =====
    // finalize reads SPAD_DEST (= O of tile i-1) and writes GMEM; the mvin DMA writes the
    // A/B operand spads.  Disjoint.  This is the ONLY place finalize can hide, because every
    // later stage either overwrites SPAD_DEST (QK mvout) or needs all the SIMT warps.
    if (warp == 0) {
        fap_cfg_mvin<QKF, /*SKIP_A=*/false>(&QK_A_in[0][0], &QK_B_in[0][0],
                                            FA_SQ, FA_SK, FA_D, QKP, tid);
    } else if (ft != 0) {
        finalize_O<FA_SQ, FA_D>(reinterpret_cast<const __shared uint32_t*>(S_SMEM),
                                reinterpret_cast<const __shared uint16_t*>(LS_SMEM),
                                reinterpret_cast<uint32_t*>(O_GMEM), stid, sthr);
    }
    FAP_BAR(3);
    MARK();   // p1: S0 done

    // ================= S1: [agent] QK matmul  ||  [agent] V scales -> SF_B half 1 ===========
    // The mesh reads B-scale half 0 while thread 0 fills half 1: different halves of the
    // scale SRAM, and the SF write port is a TL slave that is independent of the mesh's read
    // port.  The SIMT warps are parked at the barrier (no SMEM traffic) so the S mvout gets
    // its atomic 16-subbank grant (Hazard 2) unopposed.
    if (warp == 0) {
        fap_fence(tid);                                     // drain the Q/K mvin
        fap_mm_issue<QKF>(QKP, SPAD_DEST, tid);             // S = Q@K^T -> SPAD_DEST (async)
#ifndef FA_PIPE_NOVS1
        fap_scales_b<PVF>(&V_scales[0][0], FA_SQ, FA_D, PVP, tid);   // 256 words under the mesh
#endif
        fap_fence(tid);                                     // drain matmul + S mvout
    }
    FAP_BAR(4);
    MARK();   // p2: QK done (mesh 8,210 of this)

    // ========== S2: [SIMT 5 warps] softmax(i)  ||  [agent] PV cfg + V mvin + Q/K scales(i+1) =
    // The agent's 320-word Q/K scale write for the NEXT tile is the whole point of the
    // pipeline: it is the single most expensive agent item and it has no consumer until the
    // next tile's S1, so it hides completely under softmax+requant.
    if (warp == 0) {
        fap_cfg_mvin<PVF, /*SKIP_A=*/true>(&V_in[0][0], &V_in[0][0],
                                           FA_SQ, FA_D, FA_SK, PVP, tid);
#ifdef FA_PIPE_NOVS1
        fap_scales_b<PVF>(&V_scales[0][0], FA_SQ, FA_D, PVP, tid);
#endif
#ifndef FA_PIPE_NOQKPRE
        if (ft + 1 < (uint32_t)FA_PTILES) {
            fap_scales_a<QKF>(&QK_A_scales_row[0][0], FA_SQ, FA_SK, QKP, tid);
            fap_scales_b<QKF>(&QK_B_scales_col[0][0], FA_SQ, FA_SK, QKP, tid);
        }
#endif
    } else {
        online_softmax_block<FA_SQ, FA_SK>(
            reinterpret_cast<const __shared uint32_t*>(S_SMEM),
            reinterpret_cast<__shared uint32_t*>(PBF),
            reinterpret_cast<__shared uint16_t*>(M_SMEM),
            reinterpret_cast<__shared uint16_t*>(LS_SMEM),
            reinterpret_cast<__shared uint16_t*>(CORR_SMEM),
            SOFTMAX_SCALE_BF16, /*first=*/1, stid, sthr);
    }
    FAP_BAR(5);
    MARK();   // p3: softmax done (and the agent's next-tile scales are already in the SRAM)

    // ================= S3: [all 6 warps] requant bf16 P -> fp8 @ spad0 + E8M0 scales ========
    requant_P_to_spad_tiled<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint16_t*>(PBF),
        reinterpret_cast<__shared uint32_t*>(0),
        reinterpret_cast<__shared uint32_t*>(SCALE_SMEM), tid, thr);
    FAP_BAR(6);
    MARK();   // p4: requant done

    // ================= S4: [all] prepack the E8M0 words  ->  [agent] ascending SF copy ======
    // The 4-load-per-word packing math runs on 96 threads; thread 0 is then left with a pure
    // ascending 128-word copy, which is the ONLY pattern FlitMergeNode accepts.  (This split
    // was a loss in the pre-BAR_PAD-fix era only because the extra barrier cost ~10k; a
    // barrier is 3 cycles now.)
#ifndef FA_PIPE_NOPREPACK
    prepack_scales<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
        reinterpret_cast<__shared uint32_t*>(PACKED_SMEM), tid, thr);
    FAP_BAR(7);
    if (warp == 0)
        copy_scales_to_sfmem<FA_SQ, FA_SK>(
            reinterpret_cast<const __shared uint32_t*>(PACKED_SMEM),
            reinterpret_cast<volatile __shared uint32_t*>(GEMMINI_SF_MEM_A
                                                          + GEMMINI_SF_MEM_BUFFER_OFFSET), tid);
#else
    if (warp == 0)
        pack_scales_to_sfmem<FA_SQ, FA_SK>(
            reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
            reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A
                                                 + GEMMINI_SF_MEM_BUFFER_OFFSET), tid, thr);
#endif
    MARK();   // p5: pack done

    // ================= S5: [agent] PV matmul (A = requanted P @ spad0, B = V) ===============
    if (warp == 0) {
        fap_fence(tid);                              // drain the V mvin issued in S2
        fap_mm_issue<PVF>(PVP, SPAD_DEST, tid);
        fap_fence(tid);                              // drain matmul + O mvout
    }
    FAP_BAR(8);
    MARK();   // p6: PV done
    }   // end pipelined tile loop

    // EPILOGUE: the last tile's finalize (every other tile's ran inside the next tile's S0).
    finalize_O<FA_SQ, FA_D>(reinterpret_cast<const __shared uint32_t*>(S_SMEM),
                            reinterpret_cast<const __shared uint16_t*>(LS_SMEM),
                            reinterpret_cast<uint32_t*>(O_GMEM), tid, thr);
    MARK();   // p7: epilogue finalize
    }
#else   // !FA_PIPE  -- the original sequential FULL_ATTN2 body
#ifdef FA_STEADY
    // ================= STEADY-STATE UTILIZATION HARNESS (#ifdef FA_STEADY, OFF by default) =========
    // A single isolated tile charges the whole one-time cost (boot/entry, gemmini_flush, icache cold
    // miss, first-touch config) against ONE tile's 16,420 mesh-busy cycles, which UNDERSTATES the
    // utilization a real LLM attention block would see. Here the whole
    //     QK -> bar -> softmax -> PVFcfg -> requant -> pack -> PV -> finalize
    // sequence is run FA_NTILES times over the SAME input data (Q/K/V and their scales are re-read
    // from GMEM every iteration exactly as a real per-Q-block loop would re-read them). Only the LAST
    // iteration's O is meaningful for correctness -- every iteration recomputes the identical result,
    // so the standard fa_verify_out check still passes. The metric is the SLOPE:
    //     steady_per_tile = (T(4) - T(2)) / 2      (one-time boot/config cancels exactly)
    //     steady_util     = 16420 / steady_per_tile
    // A MARK is emitted at the top of every iteration so the per-iteration and per-phase deltas are
    // directly readable out of the MARK_GMEM stamp array.
    //
    // NTILES is selected by FA_NT<n> (the build harness only supports valueless -D's).
#  if   defined(FA_NT1)
#    define FA_NTILES 1
#  elif defined(FA_NT2)
#    define FA_NTILES 2
#  elif defined(FA_NT3)
#    define FA_NTILES 3
#  elif defined(FA_NT4)
#    define FA_NTILES 4
#  elif defined(FA_NT6)
#    define FA_NTILES 6
#  elif defined(FA_NT8)
#    define FA_NTILES 8
#  elif !defined(FA_NTILES)
#    define FA_NTILES 4
#  endif
    // Per-iteration state that must be re-established: NONE. online_softmax_block is called with
    // first=1 so m/l are re-initialized; S/PBF/O_acc/spad-A/spad-B/SF_MEM are all fully rewritten
    // before they are read. So the loop body is exactly the single-shot body, unmodified.
    for (uint32_t fa_tile = 0; fa_tile < (uint32_t)FA_NTILES; fa_tile++) {
    MARK();  // T: top of steady-state iteration (per-iteration cost = this stamp -> next iteration's)
#endif
    // QK_full
#ifdef FA_LANESC
    // Lane-parallel SF-scale writes (warp 0's 16 lanes, block-partitioned) instead of
    // thread-0-serial. mxgemm_core.hpp's load_scale_factors_lanes claims the B path accepts
    // this; whether the A (activation) scale merge node does is what this build measures.
    mxgemm_prefetch_tile<QKF, /*SKIP_A=*/false, /*DO_CONFIG=*/true, /*EXPLICIT_MVIN=*/false,
                         /*LANE_SCALES=*/true>(
        &QK_A_in[0][0], &QK_B_in[0][0], &QK_A_scales_row[0][0], &QK_B_scales_col[0][0],
        FA_SQ, FA_SK, FA_D, tid);
#else
    mxgemm_prefetch_tile<QKF, /*SKIP_A=*/false, /*DO_CONFIG=*/true, /*EXPLICIT_MVIN=*/false,
                         /*LANE_SCALES=*/false>(
        &QK_A_in[0][0], &QK_B_in[0][0], &QK_A_scales_row[0][0], &QK_B_scales_col[0][0],
        FA_SQ, FA_SK, FA_D, tid);
#endif
    MARK();  // 1
    mxgemm_compute_tile<QKF>(tid);
    MARK();  // 2: QK done
#ifdef FA_EARLYV
    // T1 (util): issue the V mvin NOW (QK is drained) so its ~67k DMA hides under bar2+softmax+requant+pack
    // (~42k) instead of only requant+pack (~31k). Safe: V's B-spad region is the same 2048 rows K^T used
    // (QKF PE_TILES_K*J = 8*16 == PVF 16*8) and K is already consumed; softmax reads S from SPAD_DEST, untouched.
    mxgemm_prefetch_tile<PVF, /*SKIP_A=*/true, /*DO_CONFIG=*/true, /*EXPLICIT_MVIN=*/true>(
        &V_in[0][0], &V_in[0][0], &V_scales[0][0], &V_scales[0][0], FA_SQ, FA_D, FA_SK, tid);
#endif
    mu_fence_smem(); BAR_PAD2(); mu_barrier(2, wpb); BAR_PAD2(); MARK();  // 3: bar2
    // softmax_full -> bf16 P @ PBF (row-major), l -> LS_SMEM
    online_softmax_block<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(S_SMEM),
        reinterpret_cast<__shared uint32_t*>(PBF),
        reinterpret_cast<__shared uint16_t*>(M_SMEM),
        reinterpret_cast<__shared uint16_t*>(LS_SMEM),
        reinterpret_cast<__shared uint16_t*>(CORR_SMEM),
        SOFTMAX_SCALE_BF16, /*first=*/1, tid, thr);
    mu_fence_smem(); MARK();  // 4: softmax(bf16 P) done
    // ===== PV: SKIP_A (P stays in spad0, no GMEM round-trip) + EXPLICIT_MVIN. H8 fix: the loop_ws FSM's
    // mvin with skip_lda=1 leaves a PHANTOM outstanding completion -> occupancy MMIO stuck non-zero ->
    // every gemmini_fence livelocks AND the PV matmul can't enqueue (reservation station looks full).
    // Explicit per-tile `gemmini_extended_mvin` commands for V bypass that skip accounting entirely. =====
#ifndef FA_EARLYV
#ifdef FA_SFPAR
    // ISOLATION PROBE for FA_PIPE's enabling trick: put the PV gemm's scales in scale-SRAM
    // HALF 1 instead of overwriting the QK gemm's half-0 scales, with the operand spads left
    // on the EVEN buffers.  Sequential order otherwise unchanged, so any Frobenius change is
    // attributable to the scale double-buffer alone.
    fap_cfg_mvin<PVF, /*SKIP_A=*/true>(&V_in[0][0], &V_in[0][0], FA_SQ, FA_D, FA_SK, 1u, tid);
    fap_scales_b<PVF>(&V_scales[0][0], FA_SQ, FA_D, 1u, tid);
    mu_fence_smem();
#elif defined(FA_LANESC)
    mxgemm_prefetch_tile<PVF, /*SKIP_A=*/true, /*DO_CONFIG=*/true, /*EXPLICIT_MVIN=*/false,
                         /*LANE_SCALES=*/true>(
        &V_in[0][0], &V_in[0][0], &V_scales[0][0], &V_scales[0][0],
        FA_SQ, FA_D, FA_SK, tid);
#else
    // -DFA_EXPMVIN (added by the host-offload campaign): keep the PV prefetch exactly WHERE it is
    // (after softmax, unlike FA_EARLYV which also hoists it above bar2) but issue V's move-in with
    // explicit gemmini_extended_mvin commands instead of the loop-FSM path.  FA_EARLYV changes both
    // things at once; this isolates the move-in mechanism, which is what the H8 note above says is
    // broken for SKIP_A (the loop FSM's mvin with skip_lda=1 leaves a phantom outstanding
    // completion).  Inert when the define is absent.
    mxgemm_prefetch_tile<PVF, /*SKIP_A=*/true, /*DO_CONFIG=*/true,
#ifdef FA_EXPMVIN
                         /*EXPLICIT_MVIN=*/true,
#else
                         /*EXPLICIT_MVIN=*/false,
#endif
                         /*LANE_SCALES=*/false>(
        &V_in[0][0], &V_in[0][0], &V_scales[0][0], &V_scales[0][0],
        FA_SQ, FA_D, FA_SK, tid);
#endif
#endif
    MARK();  // 5: PVF config + V mvin (explicit) issued; DMA overlaps the SIMT requant below
    // requant bf16 P -> fp8 tiled @ spad0 + per-row E8M0 scales -> SCALE_SMEM.
    requant_P_to_spad_tiled<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint16_t*>(PBF),
        reinterpret_cast<__shared uint32_t*>(0),
        reinterpret_cast<__shared uint32_t*>(SCALE_SMEM), tid, thr);
    mu_fence_smem(); MARK();  // 6: requant done
#ifdef FA_REQTEST
    // ISOLATION: dump fp8 P (spad0 tiled, 16KB=4096 words) + scales (SCALE_SMEM, 512 words) then STOP (no PVF).
    for (uint32_t i = tid; i < (FA_SQ*FA_SK)/4; i += thr)
        ((volatile uint32_t*)S_GMEM)[i] = ((const __shared uint32_t*)0)[i];
    for (uint32_t i = tid; i < (FA_SK/32)*FA_SQ; i += thr)
        ((volatile uint32_t*)PS_GMEM)[i] = ((const __shared uint32_t*)SCALE_SMEM)[i];
    mu_fence_smem(); mu_barrier(3, wpb); MARK();  // 7: reqtest dump done
    return;
#endif
    // pack per-row E8M0 scales -> SF_MEM_A (single-warp, program-order: the SF interface corrupts under
    // multi-warp parallel writes). V-scales already went to SF_MEM_B via the prefetch's load_scale_factors.
    // NOTE (2026-07-25, MEASURED): splitting this into an all-threads prepack + thread-0 ascending copy
    // made it WORSE (18.9k -> 26.0k) and broke O (Frob 51%): the required extra mu_barrier costs ~10k here,
    // which EXCEEDS the packing savings, and prepack additionally needs a barrier after requant. Reverted.
    // >>> GENERAL RULE for this machine: a barrier is ~10k cycles, so parallelizing any serial phase that is
    // itself <~10k (or that needs one more barrier) is a NET LOSS. This is why most SIMT-parallelization
    // attempts here lost. <<<
    // -DFA_NOPACK (HAZARD TEST ONLY, added by the host-offload campaign): skip the GPU's ONLY
    // write into the gemmini scale SRAM.  The output is garbage (PV runs on stale A scales); the
    // question the build answers is whether the host's per-tile 8-byte SF refill still trips
    // FlitMergeNode / the extReqXbar D-size monitor once no 4-byte GPU SF write exists to share
    // the merge node's state with.  Inert when the define is absent.
#ifndef FA_NOPACK
    pack_scales_to_sfmem<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
        reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A
#ifdef FA_SFPAR
                                             + GEMMINI_SF_MEM_BUFFER_OFFSET
#endif
                                             ), tid, thr);
#endif
    mu_fence_smem(); BAR_PAD3(); mu_barrier(3, wpb); BAR_PAD3(); MARK();  // 6: pack+bar3
#ifdef FA_DUMP2
    // dump SCALE_SMEM (normal SMEM, SIMT-readable -- unlike SF_MEM_A) = requant se's, 512 words (1 scale/word).
    // multi-thread (parses via fa_verify_out) -> PS_GMEM. Also fp8 P spad0[4096w] -> S_GMEM.
    for (uint32_t i = tid; i < (FA_SK/32)*FA_SQ; i += thr)
        ((volatile uint32_t*)PS_GMEM)[i] = ((const __shared uint32_t*)SCALE_SMEM)[i];
    for (uint32_t i = tid; i < (FA_SQ*FA_SK)/4; i += thr)
        ((volatile uint32_t*)S_GMEM)[i] = ((const __shared uint32_t*)0)[i];
    mu_fence_smem();
#endif
    // PV_full compute: A=P@spad0 (SIMT requant), B=V (explicit mvin), scales from SF_MEM. Normal BUSY fences.
    MARK();  // 7
#ifdef FA_SFPAR
    mxgemm_compute_tile<PVF>(tid, SPAD_DEST, /*a_spad_override=*/FA_A_SPAD_EVEN,
                             /*b_spad_override=*/FA_B_SPAD_EVEN, /*tile_k=*/1);
#else
    mxgemm_compute_tile<PVF>(tid);
#endif
    mu_fence_smem(); BAR_PAD4(); mu_barrier(4, wpb); BAR_PAD4(); MARK();  // 8: PV done
    // finalize directly from SPAD_DEST (O_unnorm). SINGLE finalize (common finalize guarded off for FULL_ATTN2)
    // => no double-write, no OACC copy (saved ~16k). 
    finalize_O<FA_SQ, FA_D>(
        reinterpret_cast<const __shared uint32_t*>(S_SMEM),
        reinterpret_cast<const __shared uint16_t*>(LS_SMEM),
        reinterpret_cast<uint32_t*>(O_GMEM), tid, thr);
    MARK();  // 9: finalize
#ifdef FA_STEADY
    }   // end steady-state tile loop
#endif
#endif  // FA_PIPE
    }
#elif defined(WARPSPEC)
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
#elif defined(WS_GMEM)
    // ===== GMEM-mvout overlap: the async QK_{j+1} routes its S output ACCUMULATOR -> GMEM (mxgemm_compute_
    // issue_gmem), so NO SMEM mvout runs during softmax_j -> the 16-subbank atomic-grant race (mvout vs SIMT)
    // is structurally impossible. softmax then SIMT-copies S_{j+1} GMEM->SBUF. QK mvin lands in bank0/3
    // (disjoint from softmax's bank1 SBUF/P_STAGE) and is drained pre-softmax by compute_issue_gmem's leading
    // fence, so during softmax the mesh touches ZERO SMEM. PV keeps its SMEM mvout (drained pre-rescale, no
    // concurrent SIMT -> safe). S_{j+1} uses a UNIQUE GMEM address per block (no stale-L0d reuse). =====
    {
    constexpr uint32_t P_STAGE = 0x8000;               // P (fp8) scratch; copied to row0 pre-PV
    constexpr uint32_t SBUF    = 0xA000;               // S double-buffer (bf16); softmax reads here
    constexpr uint32_t PW = (FA_SQ * FA_BK) / 4;       // P fp8 words (4 B/word)
    constexpr uint32_t SW = (FA_SQ * FA_BK) / 2;       // S bf16 words (2 elems/word)
    constexpr uint32_t SBYTES = FA_SQ * FA_BK * 2;     // S bf16 bytes per block (unique GMEM slot spacing)
    // prologue: QK_0 -> S_0@SPAD_DEST (SMEM mvout, drained pre-softmax = safe) -> copy to SBUF
    MARK();  // 1
    mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(&QK_A_in[0][0],
        &QK_B_blocks[0][0], &QK_A_scales_row[0][0], &QK_B_scales_blocks[0][0], FA_SQ, FA_BK, FA_D, tid);
    MARK();  // 2
    mxgemm_compute_tile<QK>(tid);
    MARK();  // 3
    mu_fence_smem(); BAR_PAD(); mu_barrier(2, wpb); BAR_PAD(); MARK();  // 4
    copy_smem_u32(reinterpret_cast<__shared uint32_t*>(SBUF),
                  reinterpret_cast<const __shared uint32_t*>(S_SMEM), SW, tid, thr);
    MARK();  // 5
    mu_fence_smem(); BAR_PAD(); mu_barrier(3, wpb); BAR_PAD(); MARK();  // 6
    for (uint32_t j = 0; j < FA_NBLK; j++) {
        BAR_PAD();
        const uint32_t first = (j == 0);
        // reload Q + issue QK_{j+1} -> S_{j+1}@GMEM (accmem->GMEM, NO SMEM mvout); overlaps softmax_j
        if (j + 1 < FA_NBLK) {
            mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(&QK_A_in[0][0],
                &QK_B_blocks[(j + 1) * FA_D][0], &QK_A_scales_row[0][0],
                &QK_B_scales_blocks[(j + 1) * FA_GK][0], FA_SQ, FA_BK, FA_D, tid);
            mxgemm_compute_issue_gmem<QK>(tid,
                reinterpret_cast<uint8_t*>(S_GMEM + (j + 1) * SBYTES), FA_BK);
        }
        // softmax_j: read SBUF (S_j), write P_j -> P_STAGE
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
        mu_fence_smem(); mu_barrier(4, wpb); MARK();  // 7
        BAR_PAD();
        // drain QK_{j+1} (matmul + accmem->GMEM DMA), then SIMT-copy S_{j+1} GMEM->SBUF
        if (j + 1 < FA_NBLK) {
            if (tid == 0) gemmini_fence();
            mu_fence_smem(); mu_barrier(5, wpb);
            copy_gmem_to_smem_u32(reinterpret_cast<__shared uint32_t*>(SBUF),
                reinterpret_cast<const volatile uint32_t*>(S_GMEM + (j + 1) * SBYTES), SW, tid, thr);
            mu_fence_smem(); mu_barrier(6, wpb);
        }
        MARK();  // 8
        // copy P_j: P_STAGE -> row0 (A_EVEN); clobbers Q (QK_{j+1} already read+drained it)
        copy_smem_u32(reinterpret_cast<__shared uint32_t*>(0),
                      reinterpret_cast<const __shared uint32_t*>(P_STAGE), PW, tid, thr);
        mu_fence_smem(); BAR_PAD(); mu_barrier(3, wpb); MARK();  // 9
        BAR_PAD();
        // PV_j: reads P@row0 (A_EVEN), V@B_EVEN -> O_j@SPAD_DEST (SMEM mvout, drained by compute_tile)
        mxgemm_prefetch_tile<PV, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(&V_in[j * FA_BK][0],
            &V_in[j * FA_BK][0], &V_scales[j * FA_GKB][0], &V_scales[j * FA_GKB][0],
            FA_SQ, FA_D, FA_BK, tid);
        mxgemm_compute_tile<PV>(tid);
        mu_barrier(1, wpb); MARK();  // 10
        rescale_accumulate<FA_SQ, FA_D>(
            reinterpret_cast<__shared uint32_t*>(OACC_SMEM),
            reinterpret_cast<const __shared uint32_t*>(S_SMEM),
            reinterpret_cast<const __shared uint16_t*>(CORR_SMEM), first, tid, thr);
        mu_fence_smem(); mu_barrier(2, wpb); MARK();
    }
    }
#elif defined(WS_ACCSMEM)
    // ===== Accumulator-SMEM overlap: QK_{j+1} matmul runs COMPUTE-ONLY into the accumulator during softmax_j
    // (acc_move_out=false -> off-SMEM, no mvout race), then the accmem->SPAD_DEST store (mxgemm_store_acc_to_spad,
    // 512b spad_writer) is issued POST-softmax in a SIMT-quiet window (all consumers parked at a barrier -> no
    // muon bank0 writers -> the 16-subbank atomic grant settles, no hang). Then SIMT-copy S SPAD_DEST->SBUF
    // (cheap SMEM->SMEM, NO GMEM round-trip). This is the cheap fix for WS_GMEM's 44k-cyc S transfer. =====
    {
    constexpr uint32_t P_STAGE = 0x8000;
    constexpr uint32_t SBUF    = 0xA000;
    constexpr uint32_t PW = (FA_SQ * FA_BK) / 4;
    constexpr uint32_t SW = (FA_SQ * FA_BK) / 2;
    MARK();  // 1
    mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(&QK_A_in[0][0],
        &QK_B_blocks[0][0], &QK_A_scales_row[0][0], &QK_B_scales_blocks[0][0], FA_SQ, FA_BK, FA_D, tid);
    MARK();  // 2
    mxgemm_compute_tile<QK>(tid);
    MARK();  // 3
    mu_fence_smem(); BAR_PAD(); mu_barrier(2, wpb); BAR_PAD(); MARK();  // 4
    copy_smem_u32(reinterpret_cast<__shared uint32_t*>(SBUF),
                  reinterpret_cast<const __shared uint32_t*>(S_SMEM), SW, tid, thr);
    MARK();  // 5
    mu_fence_smem(); BAR_PAD(); mu_barrier(3, wpb); BAR_PAD(); MARK();  // 6
    for (uint32_t j = 0; j < FA_NBLK; j++) {
        BAR_PAD();
        const uint32_t first = (j == 0);
        // reload Q + issue QK_{j+1} COMPUTE-ONLY into accumulator (no mvout); overlaps softmax_j
        if (j + 1 < FA_NBLK) {
            mxgemm_prefetch_tile<QK, /*SKIP_A=*/false, /*DO_CONFIG=*/true>(&QK_A_in[0][0],
                &QK_B_blocks[(j + 1) * FA_D][0], &QK_A_scales_row[0][0],
                &QK_B_scales_blocks[(j + 1) * FA_GK][0], FA_SQ, FA_BK, FA_D, tid);
            mxgemm_compute_issue_acc<QK>(tid);
        }
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
        mu_fence_smem(); mu_barrier(4, wpb); MARK();  // 7
        BAR_PAD();
        // SIMT-quiet window: store QK_{j+1} accmem->SPAD_DEST (mvout), then copy S SPAD_DEST->SBUF
        if (j + 1 < FA_NBLK) {
            if (tid == 0) mxgemm_store_acc_to_spad<QK>(tid, SPAD_DEST);  // fence inside
            mu_fence_smem(); mu_barrier(5, wpb);
            copy_smem_u32(reinterpret_cast<__shared uint32_t*>(SBUF),
                          reinterpret_cast<const __shared uint32_t*>(S_SMEM), SW, tid, thr);
            mu_fence_smem(); mu_barrier(6, wpb);
        }
        MARK();  // 8
        copy_smem_u32(reinterpret_cast<__shared uint32_t*>(0),
                      reinterpret_cast<const __shared uint32_t*>(P_STAGE), PW, tid, thr);
        mu_fence_smem(); BAR_PAD(); mu_barrier(3, wpb); MARK();  // 9
        BAR_PAD();
        mxgemm_prefetch_tile<PV, /*SKIP_A=*/true, /*DO_CONFIG=*/true>(&V_in[j * FA_BK][0],
            &V_in[j * FA_BK][0], &V_scales[j * FA_GKB][0], &V_scales[j * FA_GKB][0],
            FA_SQ, FA_D, FA_BK, tid);
        mxgemm_compute_tile<PV>(tid);
        mu_barrier(1, wpb); MARK();  // 10
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
    // FULL_ATTN2/QKF_ONLY do their own finalize (or none) -> skip the common one (avoids double-write).
#if !defined(FULL_ATTN2) && !defined(FULL_ATTN3) && !defined(QKF_ONLY)
    finalize_O<FA_SQ, FA_D>(
        reinterpret_cast<const __shared uint32_t *>(OACC_SMEM),
        reinterpret_cast<const __shared uint16_t *>(LS_SMEM),
        reinterpret_cast<uint32_t *>(O_GMEM), tid, thr);
    MARK();  // final
#endif
}

int main() {
#if defined(FA_OCC1)
    mu_schedule(fa_entry, nullptr, 1);  // occ=1: match the working requant.cpp microbench
#elif defined(FA_OCC2)
    mu_schedule(fa_entry, nullptr, 2);  // occ=2
#else
    mu_schedule(fa_entry, nullptr, 3);  // occ=3 default
#endif
                                        // occ=4 overflowed the 256 phys-reg file, occ=3 (3*57=171)
                                        // has margin. More warps hide the latency-bound SIMT softmax.
    return 0;
}
