#include <stdint.h>
#include <radiance.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
#include "mxgemmini_mmio.h"

// ============================================================================================
// FA_NS_FENCE / FA_NS_OCC -- FENCE-TOO-EARLY GUARDS APPLIED TO *EVERY* GEMMINI FENCE.
//
// gemmini_fence() (mxgemmini_mmio.h) is `while (load32_shared(GEMMINI_BUSY_ADDR) != 0) nop;`.
// It drains only what is ALREADY VISIBLE AS BUSY.  Measured from a full [ISSUE] trace of
// FULL_ATTN2 + FA_NOSCALES + FA_EARLYV at FA_NT6 (/tmp/hsruns/nt6ver.out), counting the BUSY
// loads that each fence actually performs:
//     mxgemm_prefetch_tile   configure_mxgemmini's fence   1 poll   in EVERY tile  (drains nothing)
//     mxgemm_compute_tile    leading fence (QK)            1..120 polls, and 1 (= nothing) in
//                                                          cl0 tile 2 and cl1 tile 5
//     mxgemm_compute_tile    leading fence (PV)            1 poll   in EVERY tile (drains nothing)
//     mxgemm_compute_tile    trailing fences               234..245 polls (these do drain)
// So the leading drains routinely return having waited for nothing, which is only safe if the
// gemmini's own reservation station orders the pending move-in ahead of the matmul.
//   FA_NS_FENCE: bounded wait for busy to ASSERT, then the normal drain -- at EVERY fence, not
//                just the one in mxgemm_compute_tile that FA_NS_SETTLE guards.
//   FA_NS_OCC:   drain on the OCCUPANCY counter instead (incremented when the command is
//                accepted, so there is no assert-latency race at all).  H8 warns the occupancy
//                MMIO can be phantom-poisoned on the SKIP_A path, hence the separate flag.
// ============================================================================================
// ============================================================================================
// FA_CFGSETTLE -- *** CORRECTNESS FIX: CONFIG_SCALE_MEM MUST BE APPLIED BEFORE THE MATMUL. ***
//
// ScaleFactorMem.scala computes the MX scale-SRAM READ ROW from the loop bounds that
// CONFIG_SCALE_MEM (funct 26, gemmini_mxquant_config_mvout) latches:
//     read_row_addr_act := loop_bound_i * (counter_k_runtime >> 1) + counter_i_runtime
//     read_row_addr_w   := loop_bound_j * (counter_k_runtime >> 1) + counter_j_runtime
// and the half of the double buffer from the same register's scale_mem_read_act_sel /
// _w_sel.  In this kernel the two gemms differ in loop_bound_j (QKF: N=Sk=256 -> 16;
// PVF: N=d=128 -> 8) but NOT in loop_bound_i (both M=Sq=64 -> 4).  So if a matmul starts
// while the register still holds the PREVIOUS gemm's value, the WEIGHT (B) scale rows are
// aliased and the ACT (A) scale rows are not -- a B-side-only corruption.
//
// THAT IS EXACTLY WHAT THE FAILING BUILD MEASURES.  Recovering S from the softmax's bf16 P
// and projecting onto the row space of the correct effective K matrix leaves a 22.0% residual
// (floor 2.3%); with the hw mesh model -- which reproduces golden_S BIT-EXACTLY -- an
// arbitrarily corrupted A operand or A scale set leaves the residual AT the 2.3% floor
// (mathematically: S = A''@B' for any A''), while corrupting the B scales pushes it to
// 25-63%.  |S| is also preserved to 0.08%, which is what aliasing scale bytes that only
// span 127..129 does and what substituting the act half does NOT (|S| drops 2.7x).
//
// WHY THE BASELINE IS SAFE AND BOTH OPTIMISATIONS ARE NOT -- AND WHY THE TWO FAILURES ARE ONE BUG.
// mxgemm_compute_tile issues CONFIG_SCALE_MEM and then matmul_tile_async with NOTHING in
// between, and the two take different paths inside the gemmini (CONFIG_SCALE_MEM is latched by
// ExecuteController's decode, LOOP_WS is expanded by the LoopMatmul FSM), so program order at the
// MMIO port does not order the register update against the mesh work that reads it.  The
// unmodified kernel gets away with it because mxgemm_prefetch_tile's configure_mxgemmini already
// issued the SAME CONFIG_SCALE_MEM for this gemm, and then spent ~12.6k cycles in
// load_scale_factors -- so by matmul time the register ALREADY holds the right value and a late
// second write is harmless.  BOTH optimisations delete exactly that slack:
//     FULL_ATTN2 + FA_NOSCALES : load_scale_factors is gone, so the prefetch's CONFIG_SCALE_MEM
//                                is only a few hundred cycles ahead of the matmul;
//     FA_SP + FA_SP_LEANCFG    : configure_mxgemmini is compiled out entirely, so the ONLY
//                                CONFIG_SCALE_MEM is the one inside fa_mm(), zero instructions
//                                before the matmul.
// Both therefore run a matmul whose scale addressing may still be the PREVIOUS gemm's.  Measured
// on the two failing traces (/tmp/hsruns/nt6ver.out and the FA_SP one), the signature is
// IDENTICAL and confined to ONE cluster: the bf16 P the softmax writes is bit-identical for tiles
// 0..2 and then ALL 8192 words change at tile 3 and stay changed, with every downstream stage
// (requant, pack, PV, finalize) bit-exact given S.  Measured from the trace: every "leading"
// gemmini_fence in these builds returns after ONE BUSY poll, i.e. drains nothing, so nothing else
// separates the config from the matmul.
//
// WHY IT NEVER RECOVERS.  ScaleFactorMem's counter_i/j/k_runtime are RegInits that advance per
// 16x16 tile-step and re-zero ONLY by completing a full (i,j,k) sweep of the CURRENT bounds --
// there is no reset path (gemmini_mxquant_config_mvout never sets rs1[62], and that bit resets
// the REQUANTIZER's write counter, MxRequantizer.scala:558, not these).  So a bound change that
// lands mid-sweep strands them permanently, which is exactly "correct until tile 3, wrong for
// every tile after".
//
// THE FIX is 16 MMIO round trips (~600 cyc) between the config and the matmul issue (use
// FA_CFGSETTLE2 -- the serialized form; see the note in fa_cfg_settle()).  Cost ~1.2k cyc/tile
// out of ~69k (1.7%).
//
// WHAT IS PROVEN AND WHAT IS INFERRED, so nobody has to re-derive it:
//   PROVEN from the traces: (a) S itself is wrong from the onset tile, and every downstream stage
//     is bit-exact given S; (b) the two failing configs produce the SAME wrong S -- 85-89% of the
//     16384 bf16 P halfwords are IDENTICAL between them at the same tile index, with row-space
//     residual 22.0% vs 21.5% and |S| 1298.0 vs 1298.0 -- so it is ONE bug, not two; (c) it is
//     confined to one cluster and never recovers; (d) a static A-operand or A-scale corruption is
//     mathematically incapable of producing the observed row-space residual, so the weight side of
//     the QK gemm is involved.
//   PROVEN from the RTL: the weight scale read row depends on loop_bound_j (16 for QKF, 8 for
//     PVF) while the act row depends on loop_bound_i (4 for both), and the counters that index
//     them have no reset path.
//   INFERRED: that the trigger is specifically the CONFIG_SCALE_MEM/LOOP_WS ordering.  No single
//     static aliasing model reproduces the observed S bit-for-bit (the best -- a one-k-tile
//     counter strand -- lands on 21.7% residual against the measured 22.0% but the wrong |S|),
//     which is expected if the bounds change PART-WAY through a sweep: the counters then
//     desynchronise progressively rather than by a fixed offset.  A time-varying operand
//     reproduces the residual magnitude too (18-19%), so an operand-move-in race cannot be fully
//     excluded -- which is why FA_NS_FENCE (drain harder everywhere) exists alongside this flag.
// ============================================================================================
// FA_CFGSETTLE_AFTER is the COST-MATCHED CONTROL: identical instruction count and
// identical gemmini-port traffic, moved to AFTER the matmul issue.  If "before" is
// correct and "after" is not, the ordering -- not the delay -- is what matters.
#if defined(FA_CFGSETTLE) || defined(FA_CFGSETTLE_AFTER) || defined(FA_CFGSETTLE_PRE) \
    || defined(FA_CFGSETTLE2) || defined(FA_CFGSETTLE_BIG)
static inline void fa_cfg_settle() {
#ifdef FA_CFGSETTLE_BIG
    // DIAGNOSTIC ONLY (~9.5k cyc per matmul): if even this does not close the window then the
    // CONFIG_SCALE_MEM-vs-matmul ordering is NOT the mechanism.
    { uint32_t zero = 0; asm volatile("" : "+r"(zero));
      uint32_t v = 0;
      for (uint32_t _i = 0; _i < 256u; _i++) v = load32_shared(GEMMINI_BUSY_ADDR + (v & zero));
      asm volatile("" :: "r"(v)); }
#elif defined(FA_CFGSETTLE2)
    // SERIALIZED form.  `+ (v & 0u)` is NOT enough: clang folds it away and emits 16 INDEPENDENT
    // `lw.shared a6, 0x0(a5)` (verified in the objdump), which the LSU can pipeline, and since the
    // loaded value is never used the warp never stalls -- the "settle" then costs ~16 issue slots
    // and guarantees nothing.  Launder the address through an empty asm that also takes the
    // previous result, so each load genuinely waits for its predecessor: 16 real MMIO round trips
    // (~37 cyc each, measured from the gemmini_fence spin rate) = ~590 cyc.
    uint32_t zero = 0;
    asm volatile("" : "+r"(zero));          // an OPAQUE zero: clang cannot fold `v & zero`
    uint32_t v = 0;
    for (uint32_t _i = 0; _i < 16u; _i++) v = load32_shared(GEMMINI_BUSY_ADDR + (v & zero));
    asm volatile("" :: "r"(v));
#else
    uint32_t v = 0;
    for (uint32_t _i = 0; _i < 16u; _i++) v = load32_shared(GEMMINI_BUSY_ADDR + (v & 0u));
    asm volatile("" :: "r"(v));
#endif
}
#else
static inline void fa_cfg_settle() {}
#endif

// ============================================================================================
// FA_NS_SENT -- POSITIVE COMPLETION PROOF FOR THE A-OPERAND MOVE-IN.
//
// Every drain this kernel has is a poll of the gemmini's `busy` bit, and `busy` is measurably
// unreliable here: in the failing builds mxgemm_compute_tile's LEADING fence returns after a
// single BUSY read in some tiles, and neither waiting for busy to assert first (FA_NS_FENCE),
// nor draining on OCCUPANCY instead (FA_NS_OCC), nor inserting 0.6k/9.5k cycles of MMIO round
// trips (FA_CFGSETTLE/FA_CFGSETTLE_BIG) removes the corruption -- each of them only MOVES the
// tile at which it starts.  So instead of trusting the gemmini's status bits, prove the move-in
// landed by reading the destination: the operand scratchpad IS cluster SMEM, so stamp a sentinel
// into the last word of every 16x16 destination tile before issuing the DMA and spin until the
// DMA has overwritten all of them.  That is a positive, timing-independent proof for the A
// operand (Q), whose stale content -- the previous tile's requantized P8 -- is the only candidate
// carried state that can explain the observed "the wrong S changes once and then sticks".
// A `while` loop, NOT a bounded `for (..) if (..) break;` -- see the FA_NS_SETTLE note below.
// ============================================================================================
// ============================================================================================
// FA_PHASE<k> -- MEASUREMENT HARNESS: SWEEP THE INTER-CLUSTER PHASE.
//
// The ~110-113% corruption is a race whose outcome is decided by which of the TWO clusters loses
// a contest for a shared resource: across every failing run seen so far EXACTLY ONE cluster is
// hit, never both and never neither, which for independent per-cluster events would happen with
// probability <= 0.5 each time (7 for 7 => <= 0.8%).  It is therefore anti-correlated, i.e. the
// clusters compete, and the lottery variable is their RELATIVE PHASE.  Perturbing the code
// resamples that lottery, which is why 14 of 19 arbitrary "fix" variants score 12/12 and 5 do
// not, with no correlation to which flag was set -- A SINGLE 12/12 RUN IS ~74% LIKELY BY CHANCE
// AND IS NOT EVIDENCE OF A FIX.
//
// FA_PHASE<k> delays cluster 1 (and only cluster 1) by k * 64 dependent MMIO round trips
// (~2.4k cycles each) at the top of every tile, so the phase can be swept deliberately instead of
// resampled by accident.  A real fix must be correct at EVERY k; a lucky one will not be.
//
// *** 2026-07-30: THIS HARNESS HAS NOW ACTUALLY BEEN RUN -- IT HAD NEVER BEEN, IN ANY BUILD IN
// /tmp/yruns -- AND ON ITS FIRST USE IT KILLED THE CONFIGURATION IT WAS POINTED AT. ***
//     yph2 = FULL_ATTN2 FA_SP QOVL LEANCFG QKACC PKOVL QSPLIT WCNT PAX CVTX FA_SM_2P FA_SM_2PRAW
//            FA_SP_ACCPAD (N=128)  +  FA_NT6  +  FA_PHASE2
//     cluster 0 (UNDELAYED):  6 of 6 tiles CORRECT at 3.5666%
//     cluster 1 (DELAYED):    tile 0 CORRECT, then 108.5643%  113.0664%  118.1504%  119.3714%
//                             119.3714%   -- latching, worsening, never recovering
//     SUMMARY 7 correct, 5 wrong of 12; all images 4096/4096 words, so not a truncated trace.
// The SAME binary WITHOUT FA_PHASE2 (P128, and an independent rebuild yb6 whose GPU image is
// byte-identical, rv32 sha 5028374f1b66d3ae) scores 12 of 12.  FA_PHASE is computationally inert by
// construction AND by RTL: it only issues dependent READ-ONLY loads of GEMMINI_BUSY_ADDR, whose MMIO
// field is RegField.r with an unconditionally-valid read and no side effect
// (GemminiTile.scala:428 with :404-405), and it discards the value.  It cannot change a computed
// result.  *** THEREFORE THE 12-of-12 IS THE ARTEFACT AND THE 5-WRONG IS THE TRUTH: that
// configuration is INCORRECT, and its NT6 pass was the coincidence this comment block predicted. ***
// TWO CONSEQUENCES, both worth more than the flag verdicts they invalidate:
//   * THE ONE-CLUSTER OBSERVATION IS THE MECHANISM, NOT A CURIOSITY.  Here the delayed cluster is
//     the one that fails and the undelayed one is clean, on the same instructions and the same data
//     in the same run.  (The "never both" part of the claim above is however FALSE as stated -- of 13
//     scored failing runs on disk, 10 hit exactly one cluster but P2K, hsfLD and hsfRD4 hit both.)
//   * ANY "X IS CORRECT" IN THIS TREE THAT RESTS ON A SINGLE UNPERTURBED NT6 OR NT8 RUN IS
//     UNSUPPORTED, INCLUDING THE GIT-TAGGED fa-mx-best-36.02.  Pass FA_PHASE1/2/3 before believing
//     any of them.  A single FAILURE at any k is conclusive; a single PASS at one k is not.
// ============================================================================================
#if defined(FA_PHASE1) || defined(FA_PHASE2) || defined(FA_PHASE3) || defined(FA_PHASE4) \
    || defined(FA_PHASE5)
#if   defined(FA_PHASE1)
#define FA_PHASE_N 1
#elif defined(FA_PHASE2)
#define FA_PHASE_N 2
#elif defined(FA_PHASE3)
#define FA_PHASE_N 3
#elif defined(FA_PHASE4)
#define FA_PHASE_N 4
#else
#define FA_PHASE_N 5
#endif
static inline void fa_phase_skew_impl(uint32_t tid) {
    if (tid != 0) return;
    uint32_t cl; asm volatile("csrr %0, 0xCD0" : "=r"(cl));
    // FA_PHASE_BOTH -- *** THE CONTROL THAT DECIDES WHAT AN FA_PHASE FAILURE MEANS. ***
    // FA_PHASE<k> delays cluster 1 only, so a failure confined to cluster 1 has TWO explanations:
    // (a) the RELATIVE PHASE moved and exposed a pre-existing race (the intended reading), or
    // (b) executing this skew loop at all is what damages the cluster that runs it -- it is the one
    //     place in the kernel where lane 0 of every warp runs a divergent MMIO-read loop with no
    //     enclosing vx_split, and cluster 0 never executes it, so the two clusters are NOT running
    //     the same instructions and the usual "same code, same data" argument does not apply.
    // With FA_PHASE_BOTH both clusters run the IDENTICAL loop, so the relative phase is restored to
    // ~0 while the executed code stays exactly as it is under FA_PHASE<k>.
    //     BOTH clusters correct  => the loop is innocent, (a) holds, the race is real and relative
    //                               phase is its trigger;
    //     the failure persists   => (b) holds, the harness is the bug, and every FA_PHASE verdict
    //                               including my own has to be thrown away.
    //
    // *** MEASURED -- AND THE CRITERION I JUST WROTE IS TOO CRUDE TO DECIDE IT.  I AM OVERRIDING MY
    // OWN PRE-REGISTERED RULE, AND THE REASON HAS TO BE ON THE RECORD RATHER THAN QUIETLY DROPPED. ***
    // On the FA_SP_ACCPAD(N=128) base at FA_NT6:
    //     no skew          (P128)  12 of 12
    //     FA_PHASE1        (yph1)   7 of 12   cluster 1 (delayed) tiles 1-5 WRONG, cluster 0 clean
    //     FA_PHASE2        (yph2)   7 of 12   cluster 1 (delayed) tiles 1-5 WRONG, cluster 0 clean
    //     FA_PHASE1 + BOTH (yphb1) 10 of 12   cluster 1 tiles 4,5 WRONG  (102.1825%, 132.3960%)
    //     FA_PHASE2 + BOTH (yphb2) 10 of 12   cluster 0 tiles 4,5 WRONG  ( 82.0181%, 114.5661%)
    // The failure PERSISTS under symmetric delay, so the letter of the rule says "harness is the bug".
    // The RULE is what is wrong: it conflated "the failure persists" with "running this loop corrupts
    // the cluster that runs it".  If the loop itself were the corruption then with BOTH clusters
    // running it BOTH clusters would be corrupt.  Instead exactly ONE cluster fails per run, only two
    // tiles, and *** WHICH cluster fails FLIPS between k=1 (cluster 1) and k=2 (cluster 0) *** under
    // code that is now symmetric.  Symmetric code with an asymmetric, k-dependent outcome is the
    // signature of a race resolved by residual timing asymmetry (DRAM, boot skew, arbitration), not of
    // deterministic per-cluster corruption by the loop.
    // AND THE HARNESS IS EXONERATED INDEPENDENTLY, WHICH IS WHAT ACTUALLY SETTLES IT: FA_SP_ACCPAD's
    // OWN LENGTH SWEEP IS A SECOND INERT PERTURBATION WITH NO MMIO AND NO ALL-WARP DIVERGENCE -- pure
    // register ALU ops on warp 0 only -- AND IT BREAKS THE KERNEL TOO: N=2048 scores 7 of 11 and
    // N=8192 scores 6 of 10, where N=128 scores 12 of 12.  Two structurally unrelated computationally
    // inert perturbations both break it, so the race is real and neither harness causes it.
    // WHAT *IS* REFUTED IS THE FRAMING AT THE TOP OF THIS BLOCK: relative INTER-CLUSTER phase is NOT
    // the trigger, because a symmetric delay that restores relative phase to ~0 still fails.  Asymmetry
    // makes it WORSE (5 wrong vs 2; onset at tile 1 vs tile 4) but is not necessary.  *** ANYONE
    // HUNTING A SHARED CONTENDED STRUCTURE SHOULD KNOW THAT A TWO-CLUSTER CONTEST IS NOT REQUIRED TO
    // EXPLAIN THIS BUG: perturbing the timing of a single cluster is sufficient to produce it. ***
#ifndef FA_PHASE_BOTH
    if (cl == 0) return;
#else
    (void)cl;
#endif
    uint32_t z = 0; asm volatile("" : "+r"(z));
    uint32_t v = 0;
    for (uint32_t i = 0; i < 64u * FA_PHASE_N; i++)
        v = load32_shared(GEMMINI_BUSY_ADDR + (v & z));
    asm volatile("" :: "r"(v));
}
#define fa_phase_skew(t) fa_phase_skew_impl(t)
#else
#define fa_phase_skew(t) do { } while (0)      // literally nothing at the call site
#endif

#if defined(FA_NS_SENT) || defined(FA_NS_SENTB)
#define FA_SENT_MAGIC 0xA5A5F00Du
static inline void fa_sent_stamp(uint32_t spad_row_start, uint32_t ntiles) {
    volatile __shared uint32_t *sp = reinterpret_cast<volatile __shared uint32_t *>(0);
    for (uint32_t n = 0; n < ntiles; n++)
        sp[((spad_row_start + n * 16u) * 16u + 252u) >> 2] = FA_SENT_MAGIC;
    mu_fence_smem();
}
static inline void fa_sent_wait(uint32_t spad_row_start, uint32_t ntiles) {
    volatile __shared uint32_t *sp = reinterpret_cast<volatile __shared uint32_t *>(0);
    for (uint32_t n = 0; n < ntiles; n++) {
        const uint32_t w = ((spad_row_start + n * 16u) * 16u + 252u) >> 2;
        while (sp[w] == FA_SENT_MAGIC) { asm volatile("nop"); }
    }
}
#endif

#if defined(FA_NS_FENCE) || defined(FA_NS_OCC)
static inline void fa_gemmini_fence_safe() {
#ifdef FA_NS_OCC
    while (load32_shared(GEMMINI_OCCUPANCY_ADDR) != 0) asm volatile("nop");
#else
    // *** NOT a bounded `for (..) if (busy) break;` ***  That shape -- which is what the
    // pre-existing FA_NS_SETTLE uses -- overflows the warp scheduler's IPDOM stack here
    // ("Assertion failed: ipdom stack is full", WarpScheduler.scala:456, $finish at ~112k
    // cycles), so FA_NS_SETTLE could never have produced a result.  An UNCONDITIONAL
    // dependent chain of MMIO reads has no divergence region at all.
    { uint32_t _z = 0; asm volatile("" : "+r"(_z));   // opaque 0 -> a real dependent chain
      uint32_t _v = 0;
      for (uint32_t _i = 0; _i < 16u; _i++) _v = load32_shared(GEMMINI_BUSY_ADDR + (_v & _z));
      asm volatile("" :: "r"(_v)); }
#endif
    while (load32_shared(GEMMINI_BUSY_ADDR) != 0) asm volatile("nop");
}
#define gemmini_fence() fa_gemmini_fence_safe()
#endif

// Tiling parameters -----------------------------------------------------------

enum class GemmDatatype : uint8_t {
    FP8,
    FP6,
    FP4,
};

struct GemmConfig {
    uint32_t TILE_M = 128;
    uint32_t TILE_N = 128;
    uint32_t TILE_K = 256;
    GemmDatatype DATATYPE = GemmDatatype::FP8;
    // quantize output to fp4/fp6/fp8
    bool QUANT_OUTPUT = false;

    constexpr bool IS_FP8() const { return DATATYPE == GemmDatatype::FP8; }
    constexpr uint32_t PE_M() const { return (IS_FP8() ? 16 : 32); }
    constexpr uint32_t PE_N() const { return (IS_FP8() ? 16 : 32); }
    constexpr uint32_t PE_K() const { return 16; }
    constexpr uint32_t PE_TILES_I() const { return TILE_M / PE_M(); }
    constexpr uint32_t PE_TILES_J() const { return TILE_N / PE_N(); }
    constexpr uint32_t PE_TILES_K() const { return TILE_K / PE_K(); }
    // TODO: TILE_N not differentiated
    constexpr uint32_t SCALE_FACTORS_PER_TILE() const { return TILE_M * TILE_K / 32; }
    // B (K-operand) scales are counted along N, not M: needed for non-square tiles (e.g. full-attn QKF N=Sk).
    constexpr uint32_t SCALE_FACTORS_PER_TILE_B() const { return TILE_N * TILE_K / 32; }
    constexpr uint32_t VALUES_PER_BYTE() const { return (IS_FP8() ? 1 : 2); }
    // Size of each C element *after column-packing*.
    constexpr uint32_t OUT_ELEM_SIZE() const {
        // C FP4/FP6 elem-packing is along the M dimension, not N
        return (QUANT_OUTPUT ? sizeof(uint8_t) : sizeof(uint16_t));
    }
    constexpr uint32_t TILE_M_QUANT() const {
        return (QUANT_OUTPUT ? TILE_M / VALUES_PER_BYTE() : TILE_M);
    }
    constexpr uint32_t TILE_N_QUANT() const {
        // packing of N-dimension is already reflected in OUT_ELEM_SIZE()
        return TILE_N;
    }
    constexpr bool USE_LUT() const { return DATATYPE == GemmDatatype::FP6; }
};

// Gemmini constants -----------------------------------------------------------

constexpr auto GEMMINI_FORMAT_FP8 = 0;
constexpr auto GEMMINI_FORMAT_FP6 = 1;
constexpr auto GEMMINI_FORMAT_FP4 = 2;
constexpr auto GEMMINI_FORMAT_FULL = 3;
constexpr auto QUANT_LUT_UPDATE_GRANULARITY = 1;
constexpr auto GEMMINI_ACC_ADDR = (1u << (ADDR_LEN - 1));
constexpr auto SPAD_DEST = 1024; // C-output spad row: clears the A-spad P-fp8 region up to
                                 // 128x128 fp8 (1024 rows) so it works for FA tiles up to 128.

// Performance benchmark options -----------------------------------------------

// use MxGemmini DMA for GMEM->SMEM move-in
constexpr bool GEMMINI_DMA = true;

// PMARK: cycle stamps for intra-prefetch profiling -> GMEM 0x40060000 + 4*(8*call + slot).
// Only thread 0 stamps. PMARK_CALL is bumped by the caller between prefetch invocations.
#ifndef PMARK_ENABLE
#define PMARK(slot) do {} while (0)
#else
extern uint32_t g_pmark_call;
#define PMARK(slot) do { if (tid_in_threadblock == 0) { uint32_t _c; \
    asm volatile("csrr %0, mcycle" : "=r"(_c)); \
    ((volatile uint32_t*)0x40060000)[8*g_pmark_call + (slot)] = _c; } } while (0)
#endif
// ===== GMEM DIAGNOSTIC ADDRESS MAP + A TRACE-EXTRACTION TRAP ==================================
// Every number in this campaign is read back by parsing the cyclotron [ISSUE] trace, where a store
// is reported as `rs1.data=[<per-lane base register>]` -- the S-type IMMEDIATE IS NOT SHOWN.  For a
// store whose address is a compile-time constant, clang emits `lui rX, %hi(addr); sw rY, %lo(addr)(rX)`,
// so the traced rs1.data is the 4KB-PAGE BASE, not the address.  That silently aliased every
// diagnostic in this file onto the kernel's MARK array at MARK_GMEM+0 (measured: the FA_HOST_TIMING
// diag echo, 16 constant-address stores, overwrote m[0..15] -- i.e. the first 1.5 tiles of an
// FA_STEADY mark trace -- and FA_HOST_MARK overwrote m[0]).
// TWO RULES for anything added here:
//   1. put it on its OWN 4KB page (map below), and
//   2. make the base pointer OPAQUE to the compiler (`asm volatile("" : "+r"(p))`) so the store
//      issues with imm=0 and the traced rs1.data IS the effective address.
// Map:  0x40050000 kernel MARK array (kernel.cpp, runtime index -> already exact)
//       0x40051000 host<->GPU handshake spin stamps (FA_HOST_MARK)
//       0x40052000 host diagnostic block echo (FA_HOST_TIMING)
//       0x40053000 CPROF ring (FA_CFGPROF)
//       0x40060000 PMARK
#define FA_DIAG_HSMARK 0x40051000u
#define FA_DIAG_HOSTDG 0x40052000u
#define FA_DIAG_CPROF  0x40053000u

// CPROF (-DFA_CFGPROF): fine-grained cycle stamps INSIDE mxgemm_prefetch_tile / mxgemm_compute_tile.
// MARK() in the kernel only brackets whole phases; this splits the two biggest ones (QK prefetch
// 24.3k, PVF cfg 9.2k) into {ROCC config stores | DMA-issue stores | host-scale wait} and the
// matmuls into {leading drain | issue | trailing drain}, which is what decides whether host-side
// ROCC issue is worth anything.  Thread 0 only; 1 store per point.
// ONE PACKED WORD per point -- (id << 20) | (mcycle & 0xFFFFF) -- rather than an (id, cycle) pair,
// because a 2-word pair makes the two stores share a base register and become indistinguishable in
// the trace (see the aliasing note above).  20 bits of cycle = 1,048,575, enough for NT<=4.
#ifdef FA_CFGPROF
static uint32_t g_cprof = 0;
#define CPROF_MAX  400u
#define CPROF(id) do { if (tid_in_threadblock == 0 && g_cprof < CPROF_MAX) { uint32_t _c; \
    asm volatile("csrr %0, mcycle" : "=r"(_c)); \
    volatile uint32_t *_p = (volatile uint32_t *)FA_DIAG_CPROF; \
    asm volatile("" : "+r"(_p)); \
    _p[g_cprof++] = ((uint32_t)(id) << 20) | (_c & 0xFFFFFu); } } while (0)
#else
#define CPROF(id) do {} while (0)
#endif
// always read from the first-k tile position to incur high cache hits
constexpr bool ZERO_STRIDE_K = false;
// disable GMEM->SMEM DMA copy after 0th tile and have MxGemmini work on stale
// data in SMEM
constexpr bool DISABLE_MOVE_IN_AFTER_FIRST_K = false;
// disable scale-factor write & fence from the GPU
constexpr bool DISABLE_SCALE_FACTOR_UPDATE = false;
// disable C result tensor move-out from SMEM to GMEM
constexpr bool DISABLE_GMEM_MOVE_OUT = false;
// use SIMT load/stores instead of DMA for C DMA move-out
constexpr bool SIMT_GMEM_MOVE_OUT = true;

// TODO: max size hardcoded
static uint32_t C_scale_factors[128 * 128 / 32] __attribute__((aligned(32))) = {0};

#ifdef FA_NOSCALES
// ---- HOST-ASSISTED SCALE PREFILL (see host.cpp) --------------------------------------------
// With -DFA_NOSCALES the GPU does NOT load the MX scale factors at all: the rv64 Rocket host
// writes them straight into the gemmini scale SRAM with 8-byte stores (which bypass
// FlitMergeNode's 4B pair-merge FSM).  SF_MEM_B is needed TWICE per FA pass -- K scales for
// QK^T, then V scales for PV -- and both are prefilled up front, so they live in DIFFERENT
// halves of the weight-scale double buffer (K -> buffer 0, V -> buffer 1 at byte offset
// GEMMINI_SF_MEM_BUFFER_OFFSET).  The mesh picks the half via CONFIG_SCALE_MEM's scale_w_sel
// bit (rs1[61], ScaleFactorMem.scala:196/230-237), so it must be 1 for the PV matmul.
// The prefetch knows which gemm it is (SKIP_A == "A comes from the requantizer" == PV) and
// hands that to the compute through this variable; both run on the SAME thread (tid 0).
static uint32_t g_host_sf_wsel = 0;
// ---- ACTIVATION-scale double buffer split (2026-07-25, required for FA_STEADY) --------------
// The ACT scale SRAM has the same 2-half double buffer as the weight one (ScaleFactorMem.scala:195
// `double_buffer_act_sel` <- CONFIG_SCALE_MEM rs1[60]; write half selected by addr bit 11, i.e.
// +GEMMINI_SF_MEM_BUFFER_OFFSET).  Two DIFFERENT producers need SF_MEM_A within one FA tile:
//     QK^T : A = Q      -> QK_A_scales_row, written by the HOST
//     PV   : A = P(fp8) -> runtime softmax scales, written by the GPU (pack_scales_to_sfmem,
//                          which hardcodes GEMMINI_SF_MEM_A == half 0 and is not my file)
// Single-shot they can share half 0 because QK consumes it before pack overwrites it.  Under
// FA_STEADY (NTILES>1) that is FATAL: tile t's pack leaves the P scales in half 0, and tile t+1's
// QK then multiplies Q by softmax scales.  Fix: the host puts QK_A in half **1** and QK reads with
// act_sel=1, while pack_scales/PV keep half 0 (act_sel=0).  The two halves are physically distinct
// SRAM banks (act banks 0/1 vs 2/3 -> ScaleFactorMem banks 4/5 vs 6/7), so there is no aliasing.
// This also means the host's QK_A half is never touched by the GPU -> it can be refilled per tile.
static uint32_t g_host_sf_asel = 0;

// Host<->GPU mailbox, in an unused 256B window of cluster SMEM (device 0x17F00: the kernel's SMEM
// map ends at REDBUF_SMEM=0x15000 and the gemmini B-spad starts at 0x18000).  SMEM is the only
// channel that is both cheap for the Muon and visible to a Rocket store at cluster_base+offset
// (GMEM would sit behind the non-coherent L1; the print-buffer TLRAM at +0x80000 was tried and
// host stores there never became visible).  Host side MUST use 8-byte stores.  Match host.cpp.
#define FA_HOST_MBOX      0x17F00u
#define FA_MBOX_QK_READY  0x00u
#define FA_MBOX_V_READY   0x10u
#define FA_HOST_MAGIC     0x5CA1E5u
// ---- per-tile handshake slots (-DFA_HOSTHS; see host.cpp) ---------------------------------
// GPU->host progress counters.  The host must not rewrite a scale half while the mesh is still
// reading it, so it waits for these before refilling.  MUST live in slots the host never writes
// (the "GPU must not clear the mailbox" rule is about the host's OWN flags; separate words are
// safe in both directions).  Values are "number of tiles completed", i.e. 1-based.
#define FA_MBOX_GPU_QKDONE 0x20u
#define FA_MBOX_GPU_PVDONE 0x30u
// -DFA_HOSTPACK: the host, not the GPU, packs the runtime P scales into SF_MEM_A, so that the GPU
// is not an SF requestor at all and the host's 8-byte stores never share FlitMergeNode's pairing
// state with a 4-byte GPU store.  See the long comment in host.cpp.  The GPU must additionally be
// built -DFA_NOPACK (which compiles out pack_scales_to_sfmem at its call site in the kernel body).
#define FA_MBOX_GPU_PACKREQ 0x50u  // GPU -> host: #tiles whose requant is done
#define FA_MBOX_HOST_PACKED 0x60u  // host -> GPU: #tiles whose P scales are resident in SF_MEM_A
// Diagnostics the host publishes for the GPU to echo out (see FA_HOST_TIMING echo below).
#define FA_MBOX_DIAG       0x80u   // 0x80..0xbf
// cycle stamp -> FA_DIAG_HSMARK+off (0x88/0x8c = QK scale wait, 0x90/0x94 = V scale wait).
// Own 4KB page + opaque base register: a constant-address store is traced as its PAGE BASE, which
// used to land these on top of the kernel's m[0].  See the diagnostic map at the top of this file.
// GATED ON FA_HOST_TIMING (2026-07-26).  These are pure instrumentation but were compiled into
// every -DFA_NOSCALES build, and a single-thread GMEM store is NOT cheap here: two stamps per
// prefetch x two prefetches = 4 stores per tile, measured at ~1.6k cyc/tile of the steady-state
// slope (80,054 -> 78,4xx when removed).  Never leave a measurement probe in the hot path.
#ifdef FA_HOST_TIMING
#define FA_HOST_MARK(off) do { uint32_t _c; asm volatile("csrr %0, mcycle" : "=r"(_c)); \
    volatile uint32_t *_hp = (volatile uint32_t *)(FA_DIAG_HSMARK + (off)); \
    asm volatile("" : "+r"(_hp)); *_hp = _c; } while (0)
#else
#define FA_HOST_MARK(off) do {} while (0)
#endif
#ifdef FA_HOSTHS
// Per-tile index, advanced by the PV compute (the last mesh op of an FA tile).  Used to index the
// host's per-tile scale-ready sequence numbers.  Thread 0 only.
static uint32_t g_fa_tile = 0;
static uint32_t g_fa_spin_qk = 0, g_fa_spin_v = 0;   // accumulated stall waiting on the host
#endif

#if defined(FA_HOSTCFG) && !defined(FA_HOSTHS)
#error "FA_HOSTCFG needs FA_HOSTHS: the replay is gated on the per-tile QKDONE/PVDONE handshake"
#endif
#if defined(FA_HOSTPACK) && !defined(FA_HOSTHS)
#error "FA_HOSTPACK needs FA_HOSTHS (per-tile mailbox handshake) -- and the kernel body must be built -DFA_NOPACK"
#endif
#ifdef FA_HOSTCFG
// ============================================================================================
// HOST-ISSUED GEMMINI CONFIG + MOVE-IN  (-DFA_HOSTCFG; implies FA_NOSCALES + FA_HOSTHS)
// ============================================================================================
// mxgemm_prefetch_tile's thread-0 preamble is 12 ROCC commands per gemm:
//    configure_mxgemmini      7  (config_ex, 2x config_ld, config_st, CONFIG_SCALE_MEM,
//                                 2x LOOP_WS_CONFIG_BOUNDS)  + a gemmini_fence
//    copy_gmem_to_smem_async  5  (LOOP_WS_CONFIG_ADDRS_AB, LOOP_WS_CONFIG_STRIDES_AB,
//                                 LOOP_WS_CONFIG_BOUNDS, LOOP_WS_CONFIG_SPAD_AB, LOOP_WS)
// and one ROCC command is 5 `sw.shared` (rs1 lo/hi, rs2 lo/hi, inst) -- 60 single-thread stores
// into the gemmini MMIO port, all strictly serial on the Muon's one issuing lane.  Rocket can
// drive the SAME port: GemminiTile.scala:419-430 maps it as a plain 8-byte-beat TLRegisterNode at
// cluster_base + 0x84000, and rs1/rs2 are a pair of 32-bit RegFields per 8-byte word, so ONE host
// `sd` replaces two GPU stores.  12 commands = 36 host stores per cluster.
//
// THE HOST CANNOT COMPUTE THE COMMANDS, SO IT REPLAYS THEM.  rs1/rs2 carry GPU-link-time symbol
// addresses (rad_device_to_host_address(&QK_A_in[0][0]) etc.) that the separately-compiled rv64
// host image has no way to know.  So: on tile 0 the GPU issues the stream as usual AND records
// each (rs1_lo, rs1_hi, rs2_lo, rs2_hi, inst) into cluster SMEM; the host reads the recording
// once and replays it for every later tile.  rad_device_to_host_address is a pure OR with
// RAD_HOST_GPU_DRAM_BASE (radiance.h:31-33) and carries no cluster id, so ONE recording is valid
// for BOTH clusters.  (In a real streaming FA the only per-tile-varying words are the two GMEM
// base addresses in LOOP_WS_CONFIG_ADDRS_AB, which the host would patch by adding a stride.)
//
// MUTUAL EXCLUSION.  rs1/rs2 are latched REGISTERS and the write to +0x00 is what fires the
// command, so an interleaved host/GPU command is not a corrupted queue entry -- it is a command
// built from the wrong operands.  The existing per-tile flags serialize it: for tile t the host
// replays the QK stream only after PVDONE >= t (tile t-1's PV matmul, the GPU's last ROCC write,
// has drained) and publishes QK_READY = t+1 afterwards, which is what tile t's prefetch waits on;
// the PV stream is replayed after QKDONE >= t+1 (so it cannot precede tile t's QK matmul issue,
// and cannot overwrite the B spad while the QK matmul is still reading K out of it) and gates
// V_READY = t+1.  The GPU's own gemmini accesses in between are BUSY/READY *reads*, which are
// harmless.  Writes to the command register backpressure on gemminiIO.ready
// (GemminiTile.scala:396-400), so neither side needs to poll it.
#define FA_CFGREC_QK   0x17800u   // [0] = #commands, then 5 words per command (13 cmds = 264 B)
#define FA_CFGREC_PV   0x17A00u
#define FA_MBOX_CFGREC 0x40u      // GPU -> host: 1 = QK recorded, 2 = QK and PV recorded
static volatile __shared uint32_t *g_cfgrec = nullptr;  // non-null => record every ROCC command
static uint32_t g_cfgrec_n = 0;

// Recording wrapper around the MMIO ROCC issue macro (mxgemmini_mmio.h:63-68).  Must be defined
// before configure_mxgemmini / copy_gmem_to_smem_async, which expand it.
#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    const uint64_t _r1 = gemmini_arg_to_u64(rs1); \
    const uint64_t _r2 = gemmini_arg_to_u64(rs2); \
    const uint32_t _in = (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25); \
    if (g_cfgrec) { \
        g_cfgrec[0] = (uint32_t)_r1; g_cfgrec[1] = (uint32_t)(_r1 >> 32); \
        g_cfgrec[2] = (uint32_t)_r2; g_cfgrec[3] = (uint32_t)(_r2 >> 32); \
        g_cfgrec[4] = _in; g_cfgrec += 5; g_cfgrec_n++; \
    } \
    store64_shared(GEMMINI_CTRL, GEMMINI_RS1_OFFSET, _r1); \
    store64_shared(GEMMINI_CTRL, GEMMINI_RS2_OFFSET, _r2); \
    store_shared  (GEMMINI_CTRL, GEMMINI_INST_OFFSET, _in); \
}
#endif
#endif

template <GemmConfig C>
static inline void configure_mxgemmini(const uint32_t dim_m,
                                       const uint32_t dim_n,
                                       const uint32_t dim_k,
                                       const uint32_t scale_w_sel = 0,
                                       const uint32_t scale_act_sel = 0) {
    // NOTE: non-square tiles (TILE_M != TILE_N) are supported -- the loop bounds are set
    // per-dimension from PE_TILES_I/J/K below, and the A/B spad quarters are sized
    // independently. Required for streaming FA (QK: N=Bk!=M=Sq; PV: N=d!=M=Sq).
    static_assert(C.TILE_M % C.PE_M() == 0 && C.TILE_N % C.PE_N() == 0,
                  "TILE_M/TILE_N must be multiples of the PE tile size");
    static_assert(C.TILE_K >= 32 && (C.TILE_K % 32) == 0,
                  "tile K dimension is not a multiple of block size (32)");

    // NOTE: gemmini_flush hoisted to once-per-kernel (fa_entry); between gemms the mesh is
    // fenced-idle so a per-gemm flush is redundant overhead.

    constexpr auto GEMMINI_FORMAT =
        C.DATATYPE == GemmDatatype::FP8 ? GEMMINI_FORMAT_FP8 :
        C.DATATYPE == GemmDatatype::FP6 ? GEMMINI_FORMAT_FP6 :
                                          GEMMINI_FORMAT_FP4;
    constexpr auto GEMMINI_FORMAT_OUT =
        C.QUANT_OUTPUT ? GEMMINI_FORMAT : GEMMINI_FORMAT_FULL;

    gemmini_extended3_config_ex(
        WEIGHT_STATIONARY, // dataflow
        0, 0, ACC_SCALE_IDENTITY, // sys_act, sys_shift, sys_acc_scale
        1, 1, // C_stride, A_stride
        0, 0, // A_transpose, B_transpose
        false, // set_only:strides
        GEMMINI_FORMAT, // A dtype
        GEMMINI_FORMAT, // B dtype
        GEMMINI_FORMAT_OUT, // C dtype
        C.USE_LUT()  // uselut
    );

    // Configure GMEM move-in strides for A and B
    // NOTE: FP4/FP6 packs elements by M and N dimensions
    gemmini_extended3_config_ld(dim_k * sizeof(uint8_t), MVIN_SCALE_IDENTITY,
                                false, 0);
    gemmini_extended3_config_ld(dim_n * sizeof(uint8_t) / C.VALUES_PER_BYTE(),
                                MVIN_SCALE_IDENTITY, false, 1);

    // Configure GMEM move-out stride for C
    gemmini_config_st(dim_n * C.OUT_ELEM_SIZE());

    // Configure scalefac->PE read and scalefac->GMEM write addresses; inst: 0x3420b07b
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        scale_act_sel, // A (act) scale double-buffer select (1 = host-prefilled QK_A scales)
        scale_w_sel,   // B (weight) scale double-buffer select (1 = host-prefilled V scales)
        QUANT_LUT_UPDATE_GRANULARITY);

    // Configure loop bounds for the loop FSM
    // This only needs to be done once since the kernel does not change the
    // SMEM tile size
    // Configure the two loop FSMs (issued back-to-back); a single fence drains both.
    gemmini_loop_ws_config_bounds(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        0, 0, 0 // pad_I=0, pad_J=0, pad_K=0
    );
    gemmini_loop_ws_config_bounds(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        0, 0, 0 // pad_I=0, pad_J=0, pad_K=0
    );
    gemmini_fence();
}

/** Calculate scratchpad row address for A if `is_b == false` or B if `is_b == true`. */
template <bool is_b>
static inline uint32_t calculate_spad_addr(const uint32_t tile_k) {
    constexpr auto SMEM_SIZE_ROWS = BANK_NUM * BANK_ROWS;
    constexpr auto SMEM_QUARTER_ROWS = SMEM_SIZE_ROWS / 4;
    static_assert(SMEM_QUARTER_ROWS != 0);
    constexpr auto A_SPAD_ADDR_EVEN = 0;
    constexpr auto A_SPAD_ADDR_ODD = SMEM_QUARTER_ROWS;
    // B spad address is counted from the end (SMEM_SIZE_ROWS)
    // TODO: might want to swap even and odd (do bank 0-2, 1-3 instead of 0-3, 1-2)
    constexpr auto B_SPAD_ADDR_EVEN = SMEM_SIZE_ROWS;
    constexpr auto B_SPAD_ADDR_ODD = SMEM_SIZE_ROWS - SMEM_QUARTER_ROWS;

    const uint32_t odd_k = (tile_k & 1);
    const uint32_t a_spad_addr = odd_k ? A_SPAD_ADDR_ODD : A_SPAD_ADDR_EVEN;
    const uint32_t b_spad_addr = odd_k ? B_SPAD_ADDR_ODD : B_SPAD_ADDR_EVEN;

    if constexpr (is_b) {
        return b_spad_addr;
    } else {
        return a_spad_addr;
    }
}

template <bool is_b>
static inline __shared uint32_t *
calculate_scale_factor_smem_addr(const uint32_t tile_k) {
    const uint32_t odd_k = (tile_k & 1);
    const uint32_t dbuf_offset = odd_k ? GEMMINI_SF_MEM_BUFFER_OFFSET : 0;
    auto a_sf_addr =
        reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_A + dbuf_offset);
    auto b_sf_addr =
        reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_B + dbuf_offset);

    if constexpr (is_b) {
        return b_sf_addr;
    } else {
        return a_sf_addr;
    }
}

template <GemmConfig C, bool is_b>
static inline const uint8_t *
calculate_scale_factor_gmem_addr(const uint8_t *scales_base_addr,
                                 const uint32_t tile_k, const uint32_t dim_m,
                                 const uint32_t dim_n) {
    const auto dim_mn = is_b ? dim_n : dim_m;
    const auto scales_addr =
        scales_base_addr +
        (!ZERO_STRIDE_K ? (tile_k * C.TILE_K * dim_mn / 32) * sizeof(uint8_t)
                        : 0);
    return scales_addr;
}

// LANE-PARALLEL scale load: 16 lanes of ONE warp write 16 consecutive SF-SRAM words (coalesced).
// The SF/requantizer scale interface only corrupts under MULTI-WARP parallel writes; a single warp's
// lanes are fine, and because the matmul is issued by lane 0 of the SAME warp afterwards, program
// order + `fence.s` suffice (NO cross-warp barrier needed). Single-threaded load_scale_factors was
// ~100 cyc/word => ~64k cycles total across the 3 call sites (33% of the whole kernel).
// NOTE the partitioning: lanes take CONTIGUOUS BLOCKS, not an interleave. An interleaved
// (i = lane; i += 16) pattern makes the 16 lanes write 16 CONSECUTIVE words in one cycle, which the
// coalescer merges into a 64B burst -- and the SF-SRAM scale-write path rejects that
// (FlitMergeNode_1.sv:115 "Assertion failed: start address not aligned" -> $finish). Block-partitioning
// keeps every lane's store a single 4B write to a far-apart address (no merge) while still putting 16
// writes in flight, which is what actually hides the ~100 cyc/word SF-SRAM latency.
static void __attribute__((noinline))
load_scale_factors_lanes(volatile __shared uint32_t *sf_mem, const uint8_t *scale_factors,
                         const int n, const uint32_t lane) {
    auto src = reinterpret_cast<const uint32_t *>(scale_factors);
    const uint32_t nw = (uint32_t)n / 4;
    const uint32_t per = (nw + MU_NUM_THREADS - 1) / MU_NUM_THREADS;
    const uint32_t base = lane * per;
    for (uint32_t j = 0; j < per; j++) {
        const uint32_t i = base + j;
        if (i < nw) sf_mem[i] = src[i];
    }
}

static void __attribute__((noinline))
load_scale_factors(volatile __shared uint32_t *sf_mem, const uint8_t *scale_factors,
                   const int n) {
    // asm volatile ("load_scale_factors_start_%=:" :: );
    auto word_scale_factors = reinterpret_cast<const uint32_t *>(scale_factors);

    // unroll in registers to reduce back-to-back WAW/WAR
    constexpr auto ILP = 8;
    uint32_t unrolled[ILP];
    #pragma unroll 4
    for (size_t i = 0; i < n / 4; i += ILP) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            // do full-word stores instead of 1-byte stores
            unrolled[j] = word_scale_factors[i + j];
        }
        for (int j = 0; j < ILP; j++) {
            sf_mem[i + j] = unrolled[j];
        }
    }
    // asm volatile ("load_scale_factors_end_%=:" :: );
}

template <GemmConfig C>
static inline void load_lut() {
    asm volatile ("load_lut_start_%=:" :: );

    if constexpr (C.USE_LUT()) {
        // TODO: fix to use GEMM_MN
        for (size_t i = 0; i < (C.TILE_N >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            auto *dst = reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT0_ADDR) + 3 * i;
            dst[0] = B_lut[i][0]; dst[1] = B_lut[i][1]; dst[2] = B_lut[i][2];
        }
        for (size_t i = 0; i < (C.TILE_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            auto *dst = reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT1_ADDR) + 3 * i;
            dst[0] = A_lut[i][0]; dst[1] = A_lut[i][1]; dst[2] = A_lut[i][2];
        }
        for (size_t i = 0; i < (C.TILE_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            auto *dst = reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT2_ADDR) + 3 * i;
            dst[0] = C_lut[i][0]; dst[1] = C_lut[i][1]; dst[2] = C_lut[i][2];
        }
    }

    asm volatile ("load_lut_end_%=:" :: );

}

// EXPLICIT_MVIN: use per-tile `gemmini_extended_mvin` commands instead of the loop_ws FSM.
// The loop FSM's mvin with skip_lda=1 (SKIP_A) appears to leave a PHANTOM outstanding completion
// (H8: gemmini idle + occupancy MMIO stuck non-zero -> every gemmini_fence livelocks and the next
// matmul can't even enqueue). Explicit mvin commands bypass that skip accounting entirely.
template <GemmConfig C, bool SKIP_A = false, bool EXPLICIT_MVIN = false>
static inline void copy_gmem_to_smem_async(
    const uint8_t *A_in, const uint8_t *B_in,
    const uint32_t dim_m, const uint32_t dim_n, const uint32_t dim_k,
    const uint32_t tile_i /* FIXME: unused */,
    const uint32_t tile_j /* FIXME: unused */, const uint32_t tile_k) {
    asm volatile ("copy_gmem_to_smem_async_start_%=:" :: );

    // Gemmini expects the full A/B tensor to be stored in block-level
    // row-major layout, i.e.:
    // The tensor is partitioned into DIM x DIM tiles.
    // Tiles are ordered row-by-row in memory (all tile columns of tile-row 0,
    // then tile-row 1, etc.), and each tile is stored contiguously.

    const uint32_t a_spad_addr_start = calculate_spad_addr<false>(tile_k);
    const uint32_t b_spad_addr_end = calculate_spad_addr<true>(tile_k);

    if constexpr (GEMMINI_DMA && !EXPLICIT_MVIN) {
        // Configure GMEM address for A and B
        // TODO: stride by tile_i/j
        // TODO: possibly create functions for A/B row-stride
        // inst: 0x1420b07b
        const uint32_t A_tile_start = reinterpret_cast<uint32_t>(A_in) +
                                      (!ZERO_STRIDE_K ? C.TILE_K * tile_k : 0);
        const uint32_t B_tile_start =
            reinterpret_cast<uint32_t>(B_in) +
            (!ZERO_STRIDE_K ? dim_n * C.TILE_K * tile_k / C.VALUES_PER_BYTE()
                            : 0);
        ROCC_INSTRUCTION_RS1_RS2(
            XCUSTOM_ACC,
            rad_device_to_host_address(A_tile_start),
            rad_device_to_host_address(B_tile_start),
            k_LOOP_WS_CONFIG_ADDRS_AB)

        // Configure loop FSM GMEM move-in strides for A and B
        // This only needs to be done once since the kernel does not change the
        // SMEM tile size
        // FIXME: However, moving this out of the loop breaks addresses?
        ROCC_INSTRUCTION_RS1_RS2(
            XCUSTOM_ACC,
            (uint64_t)(dim_k * sizeof(uint8_t)),
            (uint64_t)(dim_n * sizeof(uint8_t) / C.VALUES_PER_BYTE()),
            k_LOOP_WS_CONFIG_STRIDES_AB /* 0x1820b07b */)

        // Kick off DMA move-in via the loop FSM
        //
        // gemmini_loop_ws_spad issues three instructions:
        //   1. configure loop bounds (inst: 0x1220b07b, funct: k_LOOP_WS_CONFIG_BOUNDS)
        //   2. configure spad addresses (inst: 0x3020b07b, funct: k_LOOP_WS_CONFIG_SPAD_AB)
        //   3. compute loop ws with skips (inst: 0x1020b07b, funct: k_LOOP_WS)
        // TODO: skip re-configuring of loop bounds
        constexpr uint32_t skips_mvin =
            loop_matmul_skips(/*skip_lda=*/SKIP_A ? 1 : 0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                              /*skip_ex=*/1, /*skip_stc=*/1);
        constexpr auto DONTCARE = 0;
        gemmini_loop_ws_spad(
            C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), // loop bounds for I, J, K (single 16×16 PE tile)
            0, 0, 0,              // pad_I=0, pad_J=0, pad_K=0
            a_spad_addr_start,    // A scratchpad address in rows (grows upward)
            b_spad_addr_end,      // B scratchpad address in rows (grows downward)
            0,                    // D (bias) - none
            DONTCARE,             // C scratchpad address in rows
            false, false,         // A_transpose, B_transpose
            false, false, false,  // full_C, low_D, ex_accumulate
            NO_ACTIVATION,        // activation
            0, 0,                 // a_spad_id, b_spad_id
            false,                // is_resadd
            skips_mvin);          // skips

    } else { // !GEMMINI_DMA

        // FIXME: these are already done above at configure_gemmini(); should
        // be redundant
        gemmini_config_ld(dim_k * sizeof(uint8_t));

        // A layout: for each i row, store all k tiles contiguously
        // A tile (i,k) -> a_base + (i * tiles_K + k) * DIM
        // SKIP_A: A already resident in the spad (e.g. a SIMT requant wrote P there) -> no A mvin.
        if constexpr (!SKIP_A)
        for (int i = 0; i < C.PE_TILES_I(); i++) {
            for (int k = 0; k < C.PE_TILES_K(); k++) {
                // TODO: missing TILE_K offset
                const uint8_t *dram_ptr = ((uint8_t*)A_in) + i * DIM * dim_k + k * DIM;
                const uint32_t sp_addr = a_spad_addr_start + (i * C.PE_TILES_K() + k) * DIM;
                // Note gemmini needs CPU-global addresses for mvin
                gemmini_extended_mvin(rad_device_to_host_address(
                    reinterpret_cast<uint32_t>(dram_ptr)),
                                      sp_addr, DIM, DIM);
            }
        }

        gemmini_config_ld(dim_n * sizeof(uint8_t) / C.VALUES_PER_BYTE());

        // B layout: for each k row, store all j tiles contiguously
        // B tile (k,j) -> b_base + (k * tiles_J + j) * DIM
        for (int k = 0; k < C.PE_TILES_K(); k++) {
            for (int j = 0; j < C.PE_TILES_J(); j++) {
                // TODO: missing TILE_K offset
                const uint8_t *dram_ptr =
                    ((uint8_t *)B_in) + k * DIM * dim_n / C.VALUES_PER_BYTE() +
                    j * DIM;
                const uint32_t b_spad_addr_start = b_spad_addr_end - C.PE_TILES_K() * C.PE_TILES_J() * DIM;
                const uint32_t sp_addr = b_spad_addr_start + (k * C.PE_TILES_J() + j) * DIM;
                gemmini_extended_mvin(rad_device_to_host_address(
                    reinterpret_cast<uint32_t>(dram_ptr)),
                                      sp_addr, DIM, DIM);
            }
        }
    } // end !GEMMINI_DMA

    asm volatile ("copy_gmem_to_smem_async_end_%=:" :: );
}

/** Move tensor data from SMEM->GMEM using SIMT threads.
 *  Assumes row-major, packed layout (row stride == dim_col) for both src and dest.
 *  TODO: De-dup with FlashAttention */
template <uint32_t dim_row, uint32_t dim_col, uint32_t elem_size>
static void copy_smem_to_gmem_simt(const __shared uint8_t *src_smem,
                                   uint8_t *dest_gmem,
                                   const uint32_t tid_in_threadblock,
                                   const uint32_t threads_per_threadblock) {
    asm volatile("copy_smem_to_gmem_simt_start_%=:" ::);

    // Thread mapping: All warps in a threadblock cooperatively copies a
    // contiguous chunk of the same size as the threadblock per every "wave".

    // Vectorize to 32-bit words for better throughput.
    auto *src_smem_vec = reinterpret_cast<const __shared uint32_t *>(src_smem);
    auto *dest_gmem_vec = reinterpret_cast<uint32_t *>(dest_gmem);
    static_assert((dim_row * dim_col * elem_size) % sizeof(uint32_t) == 0);
    const auto iter = dim_row * dim_col * elem_size / sizeof(uint32_t) /
                      threads_per_threadblock;

#pragma unroll 32
    for (int i = 0; i < iter; i++) {
        // simple uniform-strided access
        const auto index = (threads_per_threadblock)*i + tid_in_threadblock;
        const auto smem_addr = src_smem_vec + index;
        auto gmem_addr = dest_gmem_vec + index;
        *gmem_addr = *smem_addr;
    }

    asm volatile ("copy_smem_to_gmem_simt_end_%=:" :: );
}

/** Copy tensor data from GMEM->SMEM using SIMT threads.
 *  Does 16-bit writes in order to comply with requantizer memory interface. */
template <uint32_t dim_row, uint32_t dim_col, uint32_t elem_size>
static void copy_gmem_to_smem_simt_bf16(
    const uint8_t *src_gmem, __shared uint8_t *dest_smem,
    const uint32_t tid_in_threadblock, const uint32_t threads_per_threadblock) {
    asm volatile("copy_gmem_to_smem_simt_bf16_start_%=:" ::);

    // TODO: dedup with copy_smem_to_gmem_simt

    // Vectorize to 16-bit words
    auto *src_gmem_vec = reinterpret_cast<const uint16_t *>(src_gmem);
    auto *dest_smem_vec = reinterpret_cast<__shared uint16_t *>(dest_smem);
    static_assert((dim_row * dim_col * elem_size) % sizeof(uint16_t) == 0);
    const auto iter = dim_row * dim_col * elem_size / sizeof(uint16_t) /
                      threads_per_threadblock;

#pragma unroll 32
    for (int i = 0; i < iter; i++) {
        // simple uniform-strided access
        const auto index = (threads_per_threadblock)*i + tid_in_threadblock;
        const auto src_addr = src_gmem_vec + index;
        auto dst_addr = dest_smem_vec + index;
        *dst_addr = *src_addr;
    }

    asm volatile("copy_gmem_to_smem_simt_bf16_end_%=:" ::);
}

/** Copy tensor data from GMEM->GMEM using SIMT threads.
 *  Used for generating memory traces to verify. */
template <uint32_t dim_row, uint32_t dim_col, uint32_t elem_size>
static void copy_gmem_to_gmem_simt(const uint8_t *src_gmem, uint8_t *dest_gmem,
                                   const uint32_t tid_in_threadblock,
                                   const uint32_t threads_per_threadblock) {
    asm volatile ("copy_gmem_to_gmem_simt_start_%=:" :: );

    // TODO: dedup with copy_smem_to_gmem_simt

    // Vectorize to 32-bit words for better throughput.
    auto *src_gmem_vec = reinterpret_cast<const uint32_t *>(src_gmem);
    auto *dest_gmem_vec = reinterpret_cast<uint32_t *>(dest_gmem);
    static_assert((dim_row * dim_col * elem_size) % sizeof(uint32_t) == 0);
    const auto iter = dim_row * dim_col * elem_size / sizeof(uint32_t) /
                      threads_per_threadblock;

#pragma unroll 32
    for (int i = 0; i < iter; i++) {
        // simple uniform-strided access
        const auto index = (threads_per_threadblock) * i + tid_in_threadblock;
        const auto src_addr = src_gmem_vec + index;
        auto dst_addr = dest_gmem_vec + index;
        *dst_addr = *src_addr;
    }

    asm volatile ("copy_gmem_to_gmem_simt_end_%=:" :: );
}

/** Move C result tensor from SMEM->GMEM using Gemmini DMA.
 *  `src_spad_addr` is in scratchpad row address.
 *  This call blocks and synchronizes with the completion of the DMA. */
template <GemmConfig C>
static void copy_C_smem_to_gmem_dma_sync(const uint32_t src_spad_addr,
                                         uint8_t *dest_gmem,
                                         const uint32_t dim_n,
                                         const uint32_t tid_in_threadblock) {
    asm volatile("copy_smem_to_gmem_dma_sync_start_%=:" ::);

    if (tid_in_threadblock == 0) {
        for (int i = 0; i < C.PE_TILES_I(); i++) {
#pragma unroll 32
            for (int j = 0; j < 2 * C.PE_TILES_J(); j++) {
                const uint32_t tile_spad_addr =
                    src_spad_addr + (i * 2 * C.PE_TILES_J() + j) * DIM;
                // row-major layout
                // TODO: DRAM stride is wrong for re-quantized output
                uint8_t *dram_ptr =
                    dest_gmem + (i * 2 * DIM * dim_n + j * DIM) * C.OUT_ELEM_SIZE();
                gemmini_mvout(rad_device_to_host_address(
                                  reinterpret_cast<uint32_t>(dram_ptr)),
                              tile_spad_addr);
            }
        }

        gemmini_fence();
    }

    asm volatile("copy_smem_to_gmem_dma_sync_end_%=:" ::);
}

/** Move tensor data from AccMEM->GMEM using Gemmini DMA.
 *  This call blocks and synchronizes with the completion of the DMA. */
template <GemmConfig C>
static void copy_accmem_to_gmem_dma_sync(uint8_t *dest_gmem,
                                         const uint32_t dim_n,
                                         const uint32_t tid_in_threadblock) {
    asm volatile("copy_accmem_to_gmem_dma_sync_start_%=:" ::);

    if (tid_in_threadblock == 0) {
        for (int i = 0; i < C.PE_TILES_I(); i++) {
#pragma unroll 32
            // need 4 because 4 fit in accmem row
            for (int j = 0; j < C.PE_TILES_J() / 4; j++) {
                const uint32_t tile_acc_addr =
                    GEMMINI_ACC_ADDR + (i * C.PE_TILES_J() / 4 + j) * DIM;
                // row-major layout
                // TODO: DRAM stride is wrong for re-quantized output
                uint8_t *dram_ptr =
                    dest_gmem +
                    (i * DIM * dim_n + j * DIM * 2 /*is this right?*/) *
                        C.OUT_ELEM_SIZE();
                gemmini_mvout(rad_device_to_host_address(
                                  reinterpret_cast<uint32_t>(dram_ptr)),
                              tile_acc_addr);
            }
        }

        gemmini_fence();
    }

    asm volatile("copy_accmem_to_gmem_dma_sync_end_%=:" ::);
}

/** Asynchronously kick off loop FSM matmul compute operation in MxGemmini.
 *  Move out accumulator data to SMEM if `acc_move_out` is true. */
template <GemmConfig C>
static inline void matmul_tile_async(const uint32_t tile_k, const bool acc_move_out,
                                     const bool accumulate = false,
                                     const uint32_t b_spad_override = 0xffffffffu,
                                     const uint32_t c_spad_dest = SPAD_DEST,
                                     const uint32_t a_spad_override = 0xffffffffu,
                                     const int force_first = -1) {
    asm volatile ("matmul_tile_async_start_%=:" :: );

    const uint32_t skip_stc = acc_move_out ? 0 : 1;
    const uint32_t skips_compute =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/0, /*skip_stc=*/skip_stc);

    const uint32_t a_spad_addr_start = (a_spad_override != 0xffffffffu)
                                       ? a_spad_override : calculate_spad_addr<false>(tile_k);
    const uint32_t b_spad_addr_end = (b_spad_override != 0xffffffffu)
                                     ? b_spad_override : calculate_spad_addr<true>(tile_k);

    // first_k gates ex_accumulate (overwrite vs accumulate). Normally tile_k==0, but parity-
    // decoupled callers (odd double-buffer for a single-tile OVERWRITE matmul) pass force_first.
    const bool first_k = (force_first < 0) ? (tile_k == 0) : (force_first != 0);
    // `accumulate` forces in-accumulator add (mesh-accumulate PV across blocks -> no SIMT rescale)

    // TODO: support skipping move-out to SMEM
    // TODO(perf): !first_k creates a branch
    gemmini_loop_ws_spad(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), // loop bounds for I, J, K (single 16×16 PE tile)
        0, 0, 0,                // pad_I=0, pad_J=0, pad_K=0
        a_spad_addr_start,      // A scratchpad address in rows (grows upward)
        b_spad_addr_end,        // B scratchpad address in rows (grows downward)
        0,                      // D (bias) - none
        c_spad_dest,            // C scratchpad address in rows (default SPAD_DEST; param -> double-buffer)
        false, false,           // A_transpose, B_transpose
        false, false, (!first_k || accumulate), // full_C, low_D, ex_accumulate (accumulate=cross-block)
        NO_ACTIVATION,          // activation
        0, 0,                   // a_spad_id, b_spad_id
        false,                  // is_resadd
        skips_compute);         // skips

    asm volatile ("matmul_tile_async_end_%=:" :: );
}

/** Do matmul on a single TILE_M * TILE_N output tile, accumulating over the
 *  full GEMM_K. */
template <GemmConfig C, bool barrier_tile = false, bool SKIP_A = false, bool DO_CONFIG = true>
__attribute__((noinline)) void mxgemm_single_output_tile(const uint8_t *A_in, const uint8_t *B_in,
                               const uint8_t *A_scales, const uint8_t *B_scales,
                               const uint32_t dim_m, const uint32_t dim_n,
                               const uint32_t dim_k,
                               const uint32_t tid_in_threadblock,
                               const uint32_t threads_per_threadblock) {
    asm volatile ("mxgemm_single_output_tile_start_%=:" :: );

    constexpr auto barrier_id = 2;
    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;

    if (tid_in_threadblock != 0) {
        return;
    }

    // DO_CONFIG=false: the persistent gemmini config was set once by the caller (valid
    // when successive gemms share dims -- e.g. square streaming FA blocks Bk=Sq=d, where
    // QK and PV configs are identical). Skips ~half the per-gemm ROCC command overhead.
    if constexpr (DO_CONFIG) configure_mxgemmini<C>(dim_m, dim_n, dim_k);

    // -----------------
    // Initiate pipeline
    // -----------------
    //
    int tile_k = 0;
    // TODO: change 0's for multiple SMEM tiles
    copy_gmem_to_smem_async<C, SKIP_A>(A_in, B_in, dim_m, dim_n, dim_k, 0, 0, tile_k);

    // Load scaling factors from GMEM to the scale SRAM
    // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));
    // SKIP_A: A is already in the spad AND its scales are already in the A scale SRAM
    // (e.g. a SIMT requant placed both); skip the A move-in (above) and the A scale load.
    if constexpr (!SKIP_A) {
        load_scale_factors(calculate_scale_factor_smem_addr<false>(tile_k),
                           calculate_scale_factor_gmem_addr<C, false>(
                               A_scales, tile_k, dim_m, dim_n),
                           C.SCALE_FACTORS_PER_TILE());
    }
    load_scale_factors(calculate_scale_factor_smem_addr<true>(tile_k),
                       calculate_scale_factor_gmem_addr<C, true>(
                           B_scales, tile_k, dim_m, dim_n),
                       C.SCALE_FACTORS_PER_TILE());

    // LUT is shared across the entire K, and thus loaded once per one SMEM
    // output tile
    load_lut<C>();

    // fence scale factor and LUT writes
    mu_fence_smem();

    // wait for GMEM->SMEM copy
    gemmini_fence();

    if constexpr (barrier_tile) {
        mu_barrier(barrier_id, warps_per_threadblock);
    }

    // ------------------------------
    // Main software-pipelined K-loop
    // ------------------------------
    //
    asm volatile ("main_matmul_k_loop_start_%=:" :: );

    // Potential software-pipelining loop structures:
    //                 ┌───┐   ┌───┐
    //             ┌───────────┐
    //         ┌───────┐
    // Loop 1: M0->M1->C0->M0->C1->M1->C0
    //         ┌───┐   ┌───┐   ┌───┐
    // Loop 2: M0->C0->M1->C1->M0->C0->...
    //
    for (; (tile_k * C.TILE_K) < dim_k; tile_k++) {
        const auto odd_k = (tile_k & 1);
        const auto odd_next_k = !odd_k;
        const auto last_k = ((tile_k + 1) * C.TILE_K) >= dim_k;

        // configure scalefac->PE double-buffer read; inst: 0x3420b07b
        // done for (tile_k) compute; we do this before (tile_k + 1) DMA,
        // since this may get serialized with the DMA instruction
        gemmini_mxquant_config_mvout(
            // TODO: dummy move-out space for the scale factor
            rad_device_to_host_address(
                reinterpret_cast<uint32_t>(&C_scale_factors[0])),
            C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
            odd_k, // A double-buffer toggle
            odd_k, // B double-buffer toggle
            QUANT_LUT_UPDATE_GRANULARITY);

        // GMEM->SMEM DMA for the next tile_k
        // TODO: This results in an unnecessary move-in at the last K tile
        if constexpr (!DISABLE_MOVE_IN_AFTER_FIRST_K) {
            if (!last_k) {
                copy_gmem_to_smem_async<C>(A_in, B_in, dim_m, dim_n, dim_k,
                                           0 /*FIXME*/, 0 /*FIXME*/, tile_k + 1);
            }
        }

        // asynchrously kick off matmul for this tile_k
        // gemmini_fence_ready();
        matmul_tile_async<C>(tile_k, last_k);

        // update scale factors for the next tile_k
        // make sure to place this between tile_async and fence to hide latency
        if constexpr (!DISABLE_SCALE_FACTOR_UPDATE) {
            if constexpr (!SKIP_A) {
                load_scale_factors(
                    calculate_scale_factor_smem_addr<false>(tile_k + 1),
                    calculate_scale_factor_gmem_addr<C, false>(
                        A_scales, tile_k + 1, dim_m, dim_n),
                    C.SCALE_FACTORS_PER_TILE());
            }
            load_scale_factors(
                calculate_scale_factor_smem_addr<true>(tile_k + 1),
                calculate_scale_factor_gmem_addr<C, true>(
                    B_scales, tile_k + 1, dim_m, dim_n),
                C.SCALE_FACTORS_PER_TILE());

            // fence scale factor and LUT writes before next Gemmini compute
            mu_fence_smem();
        }

        gemmini_fence();

        if constexpr (barrier_tile) {
            mu_barrier(barrier_id, warps_per_threadblock);
        }
    }

    gemmini_fence();

    asm volatile ("main_matmul_k_loop_end_%=:" :: );

    asm volatile ("mxgemm_single_output_tile_end_%=:" :: );
}

/** Software-pipelining split of mxgemm_single_output_tile for a SINGLE SMEM K-tile
 *  (dim_k == TILE_K, as in FA QK^T / PV). `mxgemm_prefetch_tile` issues config + the
 *  B (and, unless SKIP_A, A) GMEM->SMEM move-in + scale loads asynchronously and returns
 *  WITHOUT waiting (no gemmini_fence), so the caller can run other work (e.g. SIMT
 *  softmax/requant) while the DMA is in flight. `mxgemm_compute_tile` then drains the DMA
 *  (leading gemmini_fence) and runs the matmul, leaving C at SPAD_DEST. thread-0 only. */
// LANE_SCALES: do the SF-SRAM scale loads with warp-0's 16 lanes instead of thread 0 alone (~16x).
// MUST be called by all threads (lanes 1..15 participate; only lane 0 issues ROCC/config/DMA).
template <GemmConfig C, bool SKIP_A = false, bool DO_CONFIG = true, bool EXPLICIT_MVIN = false,
          bool LANE_SCALES = false>
__attribute__((noinline)) void mxgemm_prefetch_tile(
        const uint8_t *A_in, const uint8_t *B_in,
        const uint8_t *A_scales, const uint8_t *B_scales,
        const uint32_t dim_m, const uint32_t dim_n, const uint32_t dim_k,
        const uint32_t tid_in_threadblock, const uint32_t tile_k = 0) {
    asm volatile ("mxgemm_prefetch_tile_start_%=:" :: );
    if constexpr (LANE_SCALES) {
        if (tid_in_threadblock >= MU_NUM_THREADS) return;   // warp 0 only (lanes 0..15)
    } else {
        if (tid_in_threadblock != 0) return;
    }
    if (tid_in_threadblock == 0) {
        uint32_t wsel = 0, asel = 0;
#ifdef FA_NOSCALES
        // host prefilled the PV (SKIP_A) weight scales into weight-scale buffer 1
        wsel = SKIP_A ? 1u : 0u;
        // ...and the QK (!SKIP_A) ACTIVATION scales into activation buffer 1, leaving activation
        // buffer 0 to the GPU's runtime P scales (pack_scales_to_sfmem -> GEMMINI_SF_MEM_A).
        // Without this the two collide across FA_STEADY iterations (see g_host_sf_asel comment).
#ifdef FA_HOST_ASEL0
        asel = 0u;                      // DIAGNOSTIC: see FA_HOST_ASEL0 in host.cpp
#else
        asel = SKIP_A ? 0u : 1u;
#endif
        g_host_sf_wsel = wsel;
        g_host_sf_asel = asel;
        // NOTE: the GPU must NOT clear the mailbox flags here. It was tried and it deadlocks:
        // Rocket reaches main() at ~2.3k and finishes the QK scale subset well before the GPU
        // gets here (~15k), so a GPU-side clear wipes a signal the host had already raised.
        // The host clears the flags itself as the first thing it does in main().
#endif
        CPROF(SKIP_A ? 0x10 : 0x00);   // prefetch entry
#ifdef FA_HOSTCFG
        // Tile 0: issue the 12-command stream AND record it into cluster SMEM.  Tiles >= 1: issue
        // NOTHING -- the host replays the recording and hands over via QK_READY / V_READY, which
        // this function already waits on below.  (See the FA_HOSTCFG block near the top.)
        if (g_fa_tile == 0u) {
            volatile __shared uint32_t *rec = reinterpret_cast<volatile __shared uint32_t *>(
                SKIP_A ? FA_CFGREC_PV : FA_CFGREC_QK);
            g_cfgrec = rec + 1;                     // rec[0] is the command count, filled in after
            g_cfgrec_n = 0;
            if constexpr (DO_CONFIG) configure_mxgemmini<C>(dim_m, dim_n, dim_k, wsel, asel);
            CPROF(SKIP_A ? 0x11 : 0x01);
            // configure_mxgemmini ends in a gemmini_fence(); record it as a PSEUDO-COMMAND with
            // inst == 0 so the host's replay reproduces the same drain between the config group
            // and the move-in group (the loop-FSM bounds must be applied before LOOP_WS reads
            // them, and config_ld/config_ex go to different gemmini queues than LOOP_WS).
            if (g_cfgrec) {
                g_cfgrec[0] = 0; g_cfgrec[1] = 0; g_cfgrec[2] = 0; g_cfgrec[3] = 0; g_cfgrec[4] = 0;
                g_cfgrec += 5; g_cfgrec_n++;
            }
            copy_gmem_to_smem_async<C, SKIP_A, EXPLICIT_MVIN>(A_in, B_in, dim_m, dim_n, dim_k, 0, 0, tile_k);
            g_cfgrec = nullptr;
            rec[0] = g_cfgrec_n;
            mu_fence_smem();                        // recording visible before the ready flag
            volatile __shared uint32_t *mb = reinterpret_cast<volatile __shared uint32_t *>(FA_HOST_MBOX);
            mb[FA_MBOX_CFGREC / 4] = SKIP_A ? 2u : 1u;
            mu_fence_smem();
        } else {
            CPROF(SKIP_A ? 0x11 : 0x01);
        }
#else
        if constexpr (DO_CONFIG) configure_mxgemmini<C>(dim_m, dim_n, dim_k, wsel, asel);
        CPROF(SKIP_A ? 0x11 : 0x01);   // after configure_mxgemmini (7 ROCC cmds + gemmini_fence)
#if defined(FA_NS_SENT) || defined(FA_NS_SENTB)
        if constexpr (!SKIP_A) fa_sent_stamp(0, C.PE_TILES_I() * C.PE_TILES_K());
#endif
#ifdef FA_NS_SENTB
        // The B operand (K^T) is 32 KB -- FOUR TIMES the A operand -- so if the move-in DMA is
        // what loses the race against the matmul, B is the more likely victim.  Its spad grows
        // DOWN from the end: b_start = b_end - PE_TILES_K*PE_TILES_J*DIM.
        fa_sent_stamp(BANK_NUM * BANK_ROWS - C.PE_TILES_K() * C.PE_TILES_J() * DIM,
                      C.PE_TILES_K() * C.PE_TILES_J());
#endif
        copy_gmem_to_smem_async<C, SKIP_A, EXPLICIT_MVIN>(A_in, B_in, dim_m, dim_n, dim_k, 0, 0, tile_k);
#if defined(FA_NS_SENT) || defined(FA_NS_SENTB)
        if constexpr (!SKIP_A) fa_sent_wait(0, C.PE_TILES_I() * C.PE_TILES_K());
#endif
#ifdef FA_NS_SENTB
        fa_sent_wait(BANK_NUM * BANK_ROWS - C.PE_TILES_K() * C.PE_TILES_J() * DIM,
                     C.PE_TILES_K() * C.PE_TILES_J());
#endif
#endif
        CPROF(SKIP_A ? 0x12 : 0x02);   // after A/B move-in issue (5 ROCC cmds, async)
    }
    // Order lane-0's ROCC/config stores AHEAD of the lanes' SF-SRAM writes: both traverse the gemmini
    // tile's flit merge node, and previously they were strictly serial (all from thread 0).
    if constexpr (LANE_SCALES) mu_fence_smem();
    PMARK(0);   // after config + DMA issue
#ifdef FA_NOSCALES
    if (false)
#endif
    if constexpr (LANE_SCALES) {
        if constexpr (!SKIP_A) {
            load_scale_factors_lanes(calculate_scale_factor_smem_addr<false>(tile_k),
                                     calculate_scale_factor_gmem_addr<C, false>(A_scales, 0, dim_m, dim_n),
                                     C.SCALE_FACTORS_PER_TILE(), tid_in_threadblock);
        }
        load_scale_factors_lanes(calculate_scale_factor_smem_addr<true>(tile_k),
                                 calculate_scale_factor_gmem_addr<C, true>(B_scales, 0, dim_m, dim_n),
                                 C.SCALE_FACTORS_PER_TILE_B(), tid_in_threadblock);
    } else {
#ifdef FA_NOSCALES
        if (false)
#endif
        if constexpr (!SKIP_A) {
            load_scale_factors(calculate_scale_factor_smem_addr<false>(tile_k),
                               calculate_scale_factor_gmem_addr<C, false>(A_scales, 0, dim_m, dim_n),
                               C.SCALE_FACTORS_PER_TILE());
        }
        load_scale_factors(calculate_scale_factor_smem_addr<true>(tile_k),
                           calculate_scale_factor_gmem_addr<C, true>(B_scales, 0, dim_m, dim_n),
                           C.SCALE_FACTORS_PER_TILE_B());
    }
#if defined(FA_NS_KRELOAD) || defined(FA_NS_PADB)
    // ============================================================================================
    // DIAGNOSTIC PAIR for the FA_NOSCALES steady-state corruption (2026-07-27).
    //
    // WHAT THE DATA SAYS (full [ISSUE]-trace forensics on /tmp/hsruns/nt6ver.out, the 6-tile
    // FULL_ATTN2 + FA_NOSCALES + FA_EARLYV run):
    //   * the bf16 P that the softmax writes is bit-identical across tiles 0,1,2 and then ALL 8192
    //     words change at tile 3, so S ITSELF is wrong -- the fault is in the QK gemm, not in
    //     softmax/requant/PV;
    //   * recovering S from P and m and projecting it onto the row space of the correct effective
    //     K matrix leaves a 22.0% residual, against a 2.3% floor.  With the verified hw mesh model
    //     (which reproduces golden_S BIT-EXACTLY) an arbitrarily corrupted A operand or A scale set
    //     leaves the residual at 2.3-2.4% -- it is mathematically confined to the row space -- while
    //     a corrupted B operand or B scale set pushes it to 27-71%.  So the corruption is on the
    //     WEIGHT (K) side of the QK gemm;
    //   * tile 4 == tile 5 exactly, and tile 3 != tile 4: the error is SELF-SUSTAINING through a
    //     P -> (something the QK gemm reads) -> S -> P loop, which reaches a fixed point in one
    //     iteration;
    //   * the only per-tile writer of the MX scale SRAM is pack_scales_to_sfmem (the runtime P
    //     scales), and the only thing FA_NOSCALES changes about the static scales is that NOTHING
    //     REWRITES THEM -- the control build reloads all 704 words every tile and is 12/12 correct.
    // Together those say: the K scales in weight half 0 stop being right, the corrupting data is
    // P-derived, and the control build survives only because it repairs them every tile.
    //
    // FA_NS_KREPAIR tests exactly that by repairing weight half 0 from GMEM every tile.
    // FA_NS_PADB is its COST-MATCHED CONTROL: the same 256 single-thread word stores at the same
    // point in the tile, to SMEM scratch instead of the scale SRAM.  Both perturb the schedule by
    // the same ~16k cycles, so
    //     KRELOAD correct + PADB still wrong  =>  the K scales really are being corrupted;
    //     both correct                        =>  it is only the extra delay (inconclusive);
    //     both wrong                          =>  the K scales are NOT the carried state.
    // ============================================================================================
    if (tid_in_threadblock == 0) {
        if constexpr (!SKIP_A) {
#ifdef FA_NS_PADB
            load_scale_factors(reinterpret_cast<volatile __shared uint32_t *>(0x14000u /*SCALE*/
                                                                             + 0x2000u),
                               calculate_scale_factor_gmem_addr<C, true>(B_scales, 0, dim_m, dim_n),
                               C.SCALE_FACTORS_PER_TILE_B());
#else
            load_scale_factors(calculate_scale_factor_smem_addr<true>(0),
                               calculate_scale_factor_gmem_addr<C, true>(B_scales, 0, dim_m, dim_n),
                               C.SCALE_FACTORS_PER_TILE_B());
#endif
        }
    }
#endif
#ifdef FA_NOSCALES
    // ---- Wait for the host's scale write to land -------------------------------------------
    // Placed HERE (after config + DMA issue, where load_scale_factors used to be) on purpose:
    // the Q/K mvin DMA is already in flight, so this spin overlaps the drain that
    // mxgemm_compute_tile's leading gemmini_fence would have paid anyway -- the QK matmul starts
    // at max(host_ready, dma_done) instead of host_ready + dma_done.
    // The spin is bounded so a mis-built (host-side prefill disabled) image fails visibly
    // instead of hanging the simulation.
    if (tid_in_threadblock == 0) {
        volatile __shared uint32_t *mbox =
            reinterpret_cast<volatile __shared uint32_t *>(FA_HOST_MBOX);
        const uint32_t slot = (SKIP_A ? FA_MBOX_V_READY : FA_MBOX_QK_READY) / 4;
        FA_HOST_MARK(SKIP_A ? 0x90 : 0x88);          // spin start
#ifdef FA_HOSTHS
        // ---- PER-TILE handshake (-DFA_HOSTHS): the host re-writes the scale halves for EVERY
        // FA tile (as a real streaming FA, where K/V change per tile, requires).  The flag word is
        // then a monotone SEQUENCE NUMBER = "#tiles whose scales are resident", not a magic value.
        // Tile t may start its QK once the host has published t+1.
        const uint32_t need = g_fa_tile + 1u;
        uint32_t spin = 0;
        for (; spin < 400000u; spin++) { if (mbox[slot] >= need) break; }
        if (SKIP_A) g_fa_spin_v += spin; else g_fa_spin_qk += spin;
#else
        for (uint32_t spin = 0; spin < 200000u; spin++) {
            if (mbox[slot] == FA_HOST_MAGIC) break;
        }
#endif
        FA_HOST_MARK(SKIP_A ? 0x94 : 0x8c);          // spin end
    }
#endif
    PMARK(1);   // after A+B scale loads
    CPROF(SKIP_A ? 0x13 : 0x03);   // after scale loads / host-scale wait
    if (tid_in_threadblock == 0) load_lut<C>();
    PMARK(2);   // after load_lut
    mu_fence_smem();      // order the SIMT scale/LUT stores; DMA stays in flight (no fence)
    CPROF(SKIP_A ? 0x14 : 0x04);   // prefetch end
    asm volatile ("mxgemm_prefetch_tile_end_%=:" :: );
}

// ---- CISC QK path (option 2): issue the SAME loop_ws matmul via the Gemmini CISC
// microcode engine (csrw 0xacc) WITHOUT muon fences between commands. MX scales are read
// from SF_MEM (populated by load_scale_factors, as usual) -- RTL-confirmed identical.
// Hexadeciles (spadHexadecile = BANK_NUM*BANK_ROWS/16 = 512 rows): Q(A)->hex 0 (byte 0x0),
// S(C)->hex 2 (byte 0x4000 == S_SMEM, where softmax reads), K(B)->hex 15 (top, B end=8192).
// On the muon core, csrw 0xacc is an illegal CSR (that path is Vortex-only). CISC commands
// are issued via MMIO to the gemmini CISC command register at GEMMINI_CTRL + 0x30.
#define FA_CISC_CMD(x) store_shared(GEMMINI_CTRL, 0x30, (uint32_t)(x))
enum { FA_CISC_COMPUTE_AND_STORE_TO_SPAD = 1, FA_CISC_SET_AB_STRIDE = 8,
       FA_CISC_LOAD_TO_HEXADECILES = 10 };
template <GemmConfig C>
__attribute__((noinline)) void mxgemm_cisc_qk(const uint8_t *Q_in, const uint8_t *K_in,
                                              const uint8_t *A_scales, const uint8_t *B_scales,
                                              const uint32_t dim_m, const uint32_t dim_n,
                                              const uint32_t dim_k, const uint32_t tid) {
    asm volatile ("mxgemm_cisc_qk_start_%=:" :: );
    if (tid != 0) return;
    constexpr uint32_t Q_HEX = 0, K_HEX = 15, S_HEX = SPAD_DEST / ((BANK_NUM * BANK_ROWS) / 16);
    // MX scales -> SF_MEM (mesh reads these during the CISC-issued loop_ws matmul).
    load_scale_factors(calculate_scale_factor_smem_addr<false>(0),
                       calculate_scale_factor_gmem_addr<C, false>(A_scales, 0, dim_m, dim_n),
                       C.SCALE_FACTORS_PER_TILE());
    load_scale_factors(calculate_scale_factor_smem_addr<true>(0),
                       calculate_scale_factor_gmem_addr<C, true>(B_scales, 0, dim_m, dim_n),
                       C.SCALE_FACTORS_PER_TILE());
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), 0, 0, QUANT_LUT_UPDATE_GRANULARITY);
    mu_fence_smem();
    // GMEM base addresses for A(Q) and B(K) tiles (device->host address space).
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
        rad_device_to_host_address(reinterpret_cast<uint32_t>(Q_in)),
        rad_device_to_host_address(reinterpret_cast<uint32_t>(K_in)), k_LOOP_WS_CONFIG_ADDRS_AB);
    // SET_AB_STRIDE: A row stride = dim_k, B row stride = dim_n (elements). [n<<20 | k<<8 | op]
    FA_CISC_CMD((dim_n << 20) | (dim_k << 8) | FA_CISC_SET_AB_STRIDE);
    // LOAD_TO_HEXADECILES: DMA A->Q_HEX, B->K_HEX.  [b_hex<<16 | a_hex<<8 | op]
    FA_CISC_CMD((K_HEX << 16) | (Q_HEX << 8) | FA_CISC_LOAD_TO_HEXADECILES);
    // COMPUTE_AND_STORE_TO_SPAD: S = Q@K^T -> S_HEX (bf16). [d_hex<<24 | b_hex<<16 | a_hex<<8 | op]
    FA_CISC_CMD((S_HEX << 24) | (K_HEX << 16) | (Q_HEX << 8) | FA_CISC_COMPUTE_AND_STORE_TO_SPAD);
    gemmini_fence();
    asm volatile ("mxgemm_cisc_qk_end_%=:" :: );
}

// Issue-only: leading move-in drain + config + matmul issue, NO trailing fence.
// For measuring issue-cost vs drain-cost, and for async producer overlap.
template <GemmConfig C>
__attribute__((noinline)) void mxgemm_compute_issue(const uint32_t tid_in_threadblock,
                                                    const uint32_t c_spad_dest = SPAD_DEST) {
    if (tid_in_threadblock != 0) return;
    gemmini_fence();      // drain prior move-in DMA
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        0, 0, QUANT_LUT_UPDATE_GRANULARITY);
    matmul_tile_async<C>(0, /*acc_move_out=*/true, /*accumulate=*/false,
                         /*b_spad_override=*/0xffffffffu, /*c_spad_dest=*/c_spad_dest);
}

// GMEM-mvout variant: matmul C into the ACCUMULATOR (acc_move_out=false, NO SMEM store), then issue the
// accumulator->GMEM mvout ASYNC (no trailing fence). Avoids the SMEM mvout 16-subbank atomic-grant race so the
// mesh can overlap SIMT softmax. Caller drains later via gemmini_fence, then SIMT-copies GMEM->SMEM.
template <GemmConfig C>
__attribute__((noinline)) void mxgemm_compute_issue_gmem(const uint32_t tid_in_threadblock,
                                                         uint8_t *gmem_dest, const uint32_t dim_n) {
    if (tid_in_threadblock != 0) return;
    gemmini_fence();      // drain prior move-in DMA
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), 0, 0, QUANT_LUT_UPDATE_GRANULARITY);
    // matmul: result stays in accumulator (acc_move_out=false => skip_stc=1, no SMEM store), overwrite.
    matmul_tile_async<C>(0, /*acc_move_out=*/false, /*accumulate=*/false,
                         0xffffffffu, SPAD_DEST, 0xffffffffu, /*force_first=*/1);
    // issue accumulator->GMEM mvouts (async, NO fence). Mirrors copy_accmem_to_gmem_dma_sync minus the fence.
    for (int i = 0; i < C.PE_TILES_I(); i++) {
#pragma unroll 32
        for (int j = 0; j < C.PE_TILES_J() / 4; j++) {
            const uint32_t tile_acc_addr = GEMMINI_ACC_ADDR + (i * C.PE_TILES_J() / 4 + j) * DIM;
            uint8_t *dram_ptr = gmem_dest + (i * DIM * dim_n + j * DIM * 2) * C.OUT_ELEM_SIZE();
            gemmini_mvout(rad_device_to_host_address(reinterpret_cast<uint32_t>(dram_ptr)), tile_acc_addr);
        }
    }
}

// Compute-only: matmul C into the ACCUMULATOR (acc_move_out=false, skip_stc=1), NO mvout, NO trailing fence.
// The mesh uses only read-ports + the private accumulator (off-SMEM) -> safe to overlap SIMT softmax.
// Caller MUST later drain (gemmini_fence) then issue mxgemm_store_acc_to_spad in a SIMT-quiet window.
template <GemmConfig C>
__attribute__((noinline)) void mxgemm_compute_issue_acc(const uint32_t tid_in_threadblock) {
    if (tid_in_threadblock != 0) return;
    gemmini_fence();      // drain prior move-in DMA
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), 0, 0, QUANT_LUT_UPDATE_GRANULARITY);
    matmul_tile_async<C>(0, /*acc_move_out=*/false, /*accumulate=*/false,
                         0xffffffffu, SPAD_DEST, 0xffffffffu, /*force_first=*/1);
}

// Store-only: move the accumulator (result of a prior compute_issue_acc) -> SPAD (c_spad_dest) via a loop_ws
// with skip_lda=skip_ldb=skip_ldd=skip_ex=1, skip_stc=0 (store C only, no recompute). This is the cheap
// accmem->SMEM path (512b spad_writer, contends SMEM write-ports) -- issue it ONLY when SIMT is quiesced
// (post-barrier) so the 16-subbank atomic grant settles with no muon writers => no race/hang. Trailing fence.
template <GemmConfig C>
__attribute__((noinline)) void mxgemm_store_acc_to_spad(const uint32_t tid_in_threadblock,
                                                        const uint32_t c_spad_dest = SPAD_DEST) {
    if (tid_in_threadblock != 0) return;
    const uint32_t a_spad = calculate_spad_addr<false>(0);
    const uint32_t b_spad = calculate_spad_addr<true>(0);
    gemmini_loop_ws_spad(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), 0, 0, 0,
        a_spad, b_spad, 0, c_spad_dest,
        false, false, false, false, /*ex_accumulate=*/false,
        NO_ACTIVATION, 0, 0, false,
        loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1, /*skip_ex=*/1, /*skip_stc=*/0));
    gemmini_fence();
}

// FenceMode for the compute-tile drains. NONE: skip the gemmini_fence polls entirely (for the SKIP_A
// PVF path where the occupancy MMIO is phantom-poisoned per FSDB H8 -> ALL status polls livelock, but
// io_cmd_ready is healthy so the matmul still ISSUES; the mvin DATA is already resident and the caller
// guarantees drain via a fence_delay/barrier). READY: poll cmd-ready. BUSY (default): poll busy.
enum class FenceMode { BUSY, READY, NONE };
template <GemmConfig C, FenceMode FM = FenceMode::BUSY>
__attribute__((noinline)) void mxgemm_compute_tile(const uint32_t tid_in_threadblock,
                                                   const uint32_t c_spad_dest = SPAD_DEST,
                                                   const uint32_t a_spad_override = 0xffffffffu,
                                                   const uint32_t b_spad_override = 0xffffffffu,
                                                   const uint32_t tile_k = 0) {
    asm volatile ("mxgemm_compute_tile_start_%=:" :: );
    if (tid_in_threadblock != 0) return;
    const uint32_t odd = tile_k & 1u;
    uint32_t wsel = odd, asel = odd;
#ifdef FA_NOSCALES
    // Host-prefilled scales: the weight-scale half is chosen by which gemm this is (QK -> 0,
    // PV -> 1), not by tile_k parity. Set by the matching mxgemm_prefetch_tile on this thread.
    // Likewise the ACT half (QK -> 1 = host QK_A scales, PV -> 0 = GPU-packed P scales).
    wsel = g_host_sf_wsel;
    asel = g_host_sf_asel;
#endif
    CPROF(wsel ? 0x20 : 0x28);   // compute entry
#if defined(FA_HOSTPACK) && defined(FA_HOSTHS)
    // ---- hand the runtime P scales to the host (-DFA_HOSTPACK) --------------------------------
    // Reached at the top of the PV gemm (wsel == 1), i.e. after requant has filled SCALE_SMEM and
    // instead of the GPU's own pack_scales_to_sfmem (compiled out by -DFA_NOPACK).  Strictly serial
    // by design: the GPU publishes the request and blocks, so it is provably not touching the
    // gemmini tile while the host drives the scale SRAM.  Bounded so a mis-built image fails
    // visibly rather than hanging the simulation.
    if (wsel == 1u) {
        volatile __shared uint32_t *mbox =
            reinterpret_cast<volatile __shared uint32_t *>(FA_HOST_MBOX);
        mbox[FA_MBOX_GPU_PACKREQ / 4] = g_fa_tile + 1u;
        mu_fence_smem();
        const uint32_t need = g_fa_tile + 1u;
        for (uint32_t spin = 0; spin < 400000u; spin++) {
            if (mbox[FA_MBOX_HOST_PACKED / 4] >= need) break;
        }
    }
#endif
#if defined(FA_NS_SETTLE) || defined(FA_NS_PAD) || defined(FA_NS_SETTLE_LONG)
    // ---- FENCE-TOO-EARLY GUARD (-DFA_NS_SETTLE / -DFA_NS_PAD) ---------------------------------
    // gemmini_fence() is `while (load32_shared(GEMMINI_BUSY_ADDR) != 0) nop;`
    // (lib/include/mxgemmini_mmio.h:74-78) -- it drains only what is ALREADY visible as busy.  The
    // operand move-in was pushed into the command port a handful of cycles earlier, so if `busy`
    // has not risen yet the first poll reads 0 and the fence returns having drained NOTHING; the
    // mesh then multiplies a partially loaded operand tile.  A cold tile hides this (the
    // icache-cold prefetch is slow enough that busy is up by fence time), which is why it shows up
    // only in the STEADY state -- and only once -DFA_NOSCALES removes the 12.6k-cycle
    // load_scale_factors that used to sit between the move-in issue and this fence.
    //   FA_NS_SETTLE: bounded wait for busy to ASSERT, then the normal drain (~16 MMIO polls).
    //     *** DO NOT USE -- MEASURED 2026-07-27: THIS FLAG $finishes.  The bounded
    //     `for (i<N) if (busy) break;` shape overflows the warp scheduler's IPDOM stack here
    //     ("Assertion failed: ipdom stack is full", WarpScheduler.scala:456) at ~112k cycles,
    //     i.e. inside tile 0, on FULL_ATTN2 FA_NOSCALES FA_EARLYV FA_NT6.  That is why this flag
    //     has never appeared in any results table.  Use the UNCONDITIONAL dependent MMIO chain
    //     (fa_cfg_settle / FA_NS_FENCE at the top of this file) instead -- no divergence region. ***
    //   FA_NS_PAD:    a blunt fixed spin instead, purely to confirm the diagnosis.
    {
#ifdef FA_NS_PAD
        for (uint32_t i = 0; i < 3000u; i++) asm volatile("nop");
#elif defined(FA_NS_SETTLE_LONG)
        // Same idea as FA_NS_SETTLE but with a bound long enough that "busy never asserted"
        // means the command really has already retired, not "I gave up too early".  It costs
        // NOTHING when busy comes up promptly, so it is the shippable form.
        for (uint32_t i = 0; i < 3000u; i++) {
            if (load32_shared(GEMMINI_BUSY_ADDR) != 0) break;
        }
#else
        for (uint32_t i = 0; i < 16u; i++) {
            if (load32_shared(GEMMINI_BUSY_ADDR) != 0) break;
        }
#endif
    }
#endif
    if constexpr (FM == FenceMode::READY) gemmini_fence_ready();
    else if constexpr (FM == FenceMode::BUSY) gemmini_fence();
    CPROF(wsel ? 0x21 : 0x29);   // after LEADING drain (waits for the move-in DMA)
    // FM==NONE: no leading drain (V-mvin data already resident; the poisoned occupancy MMIO would livelock).
#ifdef FA_CFGSETTLE_PRE
    fa_cfg_settle();   // control placement: settle BEFORE the config, i.e. give the PREVIOUS
                       // gemm's scale reads time to finish under the OLD loop bounds.
#endif
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        asel, wsel, QUANT_LUT_UPDATE_GRANULARITY);   // A/B double-buffer parity MUST match tile_k
#if defined(FA_CFGSETTLE_AFTER) || defined(FA_CFGSETTLE_PRE)
#else
    fa_cfg_settle();   // FA_CFGSETTLE: the mesh's scale-SRAM read row depends on this command's
                       // loop_bound_j, and LOOP_WS is not ordered against it.  See the header.
#endif
    CPROF(wsel ? 0x22 : 0x2a);   // after CONFIG_SCALE_MEM
    matmul_tile_async<C>(tile_k, /*acc_move_out (last_k)=*/true, /*accumulate=*/false,
                         b_spad_override, c_spad_dest, a_spad_override, /*force_first=*/1);
#ifdef FA_CFGSETTLE_AFTER
    fa_cfg_settle();   // control placement -- provably too late to help
#endif
    CPROF(wsel ? 0x23 : 0x2b);   // after matmul ISSUE (the mesh work starts here)
    if constexpr (FM == FenceMode::READY) gemmini_fence_ready();
    else if constexpr (FM == FenceMode::BUSY) gemmini_fence();
    CPROF(wsel ? 0x24 : 0x2c);   // after TRAILING drain (mesh + mvout complete)
    // FM==NONE: caller must drain the PV mvout (fence_delay/barrier) before reading SPAD_DEST.
#if defined(FA_NOSCALES) && defined(FA_HOSTHS)
    // ---- publish GPU progress so the host knows which scale halves are dead ------------------
    // Reached only after the trailing gemmini_fence, i.e. the mesh has finished reading this
    // gemm's scale halves.  QK done  -> the host may refill act half 1 (QK_A) + weight half 0 (K).
    //                      PV done  -> the host may refill weight half 1 (V).
    // Separate words from the host's own flags, so there is no shared-word race.
    {
        volatile __shared uint32_t *mbox =
            reinterpret_cast<volatile __shared uint32_t *>(FA_HOST_MBOX);
        if (wsel == 0u) {                            // QK
            mbox[FA_MBOX_GPU_QKDONE / 4] = g_fa_tile + 1u;
        } else {                                     // PV -- last mesh op of the FA tile
            mbox[FA_MBOX_GPU_PVDONE / 4] = g_fa_tile + 1u;
            g_fa_tile++;
        }
        mu_fence_smem();
    }
#endif
#if defined(FA_NOSCALES) && defined(FA_HOST_TIMING)
    // Instrumentation only: echo the host's diagnostic block (mailbox +0x80..+0xbf) out to
    // FA_DIAG_HOSTDG so the host-side prefill/handshake timeline is readable in the same trace
    // as the GPU's mcycle marks.  The host cannot write GMEM the verifier parses, and the GPU
    // cannot read Rocket's registers, so the SMEM mailbox is the only bridge.
    // Word map (see host.cpp): 0 t_enter, 1 t_prefill_done, 2 tiles served, 3 last-tile stamp,
    //   4 total host store cycles, 5 total host wait-on-GPU cycles, 6 qk-wait timeouts,
    //   7 pv-wait timeouts, 8 last GPU_QKDONE seen, 9 last GPU_PVDONE seen, 10 GMEM-probe value.
    if (wsel) {
        const volatile __shared uint32_t *ts =
            reinterpret_cast<const volatile __shared uint32_t *>(FA_HOST_MBOX + FA_MBOX_DIAG);
        // Opaque base register (see the diagnostic map at the top of this file): with a plain
        // constant base clang folds %lo into the store immediate, all 19 stores are traced with
        // rs1.data = 0x40050000, and they overwrite the kernel's m[0..18].  That corrupted the
        // FA_STEADY mark trace of every -DFA_HOST_TIMING build.
        // ...and the base has to be re-opaqued PER STORE.  Making only the loop base opaque stops
        // clang folding %lo(addr) into the immediate but it still emits `sw rX, 4*i(base)`, so all
        // 19 stores are traced with the SAME rs1.data and only the last survives the parse
        // (measured: the whole diag block came back as one word).
        volatile uint32_t *dst = (volatile uint32_t *)FA_DIAG_HOSTDG;
        for (int i = 0; i < 19; i++) {
            uint32_t v = 0;
            if (i < 16) v = ts[i];
#ifdef FA_HOSTHS
            else if (i == 16) v = g_fa_spin_qk;   // GPU cycles spun waiting for host QK scales
            else if (i == 17) v = g_fa_spin_v;    // ... and for host V scales
            else             v = g_fa_tile;
#endif
            volatile uint32_t *p = dst + i;
            asm volatile("" : "+r"(p));
            *p = v;
        }
    }
#endif
    asm volatile ("mxgemm_compute_tile_end_%=:" :: );
}

/** Do a full GEMM and store the result C tensor at `C_gmem` GMEM address. */
template <GemmConfig C>
static void
mxgemm(const uint8_t *A_in, const uint8_t *B_in,
       const uint8_t *A_scales, const uint8_t *B_scales,
       const uint32_t dim_m, const uint32_t dim_n, const uint32_t dim_k,
       uint8_t *C_gmem, const uint32_t tid_in_threadblock,
       const uint32_t threads_per_threadblock, const uint32_t threadblock_id) {
    mxgemm_single_output_tile<C>(A_in, B_in, A_scales, B_scales,
                                 dim_m, dim_n, dim_k, tid_in_threadblock,
                                 threads_per_threadblock);

    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    mu_barrier(1, warps_per_threadblock);

    // Move-out C from SMEM to GMEM
    if constexpr (!DISABLE_GMEM_MOVE_OUT) {
        auto C_smem =
            reinterpret_cast<const __shared uint8_t *>(SPAD_DEST * DIM);
        if constexpr (SIMT_GMEM_MOVE_OUT) {
            copy_smem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                                   C.OUT_ELEM_SIZE()>(
                C_smem, C_gmem, tid_in_threadblock, threads_per_threadblock);
        } else {
            // copy_accmem_to_gmem_dma_sync<C>(C_gmem, dim_n, tid_in_threadblock);
            copy_C_smem_to_gmem_dma_sync<C>(SPAD_DEST, C_gmem, dim_n,
                                            tid_in_threadblock);

            mu_barrier(2, warps_per_threadblock);

            // we do not trace DMA move; do an additional bogus SIMT copy to
            // generate verifiable trace
            auto trace_gmem = reinterpret_cast<uint8_t *>(0x60000000);
            copy_gmem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                                   C.OUT_ELEM_SIZE()>(C_gmem, trace_gmem,
                                                      tid_in_threadblock,
                                                      threads_per_threadblock);
        }
    }
}
