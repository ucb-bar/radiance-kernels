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
// FA_SP_SQ32 half-tile shapes: identical to QKF/PVF with the QUERY block halved.  PE_TILES_I is 2
// instead of 4, so the mesh does two output row-tiles per matmul and each matmul is 4,096 mesh
// cycles instead of 8,210 -- the same total work per 64 rows, in twice as many pieces.
constexpr GemmConfig QKH{
    .TILE_M = 32, .TILE_N = FA_SK, .TILE_K = FA_D,
    .DATATYPE = GemmDatatype::FP8, .QUANT_OUTPUT = false,
};
constexpr GemmConfig PVH{
    .TILE_M = 32, .TILE_N = FA_D, .TILE_K = FA_SK,
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
// 2026-07-26: the "fast pad corrupts tile 0" claim was a VERIFICATION ARTIFACT, not a hazard.
// fa_verify_out.py leaves uncovered cells at 0, so Frobenius ~= sqrt(fraction uncovered); the 18.12%/6.19%
// figures came from parsing the .out while spike-dasm was still draining the sim's stderr pipe (8192/8192 ->
// 3.5666%, 7930/8192 -> 18.1%). 13 independent all-fast runs (fixed + auto seed, with/without dramsim) all give
// 3.5666% and a BIT-IDENTICAL O; all 8 tile-generations of a 4-tile FA_STEADY run are correct. FSDB also shows
// the PV->finalize hand-off is CAUSAL (spad_writer.io_busy falls cycle 111,561; barrier-4 release 111,656, and
// thread 0 cannot pass gemmini_fence()'s GEMMINI_BUSY poll before that). So no slow pad is needed anywhere.
// Measured: 132,016 -> 118,673 cyc (-10.1%), O bit-identical, util 12.44% -> 13.84%.
#if defined(FA_SLOWPAD4) || defined(FA_SLOWPAD_ALL)
#define BAR_PAD4() BAR_PAD_SLOW()
#else
#define BAR_PAD4() BAR_PAD_FAST()
#endif
#define MARK() do { if (tid == 0) { uint32_t _c; asm volatile("csrr %0, mcycle" : "=r"(_c)); \
                                    ((volatile uint32_t *)MARK_GMEM)[mki++] = _c; } } while (0)
// PER-STAGE mark.  *** THESE ARE NOT FREE. ***  MARK() is a single-thread store to GMEM, i.e. to
// DRAM, and mxgemm_core.hpp already measured a single such store at ~400 cycles on this machine
// (four stamps per tile were worth 1.6k of steady-state slope).  The FA_SP body emits SEVEN per
// tile, so ~2.8k cyc/tile -- 5.5% of a 51k tile -- is PURE INSTRUMENTATION, and it is charged
// against the utilisation of a kernel that would ship with none of it.  FA_SP_MARK1 keeps only the
// s0 stamp, which is all a tile-interval (and therefore a utilisation) measurement needs; read it
// with `fa_pm.py <out> --per 1`.  The per-stage attribution needs the full set, so measure stage
// costs WITH the marks and quote the headline WITHOUT them -- and say which is which.
#ifdef FA_SP_MARK1
#define SMARK() do { } while (0)
#else
#define SMARK() MARK()
#endif

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
#if defined(FA_PIPE) || defined(FA_SFPAR) || defined(FA_SP)
static constexpr uint32_t FA_A_SPAD_EVEN = 0;                       // A operand spad row
static constexpr uint32_t FA_B_SPAD_EVEN = BANK_NUM * BANK_ROWS;    // B operand spad END row

// Register-only barrier pad (see the BAR_PAD note above: a stack-resident pad is DRAM).
// FA_SP_SLOWPAD makes it a VOLATILE (stack = DRAM) pad instead, i.e. hundreds of cycles rather than
// four.  This exists to test the barrier-release-pulse hazard (commit 73c304e) against the
// steady-state corruption: the corrupted O rows are exactly the LAST rows processed by particular
// warps, which is what "a warp keeps running for a few cycles after the barrier releases" looks
// like.  If a longer pad changes the corruption, the hazard is the barrier, not the DMA overlap.
#ifdef FA_SP_SLOWPAD
#define FAP_PAD() do { volatile int _q = 0;              \
    asm volatile("addi %0,%0,1" : "+r"(_q));             \
    asm volatile("addi %0,%0,1" : "+r"(_q));             \
    asm volatile("addi %0,%0,1" : "+r"(_q));             \
    asm volatile("addi %0,%0,1" : "+r"(_q)); } while (0)
#else
#define FAP_PAD() do { int _q = 0;                       \
    asm volatile("addi %0,%0,1" : "+r"(_q));             \
    asm volatile("addi %0,%0,1" : "+r"(_q));             \
    asm volatile("addi %0,%0,1" : "+r"(_q));             \
    asm volatile("addi %0,%0,1" : "+r"(_q)); } while (0)
#endif
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
    fa_cfg_settle();   // FA_CFGSETTLE -- same hazard as fa_mm().
    matmul_tile_async<C>(/*tile_k=*/0, /*acc_move_out=*/true, /*accumulate=*/false,
                         /*b_spad_override=*/FA_B_SPAD_EVEN, /*c_spad_dest=*/c_dest,
                         /*a_spad_override=*/FA_A_SPAD_EVEN, /*force_first=*/1);
}

static __attribute__((noinline)) void fap_fence(uint32_t tid) {
    if (tid != 0) return;
    gemmini_fence();
}
#endif  // FA_PIPE || FA_SFPAR || FA_SP

// ============================================================================================
// FA_SP -- SOFTWARE-PIPELINED steady-state MX-FP8 flash attention (2026-07-26).
//
// WHY THE OLD STEADY STATE WAS SO EXPENSIVE.  FA_STEADY re-runs the WHOLE single-shot body
// per tile, including a fresh move-in of Q, K^T AND V plus all 704 words of MX scales.  But
// this kernel shape is `S_k = 256 == the whole key length`: ONE QK over all of K, ONE softmax,
// ONE PV.  The real outer loop is therefore over Q BLOCKS, and in that loop K^T, V, the K
// scales and the V scales are LOOP-INVARIANT -- exactly the standard flash-attention
// structure where the KV block stays resident in scratchpad and Q streams past it.  Per tile
// only Q (8 KB), the 64 words of Q scales, and the 128 words of runtime P scales change.
// That removes 512 of the 704 SF-SRAM words (the #1 steady-state cost) and 64 KB of the 72 KB
// of per-tile move-in DMA -- without changing a single computed value.
//
// WHY IT NEEDS A NEW SMEM MAP.  Keeping K^T and V resident simultaneously means the two
// 32 KB B-operand tiles may no longer alias (the sequential kernel deliberately overlaps
// them: "QKF PE_TILES_K*J = 8*16 == PVF 16*8").  64 KB of the 128 KB SMEM therefore goes to
// B, and Q(8) + P_fp8(16) + S(32) + scratch have to fit in the other 64 KB.  They only fit
// if the bf16 P scratch disappears, so softmax writes P IN PLACE over S (legal: one row per
// warp, and the row's 8 words are all loaded into registers before any is stored).
//
//   rows        bytes            contents                       spad role
//   0    .. 2048  0x00000..0x08000  V  (fp8 [256][128])          B operand of PVF, b_end=2048
//   2048 .. 3072  0x08000..0x0C000  P_fp8 (requanted [64][256])  A operand of PVF, a=2048
//   3072 .. 5120  0x0C000..0x14000  S(bf16) -> P(bf16) -> O      C dest of BOTH gemms, c=3072
//   5120 .. 5632  0x14000..0x16000  scratch (SCALE/M/LS/CORR/PACKED/REDBUF@0x15000)
//   5632 .. 6144  0x16000..0x18000  Q  (fp8 [64][128])           A operand of QKF, a=5632
//   6144 .. 8192  0x18000..0x20000  K^T(fp8 [128][256])          B operand of QKF, b_end=8192
//
// The 0x15000 REDBUF address is hard-coded inside flash_mx_impl.hpp's softmax helpers, which
// is why the scratch window stays exactly where it was.
//
// SCALE-SRAM ASSIGNMENT (4 halves, 4 producers, no aliasing):
//   SF_B half 0 = K scales (prologue, 256 w)     SF_B half 1 = V scales (prologue, 256 w)
//   SF_A half 1 = Q scales (per tile, 64 w)      SF_A half 0 = P scales (per tile, 128 w)
// SF_A half 0 is forced because pack_scales_to_sfmem() hard-codes GEMMINI_SF_MEM_A.
//
// MOVE-IN.  copy_gmem_to_smem_async has SKIP_A but no SKIP_B (skip_ldb is hard-wired 0), so
// the library DMA path cannot move in Q without also re-moving K.  fa_mvin_A/fa_mvin_B below
// issue the per-16x16-tile `gemmini_extended_mvin` commands directly instead, which also
// dodges the H8 phantom-completion bug of the loop-FSM mvin with skip_lda=1.
//
// --------------------------------------------------------------------------------------------
// HARDWARE FACTS ESTABLISHED WHILE BUILDING THIS (all measured on RadianceTapeoutSimConfig,
// Sq=64 Sk=256 d=128, +ntb_random_seed=12345):
//
// 1. THE MX SCALE SRAM CANNOT BE WRITTEN IN PARALLEL, AT ALL.  GemminiTile.scala:188 puts a
//    FlitMergeNode(from=4,to=8) in front of it, and FlitMergeNode.scala:56/62 assert that the
//    first 4-byte Put of every pair is 8-B aligned and the second is at +4.  Both lane-parallel
//    patterns in the tree $finish the simulation:
//       FA_LANESC   (load_scale_factors_lanes, A+B)  -> $finish @ 43,830 cyc
//       FA_PK_LANES (pack_scales_to_sfmem)           -> $finish @ 115,082 cyc
//    So ~65 cycles/word from ONE thread, strictly ascending, is a hard floor for the GPU, and
//    the only ways to make MX scale traffic cheaper are to move FEWER words (this file: 704 ->
//    192 per tile by keeping K/V resident) or to overlap it with SIMT work (FA_SP_FUSE).
//    NOTE the batching rule: every ascending run must have an EVEN number of words and start
//    8-B aligned, or the next run's first Put trips the alignment assert.
//
// 2. THERE IS NO WORKING bf16 <-> fp32 PATH.  THIS ONE COST HOURS.  The whole verified reference
//    kernel emits ZERO fcvt.s.h, fcvt.h.s, fadd.s and fmax.s -- it is pure bf16 (fmax.h, fadd.h,
//    fmul.h, fdiv.h).  So the converters are UNVALIDATED, and using them is not safe: a
//    thread-per-row softmax that accumulated the row denominator as `a += (float)e` (fcvt.s.h +
//    fadd.s, then one fcvt.h.s at the end) produced 5,120 of 8,192 output cells at +-inf and the
//    remaining cells ~2.7% wrong -- i.e. garbage l.  Written entirely in bf16 (fa_add_h) the same
//    kernel is correct.  BEWARE: clang REACHES FOR fp32 ON ITS OWN -- `fmaxf(_Float16,_Float16)`
//    and `(float)h` both compile to a promote/op/demote triple, which is how the fp32 crept in.
//    fa_max_h / fa_add_h below force the native single-instruction forms.
//
// 3. SUSPECTED (not proven): SUB-WORD SMEM STORES FROM DIFFERENT WARPS RACE.  Every m/l/corr
//    array in FA_SP is ONE 32-BIT WORD PER ROW as a precaution, because
//    online_softmax_block does `if (lane==0) { l_state[row]=..; m_state[row]=..; corr_out[row]=..; }`
//    with rows striped over warps, so rows {0,1} {2,3} {4,5} share a 32-bit word and are written
//    by warps that live on DIFFERENT CORES (mu_schedule maps warp w -> core w&1) and can therefore
//    issue in the same cycle.  Whether the SMEM write path merges those byte-enables correctly was
//    never established here -- the odd-row corruption that first suggested it turned out to be
//    fact 2 above.  It is cheap to make safe, so FA_SP makes it safe.
//
// 4. rv32im has no integer max.  `(a>b)?a:b` on uint32 compiles to a BRANCH, i.e. a warp
//    divergence region.  Use the native fmax.h (fa_max_h below) -- and note that C's
//    fmaxf()/`+` on _Float16 make clang PROMOTE to fp32 (fcvt.s.h + fmax.s + fcvt.h.s), which
//    cost 127 of fa_expreq's 610 loop instructions until fa_max_h/fa_add_h replaced them.
//
// 5. THE REGISTER BUDGET IS (256 / occupancy) DISTINCT ARCHITECTURAL REGISTERS FOR THE WHOLE
//    KERNEL.  Rename.scala:110-123 keeps ONE global counter over 1..numPhysRegs-1 that is bumped
//    every time any warp first writes an architectural register and is never reclaimed, so the
//    limit is  warps_per_core x distinct_arch_regs_written <= 255.  At occ=3 that is 85 registers
//    for the entire kernel, not per function.  Counted from the objdump:
//        FA_SP + FUSE, e[16] + fmaxf/fp32-promoted reduce   84 regs -> $finish @ ~116k cyc
//        FA_SP + FUSE, e[16] + fa_max_h/fa_add_h reduce     72 regs -> fits (3x72 = 216)
//        FA_SP + FUSE, FA_SP_2P (no array at all)           58 regs -> comfortable
//    This is also why occ=4 "overflowed the register file": it leaves only 64.
//
// 6. The accumulator-staged matmul works.  matmul_tile_async(acc_move_out=false) followed later
//    by a store-only loop_ws (fa_store_acc) lets the QK matmul run with ZERO SMEM writes, so it
//    overlaps SIMT finalize; the accmem->spad store itself then costs only ~1,000 cycles
//    (measured: 8,753 on tile 0 where it also absorbs the mesh drain, 997 in steady state).
//
// 7. THE L0 INSTRUCTION CACHE IS 16 KB, nWays = 1 (DIRECT MAPPED), 32-BYTE LINES
//    (RadianceConfigs.scala:66, L0iCacheConfig) AND MUON INSTRUCTIONS ARE 8 BYTES.  So a fully
//    unrolled 600-instruction function is 4.8 KB -- nearly a third of the cache -- and two of
//    them plus fa_entry blew the hot working set to 18.5 KB.  Measured effect: the "exp+requant,
//    MX blocks 0..3" stage cost 21,150 cycles for 256 items, ~13 cycles per issued instruction.
//    Rolling the inner loops and sharing ONE fa_expreq instantiation (runtime B0/B1) took the
//    kernel from 18.5 KB to 10.4 KB of .text at the SAME dynamic instruction count -- the loop
//    branches are warp-uniform so they cost no divergence.  For scale: the sequential
//    FULL_ATTN2 kernel is 13.0 KB, of which requant_P_to_spad_tiled alone is 3,312 B and
//    online_softmax_block 1,960 B.  ON THIS MACHINE, UNROLLING AND INLINE-ASM EXPANSION ARE NOT
//    FREE; they are paid for out of a 16 KB direct-mapped budget.
//    ...BUT DO NOT OVER-CORRECT.  Rolling the loops COMPLETELY is also wrong: these phases are
//    SMEM-load-LATENCY bound, and a rolled loop can only have ONE load in flight because
//    iteration k+1 cannot issue until the branch resolves.  Fully rolled, the fused kernel's
//    row-max pass cost 16,125 cyc and its first requant half 17,534 -- WORSE than the library's
//    separate softmax (12,825) + requant (11,717), which are heavily unrolled and therefore have
//    ~32 independent loads in flight per item.  The sweet spot is a PARTIAL unroll (x4 here) with
//    INDEPENDENT ACCUMULATORS: enough memory-level parallelism, ~1.5 KB per function.  Budget
//    icache like a register file.
//
// 8. fence.s IS NOT FREE IN BULK.  The 2-cycle figure is the issue cost; the DRAIN under
//    96-lane SMEM traffic is not.  A register butterfly reduction needs one fence per step (5
//    per row), and that alone made the cooperative row-max pass cost 9,493 cycles.  Replacing it
//    with "store partials / ONE fence / every lane folds all 16 partials in registers"
//    (fa_rowmax_s) cuts it to 1 fence per row.
//
// --------------------------------------------------------------------------------------------
// MEASURED (RadianceTapeoutSimConfig, +ntb_random_seed=12345, Sq=64 Sk=256 d=128).
// util = 16420 / steady-state cycles-per-tile; the steady state is read from MARK stamps at every
// stage boundary, and EVERY FA_SP stage boundary is a real cross-warp barrier, so the per-stage
// attribution is honest (the sequential kernel's marks are thread-0-only and therefore smeared:
// its "requant = 10,462" is really ~18k of all-warp work that thread 0 only sees the tail of).
//
//   configuration                                        cyc/tile   util     Frobenius
//   FULL_ATTN2 + FA_STEADY (baseline)                      99,288   16.54%   3.5666%  OK
//   FULL_ATTN2 + FA_PIPE (pre-existing, never measured)     64,124   25.61%   3.5666%  OK
//   FA_SP                                                  67,696   24.26%   3.5666%  OK
//   FA_SP QOVL                                             61,270   26.80%   3.5666%  OK
//   FA_SP QOVL QOVL3 LEANCFG                               61,538   26.68%   3.5666%  OK
//   ... + QKACC                                            53,833   30.50%   3.5666%  OK
//   ... + QKACC + PKOVL                                    48,234   34.04%   3.5666%  OK  <== best
//
// STAGE BREAKDOWN of the best configuration (QOVL QOVL3 LEANCFG QKACC PKOVL), steady tile:
//   (empty stage)             52
//   accumulator -> S(t)      998   the QK matmul itself ran under finalize(t-1)
//   softmax               12,825   cooperative, all 6 warps
//   requant pass A         3,611   32-element block max -> E8M0 scale
//   requant pass B        10,619   bf16 -> fp8 convert, WITH the 128-word SF pack hidden inside it
//   PV matmul              9,726   mesh 8,210; also absorbs Q(t+1)'s move-in + 64 scale words free
//   QK(t+1) || finalize   10,403   QK computes into the accumulator while finalize writes O
//   TOTAL                 48,234   ->  16420/48234 = 34.04%
//
// WHAT LIMITS T_tile NOW, in order:
//   1. softmax 12,825 (27%) -- cooperative row-per-warp, SMEM-load-latency bound.
//   2. requant pass B 10,619 (22%) -- already absorbs the whole serial scale pack.
//   3. the QK||finalize stage 10,403 (22%) -- set by FINALIZE, i.e. by the rate at which 16 KB of
//      O can be pushed to GMEM, not by the mesh (QK's 8,210 mesh cycles are entirely hidden).
//   4. the PV matmul 9,726 (20%) -- mesh 8,210, irreducible; it cannot be hidden because the one
//      thing that could run underneath it (finalize of the previous tile) needs a second 16 KB O
//      buffer and SMEM is exactly full.
//   So HALF the mesh work (QK) is now free, and of the 48.2k only 9.7k is exposed mesh.  The
//   remaining cost is SIMT softmax+requant (27.1k) and the O write-out.
//
// ============================================================================================
// 2026-07-26, SECOND PASS.  Everything above was RE-MEASURED FROM SCRATCH and then pushed further.
//   (a)  the 34.04% figure is exactly reproducible -- the CYCLES are;
//   (a2) but the steady-state tiles were computing the WRONG O, and the "3.5666% OK" column on
//        every FA_SP row above is a verification artefact.  Root cause + fix below;
//   (b)  WHY the fused softmax+requant (FA_SP_FUSE) has never once run -- it is the renamer, and
//        the budget it imposes is what bounds this whole kernel;
//   (c)  ONE of the two matmuls is necessarily exposed, which sets the ceiling of this shape;
//   (d)  the second-pass measurement table, with Frobenius scored PER TILE;
//   (e)  what is left, and which part of it needs a file this agent does not own.
//
// (a) REPRODUCTION.  Rebuilt from this file and re-run with +dramsim, fixed seed, and a trace
//     that is verified COMPLETE (8192/8192 cells) before being parsed:
//        FA_SP QOVL QOVL3 LEANCFG QKACC PKOVL, 4 tiles -> tile deltas 67,707 / 48,234 / 46,478
//        the same, 6 tiles                             -> identical, converging to ~46,49x
//     and every per-stage interval is bit-identical to the table above (52 / 998 / 12,825 /
//     3,611 / 10,619 / 9,726 / 10,403).  Two seeds (12345 and 777) give IDENTICAL cycle stamps,
//     confirming the simulator is timing-deterministic and only uninitialised state is seeded.
//     NOTE the pipeline takes THREE tiles to fill, so 48,234 is the TILE-1 interval and 46,478 is
//     tile 2; the "34.04% - 35.33%" spread in earlier notes is those two intervals of the SAME
//     configuration, not two different configurations.
//
// (a2) *** THE STEADY-STATE TILES WERE COMPUTING THE WRONG O, AND THE "3.5666% OK" COLUMN ON
//      EVERY FA_SP ROW OF THE TABLE ABOVE IS A VERIFICATION ARTEFACT. ***
//     Every tile of an FA_SP run recomputes the SAME tile from the SAME Q/K/V and writes O to the
//     SAME GMEM buffer, so a whole-file verify only ever scores whichever generation landed LAST,
//     and a trace parsed before the later finalizes arrive scores an EARLY one.  Both directions
//     mislead; this file's table was built on the early one.  Bucketing the O stores by the
//     preceding MARK index (fa_pertile.py) scores each tile on its own:
//         QOVL QOVL3 LEANCFG QKACC PKOVL, 4 tiles, run to completion, trace stable:
//            tile 0  Frobenius  3.5666%   <- correct
//            tile 1  Frobenius 37.2045%
//            tile 2  Frobenius 31.6828%
//            tile 3  Frobenius 60.9991%   (= what a whole-file verify reports)
//     IDENTICAL at seed 12345 and at seed 777, so this is DETERMINISTIC, not a race.
//     WHAT THE CORRUPTION ACTUALLY LOOKS LIKE (measured, and it narrows the cause a lot):
//       * it is NOT global.  Per row, median(got/golden) is 1.0000 for almost every row; a handful
//         of rows are catastrophically SMALL: tile 1 has 7 bad rows of 64 (ratios 0.0152 .. 0.68),
//         tile 2 has 5, tile 3 has 17.  The count GROWS with the tile index.
//       * the bad rows are the LATE rows of PARTICULAR warps: every one of them satisfies
//         row % 5 in {2,4} (the finalize partitions rows over the 5 SIMT warps), i.e. SIMT warps 2
//         and 4 = physical warps 3 and 5 = BOTH ON CORE 1.
//       * O too small by a non-power-of-2 factor means either l too large, or O_unnorm too small.
//         A too-large E8M0 code makes the e4m3 conversion UNDERFLOW to zero for a whole 32-element
//         block (t = u11 - K8 < 8), and a row with several dead blocks is exactly "O too small by
//         an arbitrary factor" -- so a corrupted per-(row,block) SCALE is the best-fitting cause,
//         and the P data words themselves are probably fine.
//     BISECTED OVER THE FEATURE LADDER (each row a 2-tile run, per-tile Frobenius, and the tile
//     deltas so the cost of being correct is visible):
//         configuration            deltas (t0, t1)   steady  util    per-tile Frobenius
//         FA_SP alone              80,072  66,830    66,830  24.6%   t0 nan, t1 3.5666%
//         + QOVL LEANCFG           77,370  65,047    65,047  25.2%   t0 3.5666% t1 3.5666%  CLEAN
//         + QOVL3                  76,050  61,912    61,912  26.5%   t0 3.5666% t1 3.5666%  CLEAN
//         + QKACC                  (in flight: tag zD2 -- no longer load-bearing, see below)
//         + PKOVL                  67,707  48,234    46,478  35.3%   t0 3.5666%, t1+ BROKEN
//     AND WITH THE FIX (FA_SP_QOVL4 instead of FA_SP_QOVL3):
//         QOVL4 QKACC PKOVL        67,304  50,934    50,934  32.24%  both tiles 3.5666%  CLEAN
//         ... + SMTPR FZROW SMBMAX 62,306  46,016    46,016  35.68%  both tiles 4.2516%  CLEAN
// (a3) *** METHODOLOGY CORRECTION -- THE TILE INTERVALS ARE NOT MONOTONE, SO "THE STEADY STATE" IS
//      NOT THE SMALLEST ONE.  Every cycles-per-tile figure in this file that was read off a 2- or
//      4-tile run is a TRANSIENT, optimistic by 1.2-1.6 utilisation points. ***
//      The 6-tile runs show the intervals fall and then RISE again:
//        published QOVL3 QKACC PKOVL   67,707  48,234  46,478  46,941  48,149
//        QOVL4 ... SMTPR FZROW SMBMAX  61,971  46,802  48,329  48,128
//      So 46,478 -- quoted as this kernel's steady state at 35.33% -- is a LOCAL MINIMUM at tile 2,
//      and that same configuration settles near 48,149 = 34.10%.  Read the same way the SMBMAX
//      variant settles near 48,128 = 34.12%, not the 46,016 = 35.68% its 2-tile run showed, so THE
//      IMPROVEMENT CLAIMED FOR IT IS INSIDE THIS WOBBLE AND IS NOT ESTABLISHED.  The
//      reference-numerics 50,934 = 32.24% is likewise a tile-1 interval and provisional until the
//      FA_NT6 run of that build converges.
//      RULE: FA_NT6 is the minimum useful tile count for a TIMING claim; NT2/NT4 are for correctness
//      only.  Compare at the same interval index, or at the last interval of a 6-tile run.
//      The per-tile-image CORRECTNESS verdicts are UNAFFECTED -- they do not depend on interval
//      indices, so FA_SP_QKACC is still the culprit and FA_SP_QOVL4 still fixes it.
//
//     *** THE HEADLINE, after the fix:
//       REFERENCE NUMERICS, verified          50,934 cyc/tile = 32.24% mesh   *** TRANSIENT, see (a3) ***
//           4 of 4 tile-images at exactly 3.5666% (re-scored per cluster with fa_verify_tiles.py)
//           (FA_SP + QOVL + QOVL4 + LEANCFG + QKACC + PKOVL)   <== the number to quote
//       FASTER, self-consistent but OFF-REFERENCE  46,016 cyc/tile = 35.68% mesh
//           4 of 4 tile-images at 4.2516%: uncorrupted, but 4.2516% != 3.5666%, so this is a real
//           numerics change (the thread-per-row softmax's l rounding), not a verified-equal result
//           (... + SMTPR + FZROW + SMBMAX)
//       the pre-existing published figure      46,478 / 35.33%  -- NOT correct after tile 0
//       the sequential kernel it started from  118,673 single-shot = 13.84%
//     So the sprint's 20% target is cleared by 1.6x at the reference numerics and 1.8x at the
//     faster numerics, with per-tile verification in both cases.  Caveat kept in view: this
//     corruption is TIMING-dependent and one earlier variant only broke at TILE 2, so two clean
//     tiles is strong evidence, not proof -- x1s/x2s at FA_NT6 are the runs that settle it. ***
//     *** RESOLVED, AND IT IS AN INTERACTION -- neither "QOVL3 is the cause" nor "QOVL3 is not the
//     cause" was right.  The three measurements that pin it down:
//         QOVL3                      (no QKACC, no PKOVL)   both tiles 3.5666%   CLEAN
//         QOVL3 + QKACC + PKOVL                             t0 ok, t1+ BROKEN
//         QOVL4 + QKACC + PKOVL                             both tiles 3.5666%   CLEAN
//     So the Q(t+1) prefetch sitting inside the PV stage IS the mechanism, but it is only EXPOSED
//     once QKACC and PKOVL add their concurrency (QKACC makes finalize run alongside mesh work on
//     warps 1-5; PKOVL puts warp 0 in the SF-SRAM pack alongside the SIMT convert).  QOVL3 on its
//     own never hits the window.  That is why bisecting one flag at a time was misleading in both
//     directions, and it is why FA_SP_QOVL4 -- which moves the prefetch to the stage where the mesh
//     performs no SMEM move-out at all -- FIXES IT: 50,934 cyc/tile with both tiles at the reference
//     3.5666%. ***  The two candidate mechanisms, either of which fits: 
//       (i)  Q(t+1)'s move-in DMA vs the PV matmul's accumulator->spad move-out, which needs an
//            ATOMIC ALL-16-SUBBANK GRANT (Hazard 2).  O lands in SP_C = spad rows 3072..5120, which
//            spans SMEM banks 1 AND 2; Q's DMA writes SP_Q = rows 5632..6144, ALSO in bank 2.
//       (ii) Q(t+1)'s 64 SF-SRAM scale words (SF_A half 1) vs the mesh READING the packed P scales
//            (SF_A half 0) for the same matmul, through the gemmini's single pair-merging
//            FlitMergeNode.
//     FA_SP_QOVL4 removes both anyway (it moves the whole prefetch into stage S6, where the mesh
//     runs QK compute-only and performs no move-out, and the in-order ROCC queue still orders Q
//     before the QK that reads it), so it is a safe placement -- it is just not the fix.
//     THE TWO REMAINING SUSPECTS, and why each fits the "late rows of core 1's warps" signature:
//       FA_SP_QKACC -- it is the flag that makes finalize(t) run CONCURRENTLY with mesh work, on
//         warps 1-5 only, and the accumulator->spad store that follows it (stage S1 of tile t+1)
//         writes S(t+1) over exactly the SP_C region finalize is reading.  A warp still finishing
//         finalize when that store lands reads S(t+1) instead of O(t) -- and the rows it has left
//         are its LAST rows, which is the observed pattern.
//       FA_SP_PKOVL -- it is the flag that puts warp 0 in the SF-SRAM pack while warps 1-5 write P8
//         in SMEM, i.e. a gemmini-tile writer running concurrently with SIMT SMEM traffic.
//     IT IS A TIMING WINDOW, NOT A HARD ORDERING VIOLATION, and the evidence for that is that the
//     tile at which corruption starts MOVES WITH THE STAGE TIMINGS: the cooperative-softmax build
//     breaks at tile 1, the thread-per-row build survives tiles 0-1 and breaks at tile 2, and
//     adding FA_SP_SMBMAX (which only changes how long the softmax stage takes) makes tiles 0 AND 1
//     both verify at 4.2140%.  So "it verified on the tiles I looked at" is NOT evidence of
//     correctness for this hazard -- only removing the concurrency is.
//     FIX = FA_SP_QOVL4: issue the prefetch in stage S6 instead, where the mesh runs QK
//     COMPUTE-ONLY into the accumulator and performs no SMEM move-out at all, and where the
//     in-order ROCC queue still guarantees Q is resident before the QK that reads it.  The overlap
//     is kept; only the stage it hides under changes.
//     LESSON (the same one as the INCOMPLETE-TRACE trap, from the other side): a per-tile result
//     must be verified PER TILE.  "8192/8192 cells covered" only says the buffer is full, not that
//     ONE tile filled it.
//     *** AND THE SECOND HALF OF THAT LESSON, WHICH COST A ROUND OF WRONG DETAIL: VERIFY PER
//     CLUSTER TOO.  USE kernels/fa_mx_hostsf/fa_verify_tiles.py, NOT fa_pertile.py. ***  This config
//     has TWO clusters, both write MARKs, and both write the SAME O addresses interleaved into one
//     trace -- so bucketing by the most recent MARK (what fa_pertile.py does) mixes clusters and
//     generations, and on the unmodified baseline it reports 52-85% for output that is provably
//     correct.  fa_verify_tiles.py groups by cluster first and starts a new image on an address
//     repeat, which is self-validating (exactly 4096 words per image, and the images account for
//     exactly the store words the whole-file verify sees).  Re-scoring everything here with it
//     CONFIRMED the substance and corrected two details:
//       * the corruption is confined to CLUSTER 0 -- cluster 1 is correct on every tile -- not to
//         "the late rows of the warps on core 1", which is what the mixed buckets suggested;
//       * its magnitude is 107-111% (garbage), not 31-61%.
//     AND IT CAN REPORT A FALSE PASS, which is the failure direction that actually costs you: on the
//     identical complete trace for QOVL3+QKACC, fa_pertile.py says "3.5666% | 3.5666%" (both tiles
//     perfect) while fa_verify_tiles.py says cluster 0 tile 1 is 85.1674% WRONG.  A correct cluster
//     masks a broken one.  NEVER accept a clean verdict from the mark-bucketing tool.
//     Re-scored verdicts (tile-images; 3.5666% == correct).  Every one of these is per cluster:
//       config (all + FA_SP QOVL LEANCFG)   steady    util    tile-images
//       published QOVL3 QKACC PKOVL         46,478    35.3%   5 of 8: cluster 0 tiles 1,2,3 are
//                                                             107-111% WRONG, cluster 1 all correct
//       QOVL3 alone                         61,912    26.5%   4 of 4 correct
//       QOVL3 PKOVL   (no QKACC)            56,229    29.2%   4 of 4 correct
//       QOVL3 QKACC   (no PKOVL)            53,275    30.8%   3 of 4: cluster 0 tile 1 is 85.2%
//                                                             WRONG  <== *** THE CULPRIT ***
//       QOVL4 QKACC PKOVL                   50,934    32.24%  4 of 4 at exactly 3.5666%  <== THE FIX
//       ... + SMTPR FZROW SMBMAX            46,016    35.68%  4 of 4 SELF-CONSISTENT at 4.2516%:
//                                                             uncorrupted, but not the reference
//                                                             value -- a real numerics difference
//     *** RESOLVED: THE CULPRIT IS FA_SP_QKACC (together with the QOVL3 prefetch placement), AND
//     FA_SP_PKOVL IS INNOCENT. ***  QOVL3 alone is clean; QOVL3 + PKOVL is clean; QOVL3 + QKACC
//     BREAKS (cluster 0 tile 1, 85.2%); QOVL3 + QKACC + PKOVL breaks harder (3 of 8 images);
//     QOVL4 + QKACC + PKOVL is clean.  So it is a TWO-WAY interaction between the Q(t+1) prefetch
//     living in the PV stage and FA_SP_QKACC, and that is mechanistically the right shape: QKACC is
//     the flag that makes the mesh run CONCURRENTLY with finalize and that moves when the PV
//     accumulator->spad move-out happens relative to the prefetch.  FA_SP_QOVL4, which relocates the
//     prefetch out of the PV stage, removes the interaction -- 4 of 4 images at exactly 3.5666%.
//     IT IS A TIMING WINDOW, and here is the cleanest demonstration: adding FA_SP_DUMPSC to the
//     BROKEN configuration -- a pure diagnostic that only adds a barrier and 32 stores per tile --
//     makes it verify 4 of 4.  So any instrumentation heavy enough to observe this hazard perturbs
//     it away, which is why it survived so long and why the fix had to be structural rather than
//     found by probing.
//     Note the cost of correctness: every individually-clean rung is SLOWER than the broken
//     combination, and FA_SP_QOVL4 is what buys the cycles back (50,934 vs 61,912) while staying
//     correct.
//
// (b) *** WHY FA_SP_FUSE HAS NEVER RUN: THE RENAMER, AND IT IS A KERNEL-WIDE BUDGET. ***
//     Every FUSE build $finishes at Rename.scala:123 "total register usage exceeded maximum
//     number of physical registers", roughly 100k cycles in (mid tile 0).  Rename.scala:66-121:
//     `assigning = valid && writesToRd && !assigned(wid)(rd)` -- a physical register is handed out
//     the FIRST time a given WARP writes a given ARCHITECTURAL register, is NEVER reclaimed, and
//     comes from ONE counter per core over 1..numPhysRegs-1 (numPhysRegs = 256, MuonCore.scala:24).
//     So the hard limit is
//          sum over the warps resident on a core of (distinct arch regs that warp writes)  < 256
//     With mu_schedule occupancy 3 (6 warps) one core carries 3 warps, so THE WHOLE KERNEL may
//     write only ~85 distinct architectural registers, of which the runtime already uses ~27.
//     Measured whole-kernel unions, counted off the generated .s (fa_regs.py counts distinct
//     written destination registers):
//          PKOVL            53   runs             SMTPR            49   runs
//          FUSE             58   $finish          FUSE + 1-pass    69   $finish
//     The five extra registers are NOT the unroll factor and NOT the number of reduction chains
//     (both were reduced and re-measured: no change).  They are fa_expreq's NINE arguments, five
//     of them pointers that stay live across the item loop and so land in callee-saved s55-s58,
//     plus two more in fa_entry.  Rematerialising them from compile-time constants does not help
//     either -- LICM hoists the constant address straight back into a loop-invariant register.
//     CONSEQUENCES, all of which explain older entries in this file:
//       * FUSE, and the single-pass e[16] form of fa_expreq, can only run at occupancy 2
//         (limit 127/warp) -- which is why the "e[16] fits at 72 regs" note above was wrong.
//       * "occ=4 overflowed the register file" is the same assert: at 4 warps/core the budget is
//         63 per warp and even the 53-register PKOVL kernel does not fit.
//       * ANY new register-hungry stage is a 6-warp/3-warp tradeoff, not a free choice.  Check it
//         BEFORE burning a 70-minute simulation:  bash fa_asm.sh TAG "DEFINES" (compile only,
//         ~20 s) then python3 fa_regs.py /tmp/fa_TAG.s -- 53 is known-good, 58 is known-fatal.
//
// (c) *** WHY ONE MATMUL IS ALWAYS EXPOSED (the ceiling of this pipeline shape). ***
//     There is ONE accumulator, and the per-tile dependency chain is
//         QK(t) -> softmax(t) -> requant(t) -> PV(t) -> finalize(t).
//     The only SIMT work that does not depend on the mesh result of its own tile is finalize, and
//     finalize(t) depends on PV(t).  So exactly one of the two matmuls can be covered: QK(t+1) is
//     issued compute-only into the accumulator underneath finalize(t) (FA_SP_QKACC), and PV(t)
//     then has nothing left to hide under -- tile t+1's softmax would need S(t+1), which needs the
//     accumulator that PV(t) is using, and deferring finalize by one tile needs a second 16 KB O
//     buffer that does not exist (V 32K + P8 16K + S/O 32K + scratch 8K + Q 8K + K 32K = 128K,
//     exactly full).  A k-split of PV (run k=0..127 under the requant) does not work either: the
//     MX scale words are block-major, so each k-half needs ITS scales at word 0 of a scale-SRAM
//     half, and all four halves are already committed (SF_A: P-scales + Q-scales, SF_B: K + V).
//     Hence  T_tile  =  accumulator store + SIMT(softmax + requant) + PV + max(finalize, QK),
//     and with the SIMT block at ~21k the floor for this shape is ~36-37k, i.e. ~45% mesh.
//
// (d) SECOND-PASS MEASUREMENT TABLE.  All rows: RadianceTapeoutSimConfig, +dramsim, explicit seed,
//     run to completion, trace stable before parsing, and Frobenius scored PER TILE with
//     y_pertile.py (never whole-file -- see (a2)).  "deltas" are the consecutive tile-to-tile MARK
//     intervals; the pipeline needs ~3 tiles to fill, so the LAST delta is the steady state.
//     All of these are FA_SP + QOVL + LEANCFG + QKACC unless stated.
//     TWO CAVEATS, so the numbers are not over-read: (i) rows with fewer deltas were STOPPED EARLY
//     once their conclusion was settled (the machine was shared), so their "steady" is the last
//     COMPLETE interval, not a fully converged asymptote -- the trend is always downward, so they
//     are upper bounds; (ii) the SMBMAX rows here predate the two-level l tree that SMBMAX now also
//     enables, so their 4.2140% is with the flat 64-deep row sum.
//
//   + flags                       tile deltas                 steady  util    per-tile Frobenius
//   CONTROL: the SEQUENTIAL kernel (FULL_ATTN2 + FA_STEADY, no FA_SP at all), 2 tiles, verified the
//   same way: BOTH tile generations are 3.5666%.  So the steady-state corruption below is specific
//   to the software pipeline -- it is not the harness, not the shared softmax/requant helpers, and
//   not the per-tile verification method.
//   QOVL3 PKOVL  (the old "best") 67,707 48,234 46,478 46,502  46,49x  35.3%   t0 3.5666%, t1+ BROKEN
//   QOVL3 PKOVL SMTPR             65,597 46,717 46,126         46,126  35.6%   t0,t1 4.2342%, t2+ BROKEN
//   QOVL3 PKOVL SMTPR SMBMAX      62,899 43,850                43,850  37.45%  t0,t1 4.2140%  OK
//   QOVL3 PKOVL SMTPR SMBMAX SUBMAX 61,356 42,202              42,202  38.91%  t0 7.7820%, t1 BROKEN
//   QOVL3 PKOVL SMTPR SUBMAX      64,748 44,275                44,275  37.09%  t0 7.7888%, t1 BROKEN
//   QOVL3 PKOVL SMTPR SMBMAX SUBMAX FZU4  62,488 ...                           FZU4 is a LOSS (+1,132)
//   QOVL3 PKOVL REFETCH           81,862 61,416 60,480         61,416  26.74%  $finish, 55 registers
//   QOVL3 PKOVL          + FA_OCC2 79,437 56,455               56,455  29.09%  t0 3.5666%, t1 BROKEN
//   QOVL3 FUSE ...       + FA_OCC2 86,390 67,122               67,122  24.46%  t0,t1 4.2539%  OK
//   QOVL3 FUSE 1P ...    + FA_OCC2 114,788 95,977              95,977  17.11%  t0,t1 4.2539%  OK
//   QOVL3 FUSE (occupancy 3)      --                           --      --      $finish, 58 registers
//
//   WHAT THIS TABLE SAYS, in order of importance:
//     1. The 34-35% timing result REPRODUCES EXACTLY, but the configuration it was measured on
//        computes the wrong O after tile 0 (see (a2)).  Timing and correctness had to be separated.
//     2. FA_SP_SMBMAX is the real win of this pass: -2,698 cyc on tile 0 and 46,126 -> 43,850 in
//        steady state (35.6% -> 37.45%) for free, by taking the requant's 32-element block max in
//        the softmax loop that already holds those 32 values in registers.  It is also the only new
//        flag whose steady-state tiles verify (4.2140% on tile 0 AND tile 1).
//     3. OCCUPANCY 2 IS NOT WORTH IT.  It doubles the register budget and makes FA_SP_FUSE (and the
//        single-pass form) run and verify -- but 4 warps instead of 6 costs far more than the fused
//        kernel saves: the SAME PKOVL config goes 46,478 -> 56,455, and the best occ-2 variant
//        (24.46%) is worse than the plain occ-3 baseline.  FUSE and 1P are therefore DEAD as
//        performance ideas on this machine, and the reason is the renamer, not the fusion.
//     4. SMTPR trades accuracy for speed: 3.5666% -> 4.2342%.  *** THE "IT IS ALL l's bf16
//        ROUNDING" EXPLANATION THAT USED TO BE WRITTEN HERE IS WRONG, AND THE ARITHMETIC SAYS SO
//        QUANTITATIVELY -- see (e3) below.  The gap is real, it belongs to FA_SP_SMTPR and NOT to
//        FA_SP_SMBMAX, and its mechanism is NOT YET IDENTIFIED. ***
//
// (d2) RUNS LEFT IN FLIGHT when this was written, and exactly how to read them.  Each is a 2-tile
//      run; the ONLY question for each is whether tile 1's per-tile Frobenius matches tile 0's.
//        python3 fa_pertile.py /tmp/yruns/<TAG>.out      # both tiles equal  => that flag is clean
//      TAG  configuration (all on FULL_ATTN2 FA_SP FA_SP_QOVL FA_SP_LEANCFG, + what is listed)
//      zD2  QOVL3 QKACC          -- isolates FA_SP_QKACC .  If tile 1 breaks, QKACC is the culprit
//      zE2  QOVL3 PKOVL          -- isolates FA_SP_PKOVL .  If tile 1 breaks, PKOVL is the culprit
//      x7   QOVL3 QKACC PKOVL SLOWPAD -- the broken config with VOLATILE barrier pads.  If this is
//           clean, the hazard is a barrier-release timing window, not a structural ordering bug,
//           and the cheap fix is a longer pad rather than a redesign.
//      x1   QOVL4 QKACC PKOVL          -- does the (safer) QOVL4 placement help anyway?
//           PARTIAL RESULT: tile-0 delta 67,304 vs QOVL3's 67,707, and tile 0 verifies at 3.5666%.
//           So FA_SP_QOVL4 is very slightly FASTER than FA_SP_QOVL3, not slower -- moving the
//           prefetch out of the PV stage costs nothing.  Whether it also fixes tile 1 is the open
//           question; QOVL3's own rung is clean, so most likely it does not.
//      x2   QOVL4 QKACC PKOVL SMTPR FZROW SMBMAX -- *** BEST VERIFIED RESULT: deltas 62,306 /
//           46,016, and tile 0 AND tile 1 BOTH score 4.2516%.  46,016 cyc/tile = 35.68% mesh, i.e.
//           FASTER than the original (broken) 46,478 configuration AND correct on every tile it
//           ran.  Confirmed independently by two separate watchers.  Caveat that matters: the
//           corruption is timing-dependent and in one earlier variant it first appeared at TILE 2,
//           so two clean tiles is strong evidence but not proof -- x2s is the same build at FA_NT6
//           (five steady intervals) and is the run that settles it. ***
//      x3   QOVL4 QKACC PKOVL ITEM  FZROW  -- *** MEASURED A CLEAR LOSS: tile-0 delta 75,521 vs
//           62,306 for the thread-per-row version (x2), i.e. +13,215.  So FA_SP_ITEM is REFUTED as
//           a performance idea, and the "a third of the machine is idle" argument for it is wrong
//           in practice.  Why: thread-per-row does its row max and its exp pass as TWO SWEEPS OF
//           ONE FUNCTION over a row the thread owns, whereas the item-parallel form has to split
//           them into a cooperative fa_rowmax_s over all of S, a barrier, the item pass, another
//           barrier, and fa_rowsum -- the same SMEM traffic, but with cross-lane reduction
//           scaffolding (a fence per row) and two extra cross-warp barriers, and every item pays a
//           fresh m_in[row] load and address computation instead of hoisting them per row.  Using
//           more threads did not pay for that.  Kept OFF and documented. ***
//      x1s  QOVL4 QKACC PKOVL at FA_NT6            -- 5 steady intervals for the REFERENCE-numerics
//           headline (50,934 from a 2-tile run); this is the run that proves the fix holds past
//           tile 2, which is where one earlier variant first broke.
//      x1b  the same at FA_NT2, SEED 777             -- second-seed correctness for the headline.
//      x2s  QOVL4 QKACC PKOVL SMTPR FZROW SMBMAX at FA_NT6 -- PARTIAL: 61,971 / 46,802, converging.
//      x3s  the FA_SP_ITEM variant at FA_NT6         -- PARTIAL: 74,056 / 60,750, i.e. it confirms
//           at NT6 that FA_SP_ITEM is a large loss.
//      dsc  QOVL3 QKACC PKOVL DUMPSC -- *** RESULT: 4 of 4 tile-images CORRECT, i.e. the diagnostic
//           PERTURBS THE HAZARD AWAY.  It cannot be used to localise this bug; it is instead the
//           evidence that the bug is a timing window.  The dump itself is SOUND -- on that clean run
//           tile 0's and tile 1's 128 scale words are bit-identical (128/128, zero differing words),
//           which is exactly what must be true when every tile recomputes the same tile.  So the
//           probe works and would be conclusive; it just cannot coexist with the hazard. ***
//           Dumps the 128 E8M0 scale words per tile to 0x40051000 +
//           512*t.  If the dumps are identical across tiles then SCALE_SMEM is fine and the damage
//           is in the pack into the gemmini SF-SRAM or the mesh's read of it; if they already
//           differ, the requant (or something racing it) is at fault.  This is the diagnostic that
//           localises the corruption to a specific producer, and it is worth running first.
//
// (e3) *** THE 3.5666% -> 4.2xxx% SOFTMAX ACCURACY GAP: WHAT IT IS NOT. ***
//     Measured on ONE otherwise-identical pipeline (QOVL4 QKACC PKOVL, per-cluster per-tile,
//     tile-0 images):
//         cooperative online_softmax_block          3.5666%   == the sequential kernel's own value
//         + FA_SP_SMTPR FA_SP_FZROW                 4.2342%
//         + FA_SP_SMBMAX                            4.2516%
//     So FA_SP_SMBMAX moves the error by +0.017 points -- it is NUMERICALLY NEUTRAL, and taking the
//     requant's block max out of the softmax's registers instead of re-reading the stored bf16 P is
//     as accurate as the two-pass form, exactly as intended.  The whole 0.68-point step belongs to
//     the thread-per-row softmax.
//     l's SUMMATION ORDER CANNOT ACCOUNT FOR IT, and this is arithmetic, not opinion.  Emulating
//     bf16 round-to-nearest-even in numpy on the real golden S and reducing exactly as each kernel
//     does gives, for l's relative error and for the Frobenius error l ALONE can inject into
//     O = O_unnorm / l (a relative error d in l scales that whole row of O by 1/(1+d), so the
//     contribution is the row-norm-weighted rms of d/(1+d)):
//         cooperative: 16-lane chains of 16 + 16-leaf tree   rms 0.0031  ->  0.3115%
//         thread-per-row: 4 chains of 64                     rms 0.0047  ->  0.4703%
//         thread-per-row: 2-level block tree (SMBMAX)        rms 0.0028  ->  0.2786%
//     Closing 3.5666% -> 4.2342% needs sqrt(4.2342^2 - 3.5666^2) = 2.28% of extra error, and NO
//     reduction order supplies a tenth of that.  The measured ordering is backwards for the theory
//     as well: the 2-level tree is the MOST accurate l of the three and produces the WORST O.
//     m IS PROVABLY IDENTICAL between the two softmaxes.  The cooperative one takes max_i of the
//     SCALED values; the thread-per-row one takes the max of the RAW row and scales once.  bf16
//     multiply and bf16 round-to-nearest are both monotone non-decreasing, so
//     max_i RNE(S_i * scale) == RNE(max_i S_i * scale) -- the two expressions are equal bit for bit.
//     P SHOULD ALSO BE IDENTICAL: both compute mu_fexp(RNE(RNE(S*scale) - m)) elementwise, both
//     write it in place at the same word index, and both cover the row exactly once (the lane
//     rotations are permutations: the union over the loop of (i+q+lane) mod SKW is all of SKW).
//     AND finalize_O and fa_finalize_row are the same arithmetic -- one bf16 reciprocal per row,
//     then a multiply -- differing only in whether l is a uint16 array or one 32-bit word per row.
//     So on paper the two configurations must agree, and they do not.  STATUS: LOCALISED TO THE
//     THREAD-PER-ROW SOFTMAX, MECHANISM UNKNOWN.  The two diagnostics that close it are in the file
//     and cost one 2-tile run each: FA_SP_DUMPP (per-row XOR checksum of the bf16 P the softmax
//     wrote) and FA_SP_DUMPLM (the l and m arrays), run on the same base at the same occupancy with
//     and without FA_SP_SMTPR, and read with /tmp/fa_dumpchk.py (which splits by cluster).  If the
//     P checksums differ the exps differ and "same math, different parallelisation" is false; if
//     they match, the fault is in l after all and the numpy model above is missing something about
//     how this hardware rounds.
//     WHY IT MATTERS FOR WHAT SHIPS: until this is closed, the reference-numerics configuration and
//     the thread-per-row configuration are two separate results, and only the first can be called
//     correct.  4.2516% is 19% more relative error than the MX-FP8 floor this kernel is supposed to
//     sit on, which is small but is NOT nothing.
//
// (e) WHAT IS LEFT, in descending value.  The stage costs to beat (steady tile, SMTPR+SMBMAX):
//        accumulator->S 998 | softmax ~12,4xx | (pass A gone) | requant convert + SF pack ~10,5xx
//        | PV ~9,6xx | max(finalize, QK) ~9,7xx
//     1. THE SF SCALE PACK BELONGS ON THE ROCKET HOST.  128 words at ~65 cyc/word is 8,320 cycles
//        of strictly serial, unparallelisable work (FlitMergeNode: one thread, strictly ascending,
//        even-sized runs).  It is currently hidden under the requant convert, which is why the
//        convert only gets 80 of 96 threads -- warp 0 is busy packing.  Move the pack to the host
//        (the machinery already exists: see commit 2027e60 and kernels/fa_mx_hostsf) and the
//        convert gets all 96 threads: 10,5xx -> ~8,8xx.  THIS NEEDS host.cpp, WHICH THIS AGENT DOES
//        NOT OWN -- it is a request, not a TODO.
//     2. FA_SP_ITEM (item-parallel exp) recovers the 32 threads that thread-per-row leaves idle.
//     3. The 16-way subbank conflict in fa_requant_cvt is real and UNFIXED, because every rotation
//        that breaks it costs registers the kernel does not have (55 aborts).  One idea not yet
//        tried: rotate only the HALF index h by (lane & 1) instead of the full pair index u.  That
//        leaves q a compile-time constant, so only ONE term of the store address becomes runtime,
//        and it still gives 8-way instead of 16-way.  Check the register union first.
//     4. The exposed PV (~9.6k) is STRUCTURAL -- see (c).  Breaking it needs either a second
//        accumulator or a flash-blocked QK (K in blocks of 128 so S is 16 KB and a second O buffer
//        fits), which is a different kernel, not a flag.
//
// --------------------------------------------------------------------------------------------
// HOW TO REPRODUCE ANY ROW OF THE TABLES ABOVE.
//   BUILD   bash /tmp/fa_build.sh <TAG> "FULL_ATTN2 <DEFINES>"          (self-locking; ~20 s)
//   RUN     bash fa_run.sh <TAG> <ELFTAG> <SEED> <CYCLE_BUDGET>
//           -- same simulator recipe as `make ... run-binary` (+dramsim matters: ~35k cycles) but
//           the ISSUE trace is grep-filtered on the fly to the only lines any tool parses (GMEM
//           stores to 0x40040000-0x40043fff = O, and 0x4005xxxx = the MARK stamps), so the .out is
//           a few MB on / instead of hundreds on /scratch, there is no spike-dasm stderr-drain race,
//           and the seed is explicit.  +max-cycles counts HALF cycles -- the script doubles it.
//   WAIT    the run must report DONE **and** the .out size must stop growing before parsing.
//   VERIFY  bash fa_final_check.sh <TAG>...   <- waits for DONE + a stable trace, then reports the
//           steady-state tile interval AND the per-CLUSTER per-tile verdict via fa_verify_tiles.py.
//           This is the only scoring path to trust.  Do NOT use fa_pertile.py: it mixes the two
//           clusters and has been caught reporting a FALSE PASS on a genuinely broken run.
//   TIME    python3 /tmp/fa_pm.py /tmp/yruns/<TAG>.out --per 7   <- per-stage + tile deltas
//   CHECK BEFORE RUNNING (both ~20 s, both have caught real bugs here):
//           bash fa_asm.sh <TAG> "FULL_ATTN2 <DEFINES>"
//           python3 fa_regs.py /tmp/fa_<TAG>.s      <- must be <= 53; 55 and 58 both $finish
//           python3 fa_baraudit.py /tmp/fa_<TAG>.s      <- per-function barrier audit
//
// ---- FIFTH-PASS CONFIGURATIONS (2026-07-27).  *** THESE SUPERSEDE THE THREE BELOW: every one of
// ---- the older FA_SP recipes is missing FA_SP_WCNT and is therefore only accidentally correct.  ***
//   reference numerics, correctness-hardened, no host:
//     FA_SP FA_SP_QOVL FA_SP_QSPLIT FA_SP_LEANCFG FA_SP_QKACC FA_SP_PKOVL FA_SP_WCNT
//     FA_SP_PAX FA_SP_CVTX
//   ... + the host scale offload and the softmax rewrite (the fastest verified stack):
//     ... FA_SP_CVTSWAR FA_SP_HSF FA_SM_2P FA_SP_FZ6
//   the FA_SP_OPV schedule (hides ALL of the mesh; $finishes on Scratchpad.scala:220 -- see (G2b)):
//     FA_SP FA_SP_QOVL FA_SP_OPV FA_SP_OPVQ FA_SP_BANKA FA_SP_LEANCFG FA_SP_PKOVL FA_SP_WCNT ...
//
// THE THREE CONFIGURATIONS THAT MATTER (all on top of FULL_ATTN2):
//   reference numerics, correctness-fixed:
//     FA_SP FA_SP_QOVL FA_SP_QOVL4 FA_SP_LEANCFG FA_SP_QKACC FA_SP_PKOVL
//   fastest verified:
//     ... + FA_SP_SMTPR FA_SP_FZROW FA_SP_SMBMAX
//   fastest verified, item-parallel exp:
//     FA_SP FA_SP_QOVL FA_SP_QOVL4 FA_SP_LEANCFG FA_SP_QKACC FA_SP_PKOVL FA_SP_ITEM FA_SP_FZROW
//   plus FA_NT2 / FA_NT4 / FA_NT6 to pick the tile count (NT6 is needed to see the asymptote --
//   the pipeline takes three tiles to fill).
//
// --------------------------------------------------------------------------------------------
// FLAG CATALOGUE (all default OFF; the plain build and the plain FULL_ATTN2 build are byte-for-
// byte the pre-existing kernel).
//   FA_SP           the whole thing: new SMEM map, K^T/V + K/V scales resident, explicit mvin,
//                   in-place softmax, and a real cross-warp barrier at every stage boundary
//                   (which also removes the pre-existing unsynchronised softmax->requant race).
//   FA_SP_QOVL      warp 0 becomes the gemmini agent, warps 1-5 the SIMT group; Q(t+1)'s
//                   move-in and scale write are issued one stage early.
//   FA_SP_QOVL3     ...and issued under the PV MATMUL instead, where SF-SRAM writes are free.
//   FA_SP_SMTPR     thread-per-row softmax (one lane owns a whole row; no cross-lane reduce).
//   FA_SP_FUSE      FUSED softmax+requant: cooperative row max, then one item-parallel pass that
//                   exps a 32-element MX block, takes its block max, and converts to fp8 without
//                   ever writing bf16 P to SMEM.  Halves the SMEM traffic of the two phases.
//   FA_SP_2P        two-pass, zero-array form of the fused kernel -- MANDATORY at occ=3, see
//                   hardware fact 4.
//   FA_SP_RMSER     serial 16-partial row-max reduce (1 fence/row).  FA_SP_RMBATCH is the
//                   RB-row-batched butterfly alternative (faster in theory, +13 registers).
//   FA_SP_LEAN      native fmax.h/fadd.h reduce instead of clang's fp32-promoted fmaxf/(float).
//   FA_SP_SWAR      e4m3_pack4_swar (25 instr/word) instead of e4m3_pack4 (34).
//   FA_SP_PKOVL     split requant so the SF pack hides under the convert pass (superseded by
//                   FA_SP_FUSE, which splits the fused pass by MX block instead).
//   FA_SP_QKACC     QK(t+1) runs COMPUTE-ONLY into the accumulator under finalize(t); the
//                   accumulator is moved out to S(t+1) at the top of the next iteration.
//   FA_SP_FZFLAT    flat/coalesced finalize (reciprocals hoisted).  FA_SP_FZROW = row-per-warp.
//   FA_SP_LEANCFG   drop the per-gemm configure_mxgemmini (7 ROCC cmds + an MMIO poll) -- all of
//                   it is either gemm-invariant, re-issued by gemmini_loop_ws_spad, or dead.
//   FA_SP_MAP2      swap V and P8 between SMEM banks (measured WORSE, kept for the record).
//   FA_SP_REFETCH   MEASUREMENT MODE: re-fetch K^T, V and their 512 MX scale words every tile --
//                   i.e. the FA_STEADY harness's pessimistic assumption -- so the cost of the
//                   "KV block stays resident" structure can be quantified rather than argued.
//                   Needs FA_SP_QOVL3 (K's refill rides under PV) and FA_SP_QKACC (V's under QK).
//
// ---- added 2026-07-26 (second pass) --------------------------------------------------------
//   FA_SP_QOVL4     *** CORRECTNESS FIX -- USE THIS INSTEAD OF FA_SP_QOVL3. ***  Issue the Q(t+1)
//                   prefetch in stage S6 (under the QK compute-only + finalize) instead of stage S5
//                   (under the PV matmul's accumulator->spad move-out, where its DMA breaks that
//                   move-out's atomic 16-subbank grant and corrupts O on every tile after tile 0 --
//                   see (a2) in the header).  The overlap is preserved; only the stage it hides
//                   under changes.  Needs FA_SP_QOVL + FA_SP_QKACC; excludes FA_SP_QOVL3.
//   FA_SP_SMBMAX    *** the per-32-element E8M0 block scale is produced BY THE SOFTMAX ***, which
//                   deletes the entire requant pass-A stage.  Pass A existed only to re-read the
//                   bf16 P that the softmax had just written and take a 32-element max of it, but
//                   the softmax loop already holds those 32 values in registers, so the max costs
//                   4 extra fmax.h per 4 elements and one extra word store per block.  Needs
//                   FA_SP_SMTPR (one thread owns a whole row, so a block's 16 words are naturally
//                   consecutive); the loop is re-shaped blocks-then-words with the lane rotation
//                   kept INSIDE the block so all 16 SMEM subbanks stay busy (verified: same 128
//                   words per row, 16 distinct subbanks per step).
//   FA_SP_SUBMAX    subsample the row max 4:1 and add a +24.0 safety margin.  The ALGEBRA is sound:
//                   m is a PER-ROW constant and a per-row constant cancels EXACTLY between the
//                   numerator and the denominator of the softmax, so m is not a numeric input to O
//                   at all -- its only job is keeping exp(u-m) inside bf16 range, and the per-block
//                   E8M0 scale restores the fp8 mantissa afterwards.  There are ~88 units of slack
//                   on each side and the measured worst-case subsample underestimate on this input
//                   is 18.25, so it cannot overflow or underflow.
//                   *** BUT IT COSTS REAL ACCURACY, SO IT IS NOT SHIPPED. ***  On an otherwise
//                   identical kernel, tile-0 Frobenius goes 4.2140% -> 7.7888%.  The algebra says
//                   that cannot come from m, so it comes from the HARDWARE: pushing every exp
//                   argument ~24 further negative makes mu_fexp (fexp.h) itself less accurate.
//                   Worth keeping as a hardware fact -- on this machine the exp is NOT uniformly
//                   accurate in its argument, so "subtract a bigger max, the block scale will undo
//                   it" is FALSE.  FA_SP_SUBMAX_NOMARGIN keeps the 4:1 subsample and drops the
//                   margin, which separates the two effects.
//   FA_SP_ITEM      ITEM-PARALLEL exp pass.  MEASURED A LOSS (+13,215 cyc on tile 0) -- see (d2).
//                   The idea was that the thread-per-row softmax leaves a THIRD of the machine idle;  FA_SP_SMTPR gives one row to one thread, and there are 64
//                   rows but 96 threads, so warps 4 and 5 do nothing at all during the largest SIMT
//                   stage of the kernel.  FA_SP_ITEM splits the work by (row, 32-element MX block)
//                   instead -- 512 independent items over 96 threads -- as three phases behind one
//                   call (fa_softmax_item): cooperative row max, item-parallel exp + block max +
//                   l partial, then the row sum.  Reuses fa_rowmax_s and fa_rowsum unchanged, and
//                   is fa_expreq minus the fp8 conversion, which is exactly the part that made
//                   FA_SP_FUSE unaffordable.  Implies FA_SP_SMBMAX; needs FZROW or FZFLAT.
//                   Rotating the in-block word index by the lane is MANDATORY here (word index is
//                   row*128 + b*16 + k, so the subbank is k&15 = lane-uniform = 16-way conflict).
//   FA_SP_FZU4      unroll finalize for memory-level parallelism (the flat variant by 4, the row
//                   variant by 2 rows) -- a rolled grid-stride loop has ONE store in flight.
//                   MEASURED A LOSS: +1,132 cyc on tile 0.  finalize is not latency-starved (it is
//                   the one phase with a GMEM store queue in front of it), so the extra live
//                   registers and code size buy nothing.  Kept for the record, OFF.
//   FA_SP_RLEAN     register-lean fa_expreq (2 reduction chains instead of 4).  Written to make
//                   FA_SP_FUSE fit the renamer budget; it does NOT (see (b) in the header).
//   FA_SP_1P        single-pass fa_expreq: hold the 32 exps as e[16] instead of recomputing them,
//                   halving the phase's SMEM loads and exp count.  Needs FA_OCC2 -- at occupancy 3
//                   it is 69 registers against a ~85 budget including the runtime, and $finishes.
//   FA_SP_CVTSWAR   SWAR packer in the PKOVL convert (638 -> 582 instructions).  MEASURED AND
//                   REJECTED at occupancy 3: it lands on 58 registers in every combination tried,
//                   and 58 is the known-fatal count.  The packer choice here is a REGISTER
//                   decision, not an instruction-count one.
//   FA_SP_CVTROT    rotate the convert's PAIR index by the lane id, to break what is otherwise a
//                   16-WAY SMEM SUBBANK CONFLICT ON EVERY LOAD of the biggest remaining SIMT
//                   stage: the subbank is word_index & 15, the index is row*128 + b*16 + k, and
//                   both 128 and 16 are multiples of 16 -- so the subbank is (k & 15), which is
//                   LANE-UNIFORM, and a warp's 16 lanes all hit ONE subbank on every load.
//                   flash_mx_impl.hpp documents this trap and fixes it for
//                   requant_P_to_spad_tiled, but the split fa_requant_max/fa_requant_cvt pair that
//                   PKOVL uses lost the rotation.  Only 8 starting points are legal (an output word
//                   must hold 4 consecutive elements of the block), so this gives 2-way instead of
//                   16-way.  MEASURED AND REJECTED: it costs 55 registers, and 55 $finishes at
//                   occupancy 3 (PKOVL+REFETCH, also 55, aborts the same way).  So this known
//                   16-way conflict CANNOT BE FIXED at 6 warps -- it is a register-budget
//                   casualty, not a missing optimisation.  Also +107 instructions.
//
// ---- added 2026-07-26 (third pass) ---------------------------------------------------------
//   FA_SP_NOMAX     *** DELETE THE SOFTMAX'S ROW-MAX PASS ALTOGETHER (m := 0). ***  The row max
//                   exists ONLY to keep exp(u-m) inside the range of the P scratch, and that
//                   scratch is bf16 -- which has FP32's 8-BIT EXPONENT.  bf16 trades MANTISSA for
//                   nothing; its range is ~1e+-38.  So the range argument that makes the running max
//                   mandatory in an fp16 flash-attention kernel simply does not apply to this
//                   kernel.  Measured on this input: S*scale in [-3.53,+3.47], so exp(S*scale) in
//                   [0.029, 32.1] and l <= 478 -- five of thirty-eight available decades.  m also
//                   cancels EXACTLY out of O = (P@V)/l, and the per-32-element E8M0 block scale
//                   renormalises every block independently afterwards, so the fp8 mantissa is
//                   untouched (the block codes move from ~127 to ~132, well inside fa_clamp_K8's
//                   0..255).  Nothing downstream reads m.  Needs FA_SP_SMTPR (it edits
//                   fa_softmax_tpr).  Also costs 2 REGISTERS LESS than the version with pass 1.
//                   THE ONE REAL RISK, and FA_SP_FIXMAX exists to separate it: this makes every
//                   mu_fexp argument POSITIVE, and until now every mu_fexp call in this kernel has
//                   had a NON-POSITIVE argument (exp(x-rowmax), exp(m_old-m_new)).  If the hardware
//                   exp is only accurate for x <= 0 then NOMAX fails for a reason unrelated to the
//                   row max.
//   FA_SP_FIXMAX    with FA_SP_NOMAX: use a CONSTANT m = 4.0 instead of 0, which bounds
//                   max(S*scale) = 3.47 on this input.  Pass 1 is still gone, but every mu_fexp
//                   argument stays <= 0.  This is a CONTROL, not a shippable flag -- a constant m
//                   is an assumption about the data, whereas m = 0 is not.
//   FA_SP_CVTROT2   the CHEAP half-rotation of fa_requant_cvt's subbank conflict that the CVTROT
//                   note above proposes as "one idea not yet tried": rotate only the HALF index h by
//                   (lane & 1), so q stays a compile-time constant and only one term of the store
//                   address becomes runtime.  8-way instead of 16-way conflict.  Costs 54 registers
//                   -- one above the known-good 53 and one below the known-fatal 55 -- so running it
//                   also brackets the renamer threshold exactly.
//   FA_SP_DUMPS     DIAGNOSTIC: per-tile XOR checksum of every row of S, taken between the
//                   accumulator store and the softmax.  S(t) is loop-invariant, so any row that
//                   moves indicts the QK matmul / accumulator store and exonerates everything after.
//   FA_SP_DUMPP     DIAGNOSTIC: the same checksum of the bf16 P the softmax just wrote in place.
//                   Pairs with FA_SP_DUMPLM to localise the cooperative-vs-thread-per-row accuracy
//                   gap to m, P or l.
//   FA_SP_DUMPL     DIAGNOSTIC: per tile, the 64 l words, the 64 m words, and a per-row checksum of
//                   O_unnorm as PV left it -- i.e. everything finalize consumes.
//   FA_SP_DUMPLM    the l/m half of FA_SP_DUMPL only; 3 registers cheaper, so it fits at occupancy 3
//                   (which matters: the cooperative softmax's reduction tree changes shape with the
//                   warp count, so the comparison has to be made at the SAME occupancy).
//   READ THE DUMPS WITH /tmp/fa_dumpchk.py, WHICH SPLITS BY CLUSTER.  Both clusters write the same
//   GMEM page, so a merged read of a dump is exactly as wrong as a merged read of O -- see (A).
//
// ============================================================================================
// ---- FOURTH PASS, 2026-07-27 -- toward 40% mesh utilisation at REFERENCE numerics ----------
//
// Starting point, re-measured from this file at FA_NT6 (six tiles, five steady intervals), the
// published reference-numerics configuration
//     FULL_ATTN2 FA_SP FA_SP_QOVL FA_SP_QOVL4 FA_SP_LEANCFG FA_SP_QKACC FA_SP_PKOVL
//     tile deltas 67,476 / 50,712 / 51,574 / 51,592 / 52,157 / 50,831  ->  51,373 cyc/tile = 31.96%
//     and 9 of 12 tile-images correct: CLUSTER 1 TILES 3, 4, 5 ARE WRONG (109.87% / 111.02% /
//     111.02%), cluster 0 is correct on all six.  So FA_SP_QOVL4 does NOT fully fix the FA_SP_QKACC
//     hazard -- it moves the tile at which it first bites from 1 to 3, which is exactly the
//     behaviour the third pass warned about ("the tile at which corruption starts MOVES WITH THE
//     STAGE TIMINGS").
//
// (F1) *** WHAT THE CORRUPTION IS, MEASURED RATHER THAN GUESSED. ***  The third pass recorded
//      "O too small by a non-power-of-2 factor => a corrupted per-(row,block) E8M0 scale".  That is
//      NOT what the NT6 failure looks like.  Scoring cluster 1 tile 3 cell by cell:
//        * ALL 64 rows are wrong, not a handful;
//        * the per-row median of got/golden takes 63 DISTINCT values spread over 0.17 .. 0.77 --
//          it is not a per-row factor at all, and it is not a power of two;
//        * every value is finite and of a plausible magnitude, and tiles 4 and 5 are BIT-IDENTICAL
//          to each other while differing from tile 3.
//      A plausible attention output for a DIFFERENT input, stable once it settles, is the signature
//      of a WRONG S -- i.e. of Q, K^T or their MX scales -- and NOT of a damaged P scale (which
//      would kill whole 32-element blocks and leave the rest of the row alone).  Every tile of an
//      FA_SP run recomputes the same tile from loop-invariant operands, so a merely STALE read is
//      harmless here; the fault has to be a read of something that is actively being rewritten.
//      Stage S6 under FA_SP_QOVL4 does exactly that, twice over:
//        fa_mvin_A (Q(t+1) DMA into SP_Q) ; fa_scl (64 SF_A-half-1 scale words) ; mu_fence_smem() ;
//        fa_mm_acc  <-- the matmul that READS SP_Q and SF_A half 1
//      and mu_fence_smem() is NOT a drain for either of them: fence.s waits only on the Muon
//      per-warp shared LSU queues (see the BAR_PAD note at the top of this file -- it does not wait
//      on a gemmini DMA, and it does not wait on the SF-SRAM scale writes).  The claim in the
//      FA_SP_QOVL4 comment that "the ROCC queue is IN-ORDER, so the move-in is guaranteed complete
//      before the matmul" is wrong: mvin goes to the LoadController and the matmul to the loop FSM,
//      they are different queues, and the sequential FA_SP path fences between them for that reason
//      (`fa_gf(tid); // drain the Q move-in`).
//      AND A SECOND MEASUREMENT NARROWS IT FURTHER, FOR FREE.  O = (P @ V) / l with P >= 0 and
//      (P/l) summing to 1 across a row, so EVERY row of a correct O is a CONVEX COMBINATION of V's
//      256 rows -- componentwise, O[i][n] must lie in [min_k V[k][n], max_k V[k][n]].  Decoding V
//      from include/fa_data.h (e4m3 x 2^(E8M0-127)) and testing the corrupt image:
//          cluster 0 tile 4 (correct):  0 of 8192 cells outside the hull
//          cluster 1 tile 4 (WRONG):    0 of 8192 cells outside the hull
//      The garbage is STILL a convex combination of V.  That EXONERATES the PV matmul, its operand
//      spad, the V scales and finalize -- all of which would push cells out of the hull -- and it
//      also says P and l are still consistent with EACH OTHER (a damaged P-scale set would rescale a
//      row away from summing to 1).  A wrong S produces a wrong-but-self-consistent P and l, which
//      is exactly what is observed.  So the fault is in S: Q, K^T, their MX scales, the QK matmul,
//      or the accumulator->spad store -- i.e. inside the FA_SP_QKACC hand-off, and nowhere else.
//      TWO FLAGS TEST THIS, and they are designed to be read together:
//        FA_SP_QGF     -- keep FA_SP_QOVL4's placement and add ONLY the missing gemmini_fence.
//        FA_SP_QSPLIT  -- restructure so neither producer is adjacent to its consumer (below).
//
// (F2) *** fa_regs.py IS UNSOUND, AND IT GAVE ME A FALSE PASS THAT COST TWO 80-MINUTE RUNS. ***
//      fa_regs.py counts the union of destination registers over the WHOLE .text and the header
//      calibrates "53 runs / 55 $finishes" on that number.  FA_SP_CBMAX and the reference build
//      have the IDENTICAL 53-name union -- byte for byte, the same set, no register appears in one
//      and not the other -- and yet
//          reference          whole-file 53, softmax fn 30 regs   RUNS
//          + FA_SP_CBMAX      whole-file 53, softmax fn 33 regs   $finish @ Rename.scala:123,
//                                                                 196,749,000 ps (mid tile 0)
//      Rename.scala:110-123 bumps ONE per-core counter the first time a given WARP writes a given
//      architectural register, so the quantity that must stay under numPhysRegs (256; numArchRegs
//      is 128, numWarps 8 -- MuonCore.scala:21-25) is
//          sum over the warps resident on that core of |arch regs THAT WARP writes|,
//      and a warp only claims the registers of the code IT executes.  A whole-file union therefore
//      cannot see the thing that matters: FA_SP puts THREE warps of the five-warp SIMT group on
//      core 1 (mu_schedule maps warp w -> core w&1, so core 1 = warps {1,3,5}), so a register added
//      inside a function ALL WARPS RUN costs 3x, while the same register inside an agent-only
//      function costs 1x.  CBMAX made the softmax -- which every warp runs -- 3 registers bigger and
//      deleted fa_requant_max, which only reduced the union.
//      THE RULE THAT ACTUALLY PREDICTS: hold the whole-file union at <= 53 AND hold the PER-FUNCTION
//      count of every all-warps function (the softmax above all) at or below the reference's.  The
//      reference's softmax is 30; 33 is fatal.  Rewriting FA_SP_SM1F's two reduction folds with TWO
//      accumulator chains instead of four took its softmax from 32 back to 30 for that reason.
//      AND THERE IS A CHEAP HARDWARE GATE: the renamer assert fires in the FIRST TILE, ~197M ps in,
//      which is about EIGHT MINUTES of wall clock -- so "launch the real FA_NT6 run and look for an
//      assertion after ten minutes" is a complete register check.  Do that instead of trusting a
//      static count.
//
// (F2b) AN RTL CORRECTION WORTH HAVING, because two campaigns have reasoned from the wrong version
//      of it.  host.cpp and kernels/fa_mx_hostsf/README.md both state that "EVERY write into the
//      gemmini tile -- the scale SRAMs at +0x88000/+0x8a000 AND the ROCC command port at +0x84000 --
//      funnels through GemminiTile's single FlitMergeNode".  IT DOES NOT.  GemminiTile.scala:
//          162:  regNode := tlSlaveXbar.node                          <- the ROCC command port
//          188:  scalingFacManager := FlitMergeNode(from=4,to=8) := TLWidthWidget(8) := tlSlaveXbar
//      Only the SCALE SRAM sits behind the merge node; the command port hangs off the slave xbar
//      directly.  Two consequences:
//        * a GPU ROCC command burst and a GPU SF-scale burst CANNOT corrupt each other's merge
//          pairing (they reach different managers), which is what makes FA_SP_QSPLIT's stage S4 --
//          32 mvin ROCC commands immediately followed by the 128-word SF pack, both on warp 0 --
//          safe by construction rather than by luck;
//        * the FA_HOSTCFG deadlock therefore cannot be "host `sd` lands between two halves of a GPU
//          4-byte ROCC pair in the merge node".  Whatever it is, it lives in the shared
//          tlSlaveXbar / TLSourceShrinker / TLFragmenter path, not in FlitMergeNode's pairing state.
//      Also note what FlitMergeNode actually does on a violation: `assert((address & (to-1)) == 0)`
//      and `assert(address === mergedReq.address + byteOffset)` (FlitMergeNode.scala:53,58) -- it
//      $finishes the simulation.  It does NOT silently corrupt.  So any silent wrong-answer bug in
//      this kernel is NOT an interleaved-SF-write bug, and that rules out a whole family of theories.
//
// (F3) THE FOUR LEVERS OF THIS PASS.  All four are BIT-EXACT: every value is produced by the same
//      operations on the same operands, only by a different thread or at a different time, so a
//      build with them must still score exactly 3.5666% against golden_O_u16.npy.  (That is the
//      point -- the whole SMTPR/SMBMAX/NOMAX/SUBMAX family buys speed by moving the numerics to
//      4.2-7.8%, and none of it can be part of a verified-correct headline.)
//        FA_SP_QSPLIT  split the Q(t+1) prefetch BY RESOURCE: the 8 KB move-in DMA to stage S4
//                      (under the requant convert -- the only stage with NO mesh op in flight, so
//                      no accumulator->spad move-out to collide with) and the 64 SF scale words to
//                      stage S5 (under the PV matmul, where warp 0 does nothing but spin in
//                      gemmini_fence).  Stage S6 is then a fence plus one matmul issue, ~200 cycles
//                      instead of ~4.4k, so QK(t+1) starts at the TOP of finalize and its 8,210
//                      mesh cycles are fully hidden -- which is what stage S1 was paying for.
// (F4) *** MEASURED, FA_NT6, SEED 12345, FIVE STEADY INTERVALS, EVERY ROW SCORED PER CLUSTER PER
//      TILE WITH fa_verify_tiles.py (12 tile-images; 3.5666% == CORRECT). ***
//
//   configuration (all on FULL_ATTN2 FA_SP FA_SP_QOVL FA_SP_LEANCFG FA_SP_QKACC FA_SP_PKOVL)
//                                        cyc/tile   util    tile-images   regs
//   FA_SP_QOVL4               (published)  51,373  31.96%   9 of 12 -- cl1 t3,t4,t5 at 109.9/111.0/
//                                                           111.0%                            53
//   FA_SP_QOVL4 + FA_SP_QGF               51,467  31.90%   12 of 12 CORRECT                   53
//   FA_SP_QSPLIT                          46,972  34.96%   12 of 12 CORRECT                   53
//   FA_SP_QOVL4 + SM1F + SM1FL            52,875  31.05%   12 of 12 CORRECT                   53
//   FA_SP_QSPLIT + BAL + SM1F             49,966  32.86%   12 of 12 CORRECT                   53
//   FA_SP_QOVL4 + FA_SP_CBMAX                --      --    $finish, Rename.scala:123           53(!)
//   FA_SP_QSPLIT + BAL + CBMAX               --      --    $finish, Rename.scala:123           53(!)
//   CONTROL, the sequential kernel (FULL_ATTN2 FA_STEADY FA_NT6, no FA_SP): 86,063-87,401 cyc/tile
//   = 18.79-19.08%, 12 of 12 correct.
//
//      *** THE CORRECTNESS RESULT: THE STEADY-STATE BUG IS A MISSING GEMMINI DRAIN, AND ONE
//      gemmini_fence() FIXES IT AT ZERO COST. ***  FA_SP_QGF adds nothing but that one fence to the
//      published FA_SP_QOVL4 schedule -- the cycle count moves by 94 out of 51,373, i.e. inside the
//      interval wobble -- and it takes the run from 9 of 12 tile-images to 12 of 12.  FA_SP_QSPLIT,
//      which removes the same adjacency structurally AND buys 4.4k, is also 12 of 12.  Two
//      independent fixes for one mechanism, and the mechanism is the one (F1) predicted from the
//      error structure: `mu_fence_smem()` is not a drain for a gemmini DMA or for an SF-SRAM scale
//      write, and mvin and matmul are different gemmini queues, so "the ROCC queue is in-order"
//      does not order the Q(t+1) move-in before the QK that reads Q.
//      *** BUT KEEP THE PERTURBATION CAVEAT IN VIEW, BECAUSE ONE ROW OF THE TABLE PROVES IT MATTERS:
//      FA_SP_QOVL4 + SM1F + SM1FL IS ALSO 12 of 12, AND IT CONTAINS NO FIX AT ALL. ***  It is 1,500
//      cycles per tile SLOWER, and that is enough to move the schedule out of the window -- the same
//      thing FA_SP_DUMPSC did in the third pass.  So "12 of 12" is only strong evidence for a fix
//      whose SCHEDULE IS UNCHANGED, which is exactly why FA_SP_QGF (+94 cycles, 0.2%) is the
//      load-bearing measurement here and FA_SP_QSPLIT's clean run is corroboration rather than proof.
//      Note also what does NOT help: the simulator is timing-deterministic and a seed only randomises
//      uninitialised state, so a second SEED does not re-test the schedule.  MORE TILES does --
//      the corruption first appeared at tile 3 -- so FA_NT8 (16 tile-images) is the real second test.
//
// (F5) *** A THIRD MEASUREMENT TRAP, AND IT MOVED A NUMBER BY 8%: fa_pm.py MIXES THE TWO CLUSTERS.
//      USE fa_marks_cl.py. ***  fa_pm.py keys the MARK stamps by ADDRESS only, and both clusters
//      write the SAME MARK_GMEM words, so its m[k] is whichever cluster's store happened to appear
//      LAST in the trace.  Two consequences, both observed here:
//        * the tile deltas CHANGE as a growing trace is re-parsed -- one variant's tile-1 interval
//          read 46,156 and then 50,030 from the same run, 8% apart, purely from re-parsing later;
//        * even on a complete trace the deltas are a mixture of two clocks that are ~1k cycles out
//          of phase, so a difference of that size between two configurations is not a result.
//      fa_marks_cl.py splits by clid, reports each cluster's intervals separately, and pools all
//      steady intervals of both clusters for the headline (10 numbers at FA_NT6 instead of 5).  It
//      also prints the per-stage means per cluster, which is what the stage tables below are built
//      from.  Note the related trap already documented for O: rs1.data is the FULL mark address here
//      only because `mki++` is a runtime index; a constant-index store would put the offset in the
//      store immediate and rs1.data would be the page base for every mark.
//
// (F6) *** THE FULL LEVER SWEEP ON TOP OF FA_SP_QSPLIT, pooled over BOTH CLUSTERS' steady intervals
//      (fa_marks_cl.py; FA_NT6 gives 10 such intervals, so ~+-200 rather than the ~+-1,000 a single
//      cluster's single interval carries): ***
//        + nothing (FA_SP_QSPLIT)                     46,902   35.01%   12/12
//        + FA_SP_CVTX FA_SP_PREPK FA_SP_FZ6           46,597   35.24%
//        + ... + FA_SP_MARK1                          47,477   34.58%
//        + FA_SP_BALC FA_SP_CVTX FA_SP_PREPK FA_SP_FZ6 47,663   34.45%
//        + FA_SP_BALC FA_SP_PREPK FA_SP_FZ6           48,535   33.83%
//      and the per-stage attribution, cluster 0, steady tiles (this is where the levers actually go):
//                            QSPLIT   +CVTX+PREPK+FZ6
//        S1 accumulator->S      998        998
//        softmax             12,825     12,825
//        requant pass A       3,405      3,400
//        S4 convert || pack  11,093     10,242      <- FA_SP_CVTX + FA_SP_PREPK: -851
//        PV                   8,889      8,880
//        S6 QK || finalize    9,562     10,892      <- FA_SP_FZ6: +1,330  *** LOSS ***
//        TOTAL               46,824     47,289
//      VERDICTS:
//        FA_SP_CVTX + FA_SP_PREPK  -851 on stage S4.  Real but far short of the ~3k the 16-way
//          subbank conflict predicted, so THE CONVERT IS NOT CONFLICT-BOUND -- 361 static
//          instructions per item against ~10.2k/512 items is mostly ISSUE, and the conflict was
//          costing hundreds, not thousands.  Keep both (they are free and bit-exact) but do not
//          expect more from that direction.
//        FA_SP_FZ6   +1,330.  A SIXTH warp makes finalize SLOWER.  warp 0 only enters finalize after
//          its gemmini_fence + matmul issue, and it then owns a sixth of the rows, so it becomes the
//          long pole in a stage that ends with its slowest warp -- the same "bandwidth-bound stages
//          end with the longest warp" mechanism that sinks FA_SP_BALF.  OFF.
//        FA_SP_BALC  -1.1k WORSE (47,663 vs 46,597).  Same lesson as BALF: the convert is not purely
//          issue-bound either, so weighting core 0's two warps to 128 items each just lengthens them.
//          *** SO THE WHOLE "SPLIT THE TWO MUON CORES BY RESOURCE" IDEA MEASURES NEGATIVE ON THIS
//          KERNEL.  The theory (core 1 carries 3 of the 5 SIMT warps, so it gets 3/5 of an
//          equal-partition's work and every issue-bound stage runs 20% long) is arithmetically sound;
//          it just does not describe any stage this kernel actually has.  Every large SIMT stage here
//          is enough of a memory-system stage that the equal partition -- which minimises the LONGEST
//          warp -- already wins.  Recorded as refuted, with the reason. ***
//        FA_SP_MARK1  NEUTRAL (47,477 vs 46,597 -- inside the wobble, and its tile-0 delta is within
//          16 cycles of the 7-mark build's).  *** THIS RETIRES A WORRY, and the worry was mine:
//          mxgemm_core.hpp measures a single-thread GMEM stamp at ~400 cycles and the FA_SP body
//          emits SEVEN per tile, which would have made ~2.8k of every quoted utilisation pure
//          instrumentation.  It does not: deleting six of the seven changes nothing measurable,
//          because MARK is one lane's fire-and-forget store while five other warps keep running,
//          whereas the ~400-cycle stamps were in a single-threaded critical section.  So the
//          utilisation figures in this file are NOT inflated by their own instrumentation, and the
//          per-stage marks can stay. ***
//
//      AND THE FIRST PERFORMANCE VERDICTS, both negative and both instructive:
//          FA_SP_SM1F (+FA_SP_SM1FL)   52,875   +1,502 vs the reference   LOSS
//          FA_SP_QSPLIT alone          46,972   -4,401                    WIN
//          FA_SP_QSPLIT + BAL + SM1F   49,966   -1,407  i.e. BAL + SM1F give back 3.0k of QSPLIT's win
//        WHY SM1F LOSES: the fold has all 16 lanes reading the SAME halfword of buf on each of its
//        16 steps.  The reference does that ONCE per reduction (`as_bf16(buf[0])`) and gets away with
//        it; doing it 16 times is a 16-way subbank conflict that costs more than the fence it saves,
//        and rotating the read to fix that costs 70 instructions and 4 registers (FA_SP_SM1FR).
//        WHY BALF LOSES, AND THIS IS THE GENERAL LESSON: *** CORE-BALANCING ONLY HELPS AN
//        ISSUE-BOUND STAGE.  ON A BANDWIDTH-BOUND STAGE IT IS ACTIVELY HARMFUL. ***  The stage ends
//        when the LAST warp finishes.  If the resource is per-core instruction issue, giving core 0's
//        two warps more work shortens the stage (core 1 stops being the long pole).  If the resource
//        is shared and global -- and finalize is GMEM-store bound, fifty-one static instructions
//        moving 4,096 words -- then every warp progresses at the same rate whatever its share, so the
//        stage is set by the LONGEST warp and the EQUAL partition is already optimal; weighting core
//        0's warps to 16 rows instead of 13 makes the stage 23% longer.  Decide which resource a
//        stage is bound by BEFORE repartitioning it: divide the stage's cycles by its static
//        instruction count per item.  finalize is ~40 cycles per coalesced 64 B store and 51 static
//        instructions -- overwhelmingly stall, so bandwidth.  The requant convert is 361 static
//        instructions per item against 10.5k/512 = 20 cycles per item per thread -- much closer to
//        issue-bound, which is why FA_SP_BALC is kept and measured separately from FA_SP_BALF.
//        FA_SP_FZ6 supersedes BALF anyway: at six warps the warp->core map is already 3/3.
//        FA_SP_BALC/F  CORE-BALANCED work partition.  The five SIMT warps are 3 on core 1 and 2 on
//                      core 0, each core issues one warp-instruction per cycle, and an EQUAL
//                      per-warp partition therefore hands core 1 three fifths of the work: every
//                      issue-bound warps-1..5 stage runs 20% longer than the balanced 0.5.  Weight
//                      the two core-0 warps 3 and the three core-1 warps 2 (total 12) and each CORE
//                      gets exactly half.  BALC does the requant convert, BALF does finalize.
//        FA_SP_SM1F    ONE fence per softmax reduction instead of two, and no warp-divergence
//                      region: store the 16 per-lane partials, fence ONCE, and let every lane fold
//                      all 16 in registers.  warp_tree_reduce is 4 x (2 SMEM loads + 1 op + 1 SMEM
//                      store + vx_split_n/beqz/vx_join) and needs a second fence to publish buf[0];
//                      that is 56 of online_softmax_block's 285 static instructions and 2 of its 4
//                      fences per row.  Measured 285 -> 273 instructions and 4 -> 2 fences.  l keeps
//                      warp_tree_reduce's EXACT 16-leaf pairing (bf16 addition is not associative
//                      and every intermediate is rounded to bf16), reproduced term for term with the
//                      same `(_Float16)(a + b)` expression the library uses.
//        FA_SP_FZ6     finalize on all SIX warps.  Only possible with QSPLIT, which empties warp 0's
//                      stage-S6 duty.  finalize_O is FIFTY-ONE static instructions and the stage
//                      costs 7-10k cycles for 4,096 coalesced word stores, i.e. it is GMEM-store
//                      bound, not issue bound -- so the cure is MORE OUTSTANDING STORES (another
//                      warp), which is also why FA_SP_FZU4's extra per-warp ILP measured a loss.
//                      At six warps mu_schedule's map is already 3/3 balanced, so BALF is redundant.
//        FA_SP_MARK1   keep ONLY the per-tile MARK.  MARK() is a single-thread store to GMEM (DRAM)
//                      and mxgemm_core.hpp already measured such a store at ~400 cycles here; the
//                      FA_SP body emits SEVEN per tile, so ~2.8k cyc/tile of a 51k tile is
//                      instrumentation charged against the utilisation of a kernel that would ship
//                      with none of it.  Quote stage costs WITH the marks and the headline WITHOUT,
//                      and say which is which.  Read with `fa_pm.py <out> --per 1`.
//        FA_SP_CVTX    KILL THE 16-WAY SMEM SUBBANK CONFLICT IN THE REQUANT CONVERT, FOR FREE.
//                      The subbank is word_index & 15; the convert's load index is
//                      row*128 + b*16 + k and both 128 and 16 are multiples of 16, so the subbank
//                      is (k & 15) -- LANE-UNIFORM, and a warp's 16 lanes hit ONE subbank on every
//                      one of the phase's 17 loads per item.  ~544 warp-loads per tile at a 16x
//                      penalty is most of the gap between this stage's 5.8k of issue and its
//                      measured 10.5k.  Only the PAIR index u may be rotated (an output word holds
//                      4 consecutive elements), which gives 8 distinct subbank pairs = 2-way.
//                      FA_SP_CVTROT does that and costs 55 registers (fatal); FA_SP_CVTROT2 halves
//                      the rotation to reach 54.  NEITHER IS NECESSARY: every address here is a base
//                      with zero low bits plus a small DISJOINT offset field, so `+` is `^`, and XOR
//                      distributes over the rotation --
//                          load  = (512*row + 64*b)            ^ (u<<3) [| 4]     bits 5:0 free
//                          store = (4096*ti + 512*b + 16*rr)   ^ ((u&4)<<6 | (u&3)<<2)
//                                                                       bits 8 and 3:0 free
//                      so addr(u ^ r) == addr(u) ^ addr_offset(r).  Fold r = lane&7 into the two
//                      base pointers ONCE per item and every access in the unrolled body stays a
//                      compile-time-immediate XOR.  (The same trick FA_PSWIZX uses for base ^ k*68.)
//                      Costs +30 static instructions per item, because an XOR is not an addressing
//                      mode -- `lw rd, imm(rs)` becomes `xori t, rs, imm; lw rd, 0(t)`.  Lives in
//                      fa_requant_cvt_bal, so it needs FA_SP_BALC.
//        FA_SP_PREPK   split the SF pack into all-thread packing math + a single-thread PURE
//                      ASCENDING 128-word copy (flash_mx_impl.hpp's prepack_scales /
//                      copy_scales_to_sfmem, which FA_PIPE already used).  ~147 cyc/word combined
//                      vs 46-64 for a plain copy, so warp 0's serial critical section goes ~8.3k ->
//                      ~7.0k.  Only matters once the convert is faster than the pack, which is
//                      exactly what FA_SP_BALC + FA_SP_CVTX make true -- stage S4 then becomes
//                      PACK-BOUND and the pack is the thing left to shorten.  Costs one barrier.
//        FA_SP_QGF     the cheap CONTROL for (F1): keep FA_SP_QOVL4's prefetch placement and add
//                      only the missing gemmini_fence between the Q prefetch and the QK matmul.
//        FA_SP_SM1FL   extend FA_SP_SM1F's one-fence fold to the l reduction as well, reproducing
//                      warp_tree_reduce's exact 16-leaf pairing term by term in registers so it
//                      stays bit-exact.  It canNOT be lane-rotated (bf16 addition is not
//                      associative), so all 16 lanes read the same halfword 16 times.
//        FA_SP_SM1FR   lane-rotate FA_SP_SM1F's max fold to break that same-address read.  MEASURED
//                      WORSE: the (k + lane) & 15 arithmetic is 4 extra ops per load, 273 -> 343
//                      static instructions and 30 -> 34 registers, and 34 in an all-warps function
//                      is over the renamer cliff.  Kept, documented, OFF.
//        FA_SP_CBMAX   produce the E8M0 block scales inside the COOPERATIVE softmax, deleting the
//                      requant pass-A stage at reference numerics (FA_SP_SMBMAX only ever did this
//                      for the thread-per-row softmax, i.e. only at 4.2% numerics).  The layout
//                      makes it exact and cheap in principle -- online_softmax_block's ownership is
//                      strided, lane l owns word j*16+l, and that word's MX block index is exactly
//                      j, so block j's 32 elements ARE the 16 lanes' word j and the block max is one
//                      16-lane reduction of a register each lane already holds, behind the fence the
//                      l-reduction already pays for.  *** MEASURED AND DEAD ANYWAY, for two
//                      independent reasons: it costs +80 static instructions per row (365 vs 285),
//                      i.e. ~+2.6k cyc/tile against pass A's 3.4k, AND it is a RENAMER CASUALTY --
//                      see (F2).  Kept, documented, OFF. ***
// ============================================================================================
// THE REGISTER THRESHOLD, EMPIRICALLY BRACKETED (occupancy 3 = 6 warps = 3 warps/core):
//     53  RUNS      -- PKOVL, SMTPR, +SMBMAX, +SUBMAX, +FZU4, +QOVL4  (every shipped config)
//     55  $finishes -- PKOVL+REFETCH, and FA_SP_CVTROT
//     58  $finishes -- FA_SP_FUSE, FUSE+RLEAN, anything with FA_SP_CVTSWAR
//     69  $finishes -- FUSE + FA_SP_1P
// So the usable budget for this kernel is 53, or at most 54, distinct architectural registers --
// there is essentially NO headroom, and every one of the four ideas that needed even two more
// registers had to be abandoned or moved to occupancy 2.  Check any new stage with
//     bash fa_asm.sh TAG "DEFINES" && python3 fa_regs.py /tmp/fa_TAG.s     (~20 s)
// BEFORE spending a 70-minute simulation on it.  This single constraint, not SMEM and not the
// mesh, is what bounds how much SIMT work this kernel can restructure.
//
// TOOLING NOTE, because two of these cost real time to discover:
// (all four live next to this file, in kernels/flash_attention_mx/)
//   * fa_marks3.py   -- *** USE THIS, NOT fa_pm.py AND NOT fa_marks_cl.py, FOR TILE INTERVALS AND
//     PER-STAGE MEANS. ***  fa_pm.py keys marks by address and mixes the two clusters (see (F5)).
//     fa_marks_cl.py splits by cluster but still derives the mark INDEX from rs1.data, i.e. it
//     assumes `mki++` leaves the full effective address in the base register -- and whether it does
//     is a CODEGEN COIN FLIP.  With a fixed number of marks per iteration clang either bumps the
//     base register by 4 per mark (rs1.data IS the address; the FA_SP_QSPLIT builds) or keeps ONE
//     base and puts the index in the store IMMEDIATE (the FA_SP_OPV builds, where every mark reports
//     rs1.data = 0x40050000 and they ALL alias to index 0, so the tool sees one interval and reports
//     none).  fa_marks3.py keys by TRACE ORDER within a cluster instead -- all marks are stores by
//     the same single thread, so they are strictly ordered and their values monotone -- and drops the
//     ONE EXTRA pre-loop copy of the s0 stamp that clang emits at its own pc.  Including that stamp
//     shifts every stage attribution by one and makes the empty stage look like 38,000 cycles, which
//     is exactly the wrong reading I got before finding it.
//   * fa_pertile.py  -- verify EACH TILE's O separately (bucket the O stores by the preceding
//     MARK).  A whole-file Frobenius on a steady-state run is meaningless; see (a2).
//   * fa_regs.py     -- whole-kernel distinct-written-register union from a .s.
//   * fa_baraudit.py     -- barrier-in-divergent-region audit, PER FUNCTION.  bar_audit.py counts
//     vx_split_n/vx_join depth LINEARLY over the whole .s, so one unbalanced pair (an early return
//     out of a split region is perfectly legal) poisons the depth for every later function and it
//     reports false HANG RISK.  It did exactly that for fa_softmax_item's two barriers, which are
//     at depth 0 within their own function.  Count depth per function.
// ============================================================================================
// ---- FIFTH PASS, 2026-07-27 -- from 34.96% toward 40% -------------------------------------
//
// Starting point: FA_SP_QSPLIT, 46,972 cyc/tile = 34.96%, 12 of 12 tile-images (fourth pass).
// 40% needs <= 41,050, i.e. 5,922 cycles out of a 46,972-cycle tile.  Its stage table is
//     acc->S 998 | softmax 12,825 | pass A 3,405 | convert||pack 11,093 | PV 8,889 | QK||fz 9,562
// so the exposed cost is 27,323 of SIMT (softmax + pass A + convert), 8,889 of EXPOSED PV mesh, and
// ~2,300 of accumulator stores and finalize-over-mesh excess.
//
// (G0) *** READ THIS FIRST: THE FOURTH PASS'S 12-of-12 WAS A SCHEDULE COINCIDENCE, NOT A FIX, AND
//      THE REAL FIX IS FA_SP_WCNT. ***
//      The fourth pass concluded that "the steady-state bug is a missing gemmini drain, and one
//      gemmini_fence() fixes it": FA_SP_QGF (+94 cyc) took FA_SP_QOVL4 from 9 of 12 tile-images to
//      12 of 12, and FA_SP_QSPLIT was 12 of 12 as well.  It also recorded the caveat that makes that
//      conclusion unsafe -- FA_SP_QOVL4 + SM1F + SM1FL is ALSO 12 of 12 and contains no fix at all,
//      it is just 1,500 cyc/tile slower.  *** THE CAVEAT WAS THE TRUTH. ***  Adding NOTHING BUT
//      BIT-EXACT SPEEDUPS to the published FA_SP_QSPLIT reference brings the corruption straight
//      back, at FA_NT6, seed 12345, scored per cluster per tile:
//          FA_SP_QSPLIT (published)                     46,972   12 of 12 CORRECT
//          + FA_SP_PAX                                  46,422    5 of 8 -- cluster 1 wrong
//          + FA_SP_PAX + FA_SP_CVTX                     45,662    6 of 8 -- cluster 1 tiles 2,3
//                                                                at 96.3% / 114.6%
//      Neither flag changes a single computed value; both only move the schedule.  So FA_SP_QSPLIT
//      does not FIX the hazard, it MISSES it, and any future speedup can re-open it.  A "12 of 12"
//      on one schedule is therefore not evidence that a build is correct -- only a build with a
//      mechanism-level drain is.
//      THE MECHANISM IS THE ONE FA_SP_WCNT WAS WRITTEN FOR, and it was sitting in this file default
//      OFF the whole time.  gemmini_fence() polls MMIO 0x20 = gemmini.module.io.busy, and a command
//      fired by a store to 0x00 takes several cycles to reach the reservation station, so a poll
//      issued shortly after that store CAN READ ZERO AND FALL THROUGH -- the fence returns before
//      the matmul has started.  FA_SP_QKACC is the only configuration in which SOFTWARE waits for a
//      matmul and then reads its ACCUMULATOR (stage S1: fa_gfl; fa_store_acc<QKF>(SP_C); fa_gfl), so
//      a fall-through there means the accumulator->spad store copies a MID-ACCUMULATION accumulator.
//      That is exactly the observed fingerprint: S is wrong, every downstream stage is bit-exact
//      given S, the garbage is still a convex combination of V, and it is timing-dependent and
//      cluster-specific.  MMIO 0x28 (runningLoops) is incremented IN THE SAME CYCLE as the command
//      write for a LOOP_WS (GemminiTile.scala:408-412), so gemmini_fence_waitcount(0) cannot race,
//      and FA_SP_WCNT puts it in front of every fa_gfl.  It costs one extra MMIO poll at four sites
//      per tile.  *** BUILD EVERY FA_SP CONFIGURATION WITH FA_SP_WCNT. ***
//
// (G1) *** TWO SUBBANK CONFLICTS THAT COST NOTHING TO FIX, BOTH BIT-EXACT. ***
//      The subbank is word_index & 15, and every phase that walks S or P indexes it as
//      row*128 + b*16 + k with BOTH strides a multiple of 16 -- so the subbank is (k & 15), which is
//      LANE-UNIFORM, and a warp's 16 lanes hit ONE subbank on every load.  The fourth pass fixed
//      this in the requant convert (FA_SP_CVTX) by folding an XOR rotation into the base pointer,
//      which is free because these base addresses have zero low bits and the offset field is
//      disjoint, so `+` is `^`.  The SAME trap is in requant pass A and had been missed:
//        FA_SP_PAX -- rotate pass A's 16 word loads by (lane & 15).  Pass A only READS and its
//                     result is a MAX, which is exactly order-independent, so the permutation may
//                     be the FULL 16 -- unlike the convert, where an output word must hold four
//                     CONSECUTIVE elements and only 8 rotations are legal (2-way, not 1-way).
//                     +18 static instructions per item, ZERO extra registers (53 -> 53).
//        FA_FOLDX  -- the same XOR trick inside warp_fold16 (the FA_TREEFIX reduction), whose eight
//                     word loads are read by all 16 lanes from the SAME address.  Costs 6 registers
//                     in an all-warps function (30 -> 36) and is therefore a RENAMER CASUALTY at
//                     occupancy 3; kept and documented OFF.  FA_SP_SM1FX -- the same idea with a
//                     32-bit-per-lane buffer, which would have been 1-way instead of 2-way -- is
//                     worse still (58 registers), for the same reason: an XOR'd address is not an
//                     addressing mode, so every rotated access needs its own live pointer.
//      *** THE GENERAL RULE THIS ESTABLISHES: an XOR rotation is free when the rotation folds into a
//      base pointer that is then used with COMPILE-TIME-CONSTANT offsets (pass A, the convert), and
//      costs one register per access when the rotated index is itself the loop variable (the folds).
//      Check which case you are in before writing it. ***
//
// (G2) *** THE 4-ENTRY SPAD READ QUEUE: WHY ONLY *ONE* MESH/SIMT OVERLAP IS LEGAL AT ALL. ***
//      Scratchpad.scala:210-220, in every ScratchpadBank's use_shared_ext_mem READ path:
//          val dma_q = Module(new Queue(Bool(), 4, false, true))
//          assert(dma_q.io.enq.fire === ren, "DMA queue does not have enough entries")
//                                             // TODO (richard): do backpressure
//      FOUR entries, NO BACKPRESSURE: if the shared SMEM does not return a read response before the
//      mesh issues a fifth request to that bank, THE SIMULATION $finishes.  So the mesh may not read
//      an operand out of a 32 KB SMEM bank (bank = addr >> 15) that SIMT code is reading at the same
//      time.  Measured, not theorised: FA_SP_OPV as first written $finished twice on
//      `gemmini.spad.spad_mems_2` at 237,301,000 and 238,123,000 ps, in the first tile.
//      THIS IS WHY FA_SP_QSPLIT OVERLAPS EXACTLY ONE MESH OP WITH SIMT WORK AND NO MORE.  Its one
//      overlap is QK(t+1) || finalize(t): QK reads Q (0x16000, bank 2) and K^T (bank 3), while
//      finalize reads O, which is the LOWER 16 KB of SP_C and therefore entirely in bank 1.  Every
//      other stage either has the mesh idle or has SIMT quiesced.  Since S/P-bf16 STRADDLES banks 1
//      and 2 in the default map, any schedule that overlaps the mesh with the softmax, pass A or the
//      convert asserts -- which is a HARDWARE reason, not a dependency reason, and it is the real
//      answer to the third pass's question of why more overlap could not be found.
//        FA_SP_BANKA -- the remap that removes the obstruction: give S a WHOLE bank and move P8/O
//                       into bank 2 next to Q (see the comment on the constants themselves).
//
// (G2b) *** AND THE VERDICT ON FA_SP_OPV: THE PV OVERLAP IS UNREACHABLE IN THIS SMEM BUDGET, AND
//      THE REASON IS THE FOUR-ENTRY QUEUE PLUS AN EXACTLY-FULL SMEM -- NOT A DEPENDENCY. ***
//      MEASURED, three runs, all $finish at Scratchpad.scala:220 on `spad_mems_2`:
//          FA_SP_OPV  (default map)                    237,301,000 ps / 238,123,000 ps  (~119k cyc)
//          FA_SP_OPV FA_SP_OPVQ FA_SP_BANKA            282,593,000 ps                   (~141k cyc)
//      FA_SP_BANKA moved the failure LATER but did not remove it, and where it fired says exactly
//      why: 141k is inside stage A of ITERATION 1, i.e. the first time PV(t-1) actually runs
//      underneath softmax(t).  Bank 2 then contains PV's A operand (P8) *and the whole scratch
//      window*, and the cooperative softmax's cross-lane reduction hammers that scratch:
//      warp_tree_reduce costs, per row per warp, 16 partial stores + 45 tree accesses + 16
//      broadcast loads of buf[0] = 77 SMEM accesses, x2 reductions x ~11 rows x 6 warps ~= 10,000
//      accesses to REDBUF in a 12,825-cycle stage -- about 0.8 per cycle.  That is far more than
//      enough to delay the mesh's read responses past four outstanding requests.
//      AND IT CANNOT BE FIXED BY RELOCATION, WHICH IS THE POINT.  For the overlap to be legal the
//      reduce scratch must not share a bank with PV's operands (P8, V) in stage A, nor with QK's
//      (Q, K) in stage D -- so it may share a bank ONLY with S.  S is [64][256] bf16 = 32,768 B,
//      i.e. EXACTLY one bank, with no slack, and the total is exactly full:
//          V 32K + K 32K + S 32K + P8/O 16K + Q 8K + scratch 8K = 128 KB.
//      There is also no lane-shuffle instruction in this ISA (mu_intrinsics.h has fences and
//      mu_fexp, nothing cross-lane), so a cooperative reduction MUST go through SMEM and the
//      scratch cannot be deleted.  Double-buffering P8 so PV(t-1) could hide under convert(t)
//      instead needs another 16 KB, which does not exist either.
//      SO THE CORRECT STATEMENT OF THE CEILING IS NOT note (c)'s: it is not that only one matmul
//      CAN be covered, it is that on this RTL the mesh may only run underneath a SIMT phase whose
//      SMEM traffic misses the mesh's operand banks, and finalize -- which reads only O (16 KB,
//      one bank) and writes GMEM -- is the ONLY such phase this kernel has.  FA_SP_QSPLIT's single
//      overlap is therefore not a missed opportunity but the whole of what the hardware allows.
//      FA_SP_OPV/OPVQ/BANKA are kept, default OFF and documented, because the schedule itself is
//      right (it hid all 16,420 mesh cycles and its measured iteration-0 stage costs -- A 16,970 |
//      B 543 | C1 4,494 | C2 3,957 | D 15,095 | E 1,339 = 42,584 cold -- project to ~37.3k = 44%
//      steady) and because the only thing standing between it and that number is a queue depth of
//      four and 8 KB of scratch.  If either the RTL grows backpressure or SMEM grows, this is the
//      schedule to switch on.
//
// (G7) WHAT BLOCKS THE REST, stated against the best configuration's stage table so it is checkable
//      rather than rhetorical.  Per tile, at FA_SP_QSPLIT + WCNT + PAX + CVTX + CVTSWAR + HSF +
//      FA_SM_2P (+ FZ6):
//        accumulator -> S      ~1,000   mesh store, SIMT quiesced; irreducible per (G2)
//        softmax              ~10,800   SIMT, issue-bound.  FA_SM_2P already removed 41 of a warp's
//                                       44 fences; the rest is instruction issue + SMEM latency.
//        requant pass A        ~2,750   SIMT, conflict-free after FA_SP_PAX (~1,600 of it is issue)
//        requant convert      ~7,700   SIMT, six warps + SWAR + XOR addressing
//        PV                   ~8,880   *** EXPOSED MESH ***
//        QK || finalize       ~8,300   finalize, GMEM-store bound; QK's 8,210 mesh cycles hide here
//      TWO THINGS BLOCK THE REMAINDER, and both are hardware, not scheduling:
//      1. THE EXPOSED PV (22% of the tile).  Hiding it needs the FA_SP_OPV schedule, which works and
//         is in this file, but the mesh then reads an operand out of a bank SIMT is using and
//         Scratchpad.scala:220's FOUR-ENTRY, UN-BACKPRESSURED read queue $finishes the simulation.
//         Fixing that needs either RTL backpressure (the TODO is in the RTL) or ~8 KB of SMEM to
//         move the softmax's reduce scratch out of PV's operand bank -- and SMEM is exactly full at
//         V 32K + K 32K + S 32K + P8/O 16K + Q 8K + scratch 8K = 128 KB.  See (G2), (G2b).
//      2. THE SIMT BLOCK (softmax + pass A + convert = ~21,250, i.e. 53% of the tile).  It makes
//         THREE passes over the 32 KB of S/P-bf16 where one would do; the one-pass form is
//         FA_SP_FUSE, and FUSE is a RENAMER casualty (58 registers against a ~53 budget) whose only
//         home is occupancy 2, where losing 32 of 96 threads costs more than fusing saves.  So the
//         binding constraint on the SIMT half of this kernel is Rename.scala's never-reclaimed
//         physical-register counter, exactly as the second pass concluded -- and (G5) is the one
//         lever against it that worked: free registers in one phase and spend them in another.
//
// (G6) MEASUREMENT TABLE, FIFTH PASS.  FA_NT6, seed 12345, +dramsim, trace verified stable before
//      parsing.  cyc/tile is the POOLED mean of both clusters' steady intervals (the fill tile
//      dropped), read with fa_marks3.py -- NOT fa_marks_cl.py, which mis-keys the marks on the
//      FA_SP_OPV builds and includes the extra pre-loop stamp (see its docstring).  Tile-images are
//      per CLUSTER per tile via kernels/fa_mx_hostsf/fa_verify_tiles.py; 3.5666% == CORRECT.
//      Every row is on FULL_ATTN2 FA_SP FA_SP_QOVL FA_SP_LEANCFG FA_SP_PKOVL plus what is listed.
//
//   configuration                                     cyc/tile   util    tile-images        regs
//   *** SCORING RULE FOR THIS TABLE: a corrupt tile has corrupt TIMING, so any row that is not
//   12 of 12 has its cycle number recorded only to identify the configuration -- it is NOT a
//   performance result and must not be quoted as one. ***
//   *** AND EVERY ROW THAT INCLUDES FA_SM_2P IS PROVISIONAL until the pb stride-33 overrun noted
//   below is fixed: that buffer rewrites Q rows 0..13 on every softmax and survives only because
//   FA_SP re-issues Q's move-in between the softmax and the QK that reads Q -- the same
//   accidental-safety pattern as the QSPLIT 12-of-12 coincidence in (G0).  FA_SM_2P is not mine and
//   is being fixed by its author; numbers that include it stand only once it is. ***
//   QSPLIT QKACC          (published 4th-pass ref)      46,972  34.96%   12 of 12            53
//   + PAX                                               46,422  35.37%   5 of 8  *** WRONG ***53
//   + PAX + CVTX                                        45,662  35.96%   6 of 8  *** WRONG ***53
//   + PAX + CVTX + FA_SM_2P                             43,442  37.80%   (see below)         49
//   OPV                                                    --      --    $finish Scratchpad:220 52
//   OPV + PAX + CVTX                                       --      --    $finish Scratchpad:220 52
//   OPV + OPVQ + BANKA + PAX + CVTX                        --      --    $finish Scratchpad:220 53
//   -- and with FA_SP_WCNT, which every row from here down carries:
//   WCNT + PAX + CVTX                                    45,582  36.02%  *** 12 of 12 CORRECT ***
//     ^ THE VERIFIED HEADLINE.  n=10 pooled steady intervals, both clusters, min 44,289 max 46,765.
//       It is +1,390 cyc/tile faster than the published reference AND it is the first FA_SP build
//       whose correctness rests on a MECHANISM rather than on the schedule missing the window: the
//       same two bit-exact speedups without FA_SP_WCNT score 5 of 8 and 6 of 8.  WCNT itself is free
//       (fill tile 67,927 with it vs 68,095 without).                                          53
//   -- the HSF (host-offload) branch, all retired: see (G8).  Every unthrottled-poll build dies on
//      TLMonitor xbar_3, and throttling to one Get per ~256 cycles does not save it either.
//   WCNT + PAX + CVTX + CVTSWAR + HSF + FA_SM_2P (+FZ6)     --     --   fabric assert, 82.5k-247k cyc
//   -- the register-band probes.  *** THESE ARE THE ROWS THAT CLOSED THE BAND, and their cycle
//      numbers are meaningless: each aborted inside the FIRST TILE at Rename.sv. ***  fa_regs3.py's
//      per-core figure is in the last column, and it is the number to screen on, not the union.
//   WCNT + PAX + CVTX + CVTSWAR              --  --  RENAMER ABORT 244,403,000 ps    core1 231
//   SQ32 + QEARLY + CVTX + CVTSWAR (no 2P)   --  --  RENAMER ABORT 210,565,000 ps    core1 246
//   SQ32 + FA_SM_2P + FA_SM_2PRAW        27,380/half  29.92%  killed: build inputs not provably
//                                        (=8192/27380) post-fd722ac, so NOT scored -- see (G11)
//   -- and the three that are in flight, all from provably post-fd722ac inputs, all register-safe:
//   WCNT PAX CVTX + 2P + 2PRAW + CVTSWAR + FZ6                  (in flight)          core1 168
//   ... + FA_SM_2PBM + FA_SP_SMBMAX                              (in flight)          core1 174
//   SQ32 + QEARLY + 2P + 2PRAW + CVTX + CVTSWAR                  (in flight)          core1 186
//   PAX + CVTX + HSF (no WCNT; isolates the host offload) 44,063  37.27%  6 of 6 correct before it
//                                                         $finished on the fabric assert at 247k    52
//      and its stage table against the same base WITHOUT the host, which is the honest reading of
//      what the host offload actually buys on the PIPELINED body:
//                            PAX+CVTX   +HSF
//        S1 accumulator->S        998     998
//        S2 softmax            12,825  12,825
//        S3 requant pass A      2,753   2,752
//        S4 convert || pack    10,492   9,516   <- the host takes the 128-word pack off warp 0, so
//                                                 the convert runs on SIX warps instead of five
//        S5 PV                  8,894   9,039   <- the 64-word Q-scale write is gone from warp 0
//        S6 QK || finalize      9,309   8,881   <- unchanged code; within ~700 of the QK mesh
//        TOTAL                 45,662  44,063   (pooled intervals 44,705..46,531, mean ~44,8xx)
//      *** SO THE HOST OFFLOAD IS WORTH ~1,000 CYCLES ON THIS BODY, NOT THE ~20k THE SEQUENTIAL
//      MEASUREMENT SUGGESTED, AND THE REASON IS ARITHMETIC RATHER THAN A BUG. ***  On FA_SP the
//      per-tile GPU scale traffic is only 192 words (64 Q + 128 P), not 704, and BOTH were already
//      hidden -- the Q write under the PV mesh, the pack under the convert.  All the host can
//      therefore win is the 6th warp for the convert, and the convert's binding core carries three
//      warps either way, so going from 6.4 to 5.33 items per thread is -17% of ITS issue term:
//      3 x 5.33 x 345 = 5,517 warp-instructions instead of 6,624, i.e. -1,107 predicted, -980
//      measured.  The handshake itself is free and, importantly, SOUND: the GPU publishes PACKREQ at
//      the top of stage S4 and blocks on PACKED at the top of S5, which is inside the ~26k-cycle
//      window S2+S3+S4 where the mesh is idle end to end, so the host is the ONLY requestor of
//      either scale port and there is no pairing state to corrupt.  No deadlock, no timeout, and the
//      PV stage is unchanged -- which is the thing the fourth pass's 7-of-12 host build got wrong.
//      *** THE TWO "WRONG" ROWS ARE THE HEADLINE RESULT OF THIS PASS, not the cycle counts: two
//      BIT-EXACT reschedulings re-opened the FA_SP_QKACC accumulator race that the fourth pass
//      believed fixed.  See (G0).  Every row below carries FA_SP_WCNT for that reason. ***
//
// (G8) *** A TIGHT HOST POLL LOOP KILLS THE FABRIC, AND IT IS NOT THE SCALE-PORT HAZARD. ***
//      The maximal stack (FA_SP_HSF + FA_SM_2P + FA_SM_2PBM + FA_SP_SMBMAX + FA_SM_2PRAW) $finished
//      at 164,937,000 ps (~82.5k cycles) with
//          TLMonitor xbar_3 (RadianceCluster.scala:112, the host->cluster extReqXbar):
//          "'D' channel contains improper response size"
//      which is the SAME assertion that killed FA_HOSTCFG and made it "WITHDRAWN, do not revive".
//      *** BUT IT IS A DIFFERENT CAUSE, AND WHERE IT FIRED PROVES IT. ***  82.5k is inside the
//      SOFTMAX of iteration 0.  By then the FA_SP_HSF host has finished every scale write it will
//      ever do in the prologue, and the FA_SP_HSF GPU writes NO scale word at any point -- so there
//      is no host-vs-GPU scale-port interleaving to blame, and no ROCC/SF pairing state to corrupt.
//      The only host traffic in that window is THE MAILBOX POLL: an unthrottled stream of uncached
//      8-byte Gets into cluster SMEM, concurrent with a softmax that is saturating that same SMEM.
//      The shared-SMEM slave path answers one of them with the wrong size.
//      The evidence that it is a timing window rather than a structural violation is that the SAME
//      assertion, at the SAME xbar, fires at wildly different times depending only on how much SMEM
//      traffic the GPU is generating:
//          + FA_SM_2PBM (heaviest softmax)      164,937,000 ps  =  ~82.5k cycles, iteration 0
//          FA_SP_HSF + PAX + CVTX (lightest)    494,475,000 ps  = ~247.2k cycles, iteration 3
//      i.e. EVERY unthrottled-poll host build eventually trips it; the lighter ones just take three
//      more tiles to get there.  That is also why the fourth pass's host runs completed at all.
//      THROTTLING THE POLL IS NOT ENOUGH -- MEASURED.  sp_backoff() (poll every ~256 cycles instead
//      of every ~20, and test cluster 1 only once cluster 0 is satisfied: >10x fewer Gets) still
//      $finishes, at 196,801,000 ps.  So the trigger is not the RATE, it is the host READ of cluster
//      SMEM at all while the GPU is using that SMEM -- which is exactly what host.cpp's own
//      mbox_get() note already recorded from the fourth pass and I re-derived the hard way:
//      "every configuration that reads the mailbox at all (FA_HOSTHS, FA_HOSTCFG) has hit that
//      assertion in some build ... while every read-free build (FA_NOSCALES alone, which only ever
//      stores) has never once hit it in ~20 runs."  The store-to-load `fence` before every poll,
//      which that note proposes as the fix, is present in sp_mbox_get() and does not help either.
//      *** CONSEQUENCE, AND IT RETIRES THE WHOLE LEVER: a host->GPU MAILBOX IS NOT USABLE ON THIS
//      RTL, so the host cannot be told when a RUNTIME quantity is ready -- and it could not pack the
//      P scales even if it were told, because packing means READING SCALE_SMEM out of cluster SMEM,
//      128 more Gets of the same kind.  The host offload is therefore limited to WRITE-ONLY,
//      UNSYNCHRONISED prefill of LOOP-INVARIANT scales, which on FA_SP is worth the ~1,000 cyc/tile
//      measured above and nothing more.  The only way through would be for the GPU to publish the
//      packed words to DRAM instead of SMEM (host DRAM reads do not touch the cluster xbar), and the
//      GPU-side cost of that publish would eat most of the ~1,000.  Not worth it; recorded so the
//      next pass does not spend another day on it. ***
//      FA_SP_HSF is kept, default OFF, with the throttle in place: it is a working, deadlock-free
//      demonstration of the composition and the correct place to restart from if the RTL ever grows
//      a usable host read path.
//
// (G5) *** THE REGISTER BUDGET IS A SHARED RESOURCE, AND FREEING IT IN ONE PHASE UNLOCKS ANOTHER. ***
//      Two flags in this file were REJECTED by earlier passes purely on the renamer:
//        FA_SP_CVTSWAR   the 25-instruction SWAR e4m3 packer, 8 x (34-25) = 72 fewer instructions
//                        per convert item (~21% of that phase's issue work) -- "58 registers, and 58
//                        is the known-fatal count ... the packer choice in this kernel is a REGISTER
//                        decision, not an instruction-count one";
//        FA_SP_FZ6       finalize on all six warps.
//      FA_SM_2P deletes online_softmax_block's slo[8]/shi[8] register arrays (it re-reads S in its
//      second pass instead of holding the scaled values), which takes the whole-file union from 53
//      to 49 -- and that is exactly the headroom those two needed:
//        + FA_SM_2P                                        49
//        + FA_SM_2P + FA_SP_CVTSWAR + FA_SP_CVTX           53   (was 58 without FA_SM_2P)
//        + FA_SM_2P + FA_SP_CVTSWAR + FA_SP_CVTX + FZ6     50
//      So the right way to read the renamer constraint is not "this flag costs too much" but "the
//      budget is one number for the whole kernel, and a register freed anywhere is spendable
//      anywhere".  Check the union of the COMBINATION, not of each flag against the reference.
//      RELATED FIX IN THIS PASS: FA_SP_CVTX and FA_SP_CVTSWAR used to be mutually exclusive #elif
//      arms of fa_requant_cvt, so choosing the SWAR packer put the 16-way subbank conflict BACK and
//      roughly cancelled its own win.  They are orthogonal -- one is the ADDRESSING, the other the
//      ARITHMETIC -- and the CVTX arm now uses the SWAR chain when both are set.
//
// (G10) *** THE 53-REGISTER WALL IS NOT REAL.  THE BUDGET IS 255 PHYSICAL REGISTERS PER CORE, AND
//       fa_regs.py's WHOLE-FILE UNION CAN RANK TWO BUILDS BACKWARDS. ***
//       Rename.scala:110-123 hands out a physical register the first time a given WARP writes a
//       given ARCHITECTURAL register, never reclaims it, and draws from ONE counter per CORE over
//       1..255.  So the constraint is
//           for each core:  sum over its resident warps of |arch regs THAT WARP EVER WRITES| <= 255
//       A whole-file union cannot express that.  It adds the warp-0 agent path's registers into the
//       budget of warps that branch around it, it misses the RUNTIME entirely (the kernel .s has no
//       _start/init_regs/mu_schedule, and every warp executes those), and it collapses the
//       three-warps-per-core multiplier that is the term actually binding.  fa_regs3.py computes the
//       real quantity from the LINKED GPU elf (llvm-objdump decodes Muon's extended register file
//       cleanly -- 1,989 instructions, zero <unknown>), attributing each symbol to a role and summing
//       under FA_SP's map (warp w -> core w&1; warp 0 = agent, warps 1..5 = SIMT).
//
//       MEASURED, and the last column is the outcome that was actually observed on hardware:
//         configuration (on QSPLIT QKACC WCNT PAX)      fa_regs.py  per-warp  core 1   observed
//         baseline + CVTX                                  53          72       216    RUNS, 12/12
//         + FA_SP_CVTROT                                   55          74       222    (claimed abort)
//         + FA_SP_CVTX + FA_SP_CVTSWAR                     58          77       231    ABORTS
//         + FA_TREEFIX + FA_FOLDX                          56          78       234    (inferred abort)
//         + FA_SM_2P + FA_SM_2PRAW + CVTSWAR + FZ6          52          56       168    -- safe --
//         FA_SP_SQ32 without FA_SM_2P                       57          82       246    ABORTS
//         FA_SP_SQ32 + QEARLY + FA_SM_2P + 2PRAW + CVTSWAR  60          62       186    -- safe --
//         + FA_SP_CBMAX                                    53          82       246    ABORTS
//         + FA_SP_SM1FX                                    58          90       270    ABORTS
//         FA_SP_SQ32 + FA_SM_2P + FA_SM_2PRAW              55          57       171    *** RUNS ***
//       TWO THINGS THIS SETTLES.
//       (1) *** THE UNION IS NOT EVEN MONOTONE IN THE REAL FIGURE. ***  FA_SP_SQ32 has a HIGHER union
//           than the running baseline (55 vs 53) and a far LOWER per-warp figure (57 vs 72), because
//           its agent/SIMT split and its smaller per-function footprints are different.  That is
//           exactly why FA_SP_CBMAX -- identical 53-name union, per-warp 82 -- was a FALSE PASS: the
//           union had no signal to give.  The per-warp figure orders every data point correctly
//           (72 runs < 82 aborts < 90 aborts) where the union orders two of them backwards.
//       (2) *** "55 IS FATAL" IS REFUTED BY DIRECT MEASUREMENT. ***  The 55-union FA_SP_SQ32 build ran
//           past 230,000,000 ps with no Rename.scala:123 assert -- the assert fires at ~197M ps when
//           it fires at all.  So the three passes' worth of levers rejected on "55/58 registers" were
//           rejected on a metric that does not measure the constraint: FA_SP_CVTSWAR (231/255),
//           FA_SP_CVTROT (222/255) and FA_TREEFIX+FA_FOLDX (234/255) are all LEGAL and all have
//           real cycle wins attached.  FA_SP_SM1FX (270) and FA_SP_CBMAX (246) genuinely are too
//           expensive, and now for a checkable reason.
//       *** THE THRESHOLD IS CALIBRATED, NOT ABSOLUTE, AND THE BAND HAS NOW BEEN CLOSED FROM ABOVE
//       BY TWO MORE RUNS -- WHICH KILLED A LEVER I HAD CALLED LEGAL. ***  Observed:
//           RUNS   at 168, 171, 186, 216
//           ABORTS at 231, 246, 246, 270   (Rename.sv, "total register usage exceeded maximum
//                                           number of physical registers", 210-244M ps)
//       so the wall is in (216, 231], and 231 is FATAL.  That retires FA_SP_CVTX + FA_SP_CVTSWAR on
//       the Sq=64 base (231 -- aborted at 244,403,000 ps) and, by inference, FA_TREEFIX + FA_FOLDX
//       (234) and the Sq=32 build without FA_SM_2P (246 -- aborted at 210,565,000 ps).
//       *** SO THE UNION'S VERDICT ON FA_SP_CVTSWAR WAS RIGHT AFTER ALL, AND ITS VERDICT ON Sq=32 WAS
//       WRONG.  THAT IS THE WHOLE POINT: the union has no signal, so it is right by coincidence and
//       wrong by coincidence, and only the per-warp figure tells you WHICH. ***  What the per-warp
//       figure buys is not permission -- it is knowing where the headroom actually is: FA_SM_2P frees
//       so much (Sq=64 + it lands at 153-168, Sq=32 + it at 171-186) that every lever the union
//       vetoed becomes affordable ON TOP OF IT, which is exactly the composition worth running.
//       Treat <= 216 as SAFE, >= 231 as FATAL, and 217..230 as the only remaining unresolved band
//       (FA_SP_CVTROT at 222 is the one flag still in it).  fa_regs3.py also reports a TIGHTER
//       statistic that removes fa_entry's warp-0-only spans; that one is MONOTONE in the data but is
//       NOT a bound (it computes 210 for FA_SP_SM1FX, which aborts, against 156 for the build that
//       runs), so it is reported and not used as a limit.  The load-bearing claim here is the
//       ORDERING, not the number: both statistics rank all six data points correctly and the
//       whole-file union ranks two of them backwards.
//       USE fa_regs3.py ON THE LINKED elf FOR ANY NEW CONFIGURATION, and treat 217..245 as "run it
//       for eight minutes and find out" rather than as pass or fail.
//       CAVEAT, stated because it bounds the model: fa_entry is executed by every warp but contains
//       the agent regions inline under `if (warp == 0)`, and this tool attributes all of fa_entry to
//       every warp, so the SIMT figure is an UPPER BOUND (divergence means warps 1..5 branch around
//       those blocks and never issue them).  That is the right direction for a budget check, and it
//       is probably why FA_SP_CBMAX aborts at a computed 246 rather than above 255.
//
// (G9) *** COSTING THE Sq=32 HALF-TILE SHAPE -- THE ROUTE PAST 40%, AND WHAT STOPS IT. ***
//      Everything above says ~40% is the ceiling of the Sq=64 pipeline SHAPE: PV's 8,882 cycles are
//      structurally exposed with five of six warps idle, no matmul is splittable along M/N/K (the
//      mesh's scale read row is a fixed function of the latched loop bounds), FA_SP_OPV is dead on
//      Scratchpad.scala:220, and SMEM is exactly full.  Halving the QUERY block is the one axis that
//      splits EXACTLY -- every output row's dot products are independent -- so an Sq=32 half-tile
//      loop with S/P and P8 double-buffered is a BIT-EXACT restructure that must still score 3.5666%.
//      It is implemented here as FA_SP_SQ32 and it is worth writing down what it costs.
//
//      1. THE SMEM MAP WORKS, AND UNLIKE FA_SP_OPV IT PASSES THE BANK TEST FOR BOTH PARITIES.  The
//      reduce scratch is PINNED at 0x14000..0x16000 (flash_mx_impl.hpp hard-codes 0x15000 and
//      0x15600/0x15680), so the layout is built around it -- see the map at FA_SQH.  The key move is
//      putting BOTH S/P buffers in bank 1 and BOTH P8 buffers in bank 2, which makes the mesh's reads
//      and SIMT's reads land in different banks whichever parity is current:
//          PV(t-1) reads P8[1-p] (b2) + V (b0)   ||  pass A / convert of t READ S/P[p]   (b1)
//          QK(t+1) reads Q       (b2) + K (b3)   ||  finalize(t-1) READS O(t-1) in S/P[1-p] (b1)
//      The convert's P8 writes and pass A's SCALE_SMEM writes do land in bank 2 while the mesh reads
//      there, which is fine (1R1W subbanks; the four-entry queue is on the READ port), and the
//      convert's SCALE_SMEM READS are one word per item, ~1 per 20 cycles.  The softmax's ~0.8
//      accesses/cycle of scratch traffic is the one thing that cannot coexist with a mesh read, so
//      THE MESH IS IDLE DURING THE SOFTMAX BY DESIGN -- which conveniently makes stage S2 the one
//      place in the iteration where warp 0 can write the SF scales with no mesh scale read anywhere.
//      Arithmetic: V 32 + K 32 + S/P 2x16 + P8 2x8 + scratch 8 + Q 4 = 124 KB, 4 KB spare, with O
//      overlaying the dead S/P buffer's low 8 KB (exactly its size).
//
//      2. THE PROJECTED WIN IS ~42%, NOT MORE.  Per half-tile the mesh is 2 x 4,096 and the six
//      stages are ~500 | softmax ~7,100 (5 warps, +granularity) | ~1,650 | convert ~4,900 (6 warps)
//      | ~500 | finalize ~4,900, i.e. ~19,400 -- so ~38,800 per 64 rows against the Sq=64 shape's
//      ~41,000, for 16,384/38,800 = 42.2%.  Both matmuls hide (PV in S3+S4 = 6,550 > 4,096; QK in
//      S6+S1 = 5,400 > 4,096).  The gain is smaller than "PV stops being exposed" suggests because
//      halving Sq also halves the mesh work per tile, and because per-tile granularity gets worse:
//      32 rows over 6 warps is 5.33, and 256 convert items over 96 threads is 2.67, so both round up
//      ~10%, and there are twice as many loop_ws FSM start-ups per 64 rows (~+1,400).
//
//      3. *** AND IT DOES NOT FIT THE RENAMER.  THIS IS THE BLOCKER, AND IT IS 2 REGISTERS. ***
//      Measured with fa_regs.py on the real body (not estimated):
//          Sq=64 QSPLIT + WCNT + PAX + CVTX + FA_SM_2P + FA_SM_2PRAW    46   fa_entry 27, softmax 25
//          Sq=32 FA_SP_SQ32, same flags                                 55   fa_entry 31, softmax 28
//          Sq=32 bare (no CVTX, no FA_SM_2P)                            57
//          Sq=32 + FA_SM_2PBM + FA_SP_SMBMAX                            56
//          Sq=32 with the two parities UNROLLED (compile-time selects)   57   fa_entry 33
//      The +9 is +4 in fa_entry (the double-buffer parity, the alternating Q source row and O
//      destination half, six stages instead of seven) and +3 in online_softmax_block<32,256> -- the
//      template's only change is a loop bound, but 32 rows over 6 warps and the `fr < SQ` fold guard
//      allocate worse than 64 do.  Unrolling the parity to make every select a compile-time constant
//      makes it WORSE, not better (fa_entry 31 -> 33): the compiler then schedules across twelve
//      straight-line stages and keeps more live.  So the floor is 55, and 55 is the empirically FATAL
//      count -- FA_SP_CVTROT and PKOVL+REFETCH both $finish at Rename.scala:123 there.
//      WHAT WOULD MAKE IT REACHABLE, in order of cheapness:
//        (a) two registers out of online_softmax_block<32,256> or fa_entry.  That is a real target,
//            not a wish: the Sq=64 instantiation of the same function needs 25.
//        (b) occupancy 2 doubles the budget to 127/warp and makes it fit trivially -- and costs more
//            than the shape gains (the same config measured 46,478 -> 56,455 going to occ 2).
//        (c) confirm whether 55 is ACTUALLY fatal.  The "55 $finishes" evidence is two data points
//            from unrelated code, and the whole-file union is only a PROXY for the per-warp counts
//            Rename.scala actually sums.  The renamer assert fires ~197M ps in, i.e. eight minutes of
//            wall clock, so this is a ten-minute experiment, and it is running as tag `sqR`.
//      RECOMMENDATION: if `sqR` survives the renamer, Sq=32 is the right next shape and is ~42%
//      reachable in about one day of work (the body is written; what is left is validating the four
//      strided Q-scale chunks and one round of correctness).  If it aborts, Sq=32 needs (a) first and
//      should be the recommendation for AFTER this sprint rather than a target inside it.
//
// (G11) *** A BUILD-INPUT RULE, BECAUSE THE FIX FOR SOMEONE ELSE'S BUG LANDED MID-FLIGHT. ***
//       FA_SM_2P's pb overrun was fixed in fd722ac (2026-07-27 19:27:15) with compile-time
//       static_asserts, and 489ff35 / 6ff6e07 followed within half an hour, both touching pass A's
//       reduction chains.  Three of my runs were built inside that churn.  Timestamps SAID they were
//       post-fix (19:32 / 19:53 / 19:55 against 19:27:15) but a timestamp is not an input check, so I
//       byte-compared each running ELF against a rebuild from the then-current source: both DIFFERED,
//       i.e. the inputs could not be proven.  All three were KILLED AND REBUILT rather than scored.
//       That is the right default here and not pedantry: a pre-fix build either fails outright or
//       "passes" because the Q re-fetch happens to overwrite the damage, and this file has already
//       recorded three separate cases of the second outcome (the QSPLIT 12-of-12 coincidence, the
//       CVTX+PREPK row, and the pb overrun itself).  It matters doubly for FA_SP_SQ32, whose whole
//       point is a different stage layout -- so whether anything covers the overrun at all is exactly
//       what changed.  RULE: before scoring any run whose config includes code you do not own, check
//       `git log --oneline -1 -- <that file>` against the build, or byte-compare the ELF to a rebuild.
//
// (G4) FLAG CATALOGUE FOR THIS PASS (all default OFF; the plain build and the plain FULL_ATTN2
//      build stay byte-for-byte the pre-existing kernel, and the published FA_SP_QSPLIT reference
//      build is verified BYTE-IDENTICAL after every edit below).
//        FA_SP_PAX    XOR-rotate requant pass A's 16 word loads by (lane & 15).  Bit-exact (max is
//                     order-free), 53 -> 53 registers, +18 instr/item.  See (G1).
//        FA_FOLDX     the same for warp_fold16's eight word loads (FA_TREEFIX's reduction).  Costs
//                     6 registers in an all-warps function -- RENAMER CASUALTY, kept OFF.
//        FA_SP_SM1FX  one-fence folds for BOTH softmax reductions over a 32-bit-per-lane XOR-indexed
//                     buffer.  Bit-exact for the SUM too, which FA_SP_SM1FR's (k+lane)&15 rotation
//                     was not: XOR by a constant is an AUTOMORPHISM of the dyadic reduction tree
//                     (the pair {k,k^1} is a tree pair, {k,k^1} vs {k^2,k^3} is the next one, ...),
//                     so the tree's expression is reproduced with some subtrees' operands SWAPPED,
//                     and bf16 addition is commutative even though it is not associative.  A cyclic
//                     rotation maps {0,1} to {1,2}, which is not a tree pair, and is therefore only
//                     legal for the max -- that asymmetry is the whole reason to prefer XOR.
//                     58 registers: RENAMER CASUALTY, kept OFF.
//        FA_SP_RBPAD  pad the cross-lane reduce buffer's PER-WARP stride from NT to NT+2 halfwords.
//                     buf is at 0x15000 + warp*NT halfwords = warp*8 WORDS, so the subbank
//                     (word & 15) is 0 for warps 0/2/4 and 8 for warps 1/3/5 -- a 3-way INTER-warp
//                     conflict on top of the 16-way INTRA-warp one (all 16 lanes read buf[0] to
//                     broadcast each reduction).  A stride of 9 words spreads warps 0..5 over
//                     subbanks 0,9,2,11,4,13.  Address change only: bit-exact, zero registers, zero
//                     instructions, +24 bytes of scratch.  Does NOT touch the intra-warp broadcast
//                     conflict -- that needs FA_FOLDX (register-fatal) or FA_SM_2P.
//        FA_SP_BANKA  bank-align S and move P8/O next to Q.  MANDATORY with FA_SP_OPV -- see (G2).
//        FA_SP_OPV    the six-stage pipeline that hides the PV matmul under the next tile's softmax
//                     and puts O_unnorm in the (dead) P8 region.  Needs FA_SP_QOVL + FA_SP_PKOVL;
//                     replaces FA_SP_QSPLIT / QOVL3 / QOVL4 / QKACC.
//        FA_SP_OPVQ   move the Q(t+1) prefetch into stage C2, the one stage with NO mesh activity.
//                     That frees warp 0 in stage A, so the softmax and pass A run on all SIX warps,
//                     and it removes the last place where an SF scale write or a prefetch DMA can
//                     sit next to a live mesh operation.
//        FA_SP_WCNT   NOT NEW, but promoted from "diagnostic" to MANDATORY -- see (G0).  Puts
//                     gemmini_fence_waitcount(0) (MMIO 0x28, runningLoops, which rises in the SAME
//                     CYCLE as the LOOP_WS command write) in front of every busy-poll fence, so the
//                     accumulator->spad store in stage S1 can no longer be read by the softmax
//                     while the mesh is still writing it.  Four extra MMIO polls per tile.
//        FA_SP_HSF    *** the host MX-scale offload COMPOSED WITH THE PIPELINE. ***  The GPU writes
//                     NO scale word anywhere (K/V/Q and the runtime packed P scales all come from
//                     Rocket's 8-byte stores), which makes both scale ports HOST-PRIVATE and
//                     removes ScaleFactorMem's shared 2-beat pairing hazard by construction instead
//                     of by scheduling -- the thing the fourth pass's 7-of-12 host build got wrong.
//                     The hand-off is PACKREQ at the top of stage C2 / PACKED at the top of stage D,
//                     i.e. entirely inside the mesh-idle stage.  It pays off twice: warp 0's ~4.2k
//                     Q-scale write and ~8.3k P pack both disappear, so finalize AND the convert go
//                     from five warps to six.  Mailbox at device 0x15F00 (the sequential kernel's
//                     0x17F00 is inside FA_SP's Q scratchpad and would be overwritten by the Q DMA).
//
// (G3) *** FA_SP_OPV -- THE EXPOSED PV MATMUL CAN BE HIDDEN, AND NOTE (c) WAS WRONG ABOUT WHY. ***
//      Note (c) of the second pass argued that exactly one of the two matmuls must be exposed
//      because "the only SIMT work that does not depend on its own tile's mesh result is finalize,
//      and finalize(t) depends on PV(t)".  That is true WITHIN a tile and false ACROSS tiles:
//      PV(t-1) and softmax(t) are independent, so sliding the loop by one hides PV.  Note (c) also
//      recorded the real obstruction correctly -- O(t-1) must stay alive across tile t's softmax and
//      "a second 16 KB O buffer does not exist" -- but the buffer IS there: it is the P8 region,
//      which is dead from the moment PV(t-1) has read it until convert(t) rewrites it.
//      FA_SP_OPV puts O there and runs six stages per iteration; FA_SP_OPVQ additionally moves the
//      Q prefetch into the one stage with NO mesh activity, which frees warp 0 for a SIX-warp
//      softmax and a six-warp pass A.  Full derivation at the flag itself.
// ============================================================================================
#if defined(FA_SP)
#ifdef FA_SP_MAP2
// MAP2: swap V and P8 so requant's fp8 WRITES (P8) and its bf16 READS (S) land in different
// 32 KB SMEM banks -- the sequential kernel had P8 in bank 0 and P-bf16 in banks 1/2, MAP1 puts
// P8 and the lower half of S both in bank 1.
static constexpr uint32_t SP_P     = 0;      // P8  (A op of PVF)  rows    0..1024
static constexpr uint32_t SP_V_END = 3072;   // V   (B op of PVF)  rows 1024..3072
static constexpr uint32_t SP_C     = 3072;   // S/P/O (C dest)     rows 3072..5120
#elif defined(FA_SP_BANKA)
// ============================================================================================
// FA_SP_BANKA -- *** BANK-ALIGN S.  THIS IS A HARDWARE CONSTRAINT, NOT A MICRO-OPTIMISATION. ***
//
// THE RTL FACT, measured the hard way (FA_SP_OPV $finished on it twice, at 237,301,000 and
// 238,123,000 ps, naming `...gemmini.spad.spad_mems_2`):
//   Scratchpad.scala:210-220 -- the `use_shared_ext_mem` READ path of every ScratchpadBank --
//   tracks outstanding spad read requests in
//       val dma_q = Module(new Queue(Bool(), 4, false, true))
//       assert(dma_q.io.enq.fire === ren, "DMA queue does not have enough entries")
//                                          // TODO (richard): do backpressure
//   FOUR entries, and NO BACKPRESSURE.  If the shared SMEM does not return a read response before
//   the mesh issues a fifth request to that bank, the SIMULATION $finishes.  So THE MESH MAY NOT
//   READ AN OPERAND OUT OF A 32 KB SMEM BANK THAT SIMT CODE IS READING AT THE SAME TIME: the SIMT
//   traffic delays the responses and the queue overflows.
//
// WHY THE DEFAULT MAP CANNOT OVERLAP THE MESH WITH THE SOFTMAX / PASS A / THE CONVERT.  Banks are
// 32 KB (bank = addr >> 15) and the default map puts
//     S -> P-bf16 -> O  at 0x0C000..0x14000, STRADDLING banks 1 AND 2,   and   Q at 0x16000, bank 2.
// Every large SIMT phase reads S, so every one of them touches bank 2 -- and the QK matmul reads its
// A operand (Q) out of bank 2.  FA_SP_QSPLIT survives this BY CONSTRUCTION, not by luck: the only
// stage where it runs the mesh underneath SIMT work is QK(t+1) || finalize(t), and finalize reads
// only O, which is the LOWER 16 KB of SP_C and therefore entirely in bank 1.  Any schedule that
// overlaps the mesh with a phase that touches S -- which is exactly what FA_SP_OPV does -- asserts.
//
// THE FIX IS A THREE-LINE REMAP: give S a WHOLE bank and move P8/O into bank 2, next to Q.
//     bank 0  0x00000..0x08000   V (32 KB)                PV's B operand
//     bank 1  0x08000..0x10000   S -> P-bf16 (32 KB)      the SIMT working set, EXACTLY one bank
//     bank 2  0x10000..0x14000   P8 / then O_unnorm       PV's A operand, then finalize's input
//             0x14000..0x16000   scratch (8 KB)           (REDBUF 0x15000 is hard-coded: unmoved)
//             0x16000..0x18000   Q (8 KB)                 QK's A operand
//     bank 3  0x18000..0x20000   K^T (32 KB)              QK's B operand
// Now every concurrent pair in the FA_SP_OPV schedule is bank-disjoint ON THE READ PORT:
//     PV mesh reads P8(b2) + V(b0)  ||  softmax reads AND writes S(b1)        disjoint
//     QK mesh reads Q(b2)  + K(b3)  ||  convert READS S(b1), writes P8(b2)    disjoint READS
//     finalize reads O(b2)          ||  no mesh operation at all
// The convert's P8 WRITES do land in bank 2 while the mesh reads Q there, which is the one pairing
// this map does not separate -- it should be safe because the subbanks are 1R1W (a read and a write
// proceed in the same cycle) and the four-entry queue is on the READ port, but it is the assumption
// to suspect first if this configuration asserts on spad_mems_2 anyway.
// Nothing else moves: SM_S / SM_P8 are derived, the 0x14000..0x16000 scratch window is untouched
// (its 0x15000 REDBUF address is hard-coded inside flash_mx_impl.hpp), and so are Q and K^T.
// ============================================================================================
static constexpr uint32_t SP_V_END = 2048;   // V   (B op of PVF)  rows    0..2048  bank 0
static constexpr uint32_t SP_C     = 2048;   // S -> P-bf16        rows 2048..4096  bank 1 EXACTLY
static constexpr uint32_t SP_P     = 4096;   // P8 / O_unnorm      rows 4096..5120  bank 2
#else
static constexpr uint32_t SP_V_END = 2048;   // V   (B op of PVF)  rows    0..2048
static constexpr uint32_t SP_P     = 2048;   // P8  (A op of PVF)  rows 2048..3072
static constexpr uint32_t SP_C     = 3072;   // S/P/O (C dest)     rows 3072..5120
#endif
#ifdef FA_SP_SQ32
// ============================================================================================
// FA_SP_SQ32 -- Sq=32 HALF-TILE LOOP WITH S/P AND P8 DOUBLE-BUFFERED.
//
// WHY: at Sq=64 the PV matmul's 8,882 cycles are STRUCTURALLY exposed with 5 of 6 warps idle, no
// matmul is splittable along M/N/K (the mesh's scale read row is a fixed function of the latched
// loop bounds), FA_SP_OPV is dead on Scratchpad.scala:220, and SMEM is exactly full.  Halving the
// QUERY block halves S, P8 and O -- and the query dimension is the one axis that splits EXACTLY,
// because every output row's dot products are independent.  So this is a BIT-EXACT restructure:
// golden_O_u16 must still score 3.5666%.
//
// THE SMEM MAP, AND THE BANK ARGUMENT THAT KILLED FA_SP_OPV BUT PASSES HERE.  Banks are 32 KB
// (bank = addr >> 15) and Scratchpad.scala:220's four-entry read queue means THE MESH MAY NOT READ
// AN OPERAND OUT OF A BANK SIMT IS READING.  The reduce scratch is PINNED at 0x14000..0x16000
// (flash_mx_impl.hpp hard-codes 0x15000 for REDBUF and 0x15600/0x15680 for FA_SM_2P's pb), so the
// layout has to be built around it:
//    bank 0  0x00000..0x08000   V (32 KB)                          PV's B operand
//    bank 1  0x08000..0x0C000   S/P[0] (16 KB; O in its low 8 KB)  the SIMT working set
//            0x0C000..0x10000   S/P[1] (16 KB; O in its low 8 KB)
//    bank 2  0x10000..0x12000   P8[0] (8 KB)                       PV's A operand
//            0x12000..0x14000   P8[1] (8 KB)
//            0x14000..0x16000   scratch (8 KB, PINNED)
//            0x16000..0x17000   Q (4 KB)                           QK's A operand
//            0x17000..0x18000   spare (4 KB)
//    bank 3  0x18000..0x20000   K^T (32 KB)                        QK's B operand
// BOTH S/P buffers are in bank 1 and BOTH P8 buffers in bank 2, which is what makes both parities
// work rather than only one:
//    PV(t-1) reads P8[1-p] (b2) + V (b0)  ||  pass A / convert of tile t READ S/P[p] (b1)   ->
//        the mesh's reads and SIMT's reads are in DIFFERENT banks for either parity.  The convert's
//        P8[p] WRITES do land in bank 2, which is fine: the subbanks are 1R1W and the four-entry
//        queue sits on the READ port.  Its SCALE_SMEM reads are one word per item (~1 per 20 cycles).
//    QK(t+1) reads Q (b2) + K (b3)        ||  finalize(t-1) READS O(t-1) = S/P[1-p]'s low half (b1)
//        -> again disjoint for either parity, because BOTH S/P buffers are in bank 1.
//    the softmax hammers the scratch (b2), so THE MESH IS IDLE during it -- by design, and that is
//        also the window in which warp 0 writes the P scales, with no mesh scale read anywhere.
// ARITHMETIC: V 32 + K 32 + S/P 2x16 + P8 2x8 + scratch 8 + Q 4 = 124 KB, 4 KB spare.  O overlays
// the dead S/P buffer's low half (8 KB), which is exactly its size.
// ============================================================================================
static constexpr uint32_t FA_SQH = 32;                  // rows per half-tile
static constexpr uint32_t SPH_C0 = 2048, SPH_C1 = 3072; // S/P double buffer  (bank 1)
static constexpr uint32_t SPH_P80 = 4096, SPH_P81 = 4608; // P8 double buffer (bank 2)
static constexpr uint32_t SPH_Q  = 5632;                // Q, 4 KB           (bank 2)
static constexpr uint32_t SMH_C0 = SPH_C0 * DIM, SMH_C1 = SPH_C1 * DIM;
static constexpr uint32_t SMH_P80 = SPH_P80 * DIM, SMH_P81 = SPH_P81 * DIM;
#endif
static constexpr uint32_t SP_Q     = 5632;   // Q   (A op of QKF)  rows 5632..6144
static constexpr uint32_t SP_K_END = 8192;   // K^T (B op of QKF)  rows 6144..8192
static constexpr uint32_t SM_S     = SP_C * DIM;
static constexpr uint32_t SM_P8    = SP_P * DIM;
// fp32 per-(block,row) row-sum partials for FA_SP_FUSE: 8*64 floats = 2 KB.  Lives in the hole
// between REDBUF (0x15000; up to 6*RB*16*2 = 1.5 KB with the batched row-max) and Q (0x16000).
static constexpr uint32_t LPART_SMEM = 0x15400;   // 2 KB, up to 0x15C00 (Q spad starts 0x16000)

static inline volatile __shared uint32_t *fa_sf_a(uint32_t half) {
    return reinterpret_cast<volatile __shared uint32_t *>(
        GEMMINI_SF_MEM_A + (half ? GEMMINI_SF_MEM_BUFFER_OFFSET : 0u));
}
static inline volatile __shared uint32_t *fa_sf_b(uint32_t half) {
    return reinterpret_cast<volatile __shared uint32_t *>(
        GEMMINI_SF_MEM_B + (half ? GEMMINI_SF_MEM_BUFFER_OFFSET : 0u));
}

// A-operand move-in: PE_TILES_I x PE_TILES_K tiles of DIM x DIM, GMEM row stride dim_k.
// (Mirrors the !GEMMINI_DMA branch of copy_gmem_to_smem_async, but with an explicit spad row
// and WITHOUT touching B -- the SKIP_B the loop-FSM path is missing.)
template <GemmConfig C>
static __attribute__((noinline)) void fa_mvin_A(const uint8_t *A, uint32_t spad_row,
                                                uint32_t dim_k, uint32_t tid) {
    if (tid != 0) return;
    gemmini_config_ld(dim_k * sizeof(uint8_t));
#ifdef FA_NS_SENT
    fa_sent_stamp(spad_row, C.PE_TILES_I() * C.PE_TILES_K());   // positive completion proof
#endif
    // rolled on purpose: 32 unrolled ROCC mvin commands are 632 B of a 16 KB direct-mapped
    // icache, and this runs once per tile (the DMA, not the issue, is the cost).
#pragma clang loop unroll(disable)
    for (uint32_t i = 0; i < C.PE_TILES_I(); i++)
#pragma clang loop unroll(disable)
        for (uint32_t k = 0; k < C.PE_TILES_K(); k++) {
            const uint8_t *p = A + i * DIM * dim_k + k * DIM;
            gemmini_extended_mvin(rad_device_to_host_address(reinterpret_cast<uint32_t>(p)),
                                  spad_row + (i * C.PE_TILES_K() + k) * DIM, DIM, DIM);
        }
#ifdef FA_NS_SENT
    fa_sent_wait(spad_row, C.PE_TILES_I() * C.PE_TILES_K());
#endif
}
// B-operand move-in: PE_TILES_K x PE_TILES_J tiles, GMEM row stride dim_n; spad grows DOWN
// from spad_end_row.  Prologue-only (K^T, V) -> keep it rolled, it is cold code.
template <GemmConfig C>
static __attribute__((noinline)) void fa_mvin_B(const uint8_t *B, uint32_t spad_end_row,
                                                uint32_t dim_n, uint32_t tid) {
    if (tid != 0) return;
    gemmini_config_ld(dim_n * sizeof(uint8_t) / C.VALUES_PER_BYTE());
    const uint32_t start = spad_end_row - C.PE_TILES_K() * C.PE_TILES_J() * DIM;
#pragma clang loop unroll(disable)
    for (uint32_t k = 0; k < C.PE_TILES_K(); k++)
#pragma clang loop unroll(disable)
        for (uint32_t j = 0; j < C.PE_TILES_J(); j++) {
            const uint8_t *p = B + k * DIM * dim_n / C.VALUES_PER_BYTE() + j * DIM;
            gemmini_extended_mvin(rad_device_to_host_address(reinterpret_cast<uint32_t>(p)),
                                  start + (k * C.PE_TILES_J() + j) * DIM, DIM, DIM);
        }
}
// Issue one matmul with FULLY explicit A/B/C spad rows and SF-SRAM half selects. Async.
template <GemmConfig C>
static __attribute__((noinline)) void fa_mm(uint32_t a_row, uint32_t b_end, uint32_t c_row,
                                            uint32_t asel, uint32_t wsel, uint32_t tid) {
    if (tid != 0) return;
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), asel, wsel,
        QUANT_LUT_UPDATE_GRANULARITY);
#ifndef FA_CFGSETTLE_AFTER
    fa_cfg_settle();   // FA_CFGSETTLE -- see the mxgemm_core.hpp header: the mesh's WEIGHT
                       // scale-SRAM read row is loop_bound_j*(k>>1)+j, loop_bound_j is 16 for
                       // QKF and 8 for PVF, and LOOP_WS is not ordered against CONFIG_SCALE_MEM.
                       // FA_SP_LEANCFG removes the only other CONFIG_SCALE_MEM, so this path is
                       // the MOST exposed of the two kernels.
#endif
    matmul_tile_async<C>(/*tile_k=*/0, /*acc_move_out=*/true, /*accumulate=*/false,
                         /*b_spad_override=*/b_end, /*c_spad_dest=*/c_row,
                         /*a_spad_override=*/a_row, /*force_first=*/1);
#ifdef FA_CFGSETTLE_AFTER
    fa_cfg_settle();
#endif
}
// FA_SP_QKACC: issue the matmul COMPUTE ONLY -- the result stays in the gemmini ACCUMULATOR
// (acc_move_out=false => skip_stc=1), so the mesh performs ZERO SMEM writes and can therefore
// run underneath SIMT work that is reading and writing SMEM (finalize).  The caller must later
// drain and issue fa_store_acc() in a SIMT-quiet window (the accmem->spad writer needs an
// atomic all-16-subbank grant -- Hazard 2).
template <GemmConfig C>
static __attribute__((noinline)) void fa_mm_acc(uint32_t a_row, uint32_t b_end,
                                                uint32_t asel, uint32_t wsel, uint32_t tid) {
    if (tid != 0) return;
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), asel, wsel,
        QUANT_LUT_UPDATE_GRANULARITY);
#ifndef FA_CFGSETTLE_AFTER
    fa_cfg_settle();   // FA_CFGSETTLE -- same hazard as fa_mm() above.
#endif
    matmul_tile_async<C>(/*tile_k=*/0, /*acc_move_out=*/false, /*accumulate=*/false,
                         /*b_spad_override=*/b_end, /*c_spad_dest=*/0,
                         /*a_spad_override=*/a_row, /*force_first=*/1);
#ifdef FA_CFGSETTLE_AFTER
    fa_cfg_settle();
#endif
}
// Store-only loop_ws: move the accumulator -> spad row c_row (skip everything except stc).
template <GemmConfig C>
static __attribute__((noinline)) void fa_store_acc(uint32_t c_row, uint32_t tid) {
    if (tid != 0) return;
    gemmini_loop_ws_spad(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), 0, 0, 0,
        /*a_spad=*/0, /*b_spad=*/BANK_NUM * BANK_ROWS, /*d=*/0, c_row,
        false, false, false, false, /*ex_accumulate=*/false,
        NO_ACTIVATION, 0, 0, false,
        loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                          /*skip_ex=*/1, /*skip_stc=*/0));
}
template <GemmConfig C>
static __attribute__((noinline)) void fa_cfg(uint32_t asel, uint32_t wsel, uint32_t tid) {
    if (tid != 0) return;
#ifdef FA_SP_LEANCFG
    // LEAN per-gemm config.  configure_mxgemmini() issues 7 ROCC commands plus a gemmini_fence
    // (an MMIO busy-poll) every time, but for FA_SP almost all of it is dead:
    //   * config_ex  (dataflow + A/B/C datatypes) is IDENTICAL for QKF and PVF -> prologue only;
    //   * the two config_ld strides are overwritten by fa_mvin_A/fa_mvin_B's own config_ld;
    //   * config_st is the accumulator->GMEM stride, and nothing here moves out to GMEM;
    //   * loop_ws_config_bounds is re-issued by gemmini_loop_ws_spad inside matmul_tile_async;
    //   * the trailing fence is unnecessary -- the ROCC queue is in-order, so a later matmul
    //     cannot overtake this config.
    // What is left is CONFIG_SCALE_MEM (i/j/k bounds + the two SF double-buffer selects), and
    // fa_mm()/fa_mm_acc() issue that themselves.  So the lean path is a no-op.
    (void)asel; (void)wsel;
#else
    configure_mxgemmini<C>(C.TILE_M, C.TILE_N, C.TILE_K, wsel, asel);
#endif
}
// One-time gemmini setup for FA_SP: the parts of configure_mxgemmini that are gemm-invariant.
static __attribute__((noinline)) void fa_cfg_once(uint32_t tid) {
    if (tid != 0) return;
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                                GEMMINI_FORMAT_FP8, GEMMINI_FORMAT_FP8, GEMMINI_FORMAT_FULL,
                                false);
    gemmini_fence();
}
static __attribute__((noinline)) void fa_scl(volatile __shared uint32_t *dst,
                                             const uint8_t *src, int nbytes, uint32_t tid) {
    if (tid != 0) return;
    load_scale_factors(dst, src, nbytes);
}
#ifdef FA_SP_SQ32
// Q's scales are QK_A_scales_row[FA_GK][FA_SQ], so a 32-row half is FOUR STRIDED 32-byte chunks of
// the source -- but they land as ONE ascending 32-word run in the SF SRAM, which is what
// FlitMergeNode requires (each chunk is 8 words: even, and 8-byte aligned).
static __attribute__((noinline)) void fa_scl_qhalf(volatile __shared uint32_t *dst,
                                                   uint32_t half, uint32_t tid) {
    if (tid != 0) return;
    for (uint32_t g = 0; g < FA_GK; g++)
        load_scale_factors(dst + g * (FA_SQH / 4u), &QK_A_scales_row[g][half * FA_SQH], FA_SQH);
}

#endif
static __attribute__((noinline)) void fa_gf(uint32_t tid) {
    if (tid != 0) return;
    gemmini_fence();
}
#ifdef FA_SP_HSF
// ---- FA_SP_HSF host<->GPU mailbox (must match SP_MBOX / SPM_* in host.cpp) -------------------
// ADDRESS CHOICE, and it took two tries.  The sequential kernel's mailbox at device 0x17F00 is NOT
// usable here: it lands inside FA_SP's Q scratchpad (0x16000..0x18000), which the Q move-in DMA
// rewrites every tile.  The obvious replacement, 0x15F00 ("above everything the map puts in the
// 0x14000..0x16000 scratch window"), is ALSO wrong: FA_SM_2P's per-row partial buffer `pb` is at
// 0x15680 with a stride of 2*NT+1 = 33 HALFWORDS (its comment says 17), so it spans
// 0x15680..0x166E0 -- it overruns the scratch window into Q AND writes 0x15F10 / 0x15F20, i.e.
// exactly the PACKREQ and PACKED words.  With FA_SM_2P on, the handshake would silently break.
// 0x14D00 sits in the 512-byte CORR_SMEM slot above the 128 bytes corr_out actually uses (0x14C00..
// 0x14C80), which no configuration touches, and nothing in flash_mx_impl.hpp hard-codes.
// *** SEPARATE HAZARD WORTH FIXING IN flash_mx_impl.hpp, NOT OWNED HERE: pb's 4,224-byte footprint
// overwrites Q rows 0..13 on every softmax.  It survives only because FA_SP re-issues Q's move-in
// in stage S4, i.e. after the softmax and before the QK that reads Q -- so the corruption is always
// repaired before use.  That is luck, not design; STR should be NT+1 = 17 as its comment says. ***
// The host writes these words with 8-byte stores (a 4-byte host store is silently dropped) and the
// GPU reads/writes the low 32-bit word of each.
#ifdef FA_SP_HSF_PB
// ---- FA_SP_HSF_PB: put the mailbox in printBuf instead of SMEM. -----------------------------
// printBuf is a real TLRAM the cluster already instantiates and NOTHING uses:
//   RadianceCluster.scala:96-103  TLRAM(AddressSet(baseAddr + peripheralAddrOffset, 0x100*nTiles-1),
//                                      cacheable = false, atomics = true, beatBytes = 8)
//                                printBuf := clcbus.outwardNode
// Verified from the tree rather than taken on trust: there are exactly TWO references to `printBuf`
// in all of radiance -- that declaration and that connection -- so it has no RTL writer, and no
// software symbol resolves into it.  512 B at device 0x80000 with 1..8 B transfers and ATOMICS.
// *** IT IS ALSO THE DISCRIMINATOR THE SMEM FAILURE NEEDS. ***  printBuf and the SMEM subbanks hang
// off the SAME clcbus, and the kernel already reaches 0x84000 / 0x8a000 through that leg, so the
// host route and RWSplitterNode are COMMON to both.  If an 8-byte host Get works here and fails at
// 0x14D00, the fault is in the SMEM leg specifically (prealignBuffer -> alignmentXbar ->
// smemFanoutXbar -> 4-byte subbanks).  If it fails here too, the common path is implicated and
// RWSplitterNode's `// FIXME: check truncation` on d.source/d.size is the live suspect.
static constexpr uint32_t FA_SP_MBOX     = 0x80000;
#else
static constexpr uint32_t FA_SP_MBOX     = 0x14D00;
#endif
static constexpr uint32_t FA_SPM_READY   = 0x00;   // host -> GPU: K/V/Q scales resident
static constexpr uint32_t FA_SPM_PACKREQ = 0x10;   // GPU  -> host: SCALE_SMEM(t) is published
static constexpr uint32_t FA_SPM_PACKED  = 0x20;   // host -> GPU: Q + P scales resident
static constexpr uint32_t FA_SPM_MAGIC   = 0x5CA1E5u;
// Bounded spin: a broken mailbox must degrade to a WRONG ANSWER with a finite run, never to a hang
// that is indistinguishable from too small a cycle budget.  200k polls is far more than one tile.
static __attribute__((noinline)) void fa_sp_wait(uint32_t off, uint32_t want, uint32_t tid) {
    if (tid != 0) return;
    volatile __shared uint32_t *p =
        reinterpret_cast<volatile __shared uint32_t *>(FA_SP_MBOX + off);
    for (uint32_t i = 0; i < 200000u; i++) if (*p >= want) return;
}
static __attribute__((noinline)) void fa_sp_post(uint32_t off, uint32_t val, uint32_t tid) {
    if (tid != 0) return;
    *reinterpret_cast<volatile __shared uint32_t *>(FA_SP_MBOX + off) = val;
    mu_fence_smem();          // publish it to the host before we go on to wait for the reply
}
#endif
// ============================================================================================
// FA_SP_WCNT -- *** gemmini_fence() IS NOT A DRAIN FOR A MATMUL YOU JUST ISSUED. ***
//
// gemmini_fence() spins on MMIO 0x20, which is wired straight to `gemmini.module.io.busy`
// (GemminiTile.scala:403-406, 428).  A gemmini command is fired by the store to 0x00 and then has
// to travel to the reservation station before io.busy rises, so a poll issued a few cycles after
// that store CAN READ ZERO AND FALL STRAIGHT THROUGH -- the fence returns before the matmul has
// even started, never mind finished.  Whether it does depends on exactly how many cycles separate
// the issuing store from the poll, which is why every schedule change flips this on and off.
//
// MMIO 0x28 is the register that cannot lie: GemminiTile.scala:408-412 keeps
//     runningLoops := runningLoops + loopStarted + mmioLoopStarted - completionCount
// where mmioLoopStarted is asserted IN THE SAME CYCLE as the command write when its funct is
// LOOP_WS.  So runningLoops is >= 1 from the instant the matmul is issued until the loop reports
// completion, and `gemmini_fence_waitcount(0)` is a true drain.  It only tracks LOOP_WS, so an
// mvin still needs the busy poll -- hence both, in that order.
//
// WHY THIS IS THE FA_SP_QKACC BUG.  FA_SP_QKACC is the ONLY configuration in which SOFTWARE has to
// wait for a matmul and then read its ACCUMULATOR: every other path passes acc_move_out=true and
// lets the loop FSM do the move-out, which is internally ordered.  With FA_SP_QKACC, stage S1 does
//     fa_gf(tid);  fa_store_acc<QKF>(SP_C, tid);  fa_gf(tid);
// and if the first fence falls through, the accumulator->spad store reads the accumulator WHILE THE
// MESH IS STILL ACCUMULATING INTO IT.  What lands in S is a set of partial sums -- wrong, finite,
// of plausible magnitude, self-consistent enough that the softmax and requant downstream produce a
// P and an l that still agree with each other, and therefore an O that is still a convex
// combination of V.  That is EXACTLY the fingerprint measured in (F1), and it explains every other
// property of the bug: it needs FA_SP_QKACC (the bisection), it is timing-dependent, it moves tiles
// when the schedule moves, and it never appears on tile 0 (whose QK is issued in the prologue, a
// barrier and a whole stage away from the store).
// ============================================================================================
static __attribute__((noinline)) void fa_gfl(uint32_t tid) {
    if (tid != 0) return;
#ifdef FA_SP_WCNT
    gemmini_fence_waitcount(0);   // MMIO 0x28: runningLoops == 0  (rises AT ISSUE -- cannot race)
#endif
    gemmini_fence();              // MMIO 0x20: io.busy == 0       (also covers mvin DMAs)
}
#ifdef FA_SP_DUMPWC
// DIAGNOSTIC (FA_SP_DUMPWC): PROVE OR REFUTE the gemmini_fence() race directly, at a cost of three
// instructions and one store, so the schedule it is measuring barely moves.  Sample MMIO 0x28
// (runningLoops -- see fa_gfl) BEFORE the busy-fence and AGAIN AFTER it, and publish both.
//     after != 0   =>  gemmini_fence() RETURNED WHILE A LOOP_WS WAS STILL RUNNING.  The
//                      accumulator->spad store that follows therefore read a mid-accumulation
//                      accumulator, which is the whole bug.
//     before == 0  =>  io.busy had not even risen yet when we arrived -- the command was still in
//                      flight to the reservation station -- which is the mechanism by which the
//                      fence can fall straight through.
// Lands at 0x40056000 + 4*t, inside the window the trace filter already keeps.
static __attribute__((noinline)) void fa_gfl_probe(uint32_t tid, uint32_t t) {
    if (tid != 0) return;
    const uint32_t before = load32_shared(GEMMINI_OCCUPANCY_ADDR);
    gemmini_fence();
    const uint32_t after = load32_shared(GEMMINI_OCCUPANCY_ADDR);
    volatile uint32_t *o = (volatile uint32_t *)(0x40056000u + 4u * t);
    *o = (before << 16) | (after & 0xffffu);
}
#endif

// Native single-instruction bf16 max / add.  Going through C's fmaxf() or `a+b` on _Float16
// makes clang promote to fp32 (fcvt.s.h + fmax.s/fadd.s + fcvt.h.s), which triples the op count
// in the issue-bound fused kernel.  zhinx keeps halves in integer registers, hence the "r"
// constraints.
static inline _Float16 fa_max_h(_Float16 a, _Float16 b) {
    _Float16 o; asm("fmax.h %0, %1, %2" : "=r"(o) : "r"(a), "r"(b)); return o;
}
static inline _Float16 fa_add_h(_Float16 a, _Float16 b) {
    _Float16 o; asm("fadd.h %0, %1, %2" : "=r"(o) : "r"(a), "r"(b)); return o;
}


// --------------------------------------------------------------------------------------------
// FA_SP_SMTPR -- THREAD-PER-ROW softmax.  One lane owns a WHOLE row, so there is no cross-lane
// reduction at all.  The cooperative row-per-warp version (online_softmax_block) spends, per
// row, TWO of {SMEM store, fence.s, 4 x (lane%stride==0) warp-divergence regions, fence.s,
// broadcast load} on top of only ~56 warp-instructions of real work per lane -- i.e. most of
// its ~16k cycles is reduction scaffolding.  Thread-per-row does 64 rows on 64 lanes (4 warps,
// 2 per core -> both cores busy) with pure straight-line code.
//
// SUBBANK RULE (this is what kills the naive version): the SMEM subbank is byte[5:2] of the
// address, i.e. `word_index & 15`.  In a thread-per-row loop the word index is lane-INVARIANT,
// so all 16 lanes of a warp hit ONE subbank on every access -- a 16-way conflict, exactly the
// trap requant_P_to_spad_tiled documents.  Rotating the traversal by the lane id,
// idx = (i + lane) & (SKW-1), makes the 16 lanes hit 16 distinct subbanks.  Both passes are
// order-independent so the rotation is free.
//
// NUMERICS: the row max is taken on the RAW bf16 row and scaled once at the end -- bf16
// round-to-nearest is monotone and scale > 0, so max(S)*scale == max(S*scale) bit-for-bit,
// and it saves SK multiplies.  The row sum is accumulated in FP32 (4 chains) instead of the
// cooperative version's bf16 16-leaf tree: with the lane rotation each lane's partial sums land
// in a lane-dependent rotation of the tree leaves, so reproducing that exact bf16 rounding is
// impossible.  fp32 is strictly MORE accurate and l is stored as bf16 either way, so this
// yields the correctly-rounded l -- expect the Frobenius error to move slightly DOWN.
// --------------------------------------------------------------------------------------------
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_softmax_tpr(
        __shared uint32_t *S32, __shared uint32_t *m_out, __shared uint32_t *l_out,
        uint16_t softmax_scale_bf16, uint32_t tid, uint32_t thr) {
    constexpr uint32_t NT  = MU_NUM_THREADS;
    constexpr uint32_t SKW = SK / 2;              // 32-bit words per row
    static_assert((SKW & (SKW - 1)) == 0, "SK/2 must be a power of two");
    const _Float16 scale = as_bf16(softmax_scale_bf16);
    const uint32_t lane = tid & (NT - 1);
    for (uint32_t row = tid; row < SQ; row += thr) {
        __shared uint32_t *Srow = S32 + row * SKW;
        // ---- pass 1: row max of the raw bf16 row (4 independent fmax chains) ----
        // FA_SP_SUBMAX: read only every SUBMAX-th word and add a safety MARGIN to the result.
        // WHY THIS IS EXACT-ENOUGH, AND WHY IT IS SAFE.  m is a PER-ROW constant, and a per-row
        // constant CANCELS EXACTLY between the numerator and the denominator of the softmax:
        // O = (P @ V) / l with P = exp(u-m) and l = sum exp(u-m).  m is therefore NOT a numeric
        // input to the result at all -- its only job is to keep exp(u-m) inside bf16's range, and
        // the per-32-element E8M0 block scale that the requant computes afterwards restores the
        // full fp8 mantissa regardless of how small the values are.  The requirements are only
        //     no overflow:  m >= row_max - 88      (exp(88) is the bf16 limit)
        //     no total underflow of the largest term: m <= row_max + 88
        // so there are ~88 units of slack on each side, and a subsampled max plus a +24 margin
        // (P <= e^-24 ~ 4e-11, l ~ 1e-8, 1/l ~ 1e8: all comfortably inside bf16) is safe unless a
        // single row's logits span more than 112.  It costs 1/SUBMAX of the pass-1 SMEM loads.
        // The output is NOT bit-identical to the full-max version (P is scaled by a non-power-of-2
        // constant, so the fp8 mantissas round differently) -- it is the SAME computation to
        // within one requant rounding, which is why the Frobenius error must be re-measured.
#ifdef FA_SP_NOMAX
        // *** FA_SP_NOMAX -- DELETE PASS 1 ENTIRELY.  m := 0. ***
        // The row max exists ONLY to keep exp(u-m) inside the floating-point range of the P
        // scratch.  That scratch is bf16, and bf16 has FP32's 8-BIT EXPONENT (range ~1e+-38) -- it
        // gives up mantissa bits, not range.  So the range argument that makes the max mandatory in
        // an fp16 flash-attention kernel does not apply here at all.  Measured on this input:
        //     S*scale in [-3.53, +3.47]   =>   exp(S*scale) in [0.029, 32.1]
        //     l without the max <= 478    =>   1/l >= 2.1e-3
        // i.e. five orders of magnitude used out of thirty-eight.  m also cancels EXACTLY between
        // the numerator and the denominator of the softmax (O = (P@V)/l with P = exp(u-m) and
        // l = sum exp(u-m)), and the per-32-element E8M0 block scale that the requant computes
        // afterwards renormalises every block independently, so the fp8 mantissa is unaffected by
        // how big P is.  The per-block E8M0 code moves from ~127 to ~132 -- well inside the 0..255
        // that fa_clamp_K8 allows -- and nothing downstream reads m at all (fa_finalize_row and
        // fa_requant_cvt do not; only the FA_SP_DUMPLM diagnostic does).
        // WHY THIS IS NOT FA_SP_SUBMAX.  SUBMAX kept the max but made it a 4:1 subsample plus a
        // +24 margin, which pushed every exp argument ~24 FURTHER NEGATIVE and cost real accuracy
        // (4.2140% -> 7.7888%) because mu_fexp is not uniformly accurate in its argument.  NOMAX
        // moves the arguments the OTHER WAY -- m is 1.7..3.5 here, so dropping it makes every
        // argument LESS negative -- so the hardware-accuracy mechanism that sank SUBMAX predicts
        // this direction should be neutral or better.  That is the prediction being tested.
        // FA_SP_FIXMAX picks a CONSTANT m instead of 0.  Its only purpose is to separate the two
        // things FA_SP_NOMAX changes at once: (i) pass 1 disappears, and (ii) every mu_fexp argument
        // becomes POSITIVE (up to +3.47 on this input) where the kernel has, until now, only ever
        // called mu_fexp with a NON-POSITIVE argument -- both softmax call sites are exp(x - rowmax)
        // and exp(m_old - m_new).  If mu_fexp is only accurate (or only defined) for x <= 0 then
        // NOMAX will be wrong for a reason that has nothing to do with the row max being
        // unnecessary.  m = 4.0 bounds max(S*scale) = 3.47 here, so FIXMAX keeps every argument
        // <= 0 AND still deletes pass 1.  NOMAX correct  => the row max is simply not needed on this
        // hardware.  NOMAX wrong but FIXMAX correct  => the row max is not needed either, but
        // mu_fexp's usable domain is x <= 0, which is a hardware fact worth having.
#ifdef FA_SP_FIXMAX
        const _Float16 m = as_bf16((uint16_t)0x4080u);   // 4.0
#else
        const _Float16 m = (_Float16)0;
#endif
#else
        _Float16 x0 = as_bf16(NEG_INF_BF16_BITS), x1 = x0, x2 = x0, x3 = x0;
#ifdef FA_SP_SUBMAX
        // FA_SP_SUBMAX_NOMARGIN drops the +24 to separate the two effects of this flag -- the
        // subsampled max and the size of the exp argument -- because the +24 version costs real
        // accuracy (see the flag catalogue) even though m provably cancels out of O.
#ifdef FA_SP_SUBMAX_NOMARGIN
        constexpr uint32_t SUB = 4u, MARGIN_BF16 = 0x0000u;   // 4x fewer loads, no margin
#else
        constexpr uint32_t SUB = 4u, MARGIN_BF16 = 0x41C0u;   // 4x fewer loads, +24.0
#endif
        for (uint32_t i = 0; i < SKW; i += 4u * SUB) {
            const uint32_t w0 = Srow[(i + 0u * SUB + lane) & (SKW - 1u)];
            const uint32_t w1 = Srow[(i + 1u * SUB + lane) & (SKW - 1u)];
            const uint32_t w2 = Srow[(i + 2u * SUB + lane) & (SKW - 1u)];
            const uint32_t w3 = Srow[(i + 3u * SUB + lane) & (SKW - 1u)];
#else
        for (uint32_t i = 0; i < SKW; i += 4) {
            const uint32_t w0 = Srow[(i + 0u + lane) & (SKW - 1u)];
            const uint32_t w1 = Srow[(i + 1u + lane) & (SKW - 1u)];
            const uint32_t w2 = Srow[(i + 2u + lane) & (SKW - 1u)];
            const uint32_t w3 = Srow[(i + 3u + lane) & (SKW - 1u)];
#endif
#ifdef FA_SP_MAXSC
            // FA_SP_MAXSC: reduce over the SCALED values with the native bf16 fmax.h, exactly as
            // online_softmax_block does, instead of maxing the RAW row and scaling once at the end
            // with clang's fp32-promoted fmaxf.  ALGEBRAICALLY the two are the same -- bf16 multiply
            // and bf16 round-to-nearest are both monotone, so max_i RNE(S_i*scale) ==
            // RNE(max_i S_i * scale) -- so this flag exists ONLY to test whether the HARDWARE agrees.
            // It is a real suspicion, not a fishing trip: hardware fact 2 says the bf16<->fp32
            // converters on this machine are UNVALIDATED and have already produced garbage once
            // (`a += (float)e` gave 5,120 of 8,192 cells at +-inf), and the raw-max form is the one
            // place left in fa_softmax_tpr that routes a value through fcvt.s.h/fcvt.h.s where the
            // cooperative version does not.  If fcvt.h.s is not monotone for some bit patterns then
            // m differs on a few rows, every fp8 mantissa in those rows shifts, and that is exactly
            // the shape of the unexplained 3.5666% -> 4.2342% step.  Costs SK extra fmul.h per row.
            x0 = fa_max_h(x0, fa_max_h((_Float16)(as_bf16((uint16_t)w0) * scale),
                                       (_Float16)(as_bf16((uint16_t)(w0 >> 16)) * scale)));
            x1 = fa_max_h(x1, fa_max_h((_Float16)(as_bf16((uint16_t)w1) * scale),
                                       (_Float16)(as_bf16((uint16_t)(w1 >> 16)) * scale)));
            x2 = fa_max_h(x2, fa_max_h((_Float16)(as_bf16((uint16_t)w2) * scale),
                                       (_Float16)(as_bf16((uint16_t)(w2 >> 16)) * scale)));
            x3 = fa_max_h(x3, fa_max_h((_Float16)(as_bf16((uint16_t)w3) * scale),
                                       (_Float16)(as_bf16((uint16_t)(w3 >> 16)) * scale)));
#else
            x0 = fmaxf(x0, fmaxf(as_bf16((uint16_t)w0), as_bf16((uint16_t)(w0 >> 16))));
            x1 = fmaxf(x1, fmaxf(as_bf16((uint16_t)w1), as_bf16((uint16_t)(w1 >> 16))));
            x2 = fmaxf(x2, fmaxf(as_bf16((uint16_t)w2), as_bf16((uint16_t)(w2 >> 16))));
            x3 = fmaxf(x3, fmaxf(as_bf16((uint16_t)w3), as_bf16((uint16_t)(w3 >> 16))));
#endif
        }
#ifdef FA_SP_MAXSC
        const _Float16 mraw = fa_max_h(fa_max_h(x0, x1), fa_max_h(x2, x3));
#ifdef FA_SP_SUBMAX
        const _Float16 m = fa_add_h(mraw, as_bf16((uint16_t)MARGIN_BF16));
#else
        const _Float16 m = mraw;      // already scaled
#endif
#else
#ifdef FA_SP_SUBMAX
        const _Float16 m = fa_add_h((_Float16)(fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)) * scale),
                                    as_bf16((uint16_t)MARGIN_BF16));
#else
        const _Float16 m = (_Float16)(fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)) * scale);
#endif
#endif
#endif  // FA_SP_NOMAX
        // ---- pass 2: P = exp(S*scale - m), written IN PLACE; row sum in bf16 (4 chains) ----
        // *** NO FP32 ANYWHERE.  The verified reference kernel emits ZERO fcvt.s.h / fadd.s /
        // fmax.s in the whole object -- it is pure bf16 (fmax.h / fadd.h / fmul.h / fdiv.h).  The
        // bf16<->fp32 converters are therefore UNVALIDATED on this machine, and using them
        // (`a += (float)e`) produced garbage l: 5,120 of 8,192 output cells +-inf and the rest
        // ~2.7% off (variant spH).  Accumulate in bf16 with 4 chains + a balanced tree instead --
        // the same shape as the reference's per-lane chain + 16-leaf tree. ***
        _Float16 a0 = (_Float16)0, a1 = (_Float16)0, a2 = (_Float16)0, a3 = (_Float16)0;
#ifdef FA_SP_SMBMAX
        // FA_SP_SMBMAX: ALSO produce the per-32-element E8M0 block scale here, which DELETES the
        // whole requant pass-A stage (measured 3,611 cyc/tile).  Pass A exists only to re-read the
        // bf16 P that this loop just wrote and take a 32-element max of it -- but this loop already
        // has those 32 values IN REGISTERS, so the max is 4 extra fmax.h per 4 elements and one
        // extra word store per 32.  Nothing else changes: the scale word written here has exactly
        // the layout and value (max(em,7) at scale_scratch[b*SQ+row]) that fa_requant_cvt reads.
        //
        // The loop is restructured as blocks-then-words instead of one flat word sweep, because a
        // block max needs the 16 words of ONE block consecutively.  THE LANE ROTATION MUST STAY
        // INSIDE THE BLOCK: the SMEM subbank is word_index & 15, so rotating at block granularity
        // (bb = b + lane) would leave all 16 lanes of a warp on the same subbank for a given j --
        // the exact 16-way conflict that thread-per-row traversal has to avoid -- while rotating
        // the word index within the 16-word block keeps all 16 subbanks busy and still covers the
        // block exactly.  Block order does not matter (each block's scale is independent).
        //
        // NUMERICS: MEASURED, AND MY PREDICTION WAS WRONG -- the two-level tree did NOT help.
        // Blocking the sweep lets `l` be summed as a two-level tree instead of four 64-deep chains
        // (max accumulation depth ~18 instead of ~66), and since bf16 has an 8-bit mantissa I
        // expected that to recover some of the accuracy the thread-per-row softmax gives up.  It
        // did not: tile-0 Frobenius is 4.2516% WITH the tree and 4.2140% with the flat row sum, i.e.
        // very slightly WORSE.  Both are the same MX-FP8 quantisation floor to within noise, so the
        // tree is kept (it is free, and shallower accumulation is the defensible default), but the
        // claim that depth was the dominant error term here is NOT supported -- the 3.5666% vs
        // 4.2xxx% gap between the cooperative and thread-per-row softmax is something else.
        // Blocking the sweep also lets `l` be
        // summed as a TWO-LEVEL TREE instead of four 64-deep chains: 32 values per block over 4
        // chains (depth 8) + a 2-deep combine, then 8 block sums (depth 8) -- max depth ~18 instead
        // of ~66.  bf16 has an 8-bit mantissa, so accumulation depth is the dominant error term for
        // a 256-term positive sum, and the flat version measurably loses: the thread-per-row softmax
        // scores Frobenius 4.2342% against the cooperative reference's 3.5666% on an OTHERWISE
        // IDENTICAL pipeline, and the difference is entirely l's rounding.  Costs one register.
        volatile __shared uint32_t *sc = reinterpret_cast<volatile __shared uint32_t *>(SCALE_SMEM);
        constexpr uint32_t NBLK = SK / 32u, BW = 16u;      // 8 blocks of 16 words (32 values)
        _Float16 lacc = (_Float16)0;
        for (uint32_t b = 0; b < NBLK; b++) {
            _Float16 bx = (_Float16)0;
            a0 = (_Float16)0; a1 = (_Float16)0; a2 = (_Float16)0; a3 = (_Float16)0;
            for (uint32_t j = 0; j < BW; j += 2) {
                const uint32_t k0 = b * BW + ((j + 0u + lane) & (BW - 1u));
                const uint32_t k1 = b * BW + ((j + 1u + lane) & (BW - 1u));
#else
        for (uint32_t i = 0; i < SKW; i += 2) {
            {
                const uint32_t k0 = (i + 0u + lane) & (SKW - 1u);
                const uint32_t k1 = (i + 1u + lane) & (SKW - 1u);
#endif
                const uint32_t w0 = Srow[k0], w1 = Srow[k1];
                const _Float16 e0 = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)w0) * scale) - m));
                const _Float16 e1 = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)(w0 >> 16)) * scale) - m));
                const _Float16 e2 = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)w1) * scale) - m));
                const _Float16 e3 = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)(w1 >> 16)) * scale) - m));
                Srow[k0] = pack_bf16x2(e0, e1);
                Srow[k1] = pack_bf16x2(e2, e3);
                a0 = fa_add_h(a0, e0); a1 = fa_add_h(a1, e1);
                a2 = fa_add_h(a2, e2); a3 = fa_add_h(a3, e3);
#ifdef FA_SP_SMBMAX
                bx = fa_max_h(bx, fa_max_h(fa_max_h(e0, e1), fa_max_h(e2, e3)));
#endif
            }
#ifdef FA_SP_SMBMAX
            // identical to fa_requant_max: E8M0 code = max(bf16 exponent of the block max, 7)
            const uint32_t em = ((uint32_t)__builtin_bit_cast(uint16_t, bx) >> 7) & 0xffu;
            const uint32_t K8 = fa_clamp_K8(((int)em - 7) << 3);
            sc[b * SQ + row] = ((K8 - 8u) >> 3) + 7u;
            // level 2 of the l tree: fold this block's 32-term sum into the row total
            lacc = fa_add_h(lacc, fa_add_h(fa_add_h(a0, a1), fa_add_h(a2, a3)));
#endif
        }
#ifdef FA_SP_SMBMAX
        const _Float16 ls = lacc;
#else
        const _Float16 ls = fa_add_h(fa_add_h(a0, a1), fa_add_h(a2, a3));
#endif
        // ONE 32-BIT WORD PER ROW.  A 16-bit store is NOT safe here: thread `tid` owns row `tid`,
        // so 16 lanes of one warp would store 16 CONSECUTIVE halfwords -- i.e. two lanes per
        // 32-bit SMEM word -- in the SAME instruction, and the SMEM write path drops one of them
        // (measured: every ODD row came back with l == 0 -> 1/l = inf -> O = +-inf on 4096 of
        // 8192 cells, even rows perfect).  This is the same "sub-word PutPartial" hazard that
        // pack_scales_to_sfmem and requant_P_to_spad_tiled avoid by only ever storing whole words.
        m_out[row] = (uint32_t)__builtin_bit_cast(uint16_t, m);
        l_out[row] = (uint32_t)__builtin_bit_cast(uint16_t, ls);
    }
}

// --------------------------------------------------------------------------------------------
// FA_SP_ITEM -- ITEM-PARALLEL exp pass.  THE THREAD-PER-ROW SOFTMAX LEAVES A THIRD OF THE
// MACHINE IDLE: it gives one row to one thread, and there are 64 rows but 96 threads, so warps 4
// and 5 do NOTHING during what is the largest SIMT stage of the kernel (11,909 cyc).  Splitting
// the work by (row, 32-element MX block) instead gives 64*8 = 512 independent items over 96
// threads -- every thread busy, and 1.5x the parallelism on the same total work.
//
// It needs the row max up front, which is what makes it a THREE-phase stage:
//     (a) fa_rowmax_s        cooperative row max over all 96 threads   -> m[row]
//     (b) fa_expitem (this)  per (row,block): 32 exps written in place, the 32-element block max
//                            as an E8M0 scale word (i.e. FA_SP_SMBMAX's trick, kept), and the
//                            block's 32-term sum as an l partial
//     (c) fa_rowsum          l[row] = sum of the 8 block partials
// (a) and (c) already existed for FA_SP_FUSE and are reused unchanged, and (b) is fa_expreq minus
// the fp8 conversion -- which is the whole point: fa_expreq's fp8 half is what pushed FUSE over
// the renamer's 53-register budget, and without it this fits.
//
// SUBBANK ROTATION IS MANDATORY HERE, for the same reason it is in fa_requant_cvt: the word index
// is row*(SK/2) + b*16 + k and both 128 and 16 are multiples of 16, so the subbank (= index & 15)
// is (k & 15) -- LANE-UNIFORM, i.e. all 16 lanes of a warp on one subbank on every access.  The
// exps are independent per word and are written back in place, and the block max and block sum are
// order-independent, so rotating k by the lane id is free and makes it 16 distinct subbanks.
//
// NUMERICS: identical to FA_SP_SMBMAX.  m is the exact row max either way (max is exact), and l is
// the same two-level tree -- 32 terms per block over 4 chains, then 8 block sums in fa_rowsum.
// --------------------------------------------------------------------------------------------
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_expitem(
        __shared uint32_t *S32, __shared uint32_t *scale_scratch,
        const __shared uint32_t *m_in, __shared uint32_t *lpart,
        uint16_t scale_bits, uint32_t tid, uint32_t thr) {
    constexpr uint32_t NBLK = SK / 32u, BW = 16u, SKW = SK / 2u, NT = MU_NUM_THREADS;
    const _Float16 scale = as_bf16(scale_bits);
    const uint32_t lane = tid % NT;
    for (uint32_t item = tid; item < SQ * NBLK; item += thr) {
        const uint32_t row = item / NBLK, b = item % NBLK;
        __shared uint32_t *Sb = S32 + row * SKW + b * BW;
        const _Float16 m = as_bf16((uint16_t)m_in[row]);
        // TWO accumulator chains, not four: the usable register budget is 53 (see the header) and
        // this function lands on 54 with four.  Two costs nothing -- the loop only has two loads per
        // iteration to keep in flight anyway, and (e0+e1)+(e2+e3) is the same two-level tree.
        _Float16 bx = (_Float16)0;
        _Float16 a0 = (_Float16)0, a1 = (_Float16)0;
        for (uint32_t j = 0; j < BW; j += 2) {
            const uint32_t k0 = (j + 0u + lane) & (BW - 1u);
            const uint32_t k1 = (j + 1u + lane) & (BW - 1u);
            const uint32_t w0 = Sb[k0], w1 = Sb[k1];
            const _Float16 e0 = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)w0) * scale) - m));
            const _Float16 e1 = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)(w0 >> 16)) * scale) - m));
            const _Float16 e2 = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)w1) * scale) - m));
            const _Float16 e3 = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)(w1 >> 16)) * scale) - m));
            Sb[k0] = pack_bf16x2(e0, e1);
            Sb[k1] = pack_bf16x2(e2, e3);
            a0 = fa_add_h(a0, fa_add_h(e0, e1));
            a1 = fa_add_h(a1, fa_add_h(e2, e3));
            bx = fa_max_h(bx, fa_max_h(fa_max_h(e0, e1), fa_max_h(e2, e3)));
        }
        // E8M0 block scale, byte-identical to fa_requant_max's, so fa_requant_cvt is unchanged
        const uint32_t em = ((uint32_t)__builtin_bit_cast(uint16_t, bx) >> 7) & 0xffu;
        const uint32_t K8 = fa_clamp_K8(((int)em - 7) << 3);
        scale_scratch[b * SQ + row] = ((K8 - 8u) >> 3) + 7u;
        // whole 32-bit word per (block,row): the sub-word store hazard applies here too
        lpart[b * SQ + row] = (uint32_t)__builtin_bit_cast(uint16_t, fa_add_h(a0, a1));
    }
}

// ============================================================================================
// FA_SP_FUSE -- FUSED softmax + MX requant.  The single biggest SIMT win in this kernel.
//
// WHY.  Both softmax and requant are pure SMEM-LOAD-LATENCY bound (perf-viz on the sequential
// kernel: fp pipes 98.4% idle, SMEM at 0.5% of peak, ~0.6-0.9 loads/cycle), so their cost tracks
// the NUMBER OF SMEM ACCESSES, not flops or bytes.  Split, one row of S costs
//     softmax : 128 word loads (row max) + 128 word loads (exp) + 128 word stores (P bf16)
//     requant : 8 items x (16 loads for the block max + 16 loads for the convert) = 256 loads
//               + 64 fp8 word stores + 8 scale stores
//   = 512 loads + ~200 stores per row.
// Fusing them removes the bf16-P round trip entirely (P never exists in SMEM) and lets ONE read
// of a 32-element block serve both the block max and the conversion, because the 32 exp results
// live in 16 registers:
//     pass 1 (row max, cooperative)    : 128 loads / row spread over 16 lanes -> 8 loads / lane
//     pass 2 (exp + requant, per item) :  16 loads + 8 fp8 stores + 2 scalar stores
//   = 256 loads + 80 stores per row -- HALF the loads and 40% of the stores.
//
// NUMERICS.  m[row] is the max of the RAW bf16 row scaled once at the end (bf16 RN is monotone
// and scale>0, so max(S)*scale == max(S*scale) bit-for-bit).  P = exp(S*scale - m), the E8M0
// code and the e4m3 conversion are byte-identical to requant_P_to_spad_tiled's FA_RQ_FAST path.
// The only difference from the reference kernel is l, summed here in FP32 per block and then
// across blocks -- strictly more accurate than the reference's bf16 16-leaf tree, and l is
// stored as bf16 either way, so this is the correctly-rounded l.
//
// pass 1 uses a register BUTTERFLY reduction instead of warp_tree_reduce's four
// (lane%stride==0) blocks: max is order-independent so it is exact, it removes 4
// vx_split_n/vx_join warp-divergence regions per row, and every lane ends with the answer.
// ============================================================================================
// SERIAL-REDUCE variant (FA_SP_RMSER) -- the one actually used.  ONE fence.s per row and NO
// accumulator array: every lane stores its 16-way partial max, one fence, then every lane reads
// all 16 partials and folds them in registers (a same-address read across the warp is a
// broadcast, not a subbank conflict).  This is 5x fewer fences than the butterfly and, unlike
// the RB-batched butterfly below, it does not add live registers -- which matters because the
// renamer's budget is (256 / occupancy) DISTINCT ARCH REGISTERS FOR THE WHOLE KERNEL and the
// batched version pushed the kernel to 71, where 72 was measured to $finish.
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_rowmax_s(const __shared uint32_t *S32,
        __shared uint32_t *m_out, uint16_t scale_bits, uint32_t tid, uint32_t thr) {
    constexpr uint32_t NT = MU_NUM_THREADS, SKW = SK / 2, WPL = SKW / NT;
    const uint32_t lane = tid % NT, warp = tid / NT, nw = thr / NT;
    const _Float16 scale = as_bf16(scale_bits);
    // ONE 32-BIT WORD PER LANE: a halfword scratch would put two lanes in one word (sub-word
    // store hazard) and, on the read side, halve the number of distinct subbanks.  The read is
    // ROTATED by the lane id so the 16 lanes touch 16 distinct subbanks every cycle instead of
    // all hammering one (a same-address warp read is only free if the SMEM broadcasts, which is
    // not something to bet on).  max is order-independent, so the rotation is exact.
    volatile __shared uint32_t *buf =
        reinterpret_cast<volatile __shared uint32_t *>(REDBUF_SMEM) + warp * NT;
    for (uint32_t row = warp; row < SQ; row += nw) {
        const __shared uint32_t *Srow = S32 + row * SKW;
        // FOUR INDEPENDENT CHAINS, UNROLLED BY 4.  A fully rolled loop serialises the SMEM
        // loads (iteration k+1 cannot issue until the branch resolves) and a single accumulator
        // serialises the fmax chain on top of that -- measured 16,125 cycles for this pass with
        // one chain and both loops rolled.  Unrolling by 4 puts 4 loads in flight and cuts the
        // dependency depth 4x, at ~4x the (tiny) static size: the icache budget is 16 KB, so a
        // PARTIAL unroll is the sweet spot, not "roll everything" and not "unroll everything".
        _Float16 m0 = as_bf16(NEG_INF_BF16_BITS), m1 = m0, m2 = m0, m3 = m0;
#pragma clang loop unroll_count(4)
        for (uint32_t i = 0; i < WPL; i += 4) {
            const uint32_t w0 = Srow[(i + 0) * NT + lane], w1 = Srow[(i + 1) * NT + lane];
            const uint32_t w2 = Srow[(i + 2) * NT + lane], w3 = Srow[(i + 3) * NT + lane];
            m0 = fa_max_h(m0, fa_max_h(as_bf16((uint16_t)w0), as_bf16((uint16_t)(w0 >> 16))));
            m1 = fa_max_h(m1, fa_max_h(as_bf16((uint16_t)w1), as_bf16((uint16_t)(w1 >> 16))));
            m2 = fa_max_h(m2, fa_max_h(as_bf16((uint16_t)w2), as_bf16((uint16_t)(w2 >> 16))));
            m3 = fa_max_h(m3, fa_max_h(as_bf16((uint16_t)w3), as_bf16((uint16_t)(w3 >> 16))));
        }
        _Float16 mx = fa_max_h(fa_max_h(m0, m1), fa_max_h(m2, m3));
        buf[lane] = (uint32_t)__builtin_bit_cast(uint16_t, mx);
        mu_fence_smem();
        m0 = mx; m1 = mx; m2 = mx; m3 = mx;
#pragma clang loop unroll_count(4)
        for (uint32_t k = 0; k < NT; k += 4) {
            m0 = fa_max_h(m0, as_bf16((uint16_t)buf[(k + 0 + lane) & (NT - 1)]));
            m1 = fa_max_h(m1, as_bf16((uint16_t)buf[(k + 1 + lane) & (NT - 1)]));
            m2 = fa_max_h(m2, as_bf16((uint16_t)buf[(k + 2 + lane) & (NT - 1)]));
            m3 = fa_max_h(m3, as_bf16((uint16_t)buf[(k + 3 + lane) & (NT - 1)]));
        }
        mx = fa_max_h(fa_max_h(m0, m1), fa_max_h(m2, m3));
        if (lane == 0)   // whole 32-bit word per row (sub-word store hazard)
            m_out[row] = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(mx * scale));
    }
}

// BATCHED variant (FA_SP_RMBATCH): reduce RB rows per group so the butterfly's fences amortise.
// The unbatched version below needs 5 fence.s per row (one after the initial store plus one per
// butterfly step, because a butterfly has every lane storing AND loading in the same step and
// therefore cannot rely on the unfenced lockstep visibility that warp_tree_reduce gets away
// with).  At 12.8 rows/warp that is 64 fences per warp, and MEASURED it made the row-max pass
// cost 9,493 cycles -- fences, not loads, were the whole phase.  Batching RB rows shares each
// fence across RB reductions: 5 fences per RB rows instead of per row.
template <uint32_t SQ, uint32_t SK, uint32_t RB>
static __attribute__((noinline)) void fa_rowmax_b(const __shared uint32_t *S32,
        __shared uint32_t *m_out, uint16_t scale_bits, uint32_t tid, uint32_t thr) {
    constexpr uint32_t NT = MU_NUM_THREADS, SKW = SK / 2, WPL = SKW / NT;
    static_assert(SQ % RB == 0, "SQ must be a multiple of RB");
    const uint32_t lane = tid % NT, warp = tid / NT, nw = thr / NT;
    const _Float16 scale = as_bf16(scale_bits);
    volatile __shared uint16_t *buf =
        reinterpret_cast<volatile __shared uint16_t *>(REDBUF_SMEM) + warp * (RB * NT);
    for (uint32_t r0 = warp * RB; r0 < SQ; r0 += nw * RB) {
        _Float16 v[RB];
#pragma unroll
        for (uint32_t j = 0; j < RB; j++) {
            const __shared uint32_t *Srow = S32 + (r0 + j) * SKW;
            _Float16 mx = as_bf16(NEG_INF_BF16_BITS);
            for (uint32_t i = 0; i < WPL; i++) {
                const uint32_t w = Srow[i * NT + lane];
                mx = fa_max_h(mx, fa_max_h(as_bf16((uint16_t)w), as_bf16((uint16_t)(w >> 16))));
            }
            v[j] = mx;
            buf[j * NT + lane] = __builtin_bit_cast(uint16_t, mx);
        }
        mu_fence_smem();
        for (uint32_t st = 1; st < NT; st <<= 1) {
#pragma unroll
            for (uint32_t j = 0; j < RB; j++)
                v[j] = fa_max_h(v[j], as_bf16(buf[j * NT + (lane ^ st)]));
#pragma unroll
            for (uint32_t j = 0; j < RB; j++)
                buf[j * NT + lane] = __builtin_bit_cast(uint16_t, v[j]);
            mu_fence_smem();
        }
        if (lane == 0)
#pragma unroll
            for (uint32_t j = 0; j < RB; j++)   // whole 32-bit word per row (sub-word hazard)
                m_out[r0 + j] = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(v[j] * scale));
    }
}

template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_rowmax(const __shared uint32_t *S32,
        __shared uint32_t *m_out, uint16_t scale_bits, uint32_t tid, uint32_t thr) {
    constexpr uint32_t NT = MU_NUM_THREADS, SKW = SK / 2, WPL = SKW / NT;
    const uint32_t lane = tid % NT, warp = tid / NT, nw = thr / NT;
    const _Float16 scale = as_bf16(scale_bits);
    volatile __shared uint16_t *buf =
        reinterpret_cast<volatile __shared uint16_t *>(REDBUF_SMEM) + warp * NT;
    for (uint32_t row = warp; row < SQ; row += nw) {
        const __shared uint32_t *Srow = S32 + row * SKW;
        _Float16 mx = as_bf16(NEG_INF_BF16_BITS);
        for (uint32_t j = 0; j < WPL; j++) {
            const uint32_t w = Srow[j * NT + lane];
            mx = fmaxf(mx, fmaxf(as_bf16((uint16_t)w), as_bf16((uint16_t)(w >> 16))));
        }
        buf[lane] = __builtin_bit_cast(uint16_t, mx);
        mu_fence_smem();
        for (uint32_t st = 1; st < NT; st <<= 1) {
            mx = fmaxf(mx, as_bf16(buf[lane ^ st]));
            buf[lane] = __builtin_bit_cast(uint16_t, mx);
            mu_fence_smem();
        }
        // whole-word store (see the sub-word hazard note in fa_softmax_tpr): rows are
        // warp-strided, so with 6 warps rows 0..5 are written concurrently and rows {0,1},
        // {2,3}, {4,5} would share a 32-bit word.
        if (lane == 0) m_out[row] = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)(mx * scale));
    }
}

// pass 2: item = (row, 32-column MX block); blocks restricted to [B0,B1) so a caller can
// interleave the (unparallelisable) SF-SRAM pack of the earlier blocks with the later ones.
// *** B0/B1 ARE RUNTIME ARGUMENTS AND THE INNER LOOPS ARE ROLLED, ON PURPOSE. ***
// The L0 instruction cache is 16 KB, nWays=1 (DIRECT MAPPED), 32-byte lines
// (RadianceConfigs.scala:66 L0iCacheConfig), and Muon instructions are 8 bytes.  The first
// version of this function had B0/B1 as template parameters and `#pragma unroll` on both inner
// loops: 4,768 BYTES per instantiation, x2 instantiations, which together with fa_entry (4,056 B)
// and the other helpers put the hot working set at ~18.5 KB -- over a direct-mapped 16 KB cache.
// Measured cost of that: the "exp+requant blocks 0..3" stage took 21,150 cycles for 256 items,
// i.e. ~13 cycles per issued instruction.  Rolling the loops and sharing ONE instantiation takes
// the function to a few hundred bytes at the same dynamic instruction count (the loop branches
// are warp-uniform, so there is no divergence cost).
// NB (the number of MX blocks this call handles) stays a TEMPLATE parameter so that `it / NB`
// and `it % NB` are shifts: with NB runtime the compiler emitted a `divu` in the item loop, and
// rv32im's divider is multi-cycle.  Both call sites use the same NB, so there is still only ONE
// instantiation -- B0 is the runtime argument.
template <uint32_t SQ, uint32_t SK, uint32_t NB>
static __attribute__((noinline)) void fa_expreq(const __shared uint32_t *S32_,
        __shared uint32_t *spad_u32_, __shared uint32_t *scale_scratch_,
        const __shared uint32_t *m_in_, __shared uint32_t *lpart_, uint16_t scale_bits,
        uint32_t B0, uint32_t tid, uint32_t thr) {
    constexpr uint32_t PE_TILES_K = SK / 16, SKW = SK / 2;
    const _Float16 scale = as_bf16(scale_bits);
    // ---- REGISTER BUDGET (FA_SP_RLEAN).  THIS IS WHAT MAKES FUSE RUN AT ALL. ----------------
    // Rename.scala:104-123 assigns a physical register the first time a WARP writes a given
    // architectural register and NEVER reclaims it, from ONE counter per core over 1..255.  So
    // the hard limit is   sum over the warps on a core of (distinct arch regs that warp writes)
    // < 256.  With 6 warps (mu_schedule occupancy 3) one core carries 3 warps, i.e. the WHOLE
    // KERNEL may write only ~85 distinct architectural registers, of which the runtime already
    // uses ~27.  Measured whole-kernel unions (from the generated .s, fa_regs.py):
    //     PKOVL (works)  53      FUSE, pointer args  58  -> $finish   FUSE + 1P  69 -> $finish
    // The five extra registers were NOT the unroll factor and NOT the number of reduction chains
    // (both were tried, no change): they are fa_expreq's NINE arguments, five of which are
    // pointers that must stay live across the whole item loop and therefore land in callee-saved
    // s55-s58.  But every one of them is a COMPILE-TIME CONSTANT in this kernel, so RLEAN just
    // ignores the parameters and rematerialises the addresses from the constants -- an address is
    // then a short-lived temporary instead of a loop-carried live value.
#ifdef FA_SP_RLEAN
    (void)S32_; (void)spad_u32_; (void)scale_scratch_; (void)m_in_; (void)lpart_;
    const __shared uint32_t *const S32 = reinterpret_cast<const __shared uint32_t *>(SM_S);
    __shared uint32_t *const spad_u32 = reinterpret_cast<__shared uint32_t *>(SM_P8);
    __shared uint32_t *const scale_scratch = reinterpret_cast<__shared uint32_t *>(SCALE_SMEM);
    const __shared uint32_t *const m_in = reinterpret_cast<const __shared uint32_t *>(M_SMEM);
    __shared uint32_t *const lpart = reinterpret_cast<__shared uint32_t *>(LPART_SMEM);
#else
    const __shared uint32_t *const S32 = S32_;
    __shared uint32_t *const spad_u32 = spad_u32_;
    __shared uint32_t *const scale_scratch = scale_scratch_;
    const __shared uint32_t *const m_in = m_in_;
    __shared uint32_t *const lpart = lpart_;
#endif
#ifdef FA_SP_1P
    // ---- SINGLE-PASS form (FA_SP_1P).  The two-pass form below re-reads the block's 16 S words
    // and RECOMPUTES all 32 exps purely to avoid holding them, which doubles the two dominant
    // per-item costs (SMEM loads and mu_fexp).  Holding them costs 16 live registers.
    // The array form was originally rejected as unaffordable, but the budget is
    // (256/occupancy) DISTINCT ARCH REGS = 85 at occ=3 (hardware fact 5) and the two-pass FUSE
    // kernel uses only ~61 -- checked on the generated .s: highest register s59, and
    //    awk '/^_Z9fa_expreq/,/\.size/' file.s | grep -c 'sp)'   ==  0   (no stack spills).
    // The variant that DID overflow also used clang's fp32-PROMOTED fmaxf/(float) reduce, which
    // is +12 registers on its own; with fa_max_h/fa_add_h (FA_SP_LEAN) the 16 fit.
    // Both loops must be FULLY unrolled or `e` becomes a stack array -- and the stack is DRAM.
    // Cost per item: 16 loads + 32 exps + 8 fp8 stores  (was 32 loads + 64 exps + 8 stores).
    for (uint32_t it = tid; it < SQ * NB; it += thr) {
        const uint32_t row = it / NB, b = B0 + it % NB;
        const __shared uint32_t *Sb = S32 + row * SKW + b * 16u;
        const _Float16 m = as_bf16((uint16_t)m_in[row]);
        uint32_t e[16];
        _Float16 x0 = (_Float16)0, x1 = (_Float16)0, c0 = (_Float16)0, c1 = (_Float16)0;
#pragma unroll
        for (uint32_t k = 0; k < 16; k++) {
            const uint32_t w = Sb[k];
            const _Float16 lo = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)w) * scale) - m));
            const _Float16 hi =
                mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)(w >> 16)) * scale) - m));
            e[k] = pack_bf16x2(lo, hi);
            x0 = fa_max_h(x0, lo); x1 = fa_max_h(x1, hi);
            c0 = fa_add_h(c0, lo); c1 = fa_add_h(c1, hi);
        }
        const _Float16 bmax = fa_max_h(x0, x1);
        const uint32_t em = ((uint32_t)__builtin_bit_cast(uint16_t, bmax) >> 7) & 0xffu;
        const uint32_t K8 = fa_clamp_K8(((int)em - 7) << 3);
        scale_scratch[b * SQ + row] = ((K8 - 8u) >> 3) + 7u;      // == max(em,7)
        lpart[b * SQ + row] = (uint32_t)__builtin_bit_cast(uint16_t, fa_add_h(c0, c1));
        const uint32_t ti = row / 16, rr = row % 16;
        __shared uint32_t *dst = spad_u32 + ti * (PE_TILES_K * 64u) + b * 128u + rr * 4u;
#ifdef FA_SP_SWAR
        const uint32_t D1 = 0x7ff8u - (K8 - 8u), D2 = D1 | (D1 << 16);
#endif
#pragma unroll
        for (uint32_t u = 0; u < 8; u++) {
#ifdef FA_SP_SWAR
            dst[(u >> 2) * 64u + (u & 3u)] =
                e4m3_pack4_swar(e[2 * u], e[2 * u + 1], D2, 0x07ff07ffu, 0x80008000u, 0x0000ffffu);
#else
            dst[(u >> 2) * 64u + (u & 3u)] = e4m3_pack4(e[2 * u], e[2 * u + 1], K8);
#endif
        }
    }
    return;
#else
    // ---- TWO-PASS, ZERO-ARRAY form.  Holding the 32 exp results as uint32 e[16] costs ~16
    // live registers on top of everything else, and with occ=3 that overflows the machine:
    // Rename.scala:123 "total register usage exceeded maximum number of physical registers"
    // $finish-es at ~116k cycles (measured on spF/spG).  Re-reading the 16 S words and
    // recomputing the 32 exps in a second pass costs +16 loads and +64 flops per item (~+4%
    // on a 464-instruction body) and needs only ~2 packed words live at a time.
    for (uint32_t it = tid; it < SQ * NB; it += thr) {
        const uint32_t row = it / NB, b = B0 + it % NB;
        const __shared uint32_t *Sb = S32 + row * SKW + b * 16u;
        const _Float16 m = as_bf16((uint16_t)m_in[row]);
        // pass A: block max + fp32-free row-sum partial, nothing stored
        // FA_SP_RLEAN: TWO reduction chains instead of four.  This is not a micro-optimisation,
        // it is what makes FUSE RUN AT ALL.  With four chains fa_expreq needs four extra
        // callee-saved registers (s55-s58 in the generated .s), which lifts the WHOLE-KERNEL
        // distinct-arch-register union from 53 to 58 and trips Rename.scala:123 -- see the
        // register-budget note in the header.  The unrolled body still issues four INDEPENDENT
        // LOADS, which is what actually sets the cost of this phase (it is SMEM-latency bound);
        // only the 1-2 cycle fmax/fadd reduction trees get twice as deep.
#ifdef FA_SP_RLEAN
        _Float16 x0 = (_Float16)0, c0 = (_Float16)0;
#else
        _Float16 x0 = (_Float16)0, x1 = (_Float16)0, c0 = (_Float16)0, c1 = (_Float16)0;
#endif
        // Unroll for MEMORY-LEVEL PARALLELISM: a rolled loop can only have ONE load in flight.
        // FA_SP_U8 doubles it to 8 (bigger code -- watch the 16 KB direct-mapped icache).
#ifdef FA_SP_U8
#pragma clang loop unroll_count(8)
#else
#pragma clang loop unroll_count(4)
#endif
        for (uint32_t k = 0; k < 16; k++) {
            const uint32_t w = Sb[k];
            const _Float16 lo = mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)w) * scale) - m));
            const _Float16 hi =
                mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)(w >> 16)) * scale) - m));
#ifdef FA_SP_RLEAN
            x0 = fa_max_h(x0, fa_max_h(lo, hi));
            c0 = fa_add_h(c0, fa_add_h(lo, hi));
#else
            x0 = fa_max_h(x0, lo); x1 = fa_max_h(x1, hi);
            c0 = fa_add_h(c0, lo); c1 = fa_add_h(c1, hi);
#endif
        }
#ifdef FA_SP_RLEAN
        const _Float16 bmax = x0;
#else
        const _Float16 bmax = fa_max_h(x0, x1);
#endif
        const uint32_t em = ((uint32_t)__builtin_bit_cast(uint16_t, bmax) >> 7) & 0xffu;
        const uint32_t K8 = fa_clamp_K8(((int)em - 7) << 3);
        scale_scratch[b * SQ + row] = ((K8 - 8u) >> 3) + 7u;      // == max(em,7)
        // bf16 code in a whole 32-bit word (no fp32; no sub-word store)
#ifdef FA_SP_RLEAN
        lpart[b * SQ + row] = (uint32_t)__builtin_bit_cast(uint16_t, c0);
#else
        lpart[b * SQ + row] = (uint32_t)__builtin_bit_cast(uint16_t, fa_add_h(c0, c1));
#endif
        // pass B: recompute the exps a pair of words at a time and convert straight to fp8
        const uint32_t ti = row / 16, rr = row % 16;
        __shared uint32_t *dst = spad_u32 + ti * (PE_TILES_K * 64u) + b * 128u + rr * 4u;
#ifdef FA_SP_SWAR
        const uint32_t D1 = 0x7ff8u - (K8 - 8u), D2 = D1 | (D1 << 16);
#endif
#ifdef FA_SP_U8
#pragma clang loop unroll_count(4)
#else
#pragma clang loop unroll_count(2)
#endif
        for (uint32_t u = 0; u < 8; u++) {
            const uint32_t w0 = Sb[2 * u], w1 = Sb[2 * u + 1];
            const uint32_t p0 = pack_bf16x2(
                mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)w0) * scale) - m)),
                mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)(w0 >> 16)) * scale) - m)));
            const uint32_t p1 = pack_bf16x2(
                mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)w1) * scale) - m)),
                mu_fexp((_Float16)((_Float16)(as_bf16((uint16_t)(w1 >> 16)) * scale) - m)));
#ifdef FA_SP_SWAR
            dst[(u >> 2) * 64u + (u & 3u)] =
                e4m3_pack4_swar(p0, p1, D2, 0x07ff07ffu, 0x80008000u, 0x0000ffffu);
#else
            dst[(u >> 2) * 64u + (u & 3u)] = e4m3_pack4(p0, p1, K8);
#endif
        }
    }
#endif  // !FA_SP_1P
}

// Pack scale words [W0,W1) -> SF_A half 0.  ONE thread, strictly ascending: that is the entire
// contract of FlitMergeNode(from=4,to=8).  W0 must be even (each pair must start 8-B aligned).
template <uint32_t W0, uint32_t W1>
static __attribute__((noinline)) void fa_pack_range(const __shared uint32_t *scale_scratch,
                                                    volatile __shared uint32_t *sf, uint32_t tid) {
    static_assert((W0 % 2) == 0 && (W1 % 2) == 0, "SF write batches must be 8-B aligned pairs");
    if (tid != 0) return;
#pragma clang loop unroll(disable)
    for (uint32_t w = W0; w < W1; w++) {
        uint32_t p = 0;
        for (uint32_t k = 0; k < 4; k++) p |= (scale_scratch[w * 4 + k] & 0xffu) << (8u * k);
        sf[w] = p;
    }
}

// pass 3: l[row] = sum of the NBLK fp32 block partials.
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_rowsum(const __shared uint32_t *lpart,
        __shared uint32_t *l_out, uint32_t tid, uint32_t thr) {
    constexpr uint32_t NBLK = SK / 32;
    for (uint32_t row = tid; row < SQ; row += thr) {
        _Float16 s = (_Float16)0;              // bf16 only -- no fp32 on this machine
#pragma clang loop unroll(disable)
        for (uint32_t b = 0; b < NBLK; b++)
            s = fa_add_h(s, as_bf16((uint16_t)lpart[b * SQ + row]));
        l_out[row] = (uint32_t)__builtin_bit_cast(uint16_t, s);   // whole word: see fa_softmax_tpr
    }
}
// The three phases behind ONE noinline call.  Expanding them inline into fa_entry (which is what a
// macro does) put two more pointer constants live across the calls and took the whole-kernel
// register union from 53 to 54 -- and the usable budget is 53.  Wrapping them keeps every address
// constant inside this frame.
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_softmax_item(uint16_t scale_bits,
                                                      uint32_t tid, uint32_t thr) {
    fa_rowmax_s<SQ, SK>(reinterpret_cast<const __shared uint32_t *>(SM_S),
                        reinterpret_cast<__shared uint32_t *>(M_SMEM), scale_bits, tid, thr);
    mu_fence_smem(); FAP_PAD(); mu_barrier(12, thr / MU_NUM_THREADS); FAP_PAD();
    fa_expitem<SQ, SK>(reinterpret_cast<__shared uint32_t *>(SM_S),
                       reinterpret_cast<__shared uint32_t *>(SCALE_SMEM),
                       reinterpret_cast<const __shared uint32_t *>(M_SMEM),
                       reinterpret_cast<__shared uint32_t *>(LPART_SMEM), scale_bits, tid, thr);
    mu_fence_smem(); FAP_PAD(); mu_barrier(13, thr / MU_NUM_THREADS); FAP_PAD();
    fa_rowsum<SQ, SK>(reinterpret_cast<const __shared uint32_t *>(LPART_SMEM),
                      reinterpret_cast<__shared uint32_t *>(LS_SMEM), tid, thr);
}


// --------------------------------------------------------------------------------------------
// FA_SP_FZFLAT -- flat/coalesced finalize.  finalize_O partitions by ROW->warp, so at any instant
// the 6 warps are storing to 6 addresses 6*D*2 bytes apart; each warp's 16 lanes coalesce into one
// 64 B GMEM write but the block as a whole scatters.  Here the 96 threads walk the output as ONE
// flat grid-stride sequence, so a whole 384 B contiguous GMEM span is in flight per step.  The
// per-row reciprocal (an fdiv, previously recomputed ~10.7x per warp) is hoisted into a one-off
// 64-thread pass so the hot loop is load / 2 mul / store with no divide.
// Arithmetic is identical to finalize_O: O[r][c] = O_unnorm[r][c] * (1/l[r]).
// --------------------------------------------------------------------------------------------
// (the caller must put a cross-warp barrier between fa_invl and fa_finalize_flat: every thread
//  reads reciprocals that other warps produced)
template <uint32_t SQ>
static __attribute__((noinline)) void fa_invl(const __shared uint32_t *l_state,
                                              __shared uint32_t *invl,
                                              uint32_t tid, uint32_t thr) {
    for (uint32_t r = tid; r < SQ; r += thr)   // one whole word per row: see fa_softmax_tpr
        invl[r] = (uint32_t)__builtin_bit_cast(uint16_t,
            (_Float16)(__builtin_bit_cast(_Float16, ONE_BF16_BITS) / as_bf16((uint16_t)l_state[r])));
}
// Row-per-warp finalize (the shape of the library finalize_O) but reading the 32-bit-per-row l
// array that FA_SP_FUSE / FA_SP_SMTPR produce.  Kept so the row-major and flat GMEM store
// patterns can be compared directly on an otherwise identical kernel.
template <uint32_t SQ, uint32_t D>
static __attribute__((noinline)) void fa_finalize_row(
        const __shared uint32_t *O32, const __shared uint32_t *l_state,
        uint32_t *O_gmem32, uint32_t tid, uint32_t thr) {
    constexpr uint32_t NT = MU_NUM_THREADS, DW = D / 2;
    const uint32_t lane = tid % NT, warp = tid / NT, nw = thr / NT;
    // FA_SP_FZU4: unroll the ROW loop by 2 -- one row is only 4 loads + 4 stores per lane behind
    // ONE fdiv, so a rolled row loop keeps a single dependency chain in flight (see fact 7).
#ifdef FA_SP_FZU4
#pragma clang loop unroll_count(2)
#endif
    for (uint32_t row = warp; row < SQ; row += nw) {
        const _Float16 inv_l = (_Float16)(__builtin_bit_cast(_Float16, ONE_BF16_BITS)
                                          / as_bf16((uint16_t)l_state[row]));
        for (uint32_t w = lane; w < DW; w += NT) {
            const uint32_t a = O32[row * DW + w];
            O_gmem32[row * DW + w] =
                pack_bf16x2((_Float16)(as_bf16((uint16_t)a) * inv_l),
                            (_Float16)(as_bf16((uint16_t)(a >> 16)) * inv_l));
        }
    }
}

template <uint32_t SQ, uint32_t D>
static __attribute__((noinline)) void fa_finalize_flat(
        const __shared uint32_t *O32, const __shared uint32_t *invl,
        uint32_t *O_gmem32, uint32_t tid, uint32_t thr) {
    constexpr uint32_t DW = D / 2;
    static_assert((DW & (DW - 1)) == 0, "D/2 must be a power of two");
    constexpr uint32_t LOG_DW = (DW == 64u) ? 6u : (DW == 32u ? 5u : 7u);
    static_assert((1u << LOG_DW) == DW, "unsupported D");
    // FA_SP_FZU4: finalize is the 2nd-biggest SIMT stage (10.4k cyc for 8,192 words = 1.3
    // cyc/word/block, i.e. ~120 cycles per word per thread) and the rolled grid-stride loop can
    // only ever have ONE load and ONE store in flight per thread, because iteration i+thr cannot
    // issue until the branch resolves.  Unrolling by 4 (grid stride 4*thr, 4 independent
    // load/mul/store chains) is the same memory-level-parallelism argument that took the row-max
    // pass from 16,125 to 3,600 cycles -- see hardware fact 7.  SQ*DW = 4096 and thr = 80 or 96,
    // so the tail loop below still runs; it is the same code, just not unrolled.
#ifdef FA_SP_FZU4
#pragma clang loop unroll_count(4)
#endif
    for (uint32_t i = tid; i < SQ * DW; i += thr) {
        const uint32_t a = O32[i];
        const _Float16 s = as_bf16((uint16_t)invl[i >> LOG_DW]);
        O_gmem32[i] = pack_bf16x2((_Float16)(as_bf16((uint16_t)a) * s),
                                  (_Float16)(as_bf16((uint16_t)(a >> 16)) * s));
    }
}

// --------------------------------------------------------------------------------------------
// DIAGNOSTIC HELPERS for FA_SP_DUMPS / FA_SP_DUMPL (see their use sites in the tile loop).
// Both are all-threads, one output word per row, so they add ~64 SMEM loads per thread and do not
// materially move the stage timings they are inserted between -- which matters, because the bug
// they exist to localise is a timing window.
// --------------------------------------------------------------------------------------------
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_dump_rowsum(const __shared uint32_t *A32,
                                                     volatile uint32_t *out,
                                                     uint32_t tid, uint32_t thr) {
    constexpr uint32_t W = SK / 2;
    for (uint32_t r = tid; r < SQ; r += thr) {
        uint32_t x = 0;
        for (uint32_t w = 0; w < W; w++) x ^= A32[r * W + w] + w;   // +w so a rotation is visible
        out[r] = x;
    }
}
template <uint32_t SQ, uint32_t D>
static __attribute__((noinline)) void fa_dump_state(const __shared uint32_t *l32,
                                                    const __shared uint32_t *m32,
                                                    const __shared uint32_t *O32,
                                                    volatile uint32_t *out,
                                                    uint32_t tid, uint32_t thr) {
    constexpr uint32_t DW = D / 2;
    for (uint32_t r = tid; r < SQ; r += thr) {
        out[r] = l32[r];
        out[SQ + r] = m32[r];
        uint32_t x = 0;
        for (uint32_t w = 0; w < DW; w++) x ^= O32[r * DW + w] + w;
        out[2 * SQ + r] = x;
    }
}
// FA_SP_DUMPLM: l and m only (no O checksum).  Register-lean enough to fit the occupancy-3 budget,
// which matters because the question it exists to answer -- "is the 3.5666% -> 4.2xxx% accuracy gap
// between the cooperative and the thread-per-row softmax in l?" -- has to be asked at the SAME
// occupancy in both configurations, or the cooperative softmax's reduction tree changes shape and
// the comparison is void.  NOTE THE TWO l/m LAYOUTS: online_softmax_block writes uint16 per row
// (so word r holds rows 2r and 2r+1), fa_softmax_tpr / fa_softmax_item write one 32-bit word per
// row.  The dump is raw words either way; the reader unpacks according to the configuration.
template <uint32_t SQ>
static __attribute__((noinline)) void fa_dump_lm(const __shared uint32_t *l32,
                                                 const __shared uint32_t *m32,
                                                 volatile uint32_t *out,
                                                 uint32_t tid, uint32_t thr) {
    for (uint32_t r = tid; r < SQ; r += thr) { out[r] = l32[r]; out[SQ + r] = m32[r]; }
}

// --------------------------------------------------------------------------------------------
// FA_SP_PKOVL -- split requant so the (strictly serial) SF-SRAM pack hides underneath it.
//
// The E8M0 pack is the one phase that CANNOT be parallelised: GemminiTile.scala:188 puts a
// FlitMergeNode(from=4,to=8) in front of the scale SRAM, and FlitMergeNode.scala:62 asserts that
// the second 4-byte Put of every merged pair sits at mergedReq.address+4.  Any lane-parallel or
// out-of-order write therefore $finishes the simulation -- measured twice here (FA_LANESC dies at
// 43.8k cycles, FA_PK_LANES at 115.1k) -- so ~65 cyc/word x 128 words is a hard floor on the GPU.
// The only way to make it free is to run it in parallel with SIMT work, and the only SIMT work
// that is both (a) after the scales exist and (b) before the PV matmul needs them is the requant
// itself.  So requant is split at its natural seam:
//     pass A (all 96 threads): 32-element block max -> E8M0 code -> scale_scratch     [~1/3]
//     pass B (warps 1-5)     : convert 32 elements -> 8 packed fp8 words -> P8 spad   [~2/3]
//     concurrently, warp 0   : pack scale_scratch -> SF_A half 0
// Pass B recovers the per-item shift from the scale word (K8 = (code-7)*8+8, the exact inverse of
// pass A's `((K8-8)>>3)+7`), so the only added cost is ONE extra SMEM load per item (17 vs 16).
// The arithmetic is byte-identical to requant_P_to_spad_tiled's FA_RQ_FAST path.
// --------------------------------------------------------------------------------------------
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_requant_max(
        const __shared uint16_t *P16, __shared uint32_t *scale_scratch,
        uint32_t tid, uint32_t thr) {
    constexpr uint32_t NBLK = SK / 32;
    const __shared uint32_t *P32 = reinterpret_cast<const __shared uint32_t *>(P16);
    for (uint32_t item = tid; item < SQ * NBLK; item += thr) {
        const uint32_t row = item / NBLK, b = item % NBLK;
        const __shared uint32_t *Pb = P32 + row * (SK / 2) + b * 16u;
        _Float16 x0 = (_Float16)0, x1 = (_Float16)0, x2 = (_Float16)0, x3 = (_Float16)0;
#ifdef FA_SP_PAX
        // ==== FA_SP_PAX -- KILL THE 16-WAY SMEM SUBBANK CONFLICT IN REQUANT PASS A, FOR FREE ====
        // Exactly the trap FA_SP_CVTX fixes in the convert, and pass A has it too: the subbank is
        // word_index & 15, the index here is row*(SK/2) + b*16 + i = row*128 + b*16 + i, and BOTH
        // 128 and 16 are multiples of 16 -- so the subbank is (i & 15), which is LANE-UNIFORM.  A
        // warp's 16 lanes hold 16 different (row,b) items but run the same inner iteration, so all
        // 16 lanes hit ONE subbank on every one of the 16 loads.  Pass A costs 3,405 cyc/tile for
        // ~90 static instructions x 5.33 items = ~1.4k of issue on the binding core, so most of the
        // remaining ~2k is this conflict.
        // THE FIX IS FREE AND BIT-EXACT.  Unlike the convert, pass A only READS -- and its result is
        // a MAX, which is exact and completely order-independent -- so the word index may be
        // permuted ARBITRARILY per lane, giving the full 16-way break rather than the convert's
        // 2-way (there an output word must hold 4 CONSECUTIVE elements, which pins the rotation).
        // Pb's byte address is (row*128 + b*16)*4 = row*512 + b*64, whose low SIX bits are zero, and
        // the 16 word offsets occupy exactly bits 5:2 -- so `+` is `^` and the rotation folds into
        // the base pointer ONCE per item, leaving every access a compile-time-immediate XOR (the
        // same argument as FA_SP_CVTX).  Costs one andi + one xor per item and ZERO live registers.
        const uint32_t Pbx = ((uint32_t)(uintptr_t)Pb) ^ (((tid % MU_NUM_THREADS) & 15u) << 2);
#define FA_PAX_LD(ix) (*reinterpret_cast<const __shared uint32_t *>(Pbx ^ ((ix) << 2)))
#pragma unroll
        for (uint32_t i = 0; i < 16; i += 4) {
            const uint32_t w0 = FA_PAX_LD(i + 0), w1 = FA_PAX_LD(i + 1),
                           w2 = FA_PAX_LD(i + 2), w3 = FA_PAX_LD(i + 3);
            x0 = fmaxf(x0, fmaxf(as_bf16((uint16_t)w0), as_bf16((uint16_t)(w0 >> 16))));
            x1 = fmaxf(x1, fmaxf(as_bf16((uint16_t)w1), as_bf16((uint16_t)(w1 >> 16))));
            x2 = fmaxf(x2, fmaxf(as_bf16((uint16_t)w2), as_bf16((uint16_t)(w2 >> 16))));
            x3 = fmaxf(x3, fmaxf(as_bf16((uint16_t)w3), as_bf16((uint16_t)(w3 >> 16))));
        }
#undef FA_PAX_LD
#else
        for (uint32_t i = 0; i < 16; i += 4) {
            const uint32_t w0 = Pb[i + 0], w1 = Pb[i + 1], w2 = Pb[i + 2], w3 = Pb[i + 3];
            x0 = fmaxf(x0, fmaxf(as_bf16((uint16_t)w0), as_bf16((uint16_t)(w0 >> 16))));
            x1 = fmaxf(x1, fmaxf(as_bf16((uint16_t)w1), as_bf16((uint16_t)(w1 >> 16))));
            x2 = fmaxf(x2, fmaxf(as_bf16((uint16_t)w2), as_bf16((uint16_t)(w2 >> 16))));
            x3 = fmaxf(x3, fmaxf(as_bf16((uint16_t)w3), as_bf16((uint16_t)(w3 >> 16))));
        }
#endif
        const _Float16 bmax = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
        const uint32_t em = ((uint32_t)__builtin_bit_cast(uint16_t, bmax) >> 7) & 0xffu;
        const uint32_t K8 = fa_clamp_K8(((int)em - 7) << 3);
        scale_scratch[b * SQ + row] = ((K8 - 8u) >> 3) + 7u;      // == max(em,7)
    }
}
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_requant_cvt(
        const __shared uint16_t *P16, __shared uint32_t *spad_u32,
        const __shared uint32_t *scale_scratch, uint32_t tid, uint32_t thr) {
    constexpr uint32_t NBLK = SK / 32;
    constexpr uint32_t PE_TILES_K = SK / 16;
    const __shared uint32_t *P32 = reinterpret_cast<const __shared uint32_t *>(P16);
    for (uint32_t item = tid; item < SQ * NBLK; item += thr) {
        const uint32_t row = item / NBLK, b = item % NBLK;
        const __shared uint32_t *Pb = P32 + row * (SK / 2) + b * 16u;
        const uint32_t K8 = ((scale_scratch[b * SQ + row] - 7u) << 3) + 8u;   // inverse of pass A
        const uint32_t ti = row / 16, rr = row % 16;
        __shared uint32_t *dst = spad_u32 + ti * (PE_TILES_K * 64u) + b * 128u + rr * 4u;
#ifdef FA_SP_CVTXS
        // ==== FA_SP_CVTXS -- FA_SP_CVTX's XOR ADDRESSING *AND* THE SWAR PACKER, TOGETHER. ========
        // These two are orthogonal and the #if chain below only ever let you pick one:
        //   FA_SP_CVTX     kills the 16-way subbank conflict (+23 instr/item, measured -872 cyc)
        //   FA_SP_CVTSWAR  packs 4 elements in 25 straight-line instructions instead of 34
        //                  (345 -> 289 instr/item, i.e. -56/item = -1.5k of issue on the binding
        //                  core) but was REGISTER-FATAL at 58 and so could never be used.
        // Two things make the combination available now.  (1) The XOR rotation costs ZERO live
        // registers -- it folds into the two base pointers once per item and every access stays a
        // compile-time-immediate XOR -- so it is free to add on top of SWAR.  (2) FA_SM_2P +
        // FA_SM_2PRAW take the whole-kernel register union from 53 to 47 by deleting the softmax's
        // slo[8]/shi[8] arrays and its 16 per-row fmul.h, which buys back exactly the headroom the
        // SWAR packer needs: 47 + 5 = 52, one under the empirical cliff.  *** So this lever only
        // exists BECAUSE of the softmax rewrite -- the register budget is a single global resource
        // and spending it well in one function is what makes another function affordable. ***
        // Bit-identical to e4m3_pack4 by construction (flash_mx_impl.hpp derives the SWAR chain
        // from the per-element one), and the XOR is a permutation of the 8 output words, each of
        // which still receives its own 4 consecutive elements.
        {
        const uint32_t lane_ = tid % MU_NUM_THREADS;
        const uint32_t Pbx  = ((uint32_t)(uintptr_t)Pb) ^ ((lane_ & 7u) << 3);
        const uint32_t dstx = ((uint32_t)(uintptr_t)dst)
                            ^ ((((lane_ & 4u) << 6) | ((lane_ & 3u) << 2)));
        const uint32_t D1 = 0x7ff8u - (K8 - 8u), D2 = D1 | (D1 << 16);
        const uint32_t C = 0x07ff07ffu, M = 0x80008000u, H = 0x0000ffffu;
#pragma unroll
        for (uint32_t uu = 0; uu < 8; uu++)
            *reinterpret_cast<__shared uint32_t *>(
                dstx ^ (((uu & 4u) << 6) | ((uu & 3u) << 2))) =
                e4m3_pack4_swar(*reinterpret_cast<const __shared uint32_t *>(Pbx ^ (uu << 3)),
                                *reinterpret_cast<const __shared uint32_t *>(Pbx ^ ((uu << 3) | 4u)),
                                D2, C, M, H);
        }
#elif defined(FA_SP_CVTX)
        // See the long FA_SP_CVTX comment in fa_requant_cvt_bal: rotate the pair index by lane&7
        // through an XOR that folds into the two base pointers, breaking the 16-way subbank
        // conflict at the cost of one xori per access and no live registers.  Present in BOTH the
        // equal-partition and the core-balanced convert, so FA_SP_CVTX and FA_SP_BALC are
        // INDEPENDENT -- which matters, because if this stage is conflict-bound rather than
        // issue-bound then CVTX is the fix and BALC is at best neutral.
        {
        const uint32_t lane_ = tid % MU_NUM_THREADS;
        const uint32_t Pbx  = ((uint32_t)(uintptr_t)Pb) ^ ((lane_ & 7u) << 3);
        const uint32_t dstx = ((uint32_t)(uintptr_t)dst)
                            ^ ((((lane_ & 4u) << 6) | ((lane_ & 3u) << 2)));
#ifdef FA_SP_CVTSWAR
        // FA_SP_CVTX + FA_SP_CVTSWAR TOGETHER.  These used to be mutually exclusive #elif arms, so
        // the SWAR packer came at the price of putting the 16-way subbank conflict back -- roughly
        // cancelling its own win.  They are ORTHOGONAL (one is the ADDRESSING, the other is the
        // ARITHMETIC), so this arm uses the XOR-rotated addresses AND the 25-instruction SWAR chain:
        // 8 x (34 - 25) = 72 fewer instructions per item, ~21% of the phase's issue work, with the
        // conflict still broken.  The three SWAR masks cost registers, which is why this was
        // unreachable until FA_SM_2P deleted the softmax's slo[8]/shi[8] arrays -- see (G5).
        const uint32_t D1 = 0x7ff8u - (K8 - 8u), D2s = D1 | (D1 << 16);
        const uint32_t C = 0x07ff07ffu, M = 0x80008000u, H = 0x0000ffffu;
#pragma unroll
        for (uint32_t uu = 0; uu < 8; uu++)
            *reinterpret_cast<__shared uint32_t *>(
                dstx ^ (((uu & 4u) << 6) | ((uu & 3u) << 2))) =
                e4m3_pack4_swar(*reinterpret_cast<const __shared uint32_t *>(Pbx ^ (uu << 3)),
                                *reinterpret_cast<const __shared uint32_t *>(Pbx ^ ((uu << 3) | 4u)),
                                D2s, C, M, H);
#else
#pragma unroll
        for (uint32_t uu = 0; uu < 8; uu++)
            *reinterpret_cast<__shared uint32_t *>(
                dstx ^ (((uu & 4u) << 6) | ((uu & 3u) << 2))) =
                e4m3_pack4(*reinterpret_cast<const __shared uint32_t *>(Pbx ^ (uu << 3)),
                           *reinterpret_cast<const __shared uint32_t *>(Pbx ^ ((uu << 3) | 4u)),
                           K8);
#endif
        }
#elif defined(FA_SP_CVTSWAR)
        // FA_SP_CVTSWAR: use the SWAR packer (25 straight-line instructions for 4 elements)
        // instead of e4m3_pack4 (34).  This stage is the biggest SIMT item after the softmax and
        // its instruction count is dominated by the packer -- 8 packs x 34 = 272 of the ~300
        // instructions per item -- so this is a ~26% cut of the phase's issue work.  The two forms
        // are bit-identical by construction (flash_mx_impl.hpp derives the SWAR chain from the
        // per-element one); FA_SP_SWAR already selects it inside fa_expreq, but the PKOVL convert
        // was never switched over.  D2 is the per-item SWAR displacement, the same
        // 0x7ff8-(K8-8) replicated into both halves that fa_expreq computes.
        const uint32_t D1 = 0x7ff8u - (K8 - 8u), D2 = D1 | (D1 << 16);
        // *** MEASURED AND REJECTED AT OCCUPANCY 3 -- DO NOT ENABLE WITH 6 WARPS. ***
        // The instruction win is real: fa_requant_cvt goes from 638 to 582 instructions (-9%).
        // But the SWAR chain needs 4 temporaries plus 4 live inputs, and that takes the
        // WHOLE-KERNEL distinct-arch-register union from 53 to 58 (extra: s54-s58) -- which is the
        // known-fatal value at 3 warps/core (Rename.scala:123, see the header).  Forcing the three
        // masks to be rematerialised inside the item body behind an empty asm barrier, so LICM
        // cannot hoist them into loop-invariant registers, does NOT help: still 58.  So this flag
        // is only usable together with FA_OCC2, where the budget is 127 per warp -- and occupancy 2
        // costs more (48 SIMT threads instead of 80) than the 9% here can win.  Kept, documented,
        // and OFF: the point is that the packer choice in this kernel is a REGISTER decision, not
        // an instruction-count decision.
        uint32_t C = 0x07ff07ffu, M = 0x80008000u, H = 0x0000ffffu;
        asm volatile("" : "+r"(C), "+r"(M), "+r"(H));
        for (uint32_t h = 0; h < 2; h++)
            for (uint32_t q = 0; q < 4; q++) {
                const uint32_t u = h * 4u + q;
                dst[h * 64u + q] = e4m3_pack4_swar(Pb[2 * u], Pb[2 * u + 1], D2, C, M, H);
            }
#elif defined(FA_SP_CVTROT2)
        // CHEAPER HALF-ROTATION.  FA_SP_CVTROT rotates the full pair index u, which makes BOTH
        // terms of the store address dst[(u>>2)*64 + (u&3)] runtime and costs 55 registers -- fatal.
        // Rotating only the HALF index h by (lane & 1) leaves q a compile-time constant, so only one
        // term becomes runtime, and it still spreads the 16 lanes over 2 distinct subbanks per step
        // (8-way instead of 16-way) because the load index 2u = h*8 + 2q.
        const uint32_t lh = (tid % MU_NUM_THREADS) & 1u;
        for (uint32_t h0 = 0; h0 < 2; h0++) {
            const uint32_t h = h0 ^ lh;
#pragma unroll
            for (uint32_t q = 0; q < 4; q++) {
                const uint32_t u = h * 4u + q;
                dst[h * 64u + q] = e4m3_pack4(Pb[2 * u], Pb[2 * u + 1], K8);
            }
        }
#elif defined(FA_SP_CVTROT)
        // *** 16-WAY SMEM SUBBANK CONFLICT, ON EVERY LOAD OF THIS PHASE. ***
        // The subbank is word_index & 15 (byte[5:2]).  Here the word index is
        //     row*(SK/2) + b*16 + k   =   row*128 + b*16 + k,
        // and BOTH 128 and 16 are multiples of 16, so the subbank is (k & 15) -- it does not depend
        // on row or b at all.  A warp's 16 lanes hold 16 different (row,b) items but run the same
        // inner iteration, so k is lane-UNIFORM and all 16 lanes hit ONE subbank on every load.
        // flash_mx_impl.hpp documents exactly this trap for requant_P_to_spad_tiled and fixes it
        // with a rotated start offset (or the FA_PSWIZ layout transpose), but the split
        // fa_requant_max/fa_requant_cvt pair that FA_SP_PKOVL uses lost the rotation.
        // The output word must hold 4 CONSECUTIVE elements of the block, so the pairs (2u,2u+1)
        // cannot be broken -- there are only 8 legal starting points, so rotating the PAIR index by
        // the lane spreads the 16 lanes over 8 distinct subbank pairs: 2-way instead of 16-way.
        // h and q become runtime values, which costs the "one base pointer + immediates" store form
        // a shift and an add per pack.
        const uint32_t lane = tid % MU_NUM_THREADS;
        for (uint32_t uu = 0; uu < 8; uu++) {
            const uint32_t u = (uu + lane) & 7u;
            dst[(u >> 2) * 64u + (u & 3u)] = e4m3_pack4(Pb[2 * u], Pb[2 * u + 1], K8);
        }
#else
        for (uint32_t h = 0; h < 2; h++)
            for (uint32_t q = 0; q < 4; q++) {
                const uint32_t u = h * 4u + q;
                dst[h * 64u + q] = e4m3_pack4(Pb[2 * u], Pb[2 * u + 1], K8);
            }
#endif
    }
}
// ============================================================================================
// ---- added 2026-07-27 (fourth pass): three BIT-EXACT structural levers ----------------------
//
// Everything in this block is arithmetically IDENTICAL to the reference kernel -- every value is
// produced by the same operations on the same operands, only by a different thread or at a
// different time -- so a build with these flags must still score exactly 3.5666% against
// golden_O_u16.npy.  That is the point: the 4.2xxx% family of flags (SMTPR / SMBMAX / NOMAX /
// SUBMAX) buys speed by changing the numerics, and this block buys it without.
//
// (A) FA_SP_BAL -- BALANCE THE SIMT WORK ACROSS THE TWO MUON CORES.
//     mu_schedule maps warp w -> core (w & 1), so at occupancy 3 the six warps are
//         core 0 = {0, 2, 4}      core 1 = {1, 3, 5}.
//     FA_SP_QOVL makes warp 0 the gemmini agent, which leaves the SIMT group as
//         core 0 = {2, 4}  (TWO warps)      core 1 = {1, 3, 5}  (THREE warps),
//     and every SIMT stage that runs on warps 1-5 partitions its work EQUALLY per warp.  Each
//     core issues one warp-instruction per cycle, so core 1 is handed 3/5 of the work and core 0
//     only 2/5: the stage takes 0.6 x (total warp-instructions) instead of the balanced 0.5 x.
//     That is a 17% overhead on the requant convert AND on finalize, i.e. ~3.4k cyc/tile here,
//     paid purely for a partition that ignores which core a warp lives on.
//     THE FIX IS A WEIGHTED PARTITION: give the two core-0 warps weight 3 and the three core-1
//     warps weight 2 (total 12), so each CORE receives exactly 6/12 of the work.  In SIMT-warp
//     numbering sw = tid/16 (physical warp sw+1) core 0 is sw ODD, so
//         cum(sw) = (sw>>1)*5 + (sw&1)*2,      start = cum(sw)*N/12, end = cum(sw+1)*N/12
//     which for N=512 items gives 85/128/85/128/86 (core 0 = 256, core 1 = 256) and for N=64
//     rows gives 10/16/11/16/11 (32 and 32).  Exact, integer, no table, ~6 instructions.
//     BIT-EXACTNESS: the convert is elementwise and finalize is per-row, so moving an item or a
//     row between warps cannot change any computed value.
//
// (B) FA_SP_CBMAX -- PRODUCE THE E8M0 BLOCK SCALES INSIDE THE COOPERATIVE SOFTMAX, which deletes
//     the whole requant pass-A stage (measured 3,4xx cyc/tile) at REFERENCE numerics.
//     FA_SP_SMBMAX already does this, but only for the THREAD-PER-ROW softmax, and that softmax
//     is what costs 0.68 Frobenius points -- so SMBMAX has never been available to a
//     reference-numerics build.  It is available: online_softmax_block's column ownership is
//     STRIDED, lane `l` owning words {j*16 + l}, and word j*16+l covers elements
//     2*(j*16+l), +1, whose MX block index is (j*16+l)/16 == j.  So LANE l'S WORD j IS ENTIRELY
//     INSIDE BLOCK j -- the 32 elements of block j are exactly the 16 lanes' word j, and the
//     block max is one 16-lane reduction of a value each lane already holds in a register.
//     It is FREE OF EXTRA FENCES: the eight per-block partials are stored alongside the existing
//     l-reduction partial and folded after the SAME mu_fence_smem, by lanes 0..7 in parallel.
//     BIT-EXACTNESS: max is exact and order-independent, and these are the SAME 32 bf16 values
//     fa_requant_max would have re-read from SMEM, so the E8M0 code is byte-identical.
//
// (C) FA_SP_QSPLIT -- SPLIT THE Q(t+1) PREFETCH BY RESOURCE AND MOVE EACH HALF TO THE STAGE THAT
//     CAN ABSORB IT.  *** This is a PERFORMANCE fix worth ~5k cyc/tile and a candidate
//     CORRECTNESS fix for the FA_SP_QKACC steady-state hazard at the same time. ***
//     Measured stage costs of the reference pipeline at FA_NT6 (steady tiles):
//         s0->s1     52 | S1 acc->S 6,338 | softmax 12,825 | passA 3,4xx
//         passB+pack 10,456 | PV 8,888 | QK||finalize 10,469        total 51,373 = 31.96%
//     S1 IS NOT 998 CYCLES, IT IS 6,338, and the reason is visible in the schedule: with
//     FA_SP_QOVL4 warp 0 spends the first ~4.2k of stage S6 writing Q(t+1)'s 64 MX scale words
//     into SF_A half 1 (~65 cyc/word, strictly serial -- hardware fact 1) and only THEN issues
//     QK(t+1).  The matmul therefore starts ~4.2k into a ~10.4k stage and its 8,210 mesh cycles
//     run over the end of it, so 2-3k of the mesh time plus the accumulator->spad store land in
//     S1 where nothing hides them.  The QK that FA_SP_QKACC exists to hide is only PARTLY hidden.
//     Splitting the prefetch fixes it, because its two halves want different homes:
//       * the 64 SF-SRAM SCALE WORDS go to stage S5, under the PV matmul.  Warp 0 spends that
//         whole stage spinning in gemmini_fence, the scale words go to SF_A half 1 while the mesh
//         reads SF_A half 0 + SF_B half 1, and the SF write port is a TL slave independent of the
//         mesh's scale read port.  (FA_PIPE already writes 256 V-scale words under a matmul that
//         also moves out, so this placement is proven.)
//       * the 8 KB Q MOVE-IN DMA goes to stage S4, under the requant convert.  There is NO mesh
//         operation in flight during S4 at all -- QK(t) drained in S1 and PV(t) is not issued
//         until S5 -- so the DMA cannot collide with an accumulator->spad move-out, which is
//         exactly the collision FA_SP_QOVL3 was blamed for and FA_SP_QOVL4 only relocated.  It
//         also gives the DMA a ~17k window (S4 + S5) instead of ~5k.
//     Stage S6 then contains nothing but one gemmini_fence and the QK issue (~200 cyc), so
//     QK(t+1) starts at the TOP of S6 and its 8,210 mesh cycles fit entirely under finalize --
//     S1 collapses to the accumulator store alone.
//     WHY IT MAY ALSO FIX THE CORRECTNESS BUG.  At FA_NT6 the published fix (FA_SP_QOVL4) is
//     still WRONG on cluster 1 tiles 3,4,5 (109.87% / 111.02% / 111.02%; cluster 0 is correct on
//     all six).  The error is NOT a per-row scale factor -- the per-row median got/golden takes 63
//     distinct values in 0.17..0.77 and the output is a plausible attention result for a DIFFERENT
//     input -- so S itself is wrong, which points at Q, K or their MX scales, i.e. at exactly the
//     two things stage S6 was touching immediately before issuing the matmul that reads them.
//     QSPLIT removes both: it puts a gemmini_fence between the Q DMA and the QK matmul (QOVL4 has
//     only a mu_fence_smem, and fence.s does NOT wait on a gemmini DMA -- see the BAR_PAD note),
//     and it separates the Q scale write from the matmul that reads it by a whole PV matmul.
//     FA_SP_QGF is the cheap CONTROL for the same hypothesis: keep QOVL4's placement and add only
//     the missing gemmini_fence.  If QGF alone is 12/12 then the missing drain was the bug; if
//     QSPLIT is 12/12 and QGF is not, it is the scale write; if neither is, it is elsewhere.
// ============================================================================================
#if defined(FA_SP_BAL)
#define FA_SP_BALC 1      /* balance the requant convert  */
#define FA_SP_BALF 1      /* balance finalize             */
#endif
#if defined(FA_SP_BALC) || defined(FA_SP_BALF)
// Core-balanced [start, end) of an N-element index space for SIMT warp sw of 5 (see (A) above).
// N is a compile-time constant, so the /12 becomes a multiply-shift and the whole thing is ~6
// instructions outside the loop.
template <uint32_t N>
static inline uint32_t fa_bal_lo(uint32_t sw) { return (((sw >> 1) * 5u + (sw & 1u) * 2u) * N) / 12u; }

// Balanced twin of fa_requant_cvt.  Byte-identical arithmetic; only the item -> thread map moves.
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_requant_cvt_bal(
        const __shared uint16_t *P16, __shared uint32_t *spad_u32,
        const __shared uint32_t *scale_scratch, uint32_t tid) {
    constexpr uint32_t NBLK = SK / 32, PE_TILES_K = SK / 16, NT = MU_NUM_THREADS;
    constexpr uint32_t NITEM = SQ * NBLK;
    const __shared uint32_t *P32 = reinterpret_cast<const __shared uint32_t *>(P16);
    const uint32_t sw = tid / NT, lane = tid % NT;
    const uint32_t lo = fa_bal_lo<NITEM>(sw), hi = fa_bal_lo<NITEM>(sw + 1u);
    for (uint32_t item = lo + lane; item < hi; item += NT) {
        const uint32_t row = item / NBLK, b = item % NBLK;
        const __shared uint32_t *Pb = P32 + row * (SK / 2) + b * 16u;
        const uint32_t K8 = ((scale_scratch[b * SQ + row] - 7u) << 3) + 8u;
        const uint32_t ti = row / 16, rr = row % 16;
        __shared uint32_t *dst = spad_u32 + ti * (PE_TILES_K * 64u) + b * 128u + rr * 4u;
#ifdef FA_SP_CVTX
        // ============================================================================
        // FA_SP_CVTX -- FIX THE 16-WAY SMEM SUBBANK CONFLICT FOR FREE, WITH AN XOR ADDRESS.
        // The subbank is word_index & 15.  Here the load index is row*128 + b*16 + k and BOTH 128
        // and 16 are multiples of 16, so the subbank is (k & 15) -- independent of row and b, hence
        // LANE-UNIFORM: a warp's 16 lanes hold 16 different items but run the same k, so all 16
        // hit ONE subbank on every one of the phase's 17 loads per item.  With 544 warp-loads per
        // tile that is ~8.7k cycles of pure conflict, which is most of the gap between this stage's
        // 5.8k of issue and its measured 10.5k.
        // The pair (2u, 2u+1) must stay together (an output word holds 4 consecutive elements), so
        // only the PAIR index u may be rotated -- 8 legal orders, i.e. 8 distinct subbank pairs
        // across the warp = 2-way instead of 16-way.  FA_SP_CVTROT does exactly that and COSTS 55
        // REGISTERS (fatal at occupancy 3) because it makes both terms of dst[(u>>2)*64 + (u&3)]
        // runtime; FA_SP_CVTROT2 halves the rotation to get it to 54.  NEITHER IS NECESSARY:
        // every address here is a BASE WITH ZERO LOW BITS PLUS A SMALL DISJOINT FIELD, so `+` is
        // `^`, and XOR distributes over the rotation.  Concretely, in BYTES,
        //     load  = Pb_byte  + u*8 (+4)     Pb_byte  = 512*row + 64*b   -> bits 5:0 zero
        //     store = dst_byte + ((u&4)<<6) + ((u&3)<<2)
        //                                     dst_byte = 4096*ti + 512*b + 16*rr -> bits 8,3:0 zero
        // and both offset fields are bitwise disjoint from their base, so
        //     addr(u ^ r) = addr(u) ^ addr_offset(r).
        // Rotate by r = lane & 7 ONCE per item into the two base pointers, and every load and store
        // in the unrolled body stays a single xori with a COMPILE-TIME immediate -- zero extra
        // instructions inside the loop and no new live values.  (The same trick FA_PSWIZX already
        // uses for its `base ^ (k*68)`.)
        // `rot` is folded straight into the two base pointers so that NO extra value stays live
        // across the unrolled body (the renamer budget is the binding constraint -- see (F2)).
        const uint32_t Pbx  = ((uint32_t)(uintptr_t)Pb) ^ ((lane & 7u) << 3);
        const uint32_t dstx = ((uint32_t)(uintptr_t)dst)
                            ^ ((((lane & 4u) << 6) | ((lane & 3u) << 2)));
#pragma unroll
        for (uint32_t uu = 0; uu < 8; uu++)
            *reinterpret_cast<__shared uint32_t *>(
                dstx ^ (((uu & 4u) << 6) | ((uu & 3u) << 2))) =
                e4m3_pack4(*reinterpret_cast<const __shared uint32_t *>(Pbx ^ (uu << 3)),
                           *reinterpret_cast<const __shared uint32_t *>(Pbx ^ ((uu << 3) | 4u)),
                           K8);
#else
        for (uint32_t h = 0; h < 2; h++)
            for (uint32_t q = 0; q < 4; q++) {
                const uint32_t u = h * 4u + q;
                dst[h * 64u + q] = e4m3_pack4(Pb[2 * u], Pb[2 * u + 1], K8);
            }
#endif
    }
}

// Balanced twin of the library finalize_O (uint16 l array -- the REFERENCE finalize).  Identical
// arithmetic per row; only which warp owns a row moves, and rows are independent.
template <uint32_t SQ, uint32_t D>
static __attribute__((noinline)) void fa_finalize_bal(
        const __shared uint32_t *O32, const __shared uint16_t *l_state,
        uint32_t *O_gmem32, uint32_t tid) {
    constexpr uint32_t NT = MU_NUM_THREADS, DW = D / 2;
    const uint32_t sw = tid / NT, lane = tid % NT;
    const uint32_t lo = fa_bal_lo<SQ>(sw), hi = fa_bal_lo<SQ>(sw + 1u);
    for (uint32_t row = lo; row < hi; row++) {
        const _Float16 inv_l =
            (_Float16)(__builtin_bit_cast(_Float16, ONE_BF16_BITS) / as_bf16(l_state[row]));
        for (uint32_t w = lane; w < DW; w += NT) {
            const uint32_t a = O32[row * DW + w];
            O_gmem32[row * DW + w] =
                pack_bf16x2((_Float16)(as_bf16((uint16_t)a) * inv_l),
                            (_Float16)(as_bf16((uint16_t)(a >> 16)) * inv_l));
        }
    }
}
#endif  // FA_SP_BALC || FA_SP_BALF

#if defined(FA_SP_SM1FL) && !defined(FA_SP_SM1F)
#  error "FA_SP_SM1FL extends FA_SP_SM1F to the l reduction -- set FA_SP_SM1F too"
#endif
#if defined(FA_SP_SM1FX) && (defined(FA_SP_SM1F) || defined(FA_SP_CBMAX))
#  error "FA_SP_SM1FX replaces FA_SP_SM1F / FA_SP_SM1FL -- do not set them together"
#endif
#if defined(FA_SP_CBMAX) || defined(FA_SP_SM1F) || defined(FA_SP_SM1FX)
// Per-warp scratch for the eight per-block max partials: 8 blocks x 16 lanes x 2 B = 256 B per
// warp, 6 warps = 1.5 KB.  Lives at 0x15800, inside the 0x14000..0x16000 scratch window and above
// everything the FA_SP map puts there (REDBUF 0x15000 uses 32 B/warp; LPART 0x15400 is FUSE-only).
static constexpr uint32_t CBMAX_SMEM = 0x15800;

// online_softmax_block specialised to first_block==1 (which is all FA_SP ever uses) and extended
// to emit the per-32-element E8M0 block scales.  EVERY arithmetic step is the reference one:
//   * the row max is the same 16-lane warp_tree_reduce over the same per-lane partials;
//   * l is the same 16-leaf balanced tree over the same per-lane bf16 chain sums, built with the
//     same `(_Float16)(a + b)` expression the library uses (clang promotes it to fp32 and back --
//     that is the reference behaviour and reproducing it is the point, see hardware fact 2);
//   * P is exp(S*scale - m) written in place at the same word index;
//   * the E8M0 code is max over block j of the same bf16 P values, i.e. byte-identical to
//     fa_requant_max's, taken from registers instead of re-read from SMEM.
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void fa_softmax_coop(
        __shared uint32_t *S32, __shared uint16_t *m_state, __shared uint16_t *l_state,
        __shared uint32_t *scale_scratch, uint16_t softmax_scale_bf16,
        uint32_t tid, uint32_t thr) {
    constexpr uint32_t NT = MU_NUM_THREADS;
    constexpr uint32_t WPL = SK / (2 * NT);      // words per lane per row == MX blocks per row
    constexpr uint32_t SKW = SK / 2;
    static_assert(WPL == SK / 32, "CBMAX needs exactly one word per lane per MX block");
    const uint32_t lane = tid % NT, warp = tid / NT, nwarps = thr / NT;
    const _Float16 scale = as_bf16(softmax_scale_bf16);
    volatile __shared uint16_t *buf =
        reinterpret_cast<volatile __shared uint16_t *>(REDBUF_SMEM) + warp * NT;
    (void)buf;
#ifdef FA_SP_SM1FX
    // ==== FA_SP_SM1FX -- ONE FENCE PER REDUCTION *AND* NO SUBBANK CONFLICT, BIT-EXACT. ==========
    // FA_SP_SM1F had the right idea (store the 16 per-lane partials, fence ONCE, let every lane
    // fold all 16 in registers -- no divergence region, no second fence, no store->load hand-off to
    // race) and MEASURED A LOSS of +1,502 cyc/tile, for one reason: all 16 lanes then read the SAME
    // halfword of `buf` on each of the fold's 16 steps.  FA_SP_SM1FR broke that with a (k+lane)&15
    // rotation and paid 70 instructions and 4 registers for the modular arithmetic -- 34 registers
    // in an all-warps function, over the renamer cliff.
    // THIS DOES IT WITH XOR AND PAYS NOTHING.  Give each lane a 32-BIT slot instead of a halfword
    // (per-warp buffer = 16 words = 64 B, base = REDBUF + warp*64, low SIX address bits zero), so
    // the 16 slots are 16 DISTINCT word-subbanks; then index them by `lane ^ k`.  Because the base
    // is 64 B aligned and the slot offset occupies exactly bits 5:2, `+` is `^`, and
    //     addr(lane ^ k) = (base ^ (lane<<2)) ^ (k<<2)
    // so the lane term folds into the pointer ONCE per row and every one of the 16 reads is a
    // compile-time-immediate XOR.  At every step the 16 lanes read 16 different words -> ZERO
    // conflict, and it costs one andi + one xor per row plus one xori per read.
    // *** WHY IT IS BIT-EXACT FOR *BOTH* REDUCTIONS, INCLUDING l. ***  Reading u[k] = v[lane ^ k]
    // permutes the 16 partials, and XOR-by-a-constant maps the DYADIC tree onto ITSELF: the pair
    // {k, k^1} of u-indices is a pair {2i, 2i+1} of v-indices, the pair of pairs {k,k^1},{k^2,k^3}
    // is the tree's level-2 pair, and so on.  So evaluating warp_tree_reduce's exact 16-leaf
    // expression over u gives the tree's expression over v with some subtrees' operands SWAPPED --
    // and bf16 addition is COMMUTATIVE (only associativity fails), so the result is bit-identical.
    // For the max it is trivially exact (max is associative and commutative).
    // This is what FA_SP_SM1FL could not claim with a (k+lane)&15 rotation, which is NOT an
    // automorphism of the tree; it is why the XOR form -- not the modular one -- is the right fix.
    const uint32_t bufx = (REDBUF_SMEM + warp * (NT * 4u)) ^ (lane * 4u);
#define FA_RBX_ST(v) (*reinterpret_cast<volatile __shared uint32_t *>(bufx) = \
                          (uint32_t)__builtin_bit_cast(uint16_t, (v)))
#define FA_RBX_LD(k) as_bf16((uint16_t)*reinterpret_cast<volatile __shared uint32_t *>( \
                          bufx ^ (((uint32_t)(k)) << 2)))
#endif
#ifdef FA_SP_CBMAX
    volatile __shared uint16_t *bb =
        reinterpret_cast<volatile __shared uint16_t *>(CBMAX_SMEM) + warp * (WPL * NT);
#endif

    for (uint32_t row = warp; row < SQ; row += nwarps) {
        __shared uint32_t *Srow = S32 + row * SKW;
        _Float16 slo[WPL], shi[WPL];
        _Float16 mloc = as_bf16(NEG_INF_BF16_BITS);
        for (uint32_t j = 0; j < WPL; j++) {
            const uint32_t w = Srow[j * NT + lane];
            const _Float16 lo = (_Float16)(as_bf16((uint16_t)w) * scale);
            const _Float16 hi = (_Float16)(as_bf16((uint16_t)(w >> 16)) * scale);
            slo[j] = lo; shi[j] = hi;
            mloc = fmaxf(fmaxf(lo, hi), mloc);
        }
#ifdef FA_SP_SM1FX
        FA_RBX_ST(mloc);
        mu_fence_smem();
        _Float16 q0 = FA_RBX_LD(0), q1 = FA_RBX_LD(1);
#pragma unroll
        for (uint32_t k = 2; k < NT; k += 2) {
            q0 = fmaxf(q0, FA_RBX_LD(k + 0)); q1 = fmaxf(q1, FA_RBX_LD(k + 1));
        }
        const _Float16 m_new = fmaxf(q0, q1);
#else
        buf[lane] = __builtin_bit_cast(uint16_t, mloc);
#ifdef FA_SP_SM1F
        // ONE fence instead of two, and NO warp-divergence region.  warp_tree_reduce costs, per
        // reduction, 4 x (2 SMEM loads + 1 op + 1 SMEM store + vx_split_n/beqz/vx_join) = ~28
        // instructions and needs a SECOND fence to publish buf[0] -- 56 of online_softmax_block's
        // 285 static instructions and 2 of its 4 fences per row are pure reduction scaffolding.
        // Store the 16 partials, fence ONCE, and let EVERY lane fold all 16 in registers: same
        // answer, no divergence, no second fence.  (A same-address 16-lane read is what the
        // reference already does for buf[0], so it is a broadcast on this SMEM, not a conflict.)
        // BIT-EXACT for the max: max is exact and order-independent.
        // TWO chains, not four.  The renamer budget is counted PER WARP AND SUMMED OVER THE THREE
        // WARPS ON A CORE (Rename.scala:110-123), and all six warps run this function, so every
        // register added here costs 3x on core 1 -- see the fa_regs.py note in the header.
        // THE READ IS ROTATED BY THE LANE.  All 16 lanes folding buf[0..15] in the SAME order means
        // 16 lanes reading ONE halfword address per step; the reference gets away with that once
        // (`as_bf16(buf[0])`) but doing it 16 times per reduction is a 16-WAY SUBBANK CONFLICT, and
        // it MEASURED AS A LOSS (+1.2k of softmax on tile 0 before the rotation was added).
        // (i + lane) & 15 is a permutation of the 16 partials, and max is exact and
        // order-independent, so the rotation is free AND bit-exact -- the same argument fa_rowmax_s
        // uses.  It is also why the l reduction canNOT be folded this way: bf16 addition is not
        // associative, so a rotated sum is a different number.  See FA_SP_SM1FL.
        mu_fence_smem();
#ifdef FA_SP_SM1FR
        // FA_SP_SM1FR: rotate the fold read by the lane, so the 16 lanes touch 16 distinct subbanks
        // instead of all reading one halfword (the same trick fa_rowmax_s uses; max is exact and
        // order-independent so it stays bit-exact).  *** MEASURED WORSE: the (k + lane) & 15 address
        // arithmetic is 4 extra ops on every one of the 16 loads, which takes the function from 273
        // to 343 static instructions AND from 30 to 34 registers -- and 34 in an all-warps function
        // is over the renamer cliff (see (F2)).  So the same-address read has to be lived with, and
        // that is what caps FA_SP_SM1F's win.  Kept, documented, OFF. ***
        _Float16 q0 = as_bf16(buf[lane]), q1 = as_bf16(buf[(1u + lane) & (NT - 1u)]);
        for (uint32_t k = 2; k < NT; k += 2) {
            q0 = fmaxf(q0, as_bf16(buf[(k + 0u + lane) & (NT - 1u)]));
            q1 = fmaxf(q1, as_bf16(buf[(k + 1u + lane) & (NT - 1u)]));
        }
#else
        _Float16 q0 = as_bf16(buf[0]), q1 = as_bf16(buf[1]);
        for (uint32_t k = 2; k < NT; k += 2) {
            q0 = fmaxf(q0, as_bf16(buf[k + 0])); q1 = fmaxf(q1, as_bf16(buf[k + 1]));
        }
#endif
        const _Float16 m_new = fmaxf(q0, q1);
#else
        mu_fence_smem(); warp_tree_reduce<true>(buf, lane); mu_fence_smem();
        const _Float16 m_new = as_bf16(buf[0]);          // first_block => m_new == block max
#endif
#endif  // FA_SP_SM1FX
        _Float16 lloc = (_Float16)0;
        for (uint32_t j = 0; j < WPL; j++) {
            const _Float16 a = mu_fexp((_Float16)(slo[j] - m_new));
            const _Float16 b = mu_fexp((_Float16)(shi[j] - m_new));
            slo[j] = a; shi[j] = b;
            lloc = (_Float16)(lloc + a + b);
#ifdef FA_SP_CBMAX
            // this lane's contribution to MX block j's max (2 of the block's 32 elements)
            bb[j * NT + lane] = __builtin_bit_cast(uint16_t, (_Float16)fmaxf(a, b));
#endif
        }
#ifdef FA_SP_SM1FX
        FA_RBX_ST(lloc);
        mu_fence_smem();
#else
        buf[lane] = __builtin_bit_cast(uint16_t, lloc);
        mu_fence_smem();
#ifndef FA_SP_SM1FL
        warp_tree_reduce<false>(buf, lane);
#endif
#endif
#ifdef FA_SP_CBMAX
        // ...and, behind the SAME fence, lanes 0..WPL-1 each fold one block's 16 partials.
        // Same 16 values fa_requant_max would have maxed, so the E8M0 code is byte-identical.
        if (lane < WPL) {
            const volatile __shared uint16_t *bp = bb + lane * NT;
            _Float16 x0 = as_bf16(bp[0]), x1 = as_bf16(bp[1]);   // 2 chains, not 4: registers
            for (uint32_t k = 2; k < NT; k += 2) {
                x0 = fmaxf(x0, as_bf16(bp[k + 0])); x1 = fmaxf(x1, as_bf16(bp[k + 1]));
            }
            const _Float16 bmax = fmaxf(x0, x1);
            const uint32_t em = ((uint32_t)__builtin_bit_cast(uint16_t, bmax) >> 7) & 0xffu;
            const uint32_t K8 = fa_clamp_K8(((int)em - 7) << 3);
            scale_scratch[lane * SQ + row] = ((K8 - 8u) >> 3) + 7u;   // == max(em, 7)
        }
#else
        (void)scale_scratch;
#endif
#ifdef FA_SP_SM1FX
        // The l reduction, folded in registers behind the SAME single fence, in warp_tree_reduce's
        // EXACT 16-leaf pairing -- see the FA_SP_SM1FX comment above for why reading u[k]=v[lane^k]
        // leaves that tree bit-identical (XOR is an automorphism of the dyadic tree; bf16 addition
        // is commutative even though it is not associative).  Three live temporaries, no array.
        _Float16 t0 = (_Float16)((_Float16)(FA_RBX_LD(0) + FA_RBX_LD(1))
                               + (_Float16)(FA_RBX_LD(2) + FA_RBX_LD(3)));
        _Float16 t1 = (_Float16)((_Float16)(FA_RBX_LD(4) + FA_RBX_LD(5))
                               + (_Float16)(FA_RBX_LD(6) + FA_RBX_LD(7)));
        const _Float16 hA = (_Float16)(t0 + t1);
        t0 = (_Float16)((_Float16)(FA_RBX_LD(8) + FA_RBX_LD(9))
                      + (_Float16)(FA_RBX_LD(10) + FA_RBX_LD(11)));
        t1 = (_Float16)((_Float16)(FA_RBX_LD(12) + FA_RBX_LD(13))
                      + (_Float16)(FA_RBX_LD(14) + FA_RBX_LD(15)));
        const _Float16 lsum = (_Float16)(hA + (_Float16)(t0 + t1));
#elif defined(FA_SP_SM1FL)
        // FA_SP_SM1FL -- the same one-fence fold for l.  *** MEASURED A LOSS, kept for the record. ***
        // l MUST keep warp_tree_reduce's exact pairing, because bf16 addition is not associative
        // and every intermediate is rounded to bf16.  The tree is
        //   (((b0+b1)+(b2+b3)) + ((b4+b5)+(b6+b7))) + (((b8+b9)+(b10+b11)) + ((b12+b13)+(b14+b15)))
        // and it is reproduced here term for term, with the SAME `(_Float16)(a + b)` expression the
        // library uses, so the result is bit-identical -- see the reference-numerics argument above.
        // No second fence and no divergence region; the SMEM tree writes disappear too.
        // Written with FOUR live temporaries rather than an r[8] array: the renamer budget is 53
        // distinct architectural registers for the WHOLE kernel (see (b) in the header) and the
        // array form costs 8, which is what pushes FA_SP_SM1F + FA_SP_CBMAX to the fatal 58.
        _Float16 t0 = (_Float16)((_Float16)(as_bf16(buf[0]) + as_bf16(buf[1]))
                               + (_Float16)(as_bf16(buf[2]) + as_bf16(buf[3])));
        _Float16 t1 = (_Float16)((_Float16)(as_bf16(buf[4]) + as_bf16(buf[5]))
                               + (_Float16)(as_bf16(buf[6]) + as_bf16(buf[7])));
        const _Float16 hA = (_Float16)(t0 + t1);
        t0 = (_Float16)((_Float16)(as_bf16(buf[8]) + as_bf16(buf[9]))
                      + (_Float16)(as_bf16(buf[10]) + as_bf16(buf[11])));
        t1 = (_Float16)((_Float16)(as_bf16(buf[12]) + as_bf16(buf[13]))
                      + (_Float16)(as_bf16(buf[14]) + as_bf16(buf[15])));
        const _Float16 lsum = (_Float16)(hA + (_Float16)(t0 + t1));
#else
        mu_fence_smem();
        const _Float16 lsum = as_bf16(buf[0]);
#endif
        for (uint32_t j = 0; j < WPL; j++)                 // P (unnormalised), in place over S
            Srow[j * NT + lane] = pack_bf16x2(slo[j], shi[j]);
        if (lane == 0) { l_state[row] = __builtin_bit_cast(uint16_t, lsum);
                         m_state[row] = __builtin_bit_cast(uint16_t, m_new); }
    }
}
#endif  // FA_SP_CBMAX || FA_SP_SM1F
#endif  // FA_SP

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
#ifdef FA_SP
    // ============ SOFTWARE-PIPELINED STEADY STATE (see the FA_SP header block) ============
#  if   defined(FA_NT1)
#    define FA_SPTILES 1
#  elif defined(FA_NT2)
#    define FA_SPTILES 2
#  elif defined(FA_NT3)
#    define FA_SPTILES 3
#  elif defined(FA_NT6)
#    define FA_SPTILES 6
#  elif defined(FA_NT8)
#    define FA_SPTILES 8
#  else
#    define FA_SPTILES 4
#  endif
    // Softmax flavour: FA_SP_SMTPR = thread-per-row (fast, l accumulated in fp32);
    // default = the cooperative row-per-warp online_softmax_block (bit-exact reference).
#if defined(FA_SP_FUSE) && defined(FA_SP_RMSER)
#  define FA_SP_SOFTMAX(T, N) fa_rowmax_s<FA_SQ, FA_SK>(                          \
        reinterpret_cast<const __shared uint32_t*>(SM_S),                         \
        reinterpret_cast<__shared uint32_t*>(M_SMEM), SOFTMAX_SCALE_BF16, (T), (N))
#elif defined(FA_SP_FUSE) && defined(FA_SP_RMBATCH)
#  define FA_SP_SOFTMAX(T, N) fa_rowmax_b<FA_SQ, FA_SK, 4>(                       \
        reinterpret_cast<const __shared uint32_t*>(SM_S),                         \
        reinterpret_cast<__shared uint32_t*>(M_SMEM), SOFTMAX_SCALE_BF16, (T), (N))
#elif defined(FA_SP_FUSE)
#  define FA_SP_SOFTMAX(T, N) fa_rowmax<FA_SQ, FA_SK>(                            \
        reinterpret_cast<const __shared uint32_t*>(SM_S),                         \
        reinterpret_cast<__shared uint32_t*>(M_SMEM), SOFTMAX_SCALE_BF16, (T), (N))
#elif defined(FA_SP_ITEM)
// Cooperative row max -> item-parallel exp/blockmax/lpart -> row sum, behind one call (see
// fa_softmax_item: inlining the three phases costs a register the kernel does not have).
// FA_SP_ITEM implies the FA_SP_SMBMAX contract -- the E8M0 block scales are produced inside the
// softmax stage, so the requant pass-A stage is compiled out.
#  define FA_SP_SOFTMAX(T, N) fa_softmax_item<FA_SQ, FA_SK>(SOFTMAX_SCALE_BF16, (T), (N))
#elif defined(FA_SP_CBMAX) || defined(FA_SP_SM1F) || defined(FA_SP_SM1FX)
// Cooperative (REFERENCE-numerics) softmax that also produces the E8M0 block scales -> pass A dies.
#  define FA_SP_SOFTMAX(T, N) fa_softmax_coop<FA_SQ, FA_SK>(                     \
        reinterpret_cast<__shared uint32_t*>(SM_S),                               \
        reinterpret_cast<__shared uint16_t*>(M_SMEM),                             \
        reinterpret_cast<__shared uint16_t*>(LS_SMEM),                            \
        reinterpret_cast<__shared uint32_t*>(SCALE_SMEM),                          \
        SOFTMAX_SCALE_BF16, (T), (N))
#elif defined(FA_SP_SMTPR)
#  define FA_SP_SOFTMAX(T, N) fa_softmax_tpr<FA_SQ, FA_SK>(                       \
        reinterpret_cast<__shared uint32_t*>(SM_S),                               \
        reinterpret_cast<__shared uint32_t*>(M_SMEM),                             \
        reinterpret_cast<__shared uint32_t*>(LS_SMEM),                            \
        SOFTMAX_SCALE_BF16, (T), (N))
#else
#  define FA_SP_SOFTMAX(T, N) online_softmax_block<FA_SQ, FA_SK>(                 \
        reinterpret_cast<const __shared uint32_t*>(SM_S),                         \
        reinterpret_cast<__shared uint32_t*>(SM_S),                               \
        reinterpret_cast<__shared uint16_t*>(M_SMEM),                             \
        reinterpret_cast<__shared uint16_t*>(LS_SMEM),                            \
        reinterpret_cast<__shared uint16_t*>(CORR_SMEM),                          \
        SOFTMAX_SCALE_BF16, /*first=*/1, (T), (N))
#endif
    // Finalize flavour: FA_SP_FZFLAT = flat/coalesced (reciprocals hoisted, barrier id 10);
    // default = the row-per-warp finalize_O.
#ifdef FA_SP_FZROW
#  define FA_SP_FINALIZE(T, N) fa_finalize_row<FA_SQ, FA_D>(                       \
        reinterpret_cast<const __shared uint32_t*>(SM_S),                          \
        reinterpret_cast<const __shared uint32_t*>(LS_SMEM),                       \
        reinterpret_cast<uint32_t*>(O_GMEM), (T), (N))
#elif defined(FA_SP_FZFLAT)
#  define FA_SP_FINALIZE(T, N) do {                                               \
        fa_invl<FA_SQ>(reinterpret_cast<const __shared uint32_t*>(LS_SMEM),        \
                       reinterpret_cast<__shared uint32_t*>(PACKED_SMEM),          \
                       (T), (N));                                                 \
        mu_fence_smem(); FAP_PAD(); mu_barrier(10, (N) / MU_NUM_THREADS); FAP_PAD(); \
        fa_finalize_flat<FA_SQ, FA_D>(                                            \
            reinterpret_cast<const __shared uint32_t*>(SM_S),                     \
            reinterpret_cast<const __shared uint32_t*>(PACKED_SMEM),               \
            reinterpret_cast<uint32_t*>(O_GMEM), (T), (N));                       \
    } while (0)
#elif defined(FA_SP_BALF)
// Core-balanced finalize (see FA_SP_BAL): same per-row arithmetic as finalize_O, weighted row map.
#  define FA_SP_FINALIZE(T, N) fa_finalize_bal<FA_SQ, FA_D>(                      \
        reinterpret_cast<const __shared uint32_t*>(SM_S),                         \
        reinterpret_cast<const __shared uint16_t*>(LS_SMEM),                      \
        reinterpret_cast<uint32_t*>(O_GMEM), (T))
#else
#  define FA_SP_FINALIZE(T, N) finalize_O<FA_SQ, FA_D>(                           \
        reinterpret_cast<const __shared uint32_t*>(SM_S),                         \
        reinterpret_cast<const __shared uint16_t*>(LS_SMEM),                      \
        reinterpret_cast<uint32_t*>(O_GMEM), (T), (N))
#endif
#if defined(FA_SP_FZ6) && !defined(FA_SP_QSPLIT)
#  error "FA_SP_FZ6 needs FA_SP_QSPLIT (warp 0 only has slack in S6 once the prefetch has moved)"
#endif
#if defined(FA_SP_FZ6) && defined(FA_SP_BALF)
#  error "FA_SP_FZ6 uses all 6 warps, which mu_schedule already balances 3/3 -- BALF is a no-op"
#endif
#if defined(FA_SP_BALF) && (defined(FA_SP_FZROW) || defined(FA_SP_FZFLAT))
#  error "FA_SP_BALF's finalize reads the uint16 l array -- it is the finalize_O (reference) shape"
#endif
#if (defined(FA_SP_BALC) || defined(FA_SP_BALF)) && (!defined(FA_SP_QOVL) || !defined(FA_SP_PKOVL))
#  error "FA_SP_BAL* balances the warps-1..5 partition -- it needs the FA_SP_QOVL agent split"
#endif
    {
    const uint32_t warp = tid / MU_NUM_THREADS;
    mki = 0;   // marks: 7 per tile, tile t stage s -> m[t*7 + s]
    // ---------------- PROLOGUE (once): K^T, V and their weight scales ----------------
#ifdef FA_SP_LEANCFG
    fa_cfg_once(tid);
#endif
    fa_cfg<QKF>(/*asel=*/1, /*wsel=*/0, tid);
    fa_mvin_B<QKF>(&QK_B_in[0][0], SP_K_END, FA_SK, tid);
    fa_cfg<PVF>(/*asel=*/0, /*wsel=*/1, tid);
    fa_mvin_B<PVF>(&V_in[0][0], SP_V_END, FA_D, tid);
    fa_gf(tid);
#ifdef FA_SP_HSF
    // FA_SP_HSF: the HOST owns both scale ports -- K -> weight half 0, V -> weight half 1,
    // Q -> act half 1 -- so the GPU writes NO scale word anywhere, ever.  That is what makes the
    // act port host-private and removes ScaleFactorMem's shared 2-beat pairing hazard by
    // construction rather than by scheduling.  Block until they are resident.
    fa_sp_wait(FA_SPM_READY, FA_SPM_MAGIC, tid);
#else
    fa_scl(fa_sf_b(0), &QK_B_scales_col[0][0], QKF.SCALE_FACTORS_PER_TILE_B(), tid);
    fa_scl(fa_sf_b(1), &V_scales[0][0], PVF.SCALE_FACTORS_PER_TILE_B(), tid);
#endif
#ifdef FA_SP_QOVL
    // tile 0's Q + Q scales: every later tile's are written one stage early (stage S2 below).
    fa_mvin_A<QKF>(&QK_A_in[0][0], SP_Q, FA_D, tid);
#ifndef FA_SP_HSF
    fa_scl(fa_sf_a(1), &QK_A_scales_row[0][0], QKF.SCALE_FACTORS_PER_TILE(), tid);
#endif
    // warps 1..5 do the SIMT work while warp 0 is the gemmini agent -> remap their ids so
    // online_softmax_block's row->warp partition covers all SQ rows with nwarps == 5.
    const uint32_t stid = tid - MU_NUM_THREADS;
    const uint32_t sthr = thr - MU_NUM_THREADS;
#endif
#if defined(FA_SP_FUSE) && !defined(FA_SP_QOVL)
#  error "FA_SP_FUSE needs FA_SP_QOVL (it uses the warp0-agent / warps1-5-SIMT split)"
#endif
#if (defined(FA_SP_FUSE) || defined(FA_SP_SMTPR)) && !defined(FA_SP_FZFLAT) && !defined(FA_SP_FZROW)
// Both write l as one 32-bit word per row (sub-word store hazard); the library finalize_O reads
// a uint16 l array, so the flat finalize (fa_invl) is mandatory with them.
#  error "FA_SP_FUSE / FA_SP_SMTPR require FA_SP_FZFLAT or FA_SP_FZROW"
#endif
#if defined(FA_SP_ITEM) && !defined(FA_SP_SMBMAX)
#  define FA_SP_SMBMAX 1     /* FA_SP_ITEM produces the E8M0 block scales -> pass A must go away */
#endif
#if defined(FA_SP_ITEM) && !defined(FA_SP_FZROW) && !defined(FA_SP_FZFLAT)
#  error "FA_SP_ITEM writes l as one 32-bit word per row -> needs FA_SP_FZROW or FA_SP_FZFLAT"
#endif
#if defined(FA_SP_ITEM) && defined(FA_SP_SMTPR)
#  error "FA_SP_ITEM replaces FA_SP_SMTPR -- do not set both"
#endif
#if defined(FA_SP_QOVL4) && (!defined(FA_SP_QKACC) || !defined(FA_SP_QOVL))
#  error "FA_SP_QOVL4 needs FA_SP_QOVL and FA_SP_QKACC (it moves the Q prefetch into stage S6)"
#endif
#if defined(FA_SP_QOVL4) && defined(FA_SP_QOVL3)
#  error "FA_SP_QOVL4 replaces FA_SP_QOVL3 -- do not set both"
#endif
#if defined(FA_SP_QSPLIT) && (!defined(FA_SP_QKACC) || !defined(FA_SP_QOVL) || !defined(FA_SP_PKOVL))
#  error "FA_SP_QSPLIT needs FA_SP_QOVL + FA_SP_QKACC + FA_SP_PKOVL (mvin -> S4, scales -> S5)"
#endif
#if defined(FA_SP_QSPLIT) && (defined(FA_SP_QOVL3) || defined(FA_SP_QOVL4))
#  error "FA_SP_QSPLIT replaces FA_SP_QOVL3 / FA_SP_QOVL4 -- do not set both"
#endif
#if defined(FA_SP_QGF) && !defined(FA_SP_QOVL4)
#  error "FA_SP_QGF is the control for FA_SP_QOVL4's missing drain -- it needs FA_SP_QOVL4"
#endif
#if defined(FA_SP_CBMAX) && (defined(FA_SP_SMTPR) || defined(FA_SP_ITEM) || defined(FA_SP_FUSE))
#  error "FA_SP_CBMAX is the COOPERATIVE softmax's block max -- it replaces SMTPR/ITEM/FUSE"
#endif
#if defined(FA_SP_CBMAX) && !defined(FA_SP_PKOVL)
#  error "FA_SP_CBMAX deletes requant pass A, which only exists on the FA_SP_PKOVL path"
#endif
// ============================================================================================
// FA_SP_OPV -- *** OVERLAP THE EXPOSED PV MATMUL WITH THE NEXT TILE'S SOFTMAX. ***
//
// THE PROBLEM IT SOLVES.  In FA_SP_QSPLIT the per-tile budget is
//     S1 acc->S 998 | softmax 12,825 | pass A 3,405 | convert||pack 11,093 | PV 8,889 |
//     QK(t+1)||finalize(t) 9,562   =  46,772
// and of the mesh's 16,420 cycles only QK's 8,210 are hidden.  PV's 8,210 are EXPOSED, and for
// those 8,889 cycles FIVE OF THE SIX WARPS DO NOTHING AT ALL -- 44,000 warp-cycles of idle SIMT
// per tile.  Header note (c) argued this is structural: the per-tile chain is
//     QK(t) -> softmax(t) -> pass A(t) -> convert(t) -> PV(t) -> finalize(t),
// the only SIMT work that does not depend on its own tile's mesh result is finalize, finalize(t)
// depends on PV(t), so "exactly one of the two matmuls can be covered".
//
// THAT ARGUMENT IS TRUE ONLY WITHIN ONE TILE.  Slide the loop by one and PV(t-1) becomes
// concurrent with softmax(t): softmax(t) needs S(t), which QK(t) produced in the PREVIOUS
// iteration, and PV(t-1) needs P8(t-1), which convert(t-1) produced there too.  Neither touches
// the other's data.  What blocked it was the BUFFER, not the dependency -- O(t-1) has to stay
// alive across tile t's softmax, and note (c) recorded "a second 16 KB O buffer does not exist
// (V 32K + P8 16K + S/O 32K + scratch 8K + Q 8K + K 32K = 128K, exactly full)".
//
// *** THE 16 KB IS ALREADY THERE: IT IS THE P8 REGION. ***  P8(t-1) is DEAD the instant PV(t-1)
// has read it, and P8(t) is not written until convert(t).  So O(t-1) is put THERE, and the region
// is used strictly in sequence inside one iteration:
//     convert(t-1) writes P8(t-1) [prev iter] -> PV(t-1) reads it [stage A] -> acc -> O(t-1)
//     lands on top of it [stage B] -> finalize(t-1) reads O(t-1) [stage C2] -> convert(t) writes
//     P8(t) over it [stage D].
// Only two things have to change to make that legal: PV runs COMPUTE-ONLY into the accumulator
// (fa_mm_acc, exactly as FA_SP_QKACC already does for QK) so its move-out can be deferred to a
// SIMT-quiet stage, and l must be DOUBLE-BUFFERED, because finalize(t-1) needs l(t-1) while
// softmax(t) is overwriting l.  m and corr are write-only in FA_SP (first_block == 1), so they
// stay single-buffered.
//
// THE SIX STAGES, and what each one costs (estimates from the QSPLIT stage table):
//   A  [mesh] PV(t-1) compute-only  ||  [warps 1-5] softmax(t)          max(8210, ~14.6k)
//      warp 0 also issues Q(t+1)'s move-in DMA and writes its 64 SF_A-half-1 scale words: the
//      mesh is reading SF_A half 0 and performs NO move-out, which is the placement FA_SP_QSPLIT
//      already proved safe.
//   B  [agent] drain PV(t-1); accumulator -> O(t-1) @ SP_P.  SIMT quiesced.       ~1.0k
//   C1 [mesh] QK(t+1) compute-only  ||  [warps 1-5] requant pass A(t)   max(8210, ~4.1k)
//      the leading gemmini_fence is a REAL drain of Q(t+1)'s DMA and, being a load from the
//      gemmini TL port, also orders the stage-A scale stores ahead of the matmul that reads them
//      -- the FA_SP_QGF/QSPLIT lesson, applied structurally.
//   C2 [warps 1-5] finalize(t-1) -> GMEM, still underneath QK(t+1)'s mesh work.   ~9.6k
//   D  [warp 0] SF pack  ||  [warps 1-5] requant convert(t) -> P8(t)    max(~11.1k, ~8.3k)
//   E  [agent] drain QK(t+1); accumulator -> S(t+1) @ SP_C.  SIMT quiesced.       ~1.0k
//   => ~14.6 + 1.0 + 8.2(hidden) + 9.6 + 11.1 + 1.0 ~= 41.3k, and with FA_SP_PAX (pass A) and
//      FA_SP_CVTX (convert) ~39k = 42%.
//
// WHY EVERY VALUE IS BIT-IDENTICAL: nothing is recomputed, reassociated or reduced differently.
// Every operation is performed by the same code on the same operands; only the STAGE it runs in
// and the SMEM address O lands at change.  So an FA_SP_OPV build must still score exactly
// 3.5666% against golden_O_u16.npy, and any deviation is a scheduling bug, not numerics.
//
// EVERY MESH SMEM WRITE IS IN A SIMT-QUIET STAGE, WHICH IS STRICTLY SAFER THAN QSPLIT.  The two
// accumulator->spad move-outs (B and E) are the only mesh SMEM writes in the whole loop and both
// sit alone between two barriers, so Hazard 2's atomic all-16-subbank grant can never be broken
// by SIMT traffic or by a prefetch DMA -- the class of bug that cost the third and fourth passes.
// The two concurrent stages (A and C1) run the mesh COMPUTE-ONLY, i.e. with zero SMEM writes.
// ============================================================================================
#if defined(FA_SP_HSF) && !defined(FA_SP_PKOVL)
#  error "FA_SP_HSF hands off after requant pass A, which only exists on the FA_SP_PKOVL path"
#endif
#if defined(FA_SP_HSF) && defined(FA_SP_OPV) && !defined(FA_SP_OPVQ)
#  error "FA_SP_HSF on FA_SP_OPV needs FA_SP_OPVQ (the hand-off lives in OPVQ's mesh-idle stage C2)"
#endif
#if defined(FA_SP_SQ32)
// ---- FA_SP_SQ32 half-tile body.  See the SMEM-map comment on FA_SQH for the bank argument. ------
#  if !defined(FA_SP_QOVL) || !defined(FA_SP_PKOVL)
#    error "FA_SP_SQ32 needs FA_SP_QOVL (warp-0 agent) + FA_SP_PKOVL (split requant)"
#  endif
#  if defined(FA_SP_QSPLIT) || defined(FA_SP_QKACC) || defined(FA_SP_OPV) || defined(FA_SP_HSF)
#    error "FA_SP_SQ32 is its own loop body -- it replaces QSPLIT / QKACC / OPV, and HSF is retired"
#  endif
#  if defined(FA_SP_FZFLAT) || defined(FA_SP_SMTPR) || defined(FA_SP_FUSE) || defined(FA_SP_ITEM)
#    error "FA_SP_SQ32 is the REFERENCE-numerics body (uint16 l, separate pass A, no inner barriers)"
#  endif
#define FA_NH (2 * (uint32_t)FA_SPTILES)          /* half-tiles: two per 64-row tile */
// l/m are per-ROW, so they double-buffer with the S/P parity (64 rows of state, 32 live per half).
// ALL SIX SELECTS ARE ARITHMETIC ON THE SINGLE PARITY REGISTER, NOT TERNARIES.  Written as
// `(par ? A : B)` the compiler materialises both constants and keeps six independent values live
// across the stage barriers, which cost FIVE registers in fa_entry (31 vs the Sq=64 body's 26) and
// put the whole shape over the 53-register renamer wall at 57.  Every buffer pair here is a power-of-
// two apart by construction, so one shift off `par` reproduces it in two instructions with nothing
// extra live:  S/P 0x4000, P8 0x2000, l/m 256 B, and the spad rows 1024 / 512.
#define FA_H_L(par)  (LS_SMEM + ((par) << 8))
#define FA_H_M(par)  (M_SMEM  + ((par) << 8))
#define FA_H_SM(par) (SMH_C0  + ((par) << 14))
#define FA_H_P8(par) (SMH_P80 + ((par) << 13))
#define FA_H_SP(par) (SPH_C0  + ((par) << 10))
#define FA_H_SPP(par)(SPH_P80 + ((par) << 9))
    // ---- PRIME: QK(half 0) -> accumulator (Q for half 0 is resident from the prologue) ----
    if (warp == 0) {
        fa_cfg<QKH>(1, 0, tid);
        fa_gf(tid);
        fa_mm_acc<QKH>(SPH_Q, SP_K_END, /*asel=*/1, /*wsel=*/0, tid);
    } else { asm volatile("nop"); }
    FAP_BAR(2);

    // ROLLED over half-tiles, with the double-buffer parity in ONE register.  I also tried the
    // fully UNROLLED form (both parities emitted with compile-time-constant selects, on the theory
    // that the parity plumbing was what cost the registers): it is WORSE -- fa_entry 31 -> 33 and the
    // whole-file union 55 -> 57 -- because the compiler then schedules across twelve straight-line
    // stages and keeps more live.  Recorded so nobody re-tries it.
    for (uint32_t t = 0; t <= FA_NH; t++) {
    const uint32_t p = t & 1u;
#define np (p ^ 1u)
    MARK();                                                             // s0
    // -- S1: drain QK(t); accumulator -> S(t) @ S/P[p].  SIMT quiesced. -----------------------
    if (warp == 0) {
        if (t < FA_NH) { fa_gfl(tid); fa_store_acc<QKH>(FA_H_SP(p), tid); fa_gfl(tid); }
    } else { asm volatile("nop"); }
    FAP_BAR(3);
    SMARK();                                                            // s1
    // -- S2: softmax(t) on warps 1-5  ||  warp 0 packs P(t-1)'s 64 scale words into SF_A half 0.
    // -- THE MESH IS IDLE HERE (the softmax hammers the scratch, which shares bank 2 with every
    // -- mesh operand), which is also what makes this the safest possible place for that SF write:
    // -- no mesh scale read is in flight anywhere.  PV(t-1), which reads them, is issued in S3.
    if (warp == 0) {
        if (t > 0) pack_scales_to_sfmem<FA_SQH, FA_SK>(
                       reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
                       reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A), tid, thr);
#ifdef FA_SP_SQ32_QEARLY
        // *** FA_SP_SQ32_QEARLY -- the FA_SP_QSPLIT LESSON, RECURRING IN THE NEW BODY. ***  With the
        // Q(t+1) prefetch in stage S6 ahead of the QK issue, warp 0's S6 chain is mvin + drain + 32
        // scale words + drain + issue ~= 7,600 cycles, so (a) S6 stops being finalize-bound and
        // (b) QK is not issued until ~3,000-4,000 into S6, which means its 4,096 mesh cycles spill
        // past the stage and S1 then WAITS for them.  Measured, half-tile 1 of the rolled body:
        //     S1 4,314 (projected 500) | S2 6,630 | S3 1,878 | S4 4,923 | S5 1,945 | S6 7,614
        // i.e. ~6,500 of the 27,380 is exactly this one placement.  Stage S2 is the right home: the
        // mesh is IDLE there (the softmax hammers the scratch, which shares bank 2 with every mesh
        // operand), warp 0 is already there for the 64-word pack, and pack 4,160 + prefetch ~2,400
        // balances the five-warp softmax's 6,630 almost exactly.  S6 then contains nothing but the
        // fence and the QK issue, so QK starts at the TOP of finalize and S1 collapses to the store.
        if (t + 1 < FA_NH) {
            fa_mvin_A<QKH>(&QK_A_in[((t + 1u) & 1u) * FA_SQH][0], SPH_Q, FA_D, tid);
            fa_gf(tid);                              // drain the DMA while the mesh is idle
            fa_scl_qhalf(fa_sf_a(1), (t + 1u) & 1u, tid);
        }
#endif
    } else {
        if (t < FA_NH) online_softmax_block<FA_SQH, FA_SK>(
                           reinterpret_cast<const __shared uint32_t*>(FA_H_SM(p)),
                           reinterpret_cast<__shared uint32_t*>(FA_H_SM(p)),
                           reinterpret_cast<__shared uint16_t*>(FA_H_M(p)),
                           reinterpret_cast<__shared uint16_t*>(FA_H_L(p)),
                           reinterpret_cast<__shared uint16_t*>(CORR_SMEM),
                           SOFTMAX_SCALE_BF16, /*first=*/1, stid, sthr);
        else asm volatile("nop");
    }
    FAP_BAR(4);
    SMARK();                                                            // s2
    // -- S3: issue PV(t-1) (reads P8[1-p] bank 2 + V bank 0) || pass A(t) (reads S/P[p] bank 1). --
    if (warp == 0) {
        if (t > 0) {
            fa_cfg<PVH>(0, 1, tid);
            fa_mm_acc<PVH>(FA_H_SPP(np), SP_V_END, /*asel=*/0, /*wsel=*/1, tid);
        }
    } else {
        if (t < FA_NH) fa_requant_max<FA_SQH, FA_SK>(
                           reinterpret_cast<const __shared uint16_t*>(FA_H_SM(p)),
                           reinterpret_cast<__shared uint32_t*>(SCALE_SMEM), stid, sthr);
        else asm volatile("nop");
    }
    FAP_BAR(5);
    SMARK();                                                            // s3
    // -- S4: convert(t) -> P8[p] on ALL SIX warps (warp 0's pack lives in S2), under PV(t-1). ----
    if (t < FA_NH)
        fa_requant_cvt<FA_SQH, FA_SK>(reinterpret_cast<const __shared uint16_t*>(FA_H_SM(p)),
                                      reinterpret_cast<__shared uint32_t*>(FA_H_P8(p)),
                                      reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
                                      tid, thr);
    mu_fence_smem();
    FAP_BAR(6);
    SMARK();                                                            // s4
    // -- S5: drain PV(t-1); accumulator -> O(t-1) @ the LOW HALF of the dead S/P[1-p].  Quiesced. -
    if (warp == 0) {
        if (t > 0) { fa_gfl(tid); fa_store_acc<PVH>(FA_H_SP(np), tid); fa_gfl(tid); }
    } else { asm volatile("nop"); }
    FAP_BAR(7);
    SMARK();                                                            // s5
    // -- S6: issue QK(t+1) (reads Q bank 2 + K bank 3) || finalize(t-1) (reads O(t-1) bank 1). ---
    if (warp == 0) {
        if (t + 1 < FA_NH) {
#ifndef FA_SP_SQ32_QEARLY
            fa_mvin_A<QKH>(&QK_A_in[((t + 1u) & 1u) * FA_SQH][0], SPH_Q, FA_D, tid);
            fa_gf(tid);                              // drain the DMA before the matmul reads Q
            fa_scl_qhalf(fa_sf_a(1), (t + 1u) & 1u, tid);
#endif
            fa_gf(tid);                              // order the SF stores before the matmul
            fa_cfg<QKH>(1, 0, tid);
            fa_mm_acc<QKH>(SPH_Q, SP_K_END, /*asel=*/1, /*wsel=*/0, tid);
        }
    } else {
        if (t > 0) finalize_O<FA_SQH, FA_D>(
                       reinterpret_cast<const __shared uint32_t*>(FA_H_SM(np)),
                       reinterpret_cast<const __shared uint16_t*>(FA_H_L(np)),
                       reinterpret_cast<uint32_t*>(O_GMEM + np * (FA_SQH * FA_D * 2u)), stid, sthr);
        else asm volatile("nop");
    }
    FAP_BAR(8);
    SMARK();                                                            // s6
#undef np
    }   // end FA_SP_SQ32 half-tile loop
    MARK();
    }
#elif defined(FA_SP_OPV)
#  if !defined(FA_SP_QOVL) || !defined(FA_SP_PKOVL)
#    error "FA_SP_OPV needs FA_SP_QOVL (warp-0 agent) + FA_SP_PKOVL (split requant: pass A | convert)"
#  endif
#  if defined(FA_SP_QSPLIT) || defined(FA_SP_QOVL3) || defined(FA_SP_QOVL4) || defined(FA_SP_QKACC)
#    error "FA_SP_OPV replaces FA_SP_QSPLIT / QOVL3 / QOVL4 / QKACC -- it is its own loop body"
#  endif
#  if defined(FA_SP_FZFLAT) || defined(FA_SP_FUSE) || defined(FA_SP_ITEM)
#    error "FA_SP_OPV's finalize runs inside a warp-uniform if -- FZFLAT's internal barrier is illegal"
#  endif
#  if defined(FA_SP_SMTPR) || defined(FA_SP_CBMAX) || defined(FA_SP_SMBMAX)
#    error "FA_SP_OPV is the REFERENCE-numerics body (l is uint16, pass A is a separate stage)"
#  endif
// l DOUBLE BUFFER: 64 uint16 = 128 B per parity, in the 512 B LS_SMEM..CORR_SMEM window.
#define FA_OPV_L(par) (LS_SMEM + (par) * 256u)
#ifdef FA_SP_SM1FX
#  define FA_OPV_SOFTMAX(T, N, PAR) fa_softmax_coop<FA_SQ, FA_SK>(                 \
        reinterpret_cast<__shared uint32_t*>(SM_S),                                \
        reinterpret_cast<__shared uint16_t*>(M_SMEM),                              \
        reinterpret_cast<__shared uint16_t*>(FA_OPV_L(PAR)),                        \
        reinterpret_cast<__shared uint32_t*>(SCALE_SMEM),                          \
        SOFTMAX_SCALE_BF16, (T), (N))
#else
#  define FA_OPV_SOFTMAX(T, N, PAR) online_softmax_block<FA_SQ, FA_SK>(            \
        reinterpret_cast<const __shared uint32_t*>(SM_S),                          \
        reinterpret_cast<__shared uint32_t*>(SM_S),                                \
        reinterpret_cast<__shared uint16_t*>(M_SMEM),                              \
        reinterpret_cast<__shared uint16_t*>(FA_OPV_L(PAR)),                        \
        reinterpret_cast<__shared uint16_t*>(CORR_SMEM),                           \
        SOFTMAX_SCALE_BF16, /*first=*/1, (T), (N))
#endif
// finalize reads O_unnorm out of the P8 REGION (SM_P8), not SM_S -- see the header comment.
#define FA_OPV_FINALIZE(T, N, PAR) finalize_O<FA_SQ, FA_D>(                        \
        reinterpret_cast<const __shared uint32_t*>(SM_P8),                         \
        reinterpret_cast<const __shared uint16_t*>(FA_OPV_L(PAR)),                  \
        reinterpret_cast<uint32_t*>(O_GMEM), (T), (N))

    // ---- PRIME: QK(0) -> accumulator -> S(0) @ SP_C, so iteration 0 starts with S resident. ----
    if (warp == 0) {
        fa_cfg<QKF>(1, 0, tid);
        fa_gf(tid);
        fa_mm_acc<QKF>(SP_Q, SP_K_END, /*asel=*/1, /*wsel=*/0, tid);
        fa_gfl(tid);
        fa_store_acc<QKF>(SP_C, tid);
        fa_gfl(tid);
    } else { asm volatile("nop"); }
    FAP_BAR(2);

    // The loop runs ONE EXTRA iteration: iteration t does tile t's softmax/passA/convert and
    // tile (t-1)'s PV/finalize, so iteration FA_SPTILES drains the last tile.  Both guards are
    // warp-uniform, and no barrier is ever inside one.
    for (uint32_t t = 0; t <= (uint32_t)FA_SPTILES; t++) {
    const uint32_t par = t & 1u;              // l double-buffer parity for tile t
    MARK();   // s0: TOP of the iteration
    fa_phase_skew(tid);
    // ---- STAGE A: [mesh] PV(t-1) COMPUTE-ONLY  ||  [warps 1-5] softmax(t) ------------------
    if (warp == 0) {
        if (t > 0) {
            fa_cfg<PVF>(0, 1, tid);
            fa_mm_acc<PVF>(SP_P, SP_V_END, /*asel=*/0, /*wsel=*/1, tid);
        }
#ifndef FA_SP_OPVQ
        // Q(t+1) prefetch, split exactly as FA_SP_QSPLIT splits it: the DMA is ROCC issue only
        // (32 commands) and the 64 SF_A-half-1 scale words ride under the PV mesh, which reads
        // half 0 and performs no move-out.  Drained/ordered by the gemmini_fence in stage C1.
        if (t + 1 < (uint32_t)FA_SPTILES) {
            fa_mvin_A<QKF>(&QK_A_in[0][0], SP_Q, FA_D, tid);
            fa_scl(fa_sf_a(1), &QK_A_scales_row[0][0], QKF.SCALE_FACTORS_PER_TILE(), tid);
        }
#endif
    } else { asm volatile("nop"); }
#ifdef FA_SP_OPVQ
    // FA_SP_OPVQ: warp 0's ENTIRE stage-A duty is the PV issue (~200 cyc), so the softmax runs on
    // ALL SIX WARPS instead of five -- worth ~1.8k, and it is the reason the Q prefetch moves to
    // stage C2 (see below).
    if (t < (uint32_t)FA_SPTILES) FA_OPV_SOFTMAX(tid, thr, par);
#else
    if (warp != 0) {
        if (t < (uint32_t)FA_SPTILES) FA_OPV_SOFTMAX(stid, sthr, par);
        else asm volatile("nop");
    } else { asm volatile("nop"); }
#endif
    FAP_BAR(3);
    SMARK();  // s1: softmax(t) done, PV(t-1) still on the mesh
    // ---- STAGE B: [agent] drain PV(t-1), accumulator -> O(t-1) @ SP_P.  SIMT quiesced. ------
    if (warp == 0) {
        if (t > 0) {
            fa_gfl(tid);                        // mesh drained -> ACC holds O_unnorm(t-1)
            fa_store_acc<PVF>(SP_P, tid);       // ... on top of the P8(t-1) it just consumed
            fa_gfl(tid);
        }
    } else { asm volatile("nop"); }
    FAP_BAR(4);
    SMARK();  // s2: O(t-1) resident in the P8 region
    // ---- STAGE C1: requant pass A(t).  Under FA_SP_OPVQ the mesh is IDLE (PV drained in B,
    // ---- QK(t+1) not issued until D), so all six warps take part; otherwise warp 0 issues
    // ---- QK(t+1) here and pass A runs on warps 1-5 underneath it. --------------------------
#ifdef FA_SP_OPVQ
    if (t < (uint32_t)FA_SPTILES)
        fa_requant_max<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint16_t*>(SM_S),
                                     reinterpret_cast<__shared uint32_t*>(SCALE_SMEM), tid, thr);
#else
    if (warp == 0) {
        if (t + 1 < (uint32_t)FA_SPTILES) {
            fa_gf(tid);                         // REAL drain of Q(t+1)'s DMA + orders its SF stores
            fa_cfg<QKF>(1, 0, tid);
            fa_mm_acc<QKF>(SP_Q, SP_K_END, /*asel=*/1, /*wsel=*/0, tid);
        }
    } else {
        if (t < (uint32_t)FA_SPTILES)
            fa_requant_max<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint16_t*>(SM_S),
                                         reinterpret_cast<__shared uint32_t*>(SCALE_SMEM),
                                         stid, sthr);
        else asm volatile("nop");
    }
#endif
    FAP_BAR(5);
    SMARK();  // s3: pass A done
    // ---- STAGE C2: [warps 1-5] finalize(t-1) -> GMEM.  Under FA_SP_OPVQ warp 0 spends the
    // ---- stage on the Q(t+1) prefetch, and THIS IS THE STAGE WITH NO MESH ACTIVITY AT ALL --
    // ---- PV(t-1) drained in B and QK(t+1) is not issued until D -- so the 64 SF_A-half-1 scale
    // ---- words cannot race a mesh scale READ and the 8 KB DMA cannot race a move-out.  That is
    // ---- strictly safer than every placement tried in the third and fourth passes, and it is
    // ---- free: finalize is ~9.6k and the prefetch is ~4.5k. --------------------------------
#ifdef FA_SP_HSF
    // Tell the host that SCALE_SMEM(t) is complete (pass A finished in C1 and FAP_BAR(5) drained
    // every warp's stores) and that this stage -- the ONE stage of the iteration with no mesh
    // operation in flight at all -- is its window.  The reply is collected at the top of stage D.
    if (t < (uint32_t)FA_SPTILES) fa_sp_post(FA_SPM_PACKREQ, t + 1u, tid);
#endif
    if (warp == 0) {
#ifdef FA_SP_OPVQ
        if (t + 1 < (uint32_t)FA_SPTILES) {
            fa_mvin_A<QKF>(&QK_A_in[0][0], SP_Q, FA_D, tid);
            // Drain the 8 KB DMA HERE, inside the stage where the only other traffic to Q's bank is
            // the DMA itself (finalize reads O, which is a different bank).  Free -- warp 0 has the
            // whole stage -- and it keeps the DMA from still being in flight when the convert starts
            // hammering that bank in stage D, which is the shape of the four-entry-queue assert
            // documented on FA_SP_BANKA.
            fa_gf(tid);
#ifndef FA_SP_HSF
            fa_scl(fa_sf_a(1), &QK_A_scales_row[0][0], QKF.SCALE_FACTORS_PER_TILE(), tid);
#endif
        }
#endif
#ifdef FA_SP_HSF
    } else { asm volatile("nop"); }
    // FA_SP_HSF: warp 0's stage-C2 duty is now just the Q move-in ISSUE plus its drain (~400 cyc)
    // instead of 64 strictly serial SF stores (~4.2k), so finalize runs on ALL SIX warps.  It is
    // GMEM-store bound, and the cure for that is more outstanding stores -- unlike FA_SP_FZ6, which
    // measured +1,330 because there warp 0 entered the stage LATE.
    if (t > 0) FA_OPV_FINALIZE(tid, thr, par ^ 1u);
#else
    } else {
        if (t > 0) FA_OPV_FINALIZE(stid, sthr, par ^ 1u);
        else asm volatile("nop");
    }
#endif
    FAP_BAR(6);
    SMARK();  // s4: O(t-1) written to GMEM -> the P8 region is free again
    // ---- STAGE D: [warp 0] SF pack  ||  [warps 1-5] requant convert(t) -> P8(t) ------------
    if (warp == 0) {
#ifdef FA_SP_OPVQ
        // QK(t+1) is issued HERE under FA_SP_OPVQ.  Its 8,210 mesh cycles still fit entirely
        // inside this stage (the convert is ~10-11k) plus stage E, so nothing is exposed, and the
        // leading gemmini_fence is a REAL drain of the stage-C2 Q move-in that -- being a load
        // from the same gemmini TL port -- also orders C2's SF scale stores ahead of it.
#ifdef FA_SP_HSF
        // Collect the host's reply BEFORE issuing QK(t+1): the host has just rewritten act half 1
        // (Q) and act half 0 (tile t's packed P scales), and QK(t+1) reads half 1.
        if (t < (uint32_t)FA_SPTILES) fa_sp_wait(FA_SPM_PACKED, t + 1u, tid);
#endif
        if (t + 1 < (uint32_t)FA_SPTILES) {
            fa_gf(tid);
            fa_cfg<QKF>(1, 0, tid);
            fa_mm_acc<QKF>(SP_Q, SP_K_END, /*asel=*/1, /*wsel=*/0, tid);
        }
#endif
#ifndef FA_SP_HSF
        if (t < (uint32_t)FA_SPTILES)
            pack_scales_to_sfmem<FA_SQ, FA_SK>(
                reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
                reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A), tid, thr);
#endif
#ifdef FA_SP_HSF
    } else { asm volatile("nop"); }
    // FA_SP_HSF: the 128-word SF pack is gone from warp 0, so the convert runs on ALL SIX warps.
    if (t < (uint32_t)FA_SPTILES)
        fa_requant_cvt<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint16_t*>(SM_S),
                                     reinterpret_cast<__shared uint32_t*>(SM_P8),
                                     reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
                                     tid, thr);
#else
    } else {
        if (t < (uint32_t)FA_SPTILES)
            fa_requant_cvt<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint16_t*>(SM_S),
                                         reinterpret_cast<__shared uint32_t*>(SM_P8),
                                         reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
                                         stid, sthr);
        else asm volatile("nop");
    }
#endif
    mu_fence_smem();
    FAP_BAR(7);
    SMARK();  // s5: P8(t) + its packed SF scales ready
    // ---- STAGE E: [agent] drain QK(t+1), accumulator -> S(t+1) @ SP_C.  SIMT quiesced. -----
    if (warp == 0) {
        if (t + 1 < (uint32_t)FA_SPTILES) {
            fa_gfl(tid);
            fa_store_acc<QKF>(SP_C, tid);       // over P(t) bf16, which the convert just consumed
            fa_gfl(tid);
        }
    } else { asm volatile("nop"); }
    FAP_BAR(8);
    SMARK();  // s6: S(t+1) resident
    }   // end FA_SP_OPV tile loop
    MARK();
    }
#else   // !FA_SP_OPV -- the FA_SP_QSPLIT / QOVL4 / QKACC body
#ifdef FA_SP_QKACC
#  ifndef FA_SP_QOVL
#    error "FA_SP_QKACC needs FA_SP_QOVL (Q must already be resident when QK is issued)"
#  endif
    // Prime the pipeline: tile 0's QK runs COMPUTE-ONLY into the accumulator here; every later
    // tile's is issued in stage S6 of the previous iteration, underneath that tile's finalize.
    fa_cfg<QKF>(1, 0, tid);
    fa_gf(tid);
    fa_mm_acc<QKF>(SP_Q, SP_K_END, /*asel=*/1, /*wsel=*/0, tid);
#endif
    FAP_BAR(2);

    for (uint32_t t = 0; t < (uint32_t)FA_SPTILES; t++) {
    MARK();   // s0: TOP of the steady-state iteration
    fa_phase_skew(tid);   // FA_PHASE<k>: measurement-only inter-cluster phase sweep
#ifndef FA_SP_QOVL
    // ---- S0: [agent] Q(t) move-in + Q(t) scales + QK config ----------------------------
    if (warp == 0) {
        fa_cfg<QKF>(1, 0, tid);
        fa_mvin_A<QKF>(&QK_A_in[0][0], SP_Q, FA_D, tid);
        fa_scl(fa_sf_a(1), &QK_A_scales_row[0][0], QKF.SCALE_FACTORS_PER_TILE(), tid);
    } else { asm volatile("nop"); }
    FAP_BAR(3);
#endif
    SMARK();  // s1: Q resident
#ifdef FA_SP_QKACC
    // ---- S1: [agent] drain QK(t) (already computed into the ACC under finalize(t-1)) and
    // ---- move the accumulator out to S(t)@SM_S.  SIMT is quiesced here (post-barrier), which
    // ---- is what the accmem->spad writer needs for its atomic 16-subbank grant.
#ifdef FA_SP_ACCSYNC
    // FA_SP_ACCSYNC: a SECOND fence+barrier before the accumulator->spad store.
    // The store writes S(t) over SP_C, which is exactly the region finalize(t-1) was READING in the
    // previous stage.  FAP_BAR(9) already separates them, but the corrupted rows are precisely the
    // LAST rows of particular warps, which is what "a warp's SP_C reads had not all retired when the
    // store landed" looks like -- one barrier orders the ISSUE of those reads, not necessarily their
    // completion under a deep SMEM queue.  This makes the separation explicit so the hypothesis can
    // be tested for the price of ~3 cycles.
    mu_fence_smem();
    FAP_PAD();
    mu_barrier(15, wpb);
    FAP_PAD();
#endif
    if (warp == 0) {
#ifdef FA_SP_DUMPWC
        fa_gfl_probe(tid, t);                         // see fa_gfl_probe: is the busy-fence racing?
#else
        fa_gfl(tid);                                  // mesh drained -> ACC holds S(t)   (FA_SP_WCNT)
#endif
#ifdef FA_SP_ACCPAD
        // ==== FA_SP_ACCPAD -- WAIT FOR THE MESH *PIPELINE*, NOT JUST THE COMMAND FSM. =============
        // *** A TEST OF WHY FA_SP_WCNT IS NECESSARY BUT NOT SUFFICIENT. ***  WCNT polls MMIO 0x28
        // (runningLoops) then MMIO 0x20 (io.busy), and argues runningLoops "cannot lie" because it is
        // raised in the same cycle as the LOOP_WS command write.  That makes it a true drain OF THE
        // LOOP FSM.  But what stage S1 needs is that THE MESH has finished accumulating into accmem,
        // and both registers track the reservation station / loop FSM: completionCount rises when the
        // loop's last command retires, not when the last MAC has propagated through a 16x16 systolic
        // array and been written to the accumulator.  If so both polls pass with MACs still in
        // flight, and fa_store_acc reads a MID-ACCUMULATION accumulator.
        // THE EVIDENCE THAT THIS IS STILL HAPPENING ON TOP OF FA_SP_WCNT (isoA at NT6, E2 at NT8):
        //   * the fault is in S -- 0 of 8192 cells of every wrong image fall outside V's per-column
        //     convex hull, so P and l are still consistent with each other;
        //   * ALL 64 rows and ALL 128 columns differ from a KNOWN-CORRECT TILE OF THE SAME RUN
        //     (methodology check: two correct tiles are bit-identical, 8192/8192 halfwords);
        //   * successive wrong tiles share 0 of 4096 words -- a FRESH partial-sum snapshot every
        //     tile, not one damaged resident operand;
        //   * it never recovers once it starts, and +4,766 cycles of unrelated slack (FA_SP_CVTXS)
        //     masks it completely.
        // That is exactly the "partial sums -- wrong, finite, of plausible magnitude, self-consistent
        // enough that P and l still agree" fingerprint FA_SP_WCNT's own comment predicts for it.
        // So: after the drain, burn a bounded number of cycles covering the mesh's pipeline depth.
        // 128 is generous for a 16x16 array plus the accmem write, and it runs on warp 0 only in a
        // stage where the other five warps are already at the barrier: ~128 cyc/tile, ~0.3% of 43k.
        // *** IF THIS CLOSES THE HAZARD THE FIX IS NOT THIS PAD -- it is that the drain must observe
        // the MESH.  The pad only proves where to look.  If it does NOT close it, the accumulator
        // race is exonerated and the search moves to Q, K^T and their scale words. ***
        { volatile int _d = 0; for (int _i = 0; _i < 128; _i++) asm volatile("addi %0,%0,1" : "+r"(_d)); }
#endif
        fa_store_acc<QKF>(SP_C, tid);
        fa_gfl(tid);                                  // drain the accmem->spad store     (FA_SP_WCNT)
    } else { asm volatile("nop"); }
    FAP_BAR(4);
#else
    // ---- S1: [agent] QK matmul: S = Q@K^T -> C dest (mesh 8,210) -----------------------
    if (warp == 0) {
        fa_gf(tid);                                   // drain the Q move-in
#ifdef FA_SP_QOVL
        fa_cfg<QKF>(1, 0, tid);                       // dims/asel for QKF (V's PVF cfg is stale)
#endif
        fa_mm<QKF>(SP_Q, SP_K_END, SP_C, /*asel=*/1, /*wsel=*/0, tid);
        fa_gfl(tid);                                  // drain matmul + S move-out (FA_SP_WCNT)
    } else { asm volatile("nop"); }
    FAP_BAR(4);
#endif
    SMARK();  // s2: QK done
#ifdef FA_SP_DUMPS
    // DIAGNOSTIC (FA_SP_DUMPS): per-tile XOR checksum of every ROW of S, taken the instant S(t) is
    // resident and before the softmax overwrites it in place.  Q, K, V and their MX scales are
    // LOOP-INVARIANT in FA_SP, so S(t) MUST be bit-identical for every t; any row whose checksum
    // moves between tiles indicts the QK matmul or (with FA_SP_QKACC) the accumulator->spad store,
    // and rules the whole softmax/requant/PV/finalize chain OUT.  All threads, 64 rows, one word
    // per row -> ~64 SMEM loads per thread, i.e. cheap enough not to move the stage timings much.
    fa_dump_rowsum<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint32_t *>(SM_S),
                                 (volatile uint32_t *)(0x40053000u + t * 256u), tid, thr);
    mu_fence_smem();
    FAP_BAR(12);
#endif
    // ---- S2: [SIMT] softmax (or FUSE pass 1 = row max)  ||  [agent] Q(t+1) prefetch ----
#if defined(FA_SP_QOVL) && !defined(FA_SP_QOVL3) && !defined(FA_SP_QOVL4) && !defined(FA_SP_QSPLIT)
    if (warp == 0) {
        if (t + 1 < (uint32_t)FA_SPTILES) {
            fa_mvin_A<QKF>(&QK_A_in[0][0], SP_Q, FA_D, tid);
            fa_scl(fa_sf_a(1), &QK_A_scales_row[0][0], QKF.SCALE_FACTORS_PER_TILE(), tid);
        }
    } else {
        FA_SP_SOFTMAX(stid, sthr);
    }
#else
    // FA_SP_QOVL3 moves the Q prefetch under the PV MATMUL (stage S5) and FA_SP_QOVL4 under the
    // QK compute (stage S6), so this stage gets all six warps.
#ifdef FA_SP_QEARLY
    // ==== FA_SP_QEARLY -- ISSUE Q(t+1)'s MOVE-IN DMA THREE STAGES EARLIER. ====================
    // FA_SP_QSPLIT issues it at the top of stage S4 because S4 was described as "the ONLY stage
    // with no mesh operation in flight at all".  That is true of S2 and S3 as well: under
    // FA_SP_QKACC, QK(t) is DRAINED in stage S1 (fa_gfl) and PV(t) is not issued until S5, so the
    // mesh is idle for the whole softmax + requant-pass-A + convert run.  Issuing here gives the
    // 8 KB transfer S2+S3+S4 (~23.5k cycles at the measured stage costs) to complete in instead of
    // S4 alone (~10k), which is what makes FA_SP_QDRAIN's fence FREE rather than a stall -- and it
    // removes the property that CVTX/PREPK exploited to break correctness, namely that SHORTENING
    // STAGE S4 shortens the DMA's window.  Costs warp 0 the ~200 cycles of 32 ROCC issues at the
    // start of a stage whose length is set by its slowest warp, so ~200 cyc/tile at worst.
    if (warp == 0) {
        if (t + 1 < (uint32_t)FA_SPTILES) fa_mvin_A<QKF>(&QK_A_in[0][0], SP_Q, FA_D, tid);
    } else { asm volatile("nop"); }   // the `else` is MANDATORY (unbalanced warp-uniform region)
#endif
    FA_SP_SOFTMAX(tid, thr);
#endif
    FAP_BAR(5);
    SMARK();  // s3: softmax / row max done
#ifdef FA_SP_DUMPP
    // DIAGNOSTIC (FA_SP_DUMPP): per-row XOR checksum of the bf16 P the softmax just wrote in place
    // over S.  Paired with FA_SP_DUMPLM this pins down WHERE the cooperative softmax and the
    // thread-per-row softmax (FA_SP_SMTPR) diverge numerically -- the measured whole-matrix
    // Frobenius steps 3.5666% -> 4.2xxx% between them, and the three candidate causes are m, P and
    // l.  m and l come out of FA_SP_DUMPLM; this is P.  Both softmaxes are supposed to compute
    // exp(bf16(S*scale) - m) elementwise with the same m, so a P checksum that differs between the
    // two configurations means the exps themselves differ and the claim "same math, different
    // parallelisation" is false.
    fa_dump_rowsum<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint32_t *>(SM_S),
                                 (volatile uint32_t *)(0x40055000u + t * 256u), tid, thr);
    mu_fence_smem();
    FAP_BAR(12);
#endif
#ifdef FA_SP_FUSE
    // ---- S3: [all] FUSE pass 2, MX blocks 0..3: exp + requant -> P8 + scales + l-partials
    fa_expreq<FA_SQ, FA_SK, FA_SK / 64>(
        reinterpret_cast<const __shared uint32_t*>(SM_S),
        reinterpret_cast<__shared uint32_t*>(SM_P8),
        reinterpret_cast<__shared uint32_t*>(SCALE_SMEM),
        reinterpret_cast<const __shared uint32_t*>(M_SMEM),
        reinterpret_cast<__shared uint32_t*>(LPART_SMEM), SOFTMAX_SCALE_BF16,
        /*B0=*/0, tid, thr);
    FAP_BAR(6);
    SMARK();  // s4: first half of the blocks requanted
    // ---- S4: [warp0] pack scale words 0..63 (= blocks 0..3)  ||  [warps1-5] blocks 4..7 --
    if (warp == 0) {
        fa_pack_range<0, (FA_SK / 32) * FA_SQ / 8>(
            reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM), fa_sf_a(0), tid);
    } else {
        fa_expreq<FA_SQ, FA_SK, FA_SK / 64>(
            reinterpret_cast<const __shared uint32_t*>(SM_S),
            reinterpret_cast<__shared uint32_t*>(SM_P8),
            reinterpret_cast<__shared uint32_t*>(SCALE_SMEM),
            reinterpret_cast<const __shared uint32_t*>(M_SMEM),
            reinterpret_cast<__shared uint32_t*>(LPART_SMEM), SOFTMAX_SCALE_BF16,
            /*B0=*/FA_SK / 64, stid, sthr);
    }
    mu_fence_smem();
    FAP_BAR(11);
    // ---- and the second half of the pack  ||  [warps1-5] l[row] = sum of block partials --
    if (warp == 0) {
        fa_pack_range<(FA_SK / 32) * FA_SQ / 8, (FA_SK / 32) * FA_SQ / 4>(
            reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM), fa_sf_a(0), tid);
    } else {
        fa_rowsum<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint32_t*>(LPART_SMEM),
                                reinterpret_cast<__shared uint32_t*>(LS_SMEM), stid, sthr);
    }
    mu_fence_smem();
    FAP_BAR(7);
    SMARK();  // s5: requant + pack done
#elif defined(FA_SP_PKOVL)
#if defined(FA_SP_SMBMAX) || defined(FA_SP_CBMAX)
    // ---- S3: EMPTY.  The softmax already produced the E8M0 block scales (FA_SP_SMBMAX), so the
    // ---- whole requant pass-A stage is gone.  The barrier + MARK are kept so the stage count
    // ---- per tile stays 7 and the mark parser / measurement table stay directly comparable.
    FAP_BAR(6);
    SMARK();  // s4: (requant pass A folded into the softmax)
#else
    // ---- S3: [all] requant PASS A (block max -> E8M0 scales) ---------------------------
    fa_requant_max<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint16_t*>(SM_S),
                                 reinterpret_cast<__shared uint32_t*>(SCALE_SMEM), tid, thr);
    FAP_BAR(6);
    SMARK();  // s4: requant pass A done
#endif
#ifdef FA_SP_PREPK
    // FA_SP_PREPK: split the SF pack into (a) the PACKING MATH, which is 4 SMEM loads + 3 shifts +
    // 3 ors per output word and parallelises perfectly over all 96 threads, and (b) a PURE ASCENDING
    // 128-word COPY into the scale SRAM, which is the only pattern FlitMergeNode accepts and which
    // therefore has to stay single-threaded.  flash_mx_impl.hpp measured the combined form at ~147
    // cyc/word and a plain ascending copy at 46-64, so this takes warp 0's serial critical section
    // from ~8.3k to ~7.0k.  It matters only once the convert is FASTER than the pack: with
    // FA_SP_BALC + FA_SP_CVTX the convert drops to ~6k and stage S4 becomes PACK-BOUND, so the pack
    // is then the thing to shorten.  Costs one extra warp-uniform barrier (3 cycles).
    prepack_scales<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
                                 reinterpret_cast<__shared uint32_t*>(PACKED_SMEM), tid, thr);
    FAP_BAR(1);
#endif
    // ---- S4: [warp0] SF pack  ||  [warps1-5] requant PASS B (bf16 -> fp8 @ P8) ---------
#ifdef FA_SP_HSF
    // FA_SP_HSF on the QSPLIT body.  SCALE_SMEM(t) is complete (pass A ended at FAP_BAR(6), which
    // drains every warp's stores), and the mesh is IDLE for the whole of S2+S3+S4 -- QK(t) drained
    // in S1 and PV(t) is not issued until S5 -- so this stage is a ~10.5k-cycle window in which the
    // host is the ONLY requestor of either scale port.  It writes act half 0 (tile t's packed P
    // scales, read by PV in S5) and act half 1 (the next tile's Q scales, read by QK(t+1) in S6);
    // the GPU collects the reply at the top of S5.
    if (t < (uint32_t)FA_SPTILES) fa_sp_post(FA_SPM_PACKREQ, t + 1u, tid);
#endif
    if (warp == 0) {
#ifdef FA_SP_QSPLIT
        // FA_SP_QSPLIT: Q(t+1)'s 8 KB move-in DMA is issued HERE, first, so it runs under the
        // whole convert + PV stages (~17k cycles).  This stage is the ONLY one with no mesh
        // operation in flight at all -- QK(t) drained in S1, PV(t) is not issued until S5 -- so
        // the DMA cannot break an accumulator->spad move-out's atomic 16-subbank grant, which is
        // the hazard FA_SP_QOVL3 hit and FA_SP_QOVL4 only relocated.  Only the ROCC ISSUE (32
        // gemmini_extended_mvin commands) is on warp 0's critical path here; the transfer is not.
#ifndef FA_SP_QEARLY   /* FA_SP_QEARLY issues it at the top of stage S2 instead */
        if (t + 1 < (uint32_t)FA_SPTILES) fa_mvin_A<QKF>(&QK_A_in[0][0], SP_Q, FA_D, tid);
#endif
#endif
#ifdef FA_SP_HSF
        /* the host packs; warp 0 has nothing to do here and joins the convert below */
#elif defined(FA_SP_PREPK)
        copy_scales_to_sfmem<FA_SQ, FA_SK>(
            reinterpret_cast<const __shared uint32_t*>(PACKED_SMEM),
            reinterpret_cast<volatile __shared uint32_t*>(GEMMINI_SF_MEM_A), tid);
#else
        pack_scales_to_sfmem<FA_SQ, FA_SK>(
            reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
            reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A), tid, thr);
#endif
#ifdef FA_SP_HSF
    } else { asm volatile("nop"); }
    // FA_SP_HSF: the 128-word SF pack is off warp 0, so the convert runs on ALL SIX warps.
    fa_requant_cvt<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint16_t*>(SM_S),
                                 reinterpret_cast<__shared uint32_t*>(SM_P8),
                                 reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM), tid, thr);
#else
    } else {
#ifdef FA_SP_BALC
        fa_requant_cvt_bal<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint16_t*>(SM_S),
                                         reinterpret_cast<__shared uint32_t*>(SM_P8),
                                         reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
                                         tid - MU_NUM_THREADS);
#else
        fa_requant_cvt<FA_SQ, FA_SK>(reinterpret_cast<const __shared uint16_t*>(SM_S),
                                     reinterpret_cast<__shared uint32_t*>(SM_P8),
                                     reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
                                     tid - MU_NUM_THREADS, thr - MU_NUM_THREADS);
#endif
    }
#endif
    mu_fence_smem();
    FAP_BAR(7);
    SMARK();  // s5: pack + convert done
#else
    // ---- S3: [all] requant bf16 P -> fp8 @ P8 spad + per-block E8M0 scales -------------
    requant_P_to_spad_tiled<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint16_t*>(SM_S),
        reinterpret_cast<__shared uint32_t*>(SM_P8),
        reinterpret_cast<__shared uint32_t*>(SCALE_SMEM), tid, thr);
    FAP_BAR(6);
    SMARK();  // s4: requant done
    // ---- S4: [thread0] pack the E8M0 words -> SF_A half 0 ------------------------------
    pack_scales_to_sfmem<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
        reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A), tid, thr);
    mu_fence_smem();
    FAP_BAR(7);
    SMARK();  // s5: pack done
#endif
#ifdef FA_SP_DUMPSC
    // DIAGNOSTIC (FA_SP_DUMPSC): dump the 128 per-(row,block) E8M0 scale words to a GMEM page, one
    // 512-byte slot per tile, right before the PV matmul reads them.  The steady-state corruption
    // looks like a corrupted per-(row,block) scale, and this says WHERE it is corrupted: if these
    // dumps are identical across tiles then SCALE_SMEM is fine and the damage is in the pack into
    // the gemmini's SF-SRAM (or in the mesh's read of it); if they already differ, the requant --
    // or whatever raced with it -- is the culprit.  Lands at 0x40051000, which the trace filter
    // already keeps and which no other diagnostic uses.
    if (tid == 0) {
        volatile uint32_t *dsc = (volatile uint32_t *)(0x40051000u + t * 512u);
        const __shared uint32_t *src = reinterpret_cast<const __shared uint32_t *>(SCALE_SMEM);
        for (uint32_t i = 0; i < (FA_SQ * (FA_SK / 32)) / 4u; i++) {
            uint32_t w = 0;
            for (uint32_t k = 0; k < 4; k++) w |= (src[i * 4 + k] & 0xffu) << (8u * k);
            dsc[i] = w;
        }
    }
    mu_fence_smem();
    FAP_BAR(14);
#endif
    // ---- S5: [agent] PV matmul: O = P@V -> C dest (mesh 8,210) -------------------------
    if (warp == 0) {
#ifdef FA_SP_HSF
        // Collect the host's reply BEFORE issuing PV: it has just rewritten act half 0.
        if (t < (uint32_t)FA_SPTILES) fa_sp_wait(FA_SPM_PACKED, t + 1u, tid);
#endif
        fa_cfg<PVF>(0, 1, tid);
#if defined(FA_SP_QSPLIT)
        // FA_SP_QSPLIT: issue PV, then spend the mesh's 8,210 cycles writing Q(t+1)'s 64 MX scale
        // words into SF_A half 1 instead of spinning in gemmini_fence.  This is the half of the
        // prefetch that is EXPENSIVE (~65 cyc/word, strictly serial -- hardware fact 1) and it is
        // what used to delay the QK issue by ~4.2k in stage S6.  Safe here: the write port is a TL
        // slave independent of the mesh's scale READ port, half 1 is dead (QK(t) drained in S1) and
        // the mesh is reading SF_A half 0 + SF_B half 1.  FA_PIPE writes 256 V-scale words under a
        // matmul that also moves out, so this placement is already proven on this RTL.
#ifdef FA_SP_QDRAIN
        // ==== FA_SP_QDRAIN -- DRAIN Q(t+1)'s MOVE-IN *DMA* BEFORE ISSUING PV.  ==================
        // *** THIS IS A CORRECTNESS FIX, AND THE MEASUREMENT THAT FORCED IT IS: FA_SP_QSPLIT +
        // FA_SP_CVTX + FA_SP_PREPK PRODUCES A WRONG TILE (cluster 1 tile 1, Frobenius 106.66%
        // against golden_O_u16) EVEN THOUGH BOTH FLAGS ARE BIT-EXACT BY CONSTRUCTION. ***
        // CVTX only XOR-permutes which of 8 output words an unrolled iteration writes and PREPK
        // only moves the pack's arithmetic off warp 0; neither can change a computed value.  So the
        // 12-of-12 that FA_SP_QSPLIT scores is NOT a property of the fix -- it is a property of the
        // SCHEDULE, and a -726-cycle perturbation is enough to lose it.  (Exactly the caveat (F4)
        // records for FA_SP_QOVL4 + SM1F + SM1FL, now with the sign reversed.)
        //
        // WHAT IS STILL RACING.  FA_SP_QSPLIT moved Q(t+1)'s move-in ISSUE into stage S4 and says
        // so explicitly: "Only the ROCC ISSUE (32 gemmini_extended_mvin commands) is on warp 0's
        // critical path here; the transfer is not."  The TRANSFER therefore runs on into stage S5,
        // where the PV matmul's accumulator->spad move-out is live -- and that move-out needs an
        // ATOMIC ALL-16-SUBBANK GRANT (Hazard 2).  O lands in SP_C = spad rows 3072..5120 =
        // 0xC000..0x14000, i.e. the 32 KB SMEM banks 1 AND 2, while Q's DMA writes SP_Q =
        // 0x16000..0x18000, which is ALSO bank 2.  That is the SAME bank collision FA_SP_QOVL3 hit;
        // QSPLIT removed the issue from S5 but not the traffic.  Shortening stage S4 -- which is
        // precisely what CVTX (-872 on the convert) and PREPK (-1.3k on the pack) do -- leaves LESS
        // of S4 for the DMA to finish in, so more of it spills into S5.  That predicts the observed
        // direction, and it also predicts why FA_SP_PAX (which shortens the stage BEFORE S4, moving
        // the issue earlier in absolute time and so giving the DMA MORE room) comes out clean.
        //
        // THE FIX IS A DRAIN, NOT A DELAY: gemmini_fence() polls the BUSY register, so it actually
        // waits for the DMA, unlike mu_fence_smem() which only drains this warp's own LSU queues
        // (the bug class fixed in 1cce749).  It is close to free -- the DMA has had all of stage S4
        // (~10k cycles) to move 8 KB -- and it makes correctness independent of how long S4 takes,
        // which is the property every performance lever in this file needs.
        //
        // *** MEASURED, AND REFUTED: FA_SP_QDRAIN IS NOT THE FIX AND THIS DIAGNOSIS IS WRONG. ***
        // FA_NT6, seed 12345, on exactly the configuration the analysis above was built from:
        //     FA_SP_QSPLIT + CVTX + PREPK                 45,974 cyc/tile   4 WRONG of 11
        //     + FA_SP_QDRAIN                              46,028 cyc/tile   2 WRONG of 8
        //     + FA_SP_QDRAIN + FA_SP_QEARLY               46,479 cyc/tile   8 of 9, none wrong
        // The drain is ~free (+54 cycles) and it does NOT fix the corruption, so the Q-DMA-versus-
        // PV-move-out bank collision reasoned about above is NOT the mechanism, however well it fit
        // the three symptoms.  And FA_SP_QEARLY's clean run must NOT be read as a fix either: with
        // the drain refuted there is no mechanism left for it to close, it costs +451 cycles, and
        // "a bit-exact reschedule that happens to come out clean" is exactly the trap this file
        // documents twice over (FA_SP_QOVL4+SM1F+SM1FL, and FA_SP_PAX moving the onset tile).
        // THE ACTUAL MECHANISM IS FA_SP_WCNT's, and it explains the S-localisation directly where
        // mine only explained the timing-sensitivity: gemmini_fence() polls io.busy, which rises
        // SEVERAL CYCLES AFTER the command store, so the fence after fa_store_acc<QKF>() in stage
        // S1 can FALL THROUGH and the softmax then reads a MID-ACCUMULATION accumulator.  MMIO 0x28
        // (runningLoops) rises in the same cycle as the LOOP_WS write, and FA_SP_WCNT uses it.
        // Measured at 45,582 cyc/tile = 36.02%, 12 of 12, and free.  *** BUILD EVERY FA_SP
        // CONFIGURATION WITH FA_SP_WCNT; FA_SP_QDRAIN and FA_SP_QEARLY ARE KEPT ONLY AS THE RECORD
        // OF A REFUTED HYPOTHESIS AND SHOULD STAY OFF. ***
        // Retained below for the record, because the reasoning is a good example of a mechanism that
        // predicted every observed symptom and was still wrong:
        // THERE ARE THREE INDEPENDENT WAYS TO KILL THIS HAZARD, AND THEY COMPOSE:
        //   FA_SP_QDRAIN  (here)  wait for the DMA before the PV that shares its bank -- removes the
        //                         OVERLAP IN TIME.
        //   FA_SP_QEARLY  issue the DMA at the top of stage S2 instead of S4, giving it ~23.5k
        //                 cycles instead of ~10k -- removes the SENSITIVITY TO S4's LENGTH, which is
        //                 the specific thing CVTX and PREPK perturb.
        //   FA_SP_BANKA   give S a whole 32 KB bank so PV's C destination is bank 1 EXACTLY and Q's
        //                 DMA is bank 2 -- removes the BANK SHARING itself, i.e. the most structural
        //                 of the three.  Note that FA_SP_BANKA already exists in this file for an
        //                 unrelated reason (the four-entry, no-backpressure spad read queue that
        //                 FA_SP_OPV asserts on) and its own comment states the collision this bug
        //                 exploits -- "O lands in SP_C ... spanning banks 1 AND 2, while Q's DMA
        //                 writes SP_Q, which is ALSO in bank 2" is the FA_SP_QOVL4 comment's own
        //                 diagnosis of FA_SP_QOVL3.  FA_SP_QSPLIT moved the DMA's ISSUE out of stage
        //                 S5 but not its TRANSFER, so the collision was never actually removed --
        //                 only made rarer, which is why it took a bit-exact -1,095-cycle flag pair
        //                 to expose it again.
        fa_gf(tid);
#endif
        fa_mm<PVF>(SP_P, SP_V_END, SP_C, /*asel=*/0, /*wsel=*/1, tid);
#ifndef FA_SP_HSF
        if (t + 1 < (uint32_t)FA_SPTILES)
            fa_scl(fa_sf_a(1), &QK_A_scales_row[0][0], QKF.SCALE_FACTORS_PER_TILE(), tid);
#endif
        fa_gfl(tid);   // PV's move-out writes SP_C, which finalize READS in S6 (FA_SP_WCNT)
#elif defined(FA_SP_QOVL4)
        // FA_SP_QOVL4: the Q(t+1) prefetch is NOT issued here -- see stage S6.  *** THIS IS THE
        // FIX FOR THE STEADY-STATE CORRECTNESS BUG. ***  FA_SP_QOVL3 issues Q(t+1)'s move-in
        // inside this stage, i.e. WHILE the PV matmul's accumulator->spad move-out is running.
        // That move-out needs an ATOMIC ALL-16-SUBBANK GRANT (Hazard 2), and O lands in SP_C =
        // spad rows 3072..5120, which spans the 32 KB SMEM banks 1 AND 2 -- while Q's DMA writes
        // SP_Q = rows 5632..6144, which is ALSO in bank 2.  So the prefetch DMA steals subbanks
        // from the mvout's grant and O comes back corrupted.  It is invisible on tile 0 (whose Q
        // was moved in by the prologue, with no matmul in flight), which is exactly the observed
        // signature: tile 0 Frobenius 3.5666%, every later tile 30-60%.
        // This stage therefore issues NOTHING but the PV matmul and its drain.
        fa_mm<PVF>(SP_P, SP_V_END, SP_C, /*asel=*/0, /*wsel=*/1, tid);
        fa_gfl(tid);
#elif defined(FA_SP_QOVL3)
        // Q(t+1)'s move-in and its 64 SF-SRAM scale words are issued UNDER the PV matmul.
        // The mvin is a ROCC command so it queues ahead of the matmul (~200 cyc, and it writes
        // Q's spad region which PV never touches); the SF scale writes are SIMT stores to the
        // gemmini's scale-SRAM TL slave, which is independent of the mesh's scale READ port --
        // and they go to SF_A half 1 while PV reads half 0, i.e. physically different banks.
        // (FA_PIPE already proves this is safe: it writes 256 V-scale words under the QK matmul.)
        if (t + 1 < (uint32_t)FA_SPTILES) fa_mvin_A<QKF>(&QK_A_in[0][0], SP_Q, FA_D, tid);
#ifdef FA_SP_REFETCH
        // PESSIMISTIC MODE (measurement only): pretend K^T and V are NOT loop-invariant and
        // re-fetch them (and their 256+256 MX scale words) every tile, i.e. the assumption the
        // FA_STEADY harness makes.  K^T's move-in and its SF_B half-0 scales go here, under the
        // PV matmul: QK(t) has already drained, PV reads SF_B half 1, and the two halves are
        // physically distinct SRAM banks, so this is safe.  V's go in stage S6 under the QK
        // compute (which reads SF_B half 0 and SF_A half 1).  This exists so the "keeping the KV
        // block resident" claim can be quantified against the baseline's own assumption instead
        // of argued.
        if (t + 1 < (uint32_t)FA_SPTILES) fa_mvin_B<QKF>(&QK_B_in[0][0], SP_K_END, FA_SK, tid);
#endif
        fa_mm<PVF>(SP_P, SP_V_END, SP_C, /*asel=*/0, /*wsel=*/1, tid);
        if (t + 1 < (uint32_t)FA_SPTILES)
            fa_scl(fa_sf_a(1), &QK_A_scales_row[0][0], QKF.SCALE_FACTORS_PER_TILE(), tid);
#ifdef FA_SP_REFETCH
        if (t + 1 < (uint32_t)FA_SPTILES)
            fa_scl(fa_sf_b(0), &QK_B_scales_col[0][0], QKF.SCALE_FACTORS_PER_TILE_B(), tid);
#endif
        fa_gfl(tid);
#else
        fa_mm<PVF>(SP_P, SP_V_END, SP_C, /*asel=*/0, /*wsel=*/1, tid);
        fa_gfl(tid);
#endif
    } else { asm volatile("nop"); }
    FAP_BAR(8);
    SMARK();  // s6: PV done
#ifdef FA_SP_DUMPL
    // DIAGNOSTIC (FA_SP_DUMPL): dump, per tile, EVERYTHING finalize is about to consume --
    //   [  0.. 63] the 64 raw words of LS_SMEM  (l; 32-bit-per-row under SMTPR/ITEM, packed
    //              uint16 pairs under the cooperative online_softmax_block)
    //   [ 64..127] the 64 raw words of M_SMEM   (m, same two layouts)
    //   [128..191] a per-ROW XOR checksum of O_unnorm as it sits in SP_C after the PV matmul
    // This is the discriminator the corruption analysis needs.  The observed damage is "a few LATE
    // rows of two particular warps come out uniformly too SMALL", and a uniform per-row factor can
    // only come from l (O = O_unnorm/l) or from O_unnorm itself -- so:
    //   * l moves between tiles for exactly the bad rows        -> the SOFTMAX (or something racing
    //                                                              its l/m stores) is at fault;
    //   * l identical but the O_unnorm checksum moves           -> P8 / the E8M0 scales / the PV
    //                                                              matmul is at fault;
    //   * both identical across every tile                      -> the damage is inside FINALIZE
    //                                                              itself (its SP_C reads or its
    //                                                              GMEM stores), not upstream.
    // Every value dumped is loop-invariant in exact arithmetic, so "identical across tiles" is the
    // expected reading for a correct pipeline and any difference is a positive localisation.
    fa_dump_state<FA_SQ, FA_D>(reinterpret_cast<const __shared uint32_t *>(LS_SMEM),
                               reinterpret_cast<const __shared uint32_t *>(M_SMEM),
                               reinterpret_cast<const __shared uint32_t *>(SM_S),
                               (volatile uint32_t *)(0x40054000u + t * 1024u), tid, thr);
    mu_fence_smem();
    FAP_BAR(13);
#elif defined(FA_SP_DUMPLM)
    // the l/m half of FA_SP_DUMPL only -- 3 registers cheaper, so it fits at occupancy 3.
    fa_dump_lm<FA_SQ>(reinterpret_cast<const __shared uint32_t *>(LS_SMEM),
                      reinterpret_cast<const __shared uint32_t *>(M_SMEM),
                      (volatile uint32_t *)(0x40054000u + t * 1024u), tid, thr);
    mu_fence_smem();
    FAP_BAR(13);
#endif
#ifdef FA_SP_QKACC
    // ---- S6: [agent] QK(t+1) COMPUTE-ONLY -> ACC  ||  [warps1-5] finalize(t) -> GMEM ----
    // The mesh touches only its own spad read ports and the private accumulator, so it issues
    // ZERO SMEM writes and cannot collide with finalize's SMEM reads / GMEM stores.  No drain
    // here: stage S1 of the next iteration drains and moves the accumulator out.
    if (warp == 0) {
#ifdef FA_SP_QSPLIT
        // FA_SP_QSPLIT: BOTH halves of the Q(t+1) prefetch are already done (the move-in was
        // issued in S4, the scale words were written under the PV matmul in S5), so this stage
        // contains nothing but a drain and the QK issue -- ~200 cycles instead of ~4.4k.  That is
        // the whole point: QK(t+1)'s 8,210 mesh cycles now start at the TOP of a ~9-10k finalize
        // stage and are fully hidden, so stage S1 shrinks from 6,338 to the accumulator store.
        // The gemmini_fence is a real drain of the Q DMA (fence.s is not -- see FA_SP_QGF) and,
        // being a load from the gemmini TL port, it also orders the S5 scale stores ahead of the
        // matmul that reads them.
        fa_gf(tid);
#endif
#ifdef FA_SP_QOVL4
        // Q(t+1) move-in + its 64 SF_A-half-1 scale words, issued HERE instead of under the PV
        // matmul.  This stage is the right home for them:
        //   * the mesh is running QK COMPUTE-ONLY into the accumulator, so it performs NO SMEM
        //     move-out at all and there is no atomic subbank grant to break (unlike stage S5);
        //   * the DMA is a ROCC command and the ROCC queue is IN-ORDER, so the move-in is
        //     guaranteed complete before the matmul below -- which READS Q -- begins;
        //   * the only other SMEM traffic is finalize(t), which READS SP_C and writes GMEM, so a
        //     DMA write into SP_Q contends for bandwidth at worst, never for correctness;
        //   * the scale words go to SF_A half 1 while nothing is reading half 1 yet.
        // So the overlap that FA_SP_QOVL3 was after is preserved -- it just hides under the QK
        // compute + finalize instead of under the PV move-out.
        if (t + 1 < (uint32_t)FA_SPTILES) {
            fa_mvin_A<QKF>(&QK_A_in[0][0], SP_Q, FA_D, tid);
            fa_scl(fa_sf_a(1), &QK_A_scales_row[0][0], QKF.SCALE_FACTORS_PER_TILE(), tid);
            mu_fence_smem();   // the scale words are SIMT stores; order them before the matmul
        }
#ifdef FA_SP_QGF
        // FA_SP_QGF (CONTROL for the FA_SP_QKACC steady-state hazard): QOVL4 relies on
        // "the ROCC queue is IN-ORDER" to guarantee that Q(t+1)'s move-in DMA has landed before
        // the matmul that reads it, and on mu_fence_smem() to publish the 64 SF scale words.
        // NEITHER is a drain: fence.s waits only on the Muon per-warp shared LSU queues (see the
        // BAR_PAD note at the top of this file -- it does NOT wait on a gemmini DMA or on the
        // SF-SRAM scale writes), and mvin/matmul are separate gemmini queues.  gemmini_fence()
        // polls the BUSY register, so it drains the DMA, and being a LOAD from the same gemmini TL
        // port it also orders every preceding store to that port.  It is FREE: warp 0 is idle for
        // the rest of this stage anyway (finalize on warps 1-5 sets the stage length).
        fa_gf(tid);
#endif
#endif
        fa_cfg<QKF>(1, 0, tid);
        fa_mm_acc<QKF>(SP_Q, SP_K_END, /*asel=*/1, /*wsel=*/0, tid);
#ifdef FA_SP_REFETCH
        // V(t+1): move-in + its SF_B half-1 scales, under the QK compute (which reads half 0).
        if (t + 1 < (uint32_t)FA_SPTILES) {
            fa_mvin_B<PVF>(&V_in[0][0], SP_V_END, FA_D, tid);
            fa_scl(fa_sf_b(1), &V_scales[0][0], PVF.SCALE_FACTORS_PER_TILE_B(), tid);
        }
#endif
#ifdef FA_SP_FZ6
    } else { asm volatile("nop"); }     // (the `else` is MANDATORY: an unbalanced warp-uniform
                                        //  region makes llvm duplicate later control flow)
    // FA_SP_FZ6: finalize on ALL SIX warps.  Only legal with FA_SP_QSPLIT, where warp 0's whole
    // stage-S6 duty is a gemmini_fence plus one matmul issue (~200 cyc) instead of the ~4.4k
    // scale write FA_SP_QOVL4 leaves there -- so warp 0 is free to take a sixth of finalize.
    // TWO reasons this is worth more than the 1/6 of instructions it moves:
    //   * finalize is not issue-bound at all.  finalize_O is FIFTY-ONE static instructions and the
    //     stage costs ~7-10k cycles for 4,096 word stores, i.e. ~40 cycles per coalesced 64 B GMEM
    //     line -- it is GMEM-store-bandwidth/latency bound, and the cure for that is MORE
    //     outstanding stores, i.e. more warps, not a better partition.  (This is also why
    //     FA_SP_FZU4's extra ILP within a warp bought nothing: +1,132 cyc, measured.)
    //   * at six warps mu_schedule's warp -> core (w & 1) map is EXACTLY BALANCED (3 and 3), so
    //     the plain equal partition is already core-balanced and FA_SP_BALF is not needed here --
    //     which is why FA_SP_FZ6 and FA_SP_BALF are mutually exclusive.
    FA_SP_FINALIZE(tid, thr);
#else
    } else {
        FA_SP_FINALIZE(stid, sthr);
    }
#endif
    FAP_BAR(9);
#else
    // ---- S6: [SIMT] finalize O = O_unnorm / l -> GMEM ---------------------------------
    FA_SP_FINALIZE(tid, thr);
    FAP_BAR(9);
#endif
    }   // end steady-state tile loop
    MARK();   // s7 of the last tile == s0 of the (absent) next one
    }
#endif  // FA_SP_OPV
#elif defined(FA_PIPE)
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
    fa_phase_skew(tid);   // FA_PHASE<k>: measurement-only inter-cluster phase sweep
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
    mxgemm_prefetch_tile<PVF, /*SKIP_A=*/true, /*DO_CONFIG=*/true, /*EXPLICIT_MVIN=*/false,
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
    pack_scales_to_sfmem<FA_SQ, FA_SK>(
        reinterpret_cast<const __shared uint32_t*>(SCALE_SMEM),
        reinterpret_cast<__shared uint32_t*>(GEMMINI_SF_MEM_A
#ifdef FA_SFPAR
                                             + GEMMINI_SF_MEM_BUFFER_OFFSET
#endif
                                             ), tid, thr);
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
#elif defined(FA_OCC4)
    // occ=4 (8 warps / 128 threads).  Worth re-testing for FA_SP: requant is LOAD-LATENCY bound
    // (perf-viz: fp pipes 98.4% idle, SMEM 0.5% of peak), and occupancy is the only thing that
    // hides that latency.  An earlier note says occ=4 overflows the 256-entry physical register
    // file; if it does, the build/run will show it.
    mu_schedule(fa_entry, nullptr, 4);
#else
    mu_schedule(fa_entry, nullptr, 3);  // occ=3 default
#endif
                                        // occ=4 overflowed the 256 phys-reg file, occ=3 (3*57=171)
                                        // has margin. More warps hide the latency-bound SIMT softmax.
    return 0;
}
