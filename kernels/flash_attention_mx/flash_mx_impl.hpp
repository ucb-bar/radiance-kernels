#ifndef _FLASH_MX_IMPL_H_
#define _FLASH_MX_IMPL_H_
// SIMT softmax + MX-FP8 requantization for the MXFP8 flash-attention kernel.
//
// Consumes S (bf16, [Sq][Sk]) in GMEM (the QK^T mesh output), applies softmax_scale,
// row softmax (max / exp / sum), and requantizes the probabilities P to MX FP8
// (e4m3 elements + E8M0 per-32-col-block scales) in the layout the PV GEMM consumes:
//   P_fp8     [Sq][Sk]      (uint8 e4m3)            -> PV A_in
//   P_scales  [Sk/32][Sq]   (uint8 E8M0, transposed)-> PV A_scales_row
//   l         [Sq]          (fp32 row denom)        -> final O normalization
//
// bf16 math via mu_fexp/_Float16 (== bf16 here). One row per warp (grid-strided);
// the 16 lanes reduce over a row cooperatively in lockstep via SMEM (no in-loop
// fences -- mirrors the working softmax kernel; avoids the muon backend's
// UnifyLoopExits crash on in-loop fences / data-dependent branches).
//
// Column ownership is STRIDED: lane owns cols {lane + i*16 : i in 0..SK/16-1}.
// With 32-element MX blocks and 16 lanes, columns i=2b and i=2b+1 fall in block b,
// so a column's block index is i/2 -- no data-dependent branch needed.
#include <stdint.h>
#include <mu_intrinsics.h>

// ===========================================================================
// MEASURED TABLE (Sq=64 Sk=256 d=128, FULL_ATTN2, VCS RadianceTapeoutSimConfig,
// requant = MARK m[5]->m[6], TOTAL = m[10]; occ=3 (6 warps / 96 thr) unless noted)
//
//   tag  configuration                        requant  softmax   TOTAL   Frobenius
//   z0   baseline (c3156c6 SIMT)               45,072   15,507  154,579   4.5954%  ok
//   z1   FA_RQ_FAST                            10,533   14,867  118,673   4.5954%  ok
//   z2   FA_RQ_FAST FA_PSWIZ                    9,546   19,507  121,943   6.86%   RACE
//   z3   + FA_RQ_SWAR                          10,971   17,846  128,677   7.93%   RACE
//   z4   + FA_SM_FAST                          10,647   19,557  132,807  15.79%   RACE
//   z7   + FA_OCC2 (4 warps)                   11,422   19,144  119,988  10.58%   RACE
//   z5   + FA_OCC2 FA_RQ_CACHE                 12,812   19,527  126,128  11.90%   RACE
//   z6   + FA_OCC1 FA_RQ_CACHE (2 warps)       21,850   27,734  127,992   6.75%   RACE
//   s1   FA_RQ_FAST FA_SYNCFIX                 18,183*  16,016  126,473   inf     BROKEN
//   s4   FA_PK_LANES FA_SYNCFIX                   --       --      --   FlitMergeNode $finish
//   k1   FA_RQ_FAST FA_RQ_SWAR                 11,984   15,731  125,130   4.5954%  ok
//   k2   + FA_SM_FAST                          12,792   14,370  122,630   9.69%   BROKEN
//   (* with FA_SYNCFIX the barrier wait lands inside the m[5]->m[6] window)
//
// k1 is the surprise: FA_RQ_SWAR removes 14% of the static instructions (357 vs
// 416 for the whole function) and is still 1,451 cycles SLOWER.  The scalar
// e4m3_pack4 is four almost-independent 7-op chains; the SWAR form is ONE
// 12-op serial dependency chain per output word plus 4 loop-invariant constant
// registers.  On this in-order machine the ILP is worth more than the op count
// -- the same lesson as the occupancy sweep below.  So SWAR stays off.
// k2-vs-k1 isolates FA_SM_FAST at -1,361 cycles of softmax -- but k2's output
// is WRONG (9.69% vs the required 4.5954%).  The butterfly has every one of the
// 16 lanes storing AND loading on every step, whereas warp_tree_reduce only
// ever has the (shrinking) set of even lanes storing while reading a slot the
// previous step's now-inactive lane wrote.  The unfenced store->load that the
// tree gets away with does not survive all-lanes traffic.  It would need a
// fence per step, which costs more than the 1,361 it saves.  Left off.
//
// ILP vs TLP (requant, with the layout/convert variants held fixed):
//   occ=3, 6 warps, 32 SMEM loads/item   requant 10,533   <-- best
//   occ=2, 4 warps, 32 loads/item        requant 11,422
//   occ=2, 4 warps, 16 loads/item (cache)requant 12,812
//   occ=1, 2 warps, 16 loads/item (cache)requant 21,850
// Issue count per core is INVARIANT in occupancy here (occ warps x 16/occ items
// x body), so occupancy buys nothing but latency hiding -- and it is the only
// thing hiding the SMEM latency, because each thread's loads are a short chain.
// Halving the loads with a register cache does not compensate for halving the
// warps: TLP wins, keep occ=3.
//
// The "RACE" rows are all FA_PSWIZ rows: the transposed layout changes WHICH
// softmax warp wrote the P rows a given requant thread reads, which turns the
// (pre-existing, unsynchronised) softmax->requant hand-off into a visible data
// race.  z1 is bit-identical to the baseline because it keeps the baseline's
// row-major P layout and therefore the baseline's (accidentally safe) pairing.
//
// MEASURED-BEST CONFIGURATION (2026-07-25).  These are ON by default; define
// FA_LEGACY_SIMT to get the pre-optimization code paths back.  See the block
// comment above requant_P_to_spad_tiled for the full measurement table.
//   FA_RQ_FAST  : branchless inline-asm requant       requant 45,072 -> 10,533
//                 (total 154,579 -> 118,673; output BIT-IDENTICAL to the baseline)
//   FA_RQ_SWAR  : 2-elements-per-register convert (25 instr/word vs 34)
//
// MEASURED LOSSES / BREAKAGE -- do NOT re-enable without re-reading the notes:
//   FA_PSWIZ, FA_PSWIZX : transposed P scratch layout.  Requant 10,533 -> 9,546,
//     but online_softmax_block's P store then hits 16 lanes x 16 distinct 64B
//     lines instead of one, costing it +4,600.  NET LOSS (+3,270 total).
//   FA_RQ_CACHE : 16-word register cache (32 SMEM loads/item -> 16).  Only pays
//     off if there is nothing else to hide the latency; at occ=1 (its only
//     spill-free-and-useful home) requant is 21,850.  NET LOSS.
//   FA_SYNCFIX  : adds the (genuinely missing) softmax->requant and
//     requant->pack cross-warp barriers.  It makes the kernel BOTH slower
//     (+7,800) and WRONG (Frobenius inf).  Left in as an #ifdef because the
//     race it targets is real -- see the FA_SYNC comment -- but something about
//     an extra mu_barrier in this phase corrupts state; needs its own
//     investigation.  Without it we are exactly as (in)correct as the baseline:
//     FA_RQ_FAST reproduces the baseline output bit-for-bit.
//   FA_PK_LANES : lane-parallel SF-SRAM scale write.  $finish -- FlitMergeNode.
//     scala:62 `assert(in.a.bits.address === mergedReq.address + byteOffset)`.
//     The A-scale merge node does not accept the block-partitioned lane pattern
//     that mxgemm_core.hpp's load_scale_factors_lanes uses on the B path.
#ifndef FA_LEGACY_SIMT
#  ifndef FA_RQ_FAST
#    define FA_RQ_FAST 1
#  endif
#endif

// Local copy of the caller's BAR_PAD: the cluster barrier RELEASE is a single-cycle
// unbuffered Valid pulse (Synchronizer.sv:87), so a few RETIRING ALU ops on either
// side of the barrier restore issue slack.  `_p` must NOT be volatile -- volatile
// forces a stack slot and the stack is in DRAM, which turned this into ~9-12 DRAM
// round-trips (that bug cost 10k cyc/barrier and is why barriers looked expensive).
#define FA_BARPAD() do { int _p = 0;                     \
    asm volatile("addi %0,%0,1" : "+r"(_p));             \
    asm volatile("addi %0,%0,1" : "+r"(_p));             \
    asm volatile("addi %0,%0,1" : "+r"(_p));             \
    asm volatile("addi %0,%0,1" : "+r"(_p)); } while (0)

// CROSS-WARP SYNC FIX (2026-07-25).  FULL_ATTN2 separates
//   online_softmax_block (row r -> warp r%nwarps)  ->  requant (item -> ALL threads)
//   requant (scale_scratch <- all threads)         ->  pack_scales (warp 0 / thread 0)
// with only a per-warp `mu_fence_smem()` in between -- there is NO cross-warp
// barrier, so a warp that finishes softmax first races into requant and reads P
// rows another warp has not written yet (and thread 0 packs scales other warps
// have not produced).  The hazard is latent in the baseline because requant was
// 45k cycles of slack; making requant 4.7x faster exposes it.  A barrier is ~3
// cycles now that BAR_PAD no longer touches DRAM, so this is essentially free.
#define FA_SYNC(id, thr) do {                            \
    mu_fence_smem(); FA_BARPAD();                        \
    mu_barrier((id), (thr) / MU_NUM_THREADS);            \
    FA_BARPAD(); } while (0)

// bf16 (uint16 code) -> e4m3 (uint8 code), round-to-nearest-even, MX FP8 saturation.
// Validated against fp8_matmul_model.tensor_to_custom_fp_codes (bf16 inputs, ~100%).
// bf16 -> e4m3 code. RNE=true rounds to nearest-even; RNE=false TRUNCATES toward zero
// (matches the golden mx_quantize_cols, which uses float_quantize_trunc -- truncating to
// the 3-bit e4m3 grid is identical from bf16 or fp32, so this reproduces the model).
template <bool RNE = true>
static inline uint8_t bf16_to_e4m3(uint16_t b) {
    uint32_t sign = (b >> 15) & 1;
    uint32_t exp = (b >> 7) & 0xff;
    uint32_t man7 = b & 0x7f;
    if (exp == 0) return 0;                        // zero/subnormal bf16 -> 0
    if (exp == 0xff) return (uint8_t)((sign << 7) | 0x7e);  // inf/nan -> saturate (448)
    int E = (int)exp - 127;
    const int emin = -6, emax = 8;
    if (E < emin) return 0;                        // underflow -> 0
    int Eu = (E <= emax) ? E : emax;
    uint32_t m3 = man7 >> 4;
    if constexpr (RNE) {
        uint32_t rb = (man7 >> 3) & 1;
        uint32_t sticky = (man7 & 0x7) != 0;
        uint32_t lsb = m3 & 1;
        if (rb && (sticky || lsb)) m3++;
        if (m3 >= 8) { m3 = 0; Eu++; }
    }
    if (Eu > emax || E > emax) { Eu = emax; m3 = 6; }
    if (Eu == emax && m3 > 6) m3 = 6;
    return (uint8_t)((sign << 7) | ((uint32_t)(Eu + 7) << 3) | m3);
}

static inline int bf16_floor_log2(uint16_t b) {  // unbiased exponent of normalized bf16
    return (int)((b >> 7) & 0xff) - 127;
}

// FUSED requant: (v * 2^-se) -> e4m3 code, in one pass. Eliminates the separate
// bf16_scale_pow2 (which only tweaked the exponent, then bf16_to_e4m3 re-extracted it).
// Truncating (RNE=false), matches bf16_scale_pow2(v,-se) then bf16_to_e4m3<false>:
//  - v exp==0 -> 0;  scaled-exp <= 0  (E < emin) -> 0;  scaled-exp overflow -> saturate 0x7e.
// P is always finite & >=0 (softmax probs), so inf/nan and sign paths are unused but handled.
static inline uint8_t bf16_to_e4m3_scaled(uint16_t b, int se) {
    // Fully BRANCHLESS, single return (no early-return control flow -> no warp divergence,
    // which was ~72% of the softmax cost). All conditionals are arithmetic selects/masks.
    const int emax = 8, emin = -6;
    int exp = (int)((b >> 7) & 0xff);
    int m3  = (int)((b >> 4) & 0x7);
    int E   = exp - 127 - se;                         // exponent after *2^-se
    int over = -(int)(E > emax);                      // all-ones if overflow else 0
    E  = (E  & ~over) | (emax & over);                // E  = over ? emax : E
    m3 = (m3 & ~over) | (6    & over);                // m3 = over ? 6    : m3
    int clampm = -(int)((E == emax) & (m3 > 6));
    m3 = (m3 & ~clampm) | (6 & clampm);               // e4m3 max mantissa at emax is 6 (=448)
    int code = ((b >> 8) & 0x80) | ((E + 7) << 3) | m3;
    int keep = -(int)!((exp == 0) | (E < emin));      // 0 if zero/underflow else all-ones
    return (uint8_t)(code & keep);
}

// ===========================================================================
// FAST branchless bf16 -> e4m3(scaled by 2^-se) code.  (2026-07-25)
//
// WHY: llvm-objdump of the "branchless" bf16_to_e4m3_scaled above shows the
// compiler turns EVERY arithmetic select back into a predicated branch:
// 4x `vx_split_n`/`vx_join` warp-divergence regions and ~30 instructions PER
// ELEMENT. requant_P_to_spad_tiled's body was 1763 instructions per (row,block)
// item (267 max-pass + 8 x 187 convert), i.e. ~9.4k instructions per warp for
// its 5.33 items -> ~28k issue cycles/core of the measured 45k.  Instruction
// issue, not SMEM, was the first-order cost.
//
// DERIVATION: let u = (b>>4) & 0x7ff = (exp<<3)|m3  (bf16 bits [14:4]) and
//   K = (120 + se) << 3.   Then
//   t = u - K = 8*(exp - 127 - se + 7) + m3 = 8*(E+7) + m3
// which IS the e4m3 code (biased-exponent<<3 | mantissa) with no further work.
//  * OVERFLOW CAN NEVER HAPPEN: se is the exponent of the block MAX and every
//    element of the block is <= that max, so E <= 0 and t <= 63.  All the
//    saturation logic (E>emax, m3>6 at emax) is dead code -> deleted.
//  * UNDERFLOW is exactly t < 8:  E < emin(-6)  <=>  E+7 <= 0  <=>  t <= 7.
//    Done with an arithmetic-shift mask, which has no comparison and therefore
//    no warp split.  (Callers clamp se >= -120 so that a zero/subnormal input,
//    u <= 7, can never alias into t >= 8.)
//  * SIGN is always 0 (P are softmax probabilities >= 0) -> dropped.
// 7 integer ops per element, zero divergence.
// WHY INLINE ASM: writing this in C as `t & ~((t-8)>>31)` is NOT enough -- the
// muon backend recognises the mask idiom, turns it back into a select, and emits
// `slti/xori/vx_split_n/beqz/mv/vx_join`, i.e. a warp-divergence region per
// element again (measured: 35 vx_split in the C version).  The asm block below
// is the only way to guarantee the straight-line form.  rv32im only here: no
// Zbb (max/min/andn) and no Zicond (czero.*) -- verified by test-assembling --
// so the mask costs srai+not+and.
//
// clamp_K8(K) = max(K,0) + 8, branchless.  Callers need max(K,0) because K<0
// (an all-zero block, se=-127) would let a zero input (u11<=7) alias to t>=8.
static inline uint32_t fa_clamp_K8(int K) {
    uint32_t r, m;
    asm("srai %1, %2, 31\n\t"
        "not  %1, %1\n\t"
        "and  %0, %2, %1\n\t"
        "addi %0, %0, 8"
        : "=&r"(r), "=&r"(m) : "r"(K));
    return r;
}

// 4 bf16 (two packed words) -> one packed word of 4 e4m3 codes.  34 straight-line
// instructions, 3 temporaries, zero branches.  K8 = max(K,0)+8.
// Per element: 2 ops to extract u11 = bits[14:4] (the shl/shr pair also clears
// the sign bit, so -0.0 still maps to code 0), then sub/srai/addi/not/and.
static inline uint32_t e4m3_pack4(uint32_t wlo, uint32_t whi, uint32_t K8) {
    uint32_t acc, t, m;
    asm("slli %[t], %[wl], 17\n\t"   // -- element 0: wlo[14:4]
        "srli %[t], %[t], 21\n\t"
        "sub  %[t], %[t], %[k]\n\t"
        "srai %[m], %[t], 31\n\t"
        "addi %[t], %[t], 8\n\t"
        "not  %[m], %[m]\n\t"
        "and  %[a], %[t], %[m]\n\t"
        "slli %[t], %[wl], 1\n\t"    // -- element 1: wlo[30:20]
        "srli %[t], %[t], 21\n\t"
        "sub  %[t], %[t], %[k]\n\t"
        "srai %[m], %[t], 31\n\t"
        "addi %[t], %[t], 8\n\t"
        "not  %[m], %[m]\n\t"
        "and  %[t], %[t], %[m]\n\t"
        "slli %[t], %[t], 8\n\t"
        "or   %[a], %[a], %[t]\n\t"
        "slli %[t], %[wh], 17\n\t"   // -- element 2: whi[14:4]
        "srli %[t], %[t], 21\n\t"
        "sub  %[t], %[t], %[k]\n\t"
        "srai %[m], %[t], 31\n\t"
        "addi %[t], %[t], 8\n\t"
        "not  %[m], %[m]\n\t"
        "and  %[t], %[t], %[m]\n\t"
        "slli %[t], %[t], 16\n\t"
        "or   %[a], %[a], %[t]\n\t"
        "slli %[t], %[wh], 1\n\t"    // -- element 3: whi[30:20]
        "srli %[t], %[t], 21\n\t"
        "sub  %[t], %[t], %[k]\n\t"
        "srai %[m], %[t], 31\n\t"
        "addi %[t], %[t], 8\n\t"
        "not  %[m], %[m]\n\t"
        "and  %[t], %[t], %[m]\n\t"
        "slli %[t], %[t], 24\n\t"
        "or   %[a], %[a], %[t]"
        : [a] "=&r"(acc), [t] "=&r"(t), [m] "=&r"(m)
        : [wl] "r"(wlo), [wh] "r"(whi), [k] "r"(K8));
    return acc;
}

// SWAR variant: both bf16 of a word are converted in parallel inside one 32-bit
// register (11-bit payloads live in [10:0] and [26:16], so they never collide).
// 25 instructions per output word vs 34 for the scalar asm above.
//   x = (w>>4) & 0x07ff07ff          per-field u11 (this also clears the sign)
//   y = x + D2,  D2 = (0x7FF8-K) broadcast  ->  y = code + 0x7FF8 per field
//     the 0x7FF8 bias is chosen so that BIT 15 OF EACH FIELD == (code >= 8),
//     i.e. the underflow test is just the field's top bit -- no compare at all.
//   g = y & 0x80008000               per-field validity bit
//   m = g - (g>>15)                  0x7FFF / 0 mask (no cross-field borrow)
//   r = (y & m) + (g>>12)            (y&m) = code-8 when valid, 0 when not;
//                                    (g>>12) adds the 8 back only where valid
//   pack: (r | r>>8) & 0xffff        -> c0 | c1<<8   (c_i <= 63, so no overlap)
// Ranges are safe: u11 <= 0x7FF and 0 <= K <= 960, so every field stays in
// [0x7C38, 0x87F7] -- no carry out of a field, no borrow into one.
static inline uint32_t e4m3_pack4_swar(uint32_t wlo, uint32_t whi, uint32_t D2,
                                       uint32_t C, uint32_t M, uint32_t H) {
    uint32_t acc, y, g, t;
    asm("srli %[y], %[wl], 4\n\t"
        "and  %[y], %[y], %[c]\n\t"
        "add  %[y], %[y], %[d]\n\t"
        "and  %[g], %[y], %[m]\n\t"
        "srli %[t], %[g], 15\n\t"
        "sub  %[t], %[g], %[t]\n\t"
        "and  %[y], %[y], %[t]\n\t"
        "srli %[g], %[g], 12\n\t"
        "add  %[y], %[y], %[g]\n\t"
        "srli %[t], %[y], 8\n\t"
        "or   %[y], %[y], %[t]\n\t"
        "and  %[a], %[y], %[h]\n\t"
        "srli %[y], %[wh], 4\n\t"
        "and  %[y], %[y], %[c]\n\t"
        "add  %[y], %[y], %[d]\n\t"
        "and  %[g], %[y], %[m]\n\t"
        "srli %[t], %[g], 15\n\t"
        "sub  %[t], %[g], %[t]\n\t"
        "and  %[y], %[y], %[t]\n\t"
        "srli %[g], %[g], 12\n\t"
        "add  %[y], %[y], %[g]\n\t"
        "srli %[t], %[y], 8\n\t"
        "or   %[y], %[y], %[t]\n\t"
        "slli %[y], %[y], 16\n\t"
        "or   %[a], %[a], %[y]"
        : [a] "=&r"(acc), [y] "=&r"(y), [g] "=&r"(g), [t] "=&r"(t)
        : [wl] "r"(wlo), [wh] "r"(whi), [d] "r"(D2),
          [c] "r"(C), [m] "r"(M), [h] "r"(H));
    return acc;
}

// ============================================================================================
// FA_SM_2P / FA_SM_2PRAW / FA_SM_2PBM  (+ FA_SP_CVTXS, FA_SP_PAX in kernel.cpp)
// A FOURTH-PASS STEADY-STATE SWEEP ON TOP OF FA_SP_QSPLIT.  All of these are BIT-EXACT by
// construction, so every one of them must still score exactly 3.5666% against golden_O_u16.npy;
// that is what lets them be part of a verified headline, unlike the whole SMTPR/SMBMAX/SUBMAX/
// NOMAX family which buys speed by moving the numerics to 4.2-7.8%.
//
// *** RETRACTED: THE "53-REGISTER CLIFF" THIS BLOCK WAS ORIGINALLY WRITTEN AROUND DOES NOT EXIST,
// AND fa_regs.py's WHOLE-FILE UNION IS NOT A MEASURE OF THE CONSTRAINT. ***  Rename.scala:110-123
// hands out a physical register the first time a given WARP writes a given ARCH register and draws
// from ONE counter per CORE, so the constraint is  sum over a core's resident warps of |arch regs
// THAT WARP writes| <= 255.  A whole-file union charges the warp-0 agent path to warps that branch
// around it, misses the RUNTIME entirely (_start / init_regs / mu_schedule are in the linked elf,
// not in the kernel .s, and every warp runs them), and collapses the 3-warps-per-core multiplier
// that is the term that actually binds.  fa_regs3.py computes it properly from the LINKED GPU elf.
// Measured there, on the FA_SP_WCNT baseline:
//     FA_SM_2P + FA_SM_2PRAW + FA_SP_PAX + FA_SP_CVTXS        core 1 = 3 x 55 = 165 / 255
//     ... + FA_SM_2PBM + FA_SP_SMBMAX                         core 1 = 3 x 57 = 171 / 255
// i.e. ~90 and ~84 PHYSICAL registers of headroom -- room for ~28-30 MORE arch registers in an
// all-warps function.  The union metric is not even MONOTONE in the real figure (FA_SP_SQ32 has a
// HIGHER union than a running baseline, 55 vs 53, and a much LOWER per-warp figure, 57 vs 72), so it
// could not have been rescued by moving its threshold.  The empirical "53 runs / 55 and 58 abort"
// bracket that this whole campaign designed around was reading a quantity that is not the budget.
// WHAT SURVIVES the retraction, because it is mechanism rather than threshold: registers ARE a
// shared per-core resource, so spending them in one all-warps function does reduce what another can
// use -- that part of the reasoning below is sound.  WHAT DOES NOT: every "X is fatal at N
// registers" judgement, including my own worry that FA_SM_2PBM and FA_SP_CVTXS could not coexist.
// They coexist at 171/255 with 84 to spare.  The one rejection that stands is FA_SP_SM1FX, and it
// stands for the right reason on the right metric: 3 x 90 = 270 / 255, genuinely over.
// The practical consequence for the code below: several of its structures were contorted to save
// registers that were never scarce -- pass A's max was cut from four independent chains to two, and
// FA_SM_2PBM's block fold from two to one -- purely to chase the union down.  Those were free
// memory-level parallelism given away for nothing, and are the first thing to undo.
//
// *** AND THE RESULT THAT MATTERS MOST IN THIS PASS IS A CORRECTNESS RESULT, NOT A SPEED ONE:
// FA_SP_QSPLIT's "12 of 12" IS A PROPERTY OF ITS SCHEDULE, NOT OF ITS FIX. ***  Measured at
// FA_NT6, seed 12345, scored per cluster per tile with fa_verify_tiles.py against golden_O_u16:
//     FA_SP_QSPLIT + FA_SP_CVTX + FA_SP_PREPK   45,807 cyc/tile  35.85%   cluster 1 tile 1 is
//                                                                        *** WRONG, 106.66% ***
// Both added flags are BIT-EXACT BY CONSTRUCTION -- CVTX only XOR-permutes which of eight output
// words an unrolled iteration writes, PREPK only moves the pack's arithmetic off warp 0 -- so
// neither can change a computed value, and the failing image is fully covered (4096/4096 words),
// i.e. it is real corruption and not a truncated trace.  A -1,095-cycle schedule perturbation is
// therefore enough to lose the baseline's correctness.  Adding FA_SP_PAX on top (another -536, also
// bit-exact) does NOT fix it -- it only moves the ONSET, which is the tell:
//     + CVTX + PREPK          cl0 t0,t1,t2 CORRECT | cl1 t0 CORRECT, t1 106.7%, t2 114.6% WRONG
//     + CVTX + PREPK + PAX    cl0 t0,t1 CORRECT, t2 119.4% WRONG | cl1 t0,t1,t2 all CORRECT
// THREE PROPERTIES OF THAT PATTERN PIN THE MECHANISM, and they are why FA_SP_QDRAIN/FA_SP_QEARLY
// target the Q(t+1) DMA rather than anything else:
//   * the onset tile differs BETWEEN THE TWO CLUSTERS of one run, which are the same instructions
//     on the same data -- so the trigger is a RACE whose timing depends on something outside the
//     instruction stream, and the only such thing per tile is DRAM contention;
//   * once a cluster goes wrong it stays wrong for every later tile, but with a DIFFERENT Frobenius
//     each time -- so it is not one frozen corrupt resident operand (K^T, V or their scales, which
//     are loop-invariant) but a fresh corruption every tile, i.e. of the one thing re-fetched every
//     tile: Q;
//   * shortening stage S4 (CVTX -872 on the convert, PREPK -1.3k on the pack) makes it appear, and
//     lengthening the stage BEFORE S4 does not -- which is what "the DMA has less of S4 to finish
//     in" predicts and what "a fixed missing fence" does not.
// See FA_SP_QDRAIN / FA_SP_QEARLY in
// kernel.cpp for the mechanism and the fix; the point here is the METHODOLOGICAL one:
// *** ON THIS PIPELINE, EVERY PERFORMANCE FLAG MUST BE RE-SCORED AT NT6, AND A "12 of 12" CARRIES
// OVER TO A NEIGHBOURING CONFIGURATION ONLY IF THE HAZARD HAS BEEN DRAINED RATHER THAN MISSED. ***
// (The published F6 lever sweep quotes cycles for the CVTX/PREPK rows but no tile verdicts -- they
// were never scored.  That is how a wrong-answer configuration came to be recorded as a free win.)
//
// MEASURED, FA_NT6, seed 12345, CONVERGED (complete traces), pooled over BOTH clusters' steady
// intervals (fa_marks_cl.py), every row scored per cluster per tile with fa_verify_tiles.py:
//   configuration (all on FULL_ATTN2 FA_SP FA_SP_QOVL FA_SP_LEANCFG FA_SP_QKACC FA_SP_PKOVL
//                  FA_SP_QSPLIT)                cyc/tile  util    tile-images        regs
//   published baseline                            46,902  35.01%  12 of 12            53
//   + CVTX + PREPK                                45,974  35.72%  7 ok, 4 WRONG /11   53
//   + CVTX + PREPK + PAX                          46,542  35.28%  8 ok, 4 WRONG /12   53
//   + CVTX + PREPK + FA_SM_2P                   **43,896  37.41%  12 of 12 CORRECT**  50
//   + CVTX + PREPK + FA_SM_2P + PAX               44,242  37.11%  8 ok, 4 WRONG /12   50
// *** THAT 43,896 / 37.41% ROW IS RETRACTED AS A *VERIFIED* NUMBER, AND THE REASON IS A BUG IN
// FA_SM_2P ITSELF THAT REVIEW CAUGHT AND TWELVE CLEAN TILE-IMAGES DID NOT. ***  The pb stride was
// coded `2 * NT + 1` = 33 where it had to be `NT + 1` = 17, so the buffer ran 1,756 bytes past
// 0x16000 into Q's operand spad on every tile -- and it scored 12 of 12 anyway, because Q(t+1)'s
// move-in DMA is issued in stage S4, AFTER this stage's barrier, and happened to rewrite exactly
// the rows the overrun had corrupted before the QK in S6 could read them.  A second, independent
// overrun sat in FA_SM_2PBM's bb buffer (3,168 B from 0x15000, colliding with pb at 0x15680).
// THREE THINGS ARE WORTH TAKING FROM THIS, because it is the third time in this campaign that a
// green result rested on an accident:
//   * the timing measurement was never in doubt (pb is pure scratch, so the cycle count is real),
//     but the CORRECTNESS verdict was, and only the correctness verdict;
//   * FA_SP_QEARLY WOULD HAVE DETONATED IT -- that flag moves the Q DMA to the top of this very
//     stage so it runs CONCURRENTLY with the softmax, after which nothing covers the overrun.  The
//     two flags were built and queued together, so the accident was one run away from becoming a
//     mystery corruption in a configuration where the softmax looked innocent;
//   * the fix is not "change 33 to 17", it is to make the bound a COMPILE-TIME property:
//     static_assert(PB_BASE + SQ*STR*2 <= 0x16000) and the matching one for bb.  Verified
//     non-vacuous by reinstating the bug, which now fails the build instead of the golden.
// The in-bounds version is re-measured at FA_NT6 before any number here is quoted as verified; the
// register counts are unchanged (50 / 51 / 55), so the register conclusions above stand.
//
// AND A THIRD MEASUREMENT TRAP, WHICH THE TWO WRONG ROWS ABOVE DEMONSTRATE: *** A CORRUPT TILE HAS
// A CORRUPT *TIMING* TOO, SO A WRONG-ANSWER RUN'S CYCLE COUNT IS NOT A PERFORMANCE MEASUREMENT. ***
// The two clean rows have interval spreads of ~1,400 cycles; the +PAX row's are
//     cl0 [45853, 43274, 45748, 50609, 53245]   cl1 [45245, 46713, 45766, 41459, 47504]
// -- 41,459 to 53,245, a 28% spread, and the outliers are exactly the corrupted tiles.  That is why
// +PAX appears to be a +568 LOSS here while its own stage (requant pass A) provably shrinks by 661,
// and why its cluster-0 tiles alone (41,488..43,702 on the FA_SM_2P row) look like a ~600 WIN.  The
// honest statement is that PAX's tile-level effect is NOT MEASURABLE until the corruption is fixed.
// Corollary for the published F6 sweep: its CVTX/PREPK rows were never scored, so both their
// verdicts AND their cycle counts have to be discarded, not just the verdicts.
//   and the per-stage attribution (cluster 0, steady tiles; +-500 of phase noise per stage):
//                          CVTX+PREPK   +PAX   +FA_SM_2P
//     S1 accumulator->S           998    1386        998
//     S2 softmax               12,825  12,825   *10,799*   <- FA_SM_2P: -2,026
//     S3 requant pass A         3,415  *2,754*     3,375   <- FA_SP_PAX: -661
//     S4 convert || pack       10,243  10,422      9,989
//     S5 PV                     8,883   8,898      8,882   <- mesh; irreducible, see below
//     S6 QK || finalize         8,938   8,614      9,346   <- already within ~700 of the QK mesh
//
// TWO STRUCTURAL LIMITS ESTABLISHED (both negative, both worth not re-deriving):
//  1. *** NO MATMUL IN THIS KERNEL CAN BE SPLIT ALONG M, N OR K. ***  The mesh's scale-SRAM read
//     row is a fixed function of the loop bounds latched by CONFIG_SCALE_MEM, so a half-size
//     matmul re-reads the FIRST half's scale rows; making the second half correct means REWRITING
//     those rows between the two matmuls (64-128 words at ~65 cyc/word, strictly serial) while the
//     mesh is reading them.  All four scale slots (SF_A/SF_B x 2 halves) are already occupied by
//     K, V, Q and packed-P, so there is no spare half to double-buffer through.  This kills the
//     whole "split the mesh op so SIMT can hide under it" family, which is why stage S5's 8,883
//     cycles of PV are exposed with five warps idle and stay that way.  THE ONE EXCEPTION is the
//     store-only loop_ws (fa_store_acc): it has no scale dependency at all and CAN be split.
//  1b. RETRACTION, for the record: I reported that HEAD "cannot build any FA_SP configuration"
//     because fa_cfg_settle was referenced five times and never defined.  That was true of the
//     commit I tested against (1cce749: zero occurrences in its mxgemm_core.hpp) and is NOT a
//     latent flag hole -- mxgemm_core.hpp guards the definition with
//     `#if defined(FA_CFGSETTLE)||...` / `#else static inline void fa_cfg_settle() {}`, so BOTH
//     branches define it and all six FA_CFGSETTLE* combinations compile (verified).  It was a
//     transient window in a shared tree: the definition lived in a concurrently-edited
//     mxgemm_core.hpp that landed in ea1be58, after 1cce749 and after my own commit.  Nothing to
//     close; the claim as I first phrased it was wrong.
//  2. The HW MxRequantizer cannot replace the convert.  copy_P_to_requant is warp-0-only and the
//     requantizer's SMEM manager takes EXACTLY 32-byte beats, so feeding it 64x256 bf16 is 1,024
//     sequential 8-lane stores on ONE warp -- the same single-thread-serial-port wall as the SF
//     pack, against a convert that runs on five warps in parallel.
//
// ============================================================================================
// SIXTH-PASS RESULTS (2026-07-29).  All on FA_SP_WCNT, which is MANDATORY; every configuration
// scored per cluster per tile with fa_verify_tiles.py, cycles from fa_marks3.py with an explicit
// --mesh 16420, and every ELF's .text byte-compared against a rebuild from the source it was
// measured on (see the provenance note below -- a whole-file compare does NOT work here).
//
//   configuration (+ FA_SP QOVL LEANCFG QKACC PKOVL QSPLIT WCNT)  cyc/tile  util   tile-images
//   PAX + CVTX                          (banked tag)                45,582  36.02%  12/12 NT6
//   PAX + CVTX + FA_SM_2P                                           43,808  37.53%  7 ok, 5 WRONG
//   PAX + CVTX + FA_SM_2P + FA_SM_2PRAW            (E1)             42,846  38.32%  12/12 NT6
//   ... the same config at FA_NT8                  (E2)             42,814  38.35%  15 ok, 1 WRONG
//   ... + FA_SP_PREPK                              (E3)             42,666  38.48%  9 ok, 3 WRONG
//   PAX + CVTX + FA_SP_PREPK                       (isoA)           45,575  36.03%  8 ok, 4 WRONG
//   PAX + CVTXS + FA_SM_2P + FA_SM_2PRAW           (zw2)            47,702  34.42%  12/12 NT6
//   ... the same config at FA_NT8                  (ybk3)           47,844  34.32%  16/16 NT8
//   PAX + CVTX + FA_SP_BANKA                       (ybk1)           45,278  36.26%  9 ok, 3 WRONG
//   FA_SM_2P + FA_SM_2PRAW + FA_SM_2PBM + SMBMAX   (pbm2)           50,692  32.39%  10 ok, 2 WRONG
//   FA_SP_OPV + OPVQ + BANKA (+/- FA_SM_2P)        (G1/G2)              --      --  HANG, 0 images
//
// VERDICTS:
//   FA_SM_2P + FA_SM_2PRAW  the fastest thing measured in this campaign, 42,846 = 38.32%, and it
//       does NOT BANK: 12/12 at NT6 but 15-of-16 at NT8.  The softmax stage really does go
//       12,825 -> 10,050 (-2,775) and that part is solid; the tile is simply not correct.
//   FA_SP_CVTXS  *** A LOSS, +4,766 ON STAGE S4, AND I SHOULD NOT HAVE SHIPPED IT INTO A STACK. ***
//       The SWAR packer is 56 FEWER static instructions per item and still much slower, because its
//       27-instruction inline-asm chain is one serial dependency where e4m3_pack4's per-element form
//       has ILP.  This stage is latency-bound, not issue-bound.  A static instruction count is not a
//       cost model.  (It is also what made my zw2 "headline" 2,120 cycles worse than the baseline
//       while I was attributing the gap to FA_SM_2P.)
//   FA_SP_PREPK   flips correctness at a SEVEN-CYCLE change in tile time (45,575 vs the banked
//       45,582).  See the hazard note below -- this is the cleanest evidence in the whole campaign
//       that what we are looking at is a data race whose window is being moved, not a schedule.
//   FA_SP_BANKA / FA_SM_2PBM / FA_SP_FZ6 / FA_SP_OPV   all LOSSES or WRONG; see the individual notes.
//
// ============================================================================================
// SEVENTH PASS (2026-07-30) -- RE-TESTING THE "REJECTED FOR CORRECTNESS" FLAGS ON TOP OF
// FA_SP_ACCPAD, AND THE RESULT IS THAT *** THE QUESTION AS POSED CANNOT BE ANSWERED BY THESE RUNS,
// BECAUSE THE BASE THEY SIT ON IS ITSELF INCORRECT. ***
//
// THE PREMISE.  FA_SP_ACCPAD took an otherwise-failing configuration to 16/16 at NT8 (H1), so the
// hypothesis was that a family of flags had been rejected while a latent accumulator-drain hazard was
// active, that they were innocent, and that ACCPAD would restore both their correctness and their
// cycle savings.  Every flag below was re-tested at FA_NT6 on the exact ACCPAD base (rv32 GPU-image
// sha 5028374f1b66d3ae, byte-identical to the ELF the P128 run loaded -- see the provenance note,
// because the `-j .text` recipe does NOT establish that), scored per cluster per tile.
//
//   configuration (+ FA_SP QOVL LEANCFG QKACC PKOVL QSPLIT WCNT PAX CVTX FA_SM_2P FA_SM_2PRAW)
//                                                    cyc/tile  util    spread  tile-images
//   FA_SP_ACCPAD (N=128) -- THE BASE          (P128)   44,565  36.85%   8.6%   12 of 12
//   ... + FA_SP_PREPK                         (ypk)    44,131  37.21%   6.0%   12 of 12   (-434)
//   ... + FA_SP_FZ6                           (yfz)    44,744  36.70%   6.1%   12 of 12   (+179)
//   ... + FA_SM_2PBM                          (ybm)    49,955  32.87%   5.4%   12 of 12 (+5,390)
//   ... + FA_SM_2PBM + FA_SP_SMBMAX + PREPK
//                      + FA_SP_BANKA          (ymax)   47,154  34.82%   7.1%   12 of 12 (+2,589)
//   ... + FA_SP_BANKA                         (ybka)      --      --   36.5%    7 ok, 5 WRONG
//   ... + FA_SM_2PBM + FA_SP_SMBMAX           (ybmx)      --      --   24.6%    8 ok, 4 WRONG
//   ... + FA_SM_2PBM + FA_SP_SMBMAX + PREPK   (ybmxpk)    --      --   26.1%    2 ok, 10 WRONG
//   (spread = (max-min)/mean over the 10 pooled steady intervals.  Every WRONG row is at 24-37%
//   against 5-9% for the clean ones -- the corrupt-timing signature this file already documents --
//   so those means are configuration labels and NOT performance results.)
//
// *** AND THEN THE RESULT THAT INVALIDATES THE WHOLE TABLE.  FA_PHASE<k> -- the inter-cluster phase
// sweep mxgemm_core.hpp has carried since ea1be58 and that HAD NEVER BEEN RUN -- BREAKS THE BASE
// ITSELF, AT EVERY k TRIED: ***
//   no skew   (P128)  12 of 12
//   FA_PHASE1 (yph1)  7 of 12   cluster 1 (THE DELAYED ONE) tiles 1-5 WRONG, cluster 0 6 of 6 clean
//   FA_PHASE2 (yph2)  7 of 12   the identical pattern: cluster 1 tiles 1-5 WRONG, cluster 0 clean
//   FA_PHASE3 (yph3)  4 of 12   cluster 0 3 WRONG *and* cluster 1 5 WRONG -- it WORSENS with skew
// yph2's cluster 1 reads 108.5643 / 113.0664 / 118.1504 / 119.3714 / 119.3714% -- latching, growing,
// never recovering: the campaign's exact fingerprint, all images 4096/4096 words so not truncation.
// FA_PHASE only issues dependent READ-ONLY loads of GEMMINI_BUSY_ADDR (RegField.r, unconditionally
// valid read, no side effect: GemminiTile.scala:428 with :404-405) and discards the value, so it
// cannot change a computed result.  *** THEREFORE THE 12-of-12 IS THE ARTEFACT AND THE 5-WRONG IS
// THE TRUTH: FA_SP_ACCPAD DOES NOT CLOSE THE HAZARD, IT MOVES THE WINDOW, AND EVERY "12 of 12" IN
// THE TABLE ABOVE IS A LOTTERY TICKET RATHER THAN A FLAG PROPERTY. ***
// (The control that decides whether the harness itself is to blame -- FA_PHASE_BOTH, which gives BOTH
// clusters the identical delay loop so the executed code is unchanged but the relative phase is not
// -- is defined in mxgemm_core.hpp and was in flight when this was written.  If FA_PHASE_BOTH is also
// wrong, the harness is the bug and this whole block must be thrown away.)
//
// THE TABLE ALSO REFUTES ITSELF INTERNALLY, WHICH NEEDS NO PHASE RUN AT ALL: THE VERDICTS ARE NOT
// MONOTONE IN THE FLAG SET.  ybmxpk (2PBM+SMBMAX+PREPK) is 2 of 12; ymax is ybmxpk PLUS FA_SP_BANKA
// and is 12 of 12 -- adding a flag to a 2-correct build makes it 12-correct.  And FA_SP_BANKA ALONE
// (ybka) is 7 of 12.  No account in which "correct/incorrect" is a property of the flag can produce
// that ordering; a schedule lottery produces it immediately.
// So the honest verdicts are:
//   * NOTHING IS RESCUED, because nothing was established as broken by its own flag to begin with.
//     FA_SP_PREPK is the sharpest case and it cuts against the original claim rather than for it:
//     -434 cyc/tile and 12 of 12 here, having been 9-ok/3-WRONG as E3 -- and NEITHER number is
//     evidence about PREPK.  (The "PREPK flips correctness at a SEVEN-CYCLE cost" line above is also
//     not admissible: 45,575 is the mean of a 4-WRONG run, and this file's own rule forbids reading a
//     corrupt run's mean as a tile time.  The two numbers being 7 apart is a coincidence of two
//     incommensurable quantities.)
//   * TWO REJECTIONS SURVIVE, ON *CYCLES*, WHICH FA_PHASE CANNOT TOUCH: FA_SM_2PBM alone is +5,390
//     (it adds the block-max fold and, without FA_SP_SMBMAX, deletes nothing -- verified statically:
//     fa_requant_max is PRESENT in ybm's GPU image and ABSENT from ybmx's and ymax's, so SMBMAX does
//     compile pass A out as claimed), and the 2PBM+SMBMAX pair still does not pay: ymax, the only
//     correct-as-built build containing it, is +2,589.  FA_SP_FZ6 is +179 here against its previously
//     measured +1,330 -- still a loss, just a smaller one.
//   * FA_SP_ACCPAD ITSELF COSTS 1,694 cyc/tile, ALL of it in stage S1 (2,692 vs 998 for the same
//     stage without it), and buys nothing that survives a phase perturbation.
// *** THE PEAK-UTILISATION RESULT OF THIS PASS: FA_SP_ACCRS -- DELETE THE PRE-STORE DRAIN AND LET
// THE RESERVATION STATION ORDER THE STORE -- IS 42,650 cyc/tile = 38.50% WITH BIT-EXACT OUTPUT ON
// 12 OF 12 TILE-IMAGES, THE FASTEST VERIFIED POINT IN THIS CAMPAIGN.  THE HONEST LIMIT ON IT IS
// "FAILS AT TILE 7 OF 8". ***
//     FA_SP_ACCRS, no pad, no pre-store fa_gfl  (yrs)  FA_NT6  42,650  38.50%  8.1%  12 of 12
//     ... the SAME build at FA_NT8              (yrs8)         42,644  38.50%  8.1%  15 of 16
//                                               cluster 1 TILE 7 at 104.7421%
// (I first wrote this row off as "also wrong".  That was the wrong call and it is worth naming why:
// the NT6 output is bit-exact against golden_O_u16.npy and the interval spread is 8.1%, i.e. a clean
// measurement, so the cycle number is real.  A later failure at a longer tile count or under a
// perturbation bounds the configuration's ROBUSTNESS; it does not retroactively unverify the tiles
// that were verified.  Peak utilisation and robustness are two results, and collapsing them loses
// the one the utilisation question actually asks for.)
// -1,915 cyc/tile against the ACCPAD base and -196 against E1 -- and E1 fails FA_NT8 15-of-16 at
// cluster 0 tile 7, the same shape on the last tile.  *** SO REMOVING THE DRAIN NEITHER FIXES NOR
// BREAKS CORRECTNESS: IT IS PURE CYCLE SAVING ON TOP OF AN EQUALLY WRONG KERNEL, AND ITS NT6
// 12-of-12 WAS ANOTHER LOTTERY TICKET (I had it written up here as a positive result on the strength
// of that one run; NT8 retracted it). ***  Kept rather than deleted for two reasons:
//   * THE RTL READING BEHIND IT STANDS AND IS THE MOST USEFUL THING IN THIS PASS.
//     ReservationStation.scala's STORE branch, for an opa_is_dst entry, computes deps_ex including
//     "additionally if ex writes, raw for st b <- ex a", i.e. new_entry.opb (fa_store_acc's
//     ACCUMULATOR SOURCE) against e.opa (the QK compute's ACCUMULATOR DESTINATION).  The hardware
//     interlock for exactly this pair exists and is exact, and the pre-store fa_gfl was DESTROYING it
//     by emptying the station before the store was ever allocated.  Two MMIO polls per tile were
//     paying for an ordering the hardware already provides.
//   * AND IT NARROWS THE HAZARD BY ELIMINATION: the one structure that could order the accumulator
//     read against the compute that wrote it DOES order it, and the failure survives anyway.  Taken
//     with AccumulatorMem.scala:619-624's same-row RAW interlock, *** THE ACCUMULATOR READ PORT IS
//     EXONERATED AND FA_SP_QKACC's accmem->spad store is not where this bug lives. ***
//   * PRACTICAL CONSEQUENCE: FA_SP_ACCRS STRICTLY DOMINATES E1 -- identical tile verdicts at both
//     NT6 and NT8, 196 cyc/tile cheaper -- so whoever ships this pipeline shape should ship it
//     without the pre-store drain, and should not ship FA_SP_ACCPAD at all (+1,694 for nothing).
// ============================================================================================
//
// ============================================================================================
// *** SEVENTH-PASS FRONTIER (2026-07-31).  PEAK UTILISATION 38.78% AT NT6 / 38.76% AT NT8, BOTH
// BIT-EXACT, AND THE CORRECTNESS GATE IS "LARGEST TILE INDEX SURVIVED" RATHER THAN NT6/NT8. ***
//
//   config = FULL_ATTN2 FA_SP QOVL LEANCFG QKACC PKOVL QSPLIT WCNT PAX CVTX FA_SM_2P FA_SM_2PRAW
//            + FA_SP_ACCRS + FA_SP_PREPK          (GPU image c97d00047b189c9b at FA_NT8)
//   FA_NT6  (zAP6)   42,344 cyc/tile  38.78%  spread 6.7%  12 of 12
//   FA_NT8  (zAP8)   42,364 cyc/tile  38.76%  spread 7.0%  16 of 16
//   FA_NT24 (mAP24)  42,373                   spread 7.4%  41 of 46 -- cluster 0 onset at TILE 17,
//                                                          cluster 1 clean through tile 23
//   vs the git-tagged fa-mx-best-36.02: -3,234 cyc/tile, +2.75 POINTS, at the same 16 of 16.
//   (mAP24's mean is NOT quotable -- 5 wrong tiles -- but it agrees with the two clean runs to
//    within 30 cycles, which is a useful consistency check on the clean numbers.)
//
// LEVER ACCOUNTING, all measured on this base, all bit-exact by construction:
//   FA_SP_ACCRS   -1,915 vs the ACCPAD base, -196 vs E1.  Deletes the pre-store fa_gfl and lets
//                 ReservationStation's "raw for st b <- ex a" clause order the accumulator read.
//   FA_SP_PREPK   -306 here (-434 on the ACCPAD base).  The only lever that survived the pass.
//   FA_SP_BANKA   *** REFUTED: -10 at NT6, +28 at NT8, i.e. noise, and WRONG on its own (zAB6,
//                 10 of 12).  Its recorded -304 came from ybk1, a 9-ok/3-WRONG run. ***
//   FA_SP_ACCPAD  +1,694 and buys nothing that survives; FA_SM_2PBM +5,390; CVTXS +4,766;
//                 FA_SP_FZ6 +179 here against a recorded +1,330 that also came from an unscored run.
//   THREE recorded lever values (BANKA -304, FZ6 +1,330, 2PBM's row) traced to runs whose own tiles
//   were wrong.  A cycle number from a corrupt run is not a measurement -- and this is the third
//   time that artifact has set a verdict in this file.
//
// *** THE CEILING OF THIS PIPELINE SHAPE IS ~38.8%, AND 40% NEEDS A DIFFERENT SHAPE. ***  My own
// projection of 39.18% failed by exactly BANKA's phantom -304.  What is left is SIMT: 23,128 of
// 42,650 (S2 softmax 10,028 | S3 pass-A 2,753 | S4 convert||pack 10,347) against a mesh-coupled
// 19,514 (S5 8,881 + S6 9,635 + S1 998) that cannot shrink without splitting a matmul, which
// structural limit 1 forbids.  Both SIMT stages are AT their floors:
//   * S2 -- the fence budget is already spent.  FA_SM_2P is 3 fences/warp = 9 per binding core
//     against the reference's 128; the reference->2P transition removes 119 fences for a MEASURED
//     -2,775, i.e. 23.3 cyc/fence, which reproduces this file's own ~22.  A GROUP-OF-TWO form goes
//     9 -> 64 fences = +1,280, SO IT IS A LOSS, NOT THE ~900 WIN PROJECTED ABOVE.  That projection
//     is correct against the REFERENCE and inverts once FA_SM_2P is in the stack; it was not built.
//   * S4 -- issue-bound, and FA_SP_CVTSWAR already lost there with 56 FEWER instructions.
// So the remaining ~1,300 to 41,050 has no identified source in this shape.  The Sq=32
// double-buffered restructure (PROJECTED, not measured) is the identified next step: it makes tile
// i's PV overlap tile i+1's softmax, the one overlap this single-buffer map cannot express.
//
// *** AND THE CORRECTNESS GATE HAS MOVED, WHICH IS THE MOST TRANSFERABLE RESULT OF THE PASS.  THE
// ONSET IS DETERMINISTIC, CONFIG-DEPENDENT, AND OFTEN BEYOND NT8. ***
//   E1+PREPK (no pad, no ACCRS)      onset c0 t3      11 of 16 at NT8
//   FA_SP_ACCPAD_N4 base             onset c0 t3       9 of 12 at NT6
//   FA_SP_ACCRS alone                onset c1 t7, c0 t12   -- NT16 AND NT24 AGREE EXACTLY
//   FA_SP_ACCPAD + PREPK             onset c1 t7, c0 t9    -- yet 16 of 16 at NT8
//   FA_SP_ACCRS + PREPK (frontier)   onset c0 t17, c1 none through t23
// TWO CONSEQUENCES:
//   1. NT8 PASSES BY LUCK.  ACCPAD+PREPK is 16-of-16 at NT8 with an onset at tile 9, and ACCRS alone
//      is 15-of-16 with an onset at 7 -- one tile of margin in each case.  A real invocation of
//      TinyLlama-1.1B at S=2048 is 144 tiles = 72/cluster, so it reaches EVERY onset here.  NO
//      configuration in this campaign survives a real invocation; the frontier's onset at 17 would
//      still corrupt 55 of 72 tiles.  Report peak utilisation and largest-tile-survived separately.
//   2. THE ONSET IS REPRODUCIBLE FOR A GIVEN BINARY -- now FOUR runs, TWO configs, FOUR tile counts,
//      all in EXACT agreement, which is the strongest single statement in this pass:
//          FA_SP_ACCRS        nA16 (NT16) c0=12 c1=7  ==  nA24 (NT24) c0=12 c1=7
//          FA_SP_ACCPAD+PREPK nP24 (NT24) c0= 9 c1=7  ==  nL72 (NT72) c0= 9 c1=7
//      So this is NOT a coin flip -- it is a deterministic function of the schedule, and the tile
//      index at which it fires is a STABLE, CHEAP OBSERVABLE of a configuration.  Two practical
//      consequences: (a) a config can be characterised by its onset from ONE run, no repetition
//      needed; (b) "it passed at NT<n>" means only "its onset is > n", which is why NT8 was
//      producing false confidence.
//      *** AND IT CLUSTERS AT TILE 7 ACROSS THREE INDEPENDENT CONFIGS, which points at a resource
//      that turns over on a period of ~8 tiles rather than a random race.  The waveform window that
//      follows is cluster 1, the tile 6->7 boundary, reproducible on any of three builds. ***
// BUDGET WARNING FOR LONG-NT RUNS: a corrupt run is MUCH slower than a clean one (nP24's cluster 0
// managed 12 tiles in 1.6M cycles, ~128k/tile against a 44k nominal), so size long-NT budgets for
// the CORRUPT case.  nP24 and mAP24 both truncated at their wall; the onset is still valid because
// the wrong images are complete (4096/4096), but neither is an NT24 result.
// ============================================================================================
// ============================================================================================
// THE PHASE SWEEP APPLIED TO EVERYTHING, INCLUDING THE GIT TAG. (2026-07-30/31)
//
// *** SCOPING NOTE, ADDED AFTER I OVER-CORRECTED THIS ONCE.  A FA_PHASE FAILURE DOES NOT
// RETROACTIVELY INVALIDATE A VERIFIED OUTPUT OR ITS CYCLE COUNT. ***  The runs below that scored
// 12/12 and 16/16 really were bit-exact against golden_O_u16.npy at the schedule they ran; what the
// sweep adds is that they are FRAGILE TO SCHEDULE PERTURBATION.  The correct phrasing is "correct at
// the schedule measured, not robust to perturbation", which is materially different from "not known
// correct".  For a peak-utilisation question -- what is the best util achievable at acceptable error
// under a fixed schedule -- a fixed schedule is a legitimate condition, so THESE NUMBERS STAND and
// the phase column is a robustness ANNOTATION, not a veto.  The corrupt-tile-implies-corrupt-timing
// rule still applies, but only to a run whose OWN tiles were wrong.
// (I originally wrote this block as "there is no correct configuration in this campaign".  That
// conflated the two statements and is withdrawn.)
//
//   configuration                       unperturbed   FA_NT8    PHASE1   PHASE2   PHASE3
//   fa-mx-best-36.02 (tagged) (ytg*)     12/12         16/16*   10/12    (pend)   12/12
//   FA_SP_ACCPAD N=128 base   (y*)       12/12         16/16     7/12     7/12     4/12
//   ... + FA_SP_PREPK         (ypk*)     12/12         16/16     7/12     5/12    10/12
//   FA_SP_ACCRS (no drain)    (yrs*)     12/12         15/16     8/12     5/12    10/12
//   FA_SP_ACCPAD N=4 (0x20+4) (q*)        9/12         11/16     --       9/12     --
//   (* the tag's 16/16 is bkNT8 from the sixth pass; ytg0 reproduces its NT6 number EXACTLY at
//    45,582 = 36.02%, 12/12, so the config under test is the tagged one.)
//
// EVERY ROW IS FRAGILE SOMEWHERE.  Three of them -- the tag, the ACCPAD base, and ACCPAD+PREPK --
// pass FA_NT8 SIXTEEN of SIXTEEN and then lose 2 to 7 tile-images to a delay that computes nothing.
// So "16 of 16 at NT8" establishes bit-exactness AT THAT SCHEDULE and does NOT establish robustness;
// report the two separately rather than collapsing them.
// ytg1's failure is cluster 0 tiles 4 and 5 at 93.3168% / 121.1437% -- the familiar latching shape,
// 4096/4096 words, on the configuration this repo has tagged and would ship.
//
// FA_SP_PREPK deserves one line of its own because it is the flag this pass was launched to rescue:
// on the ACCPAD base it is 44,131 = 37.21% and 12/12 at NT6, and 43,982 = 37.33% and 16 of 16 at
// FA_NT8 (ypk8) -- i.e. it clears every gate this campaign has ever used -- and it is 7/12, 5/12,
// 10/12 across PHASE1/2/3.  That is the single cleanest demonstration that the gates were too weak,
// and it is why "does ACCPAD rescue the rejected flags" cannot be answered as posed.
// ============================================================================================
//
// *** THE NT8 METHODOLOGY POINT, AND IT INVALIDATES A TEST WE HAVE BEEN RELYING ON: A SLOWER
// CONFIGURATION PASSING NT8 SAYS NOTHING ABOUT A FASTER ONE. ***  zw2 (with CVTXS, 47,844) passes
// NT8 16-of-16; E1/E2 (identical except CVTXS, 42,814) fails NT8 15-of-16.  CVTXS's +4,766 cycles
// were MASKING the race.  So "it passed NT8" is only evidence for the exact binary tested, and a
// correctness result may not be carried from a slow build to a fast one -- which is the direction we
// always want to carry it.
//
// *** WHERE THE SURVIVING HAZARD IS, LOCALISED BY THE CONVEX-HULL TEST RATHER THAN GUESSED. ***
// O = (P @ V) / l with P >= 0 and (P/l) summing to 1 across a row, so EVERY row of a correct O is a
// convex combination of V's 256 rows -- componentwise inside [min_k V[k][n], max_k V[k][n]].
// Decoding V from include/fa_data.h (e4m3 x 2^(E8M0-127)) and testing the WRONG images:
//     isoA cluster 1 tiles 2 and 5, and E2 cluster 0 tile 7:  0 of 8192 cells outside the hull.
// The garbage is still a convex combination of V.  That EXONERATES the PV matmul, its operand spad,
// the V scales and finalize -- every one of which would push cells out of the hull -- and it says P
// and l are still CONSISTENT WITH EACH OTHER.  So the fault is in S: Q, K^T, their MX scales, the QK
// matmul, or the accumulator->spad store.  Note this is the SAME localisation the fourth pass
// reached, now reproduced on the FA_SP_WCNT baseline -- i.e. WCNT fixed one drain in that chain and
// something else in the same chain survives it.
// AND THE FAILURE LATCHES AND WORSENS, which narrows it further:
//     isoA cluster 1:  tile 0,1 correct then 89.79%, 113.96%, 118.15%, 119.37%
//     E3   cluster 0:  tile 0,1,2 correct then 91.09%, 121.20%, 119.05%
// Once a cluster goes wrong it NEVER recovers, and the error GROWS.  But S must be bit-identical on
// every tile (Q, K^T, V and their scales are loop-invariant in FA_SP), so a permanent, accumulating
// corruption of S means a RESIDENT input to S is being progressively damaged.  The only inputs to S
// written once and never rewritten are K^T's spad (rows 6144..8192 = 0x18000..0x20000) and the K
// scales in SF_B half 0.  Q and the P scales are rewritten every tile and cannot latch.  The ratio
// got/golden is neither per-column nor per-row constant (within-line std 12..18), so it is not a
// single corrupted scale factor either -- consistent with a few damaged K^T words or K scale words
// spreading through the QK contraction.  *** THE NEXT STEP IS THE WAVEFORM, NOT ANOTHER A/B: an A/B
// cannot distinguish "fixed the race" from "moved the window", which is exactly the trap this file
// has fallen into four times.  An FSDB window over cluster 1's tile-1 stage S6 through tile-2 stage
// S1 (mcycle 191,000..204,000 on the isoA build) is where the first damaging write must be. ***
//
// *** CONSEQUENCE FOR THE BANKED TAG, STATED PLAINLY BECAUSE IT IS BETTER SAID THAN DISCOVERED:
// THE BANKED 45,582 / 36.02% CONFIGURATION IS IN THIS HAZARD'S FAMILY AND ITS "12 of 12" IS
// "12 of 12 AS BUILT AT FA_NT6", NOT A PROOF OF CORRECTNESS. ***  Three independent reasons:
//   * it is ONE bit-exact flag and SEVEN CYCLES away from isoA, which fails 4 of 12;
//   * the hazard's exposure grows with tile count -- E1 is 12/12 at NT6 and fails at tile 7 of 8 --
//     and the banked config had only ever been run at NT6 when it was tagged;
//   * every fix in this chain so far (the fourth pass's gemmini_fence, FA_SP_WCNT) removed one drain
//     and left the family alive, so there is no basis for assuming this one is closed.
// The tag is still the right thing to ship -- it is the best VERIFIED-AS-BUILT number -- but it
// should carry that qualifier until the waveform closes the mechanism.
//
// PROVENANCE, because the obvious check gives FALSE FAILURES on this build system: two consecutive
// builds of IDENTICAL defines from IDENTICAL source differ in 776 bytes, all at file offset
// >= 120,671 (tail metadata), while .text is bit-identical.  So a whole-file `cmp` reports every ELF
// as stale and tells you nothing.  Compare .text only:
//     llvm-objdump -s -j .text <elf> | grep -v "file format"
// Verified that way, all of the measured ELFs above match the source they were measured on.
//
// *** AND THAT RECIPE IS ITSELF BROKEN ON /tmp/flash_<TAG>.elf, WHICH IS THE FILE EVERY RUN ACTUALLY
// LOADS.  IT COMPARES THE RV64 *HOST* TEXT AND IS BLIND TO EVERY GPU-SIDE FLAG. ***  fa_build.sh
// copies flash_attention_mx.SOC.elf, which fuse_rv32_into_rv64.sh builds by embedding the rv32 GPU
// image as .rv32.seg0 .. .rv32.seg4 (seg2 = GPU .text at 0x110002000, seg4 = fa_data) alongside the
// Rocket host's own .text.  `-j .text` therefore selects host.cpp's code.  MEASURED: six builds
// differing in FA_NT6/FA_NT8, FA_SP_PREPK, FA_SP_FZ6, FA_SP_BANKA and FA_SM_2PBM all produced the
// IDENTICAL .text sha (4670c09a4f0f13ae) while their GPU images were all different -- i.e. the check
// passes every pair of builds and could never have failed.  The sound compare is the rv32 segments:
//     readelf -x .rv32.seg0 -x .rv32.seg1 -x .rv32.seg2 -x .rv32.seg3 -x .rv32.seg4 <elf> | sha256sum
// (or disassemble kernels/flash_attention_mx/flash_attention_mx.radiance.elf, but that file is
// OVERWRITTEN by the next build in this shared directory, so it has to be saved per tag).  Verified
// non-vacuous: it distinguishes all ten builds above, and it confirms an independent rebuild of the
// P128 defines is BYTE-IDENTICAL to the ELF the P128 run loaded (5028374f1b66d3ae).
//
// CLUSTER RAM INVENTORY (asked because SMEM being exactly full is what kills FA_SP_OPV).  TLPrintf
// (radiance/memory/Coalescing.scala:1266-1292) is NOT storage -- it is an empty class whose apply()
// emits a Chisel printf, zero RAM.  But `printBuf` (cluster/RadianceCluster.scala:96-103) IS a real,
// readable, writable, byte-maskable TLRAM: 512 B at device offset 0x80000, 1..8 B transfers, atomics
// supported, reached through the SAME clcbus leg the kernel already uses for GEMMINI_CTRL 0x84000
// and BUSY 0x84020 -- and completely unused by any RTL or software.  Free clcbus windows exist at
// 0x82000 (8 KB), 0x85000 (12 KB) and 0x8C000 (208 KB) where a second TLRAM would need NO change to
// the Muon-side address decode.  *** BUT IT IS A MAILBOX, NOT COMPUTE SCRATCH, AND THE REASON IS
// BANDWIDTH RATHER THAN CAPACITY: *** clcbus sits behind a cluster-wide TLSourceShrinker capped at
// 16 outstanding requests and a single-ported 8-byte SRAM, i.e. roughly one access per TENS of
// cycles, against the cooperative reduce scratch's measured 0.4..0.8 accesses per cycle.  That is
// one to two orders of magnitude short, so relocating REDBUF there does not rescue FA_SP_OPV.
// (I also have to correct a number I asserted: FA_SM_2P cuts REDBUF traffic 2.0x, not 31x --
// 10,164 -> 5,184 lane-accesses per tile, 0.79 -> 0.40 per cycle.  I had conflated warp-instructions
// with lane-accesses.  Whether 0.40 is under the four-entry dma_q's threshold is untested, because
// FA_SP_OPV now HANGS for an unrelated reason: G1 and G2 both ran 900,000 cycles producing ZERO
// tile-images, and G2 contains no FA_SM_2P at all, so the hang is OPV's own.)
// Explicit negatives so the search is not repeated: the requantizer window (0x40000-0x7FFFF) and the
// SF scale mem (0x88000-0x8BFFF) are WRITE-ONLY -- a load on the latter trips a $fatal; the 2,304 B
// of Gemmini LUT flops are RegField.w and unreadable; the 32 KB accumulator is hard tied off with no
// TL address; no Muon core exposes a DTIM/ITIM/spill aperture; the 16 KB contingent spad sits above
// the Muon's 31-bit global address cap; and NO SMEM subbank escapes the mesh, because the Gemmini
// spad read client is attached to all 64 (RadianceSharedMemComponents.scala:154-157, 202-204).
// ============================================================================================
// WHERE THE TILE'S TIME ACTUALLY GOES ONCE FA_SM_2P HAS LANDED (cluster 0, steady, the 12-of-12
// 43,896 configuration), and therefore what the FLOOR of this pipeline shape is:
//     S1 acc->S            998   agent-serial, 5 warps idle
//     S2 softmax        10,799   SIMT, 6 warps
//     S3 requant pass A  3,375   SIMT, 6 warps            (-661 with FA_SP_PAX, or -> 0 with 2PBM)
//     S4 convert||pack   9,989   SIMT warps 1-5 || warp-0 serial SF pack (~7.0k with PREPK)
//     S5 PV              8,882   MESH ONLY -- five warps idle, and structurally so (see limit 1)
//     S6 QK||finalize    9,346   SIMT finalize || QK mesh (8,210), i.e. ~1.1k of mesh slack
// The mesh-coupled part -- S5 + max(S6, QK) + S1 -- is ~18.2k and cannot shrink without splitting a
// matmul (impossible, limit 1) or double-buffering the tile (the Sq=32 route below).  The SIMT part
// is ~24.2k and is where every lever in this pass lives; with FA_SM_2PRAW (-500), FA_SP_PAX (-661)
// and FA_SP_CVTXS (-1,075) it projects to ~21.0k, i.e. a ~41.7k tile / ~39.4% util, and with
// FA_SM_2PBM replacing FA_SP_PAX (pass A deleted for ~+1.9k inside the softmax) ~40.8k / ~40.2%.
// *** SO ~40% IS THE CEILING OF THIS PIPELINE SHAPE, AND THE 8,882 EXPOSED CYCLES OF PV -- 20% OF
// THE TILE, WITH FIVE OF SIX WARPS IDLE -- ARE THE WHOLE REMAINING PRIZE. ***
//
// AND THE STRUCTURAL ROUTE THAT WOULD ACTUALLY PAY, worked out but NOT built: run the outer loop
// over Sq=32 HALF-TILES with S/P/O and P8 DOUBLE-BUFFERED.  Then tile i's PV overlaps tile i+1's
// softmax (different buffers), which is the one overlap the single-buffer map cannot express, and
// the floor becomes the SIMT total rather than SIMT + PV.  The SMEM arithmetic works out EXACTLY:
// 2 x 16 KB (S/P/O) + 2 x 8 KB (P8) = 48 KB, which is precisely what SP_C (32 KB) + P8 (16 KB)
// already provide, with V (32 KB), K^T (32 KB), Q (2 x 4 KB) and scratch (8 KB) unchanged = 128 KB.
// Neither matmul needs splitting (each half-tile issues its own full QK and PV), so limit 1 above
// does not bite, and one scale half per operand still suffices because PV(i) has drained long
// before PV(i+1)'s 64 words are packed.  Cost: 2x the matmuls, barriers and marks.
//
// THE 3.5666% -> 4.2342% SOFTMAX ACCURACY GAP: TWO MORE REFUTATIONS, ONE NEW MECHANISM, AND THE
// GENERAL AMPLIFIER -- BUT NOT CLOSED.  (Everything below is numpy bf16-RNE emulation on the REAL
// golden_S_u16, not argument.)
//   * l's SUMMATION ORDER IS REFUTED INDEPENDENTLY, reproducing (e3)'s numbers from scratch:
//     cooperative (16 chains of 16 + 16-leaf tree) l is 0.2919% rms from the exact sum,
//     thread-per-row (4 chains of 64 + combine) is 0.4287%, and the two l's differ from EACH OTHER
//     by 0.4986% rms -- against the 2.28% of independent error the step needs.  33 of 64 rows get a
//     BIT-IDENTICAL l either way.
//   * m IS REFUTED (as (e3) argued): x |-> RNE(x*scale) is monotone, so max_i RNE(S_i*scale) ==
//     RNE(max_i S_i * scale); emulated on the real S the two m vectors are byte-identical, 64 of 64.
//     (FA_SM_2PRAW turns that identity into a -500 cyc/tile lever rather than a worry.)
//   * NEW, AND REAL, AND FOUND IN THE DISASSEMBLY: *** clang CONTRACTS THE EXP ARGUMENT IN
//     fa_softmax_tpr. ***  It emits FOUR fmadd.h per unrolled group and ZERO fmul.h/fsub.h, i.e.
//     S*scale - m with ONE rounding, where online_softmax_block emits 16 fmul.h + 16 fsub.h, i.e.
//     RNE(RNE(S*scale) - m) with TWO.  The reference escapes contraction only because it
//     materialises the product into slo[]/shi[] first.  So the two softmaxes do NOT feed mu_fexp the
//     same numbers, which every previous analysis assumed.  Measured: the arguments differ on 13.26%
//     of the 16,384 elements, rms 0.0051 absolute, which exp turns into 0.5107% rms on P.
//   * MEASURED AMPLIFIER, and this is the part worth keeping: *** THE MX-FP8 QUANTISER MULTIPLIES
//     ANY SUB-PERCENT PERTURBATION OF P BY ~2.4x. ***  e4m3 has 3 mantissa bits, so its grid is
//     12.5% coarse; a 0.51% shift re-rounds 1.86% of elements by a FULL step and the quantised P
//     then differs by 1.2402% rms (and the row weights Q/l by 1.2425%).  THIS is why every rounding
//     change in this kernel costs ~0.7 Frobenius points regardless of what it is, why the whole
//     SMTPR/SUBMAX/NOMAX family lands at 4.2-7.8%, and why *** THE 3.5666% GATE IS IN PRACTICE A
//     BIT-EXACTNESS GATE, NOT AN ACCURACY GATE. ***  The operational conclusion: do not look for an
//     "accuracy-neutral" reordering, reproduce the rounding sequence exactly -- which is what
//     FA_SM_2P does and why it can be in a verified headline at all.
//   * STILL NOT CLOSED, and I am not going to claim otherwise: in quadrature the identified terms
//     give sqrt(3.5666^2 + 1.2402^2 + 0.4986^2) = 3.81%, so ~0.7 points of the step to 4.2342%
//     remain unaccounted for.  The remaining suspect is mu_fexp's own argument-dependent hardware
//     error being RE-SAMPLED by the contraction (an effect no software emulation can see), but the
//     arithmetic does not require it and I have not measured it.  A full numpy emulation of the
//     pipeline scores 8.74% against golden_O_u16 -- 2.4x the kernel's own 3.5666% -- so the e4m3 /
//     E8M0 model here is NOT faithful enough to settle a 0.7-point question, and any claim resting
//     on it would be worthless.  What IS settled: it is not l, not m, and not only the contraction.
//
// AND ONE MEASUREMENT THAT CORRECTS A NUMBER THIS FILE HAS BEEN REASONING FROM: *** fence.s COSTS
// ~22 CYCLES HERE, NOT ~100+. ***  FA_SM_2P removes 41 of a warp's 44 softmax fences and cuts 285
// instructions per row to 177+100/warp; the measured win is -2,015, of which the instruction count
// explains ~1,100 and the 41 fences therefore ~900, i.e. ~22 cycles each.  So the cooperative
// softmax was never "mostly reduction scaffolding fences" -- it was ~55% plain ISSUE on the binding
// core -- and that is why FA_SP_SM1F (which trades instructions for fences) had to lose, and why
// FA_SM_2PBM's per-row fence is affordable where a 16 KB warp-batched fold would have been needed
// if fences really cost 100.
// ============================================================================================
#ifdef FA_SM_2P
// Native single-instruction bf16 multiply with an EXPLICIT rounding of the product -- see the trap
// note in online_softmax_block's FA_SM_2P block.  (fa_max_h / fa_add_h in kernel.cpp
// are the same idea, but they are declared after this header is included.)
static inline _Float16 fa_sm2p_mul(_Float16 a, _Float16 b) {
    _Float16 o; asm("fmul.h %0, %1, %2" : "=r"(o) : "r"(a), "r"(b)); return o;
}
#endif

// 16-lane intra-warp tree reduction over a per-warp SMEM buffer (mirrors the
// softmax kernel's reduce_*: no in-loop fence; relies on warp lockstep). Result
// ends in buf[0]; caller fences then reads buf[0]. IS_MAX selects max vs sum.
// BUTTERFLY all-reduce (2026-07-25): 16 lanes, xor-exchange, NO divergence and
// NO broadcast read.  warp_tree_reduce below costs 4 `if (lane%stride==0)`
// blocks = 4 vx_split_n/vx_join warp-divergence regions per reduce, 8 SMEM
// half-loads + 4 half-stores, and then a trailing fence + buf[0] read to
// broadcast the answer.  The butterfly is 4 stores + 4 loads, every lane ends
// with the full result in a register, and only the FIRST cross-lane read needs
// a fence (later steps are ordered by warp lockstep, exactly as the existing
// unfenced tree steps already rely on).
template <bool IS_MAX>
static inline _Float16 warp_butterfly_reduce(volatile __shared uint16_t *buf,
                                             uint32_t lane, _Float16 v) {
    buf[lane] = __builtin_bit_cast(uint16_t, v);
#ifndef FA_SMB_NOFENCE   // (untested: the later steps already run unfenced)
    mu_fence_smem();
#endif
    for (uint32_t st = 1; st < MU_NUM_THREADS; st <<= 1) {
        _Float16 o = as_bf16(buf[lane ^ st]);
        v = IS_MAX ? fmaxf(v, o) : (_Float16)(v + o);
        if (st * 2 < MU_NUM_THREADS) buf[lane] = __builtin_bit_cast(uint16_t, v);
    }
    return v;
}

template <bool IS_MAX>
static inline void warp_tree_reduce(volatile __shared uint16_t *buf, uint32_t lane) {
    for (uint32_t stride = 2; stride <= MU_NUM_THREADS; stride <<= 1) {
        if (lane % stride == 0) {
            _Float16 a = as_bf16(buf[lane]);
            _Float16 b = as_bf16(buf[lane + (stride >> 1)]);
            _Float16 r = IS_MAX ? fmaxf(a, b) : (_Float16)(a + b);
            buf[lane] = __builtin_bit_cast(uint16_t, r);
        }
    }
}

// ============================================================================================
// FA_TREEFIX -- *** CORRECTNESS FIX for warp_tree_reduce.  PROVEN FROM AN ISSUE TRACE. ***
//
// warp_tree_reduce has NO FENCE BETWEEN TREE LEVELS: at level s+1 lane L loads
// buf[L + stride/2], which was written at level s by a DIFFERENT LANE.  The comment above
// claims warp lockstep orders that hand-off.  IT DOES NOT.  Caught red-handed in the
// FULL_ATTN2 + FA_NOSCALES + FA_EARLYV 6-tile run (trace /tmp/hsruns/nt6ver.out), cluster 1,
// global warp 1, row 61, l-reduce -- the 16 stored partials and every level's stored result are
// bit-identical across tiles, and yet:
//     tile 2 (correct): level1 buf[0]=0x403d buf[2]=0x4056 -> level2 buf[0]=0x40ca ... buf[0]=0x41e2 (28.25)
//     tile 4          : level2 buf[0]=0x4087 = 0x403d + *0x3a03*  <- buf[2] BEFORE level 1 ran
//     tile 1          : level4 buf[0]=0x41ac = 0x4147 + *0x4111*  <- buf[8] one LEVEL behind
// i.e. the load at level s+1 returned the pre-level-s value, exactly (the arithmetic closes to
// the bit).  Result: ONE row's l is a PARTIAL sum, that row of O is divided by the wrong
// denominator, and the tile scores 5.4356% / 3.7232% instead of 3.5666%.  It is a latent race in
// the SHARED softmax helper -- the unmodified sequential kernel happens not to hit the window.
//
// THE FIX: do not hand values between lanes at all.  Every lane loads all 16 partials (as 8
// 32-bit words -- buf is 4-byte aligned, warp*NT halfwords) and folds them IN REGISTERS in the
// SAME balanced binary tree order, so the result is BIT-IDENTICAL to the tree's:
//     level 1: (b0+b1) (b2+b3) ...      level 2: ((b0+b1)+(b2+b3)) ...
//     level 3: (b0..b3)+(b4..b7) ...    level 4: (b0..b7)+(b8..b15)
// One store + one fence per reduce, no further SMEM traffic, no divergence regions (the tree
// needed 4 vx_split_n/vx_join), and no store->load hand-off to race.
// ============================================================================================
// FA_FOLDX -- *** THE SAME FOLD WITH THE SUBBANK CONFLICT REMOVED, STILL BIT-EXACT. ***
// warp_fold16 as written has all 16 lanes of a warp read THE SAME EIGHT WORDS, i.e. an 8-load
// 16-WAY SMEM SUBBANK CONFLICT per reduction (the subbank is word_index & 15 and the index is
// warp*8 + k, which is lane-uniform).  That is the entire reason FA_SP_SM1F -- the same one-fence
// idea -- measured +1,502 cyc/tile instead of a win, and why FA_SP_SM1FR, which broke it with a
// (k + lane) & 15 rotation, cost 70 instructions and 4 registers and lost too.
// XOR COSTS NOTHING AND IS EXACT.  `buf` is warp*NT halfwords = warp*32 BYTES, so its low FIVE
// address bits are zero and the eight word offsets occupy exactly bits 4:2 -- hence `+` is `^`, the
// lane term folds into the base pointer ONCE, and every read is a compile-time-immediate XOR:
//     addr(k ^ r) = (base ^ (r<<2)) ^ (k<<2),     r = lane & 7.
// Over 16 lanes r takes 8 values, so each step touches 8 distinct words = 2-way instead of 16-way.
// *** WHY THE SUM STAYS BIT-IDENTICAL, which is the whole point. ***  Reading u[k] = f(word k^r)
// permutes the eight level-1 results by XOR, and XOR-by-a-constant is an AUTOMORPHISM OF THE
// DYADIC TREE: the pair {k, k^1} is a level-2 tree pair, {k,k^1} vs {k^2,k^3} is a level-3 pair,
// and so on.  So the same expression evaluated over the permuted array is the tree's expression
// with some subtrees' operands SWAPPED -- and bf16 addition is COMMUTATIVE even though it is not
// associative, so every intermediate, and the result, is bit-identical.  (Note this is exactly
// what a `(k + lane) & 15` rotation canNOT claim: a cyclic shift by an odd amount maps {0,1} to
// {1,2}, which is not a tree pair, so FA_SP_SM1FR's rotation was only legal for the max.)
// Level 1 itself is untouched: word k^r still holds the ADJACENT pair (b_{2j}, b_{2j+1}).
template <bool IS_MAX>
static inline _Float16 warp_fold16(const volatile __shared uint16_t *buf,
                                   uint32_t lane_rot = 0) {
    const volatile __shared uint32_t *w =
        reinterpret_cast<const volatile __shared uint32_t *>(buf);
    _Float16 u[8];
#ifdef FA_FOLDX
    const uint32_t wx = ((uint32_t)(uintptr_t)w) ^ ((lane_rot & 7u) << 2);
#pragma clang loop unroll(full)
    for (uint32_t k = 0; k < 8; k++) {
        const uint32_t x = *reinterpret_cast<const volatile __shared uint32_t *>(wx ^ (k << 2));
        const _Float16 a = as_bf16((uint16_t)x), b = as_bf16((uint16_t)(x >> 16));
        u[k] = IS_MAX ? fmaxf(a, b) : (_Float16)(a + b);       // == the tree's level 1
    }
#else
    (void)lane_rot;
#pragma clang loop unroll(full)
    for (uint32_t k = 0; k < 8; k++) {
        const uint32_t x = w[k];
        const _Float16 a = as_bf16((uint16_t)x), b = as_bf16((uint16_t)(x >> 16));
        u[k] = IS_MAX ? fmaxf(a, b) : (_Float16)(a + b);       // == the tree's level 1
    }
#endif
#pragma clang loop unroll(full)
    for (uint32_t s = 1; s < 8; s <<= 1)
#pragma clang loop unroll(full)
        for (uint32_t k = 0; k + s < 8; k += 2 * s)
            u[k] = IS_MAX ? fmaxf(u[k], u[k + s]) : (_Float16)(u[k] + u[k + s]);
    return u[0];
}

// multiply a bf16 value by 2^e (exact: add to exponent field).
static inline _Float16 bf16_scale_pow2(_Float16 v, int e) {
    uint16_t b = __builtin_bit_cast(uint16_t, v);
    uint32_t exp = (b >> 7) & 0xff;
    if (exp == 0 || exp == 0xff) return v;
    int ne = (int)exp + e;
    if (ne <= 0) return as_bf16((uint16_t)(b & 0x8000));
    if (ne >= 0xff) return as_bf16((uint16_t)((b & 0x8000) | 0x7f80));
    b = (uint16_t)((b & 0x807f) | ((uint32_t)ne << 7));
    return __builtin_bit_cast(_Float16, b);
}

// 32-bit word view: each lane reads consecutive bf16 pairs as words (avoids sub-word
// GMEM loads, which stall). word j (0..WPL-1) of a lane covers S cols 2*(j*NT+lane)
// and +1; with NT=16 a word spans 2 cols, and 16 words = 32 cols = one MX block, so
// word index j == MX block index j (WPL == NBLK).
// S is read from SMEM (the gemm result at SPAD_DEST, row-major bf16) -- NOT GMEM.
// Reading S back from GMEM stalled the memory system; SMEM reads are fast + safe.
template <uint32_t SQ, uint32_t SK>
static inline void softmax_requant(const __shared uint32_t *S_smem32, uint8_t *P_gmem,
                                   uint8_t *Pscales_gmem, float *l_gmem,
                                   __shared uint16_t *l_smem,
                                   uint16_t softmax_scale_bf16,
                                   uint32_t tid_in_threadblock,
                                   uint32_t threads_per_threadblock) {
    constexpr uint32_t NT = MU_NUM_THREADS;          // 16 lanes
    constexpr uint32_t WPL = SK / (2 * NT);          // S words per lane (== NBLK)
    constexpr uint32_t SKW = SK / 2;                 // words per row
    const uint32_t lane = tid_in_threadblock % NT;
    const uint32_t warp = tid_in_threadblock / NT;
    const uint32_t nwarps = threads_per_threadblock / NT;
    const _Float16 scale = as_bf16(softmax_scale_bf16);

    volatile __shared uint16_t *buf =
        reinterpret_cast<volatile __shared uint16_t *>(0x15000) + warp * NT;

    _Float16 plo[WPL], phi[WPL];
    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const __shared uint32_t *Srow = S_smem32 + row * SKW;

        // load words + scale both halves; per-lane running max
        _Float16 slo[WPL], shi[WPL];
        _Float16 mloc = as_bf16(NEG_INF_BF16_BITS);
        for (uint32_t j = 0; j < WPL; j++) {
            uint32_t w = Srow[j * NT + lane];
            _Float16 lo = (_Float16)(as_bf16((uint16_t)w) * scale);
            _Float16 hi = (_Float16)(as_bf16((uint16_t)(w >> 16)) * scale);
            slo[j] = lo; shi[j] = hi;
            mloc = fmaxf(fmaxf(lo, hi), mloc);
        }
        buf[lane] = __builtin_bit_cast(uint16_t, mloc);
        mu_fence_smem();
        warp_tree_reduce<true>(buf, lane);
        mu_fence_smem();
        _Float16 m = as_bf16(buf[0]);

        // exp(scale*S - m); per-lane running sum
        _Float16 lloc = (_Float16)0;
        for (uint32_t j = 0; j < WPL; j++) {
            _Float16 a = mu_fexp((_Float16)(slo[j] - m));
            _Float16 b = mu_fexp((_Float16)(shi[j] - m));
            plo[j] = a; phi[j] = b;
            lloc = (_Float16)(lloc + a + b);
        }
        buf[lane] = __builtin_bit_cast(uint16_t, lloc);
        mu_fence_smem();
        warp_tree_reduce<false>(buf, lane);
        mu_fence_smem();
        if (lane == 0) {
            _Float16 lval = as_bf16(buf[0]);
            l_smem[row] = buf[0];        // bf16 l in SMEM (for normalize, no GMEM load)
            l_gmem[row] = (float)lval;   // fp32 l in GMEM (for offline verification)
        }

        // requant: word j is MX block j (cols 2*(j*NT+lane), +1).
        for (uint32_t j = 0; j < WPL; j++) {
            _Float16 alo = (plo[j] < (_Float16)0) ? (_Float16)(-plo[j]) : plo[j];
            _Float16 ahi = (phi[j] < (_Float16)0) ? (_Float16)(-phi[j]) : phi[j];
            buf[lane] = __builtin_bit_cast(uint16_t, (_Float16)fmaxf(alo, ahi));
            mu_fence_smem();
            warp_tree_reduce<true>(buf, lane);
            mu_fence_smem();
            int se = bf16_floor_log2(buf[0]);          // E8M0 scale exponent (target 0)
            if (lane == 0) Pscales_gmem[j * SQ + row] = (uint8_t)(se + 127);
            uint32_t col = 2 * (j * NT + lane);
            P_gmem[row * SK + col] =
                bf16_to_e4m3(__builtin_bit_cast(uint16_t, bf16_scale_pow2(plo[j], -se)));
            P_gmem[row * SK + col + 1] =
                bf16_to_e4m3(__builtin_bit_cast(uint16_t, bf16_scale_pow2(phi[j], -se)));
        }
    }
}

// SIMT softmax variant for the HW-requantizer path: reads S from SMEM, writes P as
// bf16 (packed 2/word -> no sub-word stores) into an SMEM scratch P_smem32, and l to
// SMEM(bf16)+GMEM(fp32). The MX-FP8 requant of P is done afterward by the HW
// requantizer (not in SIMT). Same softmax math as softmax_requant.
template <uint32_t SQ, uint32_t SK>
// P is written to GMEM (packed bf16 words) -- a PRE-requant SIMT store, so not affected
// by the post-requant corruption. The requantizer must then be fed from GMEM (NOT from
// SMEM, which corrupts subsequent SIMT); feeding from GMEM keeps SIMT healthy.
static __attribute__((noinline)) void softmax_to_smem(const __shared uint32_t *S_smem32,
                                   __shared uint32_t *P_gmem32,
                                   float *l_gmem, __shared uint16_t *l_smem,
                                   uint16_t softmax_scale_bf16,
                                   uint32_t tid_in_threadblock,
                                   uint32_t threads_per_threadblock) {
    constexpr uint32_t NT = MU_NUM_THREADS;
    constexpr uint32_t WPL = SK / (2 * NT);
    constexpr uint32_t SKW = SK / 2;
    const uint32_t lane = tid_in_threadblock % NT;
    const uint32_t warp = tid_in_threadblock / NT;
    const uint32_t nwarps = threads_per_threadblock / NT;
    const _Float16 scale = as_bf16(softmax_scale_bf16);
    volatile __shared uint16_t *buf =
        reinterpret_cast<volatile __shared uint16_t *>(0x15000) + warp * NT;

    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const __shared uint32_t *Srow = S_smem32 + row * SKW;
        __shared uint32_t *Prow = P_gmem32 + row * SKW;
        _Float16 slo[WPL], shi[WPL];
        _Float16 mloc = as_bf16(NEG_INF_BF16_BITS);
        for (uint32_t j = 0; j < WPL; j++) {
            uint32_t w = Srow[j * NT + lane];
            _Float16 lo = (_Float16)(as_bf16((uint16_t)w) * scale);
            _Float16 hi = (_Float16)(as_bf16((uint16_t)(w >> 16)) * scale);
            slo[j] = lo; shi[j] = hi;
            mloc = fmaxf(fmaxf(lo, hi), mloc);
        }
        buf[lane] = __builtin_bit_cast(uint16_t, mloc);
        mu_fence_smem(); warp_tree_reduce<true>(buf, lane); mu_fence_smem();
        _Float16 m = as_bf16(buf[0]);
        _Float16 lloc = (_Float16)0;
        for (uint32_t j = 0; j < WPL; j++) {        // exp, kept in slo/shi until l known
            _Float16 a = mu_fexp((_Float16)(slo[j] - m));
            _Float16 b = mu_fexp((_Float16)(shi[j] - m));
            slo[j] = a; shi[j] = b;
            lloc = (_Float16)(lloc + a + b);
        }
        buf[lane] = __builtin_bit_cast(uint16_t, lloc);
        mu_fence_smem(); warp_tree_reduce<false>(buf, lane); mu_fence_smem();
        // Fold the softmax 1/l normalization in HERE so the PV gemm produces the FINAL
        // O directly -- no SIMT after the requant write (which corrupts SIMT state).
        _Float16 inv_l = (_Float16)(__builtin_bit_cast(_Float16, ONE_BF16_BITS) / as_bf16(buf[0]));
        for (uint32_t j = 0; j < WPL; j++)
            Prow[j * NT + lane] = pack_bf16x2((_Float16)(slo[j] * inv_l),
                                              (_Float16)(shi[j] * inv_l));
        if (lane == 0) { l_smem[row] = buf[0]; l_gmem[row] = (float)as_bf16(buf[0]); }
    }
}

// ===== Streaming (flash) online-softmax helpers =====
// FA_SP_RBPAD -- pad the PER-WARP stride of the cross-lane reduce buffer so the six warps stop
// colliding with each other.  buf sits at 0x15000 + warp*NT halfwords = warp*32 BYTES = warp*8
// WORDS, and the subbank is word_index & 15, so warps 0/2/4 all land on subbank 0 and warps 1/3/5
// all land on subbank 8 -- a 3-way INTER-warp conflict layered on top of the 16-way INTRA-warp one
// (all 16 lanes of a warp read buf[0] to broadcast each reduction's result).  A stride of NT+2 = 18
// halfwords = 9 words puts warps 0..5 on subbanks 0, 9, 2, 11, 4, 13: six distinct.  The footprint
// grows from 192 B to 216 B, still inside the scratch window, and it is an ADDRESS change only, so
// it is bit-exact and costs zero registers and zero instructions.  (It does NOT touch the intra-warp
// broadcast conflict -- that needs FA_FOLDX or FA_SM_2P.)
#ifdef FA_SP_RBPAD
#define FA_RB_STRIDE (MU_NUM_THREADS + 2u)
#else
#define FA_RB_STRIDE (MU_NUM_THREADS)
#endif
// online_softmax_block: for one key-block S_j [SQ][BK] (bf16 in SMEM), update the running
// per-row max m and denom l, emit corr = exp(m_old - m_new) for rescaling the O
// accumulator, and write the UNNORMALIZED probs P_j = exp(S_j*scale - m_new) (bf16, in
// (0,1]) to P_smem. 1/l is deferred to finalize_O. Mirrors FA.mx_attention_flash.
// One row per warp; 16 lanes reduce cooperatively (same strided scheme as softmax_to_smem).
template <uint32_t SQ, uint32_t BK>
static __attribute__((noinline)) void online_softmax_block(
        const __shared uint32_t *S_smem32, __shared uint32_t *P_smem32,
        __shared uint16_t *m_state, __shared uint16_t *l_state, __shared uint16_t *corr_out,
        uint16_t softmax_scale_bf16, uint32_t first_block,
        uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
    constexpr uint32_t NT = MU_NUM_THREADS;
    constexpr uint32_t WPL = BK / (2 * NT);
    constexpr uint32_t BKW = BK / 2;
    const uint32_t lane = tid_in_threadblock % NT;
    const uint32_t warp = tid_in_threadblock / NT;
    const uint32_t nwarps = threads_per_threadblock / NT;
    const _Float16 scale = as_bf16(softmax_scale_bf16);
    volatile __shared uint16_t *buf =
        reinterpret_cast<volatile __shared uint16_t *>(0x15000) + warp * FA_RB_STRIDE;

#ifdef FA_SM_2P
    // ==========================================================================================
    // FA_SM_2P -- TWO-PASS SOFTMAX WITH *WARP-BATCHED* CROSS-LANE REDUCTIONS.
    //
    // WHERE THE REFERENCE SPENDS ITS 12,825 cyc/tile.  online_softmax_block is 243 static
    // instructions per row with FOUR fence.s and TEN warp-divergence regions, and at FA_SP_QSPLIT
    // all six warps run it, so each core's issue port sees 32 rows x 243 = 7,776 warp-instructions
    // -- 61% of the measured stage.  The other ~5,000 cycles are the fences: 11 rows x 4 = 44 per
    // warp, and because all three warps on a core are executing the SAME code they hit them
    // together, so the port genuinely idles (~115 cyc/fence).  Both terms are reduction
    // scaffolding: per row the two warp_tree_reduce calls cost 16 lh + 13 sh + 8 ops + 8
    // vx_split_n/vx_join + 4 fence.s, i.e. ~76 of the 243 instructions AND all four fences.
    //
    // THE FIX: the reduction is per ROW, but the FENCE does not have to be.  Split the row loop
    // into two passes over the warp's ~11 rows and do ALL of that warp's reductions between them:
    //   pass A  : per-lane partial row max of RNE(S*scale)      -> pb[row][lane]      (no fence)
    //   fold A  : lane L folds row (warp + nwarps*L)'s 16 partials -> m_state[row]
    //   pass B  : a = exp(RNE(S*scale) - m), P in place over S, per-lane partial l -> pb[row][lane]
    //   fold B  : lane L folds row (warp + nwarps*L)'s 16 partials -> l_state[row]
    // THREE fence.s PER WARP instead of four per row -- 44 -> 3, measured in the disassembly -- and
    // zero divergence regions in the passes (fold B needs no trailing fence: every lane computes
    // the whole reduction itself, so there is no buf[0] to publish).  It costs pass B a RE-READ of S (8 lw + 16 extract + 16 fmul per row), which is
    // what pays for deleting the slo[8]/shi[8] register arrays: sixteen live fp registers vanish,
    // and the renamer budget is the binding constraint in this kernel (53 distinct arch regs for
    // the whole kernel -- see the header of kernel.cpp).  That is why this succeeds
    // where FA_SP_SM1FX -- which keeps the arrays and adds 15 loop-invariant XOR'd pointers -- is
    // register-fatal at 58.
    //
    // *** EVERY SYNCHRONISATION HERE IS INTRA-WARP, WHICH IS WHY mu_fence_smem() IS SUFFICIENT
    // AND NO BARRIER IS NEEDED. ***  Rows are striped row -> warp (row % nwarps), so warp w owns
    // exactly the rows {w, w+nwarps, ...}; lane L of warp w folds row w + nwarps*L, which is one
    // of THAT SAME WARP's rows.  So pb, m_state and l_state are each written and read by one warp
    // only, and fence.s -- which drains that warp's own LSU queues -- is exactly the right
    // primitive.  (mu_fence_smem is NOT a drain for a gemmini DMA or an SF-SRAM write; it is a
    // drain for this.)
    //
    // *** BIT-EXACT vs THE REFERENCE, term by term. ***
    //   m : max over the row of RNE(S*scale).  The reference accumulates one chain per lane and
    //       then a 16-leaf tree; this uses four chains and then a linear 16-fold.  fmax is exact
    //       and fully order-independent, so every grouping gives the identical bf16.
    //   a : mu_fexp(RNE(RNE(S*scale) - m)) with the same m -- the same two roundings on the same
    //       operands, recomputed from the same S word instead of read from a register.
    //   l : the per-lane chain is the reference's own `lloc = (_Float16)(lloc + a + b)` in the same
    //       j order, and fold B reproduces warp_tree_reduce's EXACT 16-leaf balanced pairing
    //         (((b0+b1)+(b2+b3)) + ((b4+b5)+(b6+b7))) + (((b8+b9)+(b10+b11))+((b12+b13)+(b14+b15)))
    //       term for term.  This matters: bf16 addition is NOT associative and every node rounds,
    //       so a linear fold here would NOT be bit-exact (which is why fold A may be linear and
    //       fold B may not).
    //   l_state: the reference stores (_Float16)(l_old*corr + lsum) with first_block=1, i.e.
    //       l_old = 0 and corr = exp(0) = 1, which is RNE(0 + lsum) == lsum.  FA_SM_2P is
    //       therefore FIRST-BLOCK ONLY (all FA_SP ever uses) and drops the dead corr/rescale
    //       arithmetic; corr_out is not written, so FA_SP_DUMPL/DUMPLM's corr column is invalid
    //       under this flag.
    //
    // SCRATCH: pb needs SQ x 17 halfwords = 2,176 B at 0x15600, inside the 0x14000..0x16000
    // scratch window (REDBUF 0x15000 ends at 0x150C0; LPART 0x15400 is FUSE-only and CBMAX_SMEM
    // 0x15800 is CBMAX-only, both mutually exclusive with this flag).  The stride is 2*NT+1 = 17,
    // ODD ON PURPOSE: fold A/B have lane L read row (w+nwarps*L), whose halfword base is
    // 17*(w+nwarps*L), so with an even stride every lane's word index would differ by a multiple
    // of 8 and the 16 lanes would land on only TWO of the 16 word-subbanks (an 8-way conflict on
    // every one of the 16 fold loads).  17 makes (6L*17)>>1 mod 16 take 11 distinct values over
    // the 11 active lanes -- essentially conflict-free -- for the price of one halfword of padding
    // per row and no instructions at all.  (Same reasoning as FA_SP_CVTX/FA_SP_PAX, but solved by
    // LAYOUT rather than by an XOR on every access, so it costs nothing in registers.)
    // *** AND ONE COMPILER TRAP THAT SILENTLY BREAKS THE BIT-EXACTNESS ABOVE, CAUGHT IN THE
    // DISASSEMBLY AND NOT IN A SIMULATION. ***  Written as `(_Float16)(as_bf16(w) * scale) - m`,
    // clang CONTRACTS the multiply and the subtract into ONE fmsub.h -- sixteen of them per row --
    // which computes S*scale - m with a SINGLE rounding.  The reference emits a separate fmul.h and
    // fsub.h, i.e. TWO roundings, RNE(RNE(S*scale) - m); the fused form is therefore a DIFFERENT
    // number and the whole bit-exactness argument above collapses.  Neither the cast to _Float16
    // nor a named local stops it.  fa_sm2p_mul forces the rounding with inline asm, and an asm
    // result cannot be folded into a later contraction.  (Same class of trap as hardware fact 2 in
    // kernel.cpp: clang reaches for fp32 -- or for contraction -- on its own, and the
    // only reliable fix is to name the instruction.)
    {
    // *** STRIDE BUG, FOUND BY REVIEW AND NOT BY A SIMULATION -- READ THIS BEFORE TOUCHING IT. ***
    // This was written `2 * NT + 1`, which is 33, not the 17 the comment above claims: a row needs
    // NT=16 halfwords plus ONE of padding to make the stride odd, i.e. NT + 1.  At 33 the buffer is
    // 64*33*2 = 4,224 B from 0x15680 and its last write lands at 0x166DC -- 1,756 BYTES INSIDE Q's
    // SPAD (SP_Q = 0x16000), i.e. it corrupted Q rows 0..13 on every tile.
    // IT PASSED 12 of 12 ANYWAY, and that is the whole lesson: Q(t+1)'s move-in DMA is issued in
    // stage S4, AFTER this stage's barrier, so it happened to rewrite exactly the rows this
    // overran before the QK in stage S6 could read them.  A correctness result that depends on a
    // later DMA coincidentally clobbering your out-of-bounds writes is not a correctness result --
    // it is the same class of accident as FA_SP_QSPLIT's schedule-dependent 12 of 12 documented
    // above, with a longer fuse.  *** AND FA_SP_QEARLY LIGHTS IT: *** that flag moves the Q DMA to
    // the TOP of this stage so it runs CONCURRENTLY with the softmax, after which the overrun is no
    // longer covered by anything and Q really is corrupt when QK reads it.
    // The static_asserts below make the bound a compile-time property instead of a coincidence.
    constexpr uint32_t STR = NT + 1;                     // 17 halfwords per row (odd: see above)
    constexpr uint32_t PB_BASE = 0x15680;
    static_assert(PB_BASE + SQ * STR * sizeof(uint16_t) <= 0x16000,
                  "FA_SM_2P: pb overruns SP_Q (the Q operand spad at 0x16000)");
    __shared uint16_t *const pb = reinterpret_cast<__shared uint16_t *>(PB_BASE);
#ifdef FA_SM_2PBM
    // FA_SM_2PBM block-max partials: 8 blocks x 17 halfwords per warp = 136, x6 warps = 1,632 B at
    // REDBUF_SMEM (0x15000), which FA_SM_2P no longer uses for anything (it returns before `buf`).
    // 17 again: the fold has lane L read block (L & 7), and with an even stride all 16 lanes would
    // land on 2 of the 16 word-subbanks.
    // NT + 1, for the same reason and with the same bug history as STR above: at 2*NT+1 this buffer
    // is 6*8*33*2 = 3,168 B from 0x15000 and reaches 0x15C60, which COLLIDES WITH pb at 0x15680 --
    // a second, independent out-of-bounds that only FA_SM_2PBM activates.
    //
    // *** NOTE FOR ANYONE HOLDING A MEASUREMENT OF FA_SM_2PBM FROM BEFORE fd722ac: THIS IS ALMOST
    // CERTAINLY WHY IT WAS WRONG, AND IT IS NOT THE BLOCK-MAX ALGORITHM. ***  bb holds the per-lane
    // block maxima and pb holds the l/m partials, so before the fix the two scribbled over each
    // other on every row -- and the observed signature of that is precisely "the stack WITH
    // FA_SM_2PBM + FA_SP_SMBMAX is wrong on every tile-image, the same stack WITHOUT it is clean",
    // which is what a concurrent pass measured and reasonably attributed to the block-max path.
    // The block-max path had never been run without the collision.  Any build predating fd722ac
    // carries BOTH overruns (pb into SP_Q for every FA_SM_2P build, plus this one for 2PBM), so
    // those tile verdicts want re-running rather than believing; the cycle counts are unaffected,
    // since pb and bb are pure scratch and nothing downstream reads them after the fold.
    constexpr uint32_t BSTR = NT + 1;
    constexpr uint32_t BB_BASE = 0x15000, BB_WARPS = 6;
    static_assert(BB_BASE + BB_WARPS * WPL * BSTR * sizeof(uint16_t) <= PB_BASE,
                  "FA_SM_2PBM: bb collides with FA_SM_2P's pb buffer");
    __shared uint16_t *const bb =
        reinterpret_cast<__shared uint16_t *>(BB_BASE) + warp * (WPL * BSTR);
    // SCALE_SMEM (0x14000): one 32-bit word per (block, row), the layout fa_requant_cvt reads.
    // Hardcoded for the same reason 0x15000 is: online_softmax_block has no scale_scratch argument
    // and this flag is FA_SP-only.  Build it together with -DFA_SP_SMBMAX, which is what compiles
    // the requant pass-A STAGE out of the FA_SP body (that branch is keyed on FA_SP_SMBMAX).
    __shared uint32_t *const scale_scratch2p = reinterpret_cast<__shared uint32_t *>(0x14000);
#endif
    const uint32_t fr = warp + nwarps * lane;            // the row THIS lane folds (may be >= SQ)

    // ---- pass A: per-lane partial row max.  Four chains for memory-level parallelism (max is
    // ---- order-free, so any grouping is exact); no reduction, no fence.
#ifdef FA_SM_2PRAW
    // FA_SM_2PRAW -- MAX THE *RAW* ROW AND SCALE ONCE, deleting 16 fmul.h per row (~500 cyc/tile).
    // x |-> RNE(x*scale) is MONOTONE NON-DECREASING for scale > 0 (bf16 multiply and
    // round-to-nearest-even both are), so
    //     max_i RNE(S_i * scale)  ==  RNE( (max_i S_i) * scale )
    // and the single product at the end of fold A is the identical bf16.  This is the one place
    // fa_softmax_tpr's shape is strictly better than the reference's, and it is free.
    // VERIFIED NUMERICALLY ON THE REAL INPUT, not just argued: emulating bf16 RNE over
    // golden_S_u16 gives byte-identical m for the two forms on all 64 rows.
    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const __shared uint32_t *Srow = S_smem32 + row * BKW;
        // FOUR independent chains, one per load, so the four loads of an iteration have no
        // dependence between them.  This was cut to two to chase fa_regs.py's whole-file union
        // under a "53-register cliff" that fa_regs3.py has since shown does not exist (this stack
        // measures 165/255 on the binding core, ~30 spare arch regs) -- i.e. it was free
        // memory-level parallelism given away for nothing.  Restored.  Bit-exact either way: fmax
        // is exact and fully order-independent, so any grouping gives the identical bf16.
        _Float16 x0 = as_bf16(NEG_INF_BF16_BITS), x1 = x0, x2 = x0, x3 = x0;
        for (uint32_t j = 0; j < WPL; j += 4) {
            const uint32_t w0 = Srow[(j + 0) * NT + lane], w1 = Srow[(j + 1) * NT + lane],
                           w2 = Srow[(j + 2) * NT + lane], w3 = Srow[(j + 3) * NT + lane];
            x0 = fmaxf(fmaxf(as_bf16((uint16_t)w0), as_bf16((uint16_t)(w0 >> 16))), x0);
            x1 = fmaxf(fmaxf(as_bf16((uint16_t)w1), as_bf16((uint16_t)(w1 >> 16))), x1);
            x2 = fmaxf(fmaxf(as_bf16((uint16_t)w2), as_bf16((uint16_t)(w2 >> 16))), x2);
            x3 = fmaxf(fmaxf(as_bf16((uint16_t)w3), as_bf16((uint16_t)(w3 >> 16))), x3);
        }
        pb[row * STR + lane] =
            __builtin_bit_cast(uint16_t, (_Float16)fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)));
    }
#else
    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const __shared uint32_t *Srow = S_smem32 + row * BKW;
        _Float16 x0 = as_bf16(NEG_INF_BF16_BITS), x1 = x0, x2 = x0, x3 = x0;
        for (uint32_t j = 0; j < WPL; j += 4) {
            const uint32_t w0 = Srow[(j + 0) * NT + lane], w1 = Srow[(j + 1) * NT + lane],
                           w2 = Srow[(j + 2) * NT + lane], w3 = Srow[(j + 3) * NT + lane];
            x0 = fmaxf(fmaxf((_Float16)(as_bf16((uint16_t)w0) * scale),
                             (_Float16)(as_bf16((uint16_t)(w0 >> 16)) * scale)), x0);
            x1 = fmaxf(fmaxf((_Float16)(as_bf16((uint16_t)w1) * scale),
                             (_Float16)(as_bf16((uint16_t)(w1 >> 16)) * scale)), x1);
            x2 = fmaxf(fmaxf((_Float16)(as_bf16((uint16_t)w2) * scale),
                             (_Float16)(as_bf16((uint16_t)(w2 >> 16)) * scale)), x2);
            x3 = fmaxf(fmaxf((_Float16)(as_bf16((uint16_t)w3) * scale),
                             (_Float16)(as_bf16((uint16_t)(w3 >> 16)) * scale)), x3);
        }
        pb[row * STR + lane] = __builtin_bit_cast(
            uint16_t, (_Float16)fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)));
    }
#endif
    mu_fence_smem();                                     // fence #1 of 4 PER WARP
    // ---- fold A: one row per lane, linear (max is order-free) -> m_state[row] ----
    if (fr < SQ) {
        const __shared uint16_t *bp = pb + fr * STR;
        _Float16 q0 = as_bf16(bp[0]), q1 = as_bf16(bp[1]);   // 2 chains (this runs once per warp)
        for (uint32_t k = 2; k < NT; k += 2) {
            q0 = fmaxf(q0, as_bf16(bp[k + 0])); q1 = fmaxf(q1, as_bf16(bp[k + 1]));
        }
#ifdef FA_SM_2PRAW
        m_state[fr] = __builtin_bit_cast(                 // ONE product, at the end (see above)
            uint16_t, fa_sm2p_mul((_Float16)fmaxf(q0, q1), scale));
#else
        m_state[fr] = __builtin_bit_cast(uint16_t, (_Float16)fmaxf(q0, q1));
#endif
    }
    mu_fence_smem();                                     // fence #2
    // ---- pass B: P = exp(RNE(S*scale) - m) in place over S; per-lane partial l ----
    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const __shared uint32_t *Srow = S_smem32 + row * BKW;
        __shared uint32_t *Prow = P_smem32 + row * BKW;
        const _Float16 m = as_bf16(m_state[row]);        // one broadcast halfword read per row
        _Float16 lloc = (_Float16)0;
        for (uint32_t j = 0; j < WPL; j++) {
            const uint32_t w = Srow[j * NT + lane];
            const _Float16 a = mu_fexp(
                (_Float16)(fa_sm2p_mul(as_bf16((uint16_t)w), scale) - m));
            const _Float16 b = mu_fexp(
                (_Float16)(fa_sm2p_mul(as_bf16((uint16_t)(w >> 16)), scale) - m));
            Prow[j * NT + lane] = pack_bf16x2(a, b);
            lloc = (_Float16)(lloc + a + b);             // the reference's exact chain and order
#ifdef FA_SM_2PBM
            // This lane's contribution to MX block j's max.  *** THE LAYOUT MAKES THIS EXACT AND
            // ALMOST FREE: *** lane L owns word j*NT+L, i.e. row elements 2*(j*NT+L) and +1, so its
            // MX block index is (j*NT+L)/NT == j and its two elements are positions 2L, 2L+1 WITHIN
            // block j.  Block j's 32 elements are therefore exactly the 16 lanes' word j, and the
            // block max is one 16-lane reduction of a register each lane already holds.
            bb[j * BSTR + lane] = __builtin_bit_cast(uint16_t, (_Float16)fmaxf(a, b));
#endif
        }
        pb[row * STR + lane] = __builtin_bit_cast(uint16_t, lloc);
#ifdef FA_SM_2PBM
        // ==== FA_SM_2PBM -- PRODUCE THE E8M0 BLOCK SCALES HERE AND DELETE REQUANT PASS A. =========
        // Pass A of the requant (fa_requant_max, 3,405 cyc/tile; 2,754 with FA_SP_PAX) exists ONLY
        // to RE-READ the bf16 P this loop just wrote and take a 32-element max of it.  Pass B above
        // has those values in registers, so the max costs 8 fmax.h + 8 halfword stores per row plus
        // ONE 16-lane fold -- and the fold is affordable because fence.s turned out to cost only
        // ~22 cycles here (measured: FA_SM_2P removes 41 of a warp's 44 fences and only ~890 of the
        // 1,742-cycle win is attributable to them), so a fence PER ROW is fine and the buffer stays
        // 256 B per warp instead of the 16 KB a fully warp-batched fold would need.
        //
        // WHY IT IS AT *REFERENCE* NUMERICS, WHICH IS THE WHOLE POINT.  FA_SP_SMBMAX does the same
        // trick but only inside the THREAD-PER-ROW softmax, so it can only be had at that softmax's
        // 4.2% numerics; FA_SP_CBMAX did it cooperatively but cost +80 instr/row and was a renamer
        // casualty at 33 registers in an all-warps function.  Here the values folded are the SAME 32
        // bf16 P values fa_requant_max would have re-read, max is exact and order-independent, and
        // the code written is byte-identical -- em/K8/max(em,7) and the scale_scratch[b*SQ+row]
        // layout are copied from fa_requant_max verbatim.  So O stays BIT-EXACT.
        //
        // NO DIVERGENCE REGION: all 16 lanes fold block (lane & 7), so lanes L and L+8 compute the
        // same block and store the same byte to the same address -- a benign duplicate store, which
        // is cheaper than an `if (lane < WPL)` split/join.  (Requires WPL == 8, static_assert'd.)
        static_assert(WPL == 8, "FA_SM_2PBM's lane&7 block fold assumes 8 MX blocks per row");
        mu_fence_smem();
        {
            const uint32_t bsel = lane & (WPL - 1u);
            const __shared uint16_t *bp = bb + bsel * BSTR;
            // TWO chains (was cut to one for the same non-existent register cliff; see the
            // retraction at the top of this file).  Order-free, so bit-exact either way.
            _Float16 y0 = as_bf16(bp[0]), y1 = as_bf16(bp[1]);
            for (uint32_t k = 2; k < NT; k += 2) {
                y0 = fmaxf(y0, as_bf16(bp[k + 0])); y1 = fmaxf(y1, as_bf16(bp[k + 1]));
            }
            const uint32_t em =
                ((uint32_t)__builtin_bit_cast(uint16_t, (_Float16)fmaxf(y0, y1)) >> 7) & 0xffu;
            const uint32_t K8 = fa_clamp_K8(((int)em - 7) << 3);
            scale_scratch2p[bsel * SQ + row] = ((K8 - 8u) >> 3) + 7u;              // max(em,7)
        }
        // *** SECOND FENCE, AND WITHOUT IT FA_SM_2PBM IS WRONG: bb IS REUSED EVERY ROW. ***
        // The fold above has lane L read bb[bsel*BSTR + k] for k = 0..15 -- sixteen halfwords that
        // OTHER LANES wrote.  The next iteration of this row loop then has those same other lanes
        // OVERWRITE exactly those halfwords for the next row.  So there is a CROSS-LANE
        // write-after-read between consecutive rows, and the fence before the fold only covers the
        // read-after-write.  Without a fence here, row r+1's stores can be serviced before row r's
        // fold loads are, and row r gets block maxima belonging to row r+1 -- which is a WRONG E8M0
        // exponent for that (row, block), i.e. a whole 32-element block quantised against the wrong
        // scale.  MEASURED before this fence: 4 correct, 7 WRONG of 11 tile-images at FA_NT6, with a
        // 50,283-cycle mean whose spread (48,954..52,088) is the corrupt-timing signature.
        // The alternative -- double-buffering bb by row parity -- does not fit: 2 x 1,632 B from
        // 0x15000 reaches 0x15CC0 and would collide with pb at 0x15680, which is the bug fixed in
        // fd722ac.  So it is a fence, at ~22 cycles x 11 rows = ~242 cyc/warp against the 2,754 that
        // deleting requant pass A saves.
        mu_fence_smem();
#endif
    }
#ifndef FA_SM_2PBM
    mu_fence_smem();                                     // fence #3
#endif
    // ---- fold B: warp_tree_reduce's EXACT 16-leaf balanced pairing (bf16 add is not
    // ---- associative, so the shape is load-bearing) -> l_state[row] ----
    if (fr < SQ) {
        const __shared uint16_t *bp = pb + fr * STR;
        _Float16 t0 = (_Float16)((_Float16)(as_bf16(bp[0]) + as_bf16(bp[1]))
                               + (_Float16)(as_bf16(bp[2]) + as_bf16(bp[3])));
        _Float16 t1 = (_Float16)((_Float16)(as_bf16(bp[4]) + as_bf16(bp[5]))
                               + (_Float16)(as_bf16(bp[6]) + as_bf16(bp[7])));
        const _Float16 hA = (_Float16)(t0 + t1);
        t0 = (_Float16)((_Float16)(as_bf16(bp[8]) + as_bf16(bp[9]))
                      + (_Float16)(as_bf16(bp[10]) + as_bf16(bp[11])));
        t1 = (_Float16)((_Float16)(as_bf16(bp[12]) + as_bf16(bp[13]))
                      + (_Float16)(as_bf16(bp[14]) + as_bf16(bp[15])));
        l_state[fr] = __builtin_bit_cast(uint16_t, (_Float16)(hA + (_Float16)(t0 + t1)));
    }
    (void)corr_out; (void)first_block; (void)buf;
    return;
    }
#endif  // FA_SM_2P

    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const __shared uint32_t *Srow = S_smem32 + row * BKW;
        __shared uint32_t *Prow = P_smem32 + row * BKW;
        _Float16 slo[WPL], shi[WPL];
        _Float16 mloc = as_bf16(NEG_INF_BF16_BITS);
        for (uint32_t j = 0; j < WPL; j++) {
            uint32_t w = Srow[j * NT + lane];
            _Float16 lo = (_Float16)(as_bf16((uint16_t)w) * scale);
            _Float16 hi = (_Float16)(as_bf16((uint16_t)(w >> 16)) * scale);
            slo[j] = lo; shi[j] = hi;
            mloc = fmaxf(fmaxf(lo, hi), mloc);
        }
#ifdef FA_SM_FAST
        _Float16 bmax = warp_butterfly_reduce<true>(buf, lane, mloc);
#elif defined(FA_TREEFIX)
        buf[lane] = __builtin_bit_cast(uint16_t, mloc);
        mu_fence_smem();
        _Float16 bmax = warp_fold16<true>(buf, lane);           // race-free, bit-identical
#else
        buf[lane] = __builtin_bit_cast(uint16_t, mloc);
        mu_fence_smem(); warp_tree_reduce<true>(buf, lane); mu_fence_smem();
        _Float16 bmax = as_bf16(buf[0]);                       // block max (scaled)
#endif
        _Float16 m_old = first_block ? bmax : as_bf16(m_state[row]);
        _Float16 m_new = fmaxf(m_old, bmax);
        _Float16 corr = mu_fexp((_Float16)(m_old - m_new));    // first-block -> exp(0)=1
        _Float16 lloc = (_Float16)0;
        for (uint32_t j = 0; j < WPL; j++) {
            _Float16 a = mu_fexp((_Float16)(slo[j] - m_new));
            _Float16 b = mu_fexp((_Float16)(shi[j] - m_new));
            slo[j] = a; shi[j] = b;
            lloc = (_Float16)(lloc + a + b);
        }
#ifdef FA_SM_FAST
        _Float16 lsum = warp_butterfly_reduce<false>(buf, lane, lloc);
#elif defined(FA_TREEFIX)
        buf[lane] = __builtin_bit_cast(uint16_t, lloc);
        mu_fence_smem();
        _Float16 lsum = warp_fold16<false>(buf, lane);          // race-free, bit-identical
#else
        buf[lane] = __builtin_bit_cast(uint16_t, lloc);
        mu_fence_smem(); warp_tree_reduce<false>(buf, lane); mu_fence_smem();
        _Float16 lsum = as_bf16(buf[0]);
#endif
#ifdef FA_PSWIZ
        // ---- TRANSPOSED P scratch layout (see requant_P_to_spad_tiled) ------
        // Word j of lane `lane` is MX block b=j, element-word k=lane of row `row`
        // (strided ownership => word index j*16+lane spans cols 2*(j*16+lane),+1,
        //  so block = (j*16+lane)/16 = j and k = lane).  We store it at
        //    item  = b*SQ + row          (b-MAJOR items)
        //    W     = (item>>4)*256 + k*16 + ((item&15) [^ k])
        // so that requant, whose 16 lanes own 16 CONSECUTIVE items, reads 16
        // consecutive words (one 64B line / 16 distinct word-subbanks) per load
        // instead of 16 different lines.  Here group=(item>>4)=j*4+row/16 and
        // il=(item&15)=row&15, so the address is base + j*1024: still a pure
        // constant stride, no extra address arithmetic in the softmax loop.
        {
            static_assert(WPL == BK / 32, "FA_PSWIZ needs 1 word-per-lane per MX block");
            const uint32_t il = row & 15u;
#ifdef FA_PSWIZX
            const uint32_t sb = il ^ lane;   // XOR swizzle: makes the 16 lanes of
#else                                        // THIS store hit 16 distinct subbanks
            const uint32_t sb = il;          // too (costs requant 1 xori/load).
#endif
            __shared uint32_t *Pb = P_smem32 + (row >> 4) * 256u + lane * 16u + sb;
            for (uint32_t j = 0; j < WPL; j++)                 // P_j UNNORMALIZED
                Pb[j * (16u * SQ)] = pack_bf16x2(slo[j], shi[j]);
        }
        (void)Prow;
#else
        for (uint32_t j = 0; j < WPL; j++)                     // P_j UNNORMALIZED
            Prow[j * NT + lane] = pack_bf16x2(slo[j], shi[j]);
#endif
        if (lane == 0) {
            _Float16 l_old = first_block ? (_Float16)0 : as_bf16(l_state[row]);
            l_state[row] = __builtin_bit_cast(uint16_t, (_Float16)(l_old * corr + lsum));
            m_state[row] = __builtin_bit_cast(uint16_t, m_new);
            corr_out[row] = __builtin_bit_cast(uint16_t, corr);
        }
    }
}

// THREAD-PER-ROW fused softmax + requant. Each lane owns a WHOLE row (grid-strided by
// the total lane count), so ALL reductions (row max, row sum, per-block max) are done in
// registers -- NO cross-lane communication, hence NO per-row mu_fence_smem. This kills the
// dominant cost of the cooperative version: ~4 fence.s/row that each drained the SMEM store
// queue under 96-lane contention (~1000+ cyc each => ~60% of total kernel cycles). S is
// re-read from SMEM per pass (cheap) to keep the register footprint (F) small. The caller
// issues ONE mu_fence_smem after this returns to publish P/scales to the PV mesh.
// Same math + same output layout as the cooperative version (verified identical results).
template <uint32_t SQ, uint32_t BK>
static __attribute__((noinline)) void fused_softmax_requant_tpr(
        const __shared uint16_t *S_smem16, __shared uint32_t *spad_u32,
        __shared uint32_t *scale_scratch,
        __shared uint16_t *m_state, __shared uint16_t *l_state, __shared uint16_t *corr_out,
        uint16_t softmax_scale_bf16, uint32_t first_block,
        uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
    constexpr uint32_t NBLK = BK / 32;            // MX blocks per row
    constexpr uint32_t PE_TILES_K = BK / 16;
    const uint32_t glane = tid_in_threadblock;    // this lane's global index
    const uint32_t nlanes = threads_per_threadblock;
    const _Float16 scale = as_bf16(softmax_scale_bf16);

    for (uint32_t row = glane; row < SQ; row += nlanes) {
        const __shared uint16_t *Srow = S_smem16 + row * BK;
        // Pass A: row max of scaled S (registers only).
        _Float16 rmax = as_bf16(NEG_INF_BF16_BITS);
        for (uint32_t c = 0; c < BK; c++)
            rmax = fmaxf(rmax, (_Float16)(as_bf16(Srow[c]) * scale));
        const _Float16 m_old = first_block ? rmax : as_bf16(m_state[row]);
        const _Float16 m_new = fmaxf(m_old, rmax);
        const _Float16 corr = mu_fexp((_Float16)(m_old - m_new));

        // Pass B: per 32-col MX block -> block max (E8M0 scale) then requant; accumulate rowsum.
        _Float16 rowsum = (_Float16)0;
        const uint32_t ti = row / 16, rr = row % 16;
        for (uint32_t b = 0; b < NBLK; b++) {
            const uint32_t base = b * 32;
            _Float16 bmax = (_Float16)0;          // block max of exp(P)
            for (uint32_t k = 0; k < 32; k++) {
                const _Float16 s = (_Float16)(as_bf16(Srow[base + k]) * scale);
                const _Float16 e = mu_fexp((_Float16)(s - m_new));
                bmax = fmaxf(bmax, e);
                rowsum = (_Float16)(rowsum + e);
            }
            const int se = bf16_floor_log2(__builtin_bit_cast(uint16_t, bmax));
            scale_scratch[b * SQ + row] = (uint32_t)(uint8_t)(se + 127);
            for (uint32_t w = 0; w < 32 / 4; w++) {   // 4 e4m3 per word, tiled store
                const uint32_t col0 = base + w * 4;
                uint32_t packed = 0;
                for (uint32_t k = 0; k < 4; k++) {
                    const _Float16 s = (_Float16)(as_bf16(Srow[col0 + k]) * scale);
                    const _Float16 e = mu_fexp((_Float16)(s - m_new));
                    const _Float16 ps = bf16_scale_pow2(e, -se);
                    packed |= (uint32_t)bf16_to_e4m3</*RNE=*/false>(
                                  __builtin_bit_cast(uint16_t, ps)) << (8 * k);
                }
                const uint32_t tk = col0 / 16, cc = col0 % 16;
                spad_u32[((ti * PE_TILES_K + tk) * 256 + rr * 16 + cc) / 4] = packed;
            }
        }
        // Update online-softmax state (this lane owns the row -> no cross-lane, no fence).
        const _Float16 l_old = first_block ? (_Float16)0 : as_bf16(l_state[row]);
        l_state[row]  = __builtin_bit_cast(uint16_t, (_Float16)(l_old * corr + rowsum));
        m_state[row]  = __builtin_bit_cast(uint16_t, m_new);
        corr_out[row] = __builtin_bit_cast(uint16_t, corr);
    }
}

// FUSED online-softmax + MX-FP8 requant (thorough perf rewrite). Contiguous 16-lane
// ownership: lane owns cols [lane*CPL, +CPL), CPL=BK/16 (register-cheap; word-packed
// disjoint spad stores, no sub-word hazard; no P_SMEM round-trip / double-read).
// Per row: load owned cols of S_j, scale, per-lane max; block-reduce (per 32-col MX block)
// + row-reduce (m/l); exp P in registers; e4m3 requant -> tiled A-spad; E8M0 -> scratch.
// CPL must be a multiple of 4 (BK a multiple of 64) so each lane writes whole words.
template <uint32_t SQ, uint32_t BK>
static __attribute__((noinline)) void fused_softmax_requant(
        const __shared uint16_t *S_smem16, __shared uint32_t *spad_u32,
        __shared uint32_t *scale_scratch,
        __shared uint16_t *m_state, __shared uint16_t *l_state, __shared uint16_t *corr_out,
        uint16_t softmax_scale_bf16, uint32_t first_block,
        uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
    constexpr uint32_t NT = MU_NUM_THREADS;      // 16 lanes
    constexpr uint32_t CPL = BK / NT;            // contiguous cols per lane (mult of 4)
    constexpr uint32_t NBLK = BK / 32;           // MX blocks per row
    constexpr uint32_t LPB = 32 / CPL;           // lanes per MX block
    constexpr uint32_t PE_TILES_K = BK / 16;
    const uint32_t lane = tid_in_threadblock % NT;
    const uint32_t warp = tid_in_threadblock / NT;
    const uint32_t nwarps = threads_per_threadblock / NT;
    const _Float16 scale = as_bf16(softmax_scale_bf16);
    volatile __shared uint16_t *buf =
        reinterpret_cast<volatile __shared uint16_t *>(0x15000) + warp * NT;
    const uint32_t b_of_lane = lane / LPB;       // which MX block this lane serves

    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const __shared uint16_t *Srow = S_smem16 + row * BK + lane * CPL;
        _Float16 s[CPL];
        _Float16 lmax = as_bf16(NEG_INF_BF16_BITS);
        for (uint32_t c = 0; c < CPL; c++) {
            s[c] = (_Float16)(as_bf16(Srow[c]) * scale);
            lmax = fmaxf(lmax, s[c]);
        }
        // block-local reduce (within LPB-lane groups) -> buf[b*LPB] = block S-max.
        // No per-step fence: warp lockstep orders it (mirrors warp_tree_reduce).
#ifdef FA_SKIP_REDUCE
        _Float16 bSmax = lmax, rmax = lmax;  // ABLATION: skip cross-lane max reduce (timing only)
#else
        buf[lane] = __builtin_bit_cast(uint16_t, lmax);
#ifndef FA_SM_NOFENCE
        mu_fence_smem();
#endif
        for (uint32_t st = 1; st < LPB; st <<= 1) {
            if ((lane % (2 * st)) == 0)
                buf[lane] = __builtin_bit_cast(uint16_t,
                    (_Float16)fmaxf(as_bf16(buf[lane]), as_bf16(buf[lane + st])));
        }
#ifndef FA_SM_NOFENCE
        mu_fence_smem();
#endif
        _Float16 bSmax = as_bf16(buf[b_of_lane * LPB]);     // this lane's block max
        _Float16 rmax = as_bf16(NEG_INF_BF16_BITS);         // row max over block leaders
        for (uint32_t b = 0; b < NBLK; b++) rmax = fmaxf(rmax, as_bf16(buf[b * LPB]));
#endif

        _Float16 m_old = first_block ? rmax : as_bf16(m_state[row]);
        _Float16 m_new = fmaxf(m_old, rmax);
        _Float16 corr = mu_fexp((_Float16)(m_old - m_new));

        // exp P (unnormalized), per-lane sum
        _Float16 lsum = (_Float16)0;
        for (uint32_t c = 0; c < CPL; c++) { s[c] = mu_fexp((_Float16)(s[c] - m_new)); lsum = (_Float16)(lsum + s[c]); }
#ifdef FA_SKIP_REDUCE
        _Float16 rowsum = lsum;  // ABLATION: skip cross-lane sum reduce (timing only)
#else
        buf[lane] = __builtin_bit_cast(uint16_t, lsum);
#ifndef FA_SM_NOFENCE
        mu_fence_smem();
#endif
        warp_tree_reduce<false>(buf, lane);
#ifndef FA_SM_NOFENCE
        mu_fence_smem();
#endif
        _Float16 rowsum = as_bf16(buf[0]);
#endif
        if (lane == 0) {
            _Float16 l_old = first_block ? (_Float16)0 : as_bf16(l_state[row]);
            l_state[row] = __builtin_bit_cast(uint16_t, (_Float16)(l_old * corr + rowsum));
            m_state[row] = __builtin_bit_cast(uint16_t, m_new);
            corr_out[row] = __builtin_bit_cast(uint16_t, corr);
        }
        // requant: per-block E8M0 scale from block P-max = exp(bSmax - m_new)
        _Float16 bPmax = mu_fexp((_Float16)(bSmax - m_new));
        int se = bf16_floor_log2(__builtin_bit_cast(uint16_t, bPmax));
        if ((lane % LPB) == 0) scale_scratch[b_of_lane * SQ + row] = (uint32_t)(uint8_t)(se + 127);
        // convert owned cols to e4m3, word-packed (CPL/4 words), store tiled.
        const uint32_t ti = row / 16, rr = row % 16;
#ifndef FA_SKIP_REQUANT
        for (uint32_t w = 0; w < CPL / 4; w++) {
            const uint32_t col0 = lane * CPL + w * 4;
            uint32_t packed = 0;
            for (uint32_t k = 0; k < 4; k++) {
                // fused, branchless scale(2^-se)+e4m3 -- eliminates bf16_scale_pow2 and the
                // divergent early-returns of bf16_to_e4m3 (was ~72% of softmax cost).
                packed |= (uint32_t)bf16_to_e4m3_scaled(
                              __builtin_bit_cast(uint16_t, s[w * 4 + k]), se) << (8 * k);
            }
            const uint32_t tk = col0 / 16, cc = col0 % 16;
#ifndef FA_SKIP_STORE
            spad_u32[((ti * PE_TILES_K + tk) * 256 + rr * 16 + cc) / 4] = packed;
#else
            asm volatile("" :: "r"(packed));  // ABLATION: keep convert, skip tiled store (isolate store bank-conflict cost)
#endif
        }
#else
        (void)ti; (void)rr; (void)bSmax; (void)se;  // ABLATION: skip requant convert+store
#endif
    }
}

// rescale_accumulate: O_acc[SQ][D] = (first ? 0 : O_acc*corr) + PV_j. PV_j is the mesh
// PV output (bf16, packed 2/word) at SPAD_DEST; O_acc is a persistent SMEM buffer. per row.
template <uint32_t SQ, uint32_t D>
static __attribute__((noinline)) void rescale_accumulate(
        __shared uint32_t *O_acc32, const __shared uint32_t *PV32,
        const __shared uint16_t *corr_out, uint32_t first_block,
        uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
    constexpr uint32_t NT = MU_NUM_THREADS;
    constexpr uint32_t DW = D / 2;
    const uint32_t lane = tid_in_threadblock % NT;
    const uint32_t warp = tid_in_threadblock / NT;
    const uint32_t nwarps = threads_per_threadblock / NT;
    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const _Float16 c = as_bf16(corr_out[row]);
        for (uint32_t w = lane; w < DW; w += NT) {
            uint32_t pv = PV32[row * DW + w];
            _Float16 plo = as_bf16((uint16_t)pv), phi = as_bf16((uint16_t)(pv >> 16));
            _Float16 olo = (_Float16)0, ohi = (_Float16)0;
            if (!first_block) {
                uint32_t a = O_acc32[row * DW + w];
                olo = as_bf16((uint16_t)a); ohi = as_bf16((uint16_t)(a >> 16));
            }
            O_acc32[row * DW + w] = pack_bf16x2((_Float16)(olo * c + plo),
                                                (_Float16)(ohi * c + phi));
        }
    }
}

// finalize_O: O[SQ][D] = O_acc / l  -> GMEM (packed bf16x2 word stores). per row.
template <uint32_t SQ, uint32_t D>
static __attribute__((noinline)) void finalize_O(
        const __shared uint32_t *O_acc32, const __shared uint16_t *l_state,
        uint32_t *O_gmem32, uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
    constexpr uint32_t NT = MU_NUM_THREADS;
    constexpr uint32_t DW = D / 2;
    const uint32_t lane = tid_in_threadblock % NT;
    const uint32_t warp = tid_in_threadblock / NT;
    const uint32_t nwarps = threads_per_threadblock / NT;
    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const _Float16 inv_l =
            (_Float16)(__builtin_bit_cast(_Float16, ONE_BF16_BITS) / as_bf16(l_state[row]));
        for (uint32_t w = lane; w < DW; w += NT) {
            uint32_t a = O_acc32[row * DW + w];
            O_gmem32[row * DW + w] = pack_bf16x2((_Float16)(as_bf16((uint16_t)a) * inv_l),
                                                 (_Float16)(as_bf16((uint16_t)(a >> 16)) * inv_l));
        }
    }
}

// Single-warp sequential copy of bf16 P from an SMEM scratch -> the requantizer SMEM
// region. The MxRequantizer needs program-order writes from one warp; we read packed
// words and write 16-bit halves in ascending address order.
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void copy_P_to_requant(const __shared uint32_t *P_smem32,
                                     __shared uint16_t *requant_smem,
                                     uint32_t tid_in_threadblock) {
    if (tid_in_threadblock >= MU_NUM_THREADS) return;   // warp 0 only
    // The requantizer smem manager requires EXACTLY 32-byte transactions
    // (reqSize = numGPUInputLanes*inputBits/8 = 16*16/8 = 32; TransferSizes(32,32), beatBytes=32;
    //  RadianceSharedMemComponents.scala:60 / GemminiTile.scala:212). A wider coalesced store
    //  (32 lanes x 4B = 128B, or a 64B cache line) exceeds the 32-byte max -> fragmented into rejected
    //  PutPartial beats (TLMonitor $finish). So emit ONE 32-byte, 32B-aligned beat per SIMT store:
    //  8 active lanes x 4B = 32B = the requantizer's 16-bf16 "fire" unit.
    __shared uint32_t *requant_smem32 = reinterpret_cast<__shared uint32_t *>(requant_smem);
    constexpr uint32_t WPB = 8;                          // 32B beat / 4B word
    constexpr uint32_t NBEATS = (SQ * SK * 2) / 32;      // SQ*SK bf16 * 2B / 32B
    const uint32_t lane = tid_in_threadblock;
    if (lane < WPB) {
        for (uint32_t beat = 0; beat < NBEATS; beat++) {
            const uint32_t idx = beat * WPB + lane;
            requant_smem32[idx] = P_smem32[idx];
        }
    }
}

// Requantize the softmax P tile (bf16 in SMEM) to MX-FP8 and write it ALL into SMEM so
// the PV gemm consumes it via the SKIP_A path (mesh reads spad + SF-SRAM directly,
// fence.s-coherent -- NO GMEM round-trip, NO global fence which is unreliable here).
//   - P elements -> the A scratchpad (spad base 0) in the EXACT Gemmini tiled layout the
//     mesh expects (verified from the DMA move-in / RTL): for element (m,c),
//       ti=m/16, rr=m%16, tk=c/16, cc=c%16, PE_TILES_K=SK/16
//       byte_offset = (ti*PE_TILES_K + tk)*256 + rr*16 + cc.
//     Each thread owns whole rows; per row the 32 cols form two disjoint, word-aligned
//     16-byte runs -> 4 e4m3 packed per 32-bit store, no cross-thread overlap.
//   - per-row E8M0 scale -> a word-per-scale SMEM scratch (packed to the A scale SRAM by
//     pack_scales_to_sfmem) so parallel scale stores never overlap.
// PARALLEL over rows; uses the validated bf16_to_e4m3 / bf16_floor_log2 encoder.
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void requant_P_to_spad_tiled(
        const __shared uint16_t *P_smem16, __shared uint32_t *spad_u32,
        __shared uint32_t *scale_scratch, uint32_t tid_in_threadblock,
        uint32_t threads_per_threadblock) {
    constexpr uint32_t NBLK = SK / 32;
    constexpr uint32_t PE_TILES_K = SK / 16;     // K dimension of A is SK
#ifdef FA_SYNCFIX
    FA_SYNC(6, threads_per_threadblock);   // P (all warps) -> requant (all threads)
#endif
    // Parallelize over (row x block) items across ALL threads (vs 1 row/thread) -> uses all 96 threads and
    // cuts serial depth. Each thread does ONE 32-elem block: max + convert + 8 tiled stores.
    // SUBBANK-CONFLICT FIX (2026-07-24): the SMEM halfword index is row*SK + b*32 + c; SK=256 and b*32
    // are multiples of 32, so the subbank (= byte[5:2] = index[4:1]) depends ONLY on c/w — which were
    // lane-UNIFORM => all 16 lanes hit the SAME subbank on every read/store (16-way conflict, and the
    // 8 lanes sharing a row also collided on the tiled store). Rotating each lane's start offset makes
    // the 16 lanes hit 16 distinct subbanks for the max pass and 8 for the convert/store pass.
#ifdef FA_RQ_FAST
    // ================= FAST PATH (2026-07-25) ==============================
    // Three changes, all of them instruction-count wins (the baseline body was
    // 1763 instructions per item, 49 vx_split/vx_join divergence regions):
    //  1. MAX PASS: P >= 0 (softmax probs), so the |x| in the old block-max loop
    //     is dead -- but the compiler implemented it as flt.h + vx_split_n +
    //     branch + fneg.h + vx_join, i.e. a WARP DIVERGENCE REGION per element
    //     (160 of the 267 max-pass instructions).  Dropping abs leaves 1 fmax.h
    //     per element.  4 independent accumulators break the fmax dep chain.
    //  2. CONVERT: e4m3_pack4() / e4m3_pack4_swar() (see above) -- 8.5 resp. 6.25
    //     straight-line integer ops per element, in INLINE ASM because the C form
    //     kept being turned back into 4 vx_split_n/vx_join regions per element.
    //  3. STORE address: the 8 tiled spad words per item are two runs of 4
    //     consecutive words 64 words apart -> one base pointer + immediates.
    // Optional FA_PSWIZ additionally transposes the P scratch layout so the 16
    // lanes of a warp read one 64B line / 16 distinct subbanks per load.
    (void)P_smem16;
    const __shared uint32_t *P32 = reinterpret_cast<const __shared uint32_t *>(P_smem16);
#ifdef FA_RQ_SWAR
    const uint32_t SWC = 0x07ff07ffu, SWM = 0x80008000u, SWH = 0x0000ffffu;
#endif
    for (uint32_t item = tid_in_threadblock; item < SQ * NBLK; item += threads_per_threadblock) {
#ifdef FA_PSWIZ
        // b-MAJOR items: a warp owns 16 consecutive items == 16 consecutive rows
        // of ONE MX block, which also collapses the 8 tiled stores from 8 lines
        // to 4 (rr*16 for rr=0..15 spans exactly 256B = 4 lines).
        const uint32_t b = item / SQ, row = item % SQ;
        const uint32_t il = item & 15u;
        const __shared uint32_t *Pb = P32 + (item >> 4) * 256u + il;
#else
        const uint32_t row = item / NBLK, b = item % NBLK;
        const __shared uint32_t *Pb = P32 + row * (SK / 2) + b * 16u;
#endif
#ifdef FA_PSWIZX
        // byte address = base ^ (k*68): base is 1024-aligned + il*4 (bits [9:6]
        // clear), k*64 lands in [9:6] and (il^k)*4 in [5:2] -> a single xori.
#define FA_PW(k) (*reinterpret_cast<const __shared uint32_t *>( \
                     reinterpret_cast<uintptr_t>(Pb) ^ (uintptr_t)((k) * 68u)))
#elif defined(FA_PSWIZ)
#define FA_PW(k) (Pb[(k) * 16u])
#else
#define FA_PW(k) (Pb[(k)])
#endif
#ifdef FA_RQ_CACHE
        // ILP knob: hold all 16 words of the block in registers so the convert
        // pass does NOT re-read them (32 SMEM loads/item -> 16).  With the fast
        // body this does NOT spill any more (objdump: no in-loop (sp) traffic,
        // only extra callee-saves) -- the 2026-07-24 ">130k, spills" result was
        // an artifact of the old 1763-instruction body.  It is still a LOSS:
        // trading loads for register pressure only helps when there is no other
        // warp to hide the latency, and the machine always has more warps.
        uint32_t pw[16];
        for (uint32_t k = 0; k < 16; k++) pw[k] = FA_PW(k);
#undef FA_PW
#define FA_PW(k) (pw[(k)])
#endif
        // ---- block max over the 32 elements (integer-free, no abs, 4 chains)
        _Float16 x0 = (_Float16)0, x1 = (_Float16)0, x2 = (_Float16)0, x3 = (_Float16)0;
        for (uint32_t i = 0; i < 16; i += 4) {
            const uint32_t w0 = FA_PW(i + 0), w1 = FA_PW(i + 1);
            const uint32_t w2 = FA_PW(i + 2), w3 = FA_PW(i + 3);
            x0 = fmaxf(x0, fmaxf(as_bf16((uint16_t)w0), as_bf16((uint16_t)(w0 >> 16))));
            x1 = fmaxf(x1, fmaxf(as_bf16((uint16_t)w1), as_bf16((uint16_t)(w1 >> 16))));
            x2 = fmaxf(x2, fmaxf(as_bf16((uint16_t)w2), as_bf16((uint16_t)(w2 >> 16))));
            x3 = fmaxf(x3, fmaxf(as_bf16((uint16_t)w3), as_bf16((uint16_t)(w3 >> 16))));
        }
        const _Float16 bmax = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
        // em = se+127 = the E8M0 code = the bf16 exponent field of the block max.
        const uint32_t em = ((uint32_t)__builtin_bit_cast(uint16_t, bmax) >> 7) & 0xffu;
        // K8 = max((em-7)<<3, 0) + 8.  The max() only fires for an all-zero block
        // (em=0 -> se=-127); it is value-neutral there (every code is 0 anyway)
        // and it keeps K >= 0 so zero/subnormal inputs can never alias to a code.
        const uint32_t K8 = fa_clamp_K8(((int)em - 7) << 3);
        scale_scratch[b * SQ + row] = ((K8 - 8u) >> 3) + 7u;   // == max(em,7)
#ifdef FA_RQ_SWAR
        const uint32_t D1 = 0x7ff8u - (K8 - 8u);
        const uint32_t D2 = D1 | (D1 << 16);
#endif
        // ---- convert 32 elements -> 8 packed words -> tiled A-spad
        // word index = ti*1024 + b*128 + rr*4 + (u&3) + (u>>2)*64
        const uint32_t ti = row / 16, rr = row % 16;
        __shared uint32_t *dst = spad_u32 + ti * (PE_TILES_K * 64u) + b * 128u + rr * 4u;
        for (uint32_t h = 0; h < 2; h++) {
            for (uint32_t q = 0; q < 4; q++) {
                const uint32_t u = h * 4u + q;
                const uint32_t wlo = FA_PW(2 * u), whi = FA_PW(2 * u + 1);
#ifdef FA_RQ_SWAR
                dst[h * 64u + q] = e4m3_pack4_swar(wlo, whi, D2, SWC, SWM, SWH);
#else
                dst[h * 64u + q] = e4m3_pack4(wlo, whi, K8);
#endif
            }
        }
#undef FA_PW
    }
#else
    const uint32_t lane = tid_in_threadblock & (MU_NUM_THREADS - 1);
    for (uint32_t item = tid_in_threadblock; item < SQ * NBLK; item += threads_per_threadblock) {
        const uint32_t row = item / NBLK, b = item % NBLK;
        const __shared uint16_t *Prow = P_smem16 + row * SK;
        const uint32_t ti = row / 16, rr = row % 16;
        {
            // block max over 32 elements (order-independent) -- lane-rotated start: c = (c0 + 2*lane) & 31
            // => index[4:1] = (c0>>1 + lane) & 15 -> 16 DISTINCT subbanks across the warp.
            // MEASURED-WORSE ALTERNATIVES (2026-07-24), do not retry blindly:
            //  * lane-rotating c/w to break the (real) lane-uniform subbank pattern: 53.2k -> 58.3k.
            //  * register-caching the block's 16 words + 4 independent accumulators: requant >130k
            //    (pw[16] SPILLS; spills are catastrophic here). Reverted both.
            // WORD loads (2 bf16 per load) instead of halfword loads: HALVES the SMEM request count.
            // MEASURED (perf-viz): requant is pure load-LATENCY stall (fp pipes 98.4% idle, SMEM 0.5% of
            // peak, 0.62 loads/cyc), so time tracks the number of requests, not bytes. No register array
            // here on purpose -- the 16-word register-cache version SPILLED and was catastrophic (>130k).
            const __shared uint32_t *Pw =
                reinterpret_cast<const __shared uint32_t *>(Prow + b * 32);
            // TWO independent load+reduce chains (2 words live, NOT 16 -> no spill; the 16-word
            // register-cache version spilled and cost >130k). Gets 2 SMEM requests in flight per thread,
            // which is what matters: requant is request-LATENCY bound (fp 1.6%, SMEM 0.5% of peak).
            _Float16 m0 = as_bf16((uint16_t)0), m1 = m0;
            for (uint32_t i = 0; i < 16; i += 2) {
                const uint32_t wa = Pw[i], wb = Pw[i + 1];      // independent -> both can be outstanding
                _Float16 a0 = as_bf16((uint16_t)wa),        a1 = as_bf16((uint16_t)(wa >> 16));
                _Float16 b0 = as_bf16((uint16_t)wb),        b1 = as_bf16((uint16_t)(wb >> 16));
                a0 = (a0 < (_Float16)0) ? (_Float16)(-a0) : a0;  a1 = (a1 < (_Float16)0) ? (_Float16)(-a1) : a1;
                b0 = (b0 < (_Float16)0) ? (_Float16)(-b0) : b0;  b1 = (b1 < (_Float16)0) ? (_Float16)(-b1) : b1;
                m0 = fmaxf(m0, fmaxf(a0, a1));
                m1 = fmaxf(m1, fmaxf(b0, b1));
            }
            _Float16 bmax = fmaxf(m0, m1);
            int se = bf16_floor_log2(__builtin_bit_cast(uint16_t, bmax));  // target 0
            scale_scratch[b * SQ + row] = (uint32_t)(uint8_t)(se + 127);   // E8M0 code (word)
            // e4m3 of the 32 elements (P / 2^se), packed 4/word into the tiled A-spad.
            // lane-rotated word order: w = (w0 + lane) & 7 -> spreads both the SMEM reads and the
            // tiled spad stores (store subbank = (rr*4 + w%4)&15) across the warp.
            for (uint32_t w = 0; w < 8; w++) {                  // 32 cols / 4 per word
                const uint32_t col0 = b * 32 + w * 4;           // global col of first of 4
                // 2 WORD loads instead of 4 halfword loads (same halving as the max pass above)
                const uint32_t wlo = Pw[w * 2], whi = Pw[w * 2 + 1];
                const uint32_t packed =
                      ((uint32_t)bf16_to_e4m3_scaled((uint16_t)wlo,         se) << 0)
                    | ((uint32_t)bf16_to_e4m3_scaled((uint16_t)(wlo >> 16), se) << 8)
                    | ((uint32_t)bf16_to_e4m3_scaled((uint16_t)whi,         se) << 16)
                    | ((uint32_t)bf16_to_e4m3_scaled((uint16_t)(whi >> 16), se) << 24);
                const uint32_t tk = col0 / 16, cc = col0 % 16;  // 4 cols stay in one tile
                const uint32_t byte_off = (ti * PE_TILES_K + tk) * 256 + rr * 16 + cc;
                spad_u32[byte_off / 4] = packed;
            }
        }
    }
#endif  // FA_RQ_FAST
}

// Pack the per-scale SMEM scratch (1 word per E8M0 byte) into the contiguous E8M0 byte
// array in the A scale SRAM (GEMMINI_SF_MEM_A), using 32-bit word stores (4 scales/word).
// Layout is linear (byte i -> A row i), matching what load_scale_factors would produce.
// Thread-0 only -> non-overlapping, ordered word stores (no sub-word PutPartial hazard).
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void pack_scales_to_sfmem(
        const __shared uint32_t *scale_scratch, __shared uint32_t *sfmem_a32,
        uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
#ifdef FA_SYNCFIX
    FA_SYNC(7, threads_per_threadblock);   // scale_scratch (all threads) -> packer
#endif
#ifdef FA_PK_LANES
    // LANE-PARALLEL (2026-07-25): same trick mxgemm_core.hpp's load_scale_factors_lanes
    // uses.  The SF-SRAM write path is ~100 cyc/word of pure LATENCY, and the single-
    // threaded loop above serialises 128 of them (measured 18.6k cyc = 145 cyc/word,
    // the second-largest SIMT phase after requant).  ONE warp's 16 lanes may write it
    // (only MULTI-WARP writes corrupt the SF interface), which puts 16 stores in flight.
    // The partitioning must be CONTIGUOUS BLOCKS per lane, never an interleave: an
    // interleave makes the 16 lanes hit 16 consecutive words, the lane collector merges
    // them into one 64B burst and FlitMergeNode $finish-es on "start address not aligned".
    // The 4-read/shift packing math rides along 16x parallel for free.
    if (tid_in_threadblock >= MU_NUM_THREADS) return;   // warp 0 only
    constexpr uint32_t NW = ((SK / 32) * SQ) / 4;
    constexpr uint32_t PER = (NW + MU_NUM_THREADS - 1) / MU_NUM_THREADS;
    volatile __shared uint32_t *sf = sfmem_a32;
    const uint32_t base = tid_in_threadblock * PER;
    for (uint32_t j = 0; j < PER; j++) {
        const uint32_t w = base + j;
        if (w < NW) {
            uint32_t packed = 0;
            for (uint32_t k = 0; k < 4; k++)
                packed |= (scale_scratch[w * 4 + k] & 0xff) << (8 * k);
            sf[w] = packed;
        }
    }
#else
    // single thread, strictly ascending words (FlitMergeNode contract)
    if (tid_in_threadblock != 0) return;
    constexpr uint32_t NS = (SK / 32) * SQ;
    for (uint32_t w = 0; w < NS / 4; w++) {
        uint32_t packed = 0;
        for (uint32_t k = 0; k < 4; k++)
            packed |= (scale_scratch[w * 4 + k] & 0xff) << (8 * k);
        sfmem_a32[w] = packed;
    }
#endif
}

// SPLIT (2026-07-25): the packing math (4 SMEM reads + shifts per word) used to run on thread 0 together
// with the SF-SRAM stores => 147 cyc/word (18.9k), vs 46-64 cyc/word for a plain ascending copy. Do the
// packing with ALL threads into a SMEM staging buffer, then let thread 0 do a PURE ASCENDING COPY to the
// SF SRAM (which is the only pattern FlitMergeNode accepts: single-thread, strictly ascending 4B pairs).
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void prepack_scales(
        const __shared uint32_t *scale_scratch, __shared uint32_t *packed,
        uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
    constexpr uint32_t NW = ((SK / 32) * SQ) / 4;
    for (uint32_t w = tid_in_threadblock; w < NW; w += threads_per_threadblock) {
        uint32_t p = 0;
        for (uint32_t k = 0; k < 4; k++)
            p |= (scale_scratch[w * 4 + k] & 0xff) << (8 * k);
        packed[w] = p;
    }
}

template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void copy_scales_to_sfmem(
        const __shared uint32_t *packed, volatile __shared uint32_t *sfmem_a32,
        uint32_t tid_in_threadblock) {
    if (tid_in_threadblock != 0) return;      // MUST be one thread, strictly ascending (merge contract)
    constexpr uint32_t NW = ((SK / 32) * SQ) / 4;
    for (uint32_t w = 0; w < NW; w++) sfmem_a32[w] = packed[w];
}

// Normalize the PV result: O[row][:] = O_unnorm[row][:] / l[row].
// O_unnorm is the bf16 PV gemm result in SMEM (row-major [SQ][D]); l is fp32 in GMEM.
// Writes the final O (bf16) to GMEM as 32-bit word stores (clean to verify).
// One row per warp; lane owns consecutive bf16 pairs (word granularity).
template <uint32_t SQ, uint32_t D>
static __attribute__((noinline)) void normalize_output(const __shared uint32_t *O_smem32, uint32_t *O_gmem32,
                                     const __shared uint16_t *l_smem,
                                     uint32_t tid_in_threadblock,
                                     uint32_t threads_per_threadblock) {
    constexpr uint32_t NT = MU_NUM_THREADS;
    constexpr uint32_t WPL = D / (2 * NT);           // O words per lane
    constexpr uint32_t DW = D / 2;                    // words per row
    const uint32_t lane = tid_in_threadblock % NT;
    const uint32_t warp = tid_in_threadblock / NT;
    const uint32_t nwarps = threads_per_threadblock / NT;

    for (uint32_t row = warp; row < SQ; row += nwarps) {
        _Float16 inv_l = (_Float16)((_Float16)__builtin_bit_cast(_Float16, ONE_BF16_BITS)
                                    / as_bf16(l_smem[row]));  // 1/l in bf16 (SMEM read)
        const __shared uint32_t *Orow = O_smem32 + row * DW;
        uint32_t *Out = O_gmem32 + row * DW;
        for (uint32_t j = 0; j < WPL; j++) {
            uint32_t w = Orow[j * NT + lane];
            _Float16 lo = (_Float16)(as_bf16((uint16_t)w) * inv_l);
            _Float16 hi = (_Float16)(as_bf16((uint16_t)(w >> 16)) * inv_l);
            Out[j * NT + lane] = pack_bf16x2(lo, hi);
        }
    }
}

#endif // _FLASH_MX_IMPL_H_
