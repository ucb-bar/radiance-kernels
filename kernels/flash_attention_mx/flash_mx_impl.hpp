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
        reinterpret_cast<volatile __shared uint16_t *>(0x15000) + warp * NT;

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
