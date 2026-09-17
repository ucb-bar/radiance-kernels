// Derived from UC Berkeley Gemmini (github.com/ucb-bar/gemmini), BSD-3-Clause. See NOTICE.
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

// Gemma-2 attention-score soft-cap primitive: cap * tanh(s / cap), all in bf16.
// tanh(x) = sign(x)*(1 - e)/(1 + e), e = exp(-2|x|) in (0,1] (numerically stable: no
// overflow -- exp of a nonpositive arg). Uses the hardware fexp (mu_fexp, bf16). This is
// the SAME map as the fp32 final-logit soft-cap (gemma_final_softcap), just in the bf16
// score domain the flash softmax already works in. inv_cap = 1/cap is precomputed by the
// data generator so the per-element work is one fexp + a few bf16 ops (no divide by cap).
static inline _Float16 bf16_softcap(_Float16 s, _Float16 cap, _Float16 inv_cap) {
    _Float16 x = (_Float16)(s * inv_cap);
    bool neg = x < (_Float16)0;
    _Float16 ax = neg ? (_Float16)(-x) : x;
    _Float16 e = mu_fexp((_Float16)((_Float16)(-2) * ax));       // exp(-2|x|) in (0,1]
    _Float16 t = (_Float16)(((_Float16)1 - e) / ((_Float16)1 + e));
    return (_Float16)(cap * (neg ? (_Float16)(-t) : t));
}

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

// 16-lane intra-warp tree reduction over a per-warp SMEM buffer (mirrors the
// softmax kernel's reduce_*: no in-loop fence; relies on warp lockstep). Result
// ends in buf[0]; caller fences then reads buf[0]. IS_MAX selects max vs sum.
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
        reinterpret_cast<volatile __shared uint16_t *>(0xFA00) + warp * NT;

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
        reinterpret_cast<volatile __shared uint16_t *>(0xFA00) + warp * NT;

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
        reinterpret_cast<volatile __shared uint16_t *>(0xFA00) + warp * NT;

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
        buf[lane] = __builtin_bit_cast(uint16_t, mloc);
        mu_fence_smem(); warp_tree_reduce<true>(buf, lane); mu_fence_smem();
        _Float16 bmax = as_bf16(buf[0]);                       // block max (scaled)
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
        buf[lane] = __builtin_bit_cast(uint16_t, lloc);
        mu_fence_smem(); warp_tree_reduce<false>(buf, lane); mu_fence_smem();
        _Float16 lsum = as_bf16(buf[0]);
        for (uint32_t j = 0; j < WPL; j++)                     // P_j UNNORMALIZED
            Prow[j * NT + lane] = pack_bf16x2(slo[j], shi[j]);
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
// GQA/causal variant: `key_offset` is this key-block's GLOBAL key base (j*BK), `q_pos0`
// the global position of query row 0. When `apply_mask` is set, an entry is masked
// (S->-inf, prob 0) whenever its global key position exceeds its global query position
// (causal). `apply_mask` is uniform across the warp (set only for blocks that straddle the
// diagonal), so the extra work is branch-free / divergence-free. Fully-below-diagonal
// blocks pass apply_mask=0 and cost exactly the baseline; fully-above blocks are skipped
// by the caller (never reach here).
// GEMMA-2 template params DO_SOFTCAP / SLIDING (both compile-time from the data header):
//  DO_SOFTCAP: fold attn_logit_softcapping into the ONLINE softmax -- the scaled score v is
//    replaced by cap*tanh(v/cap) BEFORE it enters the per-lane max (lmax), the block/row max,
//    and every exp. This is the wrinkle: the running max carried across key blocks must be a
//    max of *capped* scores, so the cap has to happen here, not as a post-softmax epilogue --
//    otherwise the cross-block max compare mixes capped and un-capped score scales.
//  SLIDING: add the sliding-window low-side mask (key_pos <= q_pos - window). Causal high-side
//    masking is unchanged. `window` and cap bits are runtime args (constant-folded).
template <uint32_t SQ, uint32_t BK, bool DO_SOFTCAP = false, bool SLIDING = false>
static __attribute__((noinline)) void fused_softmax_requant(
        const __shared uint16_t *S_smem16, __shared uint32_t *spad_u32,
        __shared uint32_t *scale_scratch,
        __shared uint16_t *m_state, __shared uint16_t *l_state, __shared uint16_t *corr_out,
        uint16_t softmax_scale_bf16, uint32_t first_block,
        uint32_t key_offset, uint32_t q_pos0, uint32_t apply_mask,
        uint16_t cap_bf16, uint16_t cap_inv_bf16, uint32_t window,
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
    const _Float16 cap = as_bf16(cap_bf16);
    const _Float16 cap_inv = as_bf16(cap_inv_bf16);
    volatile __shared uint16_t *buf =
        reinterpret_cast<volatile __shared uint16_t *>(0xFA00) + warp * NT;
    const uint32_t b_of_lane = lane / LPB;       // which MX block this lane serves
    const uint32_t key_base = key_offset + lane * CPL;  // global key of this lane's col 0
    // sliding-window layers must also mask the too-old low side, so any processed block may
    // need masking (not just diagonal-straddling ones); force per-element mask evaluation.
    const bool do_mask = apply_mask || SLIDING;

    for (uint32_t row = warp; row < SQ; row += nwarps) {
        const __shared uint16_t *Srow = S_smem16 + row * BK + lane * CPL;
        const uint32_t q_pos = q_pos0 + row;     // global query position of this row
        _Float16 s[CPL];
        bool masked[CPL];
        _Float16 lmax = as_bf16(NEG_INF_BF16_BITS);
        for (uint32_t c = 0; c < CPL; c++) {
            _Float16 v = (_Float16)(as_bf16(Srow[c]) * scale);
            if constexpr (DO_SOFTCAP) v = bf16_softcap(v, cap, cap_inv);  // cap BEFORE max/exp
            const uint32_t kpos = key_base + c;                // global key position
            bool m = do_mask && (kpos > q_pos);                // causal: key_pos > query_pos
            if constexpr (SLIDING) m = m || (do_mask && (kpos + window <= q_pos)); // too old
            masked[c] = m;
            s[c] = m ? as_bf16(NEG_INF_BF16_BITS) : v;         // masked -> -inf (drops out of max)
            lmax = fmaxf(lmax, s[c]);
        }
        // block-local reduce (within LPB-lane groups) -> buf[b*LPB] = block S-max.
        // No per-step fence: warp lockstep orders it (mirrors warp_tree_reduce).
        buf[lane] = __builtin_bit_cast(uint16_t, lmax);
        mu_fence_smem();
        for (uint32_t st = 1; st < LPB; st <<= 1) {
            if ((lane % (2 * st)) == 0)
                buf[lane] = __builtin_bit_cast(uint16_t,
                    (_Float16)fmaxf(as_bf16(buf[lane]), as_bf16(buf[lane + st])));
        }
        mu_fence_smem();
        _Float16 bSmax = as_bf16(buf[b_of_lane * LPB]);     // this lane's block max
        _Float16 rmax = as_bf16(NEG_INF_BF16_BITS);         // row max over block leaders
        for (uint32_t b = 0; b < NBLK; b++) rmax = fmaxf(rmax, as_bf16(buf[b * LPB]));

        _Float16 m_old = first_block ? rmax : as_bf16(m_state[row]);
        _Float16 m_new = fmaxf(m_old, rmax);
        _Float16 corr = mu_fexp((_Float16)(m_old - m_new));

        // exp P (unnormalized), per-lane sum. Masked cols -> exactly 0 (don't rely on
        // mu_fexp(-inf); force it), so they contribute nothing to the row sum / PV / e4m3.
        _Float16 lsum = (_Float16)0;
        for (uint32_t c = 0; c < CPL; c++) {
            _Float16 e = mu_fexp((_Float16)(s[c] - m_new));
            if (masked[c]) e = (_Float16)0;
            s[c] = e; lsum = (_Float16)(lsum + s[c]);
        }
        buf[lane] = __builtin_bit_cast(uint16_t, lsum);
        mu_fence_smem(); warp_tree_reduce<false>(buf, lane); mu_fence_smem();
        _Float16 rowsum = as_bf16(buf[0]);
        if (lane == 0) {
            _Float16 l_old = first_block ? (_Float16)0 : as_bf16(l_state[row]);
            l_state[row] = __builtin_bit_cast(uint16_t, (_Float16)(l_old * corr + rowsum));
            m_state[row] = __builtin_bit_cast(uint16_t, m_new);
            corr_out[row] = __builtin_bit_cast(uint16_t, corr);
        }
        // requant: per-block E8M0 scale from block P-max = exp(bSmax - m_new).
        // Unnormalized probs are exp(s - m_new) in (0,1], so bPmax<=1 and se<=0 ALWAYS. A
        // fully-masked MX group has bSmax=-inf -> mu_fexp(-inf)=NaN -> floor_log2 would give
        // a large positive se -> E8M0 code 2^(se) overflows to +inf in the mesh, and 0*inf
        // NaNs the whole systolic PV output. Clamp se<=0 (masked probs are 0, so the scale
        // is immaterial there) to keep the E8M0 code finite.
        _Float16 bPmax = mu_fexp((_Float16)(bSmax - m_new));
        int se = bf16_floor_log2(__builtin_bit_cast(uint16_t, bPmax));
        if (se > 0) se = 0;
        if ((lane % LPB) == 0) scale_scratch[b_of_lane * SQ + row] = (uint32_t)(uint8_t)(se + 127);
        // convert owned cols to e4m3, word-packed (CPL/4 words), store tiled.
        const uint32_t ti = row / 16, rr = row % 16;
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
            spad_u32[((ti * PE_TILES_K + tk) * 256 + rr * 16 + cc) / 4] = packed;
        }
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
        const _Float16 l = as_bf16(l_state[row]);
        // guard l<=0 (a fully-masked row -- e.g. a sliding-window row whose visible keys all
        // fell in skipped blocks): 1/l would be +inf and inf*0(=O_acc) -> NaN. Mirror the
        // model's safe_l: emit 0 for such rows instead of NaN.
        const _Float16 inv_l = (l > (_Float16)0)
            ? (_Float16)(__builtin_bit_cast(_Float16, ONE_BF16_BITS) / l)
            : (_Float16)0;
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
    constexpr uint32_t NW = (SQ * SK) / 2;              // total words
    for (uint32_t i = tid_in_threadblock; i < NW; i += MU_NUM_THREADS) {
        uint32_t w = P_smem32[i];
        requant_smem[2 * i] = (uint16_t)w;
        requant_smem[2 * i + 1] = (uint16_t)(w >> 16);
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
    for (uint32_t row = tid_in_threadblock; row < SQ; row += threads_per_threadblock) {
        const __shared uint16_t *Prow = P_smem16 + row * SK;
        const uint32_t ti = row / 16, rr = row % 16;
        for (uint32_t b = 0; b < NBLK; b++) {
            // block max over 32 elements
            _Float16 bmax = as_bf16((uint16_t)0);
            for (uint32_t c = 0; c < 32; c++) {
                _Float16 v = as_bf16(Prow[b * 32 + c]);
                _Float16 a = (v < (_Float16)0) ? (_Float16)(-v) : v;
                bmax = fmaxf(bmax, a);
            }
            int se = bf16_floor_log2(__builtin_bit_cast(uint16_t, bmax));  // target 0
            scale_scratch[b * SQ + row] = (uint32_t)(uint8_t)(se + 127);   // E8M0 code (word)
            // e4m3 of the 32 elements (P / 2^se), packed 4/word into the tiled A-spad
            for (uint32_t w = 0; w < 8; w++) {                  // 32 cols / 4 per word
                const uint32_t col0 = b * 32 + w * 4;           // global col of first of 4
                uint32_t packed = 0;
                for (uint32_t k = 0; k < 4; k++) {
                    _Float16 ps = bf16_scale_pow2(as_bf16(Prow[col0 + k]), -se);
                    // truncate (not RNE) to match the golden's float_quantize_trunc
                    uint8_t e = bf16_to_e4m3</*RNE=*/false>(__builtin_bit_cast(uint16_t, ps));
                    packed |= (uint32_t)e << (8 * k);
                }
                const uint32_t tk = col0 / 16, cc = col0 % 16;  // 4 cols stay in one tile
                const uint32_t byte_off = (ti * PE_TILES_K + tk) * 256 + rr * 16 + cc;
                spad_u32[byte_off / 4] = packed;
            }
        }
    }
}

// Pack the per-scale SMEM scratch (1 word per E8M0 byte) into the contiguous E8M0 byte
// array in the A scale SRAM (GEMMINI_SF_MEM_A), using 32-bit word stores (4 scales/word).
// Layout is linear (byte i -> A row i), matching what load_scale_factors would produce.
// Thread-0 only -> non-overlapping, ordered word stores (no sub-word PutPartial hazard).
template <uint32_t SQ, uint32_t SK>
static __attribute__((noinline)) void pack_scales_to_sfmem(
        const __shared uint32_t *scale_scratch, __shared uint32_t *sfmem_a32,
        uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
    // NOTE: MUST be single-warp, program-order writes -- the SF-SRAM/requantizer scale
    // interface corrupts under multi-warp parallel writes (verified: parallel -> 0 output).
    if (tid_in_threadblock != 0) return;
    constexpr uint32_t NS = (SK / 32) * SQ;             // total E8M0 scale bytes
    for (uint32_t w = 0; w < NS / 4; w++) {
        uint32_t packed = 0;
        for (uint32_t k = 0; k < 4; k++)
            packed |= (scale_scratch[w * 4 + k] & 0xff) << (8 * k);
        sfmem_a32[w] = packed;
    }
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
