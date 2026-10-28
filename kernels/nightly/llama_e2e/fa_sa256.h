// fa_sa256.h: fa_sa.h (nightly/fa_mxfp8_sr as llama_e2e's ATTN phase) for head dim 256 with Gemma-2's attention
// logit soft-cap, z = cap tanh(s scale / cap).  Selected by gen_data.py for hd 256 + attn_cap (Gemma-2) through
// d[29] = 1 and LL_FA_SA256 in llama_data.h (LL_FA_CAP_BITS cap, LL_FA_K1_BITS scale / cap, both f32).
//
// What hd 256 changes (every other choice is fa_sa.h's):
//   - 32-query tiles (QT = 32): O of a tile accumulates in one accumulator half across the key blocks while QK runs in
//     the other, and a half is 128 rows = 32 PE tiles of 16 x 16 (LoopMatmul c_addr: 4 rows per tile).  At d 256 a
//     64-query O is 64 tiles, so only 32 queries fit (QK: 2 x 4 tiles, PV: 2 x 16).
//   - 64-key blocks (BK = 64): a 64-key K^T block at d 256 is 256 x 64 B = 1024 scratchpad rows, a V block 64 x 256 B
//     = 1024 rows, i.e. exactly the d-128 / 128-key blocks, so fa_sa.h's scratchpad map holds unchanged (Q tile 32 x 256
//     = 512 rows per buffer, S 32 x 64 bf16 = 256 rows, P 128 rows, O 32 x 256 bf16 = 1024 rows, the flip sink 256 rows).
//     A 128-key block (2048 rows each for K and V) would leave no room for S, O, the sinks and LSUM in 8192 rows.
//   - loop shapes: QK (I, J, K) = (2, 4, 16), PV (2, 16, 4), K load B (J 4, K 16), V load A (I 4, K 16), Q load A
//     (I 2, K 16): the whole next Q tile rides along with key block 1's K load (fa_sa: two Q halves in blocks 1, 2).
//   - K scales: the ROPE / cache layout is fa_sa's [SK/128][hd/32][128]; a 64-key block's [8][64] scales are a 2-D
//     MX_LOAD_SCALES (8 rows of 64 B, DRAM pitch 128).  Q scales: ROPE writes them with TM = 32 (d[24]), so a tile's
//     [8][32] scales are contiguous.  V scales [SK/32][hd] do not depend on BK.
//   - softmax: a row (64 keys) per lane pair: warps 2, 3 take keys 0..31 of rows 0..15 / 16..31, warps 4, 5 keys
//     32..63 (two softmax warps per core).  Per score z = cap tanh_f(s k1) (kernel.cpp's tanh_f, fp32, as the unfused
//     SOFTMAX), P = fexp(bf16(z - m)).  m: block-0 max, max(cap tanh_f(k1 max_c s_c), cap - 64) (tanh is increasing,
//     so one tanh of the raw max; both lanes of a row compute it over all 64 keys, then a 4-warp barrier (BAR_M = 12)
//     before P overwrites S in place).  The floor bounds z - m <= 64 for any later key (|z| < cap), so exp never
//     overflows; a row's block-0 max scores e^0 = 1 unless the floor applies, and z - m >= -2 cap > -355 (fexp's wrap).  Each lane keeps its half-row sum l in a register and
//     stores it at the tile's last block (LSUM [2][32]); the O normalization takes 1 / (l_0 + l_1).
//   - O normalization + MX into oproj's A operand: 4 rows per call, a lane per (row, 2 groups), so each scale word
//     (4 rows of one group) is written by one store of one warp (fa_phase.h's Fa<D>::norm_o_block has 2 rows per call
//     at d 256, splitting a scale word over two warps).
#ifndef LL_FA_SA256_H
#define LL_FA_SA256_H

namespace fa_sa256 {

constexpr uint32_t D = LL_HD, BK = 64, QT = 32, SQ = LL_M, SK = LL_SK, H = LL_HQ, GQ = LL_HQ / LL_HKV;
constexpr uint32_t NB = SK / BK, NQT = H * SQ / QT, P0 = SK - SQ;
static_assert(D == 256, "fa_sa256: head dim 256");
static_assert(SQ % QT == 0 && SK % 128 == 0, "query tiles inside a head; K scales in 128-key groups");
static_assert(NQT % NIGHTLY_CLUSTERS == 0, "query tiles must split evenly over clusters");
constexpr uint32_t QT_C = NQT / NIGHTLY_CLUSTERS, G = QT_C * NB;
static_assert(NB >= 6 && NB % 2 == 0, "O groups are normalized in key blocks 2..5 of the next tile; NB even as fa_sa");
static_assert(P0 >= BK - 1, "key block 0 is never masked (its max is the tile's)");
static_assert(NIGHTLY_OCC == 4, "warp roles assume 8 warps per cluster");
constexpr uint32_t KD = D / 16;                            // QK depth tiles, PV width tiles (16)
constexpr uint32_t JB = BK / 16;                           // QK width tiles, PV depth tiles (4)
constexpr uint32_t IQ = QT / 16;                           // query tiles of 16 (2)
static_assert(IQ * KD * 4 <= mx::ACC_HALF && IQ * JB * 4 <= mx::ACC_HALF, "O and S each fit an accumulator half");
static_assert(IQ % 2 == 0 && JB % 2 == 0, "spad stores cover whole 32-row / 32-column MX blocks");

// Scratchpad map (rows of 16 B): fa_sa.h's, the operands sized for d 256 / 64 keys / 32 queries
constexpr uint32_t K_END = 1024, K_ROWS = D * BK / 16;     // K^T block [256][64]
constexpr uint32_t V_END = 2048, V_ROWS = BK * D / 16, V_START = V_END - V_ROWS;
constexpr uint32_t QA[2] = {2048, 2560}, Q_ROWS = QT * D / 16;
constexpr uint32_t PA[2] = {3072, 3584}, P_ROWS = QT * BK / 16;
constexpr uint32_t SP[2] = {4096 * 16, 5120 * 16}, S_ROWS = QT * BK * 2 / 16, ROW_B = BK * 2;   // S / P bf16 [32][64]
constexpr uint32_t FLIP_ROW = 6144, FLIP_ROWS = QT * BK * 2 / 16;   // the K-load loop's (2 x 4 tile) store
constexpr uint32_t O_ROW = 6656, O_SM = O_ROW * 16, O_PITCH = 2 * D, O_ROWS = QT * O_PITCH / 16;
constexpr uint32_t JUNK_ROW = 7712;                        // toggle()'s 32 x 32 bf16 store: 128 rows
constexpr uint32_t LSUM[2] = {7856 * 16 + 256, 7904 * 16 + 256};    // l f32 [kh][32] by tile parity
static_assert(K_ROWS <= K_END && V_START >= K_END && QA[0] >= V_END && QA[1] >= QA[0] + Q_ROWS &&
              PA[0] >= QA[1] + Q_ROWS && PA[1] >= PA[0] + P_ROWS && SP[0] / 16 >= PA[1] + P_ROWS &&
              SP[1] / 16 >= SP[0] / 16 + S_ROWS && FLIP_ROW >= SP[1] / 16 + S_ROWS &&
              FLIP_ROW + FLIP_ROWS <= O_ROW && O_ROW + O_ROWS <= JUNK_ROW && JUNK_ROW + 128 <= LSUM[0] / 16,
              "scratchpad map");
static_assert(NIGHTLY_MX_LOOPS_SMEM >= LSUM[1] + 256 && NIGHTLY_MX_LOOPS_SMEM + 4 <= mx::SPAD_ROWS * 16,
              "the LOOP_WS count word must not overlap the scratchpad map");
constexpr uint32_t KSC = D / 32 * BK, VSC = BK / 32 * D, QSC = D / 32 * QT;   // scale bytes per block / tile
constexpr uint32_t KSC128 = D / 32 * 128;                  // a 128-key group of the K scale layout

enum : uint32_t { BAR_P = 3, BAR_T = 6, BAR_S = 7, BAR_O = 8, BAR_N = 9, BAR_ND = 10, BAR_0 = 11, BAR_M = 12 };

static uint32_t scale_sink[256] __attribute__((aligned(64)));

static inline _Float16 bf(uint32_t bits16) { return __builtin_bit_cast(_Float16, (uint16_t)bits16); }
static inline uint32_t bits(_Float16 x) { return __builtin_bit_cast(uint16_t, x); }
static inline float f32(_Float16 x) { return __builtin_bit_cast(float, bits(x) << 16); }
static inline volatile __shared uint32_t *sm32(uint32_t byte) { return (volatile __shared uint32_t *)byte; }
static inline volatile __shared float *smf(uint32_t byte) { return (volatile __shared float *)byte; }
__attribute__((noinline, convergent)) static void fa_bar(uint32_t id, uint32_t warps) { mu_barrier(id, warps); }
static inline uint32_t head_of(uint32_t qt) { return qt / (SQ / QT); }
static inline uint32_t tile_of(uint32_t i, uint32_t tb) { return i * NIGHTLY_CLUSTERS + tb; }

// ---- producer helpers (lane 0 of warp 0): operand bases from the descriptor, written and read by the lead only ----
static inline uint32_t q_, qsc_, kt_, ksc_, v_, vsc_, kt_bs_, ksc_bs_, v_bs_, vsc_bs_;
static inline uint32_t kt_base(uint32_t h) { return kt_ + (h / GQ) * kt_bs_; }
static inline uint32_t ksc_base(uint32_t h) { return ksc_ + (h / GQ) * ksc_bs_; }
static inline uint32_t v_base(uint32_t h) { return v_ + (h / GQ) * v_bs_; }
static inline uint32_t vsc_base(uint32_t h) { return vsc_ + (h / GQ) * vsc_bs_; }
// K scales of 64-key block j: [hd/32] rows of 64 B inside the [SK/128][hd/32][128] layout (DRAM pitch 128)
static inline uint32_t ksc_block(uint32_t h, uint32_t j) { return ksc_base(h) + (j / 2) * KSC128 + (j % 2) * BK; }
// MX_LOAD_SCALES, 2-D: rs1[39:0] address, [63:40] DRAM pitch; rs2[31:0] bytes per row, [32] weight, [45:33] dest,
// [53:46] rows, [54] gated (loop-managed)
static inline void load_k_scales_2d(uint32_t h, uint32_t j, uint32_t dst_off, bool gated) {
  mx::cmd(mx::MX_LOAD_SCALES, (mx::host_addr(ksc_block(h, j)) & 0xFFFFFFFFFFull) | (128ull << 40),
          ((uint64_t)gated << 54) | ((uint64_t)(D / 32) << 46) | ((uint64_t)(dst_off & 0x1FFF) << 33) | (1ull << 32) | BK);
}

static inline void load_k_data(uint32_t h, uint32_t j) {
  mx::set_ab_dram(0, mx::host_addr(kt_base(h) + j * BK), D, SK);
  mx::loop_ws_spad(IQ, JB, KD, 0, K_END, 0, false, false, {true, false, true, true, true});
}
// K block j of head h + (q_src != 0) the next tile's whole Q (I = 2, K = 16: one 32 x 256 tile) + the store-half flip
static inline void load_kq(uint32_t h, uint32_t j, bool flip, uint32_t q_src, uint32_t q_row) {
  mx::set_ab_dram(q_src ? mx::host_addr(q_src) : 0, mx::host_addr(kt_base(h) + j * BK), D, SK);
  mx::loop_ws_spad(IQ, JB, KD, q_src ? q_row : 0, K_END, flip ? FLIP_ROW : 0, false, true,
                   {q_src == 0, false, true, true, !flip});
}
static inline void load_k(uint32_t h, uint32_t j) {
  load_k_data(h, j);
  load_k_scales_2d(h, j, 0, false);
}
static inline void load_v_data(uint32_t h, uint32_t j) {
  mx::set_ab_dram(mx::host_addr(v_base(h) + j * BK * D), 0, D, SK);
  mx::loop_ws_spad(JB, 4, KD, V_START, K_END, 0, false, true, {false, true, true, true, true});
}
static inline uint32_t q_tile_src(uint32_t qt) { return q_ + (head_of(qt) * SQ + (qt % (SQ / QT)) * QT) * D; }
static inline void load_q(uint32_t qt, uint32_t b) {
  mx::set_ab_dram(mx::host_addr(q_tile_src(qt)), 0, D, SK);
  mx::loop_ws_spad(IQ, 4, KD, QA[b], K_END, 0, false, false, {false, true, true, true, true});
}
static inline void load_q_scales(uint32_t qt) { mx::load_scales(mx::host_addr(qsc_ + qt * QSC), QSC, false, mx::SF_BUF1); }
static inline void load_k1_vbuf(uint32_t h) {
  mx::set_ab_dram(0, mx::host_addr(kt_base(h) + 1 * BK), D, SK);
  mx::loop_ws_spad(IQ, JB, KD, 0, V_END, 0, false, false, {true, false, true, true, true});
  load_k_scales_2d(h, 1, mx::SF_BUF1, false);
}
static inline void qk_vbuf(uint32_t b) {
  mx::config_scale(IQ, JB, KD, 1, 1, mx::host_addr(scale_sink));
  mx::loop_ws_spad(IQ, JB, KD, QA[b], V_END, 0, false, true, {true, true, true, false, true});
}
static inline void qk(uint32_t b) { mx::loop_ws_spad(IQ, JB, KD, QA[b], K_END, 0, false, true, {true, true, true, false, true}); }
static inline void qk_store(uint32_t b, uint32_t sm) {
  mx::loop_ws_spad(IQ, JB, KD, QA[b], K_END, sm / 16, false, true, {true, true, true, false, false});
}
static inline void cfg_qk(bool resident) { mx::config_scale(IQ, JB, KD, 1, 0, mx::host_addr(scale_sink), resident, false); }
static inline void cfg_pv(bool resident, bool reset) {
  mx::config_scale(IQ, KD, JB, 0, 1, mx::host_addr(scale_sink), resident, reset);
}
static inline void load_k_scales_gated(uint32_t h, uint32_t j) { load_k_scales_2d(h, j, 0, true); }
static inline void cfg_qk_managed() {   // I, J, K = 2, 4, 16; act half 1 (Q), weight half 0 (K)
  const uint64_t rs1 = (0ull << 61) | (1ull << 60) | ((uint64_t)KD << 51) | ((uint64_t)JB << 42) | ((uint64_t)IQ << 33) |
                       (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
  mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
}
static inline void load_v_scales_gated(uint32_t h, uint32_t j) {
  mx::cmd(mx::MX_LOAD_SCALES, mx::host_addr(vsc_base(h) + j * VSC) & 0xFFFFFFFFFFull,
          (1ull << 54) | ((uint64_t)mx::SF_BUF1 << 33) | (1ull << 32) | VSC);
}
static inline void cfg_pv_managed() {   // I, J, K = 2, 16, 4; act half 0 (P), weight half 1 (V)
  const uint64_t rs1 = (1ull << 61) | (0ull << 60) | ((uint64_t)JB << 51) | ((uint64_t)KD << 42) | ((uint64_t)IQ << 33) |
                       (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
  mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
}
static inline void pv(uint32_t b, bool acc) { mx::loop_ws_spad(IQ, KD, JB, PA[b], V_END, 0, acc, true, {true, true, true, false, true}); }
static inline void store_s(uint32_t sm) { mx::loop_ws_spad(IQ, JB, 1, 0, K_END, sm / 16, false, false, {true, true, true, true, false}); }
static inline void toggle() {
  mx::config_scale(2, 2, 1, 0, 0, mx::host_addr(scale_sink));
  mx::loop_ws_spad(2, 2, 1, 0, K_END, JUNK_ROW, false, true, {true, true, true, false, true});
}
static inline void store_only(uint32_t I, uint32_t J, uint32_t c_row) {
  mx::loop_ws_spad(I, J, 1, 0, K_END, c_row, false, true, {true, true, true, true, false});
}
static inline void store_o() { cfg_pv(false, false); store_only(IQ, KD, O_ROW); }
constexpr uint32_t K_SPAD_REQUANT = 34;
static inline void spad_requant(uint32_t b) {   // S / P bf16 [32][64] at SP[b] -> P operand tiles at PA[b], resident scales
  const uint64_t scale_dram = mx::host_addr(scale_sink) & 0x1FFFFFFFFull;
  const uint64_t rs1 = (uint64_t)(SP[b] / 16) | ((uint64_t)PA[b] << 14) | (1ull << 28) | (1ull << 29) | (scale_dram << 30);
  mx::cmd(K_SPAD_REQUANT, rs1, (uint64_t)QT | ((uint64_t)BK << 16));
}
// K load for block b + 2, issued after PV_{b-1} (fa_sa.h's issue_k_load; a function, not a [&] lambda)
static inline void issue_k_load(uint32_t b, uint32_t tb) {
  const uint32_t ib = b / NB, jb = b % NB;
  const bool o_tile = jb == 0 && ib > 0, qn = jb == 1 && ib + 1 < QT_C;
  load_kq(head_of(tile_of((b + 2) / NB, tb)), (b + 2) % NB, b >= 2 && !o_tile,
          qn ? q_tile_src(tile_of(ib + 1, tb)) : 0, QA[(ib + 1) & 1]);
}

// ---- softmax with the soft-cap ----
static inline float k1_() { return __builtin_bit_cast(float, (uint32_t)LL_FA_K1_BITS); }
static inline float cap_() { return __builtin_bit_cast(float, (uint32_t)LL_FA_CAP_BITS); }
// the row's m from block 0 (64 keys, 32 words at `row`): max(cap tanh_f(k1 max s), cap - 64)
static __attribute__((noinline)) float row_m(uint32_t row, uint32_t lane) {
  volatile __shared uint32_t *S = sm32(row);
  _Float16 mx = bf(0xFF80u);
  for (uint32_t w = 0; w < BK / 2; w += 8) {
    uint32_t v[8];
#pragma unroll
    for (uint32_t u = 0; u < 8; u++) v[u] = S[(w + u + lane) & (BK / 2 - 1)];
#pragma unroll
    for (uint32_t u = 0; u < 8; u++) {
      const _Float16 a = bf(v[u] & 0xFFFF), b = bf(v[u] >> 16);
      mx = a > mx ? a : mx;
      mx = b > mx ? b : mx;
    }
  }
  const float cap = cap_();
  return __builtin_fmaxf(cap * tanh_f(f32(mx) * k1_()), cap - 64.0f);
}
// one half row (32 keys, 16 words at `base`) in place, S bf16 -> P = fexp(bf16(cap tanh_f(s k1) - m)) bf16; returns
// the half row's sum.  MASK: keys c (0..31 in the half) with c > lim get P = 0.  Word order (w + u) ^ lane: the 16
// lanes (16 rows 128 B apart) hit 16 different SMEM banks.
template <bool MASK>
static __attribute__((noinline)) float sm_half(uint32_t base, uint32_t lane, float m, int32_t lim) {
  volatile __shared uint32_t *S = sm32(base);
  const float k1 = k1_(), cap = cap_();
  float l = 0.f;
  constexpr uint32_t U = 4;
  for (uint32_t w = 0; w < 16; w += U) {
    uint32_t v[U];
#pragma unroll
    for (uint32_t u = 0; u < U; u++) v[u] = S[(w + u) ^ lane];
#pragma unroll
    for (uint32_t u = 0; u < U; u++) {
      const float za = cap * tanh_f(lo(v[u]) * k1), zb = cap * tanh_f(hi(v[u]) * k1);
      uint32_t ea = bits(mu_fexp(bf(bf16(za - m)))), eb = bits(mu_fexp(bf(bf16(zb - m))));
      if (MASK) {
        const int32_t idx = (int32_t)((w + u) ^ lane);
        ea &= opaque(0u - (uint32_t)(2 * idx <= lim));
        eb &= opaque(0u - (uint32_t)(2 * idx + 1 <= lim));
      }
      l += f_bits(ea << 16) + f_bits(eb << 16);
      S[(w + u) ^ lane] = ea | (eb << 16);
    }
  }
  return l;
}

// ---- O of a tile (bf16 [32][256] at O_SM) normalized by 1 / (l_0 + l_1) and MX-quantized into oproj's A operand:
// rows 4 blk .. 4 blk + 3, lane = (row lane & 3, groups lane >> 2 and 4 + (lane >> 2)).  Bit-exact with oquant of the
// bf16 O (fa_phase.h's norm_o_block).  always_inline: 10 arguments (Muon: <= 8 for a call). ----
constexpr uint32_t O_BLOCKS = QT / 4;
static __attribute__((always_inline)) inline void norm_o4(uint32_t lane, uint32_t lsm, uint32_t blk, uint32_t q0, uint32_t h,
                                                          uint32_t A, uint32_t ASC, uint32_t TM, uint32_t TK, uint32_t KS) {
  const uint32_t row = 4 * blk + (lane & 3), t = q0 + row;
  const float inv = 1.0f / (smf(lsm)[row] + smf(lsm)[QT + row]);
#pragma unroll
  for (uint32_t p = 0; p < 2; p++) {
    const uint32_t k = (lane >> 2) + 4 * p, col = h * D + 32 * k;
    volatile __shared uint32_t *src = sm32(O_SM + row * O_PITCH + 64 * k);
    quant_group([&](uint32_t jj) __attribute__((always_inline)) {
                  const uint32_t w = src[jj];
                  const uint32_t a = bits((_Float16)(f32(bf(w & 0xFFFF)) * inv)), b = bits((_Float16)(f32(bf(w >> 16)) * inv));
                  return F2{f_bits(a << 16), f_bits(b << 16)};
                },
                lane, (volatile uint32_t *)a_code(A, TM, TK, KS, t, col), a_scale(ASC, TM, TK, KS, t, col));
  }
}

// ---- the phase (fa_sa.h's schedule; every thread of the cluster calls it) ----
static void phase(const volatile uint32_t *d, uint32_t tid, uint32_t tpb, uint32_t tb, uint32_t &acc_parity) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  const bool lead = tid == 0;
  const uint32_t OA = d[8], OASC = d[20], OTM = d[21], OTK = d[22], OKS = d[23];
#define FA_SA256_NORM_O(qt, lsm, blk) norm_o4(lane, (lsm), (blk), ((qt) % (SQ / QT)) * QT, head_of(qt), OA, OASC, OTM, OTK, OKS)
  const uint32_t qt0 = tile_of(0, tb);
  const bool smx = warp >= 2 && warp <= 5;
  if (lead) {
    q_ = d[2]; qsc_ = d[3]; kt_ = d[4]; ksc_ = d[5]; v_ = d[6]; vsc_ = d[7];
    kt_bs_ = d[16]; ksc_bs_ = d[17]; v_bs_ = d[18]; vsc_bs_ = d[19];
    if (acc_parity) toggle();                            // the GEMM phases' compute half -> 0
    acc_parity = 0;
    mx::config_ld(D, 0);
    mx::config_ld(SK, 1);
    load_q(qt0, 0);
    load_q_scales(qt0);
    load_k(head_of(qt0), 0);
    load_k1_vbuf(head_of(qt0));
    mx::config_ex(mx::FP8, mx::FP8, mx::BF16);
    cfg_qk(false);
    qk(0);
    cfg_qk(false);
    store_s(SP[0]);
  } else asm volatile("nop");
  mu_fence_smem();
  fa_bar(BAR_0, warps);

  if (warp == 0) {
    if (lead) mx::fence();
    else asm volatile("nop");
    fa_bar(BAR_S, 5);
    if (lead) {
      toggle();
      qk_vbuf(0);
    } else asm volatile("nop");
    fa_bar(BAR_P, 5);
    if (lead) spad_requant(0);
    else asm volatile("nop");
    uint32_t n_qk = 0;
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t i = g / NB, j = g % NB;
      const bool more = opaque(g + 1 < G), more2 = opaque(g + 2 < G), more3 = opaque(g + 3 < G);
      if (lead) {
        if (g == 0) {
          mx::fence();
          store_s(SP[1]);
          mx::fence();
          toggle();
          load_k(head_of(qt0), 2);
          cfg_qk(false);
          qk(0);
          store_s(SP[0]);
          n_qk = mx::loops_issued();
        } else if (more) {
          mx::wait_retired(n_qk);
        }
        if (more2 && g > 0 && (g + 2) % NB == 0) load_q_scales(tile_of((g + 2) / NB, tb));
      } else asm volatile("nop");
      if (j == 1 && i > 0) fa_bar(BAR_N, 4);
      if (j == 0 && i >= 2) fa_bar(BAR_ND, 4);
      if (more) fa_bar(BAR_S, 5);
      if (lead) {
        if (more2 && g > 0) {
          const bool o_tile = j == 0 && i > 0;
          if (g == 1) load_k_scales_gated(head_of(tile_of(3 / NB, tb)), 3 % NB);
          load_v_scales_gated(head_of(tile_of(g / NB, tb)), g % NB);
          if (more3) load_k_scales_gated(head_of(tile_of((g + 3) / NB, tb)), (g + 3) % NB);
          if (o_tile) store_o();
          cfg_qk_managed();
        }
        if (g > 0) spad_requant(g & 1);
        if (more2 && g > 0) {
          qk_store(((g + 2) / NB) & 1, SP[g & 1]);
          n_qk = mx::loops_issued();
        } else {
          if (g > 0) toggle();
          load_v_scales_gated(head_of(tile_of(g / NB, tb)), g % NB);
        }
        load_v_data(head_of(tile_of(g / NB, tb)), g % NB);
        cfg_pv_managed();
        pv(g & 1, g % NB != 0);
        if (more3) issue_k_load(g + 1, tb);
      } else asm volatile("nop");
      if (more) fa_bar(BAR_P, 5);
    }
    if (QT_C >= 2) fa_bar(BAR_ND, 4);
    if (lead) {
      mx::fence();
      store_o();
      mx::fence();
    } else asm volatile("nop");
  } else if (smx) {
    // lane pair per row: warps 2, 3 keys 0..31 of rows 0..15 / 16..31, warps 4, 5 keys 32..63 (cores w % 2)
    const uint32_t kh = (warp - 2) >> 1, row = ((warp - 2) & 1) * 16 + lane;
    float m = 0.f, lacc = 0.f;                           // registers (no runtime-indexed stack arrays)
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t i = g / NB, j = g % NB;
      const uint32_t q0 = (tile_of(i, tb) % (SQ / QT)) * QT;
      fa_bar(BAR_S, 5);
      const uint32_t rbase = SP[g & 1] + row * ROW_B;
      if (j == 0) m = row_m(rbase, lane);               // block 0 is never masked (P0 >= BK - 1)
      else asm volatile("nop");
      // row_m reads the partner lane's half too: every softmax warp has its m before any P overwrites S in place
      if (j == 0) fa_bar(BAR_M, 4);
      const int32_t lim = (int32_t)(P0 + q0 + row) - (int32_t)(j * BK + kh * (BK / 2));
      float l;
      if (j * BK + BK - 1 > P0 + q0) l = sm_half<true>(rbase + kh * BK, lane, m, lim);
      else l = sm_half<false>(rbase + kh * BK, lane, m, lim);
      lacc = (j == 0 ? 0.f : lacc) + l;
      if (j == NB - 1) smf(LSUM[i & 1])[kh * QT + row] = lacc;
      else asm volatile("nop");
      mu_fence_smem();
      fa_bar(BAR_P, 5);
    }
  } else {
    const uint32_t nw = warp == 1 ? 0 : warp - 5;
    for (uint32_t i = 1; i < QT_C; i++) {
      fa_bar(BAR_N, 4);
      for (uint32_t blk = nw; blk < O_BLOCKS; blk += 3) FA_SA256_NORM_O(tile_of(i - 1, tb), LSUM[(i - 1) & 1], blk);
      fa_bar(BAR_ND, 4);
    }
  }
  fa_bar(BAR_O, warps);
  for (uint32_t blk = warp; blk < O_BLOCKS; blk += warps) FA_SA256_NORM_O(tile_of(QT_C - 1, tb), LSUM[(QT_C - 1) & 1], blk);
  fa_bar(BAR_T, warps);
#undef FA_SA256_NORM_O
}

}  // namespace fa_sa256
#endif
