// fa_sa.h: nightly/fa_mxfp8_sr (the standalone kernel, v7h) as llama_e2e's ATTN phase, kept as close to it as the model
// allows: shapes are compile-time constants (the model's LL_HD / LL_M / LL_SK / LL_HQ / LL_HKV, as FA_D / FA_SQ / FA_SK /
// FA_H in the standalone build), block-0 max (lazy softmax), the same producer schedule, softmax and O normalization.
// Additions, each only what the model needs:
//   - GQA: query head h reads KV head h / Gq; operand base addresses from the phase descriptor (fa_phase.h's layouts)
//   - the prefill's causal mask: only key blocks with keys beyond P0 + q0 (the last one for M = 128) run the masked
//     softmax (MASK = true); the other blocks run the standalone's loop unchanged
//   - O normalized and MX-quantized straight into oproj's A operand (d[8] codes, d[20] scales, d[21..23] its TM / TK /
//     KS), by fa_phase.h's Fa<D>::norm_o_block (no oquant phase; bit-exact with oquant of the bf16 O)
//   - head dim 64 (TinyLlama): the loops' depth / width tile counts as fa_phase.h's Fa<64>
//   - LL_FA_EXACT (DeepSeek): fa_phase.h's exact-max pre-pass (QK over every key block of a tile first, the softmax
//     warps keep each row's masked max in DRAM at d[25]); key block 0 then uses that max, and fexp's argument is
//     clamped at -64 (it wraps below ~-355; exact-max rows reach s - m ~ -450)
//   - prologue: the GEMM phases' accumulator half reset (acc_parity), barrier ids clear of kernel.cpp's
// Selected by gen_data.py (--fa-impl standalone, the default for hd 64 / 128) through d[29] = 1 and LL_FA_SA in
// llama_data.h.
#ifndef LL_FA_SA_H
#define LL_FA_SA_H

namespace fa_sa {

constexpr uint32_t D = LL_HD, BK = 128, QT = 64, SQ = LL_M, SK = LL_SK, H = LL_HQ, GQ = LL_HQ / LL_HKV, DQ = H * D;
constexpr uint32_t NB = SK / BK, NQT = H * SQ / QT, P0 = SK - SQ;
static_assert(D == 64 || D == 128, "fa_sa: head dim 64 or 128");
static_assert(NQT % NIGHTLY_CLUSTERS == 0, "query tiles must split evenly over clusters");
constexpr uint32_t QT_C = NQT / NIGHTLY_CLUSTERS, G = QT_C * NB;
static_assert(NB >= 6, "O groups are normalized in key blocks 2..5 of the next tile");
static_assert(P0 >= BK - 1, "key block 0 is never masked (its max is the tile's)");
#ifndef LL_FA_EXACT
#define LL_FA_EXACT 0
#endif
constexpr bool EXACT = LL_FA_EXACT;
static_assert(!EXACT || NB % 2 == 0, "the pre-pass flips the compute / store halves NB times per tile");
static_assert(NIGHTLY_OCC == 4, "warp roles assume 8 warps per cluster");
constexpr uint32_t KD = D / 16;                            // QK depth tiles, PV width tiles (8 in the standalone)

// Scratchpad map (rows of 16 B), the standalone's
constexpr uint32_t K_END = 1024;
constexpr uint32_t V_END = 2048, V_START = V_END - BK * D / 16;
constexpr uint32_t QA[2] = {2048, 2560};
constexpr uint32_t PA[2] = {3072, 3584};
constexpr uint32_t SP[2] = {4096 * 16, 5120 * 16};
constexpr uint32_t O_ROW = 6656, O_SM = O_ROW * 16;
constexpr uint32_t JUNK_ROW = 7712;
constexpr uint32_t FLIP_ROW = 6144;
static_assert(FLIP_ROW + 512 <= O_ROW, "the store-flip sink (512 rows) must end below O");
constexpr uint32_t MREF[2] = {7856 * 16, 7904 * 16};
constexpr uint32_t LSUM[2] = {MREF[0] + 256, MREF[1] + 256};
constexpr uint32_t INV_OFF = 256;
constexpr uint32_t DUMMY_A = 0;
static_assert(MREF[1] + 768 <= mx::SPAD_ROWS * 16, "scratchpad budget");
static_assert(NIGHTLY_MX_LOOPS_SMEM >= MREF[1] + 768 && NIGHTLY_MX_LOOPS_SMEM + 4 <= mx::SPAD_ROWS * 16,
              "the LOOP_WS count word must not overlap the scratchpad map");
constexpr uint32_t QH_ROWS = (QT / 2) * D / 16;            // one Q half (256 rows at d 128)
constexpr uint32_t KSC = D / 32 * BK, VSC = BK / 32 * D, QSC = D / 32 * QT;   // scale bytes per block / tile

enum : uint32_t { BAR_P = 3, BAR_T = 6, BAR_S = 7, BAR_O = 8, BAR_N = 9, BAR_ND = 10, BAR_0 = 11 };

static uint32_t scale_sink[256] __attribute__((aligned(64)));

static inline _Float16 bf(uint32_t bits16) { return __builtin_bit_cast(_Float16, (uint16_t)bits16); }
static inline uint32_t bits(_Float16 x) { return __builtin_bit_cast(uint16_t, x); }
static inline float f32(_Float16 x) { return __builtin_bit_cast(float, bits(x) << 16); }
static inline volatile __shared uint32_t *sm32(uint32_t byte) { return (volatile __shared uint32_t *)byte; }
static inline volatile __shared float *smf(uint32_t byte) { return (volatile __shared float *)byte; }
__attribute__((noinline, convergent)) static void fa_bar(uint32_t id, uint32_t warps) { mu_barrier(id, warps); }
static inline uint32_t head_of(uint32_t qt) { return qt / (SQ / QT); }

// ---- producer helpers (lane 0 of warp 0): operand bases from the descriptor, written and read by the lead only ----
static inline uint32_t q_, qsc_, kt_, ksc_, v_, vsc_, kt_bs_, ksc_bs_, v_bs_, vsc_bs_;
static inline uint32_t kt_base(uint32_t h) { return kt_ + (h / GQ) * kt_bs_; }
static inline uint32_t ksc_base(uint32_t h) { return ksc_ + (h / GQ) * ksc_bs_; }
static inline uint32_t v_base(uint32_t h) { return v_ + (h / GQ) * v_bs_; }
static inline uint32_t vsc_base(uint32_t h) { return vsc_ + (h / GQ) * vsc_bs_; }

static inline void load_k_data(uint32_t h, uint32_t j) {
  mx::set_ab_dram(0, mx::host_addr(kt_base(h) + j * BK), D, SK);
  mx::loop_ws_spad(4, 8, KD, DUMMY_A, K_END, 0, false, false, {true, false, true, true, true});
}
static inline void load_kq(uint32_t h, uint32_t j, bool flip, uint32_t q_src, uint32_t q_row) {
  mx::set_ab_dram(q_src ? mx::host_addr(q_src) : 0, mx::host_addr(kt_base(h) + j * BK), D, SK);
  mx::loop_ws_spad(2, 8, KD, q_src ? q_row : DUMMY_A, K_END, flip ? FLIP_ROW : 0, false, true,
                   {q_src == 0, false, true, true, !flip});
}
static inline void load_k(uint32_t h, uint32_t j) {
  load_k_data(h, j);
  mx::load_scales(mx::host_addr(ksc_base(h) + j * KSC), KSC, true, 0);
}
static inline void load_v_data(uint32_t h, uint32_t j) {
  mx::set_ab_dram(mx::host_addr(v_base(h) + j * BK * D), 0, D, SK);
  mx::loop_ws_spad(8, 4, KD, V_START, K_END, 0, false, true, {false, true, true, true, true});
}
static inline uint32_t q_tile_src(uint32_t qt) { return q_ + (head_of(qt) * SQ + (qt % (SQ / QT)) * QT) * D; }
static inline void load_q(uint32_t qt, uint32_t b) {
  mx::set_ab_dram(mx::host_addr(q_tile_src(qt)), 0, D, SK);
  mx::loop_ws_spad(4, 4, KD, QA[b], K_END, 0, false, false, {false, true, true, true, true});
}
static inline uint32_t q_half_src(uint32_t qt, uint32_t half) { return q_tile_src(qt) + 32 * half * D; }
static inline void load_q_scales(uint32_t qt) { mx::load_scales(mx::host_addr(qsc_ + qt * QSC), QSC, false, mx::SF_BUF1); }
static inline void load_k1_vbuf(uint32_t h) {
  mx::set_ab_dram(0, mx::host_addr(kt_base(h) + 1 * BK), D, SK);
  mx::loop_ws_spad(4, 8, KD, DUMMY_A, V_END, 0, false, false, {true, false, true, true, true});
  mx::load_scales(mx::host_addr(ksc_base(h) + 1 * KSC), KSC, true, mx::SF_BUF1);
}
static inline void qk_vbuf(uint32_t b) {
  mx::config_scale(4, 8, KD, 1, 1, mx::host_addr(scale_sink));
  mx::loop_ws_spad(4, 8, KD, QA[b], V_END, 0, false, true, {true, true, true, false, true});
}
static inline void qk(uint32_t b) { mx::loop_ws_spad(4, 8, KD, QA[b], K_END, 0, false, true, {true, true, true, false, true}); }
static inline void qk_store(uint32_t b, uint32_t sm) {
  mx::loop_ws_spad(4, 8, KD, QA[b], K_END, sm / 16, false, true, {true, true, true, false, false});
}
static inline void cfg_qk(bool resident) { mx::config_scale(4, 8, KD, 1, 0, mx::host_addr(scale_sink), resident, false); }
static inline void cfg_pv(bool resident, bool reset) {
  mx::config_scale(4, KD, 8, 0, 1, mx::host_addr(scale_sink), resident, reset);
}
static inline void load_k_scales_gated(uint32_t h, uint32_t j) {
  mx::cmd(mx::MX_LOAD_SCALES, mx::host_addr(ksc_base(h) + j * KSC) & 0xFFFFFFFFFFull, (1ull << 54) | (1ull << 32) | KSC);
}
static inline void cfg_qk_managed() {
  const uint64_t rs1 = (0ull << 61) | (1ull << 60) | ((uint64_t)KD << 51) | (8ull << 42) | (4ull << 33) |
                       (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
  mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
}
static inline void load_v_scales_gated(uint32_t h, uint32_t j) {
  mx::cmd(mx::MX_LOAD_SCALES, mx::host_addr(vsc_base(h) + j * VSC) & 0xFFFFFFFFFFull,
          (1ull << 54) | ((uint64_t)mx::SF_BUF1 << 33) | (1ull << 32) | VSC);
}
static inline void cfg_pv_managed() {
  const uint64_t rs1 = (1ull << 61) | (0ull << 60) | (8ull << 51) | ((uint64_t)KD << 42) | (4ull << 33) |
                       (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
  mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
}
static inline void pv(uint32_t b, bool acc) { mx::loop_ws_spad(4, KD, 8, PA[b], V_END, 0, acc, true, {true, true, true, false, true}); }
static inline void store_s(uint32_t sm) { mx::loop_ws_spad(4, 8, 1, DUMMY_A, K_END, sm / 16, false, false, {true, true, true, true, false}); }
static inline void toggle() {
  mx::config_scale(2, 2, 1, 0, 0, mx::host_addr(scale_sink));
  mx::loop_ws_spad(2, 2, 1, DUMMY_A, K_END, JUNK_ROW, false, true, {true, true, true, false, true});
}
static inline void store_only(uint32_t I, uint32_t J, uint32_t c_row) {
  mx::loop_ws_spad(I, J, 1, DUMMY_A, K_END, c_row, false, true, {true, true, true, true, false});
}
static inline void store_o() { cfg_pv(false, false); store_only(4, KD, O_ROW); }
constexpr uint32_t K_SPAD_REQUANT = 34;
static inline void spad_requant(uint32_t b) {
  const uint64_t scale_dram = mx::host_addr(scale_sink) & 0x1FFFFFFFFull;
  const uint64_t rs1 = (uint64_t)(SP[b] / 16) | ((uint64_t)PA[b] << 14) | (1ull << 28) | (1ull << 29) | (scale_dram << 30);
  mx::cmd(K_SPAD_REQUANT, rs1, (uint64_t)QT | ((uint64_t)BK << 16));
}

// ---- softmax (the standalone's), MASK: keys with column c > lim (lim = P0 + query - j BK) get P = 0 ----
// m_ex: the row's scaled exact max (EXACT: the pre-pass's, read at j = 0)
template <bool MASK>
static __attribute__((noinline)) void softmax_row(uint32_t row, uint32_t j, uint32_t lane, uint32_t base, uint32_t mref,
                                                  uint32_t lsm, int32_t lim, float m_ex) {
  constexpr uint32_t SCB = LL_FA_SCALE_BITS;
  const _Float16 sc = bf((SCB + 0x7FFFu + ((SCB >> 16) & 1u)) >> 16);
  volatile __shared uint32_t *S = sm32(base + row * 256);
  _Float16 m;
  if (EXACT && j == 0) {
    smf(mref)[row] = m_ex;
    m = (_Float16)m_ex;
  } else if (j == 0) {
    _Float16 mx = bf(0xFF80);
    for (uint32_t w = 0; w < 64; w += 8) {
      uint32_t v[8];
#pragma unroll
      for (uint32_t u = 0; u < 8; u++) v[u] = S[(w + u + lane) & 63];
#pragma unroll
      for (uint32_t u = 0; u < 8; u++) {
        const _Float16 a = bf(v[u] & 0xFFFF), b = bf(v[u] >> 16);
        mx = a > mx ? a : mx;
        mx = b > mx ? b : mx;
      }
    }
    mx = mx * sc;
    smf(mref)[row] = f32(mx);
    m = mx;
  } else {
    m = (_Float16)smf(mref)[row];
  }
  float l = 0.f;
  constexpr uint32_t U = 4;
  for (uint32_t w = 0; w < 64; w += U) {
    uint32_t v[U];
#pragma unroll
    for (uint32_t u = 0; u < U; u++) v[u] = S[(w + u + lane) & 63];
#pragma unroll
    for (uint32_t u = 0; u < U; u++) {
      const _Float16 a = bf(v[u] & 0xFFFF) * sc, b = bf(v[u] >> 16) * sc;
      _Float16 xa = (_Float16)(a - m), xb = (_Float16)(b - m);
      if (EXACT) {                                       // fpEX wraps below ~-355; e^-64 ~ 1.6e-28 stands in for 0
        xa = __builtin_fmaxf16(xa, bf(0xC280u));
        xb = __builtin_fmaxf16(xb, bf(0xC280u));
      }
      uint32_t ea = bits(mu_fexp(xa)), eb = bits(mu_fexp(xb));
      if (MASK) {
        const int32_t idx = (int32_t)((w + u + lane) & 63);
        ea &= opaque(0u - (uint32_t)(2 * idx <= lim));
        eb &= opaque(0u - (uint32_t)(2 * idx + 1 <= lim));
      }
      l += f32(bf(ea)) + f32(bf(eb));
      S[(w + u + lane) & 63] = ea | (eb << 16);
    }
  }
  const float lt = (j == 0 ? 0.f : smf(lsm)[row]) + l;
  smf(lsm)[row] = lt;
  if (j == NB - 1) smf(lsm + INV_OFF)[row] = 1.0f / lt;
}

// ---- O: normalized and MX-quantized into oproj's A operand by the SIMT warps (fa_phase.h's norm_o_block: RPC rows
// per call, a lane per 32-group); same SMEM O layout (rows 2 D bytes apart at O_SM) and 1 / l at LSUM + INV_OFF ----
using FaD = fa::Fa<D>;
static_assert(FaD::O_PITCH == 2 * D && O_ROW == fa::O_ROW && LSUM[0] == fa::LSUM[0] && LSUM[1] == fa::LSUM[1] &&
              INV_OFF == fa::INV_OFF, "fa_sa and fa_phase.h O / LSUM maps must agree");
constexpr uint32_t O_BLOCKS = FaD::O_BLOCKS;

// ---- the phase (the standalone's entry; every thread of the cluster calls it) ----
static void phase(const volatile uint32_t *d, uint32_t tid, uint32_t tpb, uint32_t tb, uint32_t &acc_parity) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  const bool lead = tid == 0;
  const uint32_t OA = d[8], OASC = d[20], OTM = d[21], OTK = d[22], OKS = d[23];
  auto tile = [tb](uint32_t i) { return i * NIGHTLY_CLUSTERS + tb; };
  // a macro, not a [&] lambda: a non-inlined closure reads its captures through stack pointers that miss Muon's
  // per-lane stack interleave (fa_phase.h's FA_NORM_O)
#define FA_SA_NORM_O(qt, lsm, blk) FaD::norm_o_block(lane, (lsm), (blk), ((qt) % (SQ / QT)) * QT, head_of(qt), OA, OASC, OTM, OTK, OKS)
  const uint32_t qt0 = tile(0);
  auto issue_k_load = [&](uint32_t b) {
    const uint32_t ib = b / NB, jb = b % NB;
    const bool o_tile = jb == 0 && ib > 0, qh = (jb == 1 || jb == 2) && ib + 1 < QT_C;
    load_kq(head_of(tile((b + 2) / NB)), (b + 2) % NB, b >= 2 && !o_tile,
            qh ? q_half_src(tile(ib + 1), jb - 1) : 0, QA[(ib + 1) & 1] + QH_ROWS * (jb - 1));
  };
  const bool smx = warp >= 2 && warp <= 5;
  const uint32_t row = (warp - 2) * 16 + lane;           // softmax warps: the tile row of this lane
  // EXACT: f32 [clusters][QT_C][QT] row maxima in DRAM, explicit per-lane addresses (fa_phase.h's mrow)
  volatile float *mrow = (volatile float *)(d[25] + (tb * QT_C * QT + row) * 4);
  if (lead) {
    q_ = d[2]; qsc_ = d[3]; kt_ = d[4]; ksc_ = d[5]; v_ = d[6]; vsc_ = d[7];
    kt_bs_ = d[16]; ksc_bs_ = d[17]; v_bs_ = d[18]; vsc_bs_ = d[19];
    if (acc_parity) toggle();                            // the GEMM phases' compute half -> 0
    acc_parity = 0;
    mx::config_ld(D, 0);
    mx::config_ld(SK, 1);
  } else asm volatile("nop");
  // ---- EXACT pre-pass (fa_phase.h's): per tile, QK of every key block into SP[j & 1]; the softmax warps take each
  // row's masked raw max; NB (even) QK + S-store loops leave the compute / store halves at 0 for the prologue ----
  if (EXACT && warp == 0) {
    for (uint32_t i = 0; i < QT_C; i++) {
      const uint32_t qt = tile(i), h = head_of(qt);
      for (uint32_t j = 0; j < NB; j++) {
        if (lead) {
          if (j == 0) { load_q(qt, 0); load_q_scales(qt); }
          load_k(h, j);
          cfg_qk(false);
          qk_store(0, SP[j & 1]);
          mx::fence();                                   // S_j in SMEM; K buffer and scales free again
        } else asm volatile("nop");
        fa_bar(BAR_S, 5);
        fa_bar(BAR_P, 5);                                // the softmax warps have read S_j
      }
    }
  } else if (EXACT && smx) {
    constexpr uint32_t SCB = LL_FA_SCALE_BITS;
    const _Float16 sc = bf((SCB + 0x7FFFu + ((SCB >> 16) & 1u)) >> 16);
    for (uint32_t i = 0; i < QT_C; i++) {
      const uint32_t q0 = (tile(i) % (SQ / QT)) * QT;
      // bf16 -inf through opaque(): clang types _Float16 as IEEE half, where 0xFF80 is a NaN, so with NB a compile-time
      // constant (loop fully unrolled) it folded every `bm > m` to false and stored the constant 0xFF800000 (-inf) as
      // each row's max, discarding row_max (softmax with m = -inf: e^(s - m) = inf, l = inf, O * (1 / l) = 0).  fa_phase.h
      // only escaped through its runtime NB (m a loop phi)
      _Float16 m = bf(opaque(0xFF80u));
      for (uint32_t j = 0; j < NB; j++) {
        fa_bar(BAR_S, 5);
        const _Float16 bm = fa::row_max(row, lane, SP[j & 1], j * BK + BK - 1 > P0 + q0,
                                      (int32_t)(P0 + q0 + row) - (int32_t)(j * BK));
        m = bm > m ? bm : m;
        fa_bar(BAR_P, 5);
      }
      mrow[i * QT] = f32((_Float16)(m * sc));            // bf16(max * sc): the scaled max
    }
  } else asm volatile("nop");
  if (EXACT) mu_fence();                                 // the maxima are in DRAM before the softmax warps read them
  if (lead) {
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
        if (more2 && g > 0 && (g + 2) % NB == 0) load_q_scales(tile((g + 2) / NB));
      } else asm volatile("nop");
      if (j == 1 && i > 0) fa_bar(BAR_N, 4);
      if (j == 0 && i >= 2) fa_bar(BAR_ND, 4);
      if (more) fa_bar(BAR_S, 5);
      if (lead) {
        if (more2 && g > 0) {
          const bool o_tile = j == 0 && i > 0;
          if (g == 1) load_k_scales_gated(head_of(tile(3 / NB)), 3 % NB);
          load_v_scales_gated(head_of(tile(g / NB)), g % NB);
          if (more3) load_k_scales_gated(head_of(tile((g + 3) / NB)), (g + 3) % NB);
          if (o_tile) store_o();
          cfg_qk_managed();
        }
        if (g > 0) spad_requant(g & 1);
        if (more2 && g > 0) {
          qk_store(((g + 2) / NB) & 1, SP[g & 1]);
          n_qk = mx::loops_issued();
        } else {
          if (g > 0) toggle();
          load_v_scales_gated(head_of(tile(g / NB)), g % NB);
        }
        load_v_data(head_of(tile(g / NB)), g % NB);
        cfg_pv_managed();
        pv(g & 1, g % NB != 0);
        if (more3) issue_k_load(g + 1);
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
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t i = g / NB, j = g % NB;
      const uint32_t q0 = (tile(i) % (SQ / QT)) * QT;
      fa_bar(BAR_S, 5);
      const int32_t lim = (int32_t)(P0 + q0 + row) - (int32_t)(j * BK);
      const float m_ex = (EXACT && j == 0) ? mrow[i * QT] : 0.f;
      if (j * BK + BK - 1 > P0 + q0) softmax_row<true>(row, j, lane, SP[g & 1], MREF[i & 1], LSUM[i & 1], lim, m_ex);
      else softmax_row<false>(row, j, lane, SP[g & 1], MREF[i & 1], LSUM[i & 1], lim, m_ex);
      mu_fence_smem();
      fa_bar(BAR_P, 5);
    }
  } else {
    const uint32_t nw = warp == 1 ? 0 : warp - 5;
    for (uint32_t i = 1; i < QT_C; i++) {
      fa_bar(BAR_N, 4);
      for (uint32_t blk = nw; blk < O_BLOCKS; blk += 3) FA_SA_NORM_O(tile(i - 1), LSUM[(i - 1) & 1], blk);
      fa_bar(BAR_ND, 4);
    }
  }
  fa_bar(BAR_O, warps);
  for (uint32_t blk = warp; blk < O_BLOCKS; blk += warps) FA_SA_NORM_O(tile(QT_C - 1), LSUM[(QT_C - 1) & 1], blk);
  fa_bar(BAR_T, warps);
#undef FA_SA_NORM_O
}

}  // namespace fa_sa
#endif
