// fa_phase.h: the ATTN phase of llama_e2e, nightly/fa_mxfp8_sr (v7h, SPAD_REQUANT, retire-counter producer) as a
// phase: Q, K^T, V and their scales from the phase descriptor, GQA (query head h reads KV head h / Gq), O written
// straight into the [M][Dq] bf16 buffer that oquant reads, and the causal mask of the prefill chunk (query t sees keys
// 0 .. P0 + t; key block 0 must be unmasked, P0 >= 127).  d = 128, 128-key blocks, 64-query tiles, 8 warps per cluster.
//
// Layouts (per KV head g): K^T codes [hd][SK] at kt + g kt_bs, K scales [NB][hd/32][128] at ksc + g ksc_bs; V codes
// [SK][hd] at v + g v_bs, V scales [NB][128/32][hd] at vsc + g vsc_bs.  Q codes [Hq][M][hd] at q, Q scales
// [Hq M / 64][hd/32][64] at qsc.  O: MX A operand of oproj (codes at d[8], scales d[20], TM, TK, KS d[21..23]).
//
// Accumulator halves: FA starts and ends with the compute and store halves at 0.  The GEMM phases never store through a
// loop (store half untouched) and track the compute half in acc_parity; the phase toggles it to 0 first.
#ifndef LL_FA_PHASE_H
#define LL_FA_PHASE_H

namespace fa {

constexpr uint32_t D = 128, BK = 128, QT = 64;
// scratchpad map (rows of 16 B), as in fa_mxfp8_sr
constexpr uint32_t K_END = 1024;
constexpr uint32_t V_START = 1024, V_END = 2048;
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
static_assert(NIGHTLY_OCC == 4, "warp roles assume 8 warps per cluster");

enum : uint32_t { BAR_P = 3, BAR_T = 6, BAR_S = 7, BAR_O = 8, BAR_N = 9, BAR_ND = 10, BAR_0 = 11 };

static uint32_t scale_sink[256] __attribute__((aligned(64)));   // requantizer / SPAD_REQUANT DRAM scale flush

// phase parameters, written and read by the producer thread only (lead)
static uint32_t q_, qsc_, kt_, ksc_, v_, vsc_, Gq_, SQ_, SK_, kt_bs_, ksc_bs_, v_bs_, vsc_bs_;

static inline volatile __shared uint32_t *sm32(uint32_t byte) { return (volatile __shared uint32_t *)byte; }
static inline volatile __shared float *smf(uint32_t byte) { return (volatile __shared float *)byte; }
static inline _Float16 bf(uint32_t bits16) { return __builtin_bit_cast(_Float16, (uint16_t)bits16); }
static inline uint32_t bits(_Float16 x) { return __builtin_bit_cast(uint16_t, x); }
static inline float f32(_Float16 x) { return __builtin_bit_cast(float, bits(x) << 16); }

static inline uint32_t head_of(uint32_t qt, uint32_t SQ) { return qt / (SQ / QT); }

// ---- producer helpers (lane 0 of warp 0) -----------------------------------------------------
static inline uint32_t kt_base(uint32_t h) { return kt_ + (h / Gq_) * kt_bs_; }
static inline uint32_t ksc_base(uint32_t h) { return ksc_ + (h / Gq_) * ksc_bs_; }
static inline uint32_t v_base(uint32_t h) { return v_ + (h / Gq_) * v_bs_; }
static inline uint32_t vsc_base(uint32_t h) { return vsc_ + (h / Gq_) * vsc_bs_; }

static inline void load_k_data(uint32_t h, uint32_t j) {
  mx::set_ab_dram(0, mx::host_addr(kt_base(h) + j * BK), D, SK_);
  mx::loop_ws_spad(4, 8, 8, DUMMY_A, K_END, 0, false, false, {true, false, true, true, true});
}
static inline void load_kq(uint32_t h, uint32_t j, bool flip, uint32_t q_src, uint32_t q_row) {
  mx::set_ab_dram(q_src ? mx::host_addr(q_src) : 0, mx::host_addr(kt_base(h) + j * BK), D, SK_);
  mx::loop_ws_spad(2, 8, 8, q_src ? q_row : DUMMY_A, K_END, flip ? FLIP_ROW : 0, false, true,
                   {q_src == 0, false, true, true, !flip});
}
static inline void load_k(uint32_t h, uint32_t j) {
  load_k_data(h, j);
  mx::load_scales(mx::host_addr(ksc_base(h) + j * 512), 512, true, 0);
}
static inline void load_v_data(uint32_t h, uint32_t j) {
  mx::set_ab_dram(mx::host_addr(v_base(h) + j * BK * D), 0, D, SK_);
  mx::loop_ws_spad(8, 4, 8, V_START, K_END, 0, false, true, {false, true, true, true, true});
}
static inline uint32_t q_tile_src(uint32_t qt) {   // query row 0 of tile qt
  return q_ + (head_of(qt, SQ_) * SQ_ + (qt % (SQ_ / QT)) * QT) * D;
}
static inline void load_q(uint32_t qt, uint32_t b) {
  mx::set_ab_dram(mx::host_addr(q_tile_src(qt)), 0, D, SK_);
  mx::loop_ws_spad(4, 4, 8, QA[b], K_END, 0, false, false, {false, true, true, true, true});
}
static inline uint32_t q_half_src(uint32_t qt, uint32_t half) { return q_tile_src(qt) + 32 * half * D; }
static inline void load_q_scales(uint32_t qt) {
  mx::load_scales(mx::host_addr(qsc_ + qt * 256), 256, false, mx::SF_BUF1);
}
static inline void load_k1_vbuf(uint32_t h) {
  mx::set_ab_dram(0, mx::host_addr(kt_base(h) + 1 * BK), D, SK_);
  mx::loop_ws_spad(4, 8, 8, DUMMY_A, V_END, 0, false, false, {true, false, true, true, true});
  mx::load_scales(mx::host_addr(ksc_base(h) + 1 * 512), 512, true, mx::SF_BUF1);
}
static inline void qk_vbuf(uint32_t b) {
  mx::config_scale(4, 8, 8, 1, 1, mx::host_addr(scale_sink));
  mx::loop_ws_spad(4, 8, 8, QA[b], V_END, 0, false, true, {true, true, true, false, true});
}
static inline void qk(uint32_t b) { mx::loop_ws_spad(4, 8, 8, QA[b], K_END, 0, false, true, {true, true, true, false, true}); }
static inline void qk_store(uint32_t b, uint32_t sm) {
  mx::loop_ws_spad(4, 8, 8, QA[b], K_END, sm / 16, false, true, {true, true, true, false, false});
}
static inline void cfg_qk(bool resident) { mx::config_scale(4, 8, 8, 1, 0, mx::host_addr(scale_sink), resident, false); }
static inline void cfg_pv(bool resident, bool reset) {
  mx::config_scale(4, 8, 8, 0, 1, mx::host_addr(scale_sink), resident, reset);
}
static inline void load_k_scales_gated(uint32_t h, uint32_t j) {
  mx::cmd(mx::MX_LOAD_SCALES, mx::host_addr(ksc_base(h) + j * 512) & 0xFFFFFFFFFFull, (1ull << 54) | (1ull << 32) | 512);
}
static inline void cfg_qk_managed() {
  const uint64_t rs1 = (0ull << 61) | (1ull << 60) | (8ull << 51) | (8ull << 42) | (4ull << 33) |
                       (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
  mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
}
static inline void load_v_scales_gated(uint32_t h, uint32_t j) {
  mx::cmd(mx::MX_LOAD_SCALES, mx::host_addr(vsc_base(h) + j * 512) & 0xFFFFFFFFFFull,
          (1ull << 54) | ((uint64_t)mx::SF_BUF1 << 33) | (1ull << 32) | 512);
}
static inline void cfg_pv_managed() {
  const uint64_t rs1 = (1ull << 61) | (0ull << 60) | (8ull << 51) | (8ull << 42) | (4ull << 33) |
                       (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
  mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
}
static inline void pv(uint32_t b, bool acc) { mx::loop_ws_spad(4, 8, 8, PA[b], V_END, 0, acc, true, {true, true, true, false, true}); }
static inline void store_s(uint32_t sm) { mx::loop_ws_spad(4, 8, 1, DUMMY_A, K_END, sm / 16, false, false, {true, true, true, true, false}); }
static inline void toggle() {   // junk compute: flips the compute half
  mx::config_scale(2, 2, 1, 0, 0, mx::host_addr(scale_sink));
  mx::loop_ws_spad(2, 2, 1, DUMMY_A, K_END, JUNK_ROW, false, true, {true, true, true, false, true});
}
static inline void store_only(uint32_t I, uint32_t J, uint32_t c_row) {
  mx::loop_ws_spad(I, J, 1, DUMMY_A, K_END, c_row, false, true, {true, true, true, true, false});
}
static inline void store_o() { cfg_pv(false, false); store_only(4, 8, O_ROW); }

constexpr uint32_t K_SPAD_REQUANT = 34;
static inline void spad_requant(uint32_t b) {
  const uint64_t scale_dram = mx::host_addr(scale_sink) & 0x1FFFFFFFFull;
  const uint64_t rs1 = (uint64_t)(SP[b] / 16) | ((uint64_t)PA[b] << 14) | (1ull << 28) | (1ull << 29) | (scale_dram << 30);
  mx::cmd(K_SPAD_REQUANT, rs1, (uint64_t)QT | ((uint64_t)BK << 16));
}

// ---- softmax and O ---------------------------------------------------------------------------
// One 128-key row per lane, in place (S bf16 -> P bf16).  m: the scaled max of key block 0.  MASK: keys with column
// c > lim (lim = P0 + query - j BK) are masked (P = 0).
template <bool MASK>
__attribute__((always_inline)) static inline float softmax_pass(volatile __shared uint32_t *S, uint32_t lane, _Float16 sc,
                                                                _Float16 m, int32_t lim) {
  float l = 0.f;
  constexpr uint32_t U = 4;
  for (uint32_t w = 0; w < 64; w += U) {
    uint32_t v[U];
#pragma unroll
    for (uint32_t u = 0; u < U; u++) v[u] = S[(w + u + lane) & 63];
#pragma unroll
    for (uint32_t u = 0; u < U; u++) {
      const uint32_t idx = (w + u + lane) & 63;
      const _Float16 a = bf(v[u] & 0xFFFF) * sc, b = bf(v[u] >> 16) * sc;
      uint32_t ea = bits(mu_fexp((_Float16)(a - m))), eb = bits(mu_fexp((_Float16)(b - m)));
      if (MASK) {
        ea &= opaque(0u - (uint32_t)((int32_t)(2 * idx) <= lim));
        eb &= opaque(0u - (uint32_t)((int32_t)(2 * idx + 1) <= lim));
      }
      l += f32(bf(ea)) + f32(bf(eb));
      S[idx] = ea | (eb << 16);
    }
  }
  return l;
}
static __attribute__((noinline)) void softmax_row(uint32_t row, uint32_t j, uint32_t lane, uint32_t base, uint32_t mref,
                                                  uint32_t lsm, uint32_t scb, uint32_t NB, uint32_t mask, int32_t lim) {
  const _Float16 sc = bf((scb + 0x7FFFu + ((scb >> 16) & 1u)) >> 16);
  volatile __shared uint32_t *S = sm32(base + row * 256);
  _Float16 m;
  if (j == 0) {
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
  const float l = mask ? softmax_pass<true>(S, lane, sc, m, lim) : softmax_pass<false>(S, lane, sc, m, lim);
  const float lt = (j == 0 ? 0.f : smf(lsm)[row]) + l;
  smf(lsm)[row] = lt;
  if (j == NB - 1) smf(lsm + INV_OFF)[row] = 1.0f / lt;
}

// O, normalized and MX-quantized straight into oproj's A operand (what oquant did): 4 rows per call, lane per
// 32-group (row 4 blk + lane / 4, d group lane % 4); codes and scale byte stored directly.  Bit-exact with oquant of
// the bf16 O: bf16(O * (1 / l)), then fp8 with the group's E8M0 scale.
constexpr uint32_t O_BLOCKS = QT / 4;
static __attribute__((noinline)) void norm_o_block(uint32_t lane, uint32_t lsm, uint32_t blk, uint32_t q0, uint32_t h,
                                                   uint32_t A, uint32_t ASC, uint32_t TM, uint32_t TK, uint32_t KS) {
  const uint32_t row = 4 * blk + lane / 4, k = lane % 4, t = q0 + row, col = h * D + 32 * k;
  volatile __shared uint32_t *src = sm32(O_SM + row * 256 + 64 * k);
  const float inv = smf(lsm + INV_OFF)[row];
  quant_group([&](uint32_t jj) __attribute__((always_inline)) {
                const uint32_t w = src[jj];
                const uint32_t a = bits((_Float16)(f32(bf(w & 0xFFFF)) * inv)), b = bits((_Float16)(f32(bf(w >> 16)) * inv));
                return F2{f_bits(a << 16), f_bits(b << 16)};
              },
              lane, (volatile uint32_t *)a_code(A, TM, TK, KS, t, col), a_scale(ASC, TM, TK, KS, t, col));
}

__attribute__((noinline, convergent)) static void fbar(uint32_t id, uint32_t warps) { mu_barrier(id, warps); }

// The phase: every thread of the cluster calls it.  acc_parity: the GEMMs' compute half (reset to 0 here).
static void phase(const volatile uint32_t *d, uint32_t tid, uint32_t tpb, uint32_t tb, uint32_t &acc_parity) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  const bool lead = tid == 0;
  const uint32_t OA = d[8], H = d[9], SQ = d[11], SK = d[12], P0 = d[13], scb = d[15];
  const uint32_t OASC = d[20], OTM = d[21], OTK = d[22], OKS = d[23];
  const uint32_t NB = SK / BK, NQT = H * SQ / QT, QT_C = NQT / NIGHTLY_CLUSTERS, G = QT_C * NB;
  auto tile = [tb](uint32_t i) { return i * NIGHTLY_CLUSTERS + tb; };
  auto norm_o = [&](uint32_t qt, uint32_t lsm, uint32_t blk) {
    norm_o_block(lane, lsm, blk, (qt % (SQ / QT)) * QT, head_of(qt, SQ), OA, OASC, OTM, OTK, OKS);
  };
  const uint32_t qt0 = tile(0);
  if (lead) {
    q_ = d[2]; qsc_ = d[3]; kt_ = d[4]; ksc_ = d[5]; v_ = d[6]; vsc_ = d[7]; Gq_ = d[10]; SQ_ = SQ; SK_ = SK;
    kt_bs_ = d[16]; ksc_bs_ = d[17]; v_bs_ = d[18]; vsc_bs_ = d[19];
  } else asm volatile("nop");
  auto issue_k_load = [&](uint32_t b) {
    const uint32_t ib = b / NB, jb = b % NB;
    const bool o_tile = jb == 0 && ib > 0, qh = (jb == 1 || jb == 2) && ib + 1 < QT_C;
    load_kq(head_of(tile((b + 2) / NB), SQ), (b + 2) % NB, b >= 2 && !o_tile,
            qh ? q_half_src(tile(ib + 1), jb - 1) : 0, QA[(ib + 1) & 1] + 256 * (jb - 1));
  };
  if (lead) {
    if (acc_parity) toggle();                            // compute half -> 0
    acc_parity = 0;
    mx::config_ld(D, 0);
    mx::config_ld(SK, 1);
    load_q(qt0, 0);
    load_q_scales(qt0);
    load_k(head_of(qt0, SQ), 0);
    load_k1_vbuf(head_of(qt0, SQ));
    mx::config_ex(mx::FP8, mx::FP8, mx::BF16);
    cfg_qk(false);
    qk(0);
    cfg_qk(false);
    store_s(SP[0]);
  } else asm volatile("nop");
  mu_fence_smem();
  fbar(BAR_0, warps);

  const bool smx = warp >= 2 && warp <= 5;
  if (warp == 0) {
    if (lead) mx::fence();
    else asm volatile("nop");
    fbar(BAR_S, 5);
    if (lead) {
      toggle();
      qk_vbuf(0);
    } else asm volatile("nop");
    fbar(BAR_P, 5);
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
          load_k(head_of(qt0, SQ), 2);
          cfg_qk(false);
          qk(0);
          store_s(SP[0]);
          n_qk = mx::loops_issued();
        } else if (more) {
          mx::wait_retired(n_qk);
        }
        if (more2 && g > 0 && (g + 2) % NB == 0) load_q_scales(tile((g + 2) / NB));
      } else asm volatile("nop");
      if (j == 1 && i > 0) fbar(BAR_N, 4);
      if (j == 0 && i >= 2) fbar(BAR_ND, 4);
      if (more) fbar(BAR_S, 5);
      if (lead) {
        if (more2 && g > 0) {
          const bool o_tile = j == 0 && i > 0;
          if (g == 1) load_k_scales_gated(head_of(tile(3 / NB), SQ), 3 % NB);
          load_v_scales_gated(head_of(tile(g / NB), SQ), g % NB);
          if (more3) load_k_scales_gated(head_of(tile((g + 3) / NB), SQ), (g + 3) % NB);
          if (o_tile) store_o();
          cfg_qk_managed();
        }
        if (g > 0) spad_requant(g & 1);
        if (more2 && g > 0) {
          qk_store(((g + 2) / NB) & 1, SP[g & 1]);
          n_qk = mx::loops_issued();
        } else {
          if (g > 0) toggle();
          load_v_scales_gated(head_of(tile(g / NB), SQ), g % NB);
        }
        load_v_data(head_of(tile(g / NB), SQ), g % NB);
        cfg_pv_managed();
        pv(g & 1, g % NB != 0);
        if (more3) issue_k_load(g + 1);
      } else asm volatile("nop");
      if (more) fbar(BAR_P, 5);
    }
    if (QT_C >= 2) fbar(BAR_ND, 4);
    if (lead) {
      mx::fence();
      store_o();
      mx::fence();
    } else asm volatile("nop");
  } else if (smx) {
    const uint32_t row = (warp - 2) * 16 + lane;
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t i = g / NB, j = g % NB;
      const uint32_t q0 = (tile(i) % (SQ / QT)) * QT;
      fbar(BAR_S, 5);
      softmax_row(row, j, lane, SP[g & 1], MREF[i & 1], LSUM[i & 1], scb, NB, j * BK + BK - 1 > P0 + q0,
                  (int32_t)(P0 + q0 + row) - (int32_t)(j * BK));
      mu_fence_smem();
      fbar(BAR_P, 5);
    }
  } else {
    const uint32_t nw = warp == 1 ? 0 : warp - 5;
    for (uint32_t i = 1; i < QT_C; i++) {
      fbar(BAR_N, 4);
      for (uint32_t blk = nw; blk < O_BLOCKS; blk += 3) norm_o(tile(i - 1), LSUM[(i - 1) & 1], blk);
      fbar(BAR_ND, 4);
    }
  }
  fbar(BAR_O, warps);
  for (uint32_t blk = warp; blk < O_BLOCKS; blk += warps) norm_o(tile(QT_C - 1), LSUM[(QT_C - 1) & 1], blk);
  fbar(BAR_T, warps);
}

}  // namespace fa
#endif
