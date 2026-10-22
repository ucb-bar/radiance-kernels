// fa_phase_vla.h: fa_phase.h for kernel_vla.cpp (SmolVLA), plus a DRAM row-max buffer for the exact pre-pass,
// heads per output row block and an O block offset (d[25], d[26], d[28] / d[29]), so one ATTN phase can run a slice
// of SigLIP's 3 images x 12 heads.  Keep the shared parts in step with fa_phase.h.
//
// fa_phase.h: the ATTN phase of llama_e2e, nightly/fa_mxfp8_sr (v7h, SPAD_REQUANT, retire-counter producer) as a
// phase: Q, K^T, V and their scales from the phase descriptor, GQA (query head h reads KV head h / Gq), O written
// straight into the [M][Dq] bf16 buffer that oquant reads, and the causal mask of the prefill chunk (query t sees keys
// 0 .. P0 + t; key block 0 must be unmasked, P0 >= 127).  Head dim D = 64 or 128 (Fa<D>), 128-key blocks, 64-query
// tiles, 8 warps per cluster.
//
// Softmax max (d[27]): 0 = upstream's lazy softmax, the max of key block 0 for every block of the tile; 1 = a pre-pass
// runs QK over every key block (fenced, one block at a time) and the softmax warps keep each row's exact masked max,
// so exp(s - m) <= 1 (DeepSeek: the max of key block 0 alone overflowed exp on Qwen2's large biased logits).
//
// Layouts (per KV head g): K^T codes [hd][SK] at kt + g kt_bs, K scales [NB][hd/32][128] at ksc + g ksc_bs; V codes
// [SK][hd] at v + g v_bs, V scales [NB][128/32][hd] at vsc + g vsc_bs.  Q codes [Hq][M][hd] at q, Q scales
// [Hq M / 64][hd/32][64] at qsc.  O: MX A operand of oproj (codes at d[8], scales d[20], TM, TK, KS d[21..23]).
//
// Accumulator halves: FA starts and ends with the compute and store halves at 0.  The GEMM phases never store through a
// loop (store half untouched) and track the compute half in acc_parity; the phase toggles it to 0 first.
#ifndef LL_FA_PHASE_H
#define LL_FA_PHASE_H
// (d[25] per-row maxima in DRAM beyond QT_MAX tiles, d[26] heads per output row block, d[28] / d[29] the O block's
// row / column offset: SmolVLA's SigLIP)

namespace fa {

constexpr uint32_t BK = 128, QT = 64;
constexpr uint32_t QT_MAX = 64;                          // query tiles per cluster (the softmax lanes' row maxima)
// scratchpad map (rows of 16 B), as in fa_mxfp8_sr; B operands are end-addressed (K block ends at K_END, the V block
// and the prologue's K_1 at V_END)
constexpr uint32_t K_END = 1024, V_END = 2048;
constexpr uint32_t QA[2] = {2048, 2560};
constexpr uint32_t PA[2] = {3072, 3584};
constexpr uint32_t SP[2] = {4096 * 16, 5120 * 16};
constexpr uint32_t O_ROW = 6656, O_SM = O_ROW * 16;
constexpr uint32_t JUNK_ROW = 7712;
constexpr uint32_t FLIP_ROW = 6144;
static_assert(FLIP_ROW + 512 <= O_ROW, "the store-flip sink (512 rows) must end below O");
constexpr uint32_t LSUM[2] = {7856 * 16 + 256, 7904 * 16 + 256};
constexpr uint32_t INV_OFF = 256;
constexpr uint32_t DUMMY_A = 0;
static_assert(LSUM[1] + 512 <= mx::SPAD_ROWS * 16, "scratchpad budget");
static_assert(NIGHTLY_MX_LOOPS_SMEM >= LSUM[1] + 512 && NIGHTLY_MX_LOOPS_SMEM + 4 <= mx::SPAD_ROWS * 16,
              "the LOOP_WS count word must not overlap the scratchpad map");
static_assert(NIGHTLY_OCC == 4, "warp roles assume 8 warps per cluster");

enum : uint32_t { BAR_P = 3, BAR_T = 6, BAR_S = 7, BAR_O = 8, BAR_N = 9, BAR_ND = 10, BAR_0 = 11 };

static inline volatile __shared uint32_t *sm32(uint32_t byte) { return (volatile __shared uint32_t *)byte; }
static inline volatile __shared float *smf(uint32_t byte) { return (volatile __shared float *)byte; }
static inline _Float16 bf(uint32_t bits16) { return __builtin_bit_cast(_Float16, (uint16_t)bits16); }
static inline uint32_t bits(_Float16 x) { return __builtin_bit_cast(uint16_t, x); }
static inline float f32(_Float16 x) { return __builtin_bit_cast(float, bits(x) << 16); }
static inline uint32_t head_of(uint32_t qt, uint32_t SQ) { return qt / (SQ / QT); }
static inline _Float16 bf_scale(uint32_t scb) { return bf((scb + 0x7FFFu + ((scb >> 16) & 1u)) >> 16); }

// ---- softmax (D-independent) -----------------------------------------------------------------
// One 128-key row per lane, in place (S bf16 -> P bf16).  MASK: keys with column c > lim (lim = P0 + query - j BK)
// are masked (P = 0).
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
      // fexp's argument clamped at -64 (bf16 0xC280): fpEX wraps below ~-355 (Q10.24, no underflow check)
      uint32_t ea = bits(mu_fexp(__builtin_fmaxf16((_Float16)(a - m), bf(0xC280u))));
      uint32_t eb = bits(mu_fexp(__builtin_fmaxf16((_Float16)(b - m), bf(0xC280u))));
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
// the masked raw max of one 128-key row of S (bf16, before the scale), -inf when every key is masked
static __attribute__((noinline)) _Float16 row_max(uint32_t row, uint32_t lane, uint32_t base, uint32_t mask, int32_t lim) {
  volatile __shared uint32_t *S = sm32(base + row * 256);
  const uint32_t NEGW = 0xFF80u;                         // bf16 -inf
  _Float16 mx = bf(NEGW);
  for (uint32_t w = 0; w < 64; w += 8) {
    uint32_t v[8];
#pragma unroll
    for (uint32_t u = 0; u < 8; u++) v[u] = S[(w + u + lane) & 63];
#pragma unroll
    for (uint32_t u = 0; u < 8; u++) {
      const uint32_t idx = (w + u + lane) & 63;
      const uint32_t ka = opaque((uint32_t)(mask == 0) | (uint32_t)((int32_t)(2 * idx) <= lim));
      const uint32_t kb = opaque((uint32_t)(mask == 0) | (uint32_t)((int32_t)(2 * idx + 1) <= lim));
      const uint32_t ma = 0u - ka, mb = 0u - kb;
      const _Float16 a = bf(((v[u] & 0xFFFF) & ma) | (NEGW & ~ma)), b = bf(((v[u] >> 16) & mb) | (NEGW & ~mb));
      mx = a > mx ? a : mx;
      mx = b > mx ? b : mx;
    }
  }
  return mx;
}
// m: the row's scaled exact max (pre-pass)
static __attribute__((noinline)) void softmax_row(uint32_t row, uint32_t j, uint32_t lane, uint32_t base, float m_in,
                                                  uint32_t lsm, uint32_t scb, uint32_t NB, uint32_t mask, int32_t lim) {
  const _Float16 sc = bf_scale(scb);
  volatile __shared uint32_t *S = sm32(base + row * 256);
  const _Float16 m = (_Float16)m_in;
  const float l = mask ? softmax_pass<true>(S, lane, sc, m, lim) : softmax_pass<false>(S, lane, sc, m, lim);
  const float lt = (j == 0 ? 0.f : smf(lsm)[row]) + l;
  smf(lsm)[row] = lt;
  if (j == NB - 1) smf(lsm + INV_OFF)[row] = 1.0f / lt;
}

__attribute__((noinline, convergent)) static void fbar(uint32_t id, uint32_t warps) { mu_barrier(id, warps); }

constexpr uint32_t K_SPAD_REQUANT = 34;

template <uint32_t D>
struct Fa {
  static_assert(D == 64 || D == 128, "FA head dim");
  static constexpr uint32_t KD = D / 16;                 // QK depth tiles, PV width tiles
  static constexpr uint32_t K_ROWS = D * BK / 16, V_ROWS = BK * D / 16, V_START = V_END - V_ROWS;
  static constexpr uint32_t QH_ROWS = (QT / 2) * D / 16; // one Q half (32 queries)
  static constexpr uint32_t KSC = D / 32 * BK, VSC = BK / 32 * D, QSC = D / 32 * QT;   // scale bytes per block / tile
  static constexpr uint32_t O_PITCH = 2 * D;             // O bf16 row pitch in SMEM
  static constexpr uint32_t RPC = 16 / (D / 32), O_BLOCKS = QT / RPC;   // norm_o_block: rows per call
  static_assert(K_ROWS <= K_END && V_ROWS <= V_END - K_END && QT * D / 16 <= QA[1] - QA[0], "scratchpad map");
  static_assert(O_ROW + QT * O_PITCH / 16 <= JUNK_ROW, "O below the junk sink");

  // phase parameters, written and read by the producer thread only (lead)
  static inline uint32_t q_, qsc_, kt_, ksc_, v_, vsc_, Gq_, SQ_, SK_, kt_bs_, ksc_bs_, v_bs_, vsc_bs_;
  static inline uint32_t scale_sink[256] __attribute__((aligned(64)));   // requantizer / SPAD_REQUANT scale flush

  // ---- producer helpers (lane 0 of warp 0) -----------------------------------------------------
  static uint32_t kt_base(uint32_t h) { return kt_ + (h / Gq_) * kt_bs_; }
  static uint32_t ksc_base(uint32_t h) { return ksc_ + (h / Gq_) * ksc_bs_; }
  static uint32_t v_base(uint32_t h) { return v_ + (h / Gq_) * v_bs_; }
  static uint32_t vsc_base(uint32_t h) { return vsc_ + (h / Gq_) * vsc_bs_; }

  static void load_k_data(uint32_t h, uint32_t j) {
    mx::set_ab_dram(0, mx::host_addr(kt_base(h) + j * BK), D, SK_);
    mx::loop_ws_spad(4, 8, KD, DUMMY_A, K_END, 0, false, false, {true, false, true, true, true});
  }
  static void load_kq(uint32_t h, uint32_t j, bool flip, uint32_t q_src, uint32_t q_row) {
    mx::set_ab_dram(q_src ? mx::host_addr(q_src) : 0, mx::host_addr(kt_base(h) + j * BK), D, SK_);
    mx::loop_ws_spad(2, 8, KD, q_src ? q_row : DUMMY_A, K_END, flip ? FLIP_ROW : 0, false, true,
                     {q_src == 0, false, true, true, !flip});
  }
  static void load_k(uint32_t h, uint32_t j) {
    load_k_data(h, j);
    mx::load_scales(mx::host_addr(ksc_base(h) + j * KSC), KSC, true, 0);
  }
  static void load_v_data(uint32_t h, uint32_t j) {
    mx::set_ab_dram(mx::host_addr(v_base(h) + j * BK * D), 0, D, SK_);
    mx::loop_ws_spad(8, 4, KD, V_START, K_END, 0, false, true, {false, true, true, true, true});
  }
  static uint32_t q_tile_src(uint32_t qt) { return q_ + (head_of(qt, SQ_) * SQ_ + (qt % (SQ_ / QT)) * QT) * D; }
  static void load_q(uint32_t qt, uint32_t b) {
    mx::set_ab_dram(mx::host_addr(q_tile_src(qt)), 0, D, SK_);
    mx::loop_ws_spad(4, 4, KD, QA[b], K_END, 0, false, false, {false, true, true, true, true});
  }
  static uint32_t q_half_src(uint32_t qt, uint32_t half) { return q_tile_src(qt) + 32 * half * D; }
  static void load_q_scales(uint32_t qt) { mx::load_scales(mx::host_addr(qsc_ + qt * QSC), QSC, false, mx::SF_BUF1); }
  static void load_k1_vbuf(uint32_t h) {
    mx::set_ab_dram(0, mx::host_addr(kt_base(h) + 1 * BK), D, SK_);
    mx::loop_ws_spad(4, 8, KD, DUMMY_A, V_END, 0, false, false, {true, false, true, true, true});
    mx::load_scales(mx::host_addr(ksc_base(h) + 1 * KSC), KSC, true, mx::SF_BUF1);
  }
  static void qk_vbuf(uint32_t b) {
    mx::config_scale(4, 8, KD, 1, 1, mx::host_addr(scale_sink));
    mx::loop_ws_spad(4, 8, KD, QA[b], V_END, 0, false, true, {true, true, true, false, true});
  }
  static void qk(uint32_t b) { mx::loop_ws_spad(4, 8, KD, QA[b], K_END, 0, false, true, {true, true, true, false, true}); }
  static void qk_store(uint32_t b, uint32_t sm) {
    mx::loop_ws_spad(4, 8, KD, QA[b], K_END, sm / 16, false, true, {true, true, true, false, false});
  }
  static void cfg_qk(bool resident) { mx::config_scale(4, 8, KD, 1, 0, mx::host_addr(scale_sink), resident, false); }
  static void cfg_pv(bool resident, bool reset) {
    mx::config_scale(4, KD, 8, 0, 1, mx::host_addr(scale_sink), resident, reset);
  }
  static void load_k_scales_gated(uint32_t h, uint32_t j) {
    mx::cmd(mx::MX_LOAD_SCALES, mx::host_addr(ksc_base(h) + j * KSC) & 0xFFFFFFFFFFull, (1ull << 54) | (1ull << 32) | KSC);
  }
  static void cfg_qk_managed() {   // I, J, K = 4, 8, KD; act half 1 (Q), weight half 0 (K)
    const uint64_t rs1 = (0ull << 61) | (1ull << 60) | ((uint64_t)KD << 51) | (8ull << 42) | (4ull << 33) |
                         (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
    mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
  }
  static void load_v_scales_gated(uint32_t h, uint32_t j) {
    mx::cmd(mx::MX_LOAD_SCALES, mx::host_addr(vsc_base(h) + j * VSC) & 0xFFFFFFFFFFull,
            (1ull << 54) | ((uint64_t)mx::SF_BUF1 << 33) | (1ull << 32) | VSC);
  }
  static void cfg_pv_managed() {   // I, J, K = 4, KD, 8; act half 0 (P), weight half 1 (V)
    const uint64_t rs1 = (1ull << 61) | (0ull << 60) | (8ull << 51) | ((uint64_t)KD << 42) | (4ull << 33) |
                         (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
    mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
  }
  static void pv(uint32_t b, bool acc) { mx::loop_ws_spad(4, KD, 8, PA[b], V_END, 0, acc, true, {true, true, true, false, true}); }
  static void store_s(uint32_t sm) { mx::loop_ws_spad(4, 8, 1, DUMMY_A, K_END, sm / 16, false, false, {true, true, true, true, false}); }
  static void toggle() {   // junk compute: flips the compute half
    mx::config_scale(2, 2, 1, 0, 0, mx::host_addr(scale_sink));
    mx::loop_ws_spad(2, 2, 1, DUMMY_A, K_END, JUNK_ROW, false, true, {true, true, true, false, true});
  }
  static void store_only(uint32_t I, uint32_t J, uint32_t c_row) {
    mx::loop_ws_spad(I, J, 1, DUMMY_A, K_END, c_row, false, true, {true, true, true, true, false});
  }
  static void store_o() { cfg_pv(false, false); store_only(4, KD, O_ROW); }
  static void spad_requant(uint32_t b) {
    const uint64_t scale_dram = mx::host_addr(scale_sink) & 0x1FFFFFFFFull;
    const uint64_t rs1 = (uint64_t)(SP[b] / 16) | ((uint64_t)PA[b] << 14) | (1ull << 28) | (1ull << 29) | (scale_dram << 30);
    mx::cmd(K_SPAD_REQUANT, rs1, (uint64_t)QT | ((uint64_t)BK << 16));
  }

  // O, normalized and MX-quantized straight into oproj's A operand (what oquant did): RPC rows per call, lane per
  // 32-group (row RPC blk + lane / (D/32), d group lane % (D/32)); codes and scale byte stored directly.  Bit-exact with
  // oquant of the bf16 O: bf16(O * (1 / l)), then fp8 with the group's E8M0 scale.
  // HPB (d[26], 0 = all heads): heads per output row block; head h writes rows (h / HPB) SQ + t, columns (h % HPB) D
  // (SigLIP: 3 images x 12 heads as [3 x 1024 tokens][12 x 64])
  // always_inline: as a call its 10 / 14 arguments put 2 / 6 on the stack, and the Muon compiler stores outgoing stack
  // arguments at unscaled offsets (0, 4, 8, ...) while the callee reads them at the stack-word-stride-scaled ones
  // (0x00, 0x40, ...): every argument after the 9th arrived stale (SmolVLA's ROW0 / COL0 read 0, so the vision ATTN
  // phases overwrote one O block; the LLM's KS only matched by luck).  Keep non-inlined Muon functions at <= 8 args.
  static __attribute__((always_inline)) inline void norm_o_block(uint32_t lane, uint32_t lsm, uint32_t blk, uint32_t q0, uint32_t h,
                                                     uint32_t A, uint32_t ASC, uint32_t TM, uint32_t TK, uint32_t KS,
                                                     uint32_t HPB, uint32_t SQ, uint32_t ROW0, uint32_t COL0) {
    constexpr uint32_t NGR = D / 32;
    const uint32_t row = RPC * blk + lane / NGR, k = lane % NGR, t = ROW0 + (h / HPB) * SQ + q0 + row,
                   col = COL0 + (h % HPB) * D + 32 * k;
    volatile __shared uint32_t *src = sm32(O_SM + row * O_PITCH + 64 * k);
    const float inv = smf(lsm + INV_OFF)[row];
    quant_group([&](uint32_t jj) __attribute__((always_inline)) {
                  const uint32_t w = src[jj];
                  const uint32_t a = bits((_Float16)(f32(bf(w & 0xFFFF)) * inv)), b = bits((_Float16)(f32(bf(w >> 16)) * inv));
                  return F2{f_bits(a << 16), f_bits(b << 16)};
                },
                lane, (volatile uint32_t *)a_code(A, TM, TK, KS, t, col), a_scale(ASC, TM, TK, KS, t, col));
  }

  // The phase: every thread of the cluster calls it.  acc_parity: the GEMMs' compute half (reset to 0 here).
  static void phase(const volatile uint32_t *d, uint32_t tid, uint32_t tpb, uint32_t tb, uint32_t &acc_parity) {
    const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
    const bool lead = tid == 0;
    const uint32_t OA = d[8], H = d[9], SQ = d[11], SK = d[12], P0 = d[13], scb = d[15];
    const uint32_t OASC = d[20], OTM = d[21], OTK = d[22], OKS = d[23], HPB = d[26] ? d[26] : H;
    const uint32_t ROW0 = d[28], COL0 = d[29];   // O block offset (a phase over a slice of the heads)
    const uint32_t NB = SK / BK, NQT = H * SQ / QT, QT_C = NQT / NIGHTLY_CLUSTERS, G = QT_C * NB;
    auto tile = [tb](uint32_t i) { return i * NIGHTLY_CLUSTERS + tb; };
    // not a [&] lambda: a non-inlined closure reads its captures through pointers into the thread's stack, which do
    // not get the per-lane interleave of Muon's stack (as runtime-indexed stack arrays): SmolVLA's ROW0 / COL0 read
    // as 0 and every 4-head phase wrote over the first one's O block
#define FA_NORM_O(qt, lsm, blk) norm_o_block(lane, (lsm), (blk), ((qt) % (SQ / QT)) * QT, head_of((qt), SQ), OA, OASC, OTM, OTK, OKS, HPB, SQ, ROW0, COL0)
    const uint32_t qt0 = tile(0);
    if (lead) {
      q_ = d[2]; qsc_ = d[3]; kt_ = d[4]; ksc_ = d[5]; v_ = d[6]; vsc_ = d[7]; Gq_ = d[10]; SQ_ = SQ; SK_ = SK;
      kt_bs_ = d[16]; ksc_bs_ = d[17]; v_bs_ = d[18]; vsc_bs_ = d[19];
    } else asm volatile("nop");
    auto issue_k_load = [&](uint32_t b) {
      const uint32_t ib = b / NB, jb = b % NB;
      const bool o_tile = jb == 0 && ib > 0, qh = (jb == 1 || jb == 2) && ib + 1 < QT_C;
      load_kq(head_of(tile((b + 2) / NB), SQ), (b + 2) % NB, b >= 2 && !o_tile,
              qh ? q_half_src(tile(ib + 1), jb - 1) : 0, QA[(ib + 1) & 1] + QH_ROWS * (jb - 1));
    };
    const bool smx = warp >= 2 && warp <= 5;
    // d[27] = 0: upstream's lazy softmax, every block of a tile uses the max of its key block 0 (no rescaling of O in
    // the accumulator).  Safe unless a later key's scaled logit exceeds it by ~88 (exp overflows); d[27] = 1 runs the
    // pre-pass for the exact max (Qwen2's biased logits overflowed it).
    const bool exact = d[27] != 0;
    const uint32_t row = (warp - 2) * 16 + lane;            // a softmax lane's query row within the tile
    // its exact scaled max per tile: on the stack for <= QT_MAX tiles, else in DRAM (d[25]: f32 [clusters][QT_C][64];
    // a lane reads back only what it wrote, on its own core)
    // not a stack array (fa_phase.h: a runtime-indexed stack array aliases across lanes): block-0 max in a register,
    // exact max (pre-pass) in DRAM at d[25] (f32 [clusters][QT_C][64])
    volatile float *mrow = (volatile float *)(d[25] + (tb * QT_C * QT + row) * 4);
    const uint32_t MST = QT;                                 // mrow[i * MST]: tile i's entry
    float mcur = 0.f;

    if (lead) {
      if (acc_parity) toggle();                              // compute half -> 0
      acc_parity = 0;
      mx::config_ld(D, 0);
      mx::config_ld(SK, 1);
      mx::config_ex(mx::FP8, mx::FP8, mx::BF16);
    } else asm volatile("nop");
    // ---- pre-pass (d[27] = 1, exact max: DeepSeek): the exact masked row max of every tile.  NB QK + S-store loops per tile flip the compute and the
    // store half NB times each (NB even: back to 0 for the pipeline's prologue).
    if (exact && warp == 0) {
      for (uint32_t i = 0; i < QT_C; i++) {
        const uint32_t qt = tile(i), h = head_of(qt, SQ);
        for (uint32_t j = 0; j < NB; j++) {
          if (lead) {
            if (j == 0) { load_q(qt, 0); load_q_scales(qt); }
            load_k(h, j);
            cfg_qk(false);
            qk_store(0, SP[j & 1]);
            mx::fence();                                     // S_j in SMEM; K buffer and scales free again
          } else asm volatile("nop");
          fbar(BAR_S, 5);
          fbar(BAR_P, 5);                                    // the softmax lanes have read S_j
        }
      }
    } else if (exact && smx) {
      const _Float16 sc = bf_scale(scb);
      for (uint32_t i = 0; i < QT_C; i++) {
        const uint32_t q0 = (tile(i) % (SQ / QT)) * QT;
        _Float16 m = bf(0xFF80u);
        for (uint32_t j = 0; j < NB; j++) {
          fbar(BAR_S, 5);
          const _Float16 bm = row_max(row, lane, SP[j & 1], j * BK + BK - 1 > P0 + q0,
                                      (int32_t)(P0 + q0 + row) - (int32_t)(j * BK));
          m = bm > m ? bm : m;
          fbar(BAR_P, 5);
        }
        mrow[i * MST] = f32((_Float16)(m * sc));             // bf16(max * sc): the scaled max (monotonic rounding)
      }
    } else asm volatile("nop");

    if (lead) {
      load_q(qt0, 0);
      load_q_scales(qt0);
      load_k(head_of(qt0, SQ), 0);
      load_k1_vbuf(head_of(qt0, SQ));
      cfg_qk(false);
      qk(0);
      cfg_qk(false);
      store_s(SP[0]);
    } else asm volatile("nop");
    mu_fence_smem();
    fbar(BAR_0, warps);

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
      for (uint32_t g = 0; g < G; g++) {
        const uint32_t i = g / NB, j = g % NB;
        const uint32_t q0 = (tile(i) % (SQ / QT)) * QT;
        fbar(BAR_S, 5);
#if defined(LL_FA_DUMP) && LL_FA_DUMP
        // d[30]: cluster 0's tile 0 (as fa_phase.h): S, P per key block [j][row][128] bf16, m, l (bf16 in the high
        // half), the unnormalized O [row][D] bf16 (norm warps)
        const uint32_t dmp = (tb == 0 && i == 0) ? d[30] : 0u;
        if (dmp) {
          volatile __shared uint32_t *srow = sm32(SP[g & 1] + row * 256);
          for (uint32_t w = 0; w < 64; w++) *(volatile uint32_t *)(dmp + (j * QT + row) * 256 + 4 * w) = srow[w];
        } else asm volatile("nop");
#endif
        if (!exact && j == 0)                              // block 0 is never masked (P0 >= 127)
          mcur = f32((_Float16)(row_max(row, lane, SP[g & 1], 0, 0) * bf_scale(scb)));
        else asm volatile("nop");
        if (exact && j == 0) mcur = mrow[i * MST];
        else asm volatile("nop");
        softmax_row(row, j, lane, SP[g & 1], mcur, LSUM[i & 1], scb, NB, j * BK + BK - 1 > P0 + q0,
                    (int32_t)(P0 + q0 + row) - (int32_t)(j * BK));
        mu_fence_smem();
#if defined(LL_FA_DUMP) && LL_FA_DUMP
        if (dmp) {
          volatile __shared uint32_t *prow = sm32(SP[g & 1] + row * 256);
          for (uint32_t w = 0; w < 64; w++) *(volatile uint32_t *)(dmp + 131072 + (j * QT + row) * 256 + 4 * w) = prow[w];
          if (j == NB - 1) {
            *(volatile uint32_t *)(dmp + 262144 + 4 * row) = __builtin_bit_cast(uint32_t, mcur) & 0xFFFF0000u;
            *(volatile uint32_t *)(dmp + 262400 + 4 * row) = __builtin_bit_cast(uint32_t, smf(LSUM[i & 1])[row]) & 0xFFFF0000u;
          } else asm volatile("nop");
        } else asm volatile("nop");
#endif
        fbar(BAR_P, 5);
      }
    } else {
      const uint32_t nw = warp == 1 ? 0 : warp - 5;
      for (uint32_t i = 1; i < QT_C; i++) {
        fbar(BAR_N, 4);
#if defined(LL_FA_DUMP) && LL_FA_DUMP
        if (tb == 0 && i == 1 && warp == 1 && d[30]) {     // tile 0's unnormalized O, as stored from the accumulator
          for (uint32_t w = lane; w < QT * O_PITCH / 4; w += MU_NUM_THREADS)
            *(volatile uint32_t *)(d[30] + 262656 + 4 * w) = sm32(O_SM)[w];
          // and the 1 / l the normalization reads, per row (f32 bits)
          for (uint32_t r = lane; r < QT; r += MU_NUM_THREADS)
            *(volatile uint32_t *)(d[30] + 262656 + QT * O_PITCH + 4 * r) = sm32(LSUM[0] + INV_OFF)[r];
        } else asm volatile("nop");
#endif
        for (uint32_t blk = nw; blk < O_BLOCKS; blk += 3) FA_NORM_O(tile(i - 1), LSUM[(i - 1) & 1], blk);
        fbar(BAR_ND, 4);
      }
    }
    fbar(BAR_O, warps);
    for (uint32_t blk = warp; blk < O_BLOCKS; blk += warps) FA_NORM_O(tile(QT_C - 1), LSUM[(QT_C - 1) & 1], blk);
#undef FA_NORM_O
    fbar(BAR_T, warps);
  }
};

}  // namespace fa
#endif
