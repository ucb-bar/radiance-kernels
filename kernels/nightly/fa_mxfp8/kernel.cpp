// MXFP8 flash attention on the nightly RTL: gemmini mesh for QK^T and PV, SIMT softmax, gemmini
// requantizer (GPU path) for P.  See gen_data.py for the math (lazy softmax: reference max from key
// block 0 of each query tile).
//
// Shapes: 64-row query tiles, 128-key blocks, d = 128.  QK (I=4, J=8, K=8), PV (4, 8, 8) and the
// 64 x 128 P feed share the requantizer coalescer's I/J bounds (the coalescer reads only I and J of
// the CONFIG_SCALE_MEM in force), so PV of block g-1 runs while P of block g is fed.  The resident
// scale flush writes A scale buffer 0 in one burst after the last block of a feed, so that PV may
// read P_{g-1}'s scales there during the feed: the feeder holds its last block until the producer
// has seen PV finish (FLAG).
//
// One block stream g = i * NB + j over all query tiles i of the cluster (key block j).  Block g:
//   producer (warp 0, lane 0):
//     [bf16] fence (QK_{g+1} done) | S_{g+1} store (and O of tile i-1 when j == 1) | fence |
//            release softmax_{g+1}
//     [fp8]  coalescer reset | CONFIG_EX(fp8) | fence | release the feeder | O mvouts, Q scales and
//            Q half (non-loop commands first) | PV_{g-1} | K_{g+2} load | fence (PV done) | FLAG |
//            QK_{g+2} | V_g load | wait re-tile_g
//   feeder (warp 1):            feed_g (P_g bf16 -> requantizer -> PFLAT fp8), last block after FLAG
//   softmax (warps 2, 3, 5, 7): softmax_{g+1} in place (S -> P) during feed_g; O normalization
//   re-tile (warps 4, 6):       PFLAT -> P tiles (the mesh's 16x16-tiled A layout) during feed_g
// Warp w runs on core w % 2, so the feeder's core has no polling re-tile warp.
//
// Accumulator: S in half 0, O in half 1 (128 rows each).  Computes alternate PV (half 1) and QK
// (half 0); a block without one of them gets a 2x2x1 junk compute (outside a feed only: it needs
// its own CONFIG_SCALE_MEM, the scale-read counters wrap only at full bounds).
//
// O of tile i is stored to SMEM (bf16, unnormalized) in block (i+1)*NB + 1, normalized in place by
// the softmax warps in key blocks 2..5 of tile i+1 (1/l precomputed per row) and copied to DRAM by
// scratchpad-source MVOUTs (32 B writes).  The last tile's O is handled after the stream.
#include <nightly/device.h>
#include <nightly/mx.h>
#include <nightly/verify.h>
#include "fa_data.h"

constexpr uint32_t D = FA_D, BK = FA_BK, QT = 64;
constexpr uint32_t NB = FA_SK / BK, NQT = FA_H * FA_SQ / QT;
static_assert(D == 128 && BK == 128, "the scratchpad and accumulator maps assume d = 128, 128-key blocks");
static_assert(FA_VPERM, "the feeder sends a block's even keys, then its odd keys: generate V in that order");
static_assert(NQT % NIGHTLY_CLUSTERS == 0, "query tiles must split evenly over clusters");
constexpr uint32_t QT_C = NQT / NIGHTLY_CLUSTERS, G = QT_C * NB;
static_assert(NB >= 6, "O groups are normalized in key blocks 2..5 of the next tile");
static_assert(NIGHTLY_OCC == 4, "warp roles assume 8 warps per cluster");

#ifndef FA_ILV   // 1: cluster c gets tiles c, c + NC, ...: clusters work on one head at a time (K/V hits in L2)
#define FA_ILV 1
#endif
#ifndef FA_PROF_J
#define FA_PROF_J 6   // block whose stage boundaries are stamped (cluster 0) into printBuf phases 2..7
#endif

// Scratchpad map (rows of 16 B).  4 banks of 2048 rows (bank = row / 2048); the mesh reads A and B
// in the same cycles, so A and B operands sit in different banks (in one bank a 64x128x128 loop
// took 5.1k cycles instead of 4115):
//   bank 0: K^T, V (B operands)   bank 1: Q x2, P tiles x2 (A operands)
//   bank 2: S/P x2                bank 3: P flat, O, small buffers
constexpr uint32_t K_END = 1024;                           // K^T block [0, 1024): QK B operand
constexpr uint32_t V_START = 1024, V_END = 2048;           // V block: PV B operand
constexpr uint32_t QA[2] = {2048, 2560};                   // Q tiles (QK A operand), by tile parity
constexpr uint32_t PA[2] = {3072, 3584};                   // P tiles (PV A operand), by block parity
constexpr uint32_t SP[2] = {4096 * 16, 5120 * 16};         // S bf16 [64][128] -> P bf16 in place, pitch 256 B
constexpr uint32_t PFLAT = 6144 * 16;                      // P fp8 row-major [64][128] (requantizer output)
constexpr uint32_t O_ROW = 6656, O_SM = O_ROW * 16;        // O bf16 [64][128]
constexpr uint32_t JUNK_ROW = 7712;                        // 32x32 bf16 sink for accumulator-toggle stores
constexpr uint32_t FLAG = 7842 * 16;                       // last PV done: sequence number of the block
constexpr uint32_t MREF[2] = {7856 * 16, 7904 * 16};       // m_ref f32 [64] by tile parity
constexpr uint32_t LSUM[2] = {MREF[0] + 256, MREF[1] + 256};   // l f32 [64] by tile parity
constexpr uint32_t INV_OFF = 256;                          // 1/l f32 [64] at LSUM[p] + INV_OFF
constexpr uint32_t DUMMY_A = 0;                            // A row of store-only and junk loops
static_assert(MREF[1] + 768 <= mx::SPAD_ROWS * 16, "scratchpad budget");
constexpr uint32_t POISON = 0xFFFFFFFFu;

enum : uint32_t { BAR_P = 3, BAR_R = 4, BAR_F = 5, BAR_T = 6, BAR_S = 7, BAR_O = 8 };

// dummy destination for the requantizer's DRAM scale flush (always issued)
static uint32_t scale_sink[64] __attribute__((aligned(64)));

// ---- SIMT helpers ----------------------------------------------------------------------------
static inline _Float16 bf(uint32_t bits16) { return __builtin_bit_cast(_Float16, (uint16_t)bits16); }
static inline uint32_t bits(_Float16 x) { return __builtin_bit_cast(uint16_t, x); }
static inline float f32(_Float16 x) { return __builtin_bit_cast(float, bits(x) << 16); }

static inline volatile __shared uint32_t *sm32(uint32_t byte) { return (volatile __shared uint32_t *)byte; }
static inline volatile __shared float *smf(uint32_t byte) { return (volatile __shared float *)byte; }

// Hide a value from the optimizer.  SimplifyCFG otherwise merges tests on the loop index into a
// `switch`, which the Vortex UnifyLoopExits pass rejects ("Unsupported block terminator").
static inline uint32_t opaque(uint32_t x) { asm volatile("" : "+r"(x)); return x; }

// A barrier after lane-guarded code must not be inlined: clang tail-duplicated such barriers into
// both arms of the branch (the warp then runs it twice and hangs).
__attribute__((noinline, convergent)) static void fa_bar(uint32_t id, uint32_t warps) { mu_barrier(id, warps); }

static inline uint32_t head_of(uint32_t qt) { return qt / (FA_SQ / QT); }

// ---- producer helpers (lane 0 of warp 0) -----------------------------------------------------
// FA_SPREAD (gen --spread): odd heads' K, V and scales live in region 3, indexed by h / 2
#if FA_SPREAD
static inline uint32_t kt_base(uint32_t h) { return ((h & 1) ? FA_KT1_ADDR : FA_KT_ADDR) + (h >> 1) * D * FA_SK; }
static inline uint32_t ksc_base(uint32_t h) { return ((h & 1) ? FA_KSC1_ADDR : FA_KSC_ADDR) + (h >> 1) * NB * 512; }
static inline uint32_t v_base(uint32_t h) { return ((h & 1) ? FA_V1_ADDR : FA_V_ADDR) + (h >> 1) * FA_SK * D; }
static inline uint32_t vsc_base(uint32_t h) { return ((h & 1) ? FA_VSC1_ADDR : FA_VSC_ADDR) + (h >> 1) * NB * 512; }
#else
static inline uint32_t kt_base(uint32_t h) { return FA_KT_ADDR + h * D * FA_SK; }
static inline uint32_t ksc_base(uint32_t h) { return FA_KSC_ADDR + h * NB * 512; }
static inline uint32_t v_base(uint32_t h) { return FA_V_ADDR + h * FA_SK * D; }
static inline uint32_t vsc_base(uint32_t h) { return FA_VSC_ADDR + h * NB * 512; }
#endif

static inline void load_k(uint32_t h, uint32_t j) {   // K^T block (B loader) + its scales (B buffer 0)
  mx::set_ab_dram(0, mx::host_addr(kt_base(h) + j * BK), D, FA_SK);
  mx::loop_ws_spad(4, 8, 8, DUMMY_A, K_END, 0, false, false, {true, false, true, true, true});
  mx::load_scales(mx::host_addr(ksc_base(h) + j * 512), 512, true, 0);
}
// V [128 keys][128] is the PV B operand: tile (k, n) at V_START + (k*8 + n)*16.  The A loader
// produces exactly that order with I = 8 (key tiles) and K = 8 (d tiles), and shares the A queue's
// CONFIG_LD stride (D) with Q, leaving the B queue's stride to K^T.  Scales -> B buffer 1.
static inline void load_v(uint32_t h, uint32_t j) {
  mx::set_ab_dram(mx::host_addr(v_base(h) + j * BK * D), 0, D, FA_SK);
  mx::loop_ws_spad(8, 4, 8, V_START, K_END, 0, false, false, {false, true, true, true, true});
  mx::load_scales(mx::host_addr(vsc_base(h) + j * 512), 512, true, mx::SF_BUF1);
}
// Q is double-buffered by tile parity (the next tile's Q loads early, in two halves); its A scales
// (buffer 1) are loaded only once the current tile's last QK is done.
static inline void load_q(uint32_t qt, uint32_t b) {
  const uint32_t h = head_of(qt), q0 = (qt % (FA_SQ / QT)) * QT;
  mx::set_ab_dram(mx::host_addr(FA_Q_ADDR + (h * FA_SQ + q0) * D), 0, D, FA_SK);
  mx::loop_ws_spad(4, 4, 8, QA[b], K_END, 0, false, false, {false, true, true, true, true});
}
static inline void load_q_half(uint32_t qt, uint32_t b, uint32_t half) {   // query rows 32*half..+31
  const uint32_t h = head_of(qt), q0 = (qt % (FA_SQ / QT)) * QT + 32 * half;
  mx::set_ab_dram(mx::host_addr(FA_Q_ADDR + (h * FA_SQ + q0) * D), 0, D, FA_SK);
  // A tile (i, k) at a_start + (i * 8 + k) * 16: rows of tiles i = 2 half, 2 half + 1
  mx::loop_ws_spad(2, 4, 8, QA[b] + 256 * half, K_END, 0, false, false, {false, true, true, true, true});
}
static inline void load_q_scales(uint32_t qt) {
  mx::load_scales(mx::host_addr(FA_QSC_ADDR + qt * 256), 256, false, mx::SF_BUF1);
}
// prologue: K_1 goes to the V buffer (free until V_0) with its scales in B buffer 1, so QK_1 need not
// wait for QK_0 to release the K buffer
static inline void load_k1_vbuf(uint32_t h) {
  mx::set_ab_dram(0, mx::host_addr(kt_base(h) + 1 * BK), D, FA_SK);
  mx::loop_ws_spad(4, 8, 8, DUMMY_A, V_END, 0, false, false, {true, false, true, true, true});
  mx::load_scales(mx::host_addr(ksc_base(h) + 1 * 512), 512, true, mx::SF_BUF1);
}
static inline void qk_vbuf(uint32_t b) {
  mx::config_scale(4, 8, 8, 1, 1, mx::host_addr(scale_sink));          // A scales 1 (Q), B scales 1
  mx::loop_ws_spad(4, 8, 8, QA[b], V_END, 0, false, true, {true, true, true, false, true});
}
static inline void qk(uint32_t b) { mx::loop_ws_spad(4, 8, 8, QA[b], K_END, 0, false, true, {true, true, true, false, true}); }
static inline void cfg_qk(bool resident) {
  mx::config_scale(4, 8, 8, 1, 0, mx::host_addr(scale_sink), resident, false);
}
static inline void cfg_pv(bool resident, bool reset) {
  mx::config_scale(4, 8, 8, 0, 1, mx::host_addr(scale_sink), resident, reset);
}
static inline void pv(uint32_t b, bool acc) {
  mx::loop_ws_spad(4, 8, 8, PA[b], V_END, 0, acc, true, {true, true, true, false, true});
}
static inline void store_s(uint32_t sm) {   // 64 x 128 bf16 store-only, no store-half toggle
  mx::loop_ws_spad(4, 8, 1, DUMMY_A, K_END, sm / 16, false, false, {true, true, true, true, false});
}
static inline void toggle() {   // junk compute: flips the compute half (only outside a feed)
  mx::config_scale(2, 2, 1, 0, 0, mx::host_addr(scale_sink));
  mx::loop_ws_spad(2, 2, 1, DUMMY_A, K_END, JUNK_ROW, false, true, {true, true, true, false, true});
}
static inline void store_only(uint32_t I, uint32_t J, uint32_t c_row) {   // toggles the store half
  mx::loop_ws_spad(I, J, 1, DUMMY_A, K_END, c_row, false, true, {true, true, true, true, false});
}
static inline void store_o() {   // O (accumulator half 1) -> O_SM bf16; the store half returns to 0
  mx::config_scale(2, 2, 1, 0, 0, mx::host_addr(scale_sink));
  store_only(2, 2, JUNK_ROW);                      // store half 0 -> 128
  cfg_pv(false, false);
  store_only(4, 8, O_ROW);                         // reads the PV half, back to 0
}

// ---- feeder and re-tile --------------------------------------------------------------------------
static __attribute__((noinline)) void wait_flag(uint32_t seq) {
  while (*sm32(FLAG) < seq) asm volatile("nop");
}
// Coalescer arrival order for I = 4, J = 8 (GN = 4, two super-blocks): chunk ch = (sb, g, bib) with
// sb outer, then row group g, then block-in-super-block bib; rows 0..15 inner.  Block b = 2 sb + bib.
// Each 32-element block is two 32 B beats (16 lanes x 2 B).  Lane l loads one word (elements 2l,
// 2l+1) and sends 2l in beat 0, 2l+1 in beat 1: the block's scale does not depend on the order, and
// the requantized block holds the even keys, then the odd keys (V's rows are stored in that order).
template <uint32_t U>
static inline void feed_rows(uint32_t pst, uint32_t g, uint32_t b, uint32_t r, uint32_t lane) {
  uint16_t e[2 * U];
#pragma unroll
  for (uint32_t u = 0; u < U; u++) {
    const uint32_t w = *sm32(pst + (16 * g + r + u) * 256 + b * 64 + 4 * lane);
    e[2 * u] = w & 0xFFFF; e[2 * u + 1] = w >> 16;
  }
#pragma unroll
  for (uint32_t u = 0; u < U; u++) {
    const uint32_t dst = mx::REQUANT + 2 * (PFLAT + (16 * g + r + u) * 128 + b * 32) + 2 * lane;
    *(volatile __shared uint16_t *)dst = e[2 * u];
    *(volatile __shared uint16_t *)(dst + 32) = e[2 * u + 1];
  }
}
// feed_g.  No format probe: the producer fences after CONFIG_EX(fp8) before releasing the feeder (in
// bf16 mode the requantizer writes its zero output to (input offset) / 4, not to the fp8 address).
static __attribute__((noinline)) void feed_p(uint32_t lane, uint32_t pst, uint32_t seq) {
  for (uint32_t ch = 0; ch < 16; ch++) {
    const uint32_t g = (ch >> 1) & 3, b = 2 * (ch >> 3) + (ch & 1);
    if (ch == 15) {
      feed_rows<8>(pst, g, b, 0, lane); feed_rows<4>(pst, g, b, 8, lane);
      feed_rows<2>(pst, g, b, 12, lane); feed_rows<1>(pst, g, b, 14, lane);
    } else {
      feed_rows<8>(pst, g, b, 0, lane); feed_rows<8>(pst, g, b, 8, lane);
    }
  }
  wait_flag(seq);   // the scale flush after the last block overwrites PV_{g-1}'s A scales
  feed_rows<1>(pst, 3, 3, 15, lane);
  mu_fence_smem();
}
// re-tile_g: PFLAT -> P tiles (i = m/16, k = key/16) at pa_row + (i*8 + k)*16 + m%16, one 16 B tile
// row per thread step, in the feed's arrival order.  nt == 32 (two warps).
static __attribute__((noinline)) void retile_p(uint32_t t, uint32_t nt, uint32_t pa_row) {
  for (uint32_t ch = 0; ch < 16; ch++) {
    const uint32_t g = (ch >> 1) & 3, b = 2 * (ch >> 3) + (ch & 1);
    for (uint32_t q = t; q < 32; q += nt) {
      const uint32_t r = q >> 1, k = 2 * b + (q & 1), m = 16 * g + r;
      volatile __shared uint32_t *src = sm32(PFLAT + m * 128 + k * 16);
      uint32_t v0, v1, v2, v3;
      for (;;) {
        v0 = src[0]; v1 = src[1]; v2 = src[2]; v3 = src[3];
        if (!(v0 == POISON || v1 == POISON || v2 == POISON || v3 == POISON)) break;
      }
      src[0] = POISON; src[1] = POISON; src[2] = POISON; src[3] = POISON;
      volatile __shared uint32_t *dst = sm32((pa_row + (g * 8 + k) * 16 + r) * 16);
      dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
    }
  }
  mu_fence_smem();
}

// ---- softmax and O ---------------------------------------------------------------------------
// Softmax of one 128-key row per lane, in place (S bf16 -> P bf16); S is complete when it starts (the
// producer fenced the store).  Loads are batched (8 / 4 per step): one load per step left the pass
// SMEM-latency bound.  Lane r starts at word r, so the 16 lanes hit 16 different SMEM subbanks.
static __attribute__((noinline)) void softmax_row(uint32_t row, uint32_t j, uint32_t lane, uint32_t base,
                                                  uint32_t mref, uint32_t lsm) {
  // bf16(scale), round to nearest even.  Not (_Float16)scale: clang folds that conversion with IEEE
  // half semantics, but Muon's .h arithmetic is bf16.
  constexpr uint32_t SCB = FA_SCALE_BITS;
  const _Float16 sc = bf((SCB + 0x7FFFu + ((SCB >> 16) & 1u)) >> 16);
  volatile __shared uint32_t *S = sm32(base + row * 256);
  _Float16 m;
  if (j == 0) {
    // max over the raw S: x -> bf16(x * sc) is monotonic (sc > 0), so the max of the scaled values
    // is the scaled max, and the pass needs no multiply
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
      const _Float16 ea = mu_fexp((_Float16)(a - m)), eb = mu_fexp((_Float16)(b - m));
      l += f32(ea) + f32(eb);
      S[(w + u + lane) & 63] = bits(ea) | (bits(eb) << 16);
    }
  }
  const float lt = (j == 0 ? 0.f : smf(lsm)[row]) + l;
  smf(lsm)[row] = lt;
  if (j == NB - 1) smf(lsm + INV_OFF)[row] = 1.0f / lt;   // one division per row
}

// O leaves in 32 B DMA writes.  A scratchpad-source mvout with 32 columns writes DRAM row r from spad
// rows (addr + r) and (addr + 16 + r), so each 512 B group of O is normalized in place into that
// order: DRAM bytes 32 r + 16 k .. (chunk 2 r + k) go to spad row 16 k + r of the group.  One warp per
// group: every lane loads its two chunks before any lane stores (a warp's loads and stores issue in
// program order).  Scratchpad-source stores bypass the accumulator path and the requantizer, so they
// do not depend on the CONFIG_EX output format.
constexpr uint32_t O_GROUPS = QT * D * 2 / 512;           // 32 groups of 512 B
static __attribute__((noinline)) void norm_o_group(uint32_t lane, uint32_t lsm, uint32_t grp) {
  const uint32_t base = O_SM + grp * 512;
  volatile __shared uint32_t *src = sm32(base + lane * 32);
  uint32_t v[8];
#pragma unroll
  for (uint32_t q = 0; q < 8; q++) v[q] = src[q];
  const float inv = smf(lsm + INV_OFF)[(grp * 512 + lane * 32) / 256];   // the lane's 32 B are in one O row
#pragma unroll
  for (uint32_t q = 0; q < 8; q++) {
    const float a = f32(bf(v[q] & 0xFFFF)) * inv, b = f32(bf(v[q] >> 16)) * inv;
    v[q] = bits((_Float16)a) | (bits((_Float16)b) << 16);
  }
  volatile __shared uint32_t *d0 = sm32(base + lane * 16), *d1 = sm32(base + (16 + lane) * 16);
#pragma unroll
  for (uint32_t q = 0; q < 4; q++) { d0[q] = v[q]; d1[q] = v[4 + q]; }
}
static inline void mvout_o_groups(uint32_t qt, uint32_t g0, uint32_t g1) {   // CONFIG_ST stride 32
  const uint32_t h = head_of(qt), q0 = (qt % (FA_SQ / QT)) * QT;
  const uint64_t base = mx::host_addr(FA_O_ADDR + (h * FA_SQ + q0) * D * 2);
  for (uint32_t grp = g0; grp < g1; grp++)
    mx::cmd(mx::MVOUT, base + grp * 512, (16ull << 48) | (32ull << 32) | (O_ROW + 32 * grp));
}

// ---- kernel ----------------------------------------------------------------------------------
static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  const bool lead = tid == 0;
  // tile of the cluster's i-th position: blocked (tb * QT_C + i) or interleaved over clusters (FA_ILV)
  auto tile = [tb](uint32_t i) { return FA_ILV ? i * NIGHTLY_CLUSTERS + tb : tb * QT_C + i; };
  const uint32_t qt0 = tile(0);
  if (lead) {                                            // prologue part 1 runs under the PFLAT poison
    if (tb == 0) nightly_phase(0);
    *sm32(FLAG) = 0;
    mx::flush_tlb();
    mx::lut_disable();
    mx::config_ld(D, 0);
    mx::config_ld(FA_SK, 1);
    mx::config_st(32);                                   // O mvouts: 32 B DRAM rows, contiguous
    load_q(qt0, 0);
    load_q_scales(qt0);
    load_k(head_of(qt0), 0);
    load_k1_vbuf(head_of(qt0));
    mx::config_ex(mx::FP8, mx::FP8, mx::BF16);
    cfg_qk(false);
    qk(0);                                               // QK_0: ex 0 -> 128
    cfg_qk(false);                                       // QK_0 unrolled before its store
    store_s(SP[0]);
  } else asm volatile("nop");
  for (uint32_t w = tid; w < 2048; w += tpb) *sm32(PFLAT + 4 * w) = POISON;
  mu_fence_smem();
  fa_bar(1, warps);

  if (warp == 0) {
    if (lead) mx::fence();                               // S_0 stored
    else asm volatile("nop");
    fa_bar(BAR_S, 5);                                    // softmax_0 may start
    if (lead) {
      toggle();                                          // 128 -> 0
      qk_vbuf(0);                                        // QK_1 (K_1 in the V buffer): ex 0 -> 128
    } else asm volatile("nop");
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t i = g / NB, j = g % NB;
      const bool more = opaque(g + 1 < G), more2 = opaque(g + 2 < G);
      const bool prof = tb == 0 && g == FA_PROF_J;
      if (lead) {
        if (tb == 0) nightly_heartbeat((i << 16) | j);
        if (prof) nightly_phase(2);
        mx::fence();                                     // QK_{g+1} done
        if (g + 1 == G) toggle();                        // block g-1 had no QK: 0 -> 128
        mx::config_ex(mx::FP8, mx::FP8, mx::BF16);       // [bf16] S_{g+1} and O stores
        if (more) store_s(SP[(g + 1) & 1]);
        if (j == 1 && i > 0) store_o();                  // O of tile i-1 (its last PV ran in block g-1)
        mx::fence();
        if (prof) nightly_phase(3);
        if (g == 0) {                                    // block 0 has no PV: QK_2 runs under softmax_0
          toggle();                                      // 128 -> 0 (junk into the unused O half)
          load_k(head_of(qt0), 2);
          load_v(head_of(qt0), 0);                       // QK_1 done with the V buffer and B scales 1
          cfg_qk(false);
          qk(0);                                         // QK_2: ex 0 -> 128
        }
      } else asm volatile("nop");
      if (more) fa_bar(BAR_S, 5);                        // softmax_{g+1} may start
      if (lead) {
        cfg_pv(true, true);                              // [fp8] coalescer bounds + reset
        cfg_pv(true, false);                             // clears the latched reset before the feed
        mx::config_ex(mx::FP8, mx::FP8, mx::FP8);
        mx::fence();                                     // fp8 output format live before the feed
      } else asm volatile("nop");
      fa_bar(BAR_F, 2);                                  // feeder may start feed_g
      if (lead) {
        // Command order: a non-loop command (MVOUT, MX_LOAD_SCALES, CONFIG) is accepted only once the
        // earlier loops are unrolled, which for PV is near its end, and gemmini runs 2 loops at once.
        // So: non-loop commands first (no loop in flight here), then the next tile's Q half (loop),
        // PV (loop), and the K data loop right behind PV (its scale load waits for PV's unroll).
        // O groups of tile i-1 normalized in this block's softmax iteration (before its BAR_S)
        if (j >= 2 && j <= 5 && i > 0) mvout_o_groups(tile(i - 1), 8 * (j - 2), 8 * (j - 1));
        if (more2 && g > 0 && (g + 2) % NB == 0)
          load_q_scales(tile((g + 2) / NB));             // QK_{g+1} done with A scales 1
        // next tile's Q data in two halves (blocks 1 and 2), each 4 KiB, ahead of the K load
        if ((j == 1 || j == 2) && i + 1 < QT_C) load_q_half(tile(i + 1), (i + 1) & 1, j - 1);
        if (g > 0) pv((g - 1) & 1, (g - 1) % NB != 0);   // PV_{g-1}: ex 128 -> 0
        if (more2 && g > 0) load_k(head_of(tile((g + 2) / NB)), (g + 2) % NB);   // K_{g+2}
        if (g > 0) mx::fence();                          // PV_{g-1} done: A scales 0, V buffer free
        *sm32(FLAG) = g + 1;
        if (prof) nightly_phase(6);
        // QK before the V load: a non-loop command (cfg_qk) waits until earlier loops are unrolled,
        // and the V mvins fill the reservation station
        if (more2 && g > 0) { cfg_qk(true); qk(((g + 2) / NB) & 1); }   // QK_{g+2}: ex 0 -> 128
        if (g > 0) load_v(head_of(tile(i)), j);          // V_g
      } else asm volatile("nop");
      fa_bar(BAR_R, 3);                                  // re-tile_g done
      if (lead && prof) nightly_phase(7);
    }
    if (lead) {                                          // epilogue: last PV, last O to SMEM
      mx::fence();
      toggle();                                          // block G-1 had no QK: 0 -> 128
      cfg_pv(false, false);
      pv((G - 1) & 1, (G - 1) % NB != 0);
      mx::fence();
      mx::config_ex(mx::FP8, mx::FP8, mx::BF16);
      store_o();
      mx::fence();                                       // last O in SMEM before BAR_O
      if (tb == 0) nightly_phase(1);
    } else asm volatile("nop");
  } else if (warp == 1) {                               // feeder
    for (uint32_t g = 0; g < G; g++) {
      fa_bar(BAR_P, 7);                                  // P_g complete
      fa_bar(BAR_F, 2);                                  // fp8 window of block g issued
      if (tb == 0 && g == FA_PROF_J) nightly_phase(4);
      feed_p(lane, SP[g & 1], g + 1);
      if (tb == 0 && g == FA_PROF_J) nightly_phase(5);
    }
  } else if (warp != 4 && warp != 6) {                  // softmax: warps 2, 3, 5, 7, 16 rows each
    const uint32_t sw = warp == 2 ? 0 : warp == 3 ? 1 : warp == 5 ? 2 : 3;
    const uint32_t row = sw * 16 + lane;
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t i = g / NB, j = g % NB;
      fa_bar(BAR_S, 5);
      softmax_row(row, j, lane, SP[g & 1], MREF[i & 1], LSUM[i & 1]);
      mu_fence_smem();
      fa_bar(BAR_P, 7);
      // O of tile i-1 (stored in block i*NB+1): groups 8 (j-2) .. +7 in key blocks 2..5, two per warp.
      // They are ordered before this warp's next BAR_S, so the producer's mvouts in block g read
      // finished data.
      if (j >= 2 && j <= 5 && i > 0) {
        norm_o_group(lane, LSUM[(i - 1) & 1], 8 * (j - 2) + sw);
        norm_o_group(lane, LSUM[(i - 1) & 1], 8 * (j - 2) + 4 + sw);
        mu_fence_smem();
      }
    }
  } else {                                              // re-tile: warps 4, 6
    const uint32_t rt = (warp == 4 ? 0 : 1) * MU_NUM_THREADS + lane;
    for (uint32_t g = 0; g < G; g++) {
      fa_bar(BAR_P, 7);                                  // wait with the feeder instead of spin-polling
      retile_p(rt, 2 * MU_NUM_THREADS, PA[g & 1]);
      fa_bar(BAR_R, 3);
    }
  }
  fa_bar(BAR_O, warps);                                  // the last O is stored
  for (uint32_t grp = warp; grp < O_GROUPS; grp += warps) norm_o_group(lane, LSUM[(QT_C - 1) & 1], grp);
  mu_fence_smem();
  fa_bar(BAR_O, warps);                                  // normalized
  if (lead) { mvout_o_groups(tile(QT_C - 1), 0, O_GROUPS); mx::fence(); } else asm volatile("nop");
  fa_bar(BAR_T, warps);
  nightly_kernel_end(tid, tpb);
  nightly_verify_bf16(FA_O_ADDR, FA_G_ADDR, FA_H * FA_SQ * FA_D, tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
