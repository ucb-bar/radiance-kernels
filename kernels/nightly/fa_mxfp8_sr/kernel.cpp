// MXFP8 flash attention on Radiance with SPAD_REQUANT: upstream fa_mxfp8 (gemmini mesh for QK^T and PV, SIMT softmax,
// lazy softmax with O accumulated in the accumulator) with the Muon P feed through the requantizer window replaced by
// gemmini's SPAD_REQUANT (funct 34): the requantizer reads P (BF16) straight from SMEM and writes the PV operand tiles
// and the resident act scales itself.  No feeder / re-tile warps, no FLAG handshake, no fp8/bf16 output-format
// switching (SPAD_REQUANT carries its own format), and PV / QK issue back to back.  Needs a Radiance build with
// has_spad_requant (WithRadianceE4M3MxGemmini).  Shapes, data and golden: upstream's, keys in natural order
// (gen_data.py --novperm).
#include <nightly/device.h>
#include <nightly/mx.h>
#include <nightly/verify.h>
#include "fa_data.h"


constexpr uint32_t D = FA_D, BK = FA_BK, QT = 64;
constexpr uint32_t NB = FA_SK / BK, NQT = FA_H * FA_SQ / QT;
static_assert(D == 128 && BK == 128, "the scratchpad and accumulator maps assume d = 128, 128-key blocks");
static_assert(!FA_VPERM, "SPAD_REQUANT reads P in natural key order: generate the data with gen_data.py --novperm");
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
//   bank 2: S/P x2                bank 3: store-flip sink, O, small buffers
constexpr uint32_t K_END = 1024;                           // K^T block [0, 1024): QK B operand
constexpr uint32_t V_START = 1024, V_END = 2048;           // V block: PV B operand
constexpr uint32_t QA[2] = {2048, 2560};                   // Q tiles (QK A operand), by tile parity
constexpr uint32_t PA[2] = {3072, 3584};                   // P tiles (PV A operand), by block parity
constexpr uint32_t SP[2] = {4096 * 16, 5120 * 16};         // S bf16 [64][128] -> P bf16 in place, pitch 256 B
constexpr uint32_t O_ROW = 6656, O_SM = O_ROW * 16;        // O bf16 [64][128]
constexpr uint32_t JUNK_ROW = 7712;                        // 32x32 bf16 sink for accumulator-toggle stores
constexpr uint32_t FLIP_ROW = 6144;                        // 32x128 bf16 sink of the K-load loop's store-half flip
static_assert(FLIP_ROW + 512 <= O_ROW, "the store-flip sink (512 rows) must end below O");
constexpr uint32_t MREF[2] = {7856 * 16, 7904 * 16};       // m_ref f32 [64] by tile parity
constexpr uint32_t LSUM[2] = {MREF[0] + 256, MREF[1] + 256};   // l f32 [64] by tile parity
constexpr uint32_t INV_OFF = 256;                          // 1/l f32 [64] at LSUM[p] + INV_OFF
constexpr uint32_t DUMMY_A = 0;                            // A row of store-only and junk loops
static_assert(MREF[1] + 768 <= mx::SPAD_ROWS * 16, "scratchpad budget");

enum : uint32_t { BAR_P = 3, BAR_T = 6, BAR_S = 7, BAR_O = 8, BAR_N = 9, BAR_ND = 10 };

// dummy destination for the requantizer's DRAM scale flush (always issued)
static uint32_t scale_sink[256] __attribute__((aligned(64)));   // also SPAD_REQUANT's DRAM scale flush (256 B/block)

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

static inline void load_k_data(uint32_t h, uint32_t j) {   // K^T block (B loader)
  mx::set_ab_dram(0, mx::host_addr(kt_base(h) + j * BK), D, FA_SK);
  mx::loop_ws_spad(4, 8, 8, DUMMY_A, K_END, 0, false, false, {true, false, true, true, true});
}
// K^T block + a junk 32 x 128 store that flips the accumulator store half back (I only sizes that store; the B loads
// depend on K and J alone; I = 2, not 1: a store's row tiles must cover whole 32-row MX blocks, tilesPerMxBlock = 2).  QK_{g+2} stores its own S and so moves the store half 0 -> 128; the next block's
// K load returns it to 0 before QK_{g+3}'s store.  Its store reads O rows, so it completes behind PV_{g-1}'s computes.
// v6: the next tile's Q half rides along through the A loader (I = 2, K = 8 is exactly a Q half's A tiles), so key
// blocks 1 and 2 need no separate Q-half loop: a third load loop there held the second loop slot through PV, pushed
// the K load (and the requant, which the loop unit holds behind it) past PV, and left the mesh idle 2.7k.
// v7d: inc_acc is set even without the flip.  The loop unit lets a scale config / gated scale load pass a loop still in
// its slots only if that loop has issued its loads and computes and has inc_acc set (only_stores_left); otherwise the
// loop must have left the slots, which happens in order, i.e. behind the matmul ahead of it (QK leaves ~260 cycles
// after its last compute, once its S stores are issued).  On a loop that skips its computes and stores, inc_acc moves
// nothing: the compute half moves on an issued compute, the store half on an issued store.
static inline void load_kq(uint32_t h, uint32_t j, bool flip, uint32_t q_src, uint32_t q_row) {
  mx::set_ab_dram(q_src ? mx::host_addr(q_src) : 0, mx::host_addr(kt_base(h) + j * BK), D, FA_SK);
  mx::loop_ws_spad(2, 8, 8, q_src ? q_row : DUMMY_A, K_END, flip ? FLIP_ROW : 0, false, true,
                   {q_src == 0, false, true, true, !flip});
}
static inline void load_k(uint32_t h, uint32_t j) {   // + its scales (B buffer 0), ungated: prologue and block 0
  load_k_data(h, j);
  mx::load_scales(mx::host_addr(ksc_base(h) + j * 512), 512, true, 0);
}
// V [128 keys][128] is the PV B operand: tile (k, n) at V_START + (k*8 + n)*16.  The A loader
// produces exactly that order with I = 8 (key tiles) and K = 8 (d tiles), and shares the A queue's
// CONFIG_LD stride (D) with Q, leaving the B queue's stride to K^T.  Scales -> B buffer 1 (load_v_scales).
static inline void load_v_data(uint32_t h, uint32_t j) {   // inc_acc: see load_kq (v7d)
  mx::set_ab_dram(mx::host_addr(v_base(h) + j * BK * D), 0, D, FA_SK);
  mx::loop_ws_spad(8, 4, 8, V_START, K_END, 0, false, true, {false, true, true, true, true});
}
// Q is double-buffered by tile parity (the next tile's Q loads early, in two halves); its A scales
// (buffer 1) are loaded only once the current tile's last QK is done.
static inline void load_q(uint32_t qt, uint32_t b) {
  const uint32_t h = head_of(qt), q0 = (qt % (FA_SQ / QT)) * QT;
  mx::set_ab_dram(mx::host_addr(FA_Q_ADDR + (h * FA_SQ + q0) * D), 0, D, FA_SK);
  mx::loop_ws_spad(4, 4, 8, QA[b], K_END, 0, false, false, {false, true, true, true, true});
}
static inline uint32_t q_half_src(uint32_t qt, uint32_t half) {   // DRAM address of query rows 32*half..+31 of tile qt
  return FA_Q_ADDR + (head_of(qt) * FA_SQ + (qt % (FA_SQ / QT)) * QT + 32 * half) * D;
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
// QK that stores S (64 x 128 bf16, row-major) to SMEM itself: no store_s loop, no S-store fence.  Store half 0 -> 128.
static inline void qk_store(uint32_t b, uint32_t sm) {
  mx::loop_ws_spad(4, 8, 8, QA[b], K_END, sm / 16, false, true, {true, true, true, false, false});
}
static inline void cfg_qk(bool resident) {
  mx::config_scale(4, 8, 8, 1, 0, mx::host_addr(scale_sink), resident, false);
}
static inline void cfg_pv(bool resident, bool reset) {
  mx::config_scale(4, 8, 8, 0, 1, mx::host_addr(scale_sink), resident, reset);
}
// QK_{g+2} under a MANAGED scale config (npu attn_flash's protocol): the reservation station orders a resident
// SPAD_REQUANT behind computes whose config may read act half 0, and treats every legacy config as one that may; a
// managed config selecting act half 1 does not, so SPAD_REQUANT P_g passes QK_{g+2} and runs alongside it.  A managed
// config (rs2[17]) waits for its weight half to be LOADED by a GATED MX_LOAD_SCALES (rs2[54]: starts once the half
// is FREE, i.e. a config selected the other half and the computes ahead of it drained) and marks it INUSE; act scales
// are reused in place (rs2[18], Q's per-tile load).  PV stays legacy: V's scales load ungated after the block-top fence.
static inline void load_k_scales_gated(uint32_t h, uint32_t j) {   // weight half 0
  mx::cmd(mx::MX_LOAD_SCALES, mx::host_addr(ksc_base(h) + j * 512) & 0xFFFFFFFFFFull, (1ull << 54) | (1ull << 32) | 512);
}
static inline void cfg_qk_managed() {   // I, J, K = 4, 8, 8; act half 1 (Q), weight half 0 (K)
  const uint64_t rs1 = (0ull << 61) | (1ull << 60) | (8ull << 51) | (8ull << 42) | (4ull << 33) |
                       (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
  mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
}
// v7: PV under a managed config too (npu's protocol for every matmul): V's scales load GATED into weight half 1, which
// the next QK's managed config (or a legacy one selecting half 0) frees once PV's computes have drained.  So V_g's scales
// can be issued right behind its data, a block ahead, with no fence.  P's act scales (half 0) are SPAD_REQUANT's
// resident ones (in place, rs2[18]); the station still orders the config behind that requant (it may read act half 0).
static inline void load_v_scales_gated(uint32_t h, uint32_t j) {   // weight half 1
  mx::cmd(mx::MX_LOAD_SCALES, mx::host_addr(vsc_base(h) + j * 512) & 0xFFFFFFFFFFull,
          (1ull << 54) | ((uint64_t)mx::SF_BUF1 << 33) | (1ull << 32) | 512);
}
static inline void cfg_pv_managed() {   // I, J, K = 4, 8, 8; act half 0 (P), weight half 1 (V)
  const uint64_t rs1 = (1ull << 61) | (0ull << 60) | (8ull << 51) | (8ull << 42) | (4ull << 33) |
                       (mx::host_addr(scale_sink) & 0x1FFFFFFFFull);
  mx::cmd(mx::CONFIG_SCALE_MEM, rs1, (1ull << 18) | (1ull << 17) | (1ull << 16) | 1);
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
static inline void store_o() {   // O (accumulator half 1) -> O_SM bf16: the store half is at 128 (last QK's store), -> 0
  cfg_pv(false, false);
  store_only(4, 8, O_ROW);                         // reads the PV half, back to 0
}


// ---- SPAD_REQUANT ----------------------------------------------------------------------------------
// One block's P: BF16 [64][128] (row-major, pitch 256 B) at SP[b] -> E4M3 in the PV A-operand tiled layout at PA[b],
// E8M0 scales [BK/32][64] resident in act-scale buffer 0 (PV's) and flushed to scale_sink.  Reservation-station
// ordered: it waits for the loops reading its destination (PV_{g-1} shares PA's parity) and for any loop reading act
// buffer 0 to have issued its computes; PV_g's computes wait for it (their A rows).  rs1 = src[13:0] | dst[27:14] |
// tiled[28] | resident[29] | scale DRAM[62:30]; rs2 = M[15:0] | N[31:16] (SpadRequant.scala).
constexpr uint32_t K_SPAD_REQUANT = 34;
static inline void spad_requant(uint32_t b) {
  const uint64_t scale_dram = mx::host_addr(scale_sink) & 0x1FFFFFFFFull;
  const uint64_t rs1 = (uint64_t)(SP[b] / 16) | ((uint64_t)PA[b] << 14) | (1ull << 28) | (1ull << 29) | (scale_dram << 30);
  mx::cmd(K_SPAD_REQUANT, rs1, (uint64_t)QT | ((uint64_t)BK << 16));
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

// O is normalized by the SIMT warps and written straight to DRAM (row-major bf16, 256 B rows): no gemmini mvouts.  One
// warp per 512 B group (two O rows).  v7g: lane l handles words 16 q + l (q = 0..7), so every load / store instruction
// covers 64 contiguous bytes (one line).  Lane l owning 32 contiguous bytes made each store instruction touch 8 lines
// at a 32 B stride: 1024 scattered word stores per block took 3.6-5.6k cycles (worse with K / V loading from DRAM),
// and the softmax warps reached BAR_P that much later.
constexpr uint32_t O_GROUPS = QT * D * 2 / 512;           // 32 groups of 512 B
static inline uint32_t o_dram(uint32_t qt) {
  const uint32_t h = head_of(qt), q0 = (qt % (FA_SQ / QT)) * QT;
  return FA_O_ADDR + (h * FA_SQ + q0) * D * 2;
}
static __attribute__((noinline)) void norm_o_group(uint32_t lane, uint32_t lsm, uint32_t grp, uint32_t odst) {
  volatile __shared uint32_t *src = sm32(O_SM + grp * 512 + lane * 4);
  uint32_t v[8];
#pragma unroll
  for (uint32_t q = 0; q < 8; q++) v[q] = src[16 * q];
  const float inv0 = smf(lsm + INV_OFF)[2 * grp], inv1 = smf(lsm + INV_OFF)[2 * grp + 1];   // words 0..63: row 2 grp
#pragma unroll
  for (uint32_t q = 0; q < 8; q++) {
    const float inv = q < 4 ? inv0 : inv1;
    const float a = f32(bf(v[q] & 0xFFFF)) * inv, b = f32(bf(v[q] >> 16)) * inv;
    v[q] = bits((_Float16)a) | (bits((_Float16)b) << 16);
  }
  volatile uint32_t *dst = (volatile uint32_t *)(odst + grp * 512 + lane * 4);
#pragma unroll
  for (uint32_t q = 0; q < 8; q++) dst[16 * q] = v[q];
}


// ---- kernel ----------------------------------------------------------------------------------
// Warp roles: 0 producer (lane 0 issues every gemmini command); 2, 3, 4, 5 softmax (two per core: warp w runs on
// core w % 2); 1, 6, 7 normalize O (the previous tile's in the background, the last one with everyone).  Block g (key block j of query tile i), v7:
//   producer: wait for QK_{g+1} to retire (its S_{g+1} store landed: MMIO retire counter, no fence) | release
//             softmax_{g+1} | K_{g+2} load (+ next tile's Q half, + store-half flip; O of tile i-1 when j == 0) |
//             SPAD_REQUANT P_g | QK_{g+2} (+ S_{g+2} store) | V_g data + scales | PV_g | wait softmax_{g+1}
//   mesh:     QK_{g+1} | PV_{g-1} | QK_{g+2} | PV_g | ...   PV_{g-1} was issued last block, so it is already queued when
//             QK_{g+1} ends; nothing drains gemmini in the steady state.
//   softmax:  softmax_{g+1} in place (S -> P bf16)
//   warps 1, 6, 7: O of tile i-1 normalized to DRAM in the background during tile i (BAR_N / BAR_ND with the producer)
// Every hazard the per-block fence used to cover is ordered in hardware: operand rows by the reservation station and
// the loop unit (a load loop's mvins all leave the loop unit before the scale config behind it passes, so before the
// matmul that reads them), weight scales by the managed-half protocol, and S, Q's per-tile scales, the K / Q buffers
// and O_SM by the retire wait (in-order: QK_{g+1} retired means everything issued before it is done too).
static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  const bool lead = tid == 0;
  auto tile = [tb](uint32_t i) { return FA_ILV ? i * NIGHTLY_CLUSTERS + tb : tb * QT_C + i; };
  const uint32_t qt0 = tile(0);
  // K_{b+2} for block b (b >= 1): flip the store half 128 -> 0 unless QK_{b+1} stored no S (b < 2) or block b stores O
  // (key block 0 of tiles 1..), whose store_only flips it; in key blocks 1 and 2 the next tile's Q half rides along
  auto issue_k_load = [&](uint32_t b) {
    const uint32_t ib = b / NB, jb = b % NB;
    const bool o_tile = jb == 0 && ib > 0, qh = (jb == 1 || jb == 2) && ib + 1 < QT_C;
    load_kq(head_of(tile((b + 2) / NB)), (b + 2) % NB, b >= 2 && !o_tile,
            qh ? q_half_src(tile(ib + 1), jb - 1) : 0, QA[(ib + 1) & 1] + 256 * (jb - 1));
  };
  if (lead) {
    if (tb == 0) nightly_phase(0);
    mx::flush_tlb();
    mx::lut_disable();
    mx::config_ld(D, 0);
    mx::config_ld(FA_SK, 1);
    load_q(qt0, 0);
    load_q_scales(qt0);
    load_k(head_of(qt0), 0);
    load_k1_vbuf(head_of(qt0));
    mx::config_ex(mx::FP8, mx::FP8, mx::BF16);           // BF16 out for the whole kernel (S and O stores)
    cfg_qk(false);
    qk(0);                                               // QK_0: ex 0 -> 128
    cfg_qk(false);                                       // QK_0 unrolled before its store
    store_s(SP[0]);
  } else asm volatile("nop");
  mu_fence_smem();
  fa_bar(1, warps);

  const bool smx = warp >= 2 && warp <= 5;               // softmax warps
  if (warp == 0) {
    if (lead) mx::fence();                               // S_0 stored
    else asm volatile("nop");
    fa_bar(BAR_S, 5);                                    // softmax_0 may start
    if (lead) {
      toggle();                                          // 128 -> 0
      qk_vbuf(0);                                        // QK_1 (K_1 in the V buffer): ex 0 -> 128
    } else asm volatile("nop");
    fa_bar(BAR_P, 5);                                    // softmax_0 done: P_0 (bf16) in SP[0]
    if (lead) spad_requant(0);                           // P_0 -> PA[0] + act scales 0
    else asm volatile("nop");
    uint32_t n_qk = 0;                                   // LOOP_WS sequence number of the last issued QK
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t i = g / NB, j = g % NB;
      const bool more = opaque(g + 1 < G), more2 = opaque(g + 2 < G), more3 = opaque(g + 3 < G);
      const bool prof = tb == 0 && g == FA_PROF_J;
      if (lead) {
        if (tb == 0) nightly_heartbeat((i << 16) | j);
        if (prof) nightly_phase(2);
        if (g == 0) {                                    // block 0: fenced, as before (QK_1 and QK_2 store no S)
          mx::fence();                                   // QK_1, SPAD_REQUANT P_0
          store_s(SP[1]);                                // S_1
          mx::fence();
          toggle();                                      // 128 -> 0 (junk into the unused O half)
          load_k(head_of(qt0), 2);                       // (K_2: QK_1's K is in the V buffer)
          cfg_qk(false);
          qk(0);                                         // QK_2: ex 0 -> 128 (runs under softmax_1)
          store_s(SP[0]);                                // S_2 (QK_2 stores no S): after QK_2's computes, ahead of PV_0
          n_qk = mx::loops_issued;                       // block 1 waits for this store, not for PV_0 behind it
        } else if (more) {
          mx::wait_retired(n_qk);                        // QK_{g+1} and its S_{g+1} store retired (block 1: S_2's store)
        }
        if (more2 && g > 0 && (g + 2) % NB == 0) load_q_scales(tile((g + 2) / NB));   // QK_{g+1}: last user of A scales 1
        if (prof) nightly_phase(3);
      } else asm volatile("nop");
      // v7h: tile i-1's O is normalized by warps 1, 6, 7 in the background.  BAR_N (block i*NB+1, after the retire wait:
      // the O store issued in block i*NB has landed) starts them; BAR_ND (before the next O store, block (i+1)*NB, and
      // in the epilogue) waits for them to finish, long done by then.
      if (j == 1 && i > 0) fa_bar(BAR_N, 4);
      if (j == 0 && i >= 2) fa_bar(BAR_ND, 4);
      if (more) fa_bar(BAR_S, 5);                        // softmax_{g+1} may start
      if (lead) {
        // K_{g+2} (+ flip) | SPAD_REQUANT P_g | QK_{g+2} storing S_{g+2} into SP[g & 1].  The requant goes ahead of QK in
        // program order: QK's S store overwrites P_g, so the reservation station orders that store after it (a requant
        // behind QK would also not pass the QK loop, whose C rows it reads).  It waits at the loop unit until PV_{g-1}
        // has issued its computes (act half 0), as the managed QK config does; QK_{g+2} then follows PV_{g-1}.
        if (more2 && g > 0) {
          // K_{g+2}: K buffer and weight half 0 free (QK_{g+1} retired / freed by PV_{g-1}'s config).  The store half is
          // at 128 after QK_{g+1}'s S store: flip it back here, or (key block 0 of tiles 1..) store O of tile i-1, whose
          // last PV_{g-1} is queued ahead (the station orders the O store after its computes), which flips it too.  QK_2
          // stored no S, so block 1 flips nothing.  In key blocks 1 and 2 the next tile's Q half rides along.
          // (K_{g+2}'s data was issued at the end of block g-1, under PV_{g-1}: issue_k_load)
          const bool o_tile = j == 0 && i > 0;
          // Gated scale loads, one matmul early.  A gated MX_LOAD_SCALES passes the loop unit only once every older
          // loop has issued its loads and computes; issued right ahead of its own config it passed at the end of the
          // matmul before, and its DMA sat between the two matmuls (~400 cycles each).  Here each passes with the
          // previous matmul's config and waits in the scale loader for its half: V_g's for weight half 1 (freed by
          // QK_{g+2}'s config, so it lands under QK_{g+2}), K_{g+3}'s for half 0 (freed by PV_g's, lands under PV_g).
          // In the loader's FIFO each sits behind the load its freeing config needs (K_{g+2}'s, issued last block).
          if (g == 1) load_k_scales_gated(head_of(tile(3 / NB)), 3 % NB);   // half 0 free: QK_2 (legacy) retired
          load_v_scales_gated(head_of(tile(g / NB)), g % NB);
          if (more3) load_k_scales_gated(head_of(tile((g + 3) / NB)), (g + 3) % NB);
          if (o_tile) store_o();                         // O of tile i-1 -> O_SM (normalized by warps 1, 6, 7)
          cfg_qk_managed();
        }
        if (g > 0) spad_requant(g & 1);                  // P_g -> PA[g & 1] + act scales 0, alongside QK_{g+2}
        // QK_{g+2} right behind SR: the producer gets past the scale commands only once PV_{g-1} has issued its
        // computes, and the command path buffers ~4 commands, so every command between SR and QK's loop is issued then,
        // at ~50 cycles each (v7e put V's 5 ahead of QK: PV -> QK gap 509 cycles instead of 84).
        if (more2 && g > 0) {
          qk_store(((g + 2) / NB) & 1, SP[g & 1]);       // QK_{g+2}: ex 0 -> 128, S_{g+2} -> SP[g & 1]
          n_qk = mx::loops_issued;
        } else {
          if (g > 0) toggle();                           // no QK_{g+2} ahead of PV_g: 0 -> 128 (its legacy config also
                                                         // frees weight half 1 for V_g's gated scales)
          load_v_scales_gated(head_of(tile(g / NB)), g % NB);   // block 0: half 1 free (QK_1 done, legacy only)
        }
        // V_g and PV_g, a block ahead: V's rows are held until PV_{g-1} has issued its computes (loop WAR guard), then
        // ordered behind them (the managed config ahead of PV_g passes only once V's mvins have all left the loop unit).
        // Behind QK, V leaves the loop unit's slots only after QK (loops leave in order; QK once its S stores are
        // issued), so PV_g's loop waits ~260 cycles past QK's last compute for a slot.
        load_v_data(head_of(tile(g / NB)), g % NB);
        cfg_pv_managed();
        pv(g & 1, g % NB != 0);                          // PV_g: ex 128 -> 0
        // v7c: K_{g+3} (block g+1's K) right behind PV_g, so it takes the second loop slot as PV_g starts and loads
        // under it.  Issued after the retire wait and BAR_S of block g+1 it started 1.3k into PV and landed after it,
        // and the gated scale load / config waiting for it opened a ~400-cycle gap before QK.  Its writes are ordered
        // after QK_{g+1}'s K reads by the loop unit / reservation station, not by the wait.
        if (more3) issue_k_load(g + 1);
        if (prof) nightly_phase(6);
      } else asm volatile("nop");
      if (more) fa_bar(BAR_P, 5);                        // softmax_{g+1} done: P_{g+1} in SP (requantized next block)
      if (lead && prof) nightly_phase(7);
    }
    if (QT_C >= 2) fa_bar(BAR_ND, 4);                    // tile QT_C-2's O normalized: O_SM free
    if (lead) {                                          // epilogue: last O to SMEM (the store half is at 128)
      mx::fence();
      store_o();
      mx::fence();                                       // last O in SMEM before BAR_O
      if (tb == 0) nightly_phase(1);
    } else asm volatile("nop");
  } else if (smx) {                                      // softmax: warps 2, 3, 4, 5, 16 rows each
    const uint32_t sw = warp - 2;
    const uint32_t row = sw * 16 + lane;
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t i = g / NB, j = g % NB;
      fa_bar(BAR_S, 5);
      softmax_row(row, j, lane, SP[g & 1], MREF[i & 1], LSUM[i & 1]);
      mu_fence_smem();
      fa_bar(BAR_P, 5);
    }
  } else {
    // warps 1, 6, 7: normalize tile i-1's O (in O_SM since block i*NB) to DRAM during tile i, off the softmax warps'
    // path.  Its global stores run at ~0.3 lane-stores per cycle (~3-5k per block for the softmax warps' 8 groups, which
    // pushed them past the producer at BAR_P); here the 32 groups take ~13k cycles, under the ~7 blocks before O_SM is
    // reused.  1/l of tile i-1 (INV_OFF) was written in its last key block; LSUM[(i-1) & 1] is next reused by tile i+1.
    const uint32_t nw = warp == 1 ? 0 : warp - 5;        // 0, 1, 2
    for (uint32_t i = 1; i < QT_C; i++) {
      fa_bar(BAR_N, 4);
      for (uint32_t grp = nw; grp < O_GROUPS; grp += 3) norm_o_group(lane, LSUM[(i - 1) & 1], grp, o_dram(tile(i - 1)));
      fa_bar(BAR_ND, 4);
    }
  }
  fa_bar(BAR_O, warps);                                  // the last O is stored
  for (uint32_t grp = warp; grp < O_GROUPS; grp += warps)
    norm_o_group(lane, LSUM[(QT_C - 1) & 1], grp, o_dram(tile(QT_C - 1)));
  fa_bar(BAR_T, warps);                                  // nightly_kernel_end fences each core's L0d
  nightly_kernel_end(tid, tpb);
  nightly_verify_bf16(FA_O_ADDR, FA_G_ADDR, FA_H * FA_SQ * FA_D, tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
