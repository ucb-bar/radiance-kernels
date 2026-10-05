// MX GEMM (MXFP8 or MXFP4 operands, bf16 C) on the nightly RTL.
//
// Work split: cluster c owns m-tiles [c*MT/NC, (c+1)*MT/NC) and all n-tiles.  Each output tile
// TM x TN is computed as KS k-steps of TK; each k-step is one loop_ws over spad-resident A/B.
//
// One producer thread per cluster (threadblock thread 0) drives gemmini:
//   prologue: load operands and scales of step 0 into buffer 0
//   step g (buffer b = g & 1):
//     fence                       compute g-1 (and its C store), loads + scales of g are done
//     [tile boundary barrier]     consumers copy the previous tile's C out of SMEM
//     CONFIG_SCALE_MEM(sel = b)   ordered with computes
//     loop_ws compute g           accumulate for k-step > 0; last k-step stores C (bf16) to the
//                                 tile's SMEM C buffer and toggles the accumulator half
//     loop_ws load g+1 -> 1-b     reservation station orders it after compute g-1 (buffer 1-b)
//     MX_LOAD_SCALES g+1 -> 1-b   unordered DMA: safe because compute g-1 is done (fence)
// The consumer warps copy each finished C tile (row-major bf16 in SMEM) to C in DRAM.
//
// Schedules (compile-time, DEFS="-DNAME=value"):
//   GEMM_V3=1 (default)  A row block resident per m-tile, only B streams (entry_v3)
//   GEMM_V3=0            v2 (GEMM_V2=1, no steady-state fence) or v1 (GEMM_V2=0, fence per k-step)
//   GEMM_CDRAM=1 (default, v3) gemmini stores C from the accumulator to DRAM; 0: SIMT copies C
//   GEMM_MANAGED=1 (v3, fp8) the loop loads and orders its own scales (LOOP_WS_CONFIG_SCALES)
#include <nightly/device.h>
#include <nightly/mx.h>
#include <nightly/verify.h>
#include "gemm_data.h"

constexpr bool FP4 = GEMM_FP4;
constexpr uint32_t PE = FP4 ? 32 : 16;               // PE tile edge along M and N
constexpr uint32_t VPB = FP4 ? 2 : 1;                 // values per byte
constexpr uint32_t TM = GEMM_TM, TN = GEMM_TN, TK = GEMM_TK;
constexpr uint32_t I = TM / PE, J = TN / PE, KT = TK / 16;
constexpr uint32_t MT = GEMM_M / TM, NT = GEMM_N / TN, KS = GEMM_K / TK;
static_assert(MT % NIGHTLY_CLUSTERS == 0, "m-tiles must split evenly over clusters");
constexpr uint32_t MT_C = MT / NIGHTLY_CLUSTERS, T = MT_C * NT, G = T * KS;

// scratchpad map (rows of 16 B)
constexpr uint32_t A_ROWS = TM * TK / VPB / 16, B_ROWS = TK * TN / VPB / 16, C_ROWS = TM * TN * 2 / 16;
constexpr uint32_t A_SP[2] = {0, A_ROWS};
constexpr uint32_t B_END[2] = {2 * A_ROWS + B_ROWS, 2 * A_ROWS + 2 * B_ROWS};
#ifndef GEMM_V3
#define GEMM_V3 1
#endif
#if GEMM_V3
// v3: the whole A row block of an m-tile stays resident (KS chunks of A_ROWS), B k-steps and C tiles
// are double-buffered after it.
constexpr uint32_t A3_ROWS = KS * A_ROWS;
constexpr uint32_t B3_END[2] = {A3_ROWS + B_ROWS, A3_ROWS + 2 * B_ROWS};
constexpr uint32_t C_SP[2] = {A3_ROWS + 2 * B_ROWS, A3_ROWS + 2 * B_ROWS + C_ROWS};
#else
constexpr uint32_t C_SP[2] = {2 * A_ROWS + 2 * B_ROWS, 2 * A_ROWS + 2 * B_ROWS + C_ROWS};
#endif
static_assert(C_SP[1] + C_ROWS <= mx::SPAD_ROWS, "scratchpad budget");
constexpr uint32_t ASC_STEP = TM * TK / 32, BSC_STEP = TN * TK / 32;   // scale bytes per k-step
constexpr mx::Fmt FMT = FP4 ? mx::FP4 : mx::FP8;
#if GEMM_BDUP
static inline uint32_t b_base(uint32_t tb) { return tb ? GEMM_B2_ADDR : GEMM_B_ADDR; }
static inline uint32_t bsc_base(uint32_t tb) { return tb ? GEMM_BSC2_ADDR : GEMM_BSC_ADDR; }
#else
static inline uint32_t b_base(uint32_t) { return GEMM_B_ADDR; }
static inline uint32_t bsc_base(uint32_t) { return GEMM_BSC_ADDR; }
#endif
#if GEMM_PRETILE
// Operands pre-tiled in DRAM: A block (m-tile, k-step) is [TM/VPB][TK] contiguous, B block
// (k-step, n-tile) is [TK][TN/VPB] contiguous, so each mvin streams consecutive bytes.
constexpr uint32_t A_STRIDE = TK, B_STRIDE = TN / VPB;
static inline uint32_t a_block(uint32_t mt, uint32_t s) { return GEMM_A_ADDR + (mt * KS + s) * (TM / VPB) * TK; }
static inline uint32_t b_block(uint32_t tb, uint32_t s, uint32_t nt) { return b_base(tb) + (s * NT + nt) * TK * (TN / VPB); }
#else
constexpr uint32_t A_STRIDE = GEMM_K, B_STRIDE = GEMM_N / VPB;
static inline uint32_t a_block(uint32_t mt, uint32_t s) { return GEMM_A_ADDR + (mt * TM / VPB) * GEMM_K + s * TK; }
static inline uint32_t b_block(uint32_t tb, uint32_t s, uint32_t nt) { return b_base(tb) + s * TK * (GEMM_N / VPB) + nt * TN / VPB; }
#endif

// dummy destination for the requantizer's scale flush (never used: C is bf16)
static uint32_t scale_flush_sink[64] __attribute__((aligned(64)));

#ifndef GEMM_DIAG
#define GEMM_DIAG 0
#endif
#ifndef GEMM_V2
#define GEMM_V2 1
#endif
#ifndef GEMM_REPOISON
#define GEMM_REPOISON 1
#endif
#ifndef GEMM_HOLD   // zero-length scale load; obsolete with config_scale's wait bit (see mx.h)
#define GEMM_HOLD 0
#endif
#ifndef GEMM_V2_FENCE
#define GEMM_V2_FENCE 0
#endif

static inline void issue_loads(uint32_t g, uint32_t b, uint32_t mt0) {
  const uint32_t t = g / KS, s = g % KS, mt = mt0 + t / NT, nt = t % NT;
  mx::set_ab_dram(mx::host_addr(a_block(mt, s)), mx::host_addr(b_block(0, s, nt)), A_STRIDE, B_STRIDE);
  mx::loop_ws_spad(I, J, KT, A_SP[b], B_END[b], 0, false, false,
                   {false, false, true, true, true});
  mx::load_scales(mx::host_addr(GEMM_ASC_ADDR + (mt * KS + s) * ASC_STEP), ASC_STEP, false,
                  b ? mx::SF_BUF1 : 0);
  mx::load_scales(mx::host_addr(GEMM_BSC_ADDR + (nt * KS + s) * BSC_STEP), BSC_STEP, true,
                  b ? mx::SF_BUF1 : 0);
  // A zero-length load waits for the loader to go idle, so it holds the command stream until
  // both scale loads above have landed: the next loop cannot read half-written scales.
  if (GEMM_HOLD) mx::load_scales(mx::host_addr(scale_flush_sink), 0, false, 0);
}

constexpr uint32_t POISON = 0xFFFFFFFFu;   // two bf16 NaNs: never produced from finite operands

// Copy C tile t from its SMEM buffer to DRAM.  v2: each word is taken once its store has landed
// (the buffer is poisoned beforehand) and re-poisoned for the tile that reuses the buffer.
static void copy_out(uint32_t t, uint32_t mt0, uint32_t ctid, uint32_t cthreads) {
  const uint32_t mt = mt0 + t / NT, nt = t % NT;
  volatile __shared uint32_t *src = (volatile __shared uint32_t *)(C_SP[t & 1] * 16);
  uint32_t *dst = (uint32_t *)GEMM_C_ADDR;
  constexpr uint32_t WPR = TN / 2;   // 32-bit words per C row
  for (uint32_t w = ctid; w < TM * WPR; w += cthreads) {
    const uint32_t r = w / WPR, c = w % WPR;
    uint32_t v = src[w];
    if (GEMM_V2) { while (v == POISON) v = src[w]; if (GEMM_REPOISON) src[w] = POISON; }
    dst[((mt * TM + r) * GEMM_N + nt * TN) / 2 + c] = v;
  }
}

static void entry_v1(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t warps = tpb / MU_NUM_THREADS;
  const uint32_t mt0 = tb * MT_C;
  if (tid < MU_NUM_THREADS) {            // warp 0: producer (lane 0 issues)
    const bool lead = tid == 0;
    if (lead) {
      mx::flush_tlb();
      mx::lut_disable();
      mx::config_ex(FMT, FMT, mx::BF16);
      mx::config_ld(A_STRIDE, 0);
      mx::config_ld(B_STRIDE, 1);
      issue_loads(0, 0, mt0);
    } else asm volatile("nop");
    for (uint32_t t = 0; t < T; t++) {
      for (uint32_t s = 0; s < KS; s++) {
        const uint32_t g = t * KS + s, b = g & 1;
        const bool last = s == KS - 1;
        if (lead) mx::fence(); else asm volatile("nop");
        if (s == 0) mu_barrier(1, warps);
        if (lead) {
          if (tb == 0 && g == 0) nightly_phase(0);
          mx::config_scale(I, J, KT, b, b, mx::host_addr(scale_flush_sink));
          mx::loop_ws_spad(I, J, KT, A_SP[b], B_END[b], C_SP[t & 1], s > 0, last,
                           {true, true, true, false, !last});
          if (g + 1 < G) issue_loads(g + 1, b ^ 1, mt0);
        } else asm volatile("nop");
      }
    }
    if (lead) { mx::fence(); if (tb == 0) nightly_phase(1); } else asm volatile("nop");
    mu_barrier(1, warps);
  } else {                                // consumers: copy finished C tiles to DRAM
    const uint32_t ctid = tid - MU_NUM_THREADS, cthreads = tpb - MU_NUM_THREADS;
    for (uint32_t t = 0; t <= T; t++) {
      mu_barrier(1, warps);
      if (t > 0) copy_out(t - 1, mt0, ctid, cthreads);
    }
  }
  nightly_kernel_end(tid, tpb);
  nightly_verify_bf16(GEMM_C_ADDR, GEMM_G_ADDR, GEMM_M * GEMM_N, tid, tpb, GEMM_DIAG ? GEMM_N : 0, TM, TN);
}

// v2: no fence in the steady state.  Ordering comes from the command stream itself:
//   compute g | loads g+1 -> 1-b | scales g+1 -> 1-b | zero-length load | compute g+1 ...
// A scale load waits until the preceding loops are unrolled; compute g has far more commands than
// the reservation station holds, so by then compute g-1 (the previous user of buffer 1-b) has
// executed.  The zero-length load then holds the stream until the scales landed.  The producer
// only waits for the consumers, before a k-step whose store reuses a C buffer.
static void entry_v2(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t warps = tpb / MU_NUM_THREADS;
  const uint32_t mt0 = tb * MT_C;
  // poison both C buffers before any store can land
  for (uint32_t w = tid; w < 2 * C_ROWS * 4; w += tpb)
    ((volatile __shared uint32_t *)(C_SP[0] * 16))[w] = POISON;
  mu_fence_smem();
  mu_barrier(2, warps);
  if (tid < MU_NUM_THREADS) {            // warp 0: producer (lane 0 issues)
    const bool lead = tid == 0;
    if (lead) {
      mx::flush_tlb();
      mx::lut_disable();
      mx::config_ex(FMT, FMT, mx::BF16);
      mx::config_ld(A_STRIDE, 0);
      mx::config_ld(B_STRIDE, 1);
      issue_loads(0, 0, mt0);
      if (tb == 0) nightly_phase(0);
    } else asm volatile("nop");
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t t = g / KS, s = g % KS, b = g & 1;
      const bool last = s == KS - 1;
      if (last && t >= 2) mu_barrier(1, warps);   // consumers have copied tile t-2
      if (lead) {
        if (tb == 0) nightly_heartbeat(g);
        mx::config_scale(I, J, KT, b, b, mx::host_addr(scale_flush_sink));
        mx::loop_ws_spad(I, J, KT, A_SP[b], B_END[b], C_SP[t & 1], s > 0, last,
                         {true, true, true, false, !last});
        if (g + 1 < G) issue_loads(g + 1, b ^ 1, mt0);
        if (GEMM_V2_FENCE) mx::fence();   // bisection: v2 stream with v1's per-step drain
      } else asm volatile("nop");
    }
    if (lead) { mx::fence(); if (tb == 0) nightly_phase(1); } else asm volatile("nop");
    for (uint32_t t = (T >= 2 ? T - 2 : 0); t < T; t++) mu_barrier(1, warps);
  } else {                                // consumers: copy finished C tiles to DRAM
    const uint32_t ctid = tid - MU_NUM_THREADS, cthreads = tpb - MU_NUM_THREADS;
    for (uint32_t t = 0; t < T; t++) {
      copy_out(t, mt0, ctid, cthreads);
      mu_fence_smem();
      mu_barrier(1, warps);
    }
  }
  nightly_kernel_end(tid, tpb);
  nightly_verify_bf16(GEMM_C_ADDR, GEMM_G_ADDR, GEMM_M * GEMM_N, tid, tpb, GEMM_DIAG ? GEMM_N : 0, TM, TN);
}

#if GEMM_V3
// v3: A resident per m-tile, only B streams.  Operand traffic per MAC drops from 1/TN + 1/TM to about
// 1/N + 1/TM elements, which the gemmini DMA sustains (32 B Gets, 16 in flight).
//   step g = (m-tile, n-tile, k-step s):  CONFIG_SCALE_MEM(b), compute g (A chunk s, B buffer b),
//   loads g+1 -> 1-b, [last n-tile of the m-tile: chunk s of the next m-tile's A over chunk s].
// The reservation station orders each mvin after the computes that read its destination rows.
static inline void load_a3(uint32_t mt, uint32_t s) {
  mx::set_ab_dram(mx::host_addr(a_block(mt, s)), mx::host_addr(GEMM_B_ADDR), A_STRIDE, B_STRIDE);
  mx::loop_ws_spad(I, J, KT, s * A_ROWS, B3_END[0], 0, false, false, {false, true, true, true, true});
}

static inline void load_b3(uint32_t g, uint32_t b, uint32_t mt0, uint32_t tb) {
  const uint32_t t = g / KS, s = g % KS, mt = mt0 + t / NT, nt = t % NT;
  mx::set_ab_dram(mx::host_addr(GEMM_A_ADDR), mx::host_addr(b_block(tb, s, nt)), A_STRIDE, B_STRIDE);
  mx::loop_ws_spad(I, J, KT, 0, B3_END[b], 0, false, false, {true, false, true, true, true});
  mx::load_scales(mx::host_addr(GEMM_ASC_ADDR + (mt * KS + s) * ASC_STEP), ASC_STEP, false, b ? mx::SF_BUF1 : 0);
  mx::load_scales(mx::host_addr(bsc_base(tb) + (nt * KS + s) * BSC_STEP), BSC_STEP, true, b ? mx::SF_BUF1 : 0);
  if (GEMM_HOLD) mx::load_scales(mx::host_addr(scale_flush_sink), 0, false, 0);
}

#ifndef GEMM_CDRAM
#define GEMM_CDRAM 1
#endif
#ifndef GEMM_MANAGED
#define GEMM_MANAGED 0
#endif
static_assert(!GEMM_MANAGED || KS % 2 == 0, "managed: the A reload must be an even number of loops");
static_assert(!GEMM_MANAGED || !FP4, "loop-managed scales: E4M3 only (fp4 scale rows are not sized by the loop)");
// Managed step g: one loop loads B (k-step s, n-tile nt) into buffer b and computes with resident A
// chunk s; it also loads its own scale slices (A: KT/2 rows of TM bytes, B: KT/2 rows of TN bytes,
// contiguous per (tile, k-step) in DRAM) and orders them in hardware.  A = 0 lets the loop reuse
// the A scales already resident in its half (same slice two steps earlier, other n-tile).
static inline void step_managed(uint32_t g, uint32_t b, uint32_t mt0, uint32_t tb, uint32_t c_row,
                                bool acc, bool last) {
  const uint32_t t = g / KS, s = g % KS, mt = mt0 + t / NT, nt = t % NT;
  mx::set_ab_dram(0, mx::host_addr(b_block(tb, s, nt)), A_STRIDE, B_STRIDE);
  mx::loop_scales(mx::host_addr(GEMM_ASC_ADDR + (mt * KS + s) * ASC_STEP),
                  mx::host_addr(bsc_base(tb) + (nt * KS + s) * BSC_STEP), TM, TN);
  mx::loop_ws_spad(I, J, KT, s * A_ROWS, B3_END[b], c_row, acc, last,
                   {true, false, true, false, GEMM_CDRAM || !last});
}

// CDRAM: gemmini stores each C tile from the accumulator straight to DRAM (row-major bf16), no SIMT
// copy.  Tile t lives in accumulator half (t & 1) (only a tile's last k-step toggles the half).
constexpr uint32_t C_ROW_BYTES = GEMM_N * 2;
static inline void mvout_tile(uint32_t t, uint32_t mt0) {
  const uint32_t mt = mt0 + t / NT, nt = t % NT, half = (t & 1) * mx::ACC_HALF;
  const uint64_t base = mx::host_addr(GEMM_C_ADDR) + (uint64_t)mt * TM * C_ROW_BYTES + nt * TN * 2;
  if (FP4) {   // 4 row groups of 32 output rows x 2 column groups of 32; acc row = two output rows
    for (uint32_t i = 0; i < 4; i++)
      for (uint32_t j = 0; j < 2; j++)
        mx::mvout_acc(base + i * 32 * C_ROW_BYTES + j * 64, half + (2 * i + j) * 16, 16, 16, 0, C_ROW_BYTES / 64);
  } else {     // 8 row groups of 16 rows x 2 chunks of 32 columns
    for (uint32_t i = 0; i < I; i++)
      for (uint32_t c = 0; c < 2; c++)
        mx::mvout_acc(base + i * 16 * C_ROW_BYTES + c * 64, half + i * 16, 16, 16, c);
  }
}

static void entry_v3(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t warps = tpb / MU_NUM_THREADS;
  const uint32_t mt0 = tb * MT_C;
  if (!GEMM_CDRAM) {
    for (uint32_t w = tid; w < 2 * C_ROWS * 4; w += tpb)
      ((volatile __shared uint32_t *)(C_SP[0] * 16))[w] = POISON;
  }
  mu_fence_smem();
  mu_barrier(2, warps);
  if (tid < MU_NUM_THREADS) {            // warp 0: producer (lane 0 issues)
    const bool lead = tid == 0;
    if (lead) {
      mx::flush_tlb();
      mx::lut_disable();
      mx::config_ex(FMT, FMT, mx::BF16);
      mx::config_ld(A_STRIDE, 0);
      mx::config_ld(B_STRIDE, 1);
      if (GEMM_CDRAM) mx::config_st(FP4 ? 2 * C_ROW_BYTES : C_ROW_BYTES);
      for (uint32_t s = 0; s < KS; s++) load_a3(mt0, s);
      if (!GEMM_MANAGED) load_b3(0, 0, mt0, tb);
      if (tb == 0) nightly_phase(0);
    } else asm volatile("nop");
    for (uint32_t g = 0; g < G; g++) {
      const uint32_t t = g / KS, s = g % KS, b = g & 1;
      const bool last = s == KS - 1;
      if (!GEMM_CDRAM && last && t >= 2) mu_barrier(1, warps);   // consumers have copied tile t-2
      if (lead) {
        if (tb == 0) nightly_heartbeat(g);
        if (GEMM_MANAGED) {
          step_managed(g, b, mt0, tb, C_SP[t & 1], s > 0, last);
          // the next m-tile's A chunks, all after its last compute: an even number of loops between
          // two managed steps keeps their loop slots (and so their scale halves) alternating
          if (last && t % NT == NT - 1 && t / NT + 1 < MT_C)
            for (uint32_t c = 0; c < KS; c++) load_a3(mt0 + t / NT + 1, c);
        } else {
        mx::config_scale(I, J, KT, b, b, mx::host_addr(scale_flush_sink));
        mx::loop_ws_spad(I, J, KT, s * A_ROWS, B3_END[b], C_SP[t & 1], s > 0, last,
                         {true, true, true, false, GEMM_CDRAM || !last});
        if (g + 1 < G) load_b3(g + 1, b ^ 1, mt0, tb);
        // after load_b3: its scale loads wait until compute g is fully unrolled, so the reservation
        // station holds every command that reads chunk s and orders this mvin after them (WAR)
        if (t % NT == NT - 1 && t / NT + 1 < MT_C) load_a3(mt0 + t / NT + 1, s);
        }
        // CDRAM: tile t-1's stores go after tile t's first compute, so they never hold it back
        if (GEMM_CDRAM && s == 0 && t >= 1) mvout_tile(t - 1, mt0);
      } else asm volatile("nop");
    }
    if (lead) {
      if (GEMM_CDRAM) mvout_tile(T - 1, mt0);
      mx::fence();
      if (tb == 0) nightly_phase(1);
    } else asm volatile("nop");
    if (!GEMM_CDRAM) for (uint32_t t = (T >= 2 ? T - 2 : 0); t < T; t++) mu_barrier(1, warps);
  } else if (!GEMM_CDRAM) {               // consumers: copy finished C tiles to DRAM
    const uint32_t ctid = tid - MU_NUM_THREADS, cthreads = tpb - MU_NUM_THREADS;
    for (uint32_t t = 0; t < T; t++) {
      copy_out(t, mt0, ctid, cthreads);
      mu_fence_smem();
      mu_barrier(1, warps);
    }
  } else {
    asm volatile("nop");
  }
  nightly_kernel_end(tid, tpb);
  nightly_verify_bf16(GEMM_C_ADDR, GEMM_G_ADDR, GEMM_M * GEMM_N, tid, tpb, GEMM_DIAG ? GEMM_N : 0, TM, TN);
}
#endif

int main() {
#if GEMM_V3
  return nightly_main(entry_v3, nullptr);
#else
  return nightly_main(GEMM_V2 ? entry_v2 : entry_v1, nullptr);
#endif
}
