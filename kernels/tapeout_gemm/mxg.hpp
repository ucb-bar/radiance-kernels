// mxg.hpp: MX-FP8 GEMM on the tapeout MX mesh, driven by one thread through the MMIO command
// interface.  Shared by kernels/tapeout_gemm (benchmarks) and kernels/tapeout_encoder.
//
// C[M][N] (bf16) = A[M][K] x B[K][N] (e4m3, one e8m0 scale per 32 elements along K).
//   A codes row-major [M][K]; A scales tile-blocked [M/TM][K/TK][TK/32][TM]
//   B codes row-major [K][N]; B scales tile-blocked [N/TN][K/TK][TK/32][TN]
//   C row-major bf16, row stride N.
// Tiles: TM = TN = 128 (C fits ACCMEM: 256 rows x 64 outputs), TK = 128; A/B double-buffered in
// SMEM (A at rows 0 / 2048, B ending at 8192 / 6144), scale SRAM double-buffered by halves.
//
// Operands must not contain e4m3 subnormals: the tapeout mesh corrupts dense accumulations
// that include them.  Verified on the U250 board (bitstream fe4dc316) against exact arithmetic.
//
// Requires: radiance.h, gemmini.h + mxgemmini_mmio.h included before this header.

#ifndef TAPEOUT_MXG_HPP
#define TAPEOUT_MXG_HPP

#include <stdint.h>

namespace mxg {

static constexpr uint32_t FORMAT_FP8 = 0;
static constexpr uint32_t FORMAT_FULL = 3;
static constexpr uint32_t ACC_ADDR = 1u << (ADDR_LEN - 1);
static constexpr uint32_t SMEM_ROWS = BANK_NUM * BANK_ROWS;   // 8192 rows of 16 B
static constexpr uint32_t ACC_ELEMS_PER_ROW = 4 * DIM;        // one ACCMEM row = 64 outputs

struct Problem {
  uint32_t a, b, c, sa, sb;   // device addresses
  uint32_t m, n, k;           // shape
  uint32_t tm, tn, tk;        // tile (multiples of 16; tk multiple of 32; tm*tn <= 16384)
  uint32_t scratch;           // device address the CONFIG_SCALE_MEM command needs; never written
  // Columns this launch computes, starting at b / c (0 = all n).  n stays the row stride of B
  // and C, so a column split over clusters offsets b, sb and c and sets ncols = n / clusters.
  uint32_t ncols = 0;
  // MXG_RESACC: residual R (bf16, same layout and row stride as C) added into each C tile in ACCMEM
  // before its move-out (0 = none).
  uint32_t res = 0;
};
static inline uint32_t ncols_of(const Problem &p) { return p.ncols ? p.ncols : p.n; }

struct Tiling {
  uint32_t i, j, k;    // tile extent in 16-element PE tiles
  uint32_t lda, ldb;   // operand row strides in bytes
  uint32_t ldc;        // C row stride in bytes (bf16)
};

static inline Tiling tiling_of(const Problem &p) {
  return {p.tm / DIM, p.tn / DIM, p.tk / DIM, p.k, p.n, p.n * 2u};
}
static inline uint64_t host_addr(uint32_t device_addr) { return rad_device_to_host_address(device_addr); }

// SMEM rows of the double-buffered operands (the FSM takes B's END row).
#ifdef MXG_BANKSPLIT
// MXG_BANKSPLIT (TK = 128): both A buffers in bank 0 (rows 0-2047), both B buffers in bank 3
// (rows 6144-8191), the C slot in bank 1 and banks 1-2 free of mesh operands, so SIMT code may load
// from them while a matmul runs (memory note smem-overlap-rules).  Only tilings whose A and B tiles
// fit 1024 rows use it: the TK = 256 GEMMs of the same image keep the default map (with the split
// map their buffers overlapped: board fzl, att and everything after it wrong).
static inline bool banksplit(const Tiling &t) {
  return t.i * t.k * DIM <= SMEM_ROWS / 8 && t.k * t.j * DIM <= SMEM_ROWS / 8;
}
#else
static inline bool banksplit(const Tiling &) { return false; }
#endif
static inline uint32_t a_spad(uint32_t parity, const Tiling &t) {
  return (parity & 1) ? (banksplit(t) ? SMEM_ROWS / 8 : SMEM_ROWS / 4) : 0;
}
static inline uint32_t b_spad_end(uint32_t parity, const Tiling &t) {
  return (parity & 1) ? SMEM_ROWS - (banksplit(t) ? SMEM_ROWS / 8 : SMEM_ROWS / 4) : SMEM_ROWS;
}
// C_SPAD: C goes ACCMEM -> SMEM inside the last k-step's LOOP_WS and leaves SMEM by DMA under the
// next tile (run_pipe<true>); the default follows MXG_C_SPAD.
#ifdef MXG_C_SPAD
static constexpr bool C_SPAD_DEFAULT = true;
#else
static constexpr bool C_SPAD_DEFAULT = false;
#endif
static void configure(const Problem &p, const Tiling &t, bool c_spad = C_SPAD_DEFAULT) {
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              FORMAT_FP8, FORMAT_FP8, FORMAT_FULL, false);
  gemmini_extended3_config_ld(t.lda, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(t.ldb, MVIN_SCALE_IDENTITY, false, 1);
  if (c_spad) {
    gemmini_config_st(DIM);       // SMEM move-out: 16-byte rows, consecutive in DRAM
  } else {
    gemmini_config_st(t.ldc);     // ACCMEM move-out: C rows at the C row stride
  }
  gemmini_mxquant_config_mvout(host_addr(p.scratch), t.i, t.j, t.k, 0, 0, 1);
  // The loop FSM bounds are written twice, once per FSM, as in lib/mxgemm/mxgemm_lib.hpp.
  gemmini_loop_ws_config_bounds(t.i, t.j, t.k, 0, 0, 0);
  gemmini_fence();
  gemmini_loop_ws_config_bounds(t.i, t.j, t.k, 0, 0, 0);
  gemmini_fence();
}

// Wait until every issued command has finished.  gemmini_fence() alone is NOT enough: it polls
// io.busy, which rises only once a command reaches the reservation station, so a poll right after
// the issuing store can return before a loop_ws has started (FA_SP_WCNT in
// kernels/fa_stable_fpga/kernel.cpp).  runningLoops (MMIO 0x28) counts LOOP_WS from the issue
// cycle; the busy poll then covers the plain mvin/mvout DMAs.
static inline void drain() {
  gemmini_fence_waitcount(0);
  gemmini_fence();
}

// Select the scale-SRAM half the next matmul reads.  Bounds must stay constant.
static inline void select_scales(const Problem &p, const Tiling &t, uint32_t parity) {
  const uint32_t odd = parity & 1;
  gemmini_mxquant_config_mvout(host_addr(p.scratch), t.i, t.j, t.k, odd, odd, 1);
}

// DMA one A tile (tm x tk) and one B tile (tk x tn) from DRAM into SMEM buffer `parity`.
static inline void mvin_tiles(const Tiling &t, uint32_t a_dram, uint32_t b_dram, uint32_t parity) {
#ifdef MXG_MVIN_WIDE
  // MXG_MVIN_WIDE: plain 2-block move-ins (cols = 32 B; the tapeout's cols field is 6 bits, so 2
  // blocks is the maximum) in the loop FSM's layout: A block (i, k) at a_spad + (i K + k) DIM, B
  // block (k, j) at b_end - K J DIM + (k J + j) DIM.  Load state 0 holds A's row stride and state 1
  // B's (configure()).  Cold move-in benchmark, FireAxe pair: the loop FSM's move-ins ran at 4.3
  // B/cycle on one cluster and 3.2 on two.
  const uint32_t a0 = a_spad(parity, t), b0 = b_spad_end(parity, t) - t.k * t.j * DIM;
#pragma clang loop unroll(disable)
  for (uint32_t i = 0; i < t.i; i++)
#pragma clang loop unroll(disable)
    for (uint32_t k = 0; k < t.k; k += 2u)
      gemmini_extended_mvin(host_addr(a_dram + i * DIM * t.lda + k * DIM), a0 + (i * t.k + k) * DIM, 2u * DIM, DIM);
#pragma clang loop unroll(disable)
  for (uint32_t k = 0; k < t.k; k++)
#pragma clang loop unroll(disable)
    for (uint32_t j = 0; j < t.j; j += 2u)
      gemmini_extended_mvin2(host_addr(b_dram + k * DIM * t.ldb + j * DIM), b0 + (k * t.j + j) * DIM, 2u * DIM, DIM);
  return;
#endif
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, host_addr(a_dram), host_addr(b_dram),
                           k_LOOP_WS_CONFIG_ADDRS_AB)
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)t.lda, (uint64_t)t.ldb,
                           k_LOOP_WS_CONFIG_STRIDES_AB)
  constexpr uint32_t skips = loop_matmul_skips(0, 0, 1, 1, 1);
  gemmini_loop_ws_spad(t.i, t.j, t.k, 0, 0, 0, a_spad(parity, t), b_spad_end(parity, t), 0, 0,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, skips);
}

// MXG_C_SPAD: the last k-step's matmul also stores C (bf16, row-major tm x tn) from ACCMEM into
// SMEM at C_SPAD_ROW (free for TK = 128: A uses [0,1024) and [2048,3072), B uses [5120,6144) and
// [7168,8192)), and C leaves SMEM by DMA move-outs that overlap the next tile's work.
#ifdef MXG_BANKSPLIT
static constexpr uint32_t C_SPAD_ROW = 2048;   // bank 1
#else
static constexpr uint32_t C_SPAD_ROW = 3072;
#endif

// Matmul on SMEM buffer `parity` into ACCMEM (overwrite if `first`, else accumulate); with
// `store_c` the FSM also stores C into SMEM at C_SPAD_ROW.
static inline void matmul_tile(const Tiling &t, uint32_t parity, bool first, bool store_c = false) {
  const uint32_t skips = loop_matmul_skips(1, 1, 1, 0, store_c ? 0 : 1);
  gemmini_loop_ws_spad(t.i, t.j, t.k, 0, 0, 0, a_spad(parity, t), b_spad_end(parity, t), 0, C_SPAD_ROW,
                       false, false, false, false, !first, NO_ACTIVATION, 0, 0, false, skips);
}

// Move C of output tile (mi, nj) from SMEM (C_SPAD_ROW) to DRAM: one move-out per C row
// (tn bf16 = tn/8 SMEM rows of 16 B); needs gemmini_config_st(16) (MXG_C_SPAD configure).
static inline void mvout_c_spad(const Problem &p, uint32_t mi, uint32_t nj) {
  const uint32_t rows_per_c_row = p.tn * 2u / DIM;
  for (uint32_t r = 0; r < p.tm; r++)
    gemmini_extended_mvout(host_addr(p.c + ((mi * p.tm + r) * p.n + nj * p.tn) * 2u),
                           C_SPAD_ROW + r * rows_per_c_row, DIM, rows_per_c_row);
}

// Copy `bytes` of scales into a scale-SRAM window.  One lane, word stores in ascending order
// (FlitMergeNode requires consecutive addresses), loads batched for latency.
static void __attribute__((noinline)) load_scales(uint32_t sf_base, const uint32_t *src, uint32_t bytes) {
  volatile __shared uint32_t *dst = reinterpret_cast<volatile __shared uint32_t *>(sf_base);
  constexpr uint32_t ILP = 16;
  const uint32_t words = bytes / 4;
  uint32_t i = 0;
  uint32_t v[ILP];
  for (; i + ILP <= words; i += ILP) {
#pragma unroll
    for (uint32_t j = 0; j < ILP; j++) v[j] = src[i + j];
#pragma unroll
    for (uint32_t j = 0; j < ILP; j++) dst[i + j] = v[j];
  }
  for (; i < words; i++) dst[i] = src[i];
}

// Scales of output tile (mi, nj), k-tile kt into scale-SRAM half `parity`.
static inline void load_tile_scales(const Problem &p, uint32_t mi, uint32_t nj, uint32_t kt,
                                    uint32_t parity) {
  const uint32_t k_tiles = p.k / p.tk;
  const uint32_t half = (parity & 1) ? GEMMINI_SF_MEM_BUFFER_OFFSET : 0;
  const uint32_t a_bytes = p.tm * p.tk / 32, b_bytes = p.tn * p.tk / 32;
  load_scales(GEMMINI_SF_MEM_A + half,
              reinterpret_cast<const uint32_t *>(p.sa + (mi * k_tiles + kt) * a_bytes), a_bytes);
  load_scales(GEMMINI_SF_MEM_B + half,
              reinterpret_cast<const uint32_t *>(p.sb + (nj * k_tiles + kt) * b_bytes), b_bytes);
}

// Move the ACCMEM result of output tile (mi, nj) (tm x tn bf16) to DRAM.
static inline void mvout_c(const Problem &p, uint32_t mi, uint32_t nj) {
  const uint32_t rblk = p.tm / DIM, cblk = p.tn / ACC_ELEMS_PER_ROW;
  const uint32_t c_tile = p.c + (mi * p.tm * p.n + nj * p.tn) * 2u;
  for (uint32_t r = 0; r < rblk; r++)
    for (uint32_t c = 0; c < cblk; c++)
      gemmini_mvout(host_addr(c_tile + (r * DIM * p.n + c * ACC_ELEMS_PER_ROW) * 2u),
                    ACC_ADDR + (r * cblk + c) * DIM);
}

#ifdef MXG_RESACC
// C += R for output tile (mi, nj) in ACCMEM, after the tile's last k-step has drained: accumulating
// move-ins (load state 2) of the R tile into the accumulator rows mvout_c reads.  Each element is one
// bf16 round-to-nearest-even add, the same operation as a SIMT bf16 residual add after the move-out.
// Every ACCMEM write passes the MX scale stage and advances the scale-memory read counters, which only
// a chip reset clears (AccumulatorMem.scala:281-322, ScaleFactorMem.scala:76-105).  So, for the move-ins:
// a non-FP8 activation format (FP8 keeps 4 of every 16 elements of a DMA write and zeroes the rest),
// scale bounds 1, 1, 1 with identity scales (e8m0 127 + 127 - 254 = 2^0) in row 0 of the four scale
// banks of half h, which the next step does not read, and 16 x 32 = 512 writes per tile, a multiple
// of the counter period 16.  Then the configuration is restored.  cols = 16 accumulator elements
// (4 bf16 each, 128 B, two DMA transactions), rows = 16: cols x rows must stay below 512 (the byte
// count is (cols rows)[8:0] x 8, LoadController).  Inferred from the RTL; never exercised before.
static constexpr uint32_t ACC_ACCUMULATE = 1u << (ADDR_LEN - 2);
static inline void resacc_tile(const Problem &p, const Tiling &t, uint32_t mi, uint32_t nj, uint32_t h) {
  const uint32_t off = h * GEMMINI_SF_MEM_BUFFER_OFFSET;
  const uint32_t banks[4] = {GEMMINI_SF_MEM_A + off, GEMMINI_SF_MEM_A + off + 0x400u,
                             GEMMINI_SF_MEM_B + off, GEMMINI_SF_MEM_B + off + 0x400u};
#pragma unroll
  for (uint32_t b = 0; b < 4u; b++) {
    volatile __shared uint32_t *d = reinterpret_cast<volatile __shared uint32_t *>(banks[b]);
#pragma unroll
    for (uint32_t w = 0; w < 4u; w++) d[w] = 0x7F7F7F7Fu;
  }
  mu_fence_smem();
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              1 /* not FP8 */, FORMAT_FP8, FORMAT_FULL, false);
  gemmini_mxquant_config_mvout(host_addr(p.scratch), 1, 1, 1, h, h, 1);
  gemmini_extended3_config_ld(p.n * 2u, MVIN_SCALE_IDENTITY, false, 2);
  drain();
  const uint32_t rblk = p.tm / DIM, cblk = p.tn / ACC_ELEMS_PER_ROW;
  const uint32_t r_tile = p.res + (mi * p.tm * p.n + nj * p.tn) * 2u;
  for (uint32_t r = 0; r < rblk; r++)
    for (uint32_t c = 0; c < cblk; c++)
      gemmini_extended_mvin3(host_addr(r_tile + (r * DIM * p.n + c * ACC_ELEMS_PER_ROW) * 2u),
                             ACC_ADDR | ACC_ACCUMULATE | ((r * cblk + c) * DIM), 16u, DIM);
  drain();
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false,
                              FORMAT_FP8, FORMAT_FP8, FORMAT_FULL, false);
  select_scales(p, t, h ^ 1u);
  drain();
}
#endif

// Reference loop: per output tile, prologue, k-loop with one drain per k-tile, move-out, drain.
static void run_simple(const Problem &p, const Tiling &t) {
  const uint32_t k_tiles = p.k / p.tk;
  for (uint32_t mi = 0; mi < p.m / p.tm; mi++) {
    for (uint32_t nj = 0; nj < ncols_of(p) / p.tn; nj++) {
      const uint32_t a_row = p.a + mi * p.tm * p.k, b_col = p.b + nj * p.tn;
      mvin_tiles(t, a_row, b_col, 0);
      load_tile_scales(p, mi, nj, 0, 0);
      mu_fence_smem();
      drain();
      for (uint32_t kt = 0; kt < k_tiles; kt++) {
        const bool last = kt + 1 == k_tiles;
        select_scales(p, t, kt);
        if (!last) mvin_tiles(t, a_row + (kt + 1) * p.tk, b_col + (kt + 1) * p.tk * p.n, kt + 1);
        matmul_tile(t, kt, kt == 0);
        if (!last) { load_tile_scales(p, mi, nj, kt + 1, kt + 1); mu_fence_smem(); }
        drain();
      }
      mvout_c(p, mi, nj);
      drain();
    }
  }
}

// Pipelined loop: one stream of k-steps over all output tiles.  Step s = (tile, k-tile); the A/B
// buffers and scale halves alternate with s, so the next tile's first move-in and scales are
// issued under the current tile's last matmul.  A tile's C move-out is issued after its last drain
// and drained before the next step (overlapping it with the next tile's matmul loses the C tiles).  `on_tile_done(tile)` is called after each
// tile's move-out is issued (for timestamps).
#ifdef MXG_PROF
// MXG_PROF: thread 0's time per run_pipe phase, summed over the k-steps: command issue (scale
// select, move-in and matmul LOOP_WS), the next step's scale fill, the drain wait, and the tile-end
// C move-out.  The including kernel defines mxg_prof_report() (locals, not statics: .bss is never
// initialized on this platform).
static void mxg_prof_report(uint32_t issue, uint32_t scales, uint32_t drain_wait, uint32_t c_out);
static inline uint32_t prof_cycle() { uint32_t c; asm volatile("csrr %0, mcycle" : "=r"(c)); return c; }
#define MXG_PT(v) const uint32_t v = prof_cycle()
#else
#define MXG_PT(v) do {} while (0)
#endif
template <bool CS = C_SPAD_DEFAULT, typename OnTileDone>
static void run_pipe(const Problem &p, const Tiling &t, OnTileDone on_tile_done) {
#ifdef MXG_PROF
  uint32_t pf[4] = {0, 0, 0, 0};
#endif
  const uint32_t k_tiles = p.k / p.tk;
  const uint32_t n_tiles = ncols_of(p) / p.tn;
  const uint32_t steps = (p.m / p.tm) * n_tiles * k_tiles;
  auto a_addr = [&](uint32_t s) {
    const uint32_t tile = s / k_tiles, kt = s % k_tiles;
    return p.a + (tile / n_tiles) * p.tm * p.k + kt * p.tk;
  };
  auto b_addr = [&](uint32_t s) {
    const uint32_t tile = s / k_tiles, kt = s % k_tiles;
    return p.b + (tile % n_tiles) * p.tn + kt * p.tk * p.n;
  };
  auto scales_of = [&](uint32_t s) {
    const uint32_t tile = s / k_tiles;
    load_tile_scales(p, tile / n_tiles, tile % n_tiles, s % k_tiles, s);
  };
  mvin_tiles(t, a_addr(0), b_addr(0), 0);
  scales_of(0);
  mu_fence_smem();
  drain();
  uint32_t out_tile = ~0u;   // CS: the tile whose C sits in the C slot, moved out in the next step
  for (uint32_t s = 0; s < steps; s++) {
    const uint32_t kt = s % k_tiles;
    const bool more = s + 1 < steps;
    MXG_PT(c0);
    select_scales(p, t, s);
#ifndef MXG_MVIN_WIDE
    if (more) mvin_tiles(t, a_addr(s + 1), b_addr(s + 1), s + 1);
#endif
    matmul_tile(t, s, kt == 0, CS && kt == k_tiles - 1);
#ifdef MXG_MVIN_WIDE   // 64-128 commands that can block thread 0 on a full load queue: after the matmul
    if (more) mvin_tiles(t, a_addr(s + 1), b_addr(s + 1), s + 1);
#endif
    if constexpr (CS) {
      // C already left ACCMEM inside the FSM (the previous tile's last step); its SMEM move-outs go
      // in AFTER this step's LOOP_WS commands and drain with this step, under its matmul.  Issued
      // right after the tile's own drain instead, they sat in front of the next step's LOOP_WS, and
      // a LOOP_WS write behind a full command queue inflates runningLoops (memory note
      // gemmini-runningloops-overcount): the encoder's down GEMM hung in drain() (board fzz2).  The
      // next store into C_SPAD_ROW is k_tiles - 1 steps (and drains) later.  The C slot must not
      // share a bank with the operand buffers (MXG_BANKSPLIT): with the TK = 128 default map it
      // does, and the 2-cluster GEMM hung on the board (fzx xq128c / xd128c).
      if (out_tile != ~0u) {
#ifndef MXG_C_SPAD_NOOUT   // timing probe: C stays in SMEM (cost of the FSM's store alone)
        mvout_c_spad(p, out_tile / n_tiles, out_tile % n_tiles);
#endif
        on_tile_done(out_tile);
        out_tile = ~0u;
      }
    }
    MXG_PT(c1);
    if (more) { scales_of(s + 1); mu_fence_smem(); }
    MXG_PT(c2);
    drain();
    MXG_PT(c3);
#ifdef MXG_PROF
    pf[0] += c1 - c0; pf[1] += c2 - c1; pf[2] += c3 - c2;
#endif
    if constexpr (CS) {
      if (kt == k_tiles - 1) out_tile = s / k_tiles;
      continue;
    }
    if (kt == k_tiles - 1) {
      const uint32_t tile = s / k_tiles;
#ifdef MXG_RESACC
      if (p.res) resacc_tile(p, t, tile / n_tiles, tile % n_tiles, s & 1u);
#endif
      mvout_c(p, tile / n_tiles, tile % n_tiles);
      // Measured on the board: issuing the next step's commands (scale select, move-in, matmul)
      // while a tile's ACCMEM move-out is still running loses every move-out of a multi-tile GEMM
      // (C all poison) and the mesh drops the remaining work.  Drain the move-out first.
      drain();
      on_tile_done(tile);
#ifdef MXG_PROF
      pf[3] += prof_cycle() - c3;
#endif
    }
  }
  if constexpr (CS) {   // the last tile's C
    if (out_tile != ~0u) {
#ifndef MXG_C_SPAD_NOOUT
      mvout_c_spad(p, out_tile / n_tiles, out_tile % n_tiles);
#endif
      on_tile_done(out_tile);
    }
  }
  drain();
#ifdef MXG_PROF
  mxg_prof_report(pf[0], pf[1], pf[2], pf[3]);
#endif
}

// ---- A-resident pipeline (K * TM + 2 * TK * TN <= 128 KB, e.g. K = 512, TM = TN = 128, TK = 256) ----
// run_pipe streams one A and one B tile per k-step of every output tile.  Here the whole A row
// block (TM x K) stays in SMEM rows [0, K TM / 16) while the block's output tiles run, and only B
// tiles stream (double-buffered, ending at rows 6144 / 8192): DRAM traffic per k-step halves.  At a
// row-block change the next A block is moved in after a drain (no room to double-buffer it).
static inline void mvin_a(const Tiling &t, uint32_t a_dram, uint32_t a_row) {
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, host_addr(a_dram), 0, k_LOOP_WS_CONFIG_ADDRS_AB)
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)t.lda, (uint64_t)t.ldb, k_LOOP_WS_CONFIG_STRIDES_AB)
  constexpr uint32_t skips = loop_matmul_skips(0, 1, 1, 1, 1);   // load A only
  gemmini_loop_ws_spad(t.i, t.j, t.k, 0, 0, 0, a_row, 0, 0, 0,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, skips);
}
static inline void mvin_b(const Tiling &t, uint32_t b_dram, uint32_t b_end) {
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, 0, host_addr(b_dram), k_LOOP_WS_CONFIG_ADDRS_AB)
  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)t.lda, (uint64_t)t.ldb, k_LOOP_WS_CONFIG_STRIDES_AB)
  constexpr uint32_t skips = loop_matmul_skips(1, 0, 1, 1, 1);   // load B only
  gemmini_loop_ws_spad(t.i, t.j, t.k, 0, 0, 0, 0, b_end, 0, 0,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, skips);
}
static inline void matmul_at(const Tiling &t, uint32_t a_row, uint32_t b_end, bool first) {
  constexpr uint32_t skips = loop_matmul_skips(1, 1, 1, 0, 1);
  gemmini_loop_ws_spad(t.i, t.j, t.k, 0, 0, 0, a_row, b_end, 0, C_SPAD_ROW,
                       false, false, false, false, !first, NO_ACTIVATION, 0, 0, false, skips);
}
template <typename OnTileDone>
static void run_pipe_ares(const Problem &p, const Tiling &t, OnTileDone on_tile_done) {
  const uint32_t k_tiles = p.k / p.tk, n_tiles = ncols_of(p) / p.tn;
  const uint32_t per_block = n_tiles * k_tiles, steps = (p.m / p.tm) * per_block;
  const uint32_t a_rows = p.tm * p.tk / DIM;              // SMEM rows of one A k-tile
  auto b_end = [](uint32_t s) { return (s & 1u) ? SMEM_ROWS : SMEM_ROWS - SMEM_ROWS / 4u; };
  auto b_addr = [&](uint32_t s) {
    return p.b + ((s / k_tiles) % n_tiles) * p.tn + (s % k_tiles) * p.tk * p.n;
  };
  auto load_a_block = [&](uint32_t mi) {
    for (uint32_t kt = 0; kt < k_tiles; kt++) mvin_a(t, p.a + mi * p.tm * p.k + kt * p.tk, kt * a_rows);
  };
  auto scales_of = [&](uint32_t s) {
    const uint32_t tile = s / k_tiles;
    load_tile_scales(p, tile / n_tiles, tile % n_tiles, s % k_tiles, s);
  };
  load_a_block(0);
  mvin_b(t, b_addr(0), b_end(0));
  scales_of(0);
  mu_fence_smem();
  drain();
  for (uint32_t s = 0; s < steps; s++) {
    const uint32_t kt = s % k_tiles, tile = s / k_tiles;
    const bool more = s + 1 < steps, new_block = more && (s + 1) % per_block == 0;
    select_scales(p, t, s);
    if (more && !new_block) mvin_b(t, b_addr(s + 1), b_end(s + 1));
    matmul_at(t, kt * a_rows, b_end(s), kt == 0);
    if (more) { scales_of(s + 1); mu_fence_smem(); }
    drain();
    if (kt == k_tiles - 1) {
      mvout_c(p, tile / n_tiles, tile % n_tiles);
      drain();   // as run_pipe: a move-out overlapped with the next step's commands loses C
      on_tile_done(tile);
    }
    if (new_block) {   // the block's last matmul has drained: replace A, then the next B
      load_a_block((s + 1) / per_block);
      mvin_b(t, b_addr(s + 1), b_end(s + 1));
      drain();
    }
  }
  drain();
}


// ---- A-resident + TK = 128 + C slot (run_pipe_ares_cs; K * TM <= 64 KB, e.g. K = 512) -----------
// On two clusters the streaming GEMM is bound by its move-ins (64 KB per 16.4k-cycle k-step at TK =
// 256 against ~3.2 B/cycle per cluster: board fzk) and by the drained 32 KB ACCMEM move-out of every
// tile.  Here A stays resident (only B streams: 16 KB per 8.2k-cycle k-step) and C leaves through an
// SMEM slot under the next tile.  Each operand in its own bank, so the C move-outs never read a bank
// the mesh reads (the TK = 128 default map shared banks and hung: fzx):
//   bank 0-1  A row block, k-tile kt at rows kt * 1024     bank 2  C slot, rows 4096-6143
//   bank 3    B buffers ending at rows 7168 / 8192
static constexpr uint32_t ARES_C_ROW = 4096;
static inline void matmul_at_c(const Tiling &t, uint32_t a_row, uint32_t b_end, bool first, bool store_c) {
  const uint32_t skips = loop_matmul_skips(1, 1, 1, 0, store_c ? 0 : 1);
  gemmini_loop_ws_spad(t.i, t.j, t.k, 0, 0, 0, a_row, b_end, 0, ARES_C_ROW,
                       false, false, false, false, !first, NO_ACTIVATION, 0, 0, false, skips);
}
// C of output tile (mi, nj) from the C slot to DRAM, one move-out per C row (configure(.., true))
static inline void mvout_c_slot(const Problem &p, uint32_t mi, uint32_t nj) {
  const uint32_t rows_per_c_row = p.tn * 2u / DIM;
  for (uint32_t r = 0; r < p.tm; r++)
    gemmini_extended_mvout(host_addr(p.c + ((mi * p.tm + r) * p.n + nj * p.tn) * 2u),
                           ARES_C_ROW + r * rows_per_c_row, DIM, rows_per_c_row);
}
template <typename OnTileDone>
static void run_pipe_ares_cs(const Problem &p, const Tiling &t, OnTileDone on_tile_done) {
  const uint32_t k_tiles = p.k / p.tk, n_tiles = ncols_of(p) / p.tn;
  const uint32_t per_block = n_tiles * k_tiles, steps = (p.m / p.tm) * per_block;
  const uint32_t a_rows = p.tm * p.tk / DIM;              // SMEM rows of one A k-tile
  auto b_end = [](uint32_t s) { return (s & 1u) ? SMEM_ROWS : SMEM_ROWS - SMEM_ROWS / 8u; };
  auto b_addr = [&](uint32_t s) {
    return p.b + ((s / k_tiles) % n_tiles) * p.tn + (s % k_tiles) * p.tk * p.n;
  };
  auto load_a_block = [&](uint32_t mi) {
    for (uint32_t kt = 0; kt < k_tiles; kt++) mvin_a(t, p.a + mi * p.tm * p.k + kt * p.tk, kt * a_rows);
  };
  auto scales_of = [&](uint32_t s) {
    const uint32_t tile = s / k_tiles;
    load_tile_scales(p, tile / n_tiles, tile % n_tiles, s % k_tiles, s);
  };
  load_a_block(0);
  mvin_b(t, b_addr(0), b_end(0));
  scales_of(0);
  mu_fence_smem();
  drain();
  uint32_t out_tile = ~0u;   // the tile whose C sits in the slot: moved out during the next step
  auto c_out = [&]() {
    if (out_tile != ~0u) {
      mvout_c_slot(p, out_tile / n_tiles, out_tile % n_tiles);
      on_tile_done(out_tile);
      out_tile = ~0u;
    }
  };
  for (uint32_t s = 0; s < steps; s++) {
    const uint32_t kt = s % k_tiles, tile = s / k_tiles;
    const bool more = s + 1 < steps, new_block = more && (s + 1) % per_block == 0;
    select_scales(p, t, s);
    if (more && !new_block) mvin_b(t, b_addr(s + 1), b_end(s + 1));
    matmul_at_c(t, kt * a_rows, b_end(s), kt == 0, kt == k_tiles - 1);
    c_out();   // after this step's LOOP_WS commands (runningLoops overcount: see run_pipe)
    if (more) { scales_of(s + 1); mu_fence_smem(); }
    drain();
    if (kt == k_tiles - 1) out_tile = tile;
    if (new_block) {   // the block's last matmul has drained: replace A, then the next B
      load_a_block((s + 1) / per_block);
      mvin_b(t, b_addr(s + 1), b_end(s + 1));
      c_out();
      drain();
    }
  }
  c_out();
  drain();
}

}  // namespace mxg

#endif  // TAPEOUT_MXG_HPP
