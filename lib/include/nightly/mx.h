/*
 * nightly/mx.h: MxGemmini command interface for Muon kernels on the nightly RTL
 * (see lib/include/nightly/README.md for the RTL versions).
 *
 * Self-contained: no gemmini.h / gemmini_params.h (the rocc-tests copy of gemmini_params.h is
 * for the standalone 256 KiB rocket config and has the wrong BANK_ROWS/ACC_ROWS for Radiance).
 * Geometry below is the Radiance-generated gemmini_params.h (DIM 16, 4 banks x 2048 rows,
 * 256 accumulator rows).
 *
 * Every command is 5 sw.shared to the cluster's MMIO block at 0x84000 (RS1 lo/hi, RS2 lo/hi,
 * INST).  RS1/RS2 are single registers per cluster, so exactly ONE thread per cluster may issue
 * commands.  The INST store stalls until gemmini's command queue accepts it.
 *
 * Ordering rules (static reading of the RTL, see the nightly kernel skill for the evidence):
 *  - The reservation station orders mvin/compute/mvout by spad/accumulator address.
 *  - Any non-loop command (CONFIG_*, MX_LOAD_SCALES, mvin, mvout) waits until earlier LOOP_WS
 *    are fully UNROLLED, not executed.  CONFIG_SCALE_MEM additionally waits for the matmul in
 *    progress, so it is ordered with computes.
 *  - MX_LOAD_SCALES and GPU writes to the scale-memory window bypass the reservation station:
 *    they are NOT ordered against computes that read the same scale buffer.
 *  - BUSY (mx_fence) covers the reservation station, loop unrollers, spad reads/writes and the
 *    scale/LUT loaders.  It does not cover the requantizer pipeline or its scale flushes.
 */
#ifndef NIGHTLY_MX_H
#define NIGHTLY_MX_H

#include <stdint.h>
#include <mu_intrinsics.h>

namespace mx {

constexpr uint32_t DIM = 16;
constexpr uint32_t SPAD_ROWS = 8192;          /* 128 KiB SMEM = 8192 rows x 16 B */
constexpr uint32_t ACC_ROWS = 256;            /* 32 KiB accumulator, 64-bit (4-lane) entries */
constexpr uint32_t ACC_HALF = ACC_ROWS / 2;   /* inc_acc_addr alternates halves */

/* cluster-local MMIO */
constexpr uint32_t CTRL = 0x84000;
constexpr uint32_t INST = CTRL + 0x00, READY = CTRL + 0x08, RS1 = CTRL + 0x10, RS2 = CTRL + 0x18;
constexpr uint32_t BUSY = CTRL + 0x20, OCCUPANCY = CTRL + 0x28;
constexpr uint32_t SF_B = 0x88000;            /* weight (B) scale window, 8 KiB */
constexpr uint32_t SF_A = 0x8A000;            /* activation (A) scale window, 8 KiB */
constexpr uint32_t SF_BUF1 = 0x1000;          /* buffer 1 offset inside a window (hardware) */
constexpr uint32_t REQUANT = 0x40000;         /* GPU requantizer input window, 256 KiB */

enum Funct : uint32_t {
  CONFIG = 0, MVIN2 = 1, MVIN = 2, MVOUT = 3, FLUSH = 7,
  LOOP_WS = 8, LOOP_WS_CONFIG_BOUNDS = 9, LOOP_WS_CONFIG_ADDRS_AB = 10, LOOP_WS_CONFIG_ADDRS_DC = 11,
  LOOP_WS_CONFIG_STRIDES_AB = 12, LOOP_WS_CONFIG_STRIDES_DC = 13,
  MVOUT_SPAD = 23, LOOP_WS_CONFIG_SPAD_AB = 24, CONFIG_SCALE_MEM = 26,
  MX_LOAD_SCALES = 27, MX_LOAD_LUT = 29, MX_LUT_DISABLE = 30,
  LOOP_WS_CONFIG_SCALES = 31, LOOP_WS_CONFIG_SCALE_STRIDES = 32,
};
enum Fmt : uint32_t { FP8 = 0, FP6 = 1, FP4 = 2, BF16 = 3 };

/* GPU-local DRAM address -> the host-physical address gemmini's DMA uses */
static inline uint64_t host_addr(const void *p) {
  return 0x100000000ull | (uint64_t)(uint32_t)(uintptr_t)p;
}
static inline uint64_t host_addr(uint32_t a) { return 0x100000000ull | (uint64_t)a; }

static inline void st32(uint32_t addr, uint32_t v) {
  asm volatile("sw.shared %0, 0(%1)" :: "r"(v), "r"(addr) : "memory");
}
static inline uint32_t ld32(uint32_t addr) {
  uint32_t v;
  asm volatile("lw.shared %0, 0(%1)" : "=r"(v) : "r"(addr) : "memory");
  return v;
}

/* NIGHTLY_MX_WAIT_READY (opt-in): never let the INST store block.  A store to INST stalls while gemmini's command
 * queue is full, and a stalled store holds the issuing core's memory pipeline, slowing every other warp on that core
 * (fa_mxfp8_sr: the softmax warps sharing the producer's core ran ~2.2x slower).  So poll READY (a plain load) first,
 * with a short non-issuing stall (dependent divides) between polls. */
#ifdef NIGHTLY_MX_WAIT_READY
__attribute__((noinline)) static void wait_ready() {   /* out of line: one copy, not one per command site */
  while (ld32(READY) == 0) {
    uint32_t x = 0xFFFFFFFFu;
    asm volatile("divu %0, %0, %1\n\tdivu %0, %0, %1" : "+r"(x) : "r"(3u));
  }
}
#endif
static inline void cmd(uint32_t funct, uint64_t rs1, uint64_t rs2) {
  st32(RS1, (uint32_t)rs1); st32(RS1 + 4, (uint32_t)(rs1 >> 32));
  st32(RS2, (uint32_t)rs2); st32(RS2 + 4, (uint32_t)(rs2 >> 32));
#ifdef NIGHTLY_MX_WAIT_READY
  wait_ready();
#endif
  st32(INST, 0x7Bu | (3u << 12) | (1u << 15) | (2u << 20) | (funct << 25));
}

/* wait until gemmini is idle (all issued commands executed, loaders done) */
static inline void fence() { while (ld32(BUSY) != 0) asm volatile("nop"); }

#ifdef NIGHTLY_MX_RETIRE
/* NIGHTLY_MX_RETIRE (opt-in; needs a gemmini built with has_loop_retire_counter): MMIO 0x38 counts LOOP_WS that have
 * fully retired, in issue order (all computes done, stores landed), so "retired >= n" means the n-th LOOP_WS this
 * thread issued, and everything issued before it, is done.  Waiting for one loop this way leaves the loops queued
 * behind it running, where fence() drains everything.  loops_issued counts this thread's LOOP_WS (the issuing thread
 * only); the n of a loop is loops_issued right after issuing it. */
constexpr uint32_t RETIRED = CTRL + 0x38;
inline uint32_t loops_issued = 0;
__attribute__((noinline)) static void wait_retired(uint32_t n) {
  while ((int32_t)(ld32(RETIRED) - n) < 0) {
    uint32_t x = 0xFFFFFFFFu;
    asm volatile("divu %0, %0, %1\n\tdivu %0, %0, %1" : "+r"(x) : "r"(3u));
  }
}
#endif
/* number of LOOP_WS accepted but not yet fully unrolled */
static inline uint32_t occupancy() { return ld32(OCCUPANCY); }

/* ---- configuration ------------------------------------------------------------------------ */
/* CONFIG_EX: weight-stationary, identity scale, MX formats for A (act), B (weight), C (out). */
static inline void config_ex(Fmt act, Fmt wgt, Fmt out, uint32_t a_stride = 1, uint32_t c_stride = 1) {
  const uint64_t rs1 = ((uint64_t)a_stride << 16) | ((uint64_t)out << 14) | ((uint64_t)wgt << 12) |
                       ((uint64_t)act << 10) | (1ull << 2) /* WS */ | 0 /* CONFIG_EX */;
  cmd(CONFIG, rs1, (uint64_t)c_stride << 48);
}
/* CONFIG_LD for mvin queue `id` (0 = A, 1 = B, 2 = D): DRAM row stride in bytes. */
static inline void config_ld(uint64_t stride_bytes, uint32_t id) {
  const uint64_t rs1 = ((uint64_t)DIM << 16) | (1ull << 8) | ((uint64_t)id << 3) | 1 /* CONFIG_LD */;
  cmd(CONFIG, rs1, stride_bytes);
}
static inline void config_st(uint64_t stride_bytes) { cmd(CONFIG, 2 /* CONFIG_ST */, stride_bytes); }

/* Accumulator -> DRAM store (STORE_CMD): `rows` accumulator rows from acc_row, each written to
 * dram_host + r * CONFIG_ST stride; `chunk` selects the 32-bf16 half of a 64-bf16 accumulator row
 * (fp8 outputs).  For quad (fp4) outputs each accumulator row holds two output rows; the second goes
 * to +64 * store_j bytes (nightly G6: store_j travels in rs1[63:56], 0 = the global CONFIG_SCALE_MEM J). */
static inline void mvout_acc(uint64_t dram_host, uint32_t acc_row, uint32_t rows, uint32_t cols,
                             uint32_t chunk, uint32_t store_j = 0) {
  cmd(MVOUT, ((uint64_t)store_j << 56) | dram_host,
      ((uint64_t)chunk << 54) | ((uint64_t)rows << 48) | ((uint64_t)cols << 32) | 0x80000000ull | acc_row);
}
static inline void flush_tlb() { cmd(FLUSH, 0, 0); }
static inline void lut_disable() { cmd(MX_LUT_DISABLE, 0, 0); }

/* CONFIG_SCALE_MEM: loop bounds (tiles of 16) used for the scale-row index and the requantizer's
 * coalescer, read-buffer selects, scale-flush DRAM address (host-physical), counter reset, and
 * `resident` (requantizer writes act-block-scales into the A window buffer 0).
 * rs2[16] (`wait_loads`): the config, and so every compute behind it in the execute queue, waits
 * until the selected act and weight scale buffers have no MX_LOAD_SCALES pending.  gemmini 0901baa
 * queues scale loads (4 deep), so a zero-length load no longer holds the command stream until the
 * scales have landed: without this bit the first computes read scales that are still loading. */
static inline void config_scale(uint32_t I, uint32_t J, uint32_t K, uint32_t act_sel, uint32_t w_sel,
                                uint64_t flush_host_addr, bool resident = false, bool reset = false,
                                uint32_t lut_granularity = 1, bool wait_loads = true) {
  const uint64_t rs1 = ((uint64_t)resident << 63) | ((uint64_t)reset << 62) | ((uint64_t)w_sel << 61) |
                       ((uint64_t)act_sel << 60) | ((uint64_t)K << 51) | ((uint64_t)J << 42) |
                       ((uint64_t)I << 33) | (flush_host_addr & 0x1FFFFFFFFull);
  cmd(CONFIG_SCALE_MEM, rs1, ((uint64_t)wait_loads << 16) | (lut_granularity & 0xFFFF));
}

/* MX_LOAD_SCALES (DMA): `len` bytes from DRAM `host_src` into the A (act) or B (weight) scale
 * memory at byte offset `dst_off` (0 = buffer 0, SF_BUF1 = buffer 1).  8 B aligned source,
 * lengths and offsets multiples of 16.  1-D form of the gemmini-mx-cleanup encoding:
 * rs1[39:0] address, rs1[63:40] DRAM row pitch (0); rs2[31:0] bytes, [32] sel, [45:33] destination
 * byte offset, [53:46] rows (0 = 1-D), [54] loop-managed (0). */
static inline void load_scales(uint64_t host_src, uint32_t len, bool weight, uint32_t dst_off) {
  cmd(MX_LOAD_SCALES, host_src & 0xFFFFFFFFFFull,
      ((uint64_t)(dst_off & 0x1FFF) << 33) | ((uint64_t)weight << 32) | len);
}

/* ---- loop_ws ------------------------------------------------------------------------------ */
struct Skips { bool lda, ldb, ldd, ex, stc; };
static inline uint64_t skip_bits(Skips s) {
  return ((uint64_t)s.lda << 3) | ((uint64_t)s.ldb << 4) | ((uint64_t)s.ldd << 5) |
         ((uint64_t)s.ex << 6) | ((uint64_t)s.stc << 7);
}

/* One loop_ws in scratchpad mode (spad_only = 1):
 *   I, J, K     : tiles of 16 (FP8) along M, N, K
 *   a_start     : spad row of A tile (0,0); A tile (i,k) at a_start + (i*K + k)*16
 *   b_end       : spad row one past B; B tile (k,j) at b_end - K*J*16 + (k*J + j)*16
 *   c_spad      : spad row of the row-major C (bf16 pitch 2N B, fp8 pitch N B) when stc runs
 *   accumulate  : add into the accumulator instead of overwriting
 *   inc_acc     : toggle the accumulator base (0 <-> 128) after this loop's execute / store
 * DRAM sources for loads (lda/ldb not skipped) come from set_ab_dram() before the loop. */
static inline void loop_ws_spad(uint32_t I, uint32_t J, uint32_t K, uint32_t a_start, uint32_t b_end,
                                uint32_t c_spad, bool accumulate, bool inc_acc, Skips s) {
  cmd(LOOP_WS_CONFIG_BOUNDS, 0, ((uint64_t)K << 32) | ((uint64_t)J << 16) | I);
  cmd(LOOP_WS_CONFIG_SPAD_AB, a_start, b_end);
  const uint64_t rs2 = ((uint64_t)c_spad << 32) | (1ull << 9) | ((uint64_t)inc_acc << 8) | skip_bits(s);
  cmd(LOOP_WS, (uint64_t)accumulate, rs2);
#ifdef NIGHTLY_MX_RETIRE
  loops_issued++;
#endif
}
/* Loop-managed scales (gemmini 0901baa): the next LOOP_WS loads its own A/B scale slices -- A: K/32
 * rows of I*16 bytes at `a_pitch`, B: K/32 rows of J*16 bytes at `b_pitch` -- into the scale half of its
 * loop slot, and opens its compute stream with a CONFIG_SCALE_MEM that waits for them.  No
 * MX_LOAD_SCALES / config_scale around it.  a_host == 0 -> legacy (unmanaged) loop.  Consecutive
 * managed loops must alternate loop slots (a half is freed only when a config selects the other). */
static inline void loop_scales(uint64_t a_host, uint64_t b_host, uint64_t a_pitch, uint64_t b_pitch) {
  cmd(LOOP_WS_CONFIG_SCALES, a_host, b_host);
  cmd(LOOP_WS_CONFIG_SCALE_STRIDES, a_pitch, b_pitch);
}
/* DRAM addresses (host-physical) and row strides (bytes) for a loop's A/B loads */
static inline void set_ab_dram(uint64_t a_host, uint64_t b_host, uint64_t a_stride, uint64_t b_stride) {
  cmd(LOOP_WS_CONFIG_ADDRS_AB, a_host, b_host);
  cmd(LOOP_WS_CONFIG_STRIDES_AB, a_stride, b_stride);
}

/* ---- scale-memory window (GPU word writes; unordered against computes) ------------------- */
static inline void sf_write_words(uint32_t window_base, uint32_t off, const uint32_t *src, uint32_t bytes) {
  for (uint32_t i = 0; i < bytes / 4; i++) st32(window_base + off + 4 * i, src[i]);
}

} // namespace mx

#endif
