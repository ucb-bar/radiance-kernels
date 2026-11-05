// memlat: GPU load-to-use latency of DRAM, idle and under load (FireSim DRAM vs HBM comparisons).
//
// Per cluster, OCC = 8 warps (warp w on core w % 2):
//   warp 0 lane 0 (core 0)   the chaser: dependent loads along gen_data.py's chains, timed with mcycle
//   warp 1 lane 0 (core 1)   the DMA loader: loads-only LOOP_WS (B tiles, 32 KiB each) streaming region3
//   warps 3, 5, 7 (core 1)   the Muon loaders: every lane streams region2, 8 independent line loads per iteration
//   warps 2, 4, 6 (core 0)   idle (the chaser's core runs nothing else)
// Levels (in order): 0 idle, 1 Muon loaders, 2 DMA loader, 3 both.  Each level: the chaser turns its loaders on
// (SMEM flags), barrier, then WARM untimed + N timed loads on its seq chain, the same on its rand chain, flags off,
// the loaders drain, barrier.  Level 0 also times a re-chase of its seq chain (warm: caches) and SMALL loads over
// the first SMALL nodes, twice, the second timed (small: L0d / L1).
// Results go to printBuf slots 16.. (host.cpp prints them).  Both clusters run the same levels at the same time.
// (OCC = 8 is warps per CORE, so a cluster runs 16 warps: the Muon loaders above are warps 3, 5, .. 15.)
//
// Then the bandwidth phase, two windows of STREAM_CYC cycles each (no chaser), to find the most the memory takes:
//   warp 0 lane 0            the timer: turns the loaders on, waits STREAM_CYC cycles, turns them off
//   warp 1 lane 0            the DMA loader, up to 4 loops queued: window 0 B only (32 KiB per loop, as level 2),
//                            window 1 A and B (64 KiB per loop, both of gemmini's load streams)
//   warps 2 .. 15 (both cores, every lane)  Muon loaders: warp k of 14 reads line 14 i + k, 16 loads per iteration
// Per window: printBuf PB_STREAM + 4 w: timer cycles, Muon lines (all warps), DMA loops, DMA cycles (to its drain).
#include <nightly/device.h>
#include <nightly/mx.h>
#include "memlat_data.h"

constexpr uint32_t N = MEMLAT_N, WARM = MEMLAT_WARM, SMALL = MEMLAT_SMALL, LEVELS = MEMLAT_LEVELS;
// streamed buffers (never written; content irrelevant), 32 MiB per cluster
constexpr uint32_t MBUF = 0x40000000u, DBUF = 0x60000000u, BUF_SPAN = 32u << 20;
// DMA loop: I = 1, J = 4 (64 B rows), K = 32: 2048 spad rows (32 KiB) per loop, two spad buffers
constexpr uint32_t DJ = 4, DK = 32, DROW = DJ * 16, DBYTES = DK * 16 * DROW, DROWS = DBYTES / 16;
// SMEM control words (above the DMA buffers)
constexpr uint32_t FLAG_MUON = 0x1FF00u, FLAG_DMA = 0x1FF04u;
// printBuf result slots
constexpr uint32_t PB_RES = 16;   // + 5 * level: seq cycles, rand cycles, DMA loops, DMA cycles, Muon lines
constexpr uint32_t PB_WARM = 36, PB_SMALL = 37, PB_CHECK = 38, PB_LEVEL_CYC = 40;   // + level
constexpr uint32_t PB_STREAM = 52;   // + 4 * window (slot 48 is the heartbeat)
#ifndef MEMLAT_STREAM_CYC
#define MEMLAT_STREAM_CYC 2000000u
#endif
constexpr uint32_t STREAM_CYC = MEMLAT_STREAM_CYC, SWIN = 2, SUNROLL = 16;
constexpr uint32_t CNT_BASE = 0x1FE00u;   // SMEM: Muon lines per warp (bandwidth phase)
// window 1 DMA loop: I = 4, J = 4, K = 32: A 64 rows x 512 B and B 512 rows x 64 B (32 KiB each)
constexpr uint32_t AI = 4, ABYTES = AI * 16 * DK * 16, AROW = DK * 16;

static inline void sflag(uint32_t a, uint32_t v) { *(volatile __shared uint32_t *)a = v; }
static inline uint32_t gflag(uint32_t a) { return *(volatile __shared uint32_t *)a; }

// n dependent loads from p; returns the last pointer
static inline uint32_t chase(uint32_t p, uint32_t n) {
  #pragma unroll 1
  for (uint32_t i = 0; i < n; i += 8) {
    #pragma unroll
    for (uint32_t u = 0; u < 8; u++) {
      p = *(volatile uint32_t *)p;
      asm volatile("" : "+r"(p));
    }
  }
  return p;
}

// WARM untimed + N timed loads from head: cycles of the N; the final pointer goes into *x
static inline uint32_t timed_chase(uint32_t head, uint32_t &x) {
  uint32_t p = chase(head, WARM);
  const uint32_t t0 = nightly_mcycle();
  p = chase(p, N);
  asm volatile("" : "+r"(p));
  const uint32_t t1 = nightly_mcycle();
  x ^= p;
  return t1 - t0;
}

// one level of each role (noinline: one simple loop nest per function)
__attribute__((noinline)) static uint32_t chaser_level(uint32_t cl, uint32_t lv, uint32_t x) {
  const uint32_t lt0 = nightly_mcycle();
  const uint32_t seq = memlat_heads[cl][lv][0], rnd = memlat_heads[cl][lv][1];
  nightly_pb_put(PB_RES + 5 * lv + 0, timed_chase(seq, x));
  nightly_pb_put(PB_RES + 5 * lv + 1, timed_chase(rnd, x));
  if (lv == 0) {
    nightly_pb_put(PB_WARM, timed_chase(seq, x));
    x ^= chase(seq, SMALL);
    const uint32_t t0 = nightly_mcycle();
    uint32_t p = chase(seq, SMALL);
    asm volatile("" : "+r"(p));
    nightly_pb_put(PB_SMALL, nightly_mcycle() - t0);
    x ^= p;
  }
  sflag(FLAG_MUON, 0);
  sflag(FLAG_DMA, 0);
  mu_fence_smem();
  nightly_pb_put(PB_LEVEL_CYC + lv, nightly_mcycle() - lt0);
  nightly_heartbeat(lv + 1);
  return x;
}

// streams 32 KiB loads-only loops from dbase + *doff until FLAG_DMA clears; returns the loops issued
__attribute__((noinline)) static uint32_t dma_level(uint32_t dbase, uint32_t *doff, uint32_t *dbuf) {
  uint32_t loops = 0;
  #pragma unroll 1
  while (gflag(FLAG_DMA)) {
    if (mx::occupancy() < 2) {
      mx::set_ab_dram(0, mx::host_addr(dbase + *doff), DROW, DROW);
      mx::loop_ws_spad(1, DJ, DK, 0, (*dbuf + 1) * DROWS, 0, false, false, {true, false, true, true, true});
      *dbuf ^= 1;
      *doff = (*doff + DBYTES) % BUF_SPAN;
      loops++;
    }
  }
  mx::fence();
  return loops;
}

// the 3 Muon loader warps take interleaved lines (warp k of 3 reads line 3 i + k) until FLAG_MUON clears; returns
// this warp's lines.  *moff advances by the lines all 3 warps read.
__attribute__((noinline)) static uint32_t muon_level(uint32_t mbase, uint32_t k, uint32_t lane, uint32_t *moff,
                                                     uint32_t *sink) {
  uint32_t lines = 0, s = *sink, off = *moff;
  #pragma unroll 1
  while (gflag(FLAG_MUON)) {
    #pragma unroll
    for (uint32_t u = 0; u < 8; u++)
      s += *(volatile uint32_t *)(mbase + ((off + 3 * u + k) * 64u) % BUF_SPAN + lane * 4u);
    off += 24;
    lines += 8;
  }
  *sink = s;
  *moff = off;
  return lines;
}

// bandwidth phase: the DMA loader, loops kept queued (occupancy < 4; the spad content is never read)
__attribute__((noinline)) static uint32_t dma_stream(uint32_t dbase, uint32_t *doff, bool ab, uint32_t *cyc) {
  const uint32_t t0 = nightly_mcycle();
  uint32_t loops = 0;
  #pragma unroll 1
  while (gflag(FLAG_DMA)) {
    if (mx::occupancy() < 4) {
      if (ab) {
        mx::set_ab_dram(mx::host_addr(dbase + *doff), mx::host_addr(dbase + (*doff + ABYTES) % BUF_SPAN), AROW, DROW);
        mx::loop_ws_spad(AI, DJ, DK, DROWS, DROWS, 0, false, false, {false, false, true, true, true});
        *doff = (*doff + ABYTES + DBYTES) % BUF_SPAN;
      } else {
        mx::set_ab_dram(0, mx::host_addr(dbase + *doff), DROW, DROW);
        mx::loop_ws_spad(1, DJ, DK, 0, DROWS, 0, false, false, {true, false, true, true, true});
        *doff = (*doff + DBYTES) % BUF_SPAN;
      }
      loops++;
    }
  }
  mx::fence();
  *cyc = nightly_mcycle() - t0;
  return loops;
}

// bandwidth phase: Muon loader warp k of nw, every lane one word of each line (one 64 B line per load)
__attribute__((noinline)) static uint32_t muon_stream(uint32_t mbase, uint32_t k, uint32_t nw, uint32_t lane,
                                                      uint32_t *moff, uint32_t *sink) {
  uint32_t lines = 0, s = *sink, off = *moff;
  #pragma unroll 1
  while (gflag(FLAG_MUON)) {
    #pragma unroll
    for (uint32_t u = 0; u < SUNROLL; u++)
      s += *(volatile uint32_t *)(mbase + ((off + nw * u + k) * 64u) % (BUF_SPAN / 2) + lane * 4u);   // upper 16 MiB
    off += nw * SUNROLL;
    lines += SUNROLL;
  }
  *sink = s;
  *moff = off;
  return lines;
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  const uint32_t cl = nightly_cluster_id();
  // opaque role flags: compared on tid directly, clang folds the role tests into a switch, which the Muon
  // backend's UnifyLoopExits pass rejects ("Unsupported block terminator")
  uint32_t r_chaser = tid == 0, r_dma = tid == MU_NUM_THREADS, r_muon = warp >= 3 && (warp & 1);
  asm volatile("" : "+r"(r_chaser), "+r"(r_dma), "+r"(r_muon));
  const bool chaser = r_chaser, dma = r_dma, muon = r_muon;
  const uint32_t mbase = MBUF + cl * BUF_SPAN, dbase = DBUF + cl * BUF_SPAN;
  uint32_t x = 0, moff = 0, doff = 0, dbuf = 0, sink = 0;

  if (chaser) { sflag(FLAG_MUON, 0); sflag(FLAG_DMA, 0); } else asm volatile("nop");
  if (dma) {
    mx::flush_tlb();
    mx::lut_disable();
    mx::config_ex(mx::FP8, mx::FP8, mx::BF16);
    mx::config_ld(DROW, 1);
  } else asm volatile("nop");
  mu_barrier(1, warps);

  #pragma unroll 1
  for (uint32_t lv = 0; lv < LEVELS; lv++) {
    if (chaser) {
      sflag(FLAG_MUON, lv & 1);
      sflag(FLAG_DMA, lv >> 1);
      mu_fence_smem();
    } else asm volatile("nop");
    mu_barrier(1, warps);
    const uint32_t lt0 = nightly_mcycle();
    if (chaser) {
      x = chaser_level(cl, lv, x);
    } else if (dma) {
      const uint32_t loops = (lv >> 1) ? dma_level(dbase, &doff, &dbuf) : 0;
      nightly_pb_put(PB_RES + 5 * lv + 2, loops);
      nightly_pb_put(PB_RES + 5 * lv + 3, nightly_mcycle() - lt0);
    } else if (muon) {
      const uint32_t k = (warp - 3) / 2;
      const uint32_t lines = (lv & 1) ? muon_level(mbase, k, lane, &moff, &sink) : 0;
      if (lane == 0 && k == 0) nightly_pb_put(PB_RES + 5 * lv + 4, lines * 3);
      else asm volatile("nop");
    } else asm volatile("nop");
    mu_barrier(1, warps);
  }

  // ---- bandwidth phase ----
  uint32_t r_sload = warp >= 2;
  asm volatile("" : "+r"(r_sload));
  const bool sload = r_sload;
  const uint32_t nws = warps - 2, sk = warp - 2;
  uint32_t smoff = 0;
  #pragma unroll 1
  for (uint32_t w = 0; w < SWIN; w++) {
    if (chaser) {
      sflag(FLAG_MUON, 1);
      sflag(FLAG_DMA, 1);
      mu_fence_smem();
    } else asm volatile("nop");
    mu_barrier(1, warps);
    if (chaser) {
      const uint32_t t0 = nightly_mcycle();
      while (nightly_mcycle() - t0 < STREAM_CYC) asm volatile("nop");
      sflag(FLAG_MUON, 0);
      sflag(FLAG_DMA, 0);
      mu_fence_smem();
      nightly_pb_put(PB_STREAM + 4 * w + 0, nightly_mcycle() - t0);
    } else if (dma) {
      uint32_t cyc = 0;
      const uint32_t loops = dma_stream(dbase, &doff, w == 1, &cyc);
      nightly_pb_put(PB_STREAM + 4 * w + 2, loops);
      nightly_pb_put(PB_STREAM + 4 * w + 3, cyc);
    } else if (sload) {
      const uint32_t lines = muon_stream(mbase + (16u << 20), sk, nws, lane, &smoff, &sink);
      if (lane == 0) sflag(CNT_BASE + 4 * warp, lines);
      else asm volatile("nop");
      mu_fence_smem();
    } else asm volatile("nop");
    mu_barrier(1, warps);
    if (chaser) {
      uint32_t tot = 0;
      for (uint32_t i = 2; i < warps; i++) tot += gflag(CNT_BASE + 4 * i);
      nightly_pb_put(PB_STREAM + 4 * w + 1, tot);
    } else asm volatile("nop");
    mu_barrier(1, warps);
  }

  if (chaser) nightly_pb_put(PB_CHECK, x);
  else asm volatile("nop");
  if (muon && sink == 0x12345678u) nightly_pb_put(PB_CHECK + 1, sink);   // keep the loads
  else asm volatile("nop");
  nightly_kernel_end(tid, tpb);
}

int main() {
  return nightly_main(entry, nullptr);
}
