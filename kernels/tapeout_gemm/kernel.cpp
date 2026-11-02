// kernel.cpp: GPU side of the tapeout MX-FP8 GEMM and its microbenchmarks (mxg.hpp does the work).
//
// Thread 0 drives the MX mesh through the MMIO command interface; every other thread goes
// straight to the tapeout epilogue.  -DTG_PIPE selects the pipelined loop (mxg::run_pipe),
// otherwise the reference loop (mxg::run_simple).

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <rad_tapeout.h>
#include <radiance.h>
#include <stdint.h>

#include "include/gemmini.h"
#include <mxgemmini_mmio.h>

#include "gemm.h"
#include "mxg.hpp"

#define TG_OCCUPANCY 1u   // only thread 0 works; one warp per core is enough
#ifndef TG_CLUSTERS
#define TG_CLUSTERS 1u    // 2: split M over both clusters of the two-cluster tapeout config
#endif

static inline uint32_t mcycle() {
  uint32_t c; asm volatile("csrr %0, mcycle" : "=r"(c)); return c;
}

static void run_bench(const TGArgs &a, const mxg::Problem &p, const mxg::Tiling &t) {
  switch (a.mode) {
    case TG_BENCH_MVIN:
      // Repetition r moves k-tile (r mod K/TK) of output tile (0,0): distinct DRAM lines until
      // the K extent wraps, so with K/TK >= reps every byte comes from DRAM, not a cache.
#ifdef TG_MVIN_CONTIG
      // TG_MVIN_CONTIG: the same bytes as tile-contiguous blocks (A tile r = TM x TK bytes at
      // a + r TM TK, row stride TK; B tile r = TK x TN at b + r TK TN, row stride TN): DRAM
      // locality against the row-major strided tiles above
      {
        mxg::Tiling tc = t;
        tc.lda = a.tk; tc.ldb = a.tn;
        gemmini_extended3_config_ld(tc.lda, MVIN_SCALE_IDENTITY, false, 0);
        gemmini_extended3_config_ld(tc.ldb, MVIN_SCALE_IDENTITY, false, 1);
        for (uint32_t r = 0; r < a.reps; r++) {
          mxg::mvin_tiles(tc, a.a + r * a.tm * a.tk, a.b + r * a.tk * a.tn, r);
          mxg::drain();
        }
      }
#else
      for (uint32_t r = 0; r < a.reps; r++) {
        const uint32_t kt = r % (a.k / a.tk);
        mxg::mvin_tiles(t, a.a + kt * a.tk, a.b + kt * a.tk * a.n, r);
        mxg::drain();
      }
#endif
      break;
    case TG_BENCH_COMPUTE:
      mxg::mvin_tiles(t, a.a, a.b, 0);
      mxg::load_tile_scales(p, 0, 0, 0, 0);
      mu_fence_smem();
      mxg::drain();
      mxg::select_scales(p, t, 0);
      for (uint32_t r = 0; r < a.reps; r++) mxg::matmul_tile(t, 0, r == 0);
      mxg::drain();
      break;
    case TG_BENCH_SCALES:
      for (uint32_t r = 0; r < a.reps; r++) mxg::load_tile_scales(p, 0, 0, r % (a.k / a.tk), r);
      mu_fence_smem();
      break;
    default:
      break;
  }
}

#ifdef MXG_PROF
namespace mxg {
static void mxg_prof_report(uint32_t issue, uint32_t scales, uint32_t drain_wait, uint32_t c_out) {
  rad_pb_put(TG_PB_PROF + 0u, issue);
  rad_pb_put(TG_PB_PROF + 1u, scales);
  rad_pb_put(TG_PB_PROF + 2u, drain_wait);
  rad_pb_put(TG_PB_PROF + 3u, c_out);
}
}  // namespace mxg
#endif
static void gemm_entry(void *raw_args, uint32_t tid, uint32_t nthreads, uint32_t) {
  // Clusters beyond TG_CLUSTERS (cluster 1 running a one-cluster image on the two-cluster system)
  // only take the epilogue.
  uint32_t my_cl;
  asm volatile("csrr %0, %1" : "=r"(my_cl) : "i"(MU_CSR_CLUSTER_ID));
  if (tid == 0 && my_cl < TG_CLUSTERS) {
    const volatile TGArgs *va = reinterpret_cast<const volatile TGArgs *>(raw_args);
    TGArgs a;
    a.a = va->a; a.b = va->b; a.c = va->c; a.sa = va->sa; a.sb = va->sb;
    a.m = va->m; a.n = va->n; a.k = va->k; a.tm = va->tm; a.tn = va->tn; a.tk = va->tk;
    a.mode = va->mode; a.reps = va->reps;
    uint32_t ncols = 0;
#if TG_CLUSTERS > 1 && !defined(TG_SPLIT_N)
    // Two-cluster system, row split: cluster c computes rows [c M/2, (c+1) M/2) (whole TM blocks;
    // A scales are tile-blocked [M/TM][K/32][TM], so a row block's scales are contiguous).
    {
      const uint32_t rows = a.m / TG_CLUSTERS;
      a.a += my_cl * rows * a.k;
      a.c += my_cl * rows * a.n * 2u;
      a.sa += my_cl * rows * (a.k / 32u);
      a.m = rows;
    }
#elif TG_CLUSTERS > 1
    // Column split (TG_SPLIT_N): cluster c computes columns [c N/2, (c+1) N/2): it reads half of
    // B (the larger operand here) instead of all of it.  B scales are tile-blocked
    // [N/TN][K/32][TN], so a column block's scales are contiguous; B and C keep row stride N.
    {
      const uint32_t cols = a.n / TG_CLUSTERS;
      a.b += my_cl * cols;
      a.c += my_cl * cols * 2u;
      a.sb += my_cl * cols * (a.k / 32u);
      ncols = cols;
    }
#endif
    mxg::Problem p = {a.a, a.b, a.c, a.sa, a.sb, a.m, a.n, a.k, a.tm, a.tn, a.tk, TG_SCRATCH};
    p.ncols = ncols;
    const mxg::Tiling t = mxg::tiling_of(p);
    rad_pb_put(TG_PB_T_START, mcycle());
#ifdef TG_ARES_CS
    mxg::configure(p, t, true);
#else
    mxg::configure(p, t);
#endif
    rad_pb_put(TG_PB_T_CFG, mcycle());
    if (a.mode == TG_GEMM) {
#if defined(TG_ARES_CS)
      mxg::run_pipe_ares_cs(p, t, [](uint32_t tile) { if (tile == 0) rad_pb_put(TG_PB_T_MARK, mcycle()); });
#elif defined(TG_ARES)
      mxg::run_pipe_ares(p, t, [](uint32_t tile) { if (tile == 0) rad_pb_put(TG_PB_T_MARK, mcycle()); });
#elif defined(TG_PIPE)
      mxg::run_pipe(p, t, [](uint32_t tile) { if (tile == 0) rad_pb_put(TG_PB_T_MARK, mcycle()); });
#else
      mxg::run_simple(p, t);
      rad_pb_put(TG_PB_T_MARK, mcycle());
#endif
    } else {
      run_bench(a, p, t);
    }
    rad_pb_put(TG_PB_T_END, mcycle());
  } else {
    asm volatile("nop");
  }
  rad_tapeout_epilogue(tid, nthreads / MU_NUM_THREADS);
}

int main() {
  rad_tapeout_begin();
  mu_schedule(gemm_entry, reinterpret_cast<void *>(TG_ARGS_ADDR), TG_OCCUPANCY);
  return 0;
}
