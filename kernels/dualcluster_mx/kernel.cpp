// Dual-cluster data-parallel MX-Gemmini GEMM (fp8), for measuring the 2-cluster
// performance lever on RadianceTapeoutSimConfig (2 MX-Gemminis + 4 Muon cores).
//
// The Muon scheduler maps 1 threadblock -> 1 cluster (mu_schedule.cpp), so
// threadblock_id == cluster index. Each cluster's cores issue Gemmini ops to their
// LOCAL MX-Gemmini. We give each cluster a full 128x128x2048 fp8 output tile.
//
// DC_MODE selects the experiment:
//   0 = SINGLE          : only cluster 0 works (cluster 1 idle) -> single-cluster baseline
//   1 = DUAL_SHARED     : both clusters compute, SAME A/B (L2-cached), distinct C
//                         -> compute-concurrency ceiling, low extra DRAM traffic
//   2 = DUAL_DISTINCT   : both clusters compute fully independent A/B/C
//                         -> genuine 2x DRAM read traffic (weight-streaming / data-parallel)
//
// Cycle measurement is config-independent: each cluster's lead thread reads the mcycle
// CSR (0xB00) at kernel start/end; main() computes the wall span across active clusters
// and emits it via ECALL (sim prints "tohost=<wall_cycles>"). No trace-db needed.

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#ifndef DC_MODE
#define DC_MODE 0
#endif

#define MX_NUM_WARPS 2
extern "C" uint32_t __mu_num_warps = MX_NUM_WARPS;

#define MATMUL_M 128
#define MATMUL_N 128
#define MATMUL_K 2048
#define TILE_K_SZ 256

// Per-cluster tensor byte-sizes.
#define A_BYTES (MATMUL_M * MATMUL_K)            // 262144
#define B_BYTES (MATMUL_K * MATMUL_N)            // 262144
#define C_ELEMS (MATMUL_M * MATMUL_N)            // 16384 (bf16 = uint16)
#define SF_A_BYTES (MATMUL_M * MATMUL_K / 32)    // 8192
#define SF_B_BYTES (MATMUL_N * MATMUL_K / 32)    // 8192

// Two per-cluster copies so DUAL_DISTINCT touches genuinely disjoint DRAM.
static uint8_t A_in_arr[2 * A_BYTES] = {0};
static uint8_t B_in_arr[2 * B_BYTES] = {0};
static const uint8_t *A_in = &A_in_arr[0];
static const uint8_t *B_in = &B_in_arr[0];
static uint8_t A_scales_row[MATMUL_M][MATMUL_K / 32] = {0};
static uint8_t B_scales_col[MATMUL_N][MATMUL_K / 32] = {0};
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};
static uint16_t C_out[2 * C_ELEMS] = {0};

#include "mxgemm_lib_dc.hpp"

constexpr GemmConfig CFG{
    .TILE_M = MATMUL_M,
    .TILE_N = MATMUL_N,
    .TILE_K = TILE_K_SZ,
    .DATATYPE = GemmDatatype::FP8,
    .QUANT_OUTPUT = false,
};

// g_cyc[cluster*2 + 0] = start mcycle, [+1] = end mcycle.
__global uint32_t g_cyc[4] = {0};

static inline uint32_t rd_mcycle() {
  uint32_t c;
  asm volatile("csrr %0, 0xB00" : "=r"(c)::"memory");
  return c;
}

static inline uint32_t hart_id() {
  uint32_t id;
  asm volatile("csrr %0, mhartid" : "=r"(id)::"memory");
  return id;
}

static void kernel_body(void *, uint32_t tid_in_threadblock,
                        uint32_t threads_per_threadblock,
                        uint32_t threadblock_id) {
  const uint32_t cluster = threadblock_id;

#if DC_MODE == 0
  if (cluster != 0) return;   // single-cluster baseline: cluster 1 idle
  const uint32_t a_off = 0, b_off = 0, c_off = 0;
#elif DC_MODE == 1
  // DUAL_SHARED: same A/B for both clusters, distinct C.
  const uint32_t a_off = 0, b_off = 0;
  const uint32_t c_off = cluster * C_ELEMS;   // element offset into C_out
#else
  // DUAL_DISTINCT: fully independent A/B/C per cluster.
  const uint32_t a_off = cluster * A_BYTES;
  const uint32_t b_off = cluster * B_BYTES;
  const uint32_t c_off = cluster * C_ELEMS;
#endif

  const bool lead = (tid_in_threadblock == 0);
  if (lead) g_cyc[cluster * 2 + 0] = rd_mcycle();

  uint8_t *C_gmem =
      reinterpret_cast<uint8_t *>(reinterpret_cast<uint32_t>(&C_out[c_off]));
  mxgemm<CFG>(MATMUL_M, MATMUL_N, MATMUL_K, C_gmem, tid_in_threadblock,
              threads_per_threadblock, threadblock_id, a_off, b_off);

  if (lead) g_cyc[cluster * 2 + 1] = rd_mcycle();
  mu_fence();
}

int main() {
  mu_schedule(kernel_body, nullptr, MX_NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);

  if (hart_id() != 0) {
    for (;;) {}
  }
  mu_fence();

  // Wall span across ACTIVE clusters (end != 0), in the shared mcycle timebase.
  uint32_t mn_start = 0xFFFFFFFFu, mx_end = 0;
  for (uint32_t cl = 0; cl < 2; cl++) {
    uint32_t s = g_cyc[cl * 2 + 0], e = g_cyc[cl * 2 + 1];
    if (e == 0) continue; // cluster idle (SINGLE mode)
    if (s < mn_start) mn_start = s;
    if (e > mx_end) mx_end = e;
  }
  uint32_t wall = (mx_end >= mn_start) ? (mx_end - mn_start) : 0;

  // Emit wall-cycles via ECALL; sim prints "tohost=<wall>".
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(wall) : "memory");
  return 0;
}
