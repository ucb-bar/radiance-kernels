// FUSED QKV projection (MX-Gemmini fp4, K=2048) + RoPE (Muon SIMT), head_dim=64.
//   out[M,N] (bf16) = RoPE( bf16( X[M,K] @ W[K,N] ), cos, sin )
//
// Phase 1 (MX): C = X . W  (fp4 e2m1, per-32-K e8m0 scales) -> C[M][N] bf16 in SMEM at
//               SPAD_DEST()*DIM.  mxgemm_single_output_tile() runs config + the full K=2048
//               loop and leaves C in the scratchpad WITHOUT moving it out to DRAM.
// Phase 2 (SIMT): a SEPARATE 4-warp schedule applies RoPE to the SMEM-resident C and writes the
//               rotated bf16 STRAIGHT to GMEM. The Q/K projection never round-trips through DRAM
//               -- that saved traffic is the whole-Radiance fusion win.
//
// head_dim = N = 64  ->  HALF = 32 within the single head.
// HF Llama rotate_half, cos/sin duplicated across halves (cos[i]==cos[i+HALF]):
//   out[m][i]      = C[m][i]*cos - C[m][i+HALF]*sin
//   out[m][i+HALF] = C[m][i+HALF]*cos + C[m][i]*sin
// Paired form (one thread per (m,i) pair) reads both originals and writes both outputs -> no
// cross-thread read-after-write hazard.

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"

#ifndef MX_NUM_WARPS
#define MX_NUM_WARPS 2   // mxgemm_lib is warp-specialized for 2 warps
#endif
extern "C" uint32_t __mu_num_warps = MX_NUM_WARPS;

#include "data"

// fp4 path: mxgemm_lib references these LUTs; unused for fp4, provide zeros.
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};

#include "mxgemm_lib_fused.hpp"

// Single output tile, full bf16 output (RoPE needs the real float values, so QUANT stays false).
constexpr GemmConfig GEMM_CFG{
    .TILE_M = MATMUL_M,
    .TILE_N = MATMUL_N,
    .TILE_K = 64,
    .DATATYPE = GemmDatatype::FP4,
    .QUANT_OUTPUT = false,
};

struct KernelArgs {
  uint8_t *C;                 // bf16 output [M][N]
  const float *cosc, *sinc;   // RoPE caches [M][N]
  uint32_t M, N, K;
};

static inline uint32_t f2b(float f){ union{float f; uint32_t u;} v; v.f=f; return v.u; }
static inline float    b2f(uint32_t u){ union{float f; uint32_t u;} v; v.u=u; return v.f; }
static inline float    bf16_to_f(uint16_t h){ return b2f((uint32_t)h << 16); }
static inline uint16_t f_to_bf16(float f){ return (uint16_t)(f2b(f) >> 16); }  // truncate

static KernelArgs kernel_args;

// --- Phase 1: MX matmul X.W -> C(bf16) in SMEM (no move-out) ---
void kernel_body(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  auto* a = reinterpret_cast<KernelArgs*>(raw);
  const uint32_t nw = tpb / MU_NUM_THREADS;
  mxgemm_single_output_tile<GEMM_CFG>(a->M, a->N, a->K, tid, tpb);
  mu_barrier(1, nw);
}

// --- Phase 2: SIMT RoPE on SMEM C -> GMEM out (fused move-out), separate 4-warp schedule ---
void rope_body(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  auto* a = reinterpret_cast<KernelArgs*>(raw);
  const uint32_t N = MATMUL_N, HALF = N / 2;
  const uint32_t Cbase = GEMM_CFG.SPAD_DEST() * DIM;    // byte addr of C in SMEM (bf16)
  auto* out = reinterpret_cast<uint16_t*>(a->C);        // bf16 GMEM output [M][N]
  const uint32_t pairs = MATMUL_M * HALF;
  #pragma clang loop unroll(disable)
  for (uint32_t p = tid; p < pairs; p += tpb) {
    const uint32_t m = p / HALF, i = p % HALF;
    const uint32_t o0 = m * N + i, o1 = m * N + i + HALF;
    const float c0 = bf16_to_f(load16_shared(Cbase + o0 * 2));
    const float c1 = bf16_to_f(load16_shared(Cbase + o1 * 2));
    const float cs = a->cosc[o0];     // cos/sin duplicated: cos[i]==cos[i+HALF]
    const float sn = a->sinc[o0];
    out[o0] = f_to_bf16(c0 * cs - c1 * sn);
    out[o1] = f_to_bf16(c1 * cs + c0 * sn);
  }
}

#define VERIFY_LANES (MU_NUM_THREADS * MU_NUM_CORES)
__global uint32_t lane_errors[VERIFY_LANES] = {0};

static void verify_body(void *, uint32_t tid, uint32_t tpb, uint32_t) {
  uint32_t errors = 0;
  for (uint32_t i = tid; i < VERIFY_COUNT; i += tpb) {
    if (C_raw[i] != gold_raw[i]) errors++;
  }
  lane_errors[tid] = errors;
  mu_fence();
}


int main() {
  kernel_args = {reinterpret_cast<uint8_t *>(reinterpret_cast<uint32_t>((uint16_t*)C_raw)),
                 cos_raw, sin_raw, MATMUL_M, MATMUL_N, MATMUL_K};
  mu_schedule(kernel_body, &kernel_args, MX_NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  mu_schedule(rope_body, &kernel_args, 4);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
#ifndef DRAIN_ITERS
#define DRAIN_ITERS 0u
#endif
  for (volatile uint32_t d = 0; d < DRAIN_ITERS; d++) { asm volatile("" ::: "memory"); }

  mu_schedule(verify_body, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES);
  if (mu_hart_id() != 0) { for (;;) {} }
  mu_fence();
  uint32_t total = 0;
  for (uint32_t t = 0; t < VERIFY_LANES; t++) total += lane_errors[t];
  uint32_t code = total ? ((total << 1) | 1u) : 0u;
  mu_tohost(code);
  return 0;
}
