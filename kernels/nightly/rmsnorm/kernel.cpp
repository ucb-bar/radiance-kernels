// RMSNorm, bf16 in/out, fp32 accumulation: y = x * rsqrt(mean(x^2) + eps) * gamma.
//
// One warp per row.  Pass 1 streams the row from DRAM (UNROLL words in flight per lane),
// accumulates sum(x^2) in fp32 and keeps the raw words in the warp's SMEM stage.  The 16 lane
// partial sums are exchanged through SMEM (no warp shuffle on Muon); every lane sums all 16 so
// no broadcast is needed.  Pass 2 reads the stage and gamma (staged in SMEM once per cluster)
// and writes y.  DRAM traffic is the minimum: x read once, y written once.
#include <nightly/device.h>
#include <nightly/verify.h>
#include "rmsnorm_data.h"

#ifndef RMS_UNROLL
#define RMS_UNROLL 8
#endif

constexpr uint32_t WORDS = RMS_DIM / 2;                 // 32-bit words per row
constexpr uint32_t PER_LANE = WORDS / MU_NUM_THREADS;   // words per lane per row
static_assert(WORDS % (MU_NUM_THREADS * RMS_UNROLL) == 0, "DIM must be a multiple of 32*UNROLL");
constexpr uint32_t WARPS = MU_NUM_CORES * NIGHTLY_OCC;      // warps per cluster
// SMEM map: [0, 4*DIM/2)  gamma words | per-warp row stages | per-warp 16-word reduce buffers
constexpr uint32_t GAMMA_SM = 0;
constexpr uint32_t STAGE_SM = GAMMA_SM + WORDS * 4;
constexpr uint32_t RED_SM = STAGE_SM + WARPS * WORDS * 4;
static_assert(RED_SM + WARPS * 64 <= (128u << 10), "SMEM budget");

static inline float lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
static inline float hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xFFFF0000u); }
static inline uint32_t bf(float f) {
  const uint32_t u = __builtin_bit_cast(uint32_t, f);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t lane = tid % MU_NUM_THREADS, warp = tid / MU_NUM_THREADS;
  volatile __shared uint32_t *gamma_s = (volatile __shared uint32_t *)GAMMA_SM;
  volatile __shared uint32_t *stage = (volatile __shared uint32_t *)(STAGE_SM + warp * WORDS * 4);
  volatile __shared float *red = (volatile __shared float *)(RED_SM + warp * 64);

  // gamma -> SMEM, whole threadblock
  const uint32_t *G = (const uint32_t *)RMS_GAMMA_ADDR;
  for (uint32_t i = tid; i < WORDS; i += tpb) gamma_s[i] = G[i];
  mu_fence_smem();
  mu_barrier(1, tpb / MU_NUM_THREADS);

  const uint32_t gwarp = tb * WARPS + warp, gwarps = WARPS * NIGHTLY_CLUSTERS;
  for (uint32_t r = gwarp; r < RMS_ROWS; r += gwarps) {
    const uint32_t *X = (const uint32_t *)RMS_X_ADDR + r * WORDS;
    uint32_t *Y = (uint32_t *)RMS_Y_ADDR + r * WORDS;
    float acc0 = 0.f, acc1 = 0.f;
    for (uint32_t k = 0; k < PER_LANE; k += RMS_UNROLL) {
      uint32_t w[RMS_UNROLL];
#pragma unroll
      for (int u = 0; u < RMS_UNROLL; u++) w[u] = X[(k + u) * MU_NUM_THREADS + lane];
#pragma unroll
      for (int u = 0; u < RMS_UNROLL; u++) {
        const float a = lo(w[u]), b = hi(w[u]);
        acc0 += a * a; acc1 += b * b;
        stage[(k + u) * MU_NUM_THREADS + lane] = w[u];
      }
    }
    red[lane] = acc0 + acc1;
    mu_fence_smem();
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < MU_NUM_THREADS; j++) s += red[j];
    const float inv = 1.0f / __builtin_sqrtf(s * (1.0f / RMS_DIM) + RMS_EPS);
    for (uint32_t k = 0; k < PER_LANE; k += RMS_UNROLL) {
#pragma unroll
      for (int u = 0; u < RMS_UNROLL; u++) {
        const uint32_t i = (k + u) * MU_NUM_THREADS + lane;
        const uint32_t w = stage[i], g = gamma_s[i];
        Y[i] = bf(lo(w) * inv * lo(g)) | (bf(hi(w) * inv * hi(g)) << 16);
      }
    }
    mu_fence_smem();   // the next row's red[] writes must not pass this row's reads
  }
  nightly_kernel_end(tid, tpb);
  nightly_verify_bf16(RMS_Y_ADDR, RMS_G_ADDR, RMS_ROWS * RMS_DIM, tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
