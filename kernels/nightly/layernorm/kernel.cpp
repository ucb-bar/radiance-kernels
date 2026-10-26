// LayerNorm, bf16 in/out, fp32 accumulation: y = (x - mean) * rsqrt(var + eps) * gamma + beta.
//
// One warp per row, as in rmsnorm.  Pass 1 streams the row from DRAM (UNROLL words in flight per
// lane), accumulates sum(x) in fp32 and keeps the raw words in the warp's SMEM stage.  Pass 2
// reads the stage and accumulates sum((x - mean)^2), so the variance does not lose precision to
// cancellation when |mean| >> std.  Pass 3 reads the stage, gamma and beta (both staged in SMEM
// once per cluster) and writes y.  The 16 lane partial sums of each reduction are exchanged
// through SMEM (no warp shuffle on Muon); every lane sums all 16, so no broadcast is needed.
// DRAM traffic is the minimum: x read once, y written once.
#include <nightly/device.h>
#include <nightly/verify.h>
#include "layernorm_data.h"

#ifndef LN_UNROLL
#define LN_UNROLL 8
#endif

constexpr uint32_t WORDS = LN_DIM / 2;                  // 32-bit words per row
constexpr uint32_t PER_LANE = WORDS / MU_NUM_THREADS;   // words per lane per row
static_assert(WORDS % (MU_NUM_THREADS * LN_UNROLL) == 0, "DIM must be a multiple of 32*UNROLL");
constexpr uint32_t WARPS = MU_NUM_CORES * NIGHTLY_OCC;      // warps per cluster
// Row skew: a row is 4 KiB, the size of the direct-mapped 4 KiB L0d, so the warps of a core working on
// their rows at the same offset all hit one L0d set, and the L0d input adapter blocks a set while a nack
// on it is pending.  Each warp of a core starts its row LN_UNROLL lines further on (8 warps x 8 lines cover
// the 64 sets; mu_schedule numbers warps round-robin over cores, so warp / MU_NUM_CORES is the
// warp within its core).  LN_SKEW=0 disables it.
#ifndef LN_SKEW
#define LN_SKEW 1
#endif
static_assert((PER_LANE & (PER_LANE - 1)) == 0, "the row skew wraps with a mask");
// SMEM map: gamma words | beta words | per-warp row stages | per-warp 2 x 16-word reduce buffers
constexpr uint32_t GAMMA_SM = 0;
constexpr uint32_t BETA_SM = GAMMA_SM + WORDS * 4;
constexpr uint32_t STAGE_SM = BETA_SM + WORDS * 4;
constexpr uint32_t RED_SM = STAGE_SM + WARPS * WORDS * 4;
static_assert(RED_SM + WARPS * 128 <= (128u << 10), "SMEM budget");

static inline float lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
static inline float hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xFFFF0000u); }
static inline uint32_t bf(float f) {
  const uint32_t u = __builtin_bit_cast(uint32_t, f);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

// sum of the 16 lane partials: each lane writes its own, then every lane reads all 16
static inline float warp_sum(volatile __shared float *red, uint32_t lane, float v) {
  red[lane] = v;
  mu_fence_smem();
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < MU_NUM_THREADS; j++) s += red[j];
  return s;
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t lane = tid % MU_NUM_THREADS, warp = tid / MU_NUM_THREADS;
  const uint32_t skew = LN_SKEW ? (((warp / MU_NUM_CORES) % NIGHTLY_OCC) * LN_UNROLL) & (PER_LANE - 1) : 0;
  volatile __shared uint32_t *gamma_s = (volatile __shared uint32_t *)GAMMA_SM;
  volatile __shared uint32_t *beta_s = (volatile __shared uint32_t *)BETA_SM;
  volatile __shared uint32_t *stage = (volatile __shared uint32_t *)(STAGE_SM + warp * WORDS * 4);
  volatile __shared float *red_mean = (volatile __shared float *)(RED_SM + warp * 128);
  volatile __shared float *red_var = red_mean + MU_NUM_THREADS;

  // gamma, beta -> SMEM, whole threadblock
  const uint32_t *G = (const uint32_t *)LN_GAMMA_ADDR, *B = (const uint32_t *)LN_BETA_ADDR;
  for (uint32_t i = tid; i < WORDS; i += tpb) { gamma_s[i] = G[i]; beta_s[i] = B[i]; }
  mu_fence_smem();
  mu_barrier(1, tpb / MU_NUM_THREADS);

  const uint32_t gwarp = tb * WARPS + warp, gwarps = WARPS * NIGHTLY_CLUSTERS;
  for (uint32_t r = gwarp; r < LN_ROWS; r += gwarps) {
    const uint32_t *X = (const uint32_t *)LN_X_ADDR + r * WORDS;
    uint32_t *Y = (uint32_t *)LN_Y_ADDR + r * WORDS;
    // pass 1: DRAM -> stage, sum(x)
    float acc0 = 0.f, acc1 = 0.f;
    for (uint32_t kk = 0; kk < PER_LANE; kk += LN_UNROLL) {
      const uint32_t k = (kk + skew) & (PER_LANE - 1);
      uint32_t w[LN_UNROLL];
#pragma unroll
      for (int u = 0; u < LN_UNROLL; u++) w[u] = X[(k + u) * MU_NUM_THREADS + lane];
#pragma unroll
      for (int u = 0; u < LN_UNROLL; u++) {
        acc0 += lo(w[u]); acc1 += hi(w[u]);
        stage[(k + u) * MU_NUM_THREADS + lane] = w[u];
      }
    }
    const float mean = warp_sum(red_mean, lane, acc0 + acc1) * (1.0f / LN_DIM);
    // pass 2: stage -> sum((x - mean)^2)
    float sq0 = 0.f, sq1 = 0.f;
    for (uint32_t kk = 0; kk < PER_LANE; kk += LN_UNROLL) {
      const uint32_t k = (kk + skew) & (PER_LANE - 1);
#pragma unroll
      for (int u = 0; u < LN_UNROLL; u++) {
        const uint32_t w = stage[(k + u) * MU_NUM_THREADS + lane];
        const float a = lo(w) - mean, b = hi(w) - mean;
        sq0 += a * a; sq1 += b * b;
      }
    }
    const float var = warp_sum(red_var, lane, sq0 + sq1) * (1.0f / LN_DIM);
    const float inv = 1.0f / __builtin_sqrtf(var + LN_EPS);
    // pass 3: stage, gamma, beta -> y
    for (uint32_t kk = 0; kk < PER_LANE; kk += LN_UNROLL) {
      const uint32_t k = (kk + skew) & (PER_LANE - 1);
#pragma unroll
      for (int u = 0; u < LN_UNROLL; u++) {
        const uint32_t i = (k + u) * MU_NUM_THREADS + lane;
        const uint32_t w = stage[i], g = gamma_s[i], b = beta_s[i];
        Y[i] = bf((lo(w) - mean) * inv * lo(g) + lo(b)) | (bf((hi(w) - mean) * inv * hi(g) + hi(b)) << 16);
      }
    }
    mu_fence_smem();   // the next row's red[] writes must not pass this row's reads
  }
  nightly_kernel_end(tid, tpb);
  nightly_verify_bf16(LN_Y_ADDR, LN_G_ADDR, LN_ROWS * LN_DIM, tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
