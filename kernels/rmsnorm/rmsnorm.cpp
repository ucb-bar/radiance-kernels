#include <math.h>
#include <vx_intrinsics.h>
#include <shared_mem.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdlib.h>

#define EPS 1.0e-6f

// inlining this increases register pressure too much
__attribute__((noinline)) static void warp_reduce(__shared volatile float *sdata, unsigned int lane) {
  // muon warps are 16-wide
  if (lane < 8)  sdata[lane] += sdata[lane + 8];
  if (lane < 4)  sdata[lane] += sdata[lane + 4];
  if (lane < 2)  sdata[lane] += sdata[lane + 2];
  if (lane < 1)  sdata[lane] += sdata[lane + 1];
}

// inlining this increases register pressure too much
__attribute__((noinline)) static float inv_rms(float sum_squares, size_t n) {
  return 1.0f / sqrtf(EPS + sum_squares / n);
}

// operates in place 
// mapping is 1 warp per token to avoid synchronization overhead since D is so small (~192 elements)
// `data` dimensions = (L, D)
//   D should have a stride of 1 
//   L = batch * sequence length, where batch is usually 1
// `gamma` dimensions = (D,)
// assumes 2 cores * 8 warps/core = 16 warps in 1 threadblock
template <size_t D>
__attribute__((always_inline)) static inline void rmsnorm2(
  __global float* data,
  __global float* __restrict__ gamma,
  uint32_t L,
  uint32_t tid_in_threadblock
) {
  static_assert(D % 16 == 0, "embedding dimension should be divisible by warp size, ideally");
  static constexpr uint32_t warps_per_threadblock = 16;
  static constexpr uint32_t threads_per_warp = 16;
  
  uint32_t warp = tid_in_threadblock / 16;
  uint32_t lane = tid_in_threadblock % 16;

  __shared volatile float* warp_sdata = reinterpret_cast<__shared float*>(warp * threads_per_warp * sizeof(float));

  uint32_t i = warp;
  while (i < L) {
    __global float* warp_data = &data[i * D];

    float accum = 0.0f;
    warp_sdata[lane] = 0.0f;

    for (uint32_t k = 0; k < (D / threads_per_warp); k += 1) {
      float f = warp_data[lane + k * threads_per_warp];
      accum += f * f;
    }

    warp_sdata[lane] = accum;
    warp_reduce(warp_sdata, lane);

    if (lane == 0) {
      float sum_squares = warp_sdata[lane];
      warp_sdata[lane] = inv_rms(sum_squares, D);
    }

    // maybe a fence is needed, ideally a __syncwarp() basically
    // IPDOM scheduling should ensure that irms gets the right value...
    
    float irms = warp_sdata[0];

    // somehow, this strange code shape gets clang to split this into two loops
    // the first which premultiplies gamma, the second multiplying by irms, sharing the 
    // same instructions
    // ¯\_(ツ)_/¯ agentic descent op i suppose
    __global float* wd = warp_data + lane;
    __global float* gg = gamma + lane;
    for (uint32_t k = 0; k < (D / threads_per_warp); ++k) {
      *wd *= *gg * irms;
      wd += threads_per_warp;
      gg += threads_per_warp;
    }

    i += warps_per_threadblock;
  }
}

static struct RmsNormArgs {
  __global float* data;
  __global float* gamma;
  uint32_t L;
} rmsnorm_args;

static void rmsnorm2_entry(void* arg, uint32_t tid_in_threadblock, uint32_t _threads_per_threadblock, uint32_t _threadblock_id) {
  const auto* a = reinterpret_cast<const RmsNormArgs*>(arg);
  rmsnorm2<192>(a->data, a->gamma, a->L, tid_in_threadblock);
}

#include "data"

int main() {
  rmsnorm_args.data = data;
  rmsnorm_args.gamma = gamma;
  rmsnorm_args.L = 128;
  mu_schedule(rmsnorm2_entry, &rmsnorm_args);

  return 0;
}
