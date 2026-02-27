#include <math.h>
#include <vx_intrinsics.h>
#include <shared_mem.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdlib.h>

#define EPS 1.0e-6f

void syncthreads() {
  // barrier id 0, wait for all 2 cores * 8 warps/core = 16 warps to reach it
  vx_barrier(0, 16);
}

void warp_reduce(__shared volatile float *sdata, unsigned int lane) {
  // muon warps are 16-wide
  sdata[lane] += sdata[lane + 16];
  sdata[lane] += sdata[lane + 8];
  sdata[lane] += sdata[lane + 4];
  sdata[lane] += sdata[lane + 2];
  sdata[lane] += sdata[lane + 1];
}

float inv_rms(float sum_squares, size_t n) {
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
void rmsnorm2(
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

  __shared float* warp_sdata = reinterpret_cast<__shared float*>(warp * threads_per_warp * sizeof(float));

  uint32_t i = warp;
  while (i < L) {
    __global float* warp_data = &data[i * D];

    float accum = 0.0f;
    warp_sdata[lane] = 0.0f;

    for (uint32_t j = lane; j < D; j += threads_per_warp) {
      float f = warp_data[j];
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
    for (uint32_t j = lane; j < D; j += threads_per_warp) {
      warp_data[j] *= gamma[j] * irms;
    }

    i += warps_per_threadblock * D;
  }
}

static struct RmsNormArgs {
  __global float* data;
  __global float* gamma;
  uint32_t L;
} rmsnorm_args;

void rmsnorm2_entry(void* arg, uint32_t tid_in_threadblock, uint32_t _threads_per_threadblock, uint32_t _threadblock_id) {
  const auto* a = reinterpret_cast<const RmsNormArgs*>(arg);
  rmsnorm2<192>(a->data, a->gamma, a->L, tid_in_threadblock);
}

#include "data"

int main() {
  rmsnorm_args.data = data;
  rmsnorm_args.gamma = gamma;
  rmsnorm_args.L = 1024;
  mu_schedule(rmsnorm2_entry, &rmsnorm_args);

  return 0;
}