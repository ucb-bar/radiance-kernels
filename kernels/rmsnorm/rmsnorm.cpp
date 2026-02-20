#include <math.h>
#include <vx_intrinsics.h>
#include <shared_mem.h>
#include <mu_intrinsics.h>

#include <stdlib.h>

#define EPS 1.0e-6f

template <size_t block_size>
void warp_reduce(__shared volatile float *sdata, unsigned int tid) {
  // muon warps are 16-wide
  if constexpr (block_size >= 32) sdata[tid] += sdata[tid + 16];
  if constexpr (block_size >= 16) sdata[tid] += sdata[tid + 8];
  if constexpr (block_size >= 8) sdata[tid] += sdata[tid + 4];
  if constexpr (block_size >= 4) sdata[tid] += sdata[tid + 2];
  if constexpr (block_size >= 2) sdata[tid] += sdata[tid + 1];
}

__shared float* sdata = reinterpret_cast<__shared float*>(0x0);

// reduces block_size*2*n elements from 
// g_idata[block_idx*block_size*2*n :+ block_size*2*n] into g_odata[block_idx]
// (assuming all threads in all warps are active)
template<size_t block_size>
void square_reduce(
  __global float* g_idata, 
  __global float* g_odata,
  unsigned int block_idx,
  unsigned int n
) {
  unsigned int tid = vx_warp_id() * 16 + vx_thread_id();
  
  // since muon has 2 cores, and thus only the ability to run two threadblocks at once,
  // striding by the size of the grid to extract more parallelism at threadblock level seems 
  // kinda pointless
  
  // unsigned int i = block_idx * block_size * 2 + tid;
  // unsigned int grid_size = block_size * 2 * gridDim.x;
  unsigned int i = block_idx * block_size * 2 * n + tid;
  unsigned int grid_size = block_size * 2;

  sdata[tid] = 0.0f;

  while (i < n) {
    float a = g_idata[i];
    float b = g_idata[i + block_size];
    sdata[tid] += a * a + b * b;
    i += grid_size;
  }

  // muon has 8 warps by default, so block size is limited to 8 warps * 16 threads/warp = 128 
  if constexpr (block_size >= 64) { 
    if (tid < 32) { sdata[tid] += sdata[tid + 32]; } 
    // TODO: muon execution / shared memory barrier
    // __syncthreads(); 
  }

  if (tid < 16) warp_reduce<block_size>(sdata, tid);
  if (tid == 0) {
    g_odata[block_idx] = sdata[0];
  }
}

float inv_rms(__global float* sum_squares, size_t n) {
  return 1.0f / sqrtf(EPS + *sum_squares / n);
}

void rmsnorm(
  __global float* idata,
  __global float* odata, 
  __global float* __restrict__ gamma, 
  float inv_rms
) {
  unsigned int tid = vx_warp_id() * 16 + vx_thread_id();
   
  odata[tid] = idata[tid] * gamma[tid] / inv_rms;
}

int main() {
  return 0;
}