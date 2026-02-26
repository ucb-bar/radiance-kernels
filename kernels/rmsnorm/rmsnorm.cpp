#include <math.h>
#include <vx_intrinsics.h>
#include <shared_mem.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>

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

__shared float* const sdata = reinterpret_cast<__shared float*>(0x0);

// reduces g_idata into first num_threadblocks entries of g_odata
template<size_t block_size>
void square_reduce(
  __global float* g_idata, 
  __global float* g_odata,
  unsigned int tid_in_threadblock,
  unsigned int threadblock_id,
  unsigned int num_threadblocks,
  unsigned int n // total size of g_idata
) {
  
  // unsigned int i = block_idx * block_size * 2 + tid;
  // unsigned int grid_size = block_size * 2 * gridDim.x;
  unsigned int i = threadblock_id * block_size * 2 + tid_in_threadblock;
  unsigned int grid_size = block_size * 2 * num_threadblocks;

  sdata[tid_in_threadblock] = 0.0f;

  while (i < n) {
    float a = g_idata[i];
    float b = (i + block_size < n) ? g_idata[i + block_size] : 0.0f;
    sdata[tid_in_threadblock] += a * a + b * b;
    i += grid_size;
  }

  // muon has 2 cores and 8 warps by default, so block size (i.e. maximum resident threads per SM) 
  // is limited to 2 cores * 8 warps/core * 16 threads/warp = 256
  if constexpr (block_size >= 256) {
    if (tid_in_threadblock < 128) { sdata[tid_in_threadblock] += sdata[tid_in_threadblock + 128]; }
    // TODO: muon execution / shared memory barrier
    // __syncthreads();
  }
  if constexpr (block_size >= 128) {
    if (tid_in_threadblock < 64) { sdata[tid_in_threadblock] += sdata[tid_in_threadblock + 64]; }
    // TODO: muon execution / shared memory barrier
    // __syncthreads();
  }
  if constexpr (block_size >= 64) { 
    if (tid_in_threadblock < 32) { sdata[tid_in_threadblock] += sdata[tid_in_threadblock + 32]; } 
    // TODO: muon execution / shared memory barrier
    // __syncthreads(); 
  }

  if (tid_in_threadblock < 16) warp_reduce<block_size>(sdata, tid_in_threadblock);
  if (tid_in_threadblock == 0) {
    g_odata[threadblock_id] = sdata[0];
  }
}

struct SquareReduceArgs {
  __global float* g_idata;
  __global float* g_odata;
  unsigned int n;
};

void square_reduce_entry(void* arg, uint32_t tid_in_threadblock, uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  auto* square_reduce_args = reinterpret_cast<const SquareReduceArgs*>(arg);
  
  if (threads_per_threadblock == 256) {
    square_reduce<256>(
      square_reduce_args->g_idata, 
      square_reduce_args->g_odata, 
      tid_in_threadblock, 
      threadblock_id, 
      1,
      square_reduce_args->n
    );
  } 
}

float inv_rms(float sum_squares, size_t n) {
  return 1.0f / sqrtf(EPS + sum_squares / n);
}

void rmsnorm(
  __global float* idata,
  __global float* odata, 
  __global float* __restrict__ gamma, 
  float inv_rms
) {
  *odata = *idata * (*gamma) / inv_rms;
}

struct RmsNormArgs {
  __global float* idata;
  __global float* odata;
  __global float* gamma;
  unsigned int n;
  float inv_rms;
};

void rmsnorm_entry(void* arg, uint32_t tid_in_threadblock, uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  auto* rmsnorm_args = reinterpret_cast<const RmsNormArgs*>(arg);
  
  uint32_t block_size = threads_per_threadblock;
  uint32_t num_threadblocks = 1;
  uint32_t grid_size = block_size * num_threadblocks;

  uint32_t i = tid_in_threadblock;
  while (i < rmsnorm_args->n) {
    rmsnorm(
      &rmsnorm_args->idata[tid_in_threadblock],
      &rmsnorm_args->odata[tid_in_threadblock],
      &rmsnorm_args->gamma[tid_in_threadblock],
      rmsnorm_args->inv_rms
    );
  }
  
}

SquareReduceArgs square_reduce_args = {
  .g_idata = nullptr,
  .g_odata = nullptr,
  .n = 4096
};

RmsNormArgs rmsnorm_args = {
  .idata = nullptr,
  .odata = nullptr,
  .gamma = nullptr,
  .n = 4096,
  .inv_rms = 0.0f
};

int main() {
  mu_schedule(square_reduce_entry, &square_reduce_args);
  rmsnorm_args.inv_rms = inv_rms(square_reduce_args.g_odata[0], square_reduce_args.n);
  mu_schedule(rmsnorm_entry, &rmsnorm_args);

  return 0;
}