#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

struct SoftmaxArgs {
  float* input;
  float* output;
  uint32_t rows;
  uint32_t cols;
};

__shared float* const sdata = reinterpret_cast<__shared float*>(0x0);

// requires that cols fits in smem (one row double buffered)
void softmax(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<SoftmaxArgs*>(arg);
  uint32_t row_elems = args->cols;
  uint32_t rows_per_threadblock = (MU_BLOCK_SIZE + row_elems - 1 / row_elems) ;
  uint32_t block_row_idx = threadblock_id * rows_per_threadblock;
  uint32_t block_elem_idx = block_row_idx * row_elems;
  float *block_sdata = &sdata[block_row_idx * row_elems];
  float *block_input = &(args->input[block_elem_idx]);

  // load row to smem
  for (uint32_t load_idx = 0; load_idx < row_elems; load_idx += MU_NUM_THREADS)
    block_sdata[load_idx] = block_input[load_idx];

  uint32_t chunks_per_block = (row_elems + MU_BLOCK_SIZE - 1) / MU_BLOCK_SIZE;
  uint32_t thread_chunk_zero_idx = block_row_idx * row_elems + tid_in_threadblock;
  if (thread_chunk_zero_idx < row_elems)
      block_sdata[thread_chunk_zero_idx] = mu_fexp(block_sdata[thread_chunk_zero_idx]);

  uint32_t thread_chunk_i_idx = thread_chunk_zero_idx + MU_NUM_THREADS;
  for (uint32_t i = 0; i < chunks_per_block; i ++) {
    if (thread_chunk_i_idx < row_elems)
      block_sdata[thread_chunk_zero_idx] += mu_fexp(block_sdata[thread_chunk_i_idx]);
  }

  // __syncthreads()
  for (uint32_t active_stride = 2; active_stride < MU_NUM_THREADS; active_stride *= 2)
    if (thread_chunk_zero_idx % active_stride)
      block_sdata[thread_chunk_zero_idx] += block_sdata[thread_chunk_zero_idx + (active_stride >> 1)];
}

SoftmaxArgs softmax_args = {
  .input = nullptr,
  .output = nullptr,
  .rows = 0,
  .cols = 0,
};

int main() {
  mu_schedule(softmax, &softmax_args);
  return 0;
}
