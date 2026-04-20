#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#define FAN2_NUM_WARPS 4

struct Fan2Args {
  __global float* m;
  __global float* a;
  __global float* b;
  uint32_t size;
  uint32_t t;
};

static inline void fan2(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<Fan2Args*>(arg);
  (void)threadblock_id;

  uint32_t rows = args->size - 1 - args->t;
  uint32_t cols = args->size - args->t;
  uint32_t total = rows * cols;
  float b_t = args->b[args->t];

  for (uint32_t idx = tid_in_threadblock; idx < total; idx += threads_per_threadblock) {
    uint32_t global_idx = idx / cols;
    uint32_t global_idy = idx % cols;
    uint32_t row = global_idx + 1 + args->t;
    uint32_t col = global_idy + args->t;
    float factor = args->m[args->size * row + args->t];
    args->a[args->size * row + col] -= factor * args->a[args->size * args->t + col];
    if (global_idy == 0) {
      args->b[row] -= factor * b_t;
    }
  }
}

Fan2Args fan2_args = {
  .m = nullptr,
  .a = nullptr,
  .b = nullptr,
  .size = 0,
  .t = 0,
};

#include GAUSSIAN_DATA_HEADER

int main() {
  fan2_args.m = reinterpret_cast<__global float*>(m_raw);
  fan2_args.a = reinterpret_cast<__global float*>(a_raw);
  fan2_args.b = reinterpret_cast<__global float*>(b_raw);
  fan2_args.size = size_val;
  fan2_args.t = t_val;
  mu_schedule(fan2, &fan2_args, FAN2_NUM_WARPS);
  return 0;
}
