#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#ifndef GAUSSIAN_NUM_WARPS
#define GAUSSIAN_NUM_WARPS 4
#endif

#ifndef GAUSSIAN_DATA_HEADER
#error "GAUSSIAN_DATA_HEADER must be defined before including kernel_impl.hpp"
#endif

extern "C" uint32_t __mu_num_warps = GAUSSIAN_NUM_WARPS;

struct GaussianArgs {
  __global float* m;
  __global float* a;
  __global float* b;
  uint32_t size;
  uint32_t t;
};

static inline void fan1(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<GaussianArgs*>(arg);
  (void)threadblock_id;
  (void)args->b;

  uint32_t count = args->size - 1 - args->t;
  float pivot = args->a[args->size * args->t + args->t];
  for (uint32_t global_id = tid_in_threadblock; global_id < count;
       global_id += threads_per_threadblock) {
    uint32_t row = global_id + args->t + 1;
    args->m[args->size * row + args->t] =
      args->a[args->size * row + args->t] / pivot;
  }
}

static inline void fan2(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<GaussianArgs*>(arg);
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

GaussianArgs gaussian_args = {
  .m = nullptr,
  .a = nullptr,
  .b = nullptr,
  .size = 0,
  .t = 0,
};

#include GAUSSIAN_DATA_HEADER

int main() {
  gaussian_args.m = reinterpret_cast<__global float*>(m_raw);
  gaussian_args.a = reinterpret_cast<__global float*>(a_raw);
  gaussian_args.b = reinterpret_cast<__global float*>(b_raw);
  gaussian_args.size = size_val;
  gaussian_args.t = t_val;

#if defined(GAUSSIAN_FAN1)
  mu_schedule(fan1, &gaussian_args, GAUSSIAN_NUM_WARPS);
#elif defined(GAUSSIAN_FAN2)
  mu_schedule(fan2, &gaussian_args, GAUSSIAN_NUM_WARPS);
#else
#error "Define GAUSSIAN_FAN1 or GAUSSIAN_FAN2 before including kernel_impl.hpp"
#endif

  return 0;
}
