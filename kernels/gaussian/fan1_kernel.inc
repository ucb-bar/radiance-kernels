#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#define FAN1_NUM_WARPS 4

struct Fan1Args {
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
  auto* args = reinterpret_cast<Fan1Args*>(arg);
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

Fan1Args fan1_args = {
  .m = nullptr,
  .a = nullptr,
  .b = nullptr,
  .size = 0,
  .t = 0,
};

#include GAUSSIAN_DATA_HEADER

int main() {
  fan1_args.m = reinterpret_cast<__global float*>(m_raw);
  fan1_args.a = reinterpret_cast<__global float*>(a_raw);
  fan1_args.b = reinterpret_cast<__global float*>(b_raw);
  fan1_args.size = size_val;
  fan1_args.t = t_val;
  mu_schedule(fan1, &fan1_args, FAN1_NUM_WARPS);
  return 0;
}
