#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <stdint.h>

#ifndef SFILTER_NUM_WARPS
#define SFILTER_NUM_WARPS 4
#endif

extern "C" uint32_t __mu_num_warps = SFILTER_NUM_WARPS;

struct SFilterArgs {
  __global float* src;
  __global float* dst;
  uint32_t ldc;
  float m0;
  float m1;
  float m2;
  float m3;
  float m4;
  float m5;
  float m6;
  float m7;
  float m8;
};

static inline void sfilter(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<SFilterArgs*>(arg);
  (void)threadblock_id;

  uint32_t interior = (args->ldc - 2) * (args->ldc - 2);
  for (uint32_t idx = tid_in_threadblock; idx < interior; idx += threads_per_threadblock) {
    uint32_t x = 1 + (idx % (args->ldc - 2));
    uint32_t y = 1 + (idx / (args->ldc - 2));
    uint32_t addr = x + y * args->ldc;

    float i0 = args->src[addr - 1 - args->ldc] * args->m0;
    float i1 = args->src[addr - args->ldc] * args->m1;
    float i2 = args->src[addr + 1 - args->ldc] * args->m2;
    float i3 = args->src[addr - 1] * args->m3;
    float i4 = args->src[addr] * args->m4;
    float i5 = args->src[addr + 1] * args->m5;
    float i6 = args->src[addr - 1 + args->ldc] * args->m6;
    float i7 = args->src[addr + args->ldc] * args->m7;
    float i8 = args->src[addr + 1 + args->ldc] * args->m8;

    args->dst[addr] = i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8;
  }
}

SFilterArgs sfilter_args = {
  .src = nullptr,
  .dst = nullptr,
  .ldc = 0,
  .m0 = 0.0f,
  .m1 = 0.0f,
  .m2 = 0.0f,
  .m3 = 0.0f,
  .m4 = 0.0f,
  .m5 = 0.0f,
  .m6 = 0.0f,
  .m7 = 0.0f,
  .m8 = 0.0f,
};

#include "data"

int main() {
  sfilter_args.src = reinterpret_cast<__global float*>(src_raw);
  sfilter_args.dst = reinterpret_cast<__global float*>(dst_raw);
  sfilter_args.ldc = ldc;
  sfilter_args.m0 = __builtin_bit_cast(float, m0_raw);
  sfilter_args.m1 = __builtin_bit_cast(float, m1_raw);
  sfilter_args.m2 = __builtin_bit_cast(float, m2_raw);
  sfilter_args.m3 = __builtin_bit_cast(float, m3_raw);
  sfilter_args.m4 = __builtin_bit_cast(float, m4_raw);
  sfilter_args.m5 = __builtin_bit_cast(float, m5_raw);
  sfilter_args.m6 = __builtin_bit_cast(float, m6_raw);
  sfilter_args.m7 = __builtin_bit_cast(float, m7_raw);
  sfilter_args.m8 = __builtin_bit_cast(float, m8_raw);
  mu_schedule(sfilter, &sfilter_args, SFILTER_NUM_WARPS);
  return 0;
}
