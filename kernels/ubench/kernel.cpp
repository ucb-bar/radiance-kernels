#include <stdint.h>
#include <vx_intrinsics.h>

int main() {
  vx_tmc(vx_num_threads());

  constexpr int inst_parallelism = 4;
  constexpr int N = 1000 / inst_parallelism;

  int sum0 = 0;
  int sum1 = 0;
  int sum2 = 0;
  int sum3 = 0;
  int incr = 1;

#pragma unroll N
  for (int i = 0; i < N; i++) {
    // sum += i;
    asm volatile("add %0, %0, %1" : "+r"(sum0) : "r"(incr) : "memory");
    if constexpr (inst_parallelism >= 2) {
      asm volatile("add %0, %0, %1" : "+r"(sum1) : "r"(incr) : "memory");
    }
    if constexpr (inst_parallelism >= 3) {
      asm volatile("add %0, %0, %1" : "+r"(sum2) : "r"(incr) : "memory");
    }
    if constexpr (inst_parallelism >= 4) {
      asm volatile("add %0, %0, %1" : "+r"(sum3) : "r"(incr) : "memory");
    }
  }

  vx_tmc(1);

  int sum = sum0 + sum1 + sum2 + sum3;
  return sum;
}
