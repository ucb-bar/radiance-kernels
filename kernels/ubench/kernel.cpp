#include <stdint.h>
#include <vx_intrinsics.h>

int main() {
  vx_tmc(vx_num_threads());

  int sum0 = 0;
  int sum1 = 0;
  int sum2 = 0;
  int sum3 = 0;
  int incr = 1;
#pragma unroll 10
  for (int i = 0; i < 1000; i++) {
    // sum += i;
    asm volatile("add %0, %0, %1" : "+r"(sum0) : "r"(incr) : "memory");
    asm volatile("add %0, %0, %1" : "+r"(sum1) : "r"(incr) : "memory");
    asm volatile("add %0, %0, %1" : "+r"(sum2) : "r"(incr) : "memory");
    asm volatile("add %0, %0, %1" : "+r"(sum3) : "r"(incr) : "memory");
  }

  vx_tmc(1);

  int sum = sum0 + sum1 + sum2 + sum3;
  return sum;
}
