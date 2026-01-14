#include <stdint.h>
#include <type_traits>
#include <vx_intrinsics.h>

template <typename T>
inline T sum() {
  constexpr int ILP = 16;
  constexpr int N = 2048 / ILP;

  T sum[ILP] = {static_cast<T>(0),};
  T incr = 1.;

#pragma unroll 100
  for (int i = 0; i < N; i++) {
    // sum += i;
#pragma unroll ILP
    for (int j = 0; j < ILP; j++) {
      if constexpr (std::is_integral_v<T>) {
        asm volatile("add %0, %0, %1" : "+r"(sum[j]) : "r"(incr) : "memory");
      } else {
        asm volatile("fadd.s %0, %0, %1" : "+r"(sum[j]) : "r"(incr) : "memory");
      }
    }
  }

  return sum[0];
}

int main() {
  vx_tmc(vx_num_threads());

  sum<int32_t>();

  vx_tmc(1);

  return 0;
}
