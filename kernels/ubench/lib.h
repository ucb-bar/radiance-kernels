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
      } else if constexpr (std::is_same_v<T, _Float16>) {
        asm volatile("fadd.h %0, %0, %1" : "+r"(sum[j]) : "r"(incr) : "memory");
      } else if constexpr (std::is_same_v<T, float>) {
        asm volatile("fadd.s %0, %0, %1" : "+r"(sum[j]) : "r"(incr) : "memory");
      } else {
        static_assert(std::is_integral_v<T> || std::is_same_v<T, _Float16> || std::is_same_v<T, float>,
                      "sum supports only int, _Float16, float");
      }
    }
  }

  return sum[0];
}
