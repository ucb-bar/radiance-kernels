#include <stdint.h>
#include <type_traits>
#include <vx_intrinsics.h>

template <typename T>
inline T mem() {
  constexpr int ILP = 16;
  constexpr int N = 1024 / ILP;

  // text region
  T *base = reinterpret_cast<T *>(0x10000000);
  T val[ILP];

#pragma unroll 100
  for (int i = 0; i < N; i++) {
#pragma unroll ILP
    for (int j = 0; j < ILP; j++) {
      const uint32_t off = sizeof(T) * ((i * ILP) + j);
      const T *addr = base + off;
      // asm volatile("lw %0, %2(%1)" : "+r"(val[j]) : "r"(base), "i"(off) : "memory");
      asm volatile("lw %0, (%1)" : "+r"(val[j]) : "r"(addr) : "memory");
    }
  }

  return 0;
}

int main() {
  vx_tmc(vx_num_threads());

  mem<int32_t>();

  vx_tmc(1);

  return 0;
}
