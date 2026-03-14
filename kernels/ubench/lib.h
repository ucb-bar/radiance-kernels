#include <stdint.h>
#include <type_traits>
#include <mu_intrinsics.h>

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

inline _Float16 exp() {
  constexpr int ILP = 32;
  constexpr int N = 2048 / ILP;

  _Float16 x[ILP] = {static_cast<_Float16>(0.001f),};

#pragma unroll 100
  for (int i = 0; i < N; i++) {
#pragma unroll ILP
    for (int j = 0; j < ILP; j++) {
      asm volatile("fexp.h %0, %0" : "+r"(x[j]) : : "memory");
    }
    for (int j = 0; j < ILP; j++) {
      asm volatile("fnexp.h %0, %0" : "+r"(x[j]) : : "memory");
    }
  }

  return x[0];
}

template <typename T>
inline T load_gmem_coalesced(const volatile __global T *base) {
    constexpr int ILP = 16;
    constexpr int N = 1 << 18;
    constexpr int UNROLL = 8;
    constexpr auto NT = MU_NUM_THREADS;
    static_assert(UNROLL * ILP * NT < N);

    const auto tid = vx_thread_id();

    #pragma unroll UNROLL;
    for (int i = 0; i < N; i += ILP * NT) {
        T val[ILP];
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            // const uint32_t index = i + j * NT + tid;
            const uint32_t index = tid;
            val[j] = base[index];
        }
    }

    return 0;
}
