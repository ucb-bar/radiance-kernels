#ifndef __MU_INTRINSICS_H__
#define __MU_INTRINSICS_H__

#include <stdint.h>
#include <type_traits>
#include <vx_intrinsics.h>

inline void store_shared(uint32_t base, uint32_t offset, uint32_t data) {
    asm volatile("sw.shared %2, %1(%0)" :: "r"(base), "I"(offset), "r"(data)
                 : "memory");
}

inline void store64_shared(uint32_t base, uint32_t offset, uint64_t data) {
    uint32_t lo = static_cast<uint32_t>(data);
    uint32_t hi = static_cast<uint32_t>(data >> 32);
    store_shared(base, offset,     lo);
    store_shared(base, offset + 4, hi);
}

inline void store_shared_from_global(uint32_t dst, uint32_t src) {
    uint32_t data;
    asm volatile("lw.global %0, 0(%1)" : "=r"(data) : "r"(src) : "memory");
    asm volatile("sw.shared %1, 0(%0)" :: "r"(dst), "r"(data) : "memory");
}

inline uint16_t load16_shared(uint32_t address) {
    uint16_t data;
    asm volatile("lh.shared %0, %1(%2)" : "=r"(data) : "I"(0), "r"(address)
                 : "memory");
    return data;
}

template <typename T>
inline std::remove_cv_t<T> load16_shared(const T *address) {
    // need bit_cast to re-interpret uint16_t bits as _Float16
    using U = std::remove_cv_t<T>;
    static_assert(sizeof(U) == sizeof(uint16_t), "load16_shared<T*> expects 16-bit T");
    uint16_t bits = load16_shared(reinterpret_cast<uint32_t>(address));
    return __builtin_bit_cast(U, bits);
}

// Number of threads per warp.
// This hard-codes architecture detail into kernel, but this allows efficient
// compile-time unrolling and constant propagation.
#define MU_NUM_THREADS 16

// hard-coded block size for now, change if tapeout config changes
#define MU_BLOCK_SIZE 256

// This compiles to CSR reads which stalls the pipeline. Use sparingly & cache.
inline int mu_num_threads() {
    return vx_num_threads();
}

inline float mu_fexp(float arg) {
    float output;
    asm volatile("fexp.s %0, %1" ::"r"(output), "r"(arg));
    return output;
}

inline float mu_fnexp(float arg) {
    float output;
    asm volatile("fnexp.s %0, %1" ::"r"(output), "r"(arg));
    return output;
}

// TODO: half?

#endif // __MU_INTRINSICS_H__
