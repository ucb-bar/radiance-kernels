#ifndef __MU_INTRINSICS_H__
#define __MU_INTRINSICS_H__

#include <stdint.h>
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

inline uint16_t load16_shared(uint32_t address) {
    uint16_t data;
    asm volatile("lh.shared %0, %1(%2)" : "=r"(data) : "I"(0), "r"(address)
                 : "memory");
    return data;
}

template <typename T>
inline uint16_t load16_shared(const T *address) {
    return load16_shared(reinterpret_cast<uint32_t>(address));
}

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
