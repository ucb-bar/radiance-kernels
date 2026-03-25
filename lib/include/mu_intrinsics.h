#ifndef __MU_INTRINSICS_H__
#define __MU_INTRINSICS_H__

#include <stdint.h>
#include <type_traits>
#include <vx_intrinsics.h>
#include "shared_mem.h"

// You need to use __builtin_bit_cast(_Float16, ONE_BF16_BITS) for the compiler to correctly emit it.
// use as_bf16 to quickly convert
#define ONE_BF16_BITS ((uint16_t)0x3f80)
#define NEG_INF_BF16_BITS ((uint16_t) 0xFF80)

// 128 KiB SMEM
#define MU_SMEM_SIZE_BYTES (128 << 10)

// 64 KB cache line
#define CACHE_LINE_BYTES 64

#define MU_CSR_CLUSTER_ID 0xCD0

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

inline uint32_t load32_shared(uint32_t address) {
    uint32_t data;
    asm volatile("lw.shared %0, %1(%2)" : "=r"(data) : "I"(0), "r"(address)
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

__attribute__((convergent))
inline void mu_fence() {
    asm volatile ("fence" ::: "memory");
}
__attribute__((convergent))
inline void mu_fence_smem() {
    asm volatile ("fence.s" ::: "memory");
}

/** NOTE about barriers: Placing barriers around thread-divergent branches
 *  may cause bugs.  The compiler might decide to duplicate mu_barrier() into
 *  both paths of a warp-divergent branch, which will cause the barrier to
 *  execute twice via SIMT serialization, and cause potential deadlocks.
 *  mu_barrier() doesn't check for participating tmasks.
 *
 *  We wrap mu_barrier() with convergent/noduplicate/noinline, but that doesn't
 *  seem to be sufficient.
 *
 *  This seems to happen the most around single-thread-guarded code, e.g.:
 *
 *    if (tid == 0) {
 *        // do something
 *    }
 *    mu_barrier(...);
 *
 *  A workaround that _may_ work is to put an explicit else clause with a
 *  nop in it:
 *
 *    if (tid == 0) {
 *        // do something
 *    } else {
 *        asm volatile("nop");
 *    }
 *    mu_barrier(...);
 *
 *  Another workaround is to use -Os for the optimization, which keeps
 *  the compiler from branch-duplicating to save code size.
 *
 *  None of these workarounds are fundamental, and we need proper compiler
 *  support to reason about warp-convergence.  TODO.
 */
__attribute__((convergent))
static void mu_barrier(unsigned barried_id, unsigned num_warps) {
    asm volatile ("vx_bar %0, %1" :: "r"(barried_id), "r"(num_warps) : "memory");
}

// This hard-codes hardware config into kernel, but this allows efficient
// compile-time unrolling and constant propagation.
#define MU_NUM_THREADS 16
#define MU_NUM_WARPS 8
#define MU_NUM_CORES 2
#define MU_NUM_MAX_WARPS 8
#define MU_NUM_CLUSTERS 1
#define MU_BLOCK_NUM_WARPS(n) (MU_NUM_CORES * (n))
#define MU_BLOCK_SIZE(n) (MU_BLOCK_NUM_WARPS(n) * MU_NUM_THREADS)
#define MU_DOUBLE_BLOCK_SIZE(n) (MU_BLOCK_SIZE(n) * 2)

// This compiles to CSR reads which stalls the pipeline. Use sparingly & cache.
inline int mu_num_threads() {
    return vx_num_threads();
}

inline _Float16 mu_fexp(_Float16 arg) {
    _Float16 output;
    asm volatile("fexp.h %0, %1" : "=r"(output) : "r"(arg));
    return output;
}

inline _Float16 mu_fnexp(_Float16 arg) {
    _Float16 output;
    asm volatile("fnexp.h %0, %1" : "=r"(output) : "r"(arg));
    return output;
}


// TODO: half?

static inline _Float16 as_bf16(uint16_t bits) {
  return __builtin_bit_cast(_Float16, bits);
}

struct bf16x2 { _Float16 lo, hi; };

static inline bf16x2 unpack_bf16x2(uint32_t packed) {
  return { as_bf16((uint16_t)packed), as_bf16((uint16_t)(packed >> 16)) };
}

static inline uint32_t pack_bf16x2(_Float16 lo, _Float16 hi) {
  return (uint32_t)__builtin_bit_cast(uint16_t, lo)
       | ((uint32_t)__builtin_bit_cast(uint16_t, hi) << 16);
}

#define LABEL(name) asm volatile(#name ":" ::: "memory")

#endif // __MU_INTRINSICS_H__
