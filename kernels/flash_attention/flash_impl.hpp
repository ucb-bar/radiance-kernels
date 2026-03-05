#ifndef _FLASH_IMPL_H_
#define _FLASH_IMPL_H_

#include <cstdint>
#include <cmath>
#include <mu_intrinsics.h>

#define SEQLEN 1024
#define HEADDIM 64
#define B_ROW 64
#define B_COL 64

/** Move BF16 tensor data from GMEM->SMEM.
 *  Assumes row-major layout for both src and dest. */
template <uint32_t dim_row, uint32_t dim_col>
inline void copy_gmem_to_smem(const volatile _Float16 *src_gmem, volatile _Float16 *dest_smem,
                              const uint32_t tid_in_threadblock,
                              const uint32_t threads_per_threadblock) {
    // Thread mapping: All warps in a threadblock cooperatively copies a
    // contiguous chunk of the same size as the threadblock per every "wave".
    // The number of waves are determined with:
    const auto iter = (dim_row * dim_col) / threads_per_threadblock;

#pragma unroll 16
    for (int i = 0; i < iter; i++) {
        const auto index = (threads_per_threadblock) * i + tid_in_threadblock;
        const auto data = src_gmem[index];
        auto smem_address = &dest_smem[index];
        asm volatile("sh.shared %1, 0(%0)" :: "r"(smem_address), "r"(data) : "memory");
    }
}

/** Row-wise max reduction for softmax. */
template <uint32_t dim_row, uint32_t dim_col>
inline void rowmax(const _Float16 *tensor, _Float16 *rowmax, _Float16 *scratch,
                   const uint32_t tid_in_threadblock,
                   const uint32_t threads_per_threadblock,
                   const uint32_t threadblock_id) {
    // Thread mapping scheme
    // ---------------------
    //
    // Map each row to a single warp; each contiguous warplen-elements map to
    // neighboring threads in a single warp, allowing conflict-free accesses.
    // Each thread iterates through all elements, reducing down to single-warp,
    // per-thread intermediate max values.  Finally, we elect a single thread
    // to reduce them to the single maximum.
    //
    // Row 0: Warp 0 [ T0 T1 T2 T3 T0 T1 T2 T3 ... ]
    // Row 1: Warp 1 [ T0 T1 T2 T3 T0 T1 T2 T3 ... ]
    // Row 2: Warp 2 [ T0 T1 T2 T3 T0 T1 T2 T3 ... ]
    //       ↓
    // reduced per-thread max
    // [ T0' T1' T2' T3' ]
    //       ↓
    // Single-thread max reduction
    // [ T0'' ]

    constexpr auto NT = MU_NUM_THREADS;
    const auto tid_in_warp = tid_in_threadblock % NT;
    const auto warp_id     = tid_in_threadblock / NT;
    const auto row_stride  = threads_per_threadblock / NT;
    const auto row_iter    = dim_row / row_stride;

    for (int j = 0; j < row_iter; j++) {
        const auto row = (j * row_stride) + warp_id;
        if (row >= dim_row) {
            return;
        }
        const auto col_iter = dim_col / NT;
        const auto *elem_addr = &tensor[row * dim_col + tid_in_warp];
        _Float16 max_per_thread = load16_shared(elem_addr);
        for (int i = 0; i < col_iter; i++) {
            const auto elem = load16_shared(elem_addr);
            max_per_thread = fmax(elem, max_per_thread);
            elem_addr += NT;
        }

        auto temp_addr = &scratch[tid_in_threadblock];
        asm volatile("sh.shared %1, 0(%0)" :: "r"(temp_addr), "r"(max_per_thread) : "memory");

        // elect a single thread to reduce within the warp
        if (tid_in_warp == 0) {
            _Float16 max = load16_shared(temp_addr);
            for (int i = 0; i < NT; i++) {
                const auto elem = load16_shared(temp_addr);
                max = fmax(elem, max);
                temp_addr++;
            }
            auto result_addr = &rowmax[row];
            asm volatile("sh.shared %1, 0(%0)" :: "r"(result_addr), "r"(max) : "memory");
        }
    }

    // TODO: threadblock barrier here
}

#endif
