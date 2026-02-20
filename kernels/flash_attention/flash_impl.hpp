#ifndef _FLASH_IMPL_H_
#define _FLASH_IMPL_H_

#include <stdint.h>
#include <mu_intrinsics.h>

#define B_ROW 64
#define B_COL 64
#define HEADDIM 128

/** Row-wise max reduction for softmax. */
template <uint32_t dim_row, uint32_t dim_col>
inline void rowmax(const _Float16 *tensor, _Float16 *result,
                   const uint32_t tid_in_threadblock,
                   const uint32_t threads_per_threadblock,
                   const uint32_t threadblock_id_in_cluster) {
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

    const auto NT = mu_num_threads();
    const auto tid_in_warp = tid_in_threadblock % NT;
    const auto row         = tid_in_threadblock / NT;

    const auto iter = dim_col / NT;
    const auto *elem = &tensor[row * dim_col + tid_in_warp];
    _Float16 max_per_thread = *elem;
    for (int i = 0; i < iter; i++) {
        if (*elem > max_per_thread) {
            max_per_thread = *elem;
        }
        elem += NT;
    }

    // TODO: threadblock barrier here

    result[0] = max_per_thread;
}

#endif
