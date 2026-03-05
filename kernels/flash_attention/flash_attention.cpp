#include <radiance.h>
#include <mu_schedule.h>

#include "flash_impl.hpp"
#include "args.hpp"
#include "include/matmul_data.h"

extern _Float16 numpy_a_bin[];
extern _Float16 numpy_b_bin[];

void kernel_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock, uint32_t threadblock_id) {
    auto kernel_arg = reinterpret_cast<const FlashAttentionKernelArgs *>(arg);

    // const auto tid = mu_mhartid; // TODO

    // auto tensor = reinterpret_cast<const _Float16 *>(A_in);

    // get argument data written by host
    auto host_arg = reinterpret_cast<volatile uint32_t *>(RAD_DEVICE_ARG_BASE);
    auto temp = *host_arg;

    auto Q_gmem = kernel_arg->addr_q;
    _Float16 *Q_smem = 0;
    copy_gmem_to_smem<64, 64>(Q_gmem, Q_smem, tid_in_threadblock,
                              threads_per_threadblock);

    auto rowmax_smem = reinterpret_cast<_Float16 *>(0x10000);
    rowmax<64, 64>(Q_smem, rowmax_smem, tid_in_threadblock,
                   threads_per_threadblock, threadblock_id);
}

int main() {
    FlashAttentionKernelArgs arg {
        .dim_seqlen = 4096,
        .dim_headdim = 128,
        .addr_q = &numpy_a_bin[0],
        .addr_k = reinterpret_cast<_Float16 *>(0x30000000), // FIXME temporary
        .addr_v = reinterpret_cast<_Float16 *>(0x40000000), // FIXME temporary
        .addr_o = reinterpret_cast<_Float16 *>(0x50000000), // FIXME temporary
    };

    // TODO: &arg may come from the CPU
    mu_schedule(kernel_entry, &arg);

    return 0;
}
