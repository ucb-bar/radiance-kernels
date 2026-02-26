#include <radiance.h>
#include <mu_schedule.h>
#include "flash_impl.hpp"

#include "include/matmul_data.h"

extern const _Float16 numpy_a_bin[];
extern const _Float16 numpy_b_bin[];

void kernel_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock, uint32_t threadblock_id) {
    constexpr int DIM = 16;
    // const auto tid = mu_mhartid; // TODO

    _Float16 result[16] = {0};
    // auto tensor = reinterpret_cast<const _Float16 *>(A_in);

    // get argument data written by host
    auto host_arg = reinterpret_cast<volatile uint32_t *>(RAD_DEVICE_ARG_BASE);
    auto temp = *host_arg;

    auto dest_gmem = reinterpret_cast<volatile _Float16 *>(0x20000000); // yolo
    copy_gmem_to_smem<DIM, DIM>(0, dest_gmem, tid_in_threadblock,
                                threads_per_threadblock);

    // rowmax<1024, 64>(numpy_a_bin, result, 0, 0, 0);
    // rowmax<1024, 64>(0, result, 0, 0, 0);
}

int main() {
    int void_arg = 42;

    mu_schedule(kernel_entry, &void_arg);

    return void_arg;
}
