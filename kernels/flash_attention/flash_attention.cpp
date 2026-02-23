#include <radiance.h>
#include "flash_impl.hpp"

#include "include/matmul_data.h"

extern const _Float16 numpy_a_bin[];
extern const _Float16 numpy_b_bin[];

// TODO: use scheduler entry point
int entry() {
    vx_tmc(-1);

    constexpr int DIM = 16;

    _Float16 result[16] = {0};
    auto tensor = reinterpret_cast<const _Float16 *>(A_in);

    // get argument data written by host
    auto arg = reinterpret_cast<volatile uint32_t *>(RAD_DEVICE_ARG_BASE);
    auto temp = *arg;

    // rowmax<1024, 64>(numpy_a_bin, result, 0, 0, 0);
    rowmax<1024, 64>(0, result, 0, 0, 0);

    vx_tmc(1);
    vx_tmc(0);
    return static_cast<int>(result[0]);
}

int main() {
    // vx_wspawn(8, (void (*)())entry);
    int result = entry();
    return result;
}
