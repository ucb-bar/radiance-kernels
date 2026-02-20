#include "flash_impl.hpp"

#include "include/matmul_data.h"

// TODO: use scheduler entry point
int entry() {
    vx_tmc(-1);

    constexpr int DIM = 16;

    auto tensor = reinterpret_cast<const _Float16 *>(A_in);
    _Float16 result[16] = {0};
    rowmax<1024, 4096>(tensor, result, 0, 0, 0);

    vx_tmc(1);
    vx_tmc(0);
    return static_cast<int>(result[0]);
}

int main() {
    vx_wspawn(8, (void (*)())entry);
    int result = entry();
    return result;
}
