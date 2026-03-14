#include <stdint.h>
#include <type_traits>
#include <mu_intrinsics.h>
#include "lib.h"

int main() {
    vx_tmc(-1);

    constexpr auto N = 1 << 18;
    auto base = reinterpret_cast<volatile __global uint32_t *>(0x40000000);
    store_gmem_coalesced<N>(base);

    return 0;
}
