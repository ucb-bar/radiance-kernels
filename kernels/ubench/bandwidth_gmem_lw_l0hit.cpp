#include <stdint.h>
#include <type_traits>
#include <mu_intrinsics.h>
#include "lib.h"

int main() {
    vx_tmc(-1);

    // text region
    const auto base = reinterpret_cast<const volatile __global uint32_t *>(0x10000000);
    load_gmem_coalesced(base);

    vx_tmc(1);

    return 0;
}
