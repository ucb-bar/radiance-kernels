#include <stdint.h>
#include <type_traits>
#include <mu_intrinsics.h>
#include "lib.h"

int main() {
    if (vx_core_id() != 0) {
        return 0;
    }

    vx_tmc(-1);

    constexpr auto N = 1 << 18;
    // text region
    const auto base = reinterpret_cast<volatile __shared uint32_t *>(0x0);
    store_smem<N>(base, 0xdeadbeef);

    return 0;
}
