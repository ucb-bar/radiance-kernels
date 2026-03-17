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
    const auto base = reinterpret_cast<const volatile __shared uint32_t *>(0x0);
    load_smem<N>(base);

    return 0;
}
