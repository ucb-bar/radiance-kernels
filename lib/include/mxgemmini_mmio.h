/** mxgemmini_mmio.h
 *  ================
 *
 *  Macro definitions that implement MMIO interface between Muon - MxGemmini.
 *  Include it *after* gemmini.h so that it correctly overrides APIs such as
 *  ROCC_INSTRUCTION_RS1_RS2.
 *
 *  Note that all of these addresses are within the GPU-local address space,
 *  different from the CPU-global space.
 *
 *  @cleanup: These only differ marginally from gemmini_mmio.h.  Merging the
 *  two would be ideal.
 */

#include <mu_intrinsics.h>
#include <type_traits>

#define GEMMINI_SF_MEM    0x00088000
#define GEMMINI_SF_MEM_A (GEMMINI_SF_MEM + 0x2000)
#define GEMMINI_SF_MEM_B (GEMMINI_SF_MEM)

#define GEMMINI_CTRL        0x00084000
#define GEMMINI_RS1_OFFSET  0x10
#define GEMMINI_RS2_OFFSET  0x18
#define GEMMINI_INST_OFFSET 0x0
#define GEMMINI_RS1_ADDR   (GEMMINI_CTRL + GEMMINI_RS1_OFFSET)
#define GEMMINI_RS2_ADDR   (GEMMINI_CTRL + GEMMINI_RS2_OFFSET)
#define GEMMINI_INST_ADDR  (GEMMINI_CTRL + GEMMINI_INST_OFFSET)

template <typename T>
inline uint64_t gemmini_arg_to_u64(T value) {
    if constexpr (std::is_pointer_v<T>) {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(value));
    } else {
        static_assert(std::is_integral_v<T> || std::is_enum_v<T>,
                      "Gemmini argument must be integral/enum or pointer");
        return static_cast<uint64_t>(value);
    }
}

#undef ROCC_INSTRUCTION_RS1_RS2
#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) { \
    store64_shared(GEMMINI_CTRL, GEMMINI_RS1_OFFSET, gemmini_arg_to_u64(rs1)); \
    store64_shared(GEMMINI_CTRL, GEMMINI_RS2_OFFSET, gemmini_arg_to_u64(rs2)); \
    store_shared  (GEMMINI_CTRL, GEMMINI_INST_OFFSET, (0x7B) | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct) << 25)); \
}
