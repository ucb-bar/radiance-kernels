/** mxgemmini_mmio.h
 *  ================
 *
 *  Macro definitions that implement MMIO interface between Muon<->MxGemmini.
 *  Include it *after* gemmini.h so that it correctly overrides APIs such as
 *  ROCC_INSTRUCTION_RS1_RS2.
 *
 *  These are meant to be used from the Muon kernel, and as such the addresses
 *  are within the GPU-local address space.
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
#define GEMMINI_INST_OFFSET 0x0
#define GEMMINI_RS1_OFFSET  0x10
#define GEMMINI_RS2_OFFSET  0x18
#define GEMMINI_BUSY_OFFSET  0x20
#define GEMMINI_OCCUPANCY_OFFSET  0x28
#define GEMMINI_INST_ADDR  (GEMMINI_CTRL + GEMMINI_INST_OFFSET)
#define GEMMINI_RS1_ADDR   (GEMMINI_CTRL + GEMMINI_RS1_OFFSET)
#define GEMMINI_RS2_ADDR   (GEMMINI_CTRL + GEMMINI_RS2_OFFSET)
#define GEMMINI_BUSY_ADDR  (GEMMINI_CTRL + GEMMINI_BUSY_OFFSET)
#define GEMMINI_OCCUPANCY_ADDR (GEMMINI_CTRL + GEMMINI_OCCUPANCY_OFFSET)

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

// synchronization
// ---------------

#define gemmini_status() ({uint32_t status; asm volatile ("csrr %0, 0xacc" : "=r" (status)); status;})
#undef gemmini_fence
inline void gemmini_fence() {
    while (load32_shared(GEMMINI_BUSY_ADDR) != 0) {
        asm volatile("nop");
    }
}
inline void gemmini_fence_outstanding(const int n) {
    while (load32_shared(GEMMINI_OCCUPANCY_ADDR) > n) {
        asm volatile("nop");
    }
}

// MMIO helpers
// ------------

#define loop_matmul_skips(skip_lda, skip_ldb, skip_ldd, skip_ex, skip_stc) \
  (((skip_lda) | ((skip_ldb) << 1) | ((skip_ldd) << 2) | ((skip_ex) << 3) | ((skip_stc) << 4)) << 3)

#define sp_tiled_matmul_full_spad_ws(A_sp_addr_start, B_sp_addr_start, D_sp_addr_start, C_dst_sp_addr_start,\
  I, J, K, pad_I, pad_J, pad_K, a_transpose, b_transpose, full_C, low_D, acc, act, skips) \
  gemmini_loop_ws_spad(I, J, K, pad_I, pad_J, pad_K, A_sp_addr_start, (B_sp_addr_start) + (K) * (J) * DIM, NULL, \
  C_dst_sp_addr_start, a_transpose, b_transpose, full_C, low_D, acc, act, 0, 0, false, skips)
