#include <stdint.h>
#include <radiance.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
#include "include/matmul_data.h"
#include "mxgemmini_mmio.h"

// TODO: cleanup
typedef uint8_t elem_t;   // A_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint8_t welem_t;  // B_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint64_t  out_t;    // C_scaled: fp8:e4m3 (1 byte per output)

// global section that will store fp8 outputs from Gemmini
static uint64_t C_out_got[DIM][DIM] = {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF,};

inline void load_scale_factors(volatile uint64_t *sf_mem, const uint8_t *scale_factors, int n) {
    uint64_t *dword_scale_factors = (uint64_t *)scale_factors;
    for (size_t i = 0; i < n / 8; i ++) {
        // do wide stores instead of individual 1-byte stores
        store64_shared(reinterpret_cast<uint32_t>(&sf_mem[i]), 0, dword_scale_factors[i]);
    }
}

/** Convert GPU-local GMEM address to CPU-addressable global address */
inline uint64_t convert_to_global(uint32_t addr) {
    return (static_cast<uint64_t>(addr) | RAD_HOST_GPU_DRAM_BASE);
}

void mxgemm(void *arg, uint32_t tid_in_threadblock,
            uint32_t threads_per_threadblock,
            uint32_t threadblock_id) {
    if (!(tid_in_threadblock == 0 && threadblock_id == 0)) {
        return;
    }

    // ---- Buffers ----
    // Inputs come from MATMUL_DATA_H: A_in[16][16], B_in[16][16]
    static out_t C_hw[DIM][DIM] = {0};  // fp8 outputs from HW

    // ---------- Run Gemmini (fp8 WS test) ----------
    gemmini_flush(0);
    //  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

    // We want 1 byte per output element in DRAM
    gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);

#if 0
    // Load per-element scaling factors into the scale SRAM
    // (C_scale is uint8_t[DIM][DIM], packed row-major)
    // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, A_scales_row , 32);
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, B_scales_col , 32);
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, A_scales_row , 32);
    load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, B_scales_col , 32);
#endif

    // MVIN B and A
    // Note gemmini needs CPU-global addresses for mvin
    gemmini_config_ld(DIM * sizeof(elem_t));
    gemmini_mvin(convert_to_global(reinterpret_cast<uint32_t>(B_in)), 1 * DIM);
    gemmini_mvin(convert_to_global(reinterpret_cast<uint32_t>(A_in)), 0 * DIM);

    uint32_t acc_addr = (1u << (ADDR_LEN - 1));
    //  gemmini_preload(1 * DIM, acc_addr);  // Read B from spad addr 1*DIM, results -> acc_addr
    //
    //  // Compute: A (from spad 0*DIM) × B (preloaded) -> accumulator (at acc_addr)
    //  gemmini_config_ld(DIM * sizeof(elem_t));
    //  gemmini_compute_preloaded(0 * DIM, GARBAGE_ADDR);
    //
    //  uint32_t mvout_addr = acc_addr & ~(1 << (ADDR_LEN - 2));  // Clear accumulate bit
    //  mvout_addr |= (1 << 29);  // Set full row bit
    ////  gemmini_mvout((void *) C_hw, mvout_addr);
    //  gemmini_mvout_spad(0, acc_addr);

    gemmini_loop_ws_spad(
        1, 1, 1,              // I=1, J=1, K=1 (single 16×16 tile)
        0, 0, 0,              // pad_I=0, pad_J=0, pad_K=0
        0 * DIM,              // A scratchpad address
        2 * DIM,              // B scratchpad address
        0,                    // D (bias) - none
        acc_addr,             // C accumulator address
        false, false,         // A_transpose, B_transpose
        false, false, false,  // full_C, low_D, ex_accumulate
        NO_ACTIVATION,        // activation
        0, 0,                 // a_spad_id, b_spad_id
        false,                // is_resadd
        0x38);                // skips

    //  gemmini_mvout_spad(0, acc_addr);

    // Single fence at the end, like your fp6 test
    // gemmini_fence();

#if 0
    // read back C_out_got from DMEM to generate some traces
    volatile uint64_t sum = 0;
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            uint64_t got = C_out_got[i][j];
            sum += got;
        }
    }
#endif
}

int main() {
    mu_schedule(mxgemm, nullptr);
    return 0;
}
