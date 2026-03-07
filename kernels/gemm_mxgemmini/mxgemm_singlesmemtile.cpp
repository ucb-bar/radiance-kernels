#include <stdint.h>
#include <radiance.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
#include "include/matmul_data_mx_fp8.h"
#include "mxgemmini_mmio.h"

// TODO: cleanup
typedef uint8_t elem_t;   // A_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint8_t welem_t;  // B_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint64_t  out_t;    // C_scaled: fp8:e4m3 (1 byte per output)

// global section that will store fp8 outputs from Gemmini
static uint64_t C_out_got[DIM][DIM] = {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF,};

static void load_scale_factors(volatile uint32_t *sf_mem, const uint8_t *scale_factors, int n) {
    auto word_scale_factors = reinterpret_cast<const uint32_t *>(scale_factors);
    for (size_t i = 0; i < n / 4; i ++) {
        // do full-word stores instead of 1-byte stores
        store_shared(reinterpret_cast<uint32_t>(&sf_mem[i]), 0, word_scale_factors[i]);
    }
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

    // Load per-element scaling factors into the scale SRAM
    // (C_scale is uint8_t[DIM][DIM], packed row-major)
    // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_A), &A_scales_row[0][0], 32);
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_A), &A_scales_row[0][0], 32);
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_B), &B_scales_col[0][0], 32);
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_B), &B_scales_col[0][0], 32);

    // MVIN B and A
    gemmini_config_ld(MATMUL_M * sizeof(elem_t));

    int tiles_I = MATMUL_M / DIM;
    int tiles_J = MATMUL_N / DIM;
    int tiles_K = MATMUL_K / DIM;

    uint32_t a_base = 0;
    uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

    // A layout: for each i row, store all k tiles contiguously
    // A tile (i,k) -> a_base + (i * tiles_K + k) * DIM
    for (int i = 0; i < tiles_I; i++) {
        for (int k = 0; k < tiles_K; k++) {
            elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
            uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
            // Note gemmini needs CPU-global addresses for mvin
            gemmini_extended_mvin(rad_device_to_host_address(reinterpret_cast<uint32_t>(dram_ptr)), sp_addr, DIM, DIM);
        }
    }

    // B layout: for each k row, store all j tiles contiguously
    // B tile (k,j) -> b_base + (k * tiles_J + j) * DIM
    for (int j = 0; j < tiles_J; j++) {
        for (int k = 0; k < tiles_K; k++) {
            elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
            uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
            gemmini_extended_mvin(rad_device_to_host_address(reinterpret_cast<uint32_t>(dram_ptr)), sp_addr, DIM, DIM);
        }
    }

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
        2, 2, 2,              // I=1, J=1, K=1 (single 16×16 tile)
        0, 0, 0,              // pad_I=0, pad_J=0, pad_K=0
        0 * DIM,              // A scratchpad address
        BANK_NUM * BANK_ROWS,              // B scratchpad address
        0,                    // D (bias) - none
        acc_addr,             // C accumulator address
        false, false,         // A_transpose, B_transpose
        false, false, false,  // full_C, low_D, ex_accumulate
        NO_ACTIVATION,        // activation
        0, 0,                 // a_spad_id, b_spad_id
        false,                // is_resadd
        0x38);                // skips

    // gemmini_mvout_spad(0, acc_addr);

    gemmini_fence();

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
    mu_schedule(mxgemm, nullptr, vx_num_warps());
    return 0;
}
