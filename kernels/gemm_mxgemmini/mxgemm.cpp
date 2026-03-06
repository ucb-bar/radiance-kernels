#include <stdint.h>
#include <radiance.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
#include "include/matmul_data_mx_fp8.h"
#include "mxgemmini_mmio.h"

constexpr auto TILE_M = 64;
constexpr auto TILE_N = 64;
constexpr auto TILE_K = 64;
constexpr auto PE_TILES_I = TILE_M / DIM;
constexpr auto PE_TILES_J = TILE_N / DIM;
constexpr auto PE_TILES_K = TILE_K / DIM;

// TODO: cleanup
typedef uint8_t elem_t;   // A_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint8_t welem_t;  // B_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint64_t  out_t;    // C_scaled: fp8:e4m3 (1 byte per output)

// global section that will store fp8 outputs from Gemmini
static uint64_t C_out_got[DIM][DIM] = {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF,};

static inline void load_scale_factors(volatile uint32_t *sf_mem, const uint8_t *scale_factors, int n) {
    auto word_scale_factors = reinterpret_cast<const uint32_t *>(scale_factors);
    for (size_t i = 0; i < n / 4; i ++) {
        // do full-word stores instead of 1-byte stores
        store_shared(reinterpret_cast<uint32_t>(&sf_mem[i]), 0, word_scale_factors[i]);
    }
}

static inline void configure_mxgemmini() {
    gemmini_flush(0);
    gemmini_extended3_config_ex(
        WEIGHT_STATIONARY, // dataflow
        0, 0, ACC_SCALE_IDENTITY, // sys_act, sys_shift, sys_acc_scale
        1, 1, // C_stride, A_stride
        0, 0, // A_transpose, B_transpose
        false, // set_only:strides
        0, // act_mx_fmt
        0, // wgt_mx_fmt
        3, // out_mx_fmt
        0  // uselut
    );

    // Configure move-in strides for B and A
    // gemmini_config_ld(TILE_M * sizeof(elem_t)); // nicolas's
    gemmini_extended3_config_ld(TILE_K * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 0);
    gemmini_extended3_config_ld(TILE_N * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 1);

    // Configure move-out for C
    // We want 1 byte per output element in DRAM
    // gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1); // nicolas's
    gemmini_extended_config_st(TILE_N * sizeof(elem_t), NO_ACTIVATION, 1 /*TODO: what is this?*/);

    // wait for configuration finish
    gemmini_fence();
}

static inline void copy_a_b_gmem_to_smem(const uint32_t tile_i,
                                         const uint32_t tile_j,
                                         const uint32_t tile_k) {
  // Gemmini expects the full A/B tensor to be stored in block-level
  // row-major layout, i.e.:
  // The tensor is partitioned into DIM x DIM tiles.
  // Tiles are ordered row-by-row in memory (all tile columns of tile-row 0,
  // then tile-row 1, etc.), and each tile is stored contiguously.

#define GEMMINI_DMA 1
#if GEMMINI_DMA
    // config GMEM address for A and B
    // inst: 0x1420b07b
    ROCC_INSTRUCTION_RS1_RS2(
        XCUSTOM_ACC,
        rad_device_to_host_address(reinterpret_cast<uint32_t>(A_in)),
        rad_device_to_host_address(reinterpret_cast<uint32_t>(B_in)),
        k_LOOP_WS_CONFIG_ADDRS_AB)

    // gemmini_fence();

    // TODO: config DRAM strides
    // inst: 0x1820b07b
    ROCC_INSTRUCTION_RS1_RS2(
        XCUSTOM_ACC,
        (uint64_t)(TILE_K),
        (uint64_t)(TILE_N),
        k_LOOP_WS_CONFIG_STRIDES_AB)

    // gemmini_fence();

    // gemmini_loop_ws_spad issues three instructions:
    //
    // 1. configure loop bounds (inst: 0x1220b07b, funct: k_LOOP_WS_CONFIG_BOUNDS)
    // 2. configure spad addresses (inst: 0x3020b07b, funct: k_LOOP_WS_CONFIG_SPAD_AB)
    // 3. compute loop ws with skips (inst: 0x1020b07b, funct: k_LOOP_WS)
    constexpr uint32_t skips_mvin =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);
    constexpr auto DONTCARE = 0;

    gemmini_loop_ws_spad(
        PE_TILES_I, PE_TILES_J, PE_TILES_K, // loop bounds for I, J, K (single 16×16 PE tile)
        0, 0, 0,              // pad_I=0, pad_J=0, pad_K=0
        0 * DIM,              // A scratchpad address
        BANK_NUM * BANK_ROWS, // B scratchpad address
        0,                    // D (bias) - none
        DONTCARE,             // C accumulator address
        false, false,         // A_transpose, B_transpose
        false, false, false,  // full_C, low_D, ex_accumulate
        NO_ACTIVATION,        // activation
        0, 0,                 // a_spad_id, b_spad_id
        false,                // is_resadd
        skips_mvin);        // skips

#else

    // Gemmini spad address is in DIM elems (16 fp8), not bytes
    const uint32_t a_base = 0;
    // Put B at the end of the spad
    const uint32_t b_base = BANK_NUM * BANK_ROWS - PE_TILES_K * PE_TILES_J * DIM;

    // A layout: for each i row, store all k tiles contiguously
    // A tile (i,k) -> a_base + (i * tiles_K + k) * DIM
    for (int i = 0; i < tiles_I; i++) {
        for (int k = 0; k < tiles_K; k++) {
            elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * TILE_M + k * DIM;
            uint32_t sp_addr = a_base + (i * PE_TILES_K + k) * DIM;
            // Note gemmini needs CPU-global addresses for mvin
            gemmini_extended_mvin(rad_device_to_host_address(reinterpret_cast<uint32_t>(dram_ptr)), sp_addr, DIM, DIM);
            gemmini_fence();
        }
    }

    // B layout: for each k row, store all j tiles contiguously
    // B tile (k,j) -> b_base + (k * tiles_J + j) * DIM
    for (int j = 0; j < PE_TILES_J; j++) {
        for (int k = 0; k < PE_TILES_K; k++) {
            elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * TILE_M + k * DIM;
            uint32_t sp_addr = b_base + (j * PE_TILES_K + k) * DIM;
            gemmini_extended_mvin(rad_device_to_host_address(reinterpret_cast<uint32_t>(dram_ptr)), sp_addr, DIM, DIM);
            gemmini_fence();
        }
    }
#endif
}

static inline void matmul_smem_tile() {
    uint32_t acc_addr = (1u << (ADDR_LEN - 1));
    gemmini_loop_ws_spad(
        PE_TILES_I, PE_TILES_J, PE_TILES_K, // loop bounds for I, J, K (single 16×16 PE tile)
        0, 0, 0,              // pad_I=0, pad_J=0, pad_K=0
        0 * DIM,              // A scratchpad address
        BANK_NUM * BANK_ROWS, // B scratchpad address
        0,                    // D (bias) - none
        acc_addr,             // C accumulator address
        false, false,         // A_transpose, B_transpose
        false, false, false,  // full_C, low_D, ex_accumulate
        NO_ACTIVATION,        // activation
        0, 0,                 // a_spad_id, b_spad_id
        false,                // is_resadd
        0x38);                // skips
}

void mxgemm(void *arg, uint32_t tid_in_threadblock,
            uint32_t threads_per_threadblock,
            uint32_t threadblock_id) {
    if (!(tid_in_threadblock == 0 && threadblock_id == 0)) {
        return;
    }

    static out_t C_hw[DIM][DIM] = {0};  // fp8 outputs from HW

    configure_mxgemmini();

    // Load scaling factors from GMEM to the scale SRAM
    // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_A), &A_scales_row[0][0], 32);
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_A), &A_scales_row[0][0], 32);
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_B), &B_scales_col[0][0], 32);
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_B), &B_scales_col[0][0], 32);

    // initiate pipeline
    int tile_k = 0;
    // copy_a_b_gmem_to_smem(0, 0/*FIXME*/, tile_k);

    // // wait for GMEM->SMEM copy
    // gemmini_fence();

    const uint32_t dim_k = 512; // FIXME export to somewhere
    for (; (tile_k * TILE_K) < dim_k; tile_k++) {
        // NOTE: This results in an unnecessary move-in at the last K tile
        copy_a_b_gmem_to_smem(0, 0, tile_k);

        // wait for compute and GMEM->SMEM copy
        gemmini_fence();

        matmul_smem_tile();
    }

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
    mu_schedule(mxgemm, nullptr);
    return 0;
}
