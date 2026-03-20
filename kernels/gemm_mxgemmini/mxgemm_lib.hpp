#include <stdint.h>
#include <radiance.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
// #include "include/matmul_data_mx_fp8.h"
// #include "include/matmul_fp8_64x64.h"
#include "include/matmul_fp8_128x128x256.h"
// #include "include/matmul_fp8_128x128.h"
// #include "include/matmul_data_mx_lut_hw.h"
#include "mxgemmini_mmio.h"

// Tiling parameters -----------------------------------------------------------

constexpr auto GEMM_K = 1024;
constexpr auto TILE_M = 128;
constexpr auto TILE_N = 128;
constexpr auto TILE_K = 512;
constexpr auto FP4FP6 = true;
constexpr auto PE_M = (FP4FP6 ? 32 : 16);
constexpr auto PE_N = (FP4FP6 ? 32 : 16);
constexpr auto PE_K = (FP4FP6 ? 16 : 16);
constexpr auto PE_TILES_I = TILE_M / PE_M;
constexpr auto PE_TILES_J = TILE_N / PE_N;
constexpr auto PE_TILES_K = TILE_K / PE_K;
constexpr auto SCALE_FAC_DIM = TILE_M * TILE_K / 32; // TODO: TILE_N not differentiated
constexpr auto USE_LUT = FP4FP6;
constexpr auto VALUES_PER_BYTE = (FP4FP6 ? 2 : 1);
constexpr auto QUANT_LUT_UPDATE_GRANULARITY = 1;
static_assert(TILE_M == TILE_N, "currently only supports square SMEM tile dimensions");

constexpr auto GEMMINI_FORMAT_FP8 = 0;
constexpr auto GEMMINI_FORMAT_FP6 = 1;
constexpr auto GEMMINI_FORMAT_FP4 = 2;
constexpr auto GEMMINI_FORMAT_FULL = 3;

// Performance benchmark options -----------------------------------------------
constexpr bool GEMMINI_DMA = true;
// disable GMEM->SMEM DMA copy and have MxGemmini work on stale data in SMEM
constexpr bool DISABLE_DMA_MOVE_IN = false;
// disable scale-factor write & fence from the GPU
constexpr bool DISABLE_SCALE_FACTOR_UPDATE = false;

// TODO: cleanup
typedef uint64_t out_t;   // UNUSED: C_scaled: fp8:e4m3 (1 byte per output)

static uint32_t C_scale_factors[TILE_M * TILE_N / 32] __attribute__((aligned(32))) = {0};

static void __attribute__((noinline))
load_scale_factors(volatile __shared uint32_t *sf_mem, const uint8_t *scale_factors,
                   int n) {
    // asm volatile ("load_scale_factors_start_%=:" :: );
    auto word_scale_factors = reinterpret_cast<const uint32_t *>(scale_factors);

    // unroll in registers to reduce back-to-back WAW/WAR
    constexpr auto ILP = 8;
    uint32_t unrolled[ILP];
    #pragma unroll 4
    for (size_t i = 0; i < n / 4; i += ILP) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            // do full-word stores instead of 1-byte stores
            unrolled[j] = word_scale_factors[i + j];
        }
        for (int j = 0; j < ILP; j++) {
            sf_mem[i + j] = unrolled[j];
        }
    }
    // asm volatile ("load_scale_factors_end_%=:" :: );
}

static void load_lut_row(volatile __shared uint32_t *lut_mem, uint8_t *lut) {
    lut_mem[0] = (uint32_t)(lut[0] | (lut[1] << 6) | (lut[2] << 12) |
                            (lut[3] << 18) | (lut[4] << 24) | (lut[5] << 30));
    lut_mem[1] = (uint32_t)((lut[5] >> 2) | (lut[6] << 4) | (lut[7] << 10) |
                            (lut[8] << 16) | (lut[9] << 22) | (lut[10] << 28));
    lut_mem[2] = (uint32_t)((lut[10] >> 4) | (lut[11] << 2) | (lut[12] << 8) |
                            (lut[13] << 14) | (lut[14] << 20) | (lut[15] << 26));
}

static inline void configure_mxgemmini() {
    gemmini_flush(0);

    // TODO: FP4
    constexpr auto GEMMINI_FORMAT =
        FP4FP6 ? GEMMINI_FORMAT_FP6 : GEMMINI_FORMAT_FP8;

    gemmini_extended3_config_ex(
        WEIGHT_STATIONARY, // dataflow
        0, 0, ACC_SCALE_IDENTITY, // sys_act, sys_shift, sys_acc_scale
        1, 1, // C_stride, A_stride
        0, 0, // A_transpose, B_transpose
        false, // set_only:strides
        GEMMINI_FORMAT, // A dtype
        GEMMINI_FORMAT, // B dtype
        GEMMINI_FORMAT_FULL, // C dtype
        USE_LUT  // uselut
    );
    // ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,
    //   ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)
    //   | ((uint64_t)(1) << 16) // A stride
    //   // | (GEMMINI_FORMAT << 14) // C format
    //   | (3 << 14) // C format
    //   | (GEMMINI_FORMAT << 12) // B format
    //   | (GEMMINI_FORMAT << 10) // A format
    //   | (0 << 9) // B transpose
    //   | (0 << 8) // A transpose
    //   | ((false) << 7) // Set only strides
    //   | ((USE_LUT) << 4)
    //   | ((0) << 3) // Activation function
    //   | ((WEIGHT_STATIONARY) << 2)
    //   | CONFIG_EX,
    //   ((uint64_t)(1) << 48) // C stride
    //   | (0),
    //   k_CONFIG);

    // Configure GMEM move-in strides for A and B
    gemmini_extended3_config_ld(
        TILE_K * sizeof(uint8_t) / VALUES_PER_BYTE,
        MVIN_SCALE_IDENTITY, false, 0
    );
    gemmini_extended3_config_ld(
        TILE_N * sizeof(uint8_t) / VALUES_PER_BYTE,
        MVIN_SCALE_IDENTITY, false, 1
    );

    // Configure GMEM move-out stride for C
    // gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1); // nicolas's
    gemmini_config_st(
        TILE_N * sizeof(uint16_t) // FIXME: need to change by GEMMINI_FORMAT_FULL
    );

    // Configure scalefac->PE read and scalefac->GMEM write addresses; inst: 0x3420b07b
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        PE_TILES_I, PE_TILES_J, PE_TILES_K,
        0, // A double-buffer toggle
        0, // B double-buffer toggle
        QUANT_LUT_UPDATE_GRANULARITY);

    // Configure loop bounds for the loop FSM
    // This only needs to be done once since the kernel does not change the
    // SMEM tile size
    gemmini_loop_ws_config_bounds(
        PE_TILES_I, PE_TILES_J, PE_TILES_K,
        0, 0, 0 // pad_I=0, pad_J=0, pad_K=0
    );

    // wait for configuration finish
    gemmini_fence();

    // NOTE: we need to run this twice to configure the two FSMs
    gemmini_loop_ws_config_bounds(
        PE_TILES_I, PE_TILES_J, PE_TILES_K,
        0, 0, 0 // pad_I=0, pad_J=0, pad_K=0
    );
    // wait for configuration finish
    gemmini_fence();
}

/** Calculate scratchpad row address for A if `is_b == false` or B if `is_b == true`. */
template <bool is_b>
static inline uint32_t calculate_spad_addr(const uint32_t tile_k) {
    constexpr auto SMEM_SIZE_ROWS = BANK_NUM * BANK_ROWS;
    constexpr auto SMEM_QUARTER_ROWS = SMEM_SIZE_ROWS / 4;
    static_assert(SMEM_QUARTER_ROWS != 0);
    constexpr auto A_SPAD_ADDR_EVEN = 0;
    constexpr auto A_SPAD_ADDR_ODD = SMEM_QUARTER_ROWS;
    // B spad address is counted from the end (SMEM_SIZE_ROWS)
    constexpr auto B_SPAD_ADDR_EVEN = SMEM_SIZE_ROWS;
    constexpr auto B_SPAD_ADDR_ODD = SMEM_SIZE_ROWS - SMEM_QUARTER_ROWS;

    const uint32_t odd_k = (tile_k & 1);
    const uint32_t a_spad_addr = odd_k ? A_SPAD_ADDR_ODD : A_SPAD_ADDR_EVEN;
    const uint32_t b_spad_addr = odd_k ? B_SPAD_ADDR_ODD : B_SPAD_ADDR_EVEN;

    if constexpr (is_b) {
        return b_spad_addr;
    } else {
        return a_spad_addr;
    }
}

template <bool is_b>
static inline __shared uint32_t *calculate_scale_factor_addr(const uint32_t tile_k) {
    const uint32_t odd_k = (tile_k & 1);
    const uint32_t dbuf_offset = odd_k ? GEMMINI_SF_MEM_BUFFER_SIZE : 0;
    auto a_sf_addr = reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_A + dbuf_offset);
    auto b_sf_addr = reinterpret_cast<__shared uint32_t *>(GEMMINI_SF_MEM_B + dbuf_offset);

    if constexpr (is_b) {
        return b_sf_addr;
    } else {
        return a_sf_addr;
    }
}

static inline void copy_gmem_to_smem_async(const uint32_t tile_i /* FIXME: unused */,
                                           const uint32_t tile_j /* FIXME: unused */,
                                           const uint32_t tile_k) {
    asm volatile ("copy_gmem_to_smem_async_start_%=:" :: );

    // Gemmini expects the full A/B tensor to be stored in block-level
    // row-major layout, i.e.:
    // The tensor is partitioned into DIM x DIM tiles.
    // Tiles are ordered row-by-row in memory (all tile columns of tile-row 0,
    // then tile-row 1, etc.), and each tile is stored contiguously.

    const uint32_t a_spad_addr_start = calculate_spad_addr<false>(tile_k);
    const uint32_t b_spad_addr_end = calculate_spad_addr<true>(tile_k);

    if constexpr (GEMMINI_DMA) {
        // Configure GMEM address for A and B
        // inst: 0x1420b07b
        ROCC_INSTRUCTION_RS1_RS2(
            XCUSTOM_ACC,
#if USE_LUT // FIXME: remove
            rad_device_to_host_address(reinterpret_cast<uint32_t>(A_in_hw)),
#else
            rad_device_to_host_address(reinterpret_cast<uint32_t>(A_in)),
#endif
            rad_device_to_host_address(reinterpret_cast<uint32_t>(B_in)),
            k_LOOP_WS_CONFIG_ADDRS_AB)

        // Configure loop FSM GMEM move-in strides for A and B
        // This only needs to be done once since the kernel does not change the
        // SMEM tile size
        // FIXME: However, moving this out of the loop breaks addresses?
        ROCC_INSTRUCTION_RS1_RS2(
            XCUSTOM_ACC,
            (uint64_t)(TILE_K),
            (uint64_t)(TILE_N),
            k_LOOP_WS_CONFIG_STRIDES_AB /* 0x1820b07b */)

        // gemmini_loop_ws_spad issues three instructions:
        //
        // 1. configure loop bounds (inst: 0x1220b07b, funct: k_LOOP_WS_CONFIG_BOUNDS)
        // 2. configure spad addresses (inst: 0x3020b07b, funct: k_LOOP_WS_CONFIG_SPAD_AB)
        // 3. compute loop ws with skips (inst: 0x1020b07b, funct: k_LOOP_WS)
        //
        // TODO: skip re-configuring of loop bounds
        constexpr uint32_t skips_mvin =
            loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                              /*skip_ex=*/1, /*skip_stc=*/1);
        constexpr auto DONTCARE = 0;

        gemmini_loop_ws_spad(
            PE_TILES_I, PE_TILES_J, PE_TILES_K, // loop bounds for I, J, K (single 16×16 PE tile)
            0, 0, 0,              // pad_I=0, pad_J=0, pad_K=0
            a_spad_addr_start,    // A scratchpad address in rows (grows upward)
            b_spad_addr_end,      // B scratchpad address in rows (grows downward)
            0,                    // D (bias) - none
            DONTCARE,             // C scratchpad address in rows
            false, false,         // A_transpose, B_transpose
            false, false, false,  // full_C, low_D, ex_accumulate
            NO_ACTIVATION,        // activation
            0, 0,                 // a_spad_id, b_spad_id
            false,                // is_resadd
            skips_mvin);        // skips

    } else { // GEMMINI_DMA

        gemmini_config_ld(
            TILE_K * sizeof(uint8_t) / VALUES_PER_BYTE
        );

        // A layout: for each i row, store all k tiles contiguously
        // A tile (i,k) -> a_base + (i * tiles_K + k) * DIM
        for (int i = 0; i < PE_TILES_I; i++) {
            for (int k = 0; k < PE_TILES_K; k++) {
                // FIXME: TILE_K may have to be GEMM_K
                const uint8_t *dram_ptr = ((uint8_t*)A_in) + i * DIM * TILE_K + k * DIM;
                const uint32_t sp_addr = a_spad_addr_start + (i * PE_TILES_K + k) * DIM;
                // Note gemmini needs CPU-global addresses for mvin
                gemmini_extended_mvin(rad_device_to_host_address(
                    reinterpret_cast<uint32_t>(dram_ptr)),
                                      sp_addr, DIM, DIM);
            }
        }

        gemmini_config_ld(
            TILE_N * sizeof(uint8_t) / VALUES_PER_BYTE
        );

        // B layout: for each k row, store all j tiles contiguously
        // B tile (k,j) -> b_base + (k * tiles_J + j) * DIM
        for (int k = 0; k < PE_TILES_K; k++) {
            for (int j = 0; j < PE_TILES_J; j++) {
                const uint8_t *dram_ptr = ((uint8_t*)B_in) + k * DIM * TILE_N + j * DIM;
                const uint32_t b_spad_addr_start = b_spad_addr_end - PE_TILES_K * PE_TILES_J * DIM;
                const uint32_t sp_addr = b_spad_addr_start + (k * PE_TILES_J + j) * DIM;
                gemmini_extended_mvin(rad_device_to_host_address(
                    reinterpret_cast<uint32_t>(dram_ptr)),
                                      sp_addr, DIM, DIM);
            }
        }
    }

    asm volatile ("copy_gmem_to_smem_async_end_%=:" :: );
}

/** Asynchronously kick off loop FSM matmul compute operation in MxGemmini.
 *  Move out accumulator data to SMEM if `acc_move_out` is true. */
static inline void matmul_tile_async(const uint32_t tile_k, const bool acc_move_out) {
    asm volatile ("matmul_tile_async_start_%=:" :: );

    const uint32_t skip_stc = acc_move_out ? 0 : 1;
    const uint32_t skips_compute =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/0, /*skip_stc=*/skip_stc);
    // uint32_t acc_addr = (1u << (ADDR_LEN - 1));
    const uint32_t SPAD_DEST = 256;

    const uint32_t a_spad_addr_start = calculate_spad_addr<false>(tile_k);
    const uint32_t b_spad_addr_end = calculate_spad_addr<true>(tile_k);

    gemmini_loop_ws_spad(
        PE_TILES_I, PE_TILES_J, PE_TILES_K, // loop bounds for I, J, K (single 16×16 PE tile)
        0, 0, 0,              // pad_I=0, pad_J=0, pad_K=0
        a_spad_addr_start,    // A scratchpad address in rows (grows upward)
        b_spad_addr_end,      // B scratchpad address in rows (grows downward)
        0,                    // D (bias) - none
        SPAD_DEST,            // C scratchpad address in rows
        false, false,         // A_transpose, B_transpose
        false, false, false,  // full_C, low_D, ex_accumulate
        NO_ACTIVATION,        // activation
        0, 0,                 // a_spad_id, b_spad_id
        false,                // is_resadd
        skips_compute);       // skips

    asm volatile ("matmul_tile_async_end_%=:" :: );
}

