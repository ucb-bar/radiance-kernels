#include <stdint.h>
#include <radiance.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
#include "include/matmul_data_mx_fp8.h"
#include "mxgemmini_mmio.h"

// Tiling parameters -----------------------------------------------------------

constexpr auto TILE_M = 64;
constexpr auto TILE_N = 64;
constexpr auto TILE_K = 64;
constexpr auto PE_TILES_I = TILE_M / DIM;
constexpr auto PE_TILES_J = TILE_N / DIM;
constexpr auto PE_TILES_K = TILE_K / DIM;
constexpr auto SCALE_FAC_DIM = TILE_M * TILE_K / 32;

// Performance benchmark options -----------------------------------------------
// disable GMEM->SMEM DMA copy and have MxGemmini work on stale data in SMEM
constexpr bool DISABLE_DMA_MOVE_IN = false;
// disable scale-factor write & fence from the GPU
constexpr bool DISABLE_SCALE_FACTOR_UPDATE = false;


// TODO: cleanup
typedef uint8_t elem_t;   // A_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint8_t welem_t;  // B_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint64_t  out_t;    // C_scaled: fp8:e4m3 (1 byte per output)

// global section that will store fp8 outputs from Gemmini
static uint64_t C_out_got[DIM][DIM] = {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF,};

static void __attribute__((noinline))
load_scale_factors(volatile uint32_t *sf_mem, const uint8_t *scale_factors,
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
            store_shared(reinterpret_cast<uint32_t>(&sf_mem[i + j]), 0, unrolled[j]);
        }
    }
    // asm volatile ("load_scale_factors_end_%=:" :: );
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
    // We want 1 byte per output element in GMEM
    // gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1); // nicolas's
    gemmini_extended_config_st(TILE_N * sizeof(elem_t), NO_ACTIVATION, 1 /*TODO: what is this?*/);

    // Configure loop bounds for the FSM
    // This only needs to be done once since the kernel does not change the
    // SMEM tile size
    gemmini_loop_ws_config_bounds(
        PE_TILES_I, PE_TILES_J, PE_TILES_K, // loop bounds for I, J, K (single 16×16 PE tile)
        0, 0, 0 // pad_I=0, pad_J=0, pad_K=0
    );

    // wait for configuration finish
    gemmini_fence();

    // NOTE: we need to run this twice to configure the two FSMs
    gemmini_loop_ws_config_bounds(
        PE_TILES_I, PE_TILES_J, PE_TILES_K, // loop bounds for I, J, K (single 16×16 PE tile)
        0, 0, 0 // pad_I=0, pad_J=0, pad_K=0
    );
    // wait for configuration finish
    gemmini_fence();
}

static inline void copy_gmem_to_smem_async(const uint32_t tile_i,
                                           const uint32_t tile_j,
                                           const uint32_t tile_k /* FIXME: unused */) {
    asm volatile ("copy_gmem_to_smem_async_start_%=:" :: );

    // Gemmini expects the full A/B tensor to be stored in block-level
    // row-major layout, i.e.:
    // The tensor is partitioned into DIM x DIM tiles.
    // Tiles are ordered row-by-row in memory (all tile columns of tile-row 0,
    // then tile-row 1, etc.), and each tile is stored contiguously.

#define GEMMINI_DMA 1
#if GEMMINI_DMA
    // Configure GMEM address for A and B
    // inst: 0x1420b07b
    ROCC_INSTRUCTION_RS1_RS2(
        XCUSTOM_ACC,
        rad_device_to_host_address(reinterpret_cast<uint32_t>(A_in)),
        rad_device_to_host_address(reinterpret_cast<uint32_t>(B_in)),
        k_LOOP_WS_CONFIG_ADDRS_AB)

    // Configure loop FSM GMEM move-in strides for A and B
    // This only needs to be done once since the kernel does not change the
    // SMEM tile size
    // FIXME: Moving this out of the loop breaks addresses?
    ROCC_INSTRUCTION_RS1_RS2(
        XCUSTOM_ACC,
        (uint64_t)(TILE_K), // TODO: this might be wrong
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
        0 * DIM,              // A scratchpad address (TODO: double-buffer)
        BANK_NUM * BANK_ROWS, // B scratchpad address (TODO: double-buffer)
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

    asm volatile ("copy_gmem_to_smem_async_end_%=:" :: );
}

/** Asynchronously kick off loop FSM matmul compute operation in MxGemmini.
 *  Move out accumulator data to SMEM if `acc_move_out` is true.
 *  TODO: double-buffer */
static inline void matmul_tile_async(const bool acc_move_out) {
    asm volatile ("matmul_tile_async_start_%=:" :: );

    const uint32_t skip_stc = acc_move_out ? 0 : 1;
    const uint32_t skips_compute =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/0, /*skip_stc=*/skip_stc);
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
        skips_compute);       // skips

    asm volatile ("matmul_tile_async_end_%=:" :: );
}

void mxgemm(void *arg, uint32_t tid_in_threadblock,
            uint32_t threads_per_threadblock,
            uint32_t threadblock_id) {
    if (tid_in_threadblock != 0) {
        return;
    }

    // FIXME
    static uint32_t scale_factors[16] = {
        0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000,  // e.g. 1.0f in IEEE754
        0x40000000, 0x40000000, 0x40000000, 0x40000000,  // e.g. 2.0f
        0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000,  // e.g. 0.5f
        0x3FC00000, 0x3FC00000, 0x3FC00000, 0x3FC00000,  // e.g. 1.5f
    };

    static out_t C_hw[DIM][DIM] = {0};  // fp8 outputs from HW

    const uint32_t dim_k = 512; // FIXME export to somewhere

    configure_mxgemmini();

    // -----------------
    // Initiate pipeline
    // -----------------
    //
    int tile_k = 0;
    copy_gmem_to_smem_async(0, 0/*FIXME*/, tile_k);

    // wait for GMEM->SMEM copy
    // gemmini_fence();

    // Load scaling factors from GMEM to the scale SRAM
    // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_A), &A_scales_row[0][0], SCALE_FAC_DIM);
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_A), &A_scales_row[0][0], SCALE_FAC_DIM);
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_B), &B_scales_col[0][0], SCALE_FAC_DIM);
    load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_B), &B_scales_col[0][0], SCALE_FAC_DIM);

    // wait for scale factor write
    mu_fence();

    // ------------------------------
    // Main software-pipelined K-loop
    // ------------------------------
    //
    asm volatile ("main_matmul_k_loop_start_%=:" :: );

    //                 ┌───┐   ┌───┐
    //             ┌───────────┐
    //         ┌───────┐
    // Loop 1: M0->M1->C0->M0->C1->M1->C0
    //         ┌───┐   ┌───┐   ┌───┐
    // Loop 2: M0->C0->M1->C1->M0->C0->...
    //
    // FIXME: below is not possible in current fence_ready():
    // Operations of the same type are serialized (e.g. M0->M1->M2); therefore
    // no need to fence against previous same-double-buffer op.
    for (; (tile_k * TILE_K) < dim_k; tile_k++) {
        const auto last_k = ((tile_k + 1) * TILE_K) >= dim_k;

        // TODO: This results in an unnecessary move-in at the last K tile
        if constexpr (!DISABLE_DMA_MOVE_IN) {
            // gemmini_fence_ready();
            copy_gmem_to_smem_async(0, 0, tile_k + 1);
        }

        // TODO: SMEM tile must be double-buffered
        // gemmini_fence_ready();
        matmul_tile_async(last_k);

        // update scale factors between async kickoff & fence to hide latency
        // FIXME: fix double-buffer index
        if constexpr (!DISABLE_SCALE_FACTOR_UPDATE) {
            load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_A), &A_scales_row[0][0], SCALE_FAC_DIM);
            load_scale_factors(reinterpret_cast<uint32_t *>(GEMMINI_SF_MEM_B), &B_scales_col[0][0], SCALE_FAC_DIM);

            // ensure scale factor write -> mxgemmini kickoff ordering
            mu_fence();

            // configure scalefac->PE double-buffer read; inst: 0x3420b07b
            gemmini_mxquant_config_mvout(
                rad_device_to_host_address(0x20000000) /*FIXME*/,
                PE_TILES_I, PE_TILES_J, PE_TILES_K, 0 /*FIXME*/, 0);
        }

        // TODO: add LUT loading

        gemmini_fence();
    }

    gemmini_fence();

    asm volatile ("main_matmul_k_loop_end_%=:" :: );

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
    mu_schedule(mxgemm, nullptr, 2);
    return 0;
}
