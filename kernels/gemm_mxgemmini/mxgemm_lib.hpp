#include <stdint.h>
#include <radiance.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
#include "mxgemmini_mmio.h"

// Tiling parameters -----------------------------------------------------------

struct GemmConfig {
    uint32_t TILE_M = 128;
    uint32_t TILE_N = 128;
    uint32_t TILE_K = 256;
    bool FP4FP6 = false;
    // quantize output to fp4/fp6/fp8
    bool QUANT_OUTPUT = false;

    constexpr uint32_t PE_M() const { return (FP4FP6 ? 32 : 16); }
    constexpr uint32_t PE_N() const { return (FP4FP6 ? 32 : 16); }
    constexpr uint32_t PE_K() const { return (FP4FP6 ? 16 : 16); }
    constexpr uint32_t PE_TILES_I() const { return TILE_M / PE_M(); }
    constexpr uint32_t PE_TILES_J() const { return TILE_N / PE_N(); }
    constexpr uint32_t PE_TILES_K() const { return TILE_K / PE_K(); }
    // TODO: TILE_N not differentiated
    constexpr uint32_t SCALE_FAC_DIM() const { return TILE_M * TILE_K / 32; }
    constexpr uint32_t VALUES_PER_BYTE() const { return (FP4FP6 ? 2 : 1); }
    // Size of each C element *after column-packing*.
    constexpr uint32_t OUT_ELEM_SIZE() const {
        // C FP4/FP6 elem-packing is along the M dimension, not N
        return (QUANT_OUTPUT ? sizeof(uint8_t) : sizeof(uint16_t));
    }
    constexpr uint32_t TILE_M_QUANT() const {
        return (QUANT_OUTPUT ? TILE_M / VALUES_PER_BYTE() : TILE_M);
    }
    constexpr uint32_t TILE_N_QUANT() const {
        // packing of N-dimension is already reflected in OUT_ELEM_SIZE()
        return TILE_N;
    }
    constexpr bool USE_LUT() const { return FP4FP6; }
};

// Gemmini constants -----------------------------------------------------------

constexpr auto GEMMINI_FORMAT_FP8 = 0;
constexpr auto GEMMINI_FORMAT_FP6 = 1;
constexpr auto GEMMINI_FORMAT_FP4 = 2;
constexpr auto GEMMINI_FORMAT_FULL = 3;
constexpr auto QUANT_LUT_UPDATE_GRANULARITY = 1;
constexpr auto GEMMINI_ACC_ADDR = (1u << (ADDR_LEN - 1));
constexpr auto SPAD_DEST = 256; // TODO: arbitrary

// Performance benchmark options -----------------------------------------------

constexpr bool GEMMINI_DMA = true;
// disable GMEM->SMEM DMA copy and have MxGemmini work on stale data in SMEM
constexpr bool DISABLE_DMA_MOVE_IN = false;
// disable scale-factor write & fence from the GPU
constexpr bool DISABLE_SCALE_FACTOR_UPDATE = false;
// disable C result tensor move-out from SMEM to GMEM
constexpr bool DISABLE_GMEM_MOVEOUT = false;
// use SIMT load/stores instead of DMA for C DMA move-out
// TODO: enabled to avoid current mvout bug in DMA
constexpr bool SIMT_GMEM_MOVEOUT = false;

// TODO: max size hardcoded
static uint32_t C_scale_factors[128 * 128 / 32] __attribute__((aligned(32))) = {0};

template <GemmConfig C>
static inline void configure_mxgemmini() {
    static_assert(C.TILE_M == C.TILE_N,
                  "currently only supports square SMEM tile dimensions");

    gemmini_flush(0);

    // TODO: FP4
    constexpr auto GEMMINI_FORMAT =
        C.FP4FP6 ? GEMMINI_FORMAT_FP6 : GEMMINI_FORMAT_FP8;
    constexpr auto GEMMINI_FORMAT_OUT =
        C.QUANT_OUTPUT ? GEMMINI_FORMAT : GEMMINI_FORMAT_FULL;

    gemmini_extended3_config_ex(
        WEIGHT_STATIONARY, // dataflow
        0, 0, ACC_SCALE_IDENTITY, // sys_act, sys_shift, sys_acc_scale
        1, 1, // C_stride, A_stride
        0, 0, // A_transpose, B_transpose
        false, // set_only:strides
        GEMMINI_FORMAT, // A dtype
        GEMMINI_FORMAT, // B dtype
        GEMMINI_FORMAT_OUT, // C dtype
        C.USE_LUT()  // uselut
    );

    // Configure GMEM move-in strides for A and B
    // NOTE: FP4/FP6 packs elements by M and N dimensions
    gemmini_extended3_config_ld(
        C.TILE_K * sizeof(uint8_t),
        MVIN_SCALE_IDENTITY, false, 0
    );
    gemmini_extended3_config_ld(
        C.TILE_N * sizeof(uint8_t) / C.VALUES_PER_BYTE(),
        MVIN_SCALE_IDENTITY, false, 1
    );

    // Configure GMEM move-out stride for C
    gemmini_config_st(C.TILE_N * C.OUT_ELEM_SIZE());

    // Configure scalefac->PE read and scalefac->GMEM write addresses; inst: 0x3420b07b
    gemmini_mxquant_config_mvout(
        rad_device_to_host_address(reinterpret_cast<uint32_t>(&C_scale_factors[0])),
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        0, // A double-buffer toggle
        0, // B double-buffer toggle
        QUANT_LUT_UPDATE_GRANULARITY);

    // Configure loop bounds for the loop FSM
    // This only needs to be done once since the kernel does not change the
    // SMEM tile size
    gemmini_loop_ws_config_bounds(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
        0, 0, 0 // pad_I=0, pad_J=0, pad_K=0
    );

    // wait for configuration finish
    gemmini_fence();

    // NOTE: we need to run this twice to configure the two FSMs
    gemmini_loop_ws_config_bounds(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
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

template <GemmConfig C>
static inline void load_lut() {
    asm volatile ("load_lut_start_%=:" :: );

    if constexpr (C.USE_LUT()) {
        // TODO: fix to use GEMM_MNK
        for (size_t i = 0; i < (C.TILE_N >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            auto *dst = reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT0_ADDR) + 3 * i;
            dst[0] = B_lut[i][0]; dst[1] = B_lut[i][1]; dst[2] = B_lut[i][2];
        }
        for (size_t i = 0; i < (C.TILE_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            auto *dst = reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT1_ADDR) + 3 * i;
            dst[0] = A_lut[i][0]; dst[1] = A_lut[i][1]; dst[2] = A_lut[i][2];
        }
        for (size_t i = 0; i < (C.TILE_M >> QUANT_LUT_UPDATE_GRANULARITY); i++) {
            auto *dst = reinterpret_cast<volatile __shared uint32_t *>(GEMMINI_LUT2_ADDR) + 3 * i;
            dst[0] = C_lut[i][0]; dst[1] = C_lut[i][1]; dst[2] = C_lut[i][2];
        }
    }

    asm volatile ("load_lut_end_%=:" :: );

}

template <GemmConfig C>
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
        // TODO: stride by tile_k
        // inst: 0x1420b07b
        ROCC_INSTRUCTION_RS1_RS2(
            XCUSTOM_ACC,
            rad_device_to_host_address(reinterpret_cast<uint32_t>(A_in)),
            rad_device_to_host_address(reinterpret_cast<uint32_t>(B_in)),
            k_LOOP_WS_CONFIG_ADDRS_AB)

        // Configure loop FSM GMEM move-in strides for A and B
        // This only needs to be done once since the kernel does not change the
        // SMEM tile size
        // FIXME: However, moving this out of the loop breaks addresses?
        ROCC_INSTRUCTION_RS1_RS2(
            XCUSTOM_ACC,
            (uint64_t)(C.TILE_K * sizeof(uint8_t)),
            (uint64_t)(C.TILE_N * sizeof(uint8_t) / C.VALUES_PER_BYTE()),
            k_LOOP_WS_CONFIG_STRIDES_AB /* 0x1820b07b */)

        // Kick off DMA move-in via the loop FSM
        //
        // gemmini_loop_ws_spad issues three instructions:
        //   1. configure loop bounds (inst: 0x1220b07b, funct: k_LOOP_WS_CONFIG_BOUNDS)
        //   2. configure spad addresses (inst: 0x3020b07b, funct: k_LOOP_WS_CONFIG_SPAD_AB)
        //   3. compute loop ws with skips (inst: 0x1020b07b, funct: k_LOOP_WS)
        // TODO: skip re-configuring of loop bounds
        constexpr uint32_t skips_mvin =
            loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                              /*skip_ex=*/1, /*skip_stc=*/1);
        constexpr auto DONTCARE = 0;
        gemmini_loop_ws_spad(
            C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), // loop bounds for I, J, K (single 16×16 PE tile)
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
            skips_mvin);          // skips

    } else { // !GEMMINI_DMA

        gemmini_config_ld(
            C.TILE_K * sizeof(uint8_t) / C.VALUES_PER_BYTE()
        );

        // A layout: for each i row, store all k tiles contiguously
        // A tile (i,k) -> a_base + (i * tiles_K + k) * DIM
        for (int i = 0; i < C.PE_TILES_I(); i++) {
            for (int k = 0; k < C.PE_TILES_K(); k++) {
                // FIXME: TILE_K may have to be GEMM_K
                const uint8_t *dram_ptr = ((uint8_t*)A_in) + i * DIM * C.TILE_K + k * DIM;
                const uint32_t sp_addr = a_spad_addr_start + (i * C.PE_TILES_K() + k) * DIM;
                // Note gemmini needs CPU-global addresses for mvin
                gemmini_extended_mvin(rad_device_to_host_address(
                    reinterpret_cast<uint32_t>(dram_ptr)),
                                      sp_addr, DIM, DIM);
            }
        }

        gemmini_config_ld(
            C.TILE_N * sizeof(uint8_t) / C.VALUES_PER_BYTE()
        );

        // B layout: for each k row, store all j tiles contiguously
        // B tile (k,j) -> b_base + (k * tiles_J + j) * DIM
        for (int k = 0; k < C.PE_TILES_K(); k++) {
            for (int j = 0; j < C.PE_TILES_J(); j++) {
                const uint8_t *dram_ptr = ((uint8_t*)B_in) + k * DIM * C.TILE_N + j * DIM;
                const uint32_t b_spad_addr_start = b_spad_addr_end - C.PE_TILES_K() * C.PE_TILES_J() * DIM;
                const uint32_t sp_addr = b_spad_addr_start + (k * C.PE_TILES_J() + j) * DIM;
                gemmini_extended_mvin(rad_device_to_host_address(
                    reinterpret_cast<uint32_t>(dram_ptr)),
                                      sp_addr, DIM, DIM);
            }
        }
    } // end !GEMMINI_DMA

    asm volatile ("copy_gmem_to_smem_async_end_%=:" :: );
}

/** Move tensor data from SMEM->GMEM using SIMT threads.
 *  Assumes row-major, packed layout (row stride == dim_col) for both src and dest.
 *  TODO: De-dup with FlashAttention */
template <uint32_t dim_row, uint32_t dim_col, uint32_t elem_size>
inline void copy_output_smem_to_gmem_simt(const __shared uint8_t *src_smem,
                                          uint8_t *dest_gmem,
                                          const uint32_t tid_in_threadblock,
                                          const uint32_t threads_per_threadblock) {
    asm volatile ("copy_output_smem_to_gmem_simt_start_%=:" :: );

    // Thread mapping: All warps in a threadblock cooperatively copies a
    // contiguous chunk of the same size as the threadblock per every "wave".
    // TODO: use GEMM_ instead of TILE_ here

    // Vectorize to 32-bit words for better throughput.
    auto *src_smem_vec = reinterpret_cast<const __shared uint32_t *>(src_smem);
    auto *dest_gmem_vec = reinterpret_cast<uint32_t *>(dest_gmem);
    static_assert((dim_row * dim_col * elem_size) % sizeof(uint32_t) == 0);
    const auto iter = dim_row * dim_col * elem_size / sizeof(uint32_t) /
                      threads_per_threadblock;

#pragma unroll 32
    for (int i = 0; i < iter; i++) {
        // simple uniform-strided access
        const auto index = (threads_per_threadblock) * i + tid_in_threadblock;
        const auto smem_addr = src_smem_vec + index;
        auto gmem_addr = dest_gmem_vec + index;
        *gmem_addr = *smem_addr;
    }

    asm volatile ("copy_output_smem_to_gmem_simt_end_%=:" :: );
}

/** Move tensor data from SMEM->GMEM using Gemmini DMA.
 *  `src_spad_addr` is in scratchpad row address. */
template <GemmConfig C>
static inline void copy_output_smem_to_gmem_async(const uint32_t src_spad_addr,
                                                  uint8_t *dest_gmem,
                                                  const uint32_t dim_n) {
    asm volatile ("copy_output_smem_to_gmem_async_start_%=:" :: );

    for (int i = 0; i < C.PE_TILES_I(); i++) {
        for (int j = 0; j < C.PE_TILES_J(); j++) {
            // TODO: stride is wrong for re-quantized output
            const uint32_t tile_spad_addr = src_spad_addr + (i * C.PE_TILES_J() + j) * DIM;
            // row-major layout
            // FIXME: hardcoded output dtype
            auto *dram_ptr = dest_gmem + (i * DIM * dim_n + j * DIM) * C.OUT_ELEM_SIZE();
            gemmini_mvout((void *) dram_ptr, tile_spad_addr);
        }
    }

    asm volatile ("copy_output_smem_to_gmem_async_end_%=:" :: );
}

/** Asynchronously kick off loop FSM matmul compute operation in MxGemmini.
 *  Move out accumulator data to SMEM if `acc_move_out` is true. */
template <GemmConfig C>
static inline void matmul_tile_async(const uint32_t tile_k, const bool acc_move_out) {
    asm volatile ("matmul_tile_async_start_%=:" :: );

    const uint32_t skip_stc = acc_move_out ? 0 : 1;
    const uint32_t skips_compute =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/0, /*skip_stc=*/skip_stc);

    const uint32_t a_spad_addr_start = calculate_spad_addr<false>(tile_k);
    const uint32_t b_spad_addr_end = calculate_spad_addr<true>(tile_k);

    const bool first_k = tile_k == 0;

    // TODO: support skipping move-out to SMEM
    // TODO(perf): !first_k creates a branch
    gemmini_loop_ws_spad(
        C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(), // loop bounds for I, J, K (single 16×16 PE tile)
        0, 0, 0,                // pad_I=0, pad_J=0, pad_K=0
        a_spad_addr_start,      // A scratchpad address in rows (grows upward)
        b_spad_addr_end,        // B scratchpad address in rows (grows downward)
        0,                      // D (bias) - none
        SPAD_DEST,              // C scratchpad address in rows
        false, false,           // A_transpose, B_transpose
        false, false, !first_k, // full_C, low_D, ex_accumulate
                                // only start in-mem accumulation after first k
        NO_ACTIVATION,          // activation
        0, 0,                   // a_spad_id, b_spad_id
        false,                  // is_resadd
        skips_compute);         // skips

    asm volatile ("matmul_tile_async_end_%=:" :: );
}

/** Do matmul on a single TILE_M * TILE_N output tile, accumulating over the
 *  full GEMM_K. */
template <GemmConfig C> void mxgemm_single_output_tile(const uint32_t dim_k) {
    configure_mxgemmini<C>();

    // -----------------
    // Initiate pipeline
    // -----------------
    //
    int tile_k = 0;
    // TODO: change 0's for multiple SMEM tiles
    copy_gmem_to_smem_async<C>(0, 0, tile_k);

    // Load scaling factors from GMEM to the scale SRAM
    // load_scale_factors((const uint64_t *) C_scale, sizeof(C_scale));
    load_scale_factors(calculate_scale_factor_addr<false>(tile_k), &A_scales_row[0][0], C.SCALE_FAC_DIM());
    load_scale_factors(calculate_scale_factor_addr<true> (tile_k), &B_scales_col[0][0], C.SCALE_FAC_DIM());

    load_lut<C>();

    // fence scale factor and LUT writes
    mu_fence_smem();

    // wait for GMEM->SMEM copy
    gemmini_fence();

    // ------------------------------
    // Main software-pipelined K-loop
    // ------------------------------
    //
    asm volatile ("main_matmul_k_loop_start_%=:" :: );

    // Potential software-pipelining loop structures:
    //                 ┌───┐   ┌───┐
    //             ┌───────────┐
    //         ┌───────┐
    // Loop 1: M0->M1->C0->M0->C1->M1->C0
    //         ┌───┐   ┌───┐   ┌───┐
    // Loop 2: M0->C0->M1->C1->M0->C0->...
    //
    for (; (tile_k * C.TILE_K) < dim_k; tile_k++) {
        const auto last_k = ((tile_k + 1) * C.TILE_K) >= dim_k;
        const auto odd_k = (tile_k & 1);

        // GMEM->SMEM DMA for the next tile_k
        // TODO: This results in an unnecessary move-in at the last K tile
        if constexpr (!DISABLE_DMA_MOVE_IN) {
            // gemmini_fence_ready();
            copy_gmem_to_smem_async<C>(0, 0, tile_k + 1);
        }

        // asynchrously kick off matmul for this tile_k
        // gemmini_fence_ready();
        matmul_tile_async<C>(tile_k, last_k);

        // update scale factors for the next tile_k
        // make sure to place this between tile_async and fence to hide latency
        if constexpr (!DISABLE_SCALE_FACTOR_UPDATE) {
            load_scale_factors(calculate_scale_factor_addr<false>(tile_k + 1),
                               &A_scales_row[0][0], C.SCALE_FAC_DIM());
            load_scale_factors(calculate_scale_factor_addr<true>(tile_k + 1),
                               &B_scales_col[0][0], C.SCALE_FAC_DIM());

            // fence scale factor and LUT writes before next Gemmini compute
            mu_fence_smem();

            // configure scalefac->PE double-buffer read; inst: 0x3420b07b
            gemmini_mxquant_config_mvout(
                // TODO: dummy move-out space for the scale factor
                rad_device_to_host_address(
                    reinterpret_cast<uint32_t>(&C_scale_factors[0])),
                C.PE_TILES_I(), C.PE_TILES_J(), C.PE_TILES_K(),
                odd_k, // A double-buffer toggle
                odd_k, // B double-buffer toggle
                QUANT_LUT_UPDATE_GRANULARITY);
        }

        gemmini_fence();
    }

    gemmini_fence();

    asm volatile ("main_matmul_k_loop_end_%=:" :: );
}

/** Do a full GEMM and store the result C tensor at `C_gmem` GMEM address. */
template <GemmConfig C>
static inline void
mxgemm(const uint32_t dim_m, const uint32_t dim_n, const uint32_t dim_k,
       uint8_t *C_gmem, const uint32_t tid_in_threadblock,
       const uint32_t threads_per_threadblock, const uint32_t threadblock_id) {
    if (tid_in_threadblock == 0) {
        mxgemm_single_output_tile<C>(dim_k);
    }

    const auto warps_per_threadblock = threads_per_threadblock / MU_NUM_THREADS;
    mu_barrier(0, warps_per_threadblock);

    // Move-out C from SMEM to GMEM
    if constexpr (!DISABLE_GMEM_MOVEOUT) {
        auto C_smem =
            reinterpret_cast<const __shared uint8_t *>(SPAD_DEST * DIM);
        if constexpr (SIMT_GMEM_MOVEOUT) {
            copy_output_smem_to_gmem_simt<C.TILE_M_QUANT(), C.TILE_N_QUANT(),
                                          C.OUT_ELEM_SIZE()>(
                C_smem, C_gmem, tid_in_threadblock, threads_per_threadblock);
        } else {
            if (tid_in_threadblock == 0) {
                copy_output_smem_to_gmem_async<C>(SPAD_DEST, C_gmem, dim_n);
                gemmini_fence();
            }
        }
    }
}
