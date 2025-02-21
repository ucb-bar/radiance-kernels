#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "sgemm_impl.hpp"
#include "include/gemmini.h"
#include "gemmini_mmio.h"
#include "flash_impl.hpp"

#define FENCE_GEMM_II

#define GEMMINI_NEW_CISC 1
static_assert(GEMMINI_NEW_CISC, "NOTE: old non-CISC code is untested; look for "
                                "any misalignment of fields in ciscArgs.");

constexpr bool DEBUG = false;

static_assert(GEMMINI_DMA && !WARP_SPECIALIZED,
              "GEMMINI_DMA should be set and WARP_SPECIALIZED unset");

void kernel_body(int task_id, kernel_arg_t *__UNIFORM__ arg) {
  // @perf: All threads are running these compute whose result is mostly same
  // across the threadblock

#ifdef RADIANCE
  constexpr uint32_t cores_per_cluster = CORES_PER_CLUSTER;
#else
  constexpr uint32_t cores_per_cluster = 1;
#endif

  // FIXME: headdim not considered
  constexpr uint32_t threads_per_threadblock_theoretical =
      (B_ROW * B_COL) / (ELEM_PER_THREAD);
  constexpr uint32_t hw_threads_per_cluster =
      CORES_PER_CLUSTER * NUM_THREADS * NUM_WARPS;
  // cap maximum threadblock size to # of HW threads in cluster, to prevent
  // multiple "wave" invocations which slows down the kernel
  constexpr uint32_t threads_per_threadblock =
      (threads_per_threadblock_theoretical > hw_threads_per_cluster)
          ? hw_threads_per_cluster
          : threads_per_threadblock_theoretical;
  constexpr uint32_t threadblocks_per_cluster =
      hw_threads_per_cluster / threads_per_threadblock;
  constexpr uint32_t warps_per_threadblock_per_core =
      NUM_WARPS / threadblocks_per_cluster;

  const uint32_t threadblock_id = task_id / threads_per_threadblock;
  const uint32_t threadblock_id_in_cluster =
      threadblock_id % threadblocks_per_cluster;
  const uint32_t tid_in_threadblock = task_id % threads_per_threadblock;
  const uint32_t warp_id = tid_in_threadblock / NUM_THREADS;
  constexpr uint32_t warps_in_threadblock =
      threads_per_threadblock / NUM_THREADS;

  // warpgroup context
  constexpr uint32_t threads_per_warpgroup =
      threads_per_threadblock / (WARP_SPECIALIZED ? 2 : 1);
  constexpr uint32_t warpgroups_per_cluster =
      threadblocks_per_cluster * (WARP_SPECIALIZED ? 2 : 1);
  const uint32_t warps_per_warpgroup_per_core =
      NUM_WARPS / warpgroups_per_cluster;
  const uint32_t warpgroup_id = task_id / threads_per_warpgroup;
  const uint32_t warpgroup_id_in_cluster =
      warpgroup_id % warpgroups_per_cluster;
  const uint32_t tid_in_warpgroup = tid_in_threadblock % threads_per_warpgroup;
  // // warpgroup 0: warp 0
  // // warpgroup 1: warp 1~7
  // const uint32_t warpgroup_id = (warp_id != 0);

  const uint32_t dim_seqlen = arg->dim_seqlen;
  const uint32_t dim_headdim = arg->dim_headdim;

  // get global memory addresses from kernel arguments
  const float *gmem_Q = reinterpret_cast<float *>(arg->addr_q);
  const float *gmem_K = reinterpret_cast<float *>(arg->addr_k);
  const float *gmem_V = reinterpret_cast<float *>(arg->addr_v);
  float *gmem_O = reinterpret_cast<float *>(arg->addr_o);

  float *gmem_tmp_d0 = reinterpret_cast<float *>(0xd0000000UL);
  float *gmem_tmp_d1 = reinterpret_cast<float *>(0xd1000000UL);
  float *gmem_tmp_d2 = reinterpret_cast<float *>(0xd2000000UL);
  float *gmem_tmp_d3 = reinterpret_cast<float *>(0xd3000000UL);
  float *gmem_tmp_d4 = reinterpret_cast<float *>(0xd4000000UL);
  float *gmem_tmp_d5 = reinterpret_cast<float *>(0xd5000000UL);
  float *gmem_tmp_d6 = reinterpret_cast<float *>(0xd6000000UL);
  float *gmem_tmp_d7 = reinterpret_cast<float *>(0xd7000000UL);
  float *gmem_tmp_e0 = reinterpret_cast<float *>(0xe0000000UL);
  float *gmem_tmp_e1 = reinterpret_cast<float *>(0xe1000000UL);
  float *gmem_tmp_e2 = reinterpret_cast<float *>(0xe2000000UL);
  float *gmem_tmp_e3 = reinterpret_cast<float *>(0xe3000000UL);

  // static shared memory allocation
  // these are in float elements, not bytes
  constexpr uint32_t smem_Q_size = B_ROW * HEADDIM;
  constexpr uint32_t smem_K_size = B_COL * HEADDIM;
  constexpr uint32_t smem_QK_size = B_ROW * B_COL;
  constexpr uint32_t smem_V_size = B_COL * HEADDIM;
  constexpr uint32_t smem_O_size = B_COL * HEADDIM;
  static_assert(
      threads_per_threadblock == NUM_WARPS * NUM_THREADS * CORES_PER_CLUSTER,
      "flashattention kernel assumes 1 threadblock occupancy per cluster");
  uint8_t *smem_per_threadblock = reinterpret_cast<uint8_t *>(DEV_SMEM_START_ADDR);
  constexpr uint32_t smem_start = DEV_SMEM_START_ADDR;
  constexpr uint32_t smem_hexadecile_size = (SMEM_SIZE / 16);
  // currently assumes the Q/K/V tile sizes exactly match the hexadecile size
  static_assert(smem_hexadecile_size == smem_Q_size * sizeof(float));
  static_assert(smem_hexadecile_size == smem_K_size * sizeof(float));
  static_assert(smem_hexadecile_size == smem_QK_size * sizeof(float));
  static_assert(smem_hexadecile_size == smem_V_size * sizeof(float));
  static_assert(smem_hexadecile_size == smem_O_size * sizeof(float));

  // Q/V/S in quart0/1, K/P/O in quart2/3
  constexpr uint32_t smem_Q0_hexadecile = 4 * 0;
  constexpr uint32_t smem_Q1_hexadecile = 4 * 1;
  constexpr uint32_t smem_K0_hexadecile = 4 * 2;
  constexpr uint32_t smem_K1_hexadecile = 4 * 3;
  constexpr uint32_t smem_V0_hexadecile = smem_Q0_hexadecile + 1;
  constexpr uint32_t smem_V1_hexadecile = smem_Q1_hexadecile + 1;
  // put S1/S0 with V0/V1 so that softmax and GEMM-II doesn't cause bank
  // conflicts
  constexpr uint32_t smem_S0_hexadecile = smem_V1_hexadecile + 1;
  constexpr uint32_t smem_S1_hexadecile = smem_V0_hexadecile + 1;
  constexpr uint32_t smem_P0_hexadecile = smem_K0_hexadecile + 1;
  constexpr uint32_t smem_P1_hexadecile = smem_K1_hexadecile + 1;
  // reversed!
  constexpr uint32_t smem_O0_hexadecile = smem_P1_hexadecile + 1;
  constexpr uint32_t smem_O1_hexadecile = smem_P0_hexadecile + 1; // unused

  float *smem_Q0 = reinterpret_cast<float *>(smem_start + smem_Q0_hexadecile * smem_hexadecile_size);
  float *smem_Q1 = reinterpret_cast<float *>(smem_start + smem_Q1_hexadecile * smem_hexadecile_size);
  float *smem_K0 = reinterpret_cast<float *>(smem_start + smem_K0_hexadecile * smem_hexadecile_size);
  float *smem_K1 = reinterpret_cast<float *>(smem_start + smem_K1_hexadecile * smem_hexadecile_size);
  float *smem_V0 = reinterpret_cast<float *>(smem_start + smem_V0_hexadecile * smem_hexadecile_size);
  float *smem_V1 = reinterpret_cast<float *>(smem_start + smem_V1_hexadecile * smem_hexadecile_size);
  float *smem_S0 = reinterpret_cast<float *>(smem_start + smem_S0_hexadecile * smem_hexadecile_size);
  float *smem_S1 = reinterpret_cast<float *>(smem_start + smem_S1_hexadecile * smem_hexadecile_size);
  float *smem_P0 = reinterpret_cast<float *>(smem_start + smem_P0_hexadecile * smem_hexadecile_size);
  float *smem_P1 = reinterpret_cast<float *>(smem_start + smem_P1_hexadecile * smem_hexadecile_size);
  float *smem_O0 = reinterpret_cast<float *>(smem_start + smem_O0_hexadecile * smem_hexadecile_size);
  float *smem_O1 = reinterpret_cast<float *>(smem_start + smem_O1_hexadecile * smem_hexadecile_size);

  // allocate rowmax/rowsum storage at the end of the sharedmem address space
  constexpr uint32_t smem_rowmax_size = B_ROW * ROWMAX_SETS;
  constexpr uint32_t smem_rowsum_size = B_ROW;
  constexpr uint32_t smem_O_row_scale_size = B_ROW;

  float *smem_cursor = smem_O1 + smem_O_size;
  // // FIXME: dangerous
  // smem_cursor = reinterpret_cast<float *>(0xff038000);
  float *smem_rowmax_0 = smem_cursor;
  smem_cursor += smem_rowmax_size;
  float *smem_rowmax_1 = smem_cursor;
  smem_cursor += smem_rowmax_size;
  float *smem_rowsum_0 = smem_cursor;
  smem_cursor += smem_rowsum_size;
  float *smem_rowsum_1 = smem_cursor;
  smem_cursor += smem_rowsum_size;
  float *smem_O_row_scale_0 = smem_cursor;
  smem_cursor += smem_O_row_scale_size;
  float *smem_O_row_scale_1 = smem_cursor;
  smem_cursor += smem_O_row_scale_size;

  // sharedmem "scratchpad" area to put temporary data, e.g. for tree reduction
  // in rowsum
  // NOTE: out-of bounds is not checked
  constexpr uint32_t smem_scratchpad_size =
      threads_per_warpgroup * 2 /*arbitrary slack*/;
  float *smem_scratchpad_0 = smem_cursor;
  smem_cursor += smem_scratchpad_size;
  float *smem_scratchpad_1 = smem_cursor;
  smem_cursor += smem_scratchpad_size;
  uint32_t *smem_O_flag = reinterpret_cast<uint32_t *>(smem_cursor);
  smem_cursor += 1 /* 4Byte */;

  static_assert(sizeof(elem_t) == sizeof(float));
  constexpr uint32_t spad_addr_factor = DIM * sizeof(elem_t);
  constexpr uint32_t spad_addr_Q0 = smem_Q0_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_Q1 = smem_Q1_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_K0 = smem_K0_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_K1 = smem_K1_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_V0 = smem_V0_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_V1 = smem_V1_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_S0 = smem_S0_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_S1 = smem_S1_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_P0 = smem_P0_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_P1 = smem_P1_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_O0 = smem_O0_hexadecile * smem_hexadecile_size / spad_addr_factor;
  constexpr uint32_t spad_addr_O1 = smem_O1_hexadecile * smem_hexadecile_size / spad_addr_factor;

  constexpr uint32_t global_barrier_id = NUM_WARPS - 1; // arbitrary
  static_assert(warps_per_threadblock_per_core == NUM_WARPS);

  // initialize rowmax/rowsum values in sharedmem
  thread_block_init_sharedmem(tid_in_warpgroup, threads_per_warpgroup, smem_O0,
                              smem_rowmax_0, smem_rowsum_0, smem_O_row_scale_0);
  thread_block_init_sharedmem(tid_in_warpgroup, threads_per_warpgroup, smem_O1,
                              smem_rowmax_1, smem_rowsum_1, smem_O_row_scale_1);

  threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);

  // skip everything except DMA in the loop FSM
  constexpr uint32_t skips =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);
  constexpr uint32_t skips_only_a =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);
  constexpr uint32_t skips_only_b =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/0, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);
  constexpr uint32_t skips_mvout_spad =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/0);
  constexpr uint32_t skips_matmul =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/1,
                        /*skip_ex=*/0, /*skip_stc=*/0);
  constexpr uint32_t skips_matmul_preload =
      loop_matmul_skips(/*skip_lda=*/1, /*skip_ldb=*/1, /*skip_ldd=*/0,
                        /*skip_ex=*/0, /*skip_stc=*/1);

  MARK_BEG();

  if (tid_in_warpgroup == 0) {
    gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);

    // configure DMA with GMEM address strides
    // Q matrix
    gemmini_extended3_config_ld(HEADDIM * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 0);
    // K matrix
    gemmini_extended3_config_ld(dim_seqlen * sizeof(elem_t),
                                MVIN_SCALE_IDENTITY, false, 1);
    // configure DMA for Q*K store
    gemmini_extended_config_st(B_COL * sizeof(elem_t), 0, MVIN_SCALE_IDENTITY);
    gemmini_fence();
  }

  // NOTE about barriers: Placing barriers around thread-divergent branches may
  // cause bugs, because the Vortex core doesn't check for tmask for barriers.
  // The compiler might decide to duplicate vx_bar into both paths of a
  // conditional branch, which will get evaluated twice because of the way
  // branches are handled in SIMT; this might result in stalls especially when
  // other warps behave differently on the branch condition.
  // threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

  static_assert(B_ROW == B_COL, "currently only supports square tiles");

  // move Q and K into SMEM before the loop starts
  //
  asm volatile("dma_move_start_%=:" ::);
  if (tid_in_warpgroup == 0) {
    // make sure to read from the correct row of Q
    const float *gmem_Q_tile = gmem_Q + HEADDIM * B_ROW * warpgroup_id;
    const float *gmem_K_tile = gmem_K;

    // do DMA
    //
    // move Q to spad_addr_Q0 for the first iteration
    //
#ifdef GEMMINI_NEW_CISC
    // the target addresses of this should match with spad_addr_Q0 and
    // spad_addr_K0 set in this kernel
    gemmini_tile_load_ab(gmem_Q_tile, gmem_K_tile, smem_Q0_hexadecile,
                         smem_K0_hexadecile, 0 /*tile_idx_i*/, 0 /*tile_idx_j*/,
                         0 /*tile_idx_k*/, dim_seqlen, dim_seqlen, HEADDIM,
                         B_ROW, B_COL, HEADDIM);
#else
    // configure the GMEM addresses for the DMA to read from
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(gmem_Q_tile),
                             (uint64_t)(gmem_K_tile), k_LOOP_WS_CONFIG_ADDRS_AB)
    // configure address strides for the DMA
    GEMMINI_CISC_CMD_R((dim_seqlen << 20) | (HEADDIM << 8) |
                       GEMMINI_CISC_SET_AB_STRIDE);
    gemmini_fence();

    // among other things, this also configures CONFIG_BOUNDS so that the
    // DMA knows the full matrix dimensions
    sp_tiled_matmul_full_spad_ws(
        spad_addr_Q0, spad_addr_K0,
        /*spad_D=*/0, /*spad_C=*/spad_addr_S0/*bogus*/,
        /*I=*/(B_ROW / DIM), /*J=*/(B_COL / DIM), /*K=*/(HEADDIM / DIM),
        /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
        /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
        /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips);
#endif

    // block until DMA complete
    gemmini_fence();

    // also move Q to spad_addr_Q1 for the second iteration
    //
#ifdef GEMMINI_NEW_CISC
    gemmini_tile_load_ab(gmem_Q_tile, gmem_K_tile, smem_Q1_hexadecile,
                         smem_K1_hexadecile, 0 /*tile_idx_i*/, 0 /*tile_idx_j*/,
                         0 /*tile_idx_k*/, dim_seqlen, dim_seqlen, HEADDIM,
                         B_ROW, B_COL, HEADDIM);
#else
    // FIXME: re-configure necessary?
    gmem_K_tile = gmem_K + (B_COL * 1);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(gmem_Q_tile),
                             (uint64_t)(gmem_K_tile), k_LOOP_WS_CONFIG_ADDRS_AB)
    GEMMINI_CISC_CMD_R((dim_seqlen << 20) | (HEADDIM << 8) |
                       8 /*k_LOOP_WS_CONFIG_STRIDES_AB*/);
    gemmini_fence();

    sp_tiled_matmul_full_spad_ws(
        spad_addr_Q1, spad_addr_K1/*bogus*/,
        /*spad_D=*/0, /*spad_C=*/spad_addr_S0/*bogus*/,
        /*I=*/(B_ROW / DIM), /*J=*/(B_COL / DIM), /*K=*/(HEADDIM / DIM),
        /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
        /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
        /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips);
#endif

    // block until DMA complete
    gemmini_fence();

    // re-configure DMA for K and V load that will later happen in the loop
    // FIXME: not sure necessary with new CISC
    //
    // GMEM addr stride for K
    gemmini_extended3_config_ld(dim_seqlen * sizeof(elem_t),
                                MVIN_SCALE_IDENTITY, false, 0);
    // GMEM addr stride for V
    gemmini_extended3_config_ld(HEADDIM * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, 1);
    gemmini_fence();
  }

  asm volatile("dma_move_end_%=:" ::);

  // protect write to SMEM
  // threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);
  threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);

  // if constexpr (DEBUG) {
  //   thread_block_copy_tile<B_ROW, HEADDIM>(smem_Q0, gmem_tmp_d0, tid_in_warpgroup,
  //                          threads_per_warpgroup, warpgroup_id_in_cluster);
  //   thread_block_copy_tile<HEADDIM, B_COL>(smem_K0, gmem_tmp_d1, tid_in_warpgroup,
  //                          threads_per_warpgroup, warpgroup_id_in_cluster);
  //   threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);
  // }

  constexpr uint32_t threads_per_warpgroup_simt =
      threads_per_warpgroup -
      CORES_PER_CLUSTER * NUM_THREADS /*warp 0, 4, 8, 12*/;
  constexpr uint32_t warpgroup_id_simt = 1;
  constexpr uint32_t barrier_id_simt = 1;
  constexpr uint32_t barrier_count_simt = NUM_WARPS - 1;
  const uint32_t tid_in_warpgroup_simt =
      tid_in_warpgroup - (CORES_PER_CLUSTER * NUM_THREADS);
  static_assert(barrier_id_simt == 1 && barrier_count_simt == 7);

  asm volatile ("tile_loop_start_%=:" :: );

  // "inner loop" along the columns of K^T
  const uint32_t k_tiles = (dim_seqlen / B_COL);
  for (uint32_t tile_k = 0;
       tile_k < (4 /*for perf measurement*/ *
                 // virgo kernel is fully pipelined around (2 GEMMs | softmax);
                 // requires two loop iterations to finish one tile compute
                 (2 * k_tiles)) +
                    2 /*pipeline latency*/;
       tile_k++) {
    if constexpr (DEBUG || true) {
      threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);
    }

    // select the correct double buffer by tile iteration
    // all iterations work on the same Q row tile; no ping-pong necessary
    asm volatile ("dbuf_sel_start_%=:" :: );
    // FIXME speedup by doing arithmetic
    float *smem_Q = smem_Q0;
    float *smem_K_consume = (tile_k & 1) ? smem_K1 : smem_K0;
    float *smem_K_produce = (tile_k & 1) ? smem_K0 : smem_K1;
    float *smem_V_consume = (tile_k & 1) ? smem_V1 : smem_V0;
    float *smem_V_produce = (tile_k & 1) ? smem_V0 : smem_V1;
    float *smem_S_consume = (tile_k & 1) ? smem_S1 : smem_S0;
    float *smem_S_produce = (tile_k & 1) ? smem_S0 : smem_S1;
    float *smem_P_consume = (tile_k & 1) ? smem_P1 : smem_P0;
    float *smem_P_produce = (tile_k & 1) ? smem_P0 : smem_P1;
    // O, rowmax/rowsum etc. is sequentially updated at every iteration; no
    // ping-pong necessary
    float *smem_O = smem_O0;
    float *smem_O_row_scale = smem_O_row_scale_0;
    float *smem_rowmax = smem_rowmax_0;
    float *smem_rowsum = smem_rowsum_0;
    float *smem_scratchpad = smem_scratchpad_0;

    const auto spad_addr_Q = spad_addr_Q0;
    const auto spad_addr_K_consume = (tile_k & 1) ? spad_addr_K1 : spad_addr_K0;
    const auto spad_addr_K_produce = (tile_k & 1) ? spad_addr_K0 : spad_addr_K1;
    const auto spad_addr_V_consume = (tile_k & 1) ? spad_addr_V1 : spad_addr_V0;
    const auto spad_addr_V_produce = (tile_k & 1) ? spad_addr_V0 : spad_addr_V1;
    const auto spad_addr_S_consume = (tile_k & 1) ? spad_addr_S1 : spad_addr_S0;
    const auto spad_addr_S_produce = (tile_k & 1) ? spad_addr_S0 : spad_addr_S1;
    const auto spad_addr_P_consume = (tile_k & 1) ? spad_addr_P1 : spad_addr_P0;
    const auto spad_addr_P_produce = (tile_k & 1) ? spad_addr_P0 : spad_addr_P1;
    const auto spad_addr_O = spad_addr_O0; // NOTE: there's only single O tile

    const auto spad_hex_Q = smem_Q0_hexadecile;
    const auto spad_hex_K_consume = (tile_k & 1) ? smem_K1_hexadecile : smem_K0_hexadecile;
    const auto spad_hex_K_produce = (tile_k & 1) ? smem_K0_hexadecile : smem_K1_hexadecile;
    const auto spad_hex_V_consume = (tile_k & 1) ? smem_V1_hexadecile : smem_V0_hexadecile;
    const auto spad_hex_V_produce = (tile_k & 1) ? smem_V0_hexadecile : smem_V1_hexadecile;
    const auto spad_hex_S_consume = (tile_k & 1) ? smem_S1_hexadecile : smem_S0_hexadecile;
    const auto spad_hex_S_produce = (tile_k & 1) ? smem_S0_hexadecile : smem_S1_hexadecile;
    const auto spad_hex_P_consume = (tile_k & 1) ? smem_P1_hexadecile : smem_P0_hexadecile;
    const auto spad_hex_P_produce = (tile_k & 1) ? smem_P0_hexadecile : smem_P1_hexadecile;
    const auto spad_hex_O = smem_O0_hexadecile; // NOTE: there's only single O tile
    asm volatile ("dbuf_sel_end_%=:" :: );

    if (vx_warp_id() == 0 /* warp 0 in every core */) {
      if (tile_k >= 2) // delay by 2 iters for pipelining
      {
        const uint32_t tile_k_ = tile_k - 2;

        // GEMM II: O = O + P*V
        // --------------------
        // This is done *before* GEMM I in the software pipeline, working on the
        // online softmax result tile from the previous iteration

        asm volatile("gemm_pv_start_%=:" ::);

        if (tid_in_warpgroup == 0) {
          // kickoff matmul

          // FIXME: perf: prevent GMEM->SMEM load for O tile
          gemmini_fence();
#ifdef GEMMINI_NEW_CISC
          gemmini_tile_compute</*store_to_spad=*/true>(
              spad_hex_P_consume, spad_hex_V_consume, spad_hex_O,
              0 /*accumulate.
                  FIXME: Gemmini doens't support accumulation from a spad tile*/);
#else
          sp_tiled_matmul_full_spad_ws(
              spad_addr_P_consume, spad_addr_V_consume,
              /*spad_D=*/spad_addr_O, /*spad_C=*/spad_addr_O,
              /*I=*/(B_ROW / DIM), /*J=*/(HEADDIM / DIM), /*K=*/(B_COL / DIM),
              /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
              /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
              /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips_matmul);
#endif
        }

        // // reconverge from mmio divergence
        // threadblock_barrier(warpgroup_id_in_cluster,
        //                     warps_per_warpgroup_per_core);

        asm volatile("gemm_pv_finish_%=:" ::);
      }

      // GEMM I: S = Q*K
      //
      // kick off asynchronously; fence later
      asm volatile("gemm_qk_start_%=:" ::);

      if (tid_in_warpgroup == 0) {
        // fence to GEMM II completion
        gemmini_fence();

#ifdef FENCE_GEMM_II
        asm volatile("rescale_fence_write_start_%=:" ::);
        // signal that GEMM II is finished to O rescale step
        *smem_O_flag = 1;
        vx_fence();
        asm volatile("rescale_fence_write_end_%=:" ::);
#endif

        // Kick off GEMM I
        //
#ifdef GEMMINI_NEW_CISC
        gemmini_tile_compute</*store_to_spad=*/true>(
            spad_hex_Q, spad_hex_K_consume, spad_hex_S_produce,
            0 /*accumulate*/);
#else
        sp_tiled_matmul_full_spad_ws(
            spad_addr_Q, spad_addr_K_consume,
            /*spad_D=*/0, /*spad_C=*/spad_addr_S_produce,
            /*I=*/(B_ROW / DIM), /*J=*/(B_COL / DIM), /*K=*/(HEADDIM / DIM),
            /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
            /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
            /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips_matmul);
#endif

        asm volatile("gemm_qk_finish_%=:" ::);

        // data move for K and V
        //
        // Q stays in SMEM for the entire loop
        asm volatile("move_k_v_start_%=:" ::);

        // configure GMEM addresses for K and V tiles
        // load K for the next iteration
        const float *gmem_K_tile = gmem_K + (B_COL * (tile_k + 1 /*runahead*/));
        // load V for the *previous* iteration; this will be consumed 2
        // iterations later
        const float *gmem_V_tile =
            gmem_V + (HEADDIM * B_COL * (tile_k - 1 /*dragbehind*/));

#if 0
        // fence mvout S to SMEM
        gemmini_fence();
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(gmem_K_tile),
                                 (uint64_t)(gmem_V_tile),
                                 k_LOOP_WS_CONFIG_ADDRS_AB)
#endif

        // do DMA
        if (tile_k == 0) {
          // // configure address strides for the DMA
          // // FIXME: unnecessary?
          // GEMMINI_CISC_CMD_R((HEADDIM /*V*/ << 20) | (dim_seqlen /*KT*/ << 8) |
          //                    8 /*k_LOOP_WS_CONFIG_STRIDES_AB*/);
          // gemmini_fence();
          //
          // we load (k-1)th tile for V; skip V for the 1st iteration,
          // sp_tiled_matmul_full_spad_ws(
          //     spad_addr_K_produce, spad_addr_V_produce,
          //     /*spad_D=*/0, /*spad_C=*/0,
          //     /*I=*/(B_ROW / DIM), /*J=*/(HEADDIM / DIM), /*K=*/(B_COL / DIM),
          //     /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
          //     /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
          //     /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips_only_a);
        } else {
#ifdef GEMMINI_NEW_CISC
          gemmini_tile_load_ab(
              gmem_K_tile, gmem_V_tile, spad_hex_K_produce, spad_hex_V_produce,
              0 /*tile_idx_i*/, 0 /*tile_idx_j*/, 0 /*tile_idx_k*/,
              HEADDIM /*dim_m of KT*/, HEADDIM /*dim_n of V*/,
              dim_seqlen /*dim_k of KT*/, B_ROW, HEADDIM, B_COL);
#else
          // configure address strides for the DMA
          // FIXME: unnecessary?
          GEMMINI_CISC_CMD_R((HEADDIM /*V*/ << 20) | (dim_seqlen /*KT*/ << 8) |
                             8 /*k_LOOP_WS_CONFIG_STRIDES_AB*/);
          gemmini_fence();
          sp_tiled_matmul_full_spad_ws(
              spad_addr_K_produce, spad_addr_V_produce,
              /*spad_D=*/0, /*spad_C=*/0,
              /*I=*/(B_ROW / DIM), /*J=*/(HEADDIM / DIM), /*K=*/(B_COL / DIM),
              /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
              /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
              /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips);
#endif
        }

        // fence everything before going to the next tile
        gemmini_fence();
      }

      // threadblock_barrier(warpgroup_id_in_cluster,
      //                     warps_per_warpgroup_per_core);

      asm volatile("move_k_v_finish_%=:" ::);

      // NOTE: cannot put barrier here; thread 1-7 in warp 0 will skip the
      // branch and call this barrier earlier than when thread 0 finishes.
      // Since tmask is not considered, that will be a barrier resolve done too
      // early
      // threadblock_barrier(0, 1);

    } else /* warp_id != 0 */ {

      if (tile_k >= 1) // delay online softmax by 1 iters
      {
        const uint32_t tile_k_ = tile_k - 1;

        if constexpr (DEBUG) {
          // verify S = Q*K before softmax
          if (warpgroup_id == 0) {
            if (tile_k_ == 0) {
              thread_block_copy_tile<B_ROW, B_COL, GEMMINI_DMA>(
                  smem_S_consume, gmem_tmp_d0, tid_in_warpgroup_simt,
                  threads_per_warpgroup_simt, warpgroup_id_simt);
            } else if (tile_k_ == 1) {
              thread_block_copy_tile<B_ROW, B_COL, GEMMINI_DMA>(
                  smem_S_consume, gmem_tmp_d1, tid_in_warpgroup_simt,
                  threads_per_warpgroup_simt, warpgroup_id_simt);
            }

            threadblock_barrier(barrier_id_simt, barrier_count_simt);
          }
        }

        // Online softmax
        //
        thread_block_online_softmax</*block_row_major=*/GEMMINI_DMA>(
            smem_S_consume, smem_P_produce, tid_in_warpgroup_simt,
            threads_per_warpgroup_simt, warpgroup_id_simt, smem_scratchpad,
            smem_rowmax, smem_rowsum, smem_O_row_scale);

        threadblock_barrier(barrier_id_simt, barrier_count_simt);

        if constexpr (DEBUG) {
          if (warpgroup_id == 0) {
            if (tile_k_ == 0) {
              thread_block_copy_rowmax(
                  smem_rowmax, gmem_tmp_e0, tid_in_warpgroup_simt,
                  threads_per_warpgroup_simt, warpgroup_id_simt);
              thread_block_copy_rowmax(
                  smem_rowsum, gmem_tmp_e2, tid_in_warpgroup_simt,
                  threads_per_warpgroup_simt, warpgroup_id_simt);
            } else if (tile_k_ == 1) {
              thread_block_copy_rowmax(smem_rowmax, gmem_tmp_e1,
                                       tid_in_warpgroup_simt, threads_per_warpgroup_simt,
                                       warpgroup_id_simt);
              thread_block_copy_rowmax(smem_rowsum, gmem_tmp_e3,
                                       tid_in_warpgroup_simt, threads_per_warpgroup_simt,
                                       warpgroup_id_simt);
            }

            threadblock_barrier(barrier_id_simt, barrier_count_simt);
          }
        }

#ifdef FENCE_GEMM_II
        asm volatile("rescale_fence_read_start_%=:" ::);
        // check flag to make sure GEMM II finished and read-after-write
        // dependency on O tile is settled for rescale
        if (tid_in_warpgroup_simt == 0) {
          while ((*smem_O_flag) != 1)
            ;
          // set it back to 0 for the next tile iteration
          *smem_O_flag = 0;
          vx_fence();
        }
        asm volatile("rescale_fence_read_end_%=:" ::);
#endif

#if 0
        if (tid_in_warpgroup == 0) {
          gemmini_fence();
          gemmini_fence();
          gemmini_fence();
          gemmini_fence();
        }

        // reconverge from mmio divergence
        threadblock_barrier(warpgroup_id_in_cluster,
                            warps_per_warpgroup_per_core);
#endif

        if constexpr (DEBUG) {
          if (warpgroup_id == 0) {
            gemmini_fence();
            gemmini_fence();

            // O after PV
            if (tile_k_ == 1 /*wait until GEMM II finshes */) {
              thread_block_copy_tile<B_ROW, HEADDIM, GEMMINI_DMA>(
                  smem_O, gmem_tmp_d6, tid_in_warpgroup_simt, threads_per_warpgroup_simt,
                  warpgroup_id_simt);
            } else if (tile_k_ == 2) {
              thread_block_copy_tile<B_ROW, HEADDIM, GEMMINI_DMA>(
                  smem_O, gmem_tmp_d7, tid_in_warpgroup_simt, threads_per_warpgroup_simt,
                  warpgroup_id_simt);
            }

            threadblock_barrier(barrier_id_simt, barrier_count_simt);
          }
        }

        // Oi rescale
        thread_block_O_rescale</*block_row_major=*/GEMMINI_DMA>(
            smem_O, smem_O /*in-place*/, smem_O_row_scale,
            tid_in_warpgroup_simt, threads_per_warpgroup_simt,
            warpgroup_id_simt);

        // rescale-to-PV-GEMM barrier
        threadblock_barrier(barrier_id_simt, barrier_count_simt);

        if constexpr (DEBUG) {
          if (warpgroup_id == 0) {
            // O before PV
            if (tile_k_ == 0) {
              thread_block_copy_tile<B_ROW, B_COL, GEMMINI_DMA>(
                  smem_P_produce, gmem_tmp_d2, tid_in_warpgroup_simt,
                  threads_per_warpgroup_simt, warpgroup_id_simt);
              thread_block_copy_tile<B_ROW, HEADDIM, GEMMINI_DMA>(
                  smem_O, gmem_tmp_d4, tid_in_warpgroup_simt,
                  threads_per_warpgroup_simt, warpgroup_id_simt);
            } else if (tile_k_ == 1) {
              thread_block_copy_tile<B_ROW, B_COL, GEMMINI_DMA>(
                  smem_P_produce, gmem_tmp_d3, tid_in_warpgroup_simt,
                  threads_per_warpgroup_simt, warpgroup_id_simt);
              thread_block_copy_tile<B_ROW, HEADDIM, GEMMINI_DMA>(
                  smem_O, gmem_tmp_d5, tid_in_warpgroup_simt,
                  threads_per_warpgroup_simt, warpgroup_id_simt);
            }

            threadblock_barrier(barrier_id_simt, barrier_count_simt);
          }
        }
      }

#if 0
      // fence GEMM I after Oi rescale
      if (tid_in_warpgroup == 0) {
        gemmini_fence();
        gemmini_fence();
        gemmini_fence();
        gemmini_fence();
      }

      // reconverge from mmio divergence
      threadblock_barrier(warpgroup_id_in_cluster,
                          warps_per_warpgroup_per_core);
#endif

      // intra-warpgroup barrier
      threadblock_barrier(barrier_id_simt, barrier_count_simt);
    }
  }

  asm volatile ("tile_loop_finish_%=:" :: );

  MARK_END();
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  const uint32_t hw_threads_per_cluster =
      CORES_PER_CLUSTER * vx_num_threads() * vx_num_warps();
  // fix to 1 threadblock per cluster
  const uint32_t grid_size = hw_threads_per_cluster;

#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
