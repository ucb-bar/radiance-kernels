#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"
#include "sgemm_impl.hpp"
#include "include/gemmini.h"
#include "gemmini_mmio.h"
#include "flash_impl.hpp"

#define GEMMINI_NEW_CISC 1

constexpr bool DEBUG = false;
constexpr bool Q_IS_K_MAJOR = true;

// temporary safety stop
static_assert(TENSOR_CORE && WARP_SPECIALIZED);

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
  constexpr uint32_t smem_Q_size = B_ROW * HEADDIM;
  constexpr uint32_t smem_K_size = B_COL * HEADDIM;
  constexpr uint32_t smem_QK_size = B_ROW * B_COL;
  constexpr uint32_t smem_V_size = B_COL * HEADDIM;
  constexpr uint32_t smem_O_size = B_COL * HEADDIM;
  static_assert(
      threads_per_threadblock == NUM_WARPS * NUM_THREADS * CORES_PER_CLUSTER,
      "flashattention kernel assumes 1 threadblock occupancy per cluster");
  uint8_t *smem_per_threadblock = reinterpret_cast<uint8_t *>(
      DEV_SMEM_START_ADDR);
  constexpr uint32_t smem_start = DEV_SMEM_START_ADDR;
  constexpr uint32_t smem_hexadecile_size = (SMEM_SIZE / 16);
  // currently assumes the Q/K/V tile sizes exactly match the hexadecile size
  static_assert(smem_hexadecile_size == smem_Q_size * sizeof(float));
  static_assert(smem_hexadecile_size == smem_K_size * sizeof(float));
  static_assert(smem_hexadecile_size == smem_QK_size * sizeof(float));
  static_assert(smem_hexadecile_size == smem_V_size * sizeof(float));
  static_assert(smem_hexadecile_size == smem_O_size * sizeof(float));

  constexpr uint32_t smem_octet0 = 0 * (SMEM_SIZE / 8);
  constexpr uint32_t smem_octet1 = 1 * (SMEM_SIZE / 8);
  constexpr uint32_t smem_octet2 = 2 * (SMEM_SIZE / 8);
  constexpr uint32_t smem_octet3 = 3 * (SMEM_SIZE / 8);
  constexpr uint32_t smem_octet4 = 4 * (SMEM_SIZE / 8);
  constexpr uint32_t smem_octet5 = 5 * (SMEM_SIZE / 8);
  constexpr uint32_t smem_octet6 = 6 * (SMEM_SIZE / 8);
  constexpr uint32_t smem_octet7 = 7 * (SMEM_SIZE / 8);

  // allocation strategy: since the two warpgroups only access *0 and *1
  // buffers each, allocate *0 in the first half of SMEM, and *1 in the latter
  // half
  // at the same time, make sure Q and K are in different banks so that they
  // can be accessed in parallel for GEMM; same for P and V
  constexpr uint32_t smem_Q0_hexadecile = 2 * 0; // octet0
  constexpr uint32_t smem_Q1_hexadecile = 2 * 4; // octet4
  constexpr uint32_t smem_K0_hexadecile = 2 * 1; // octet1
  constexpr uint32_t smem_K1_hexadecile = 2 * 5; // octet5
  constexpr uint32_t smem_V0_hexadecile = smem_K0_hexadecile + 1;
  constexpr uint32_t smem_V1_hexadecile = smem_K1_hexadecile + 1;
  constexpr uint32_t smem_S0_hexadecile = 2 * 2; // octet2
  constexpr uint32_t smem_S1_hexadecile = 2 * 6; // octet6
  constexpr uint32_t smem_P0_hexadecile = smem_Q0_hexadecile + 1;
  constexpr uint32_t smem_P1_hexadecile = smem_Q1_hexadecile + 1;
  constexpr uint32_t smem_O0_hexadecile = 2 * 3; // octet3
  constexpr uint32_t smem_O1_hexadecile = 2 * 7; // octet7

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

  float *smem_cursor_0 = smem_O0 + smem_O_size;
  float *smem_cursor_1 = smem_O1 + smem_O_size;
  // // FIXME: dangerous
  // smem_cursor = reinterpret_cast<float *>(0xff038000);
  float *smem_rowmax_0 = smem_cursor_0;
  smem_cursor_0 += smem_rowmax_size;
  float *smem_rowmax_1 = smem_cursor_1;
  smem_cursor_1 += smem_rowmax_size;
  float *smem_rowsum_0 = smem_cursor_0;
  smem_cursor_0 += smem_rowsum_size;
  float *smem_rowsum_1 = smem_cursor_1;
  smem_cursor_1 += smem_rowsum_size;
  float *smem_O_row_scale_0 = smem_cursor_0;
  smem_cursor_0 += smem_O_row_scale_size;
  float *smem_O_row_scale_1 = smem_cursor_1;
  smem_cursor_1 += smem_O_row_scale_size;

  // sharedmem "scratchpad" area to put temporary data, e.g. for tree reduction
  // in rowsum
  // NOTE: out-of bounds is not checked
  constexpr uint32_t smem_scratchpad_size =
      threads_per_warpgroup * 2 /*arbitrary slack*/;
  float *smem_scratchpad_0 = smem_cursor_0;
  smem_cursor_0 += smem_scratchpad_size;
  float *smem_scratchpad_1 = smem_cursor_1;
  smem_cursor_1 += smem_scratchpad_size;

  // select the correct buffer by warpgroup
  float *smem_Q = (warpgroup_id % 2) ? smem_Q1 : smem_Q0;
  float *smem_K = (warpgroup_id % 2) ? smem_K1 : smem_K0;
  float *smem_V = (warpgroup_id % 2) ? smem_V1 : smem_V0;
  float *smem_S = (warpgroup_id % 2) ? smem_S1 : smem_S0;
  float *smem_O = (warpgroup_id % 2) ? smem_O1 : smem_O0;
  float *smem_P = smem_S;
  float *smem_O_row_scale =
      (warpgroup_id % 2) ? smem_O_row_scale_1 : smem_O_row_scale_0;
  float *smem_rowmax = (warpgroup_id % 2) ? smem_rowmax_1 : smem_rowmax_0;
  float *smem_rowsum = (warpgroup_id % 2) ? smem_rowsum_1 : smem_rowsum_0;
  float *smem_scratchpad =
      (warpgroup_id % 2) ? smem_scratchpad_1 : smem_scratchpad_0;

#ifdef GEMMINI_NEW_CISC
  const auto spad_hex_Q = (warpgroup_id % 2) ? smem_Q1_hexadecile : smem_Q0_hexadecile;
  const auto spad_hex_K = (warpgroup_id % 2) ? smem_K1_hexadecile : smem_K0_hexadecile;
  const auto spad_hex_V = (warpgroup_id % 2) ? smem_V1_hexadecile : smem_V0_hexadecile;
  const auto spad_hex_S = (warpgroup_id % 2) ? smem_S1_hexadecile : smem_S0_hexadecile;
#else
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

  const auto spad_addr_Q = (warpgroup_id % 2) ? spad_addr_Q1 : spad_addr_Q0;
  const auto spad_addr_K = (warpgroup_id % 2) ? spad_addr_K1 : spad_addr_K0;
  const auto spad_addr_V = (warpgroup_id % 2) ? spad_addr_V1 : spad_addr_V0;
  const auto spad_addr_S = (warpgroup_id % 2) ? spad_addr_S1 : spad_addr_S0;
#endif

  // initialize rowmax/rowsum values in sharedmem
  thread_block_init_sharedmem(tid_in_warpgroup, threads_per_warpgroup, smem_O,
                              smem_rowmax, smem_rowsum, smem_O_row_scale);

  constexpr uint32_t global_barrier_id = NUM_WARPS - 1; // arbitrary

  // delay warpgroup 0 by 1 iteration to do ping-pong scheduling
  if (WARP_SPECIALIZED && warpgroup_id == 1) {
    threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);
  }

  static_assert(!GEMMINI_DMA || Q_IS_K_MAJOR,
                "DMA code assumes Q matrix is stored K-major");

  // skip everything except DMA in the loop FSM
  constexpr uint32_t skips =
      loop_matmul_skips(/*skip_lda=*/0, /*skip_ldb=*/0, /*skip_ldd=*/1,
                        /*skip_ex=*/1, /*skip_stc=*/1);

  MARK_BEG();

  if constexpr (GEMMINI_DMA) {
    if (tid_in_warpgroup == 0) {
      gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, 1, 0, 0);

      // configure DMA for the full Q matrix
      gemmini_extended3_config_ld(HEADDIM * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                  false, 0);
      // configure DMA for the full K matrix
      gemmini_extended3_config_ld(dim_seqlen * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                  false, 1);
      // configure DMA for Q*K store
      gemmini_extended_config_st(B_COL * sizeof(elem_t), 0,
                                 MVIN_SCALE_IDENTITY);
      gemmini_fence();
    }
  }

  // NOTE about barriers: Placing barriers around thread-divergent branches may
  // cause bugs, because the Vortex core doesn't check for tmask for barriers.
  // The compiler might decide to duplicate vx_bar into both paths of a
  // conditional branch, which will get evaluated twice because of the way
  // branches are handled in SIMT; this might result in stalls especially when
  // other warps behave differently on the branch condition.
  // threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

  static_assert(B_ROW == B_COL, "currently only supports square tiles");

  if constexpr (GEMMINI_DMA) {
    asm volatile("dma_move_start_%=:" ::);

    if (tid_in_warpgroup == 0) {
      const float *gmem_Q_tile = gmem_Q + HEADDIM * B_ROW * warpgroup_id;
      const float *gmem_K_tile = gmem_K;
      // do DMA
      //
      // move Q and K into SMEM before the loop starts.  Note this will be done
      // separately for the two warpgroups
      //
#ifdef GEMMINI_NEW_CISC
      // the target addresses of this should match with spad_addr_Q0 and
      // spad_addr_K0 set in this kernel
      gemmini_tile_load_ab(gmem_Q_tile, gmem_K_tile, spad_hex_Q,
                           spad_hex_K, 0 /*tile_idx_i*/,
                           0 /*tile_idx_j*/, 0 /*tile_idx_k*/, dim_seqlen,
                           dim_seqlen, HEADDIM, B_ROW, B_COL, HEADDIM);
#else
      // configure the GMEM addresses for the DMA to read from
      ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(gmem_Q_tile),
                               (uint64_t)(gmem_K_tile),
                               k_LOOP_WS_CONFIG_ADDRS_AB)
      // configure address strides for the DMA
      GEMMINI_CISC_CMD_R((dim_seqlen << 20) | (HEADDIM << 8) |
                         8 /*k_LOOP_WS_CONFIG_STRIDES_AB*/);
      gemmini_fence();

      // among other things, this also configures CONFIG_BOUNDS so that the
      // DMA knows the full matrix dimensions
      sp_tiled_matmul_full_spad_ws(
          spad_addr_Q, spad_addr_K,
          /*spad_D=*/0, /*spad_C=*/spad_addr_S,
          /*I=*/(B_ROW / DIM), /*J=*/(B_COL / DIM), /*K=*/(HEADDIM / DIM),
          /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
          /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
          /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips);
#endif

      // block until DMA complete
      gemmini_fence();

      // re-configure DMA for K and V load that will later happen in the loop
      // GMEM addr stride for K
      gemmini_extended3_config_ld(dim_seqlen * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                  false, 0);
      // GMEM addr stride for V
      gemmini_extended3_config_ld(HEADDIM * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                  false, 1);
      gemmini_fence();
    }

    asm volatile("dma_move_end_%=:" ::);
  } else {
    // load Q; this stays in SMEM for the entire loop
    if constexpr (Q_IS_K_MAJOR) {
      load_tile_to_smem<float, MemLayout::K_major, MemLayout::K_major, B_ROW,
                        HEADDIM, threads_per_warpgroup>(
          HEADDIM, warpgroup_id, 0 /* dim_k == headdim */, gmem_Q, smem_Q,
          tid_in_warpgroup);
    } else {
      load_tile_to_smem<float, MemLayout::MN_major, MemLayout::MN_major, B_ROW,
                        HEADDIM, threads_per_warpgroup>(
          dim_seqlen, warpgroup_id, 0 /* dim_k == headdim */, gmem_Q, smem_Q,
          tid_in_warpgroup);
    }

    // load K
    load_tile_to_smem<float, MemLayout::MN_major, MemLayout::MN_major, B_COL,
                      HEADDIM, threads_per_warpgroup>(
        dim_seqlen, /*tile_k=*/0, 0 /* dim_k == headdim */, gmem_K, smem_K,
        tid_in_warpgroup);
  }

  // protect write to SMEM
  threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

  // if constexpr (DEBUG) {
  //   thread_block_copy_tile<B_ROW, HEADDIM>(smem_Q0, gmem_tmp_d0, tid_in_warpgroup,
  //                          threads_per_warpgroup, warpgroup_id_in_cluster);
  //   thread_block_copy_tile<HEADDIM, B_COL>(smem_K0, gmem_tmp_d1, tid_in_warpgroup,
  //                          threads_per_warpgroup, warpgroup_id_in_cluster);

  //   threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);
  // }

  asm volatile ("tile_loop_start_%=:" :: );

  // "inner loop" along the columns of K^T
  const uint32_t k_tiles = (dim_seqlen / B_COL);
  for (uint32_t tile_k = 0; tile_k < (1 /* for perf measurement */ * k_tiles);
       tile_k++) {
    // float *smem_P_produce = (tile_k % 2) ? smem_P0 : smem_P1;
    // float *smem_P_consume = (tile_k % 2) ? smem_P1 : smem_P0;
    // float *smem_V_produce = (tile_k % 2) ? smem_V0 : smem_V1;
    // float *smem_V_consume = (tile_k % 2) ? smem_V1 : smem_V0;
    // float *smem_O_row_scale_produce =
    //     (tile_k % 2) ? smem_O_row_scale_0 : smem_O_row_scale_1;
    // float *smem_O_row_scale_consume =
    //     (tile_k % 2) ? smem_O_row_scale_1 : smem_O_row_scale_0;

    asm volatile("gemm_qk_start_%=:" ::);

    constexpr bool skip_gemm_qk = false;
    if constexpr (!skip_gemm_qk) {
      // GEMM I: S = Q*K
      //
      // FIXME: deduplicate this between GEMM II
      if constexpr (!WARP_SPECIALIZED) {
        // clear out accumulators before GEMM
        initialize_accum_regs<0>();
        initialize_accum_regs<1>();

        if constexpr (GEMMINI_DMA) {
          thread_block_gemm_single_tile<
              float, MemLayout::block_row_major, MemLayout::block_row_major,
              B_ROW, B_COL, HEADDIM, /*leading_dim_a=*/0, /*leading_dim_b=*/0,
              /*load_accum=*/false,
              /*write_to_smem=*/true>(
              smem_Q, smem_K, nullptr /*ignore accum*/, smem_S,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        } else if constexpr (Q_IS_K_MAJOR) {
          thread_block_gemm_single_tile<
              float, MemLayout::K_major, MemLayout::MN_major, B_ROW, B_COL,
              HEADDIM, /*leading_dim_a=*/0, /*leading_dim_b=*/0,
              /*load_accum=*/false,
              /*write_to_smem=*/true>(
              smem_Q, smem_K, nullptr /*ignore accum*/, smem_S,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        } else {
          thread_block_gemm_single_tile<
              float, MemLayout::MN_major, MemLayout::MN_major, B_ROW, B_COL,
              HEADDIM, /*leading_dim_a=*/0, /*leading_dim_b=*/0,
              /*load_accum=*/false,
              /*write_to_smem=*/true>(
              smem_Q, smem_K, nullptr /*ignore accum*/, smem_S,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        }
      } else {
        // when warp-specialized, there's only enough warps to do 64x32 tile
        // size so we need to do 2 GEMM calls
        static_assert(B_ROW / 2 == 32,
                      "tile size assumption for warp-specialization not met");

        float *smem_Q_half0 = smem_Q;
        float *smem_Q_half1 = (Q_IS_K_MAJOR || GEMMINI_DMA)
                                  ? smem_Q + (B_ROW / 2) * HEADDIM
                                  : smem_Q + (B_ROW / 2);
        float *smem_S_half0 = smem_S;
        float *smem_S_half1 = smem_S + (B_ROW / 2) * B_COL;

        // clear out accumulators before GEMM
        initialize_accum_regs<0>();
        initialize_accum_regs<1>();

        // split by rows into 2 chunks
        if constexpr (GEMMINI_DMA) {
          if constexpr (GEMMINI_DMA_FAST) {
            thread_block_gemm_single_tile<float, MemLayout::MN_major,
                                          MemLayout::MN_major, B_ROW / 2,
                                          B_COL, HEADDIM, /*leading_dim_a=*/0,
                                          /*leading_dim_b=*/0,
                                          /*load_accum=*/false,
                                          /*write_to_smem=*/true>(
                smem_Q_half0, smem_K, nullptr /*ignore accum*/, smem_S_half0,
                tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
                warpgroup_id_in_cluster);
          } else {
            thread_block_gemm_single_tile<float, MemLayout::block_row_major,
                                          MemLayout::block_row_major, B_ROW / 2,
                                          B_COL, HEADDIM, /*leading_dim_a=*/0,
                                          /*leading_dim_b=*/0,
                                          /*load_accum=*/false,
                                          /*write_to_smem=*/true>(
                smem_Q_half0, smem_K, nullptr /*ignore accum*/, smem_S_half0,
                tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
                warpgroup_id_in_cluster);
          }
        } else if constexpr (Q_IS_K_MAJOR) {
          thread_block_gemm_single_tile<
              float, MemLayout::K_major, MemLayout::MN_major, B_ROW / 2, B_COL,
              HEADDIM, /*leading_dim_a=*/0, /*leading_dim_b=*/0,
              /*load_accum=*/false,
              /*write_to_smem=*/true>(
              smem_Q_half0, smem_K, nullptr /*ignore accum*/, smem_S_half0,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        } else {
          thread_block_gemm_single_tile<
              float, MemLayout::MN_major, MemLayout::MN_major, B_ROW / 2, B_COL,
              HEADDIM, /*leading_dim_a=*/B_ROW, /*leading_dim_b=*/0,
              /*load_accum=*/false,
              /*write_to_smem=*/true>(
              smem_Q_half0, smem_K, nullptr /*ignore accum*/, smem_S_half0,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        }

        initialize_accum_regs<0>();
        initialize_accum_regs<1>();

        if constexpr (GEMMINI_DMA) {
          if constexpr (GEMMINI_DMA_FAST) {
            thread_block_gemm_single_tile<float, MemLayout::MN_major,
                                          MemLayout::MN_major, B_ROW / 2,
                                          B_COL, HEADDIM, /*leading_dim_a=*/0,
                                          /*leading_dim_b=*/0,
                                          /*load_accum=*/false,
                                          /*write_to_smem=*/true>(
                smem_Q_half1, smem_K, nullptr /*ignore accum*/, smem_S_half1,
                tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
                warpgroup_id_in_cluster);
          } else {
            thread_block_gemm_single_tile<float, MemLayout::block_row_major,
                                          MemLayout::block_row_major, B_ROW / 2,
                                          B_COL, HEADDIM, /*leading_dim_a=*/0,
                                          /*leading_dim_b=*/0,
                                          /*load_accum=*/false,
                                          /*write_to_smem=*/true>(
                smem_Q_half1, smem_K, nullptr /*ignore accum*/, smem_S_half1,
                tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
                warpgroup_id_in_cluster);
          }
        } else if constexpr (Q_IS_K_MAJOR) {
          thread_block_gemm_single_tile<
              float, MemLayout::K_major, MemLayout::MN_major, B_ROW / 2, B_COL,
              HEADDIM, /*leading_dim_a=*/0, /*leading_dim_b=*/0,
              /*load_accum=*/false,
              /*write_to_smem=*/true>(
              smem_Q_half1, smem_K, nullptr /*ignore accum*/, smem_S_half1,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        } else {
          thread_block_gemm_single_tile<
              float, MemLayout::MN_major, MemLayout::MN_major, B_ROW / 2, B_COL,
              HEADDIM, /*leading_dim_a=*/B_ROW, /*leading_dim_b=*/0,
              /*load_accum=*/false,
              /*write_to_smem=*/true>(
              smem_Q_half1, smem_K, nullptr /*ignore accum*/, smem_S_half1,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        }
      }
    } else {
      // load Q*K
      load_tile_to_smem<float, MemLayout::K_major, MemLayout::K_major, B_COL,
                        HEADDIM, threads_per_warpgroup>(
          dim_seqlen, warpgroup_id /* parallelize across rows */, tile_k,
          gmem_Q /*contains S*/, smem_S, tid_in_warpgroup);
    }

    // protect write to SMEM (smem_S) before softmax
    threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

    asm volatile("gemm_qk_finish_%=:" ::);

    if constexpr (DEBUG) {
      if (warpgroup_id == 0) {
        if (tile_k == 0) {
          thread_block_copy_tile<B_ROW, B_COL>(smem_S, gmem_tmp_d0,
                                 tid_in_warpgroup, threads_per_warpgroup,
                                 warpgroup_id_in_cluster);
        } else if (tile_k == 1) {
          thread_block_copy_tile<B_ROW, B_COL>(smem_S, gmem_tmp_d1,
                                 tid_in_warpgroup, threads_per_warpgroup,
                                 warpgroup_id_in_cluster);
        }

        threadblock_barrier(warpgroup_id_in_cluster,
                            warps_per_warpgroup_per_core);
      }
    }

    // inter-warpgroup barrier before online softmax
    threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);

    // Online softmax
    //
    thread_block_online_softmax(smem_S, smem_P, tid_in_warpgroup,
                                threads_per_warpgroup, warpgroup_id_in_cluster,
                                smem_scratchpad, smem_rowmax, smem_rowsum,
                                smem_O_row_scale);

    // FIXME: unnecessary?
    threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

    // data movement for K and V
    //
    // Q stays in SMEM for the entire loop
    asm volatile("move_k_v_start_%=:" ::);
    if constexpr (GEMMINI_DMA) {
      // NOTE: Beware of race conditions; with warp specialization, we need to
      // make sure below command code to DMA is not executed simultaneously
      // from the two warpgroups (which will result in hardware fault).
      // Currently the ping-pong scheduling scheme prevents that.
      if (tid_in_warpgroup == 0) {
        // configure GMEM addresses for K and V tiles
        // load K for the next iteration
        const float *gmem_K_tile = gmem_K + (B_COL * (tile_k + 1));
        // load V for the current iteration
        const float *gmem_V_tile = gmem_V + (HEADDIM * B_COL * tile_k);
        ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, (uint64_t)(gmem_K_tile),
                                 (uint64_t)(gmem_V_tile),
                                 k_LOOP_WS_CONFIG_ADDRS_AB)
        // do DMA
#ifdef GEMMINI_NEW_CISC
        gemmini_tile_load_ab(gmem_K_tile, gmem_V_tile, spad_hex_K, spad_hex_V,
                             0 /*tile_idx_i*/, 0 /*tile_idx_j*/,
                             0 /*tile_idx_k*/, HEADDIM /*dim_m of KT*/,
                             HEADDIM /*dim_n of V*/, dim_seqlen /*dim_k of KT*/,
                             B_ROW, HEADDIM, B_COL);
#else
        // configure address strides for the DMA
        // FIXME: unnecessary?
        GEMMINI_CISC_CMD_R((HEADDIM /*V*/ << 20) | (dim_seqlen /*KT*/ << 8) |
                           8 /*k_LOOP_WS_CONFIG_STRIDES_AB*/);
        gemmini_fence();

        sp_tiled_matmul_full_spad_ws(
            spad_addr_K, spad_addr_V,
            /*spad_D=*/0, /*spad_C=*/spad_addr_S,
            /*I=*/(HEADDIM / DIM), /*J=*/(HEADDIM / DIM), /*K=*/(B_COL / DIM),
            /*pad_I=*/0, /*pad_J=*/0, /*pad_K=*/0,
            /*a_transpose=*/0, /*b_transpose=*/0, /*full_C=*/0, /*low_D=*/0,
            /*acc=*/0, /*act=*/NO_ACTIVATION, /*skips=*/skips);
#endif
        // FIXME: necessary?
        gemmini_fence();
      }
    } else {
      // load K for the next iteration
      load_tile_to_smem<float, MemLayout::MN_major, MemLayout::MN_major, B_COL,
                        HEADDIM, threads_per_warpgroup>(
          dim_seqlen, tile_k + 1, 0 /* dim_k == headdim */, gmem_K, smem_K,
          tid_in_warpgroup);

      // load V for the current iteration
      // V dimension is [seqlen, headdim], stored N(headdim)-major
      load_tile_to_smem<float, MemLayout::MN_major, MemLayout::MN_major, B_COL,
                        HEADDIM, threads_per_warpgroup>(
          HEADDIM, 0 /* full N-dimension */, tile_k, gmem_V, smem_V,
          tid_in_warpgroup);
    }
    asm volatile("move_k_v_finish_%=:" ::);

    // protect write to SMEM
    threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

    if constexpr (DEBUG) {
      if (warpgroup_id == 0) {
        if (tile_k == 0) {
          thread_block_copy_rowmax(smem_rowmax, gmem_tmp_e0, tid_in_warpgroup,
                                   threads_per_warpgroup,
                                   warpgroup_id_in_cluster);
          thread_block_copy_rowmax(smem_rowsum, gmem_tmp_e2, tid_in_warpgroup,
                                   threads_per_warpgroup,
                                   warpgroup_id_in_cluster);
        } else if (tile_k == 1) {
          thread_block_copy_rowmax(smem_rowmax, gmem_tmp_e1, tid_in_warpgroup,
                                   threads_per_warpgroup,
                                   warpgroup_id_in_cluster);
          thread_block_copy_rowmax(smem_rowsum, gmem_tmp_e3, tid_in_warpgroup,
                                   threads_per_warpgroup,
                                   warpgroup_id_in_cluster);
        }

        threadblock_barrier(warpgroup_id_in_cluster,
                            warps_per_warpgroup_per_core);
      }
    }

    // inter-warpgroup barrier before GEMM II
    threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);

    // Oi rescale
    // TODO: move this back to after softmax for better load-balancing
    thread_block_O_rescale(smem_O, smem_O /*in-place*/,
                           smem_O_row_scale, tid_in_warpgroup,
                           threads_per_warpgroup, warpgroup_id_in_cluster);

    // rescale-to-PV-GEMM barrier
    threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

    if constexpr (DEBUG) {
      if (warpgroup_id == 0) {
        // O before PV
        if (tile_k == 0) {
          thread_block_copy_tile<B_ROW, B_COL>(smem_P, gmem_tmp_d2, tid_in_warpgroup,
                                 threads_per_warpgroup,
                                 warpgroup_id_in_cluster);
          thread_block_copy_tile<B_ROW, HEADDIM>(smem_O, gmem_tmp_d4, tid_in_warpgroup,
                                 threads_per_warpgroup,
                                 warpgroup_id_in_cluster);
        } else if (tile_k == 1) {
          thread_block_copy_tile<B_ROW, B_COL>(smem_P, gmem_tmp_d3, tid_in_warpgroup,
                                 threads_per_warpgroup,
                                 warpgroup_id_in_cluster);
          thread_block_copy_tile<B_ROW, HEADDIM>(smem_O, gmem_tmp_d5, tid_in_warpgroup,
                                 threads_per_warpgroup,
                                 warpgroup_id_in_cluster);
        }

        threadblock_barrier(warpgroup_id_in_cluster,
                            warps_per_warpgroup_per_core);
      }
    }

    // GEMM II: O = O + P*V

    asm volatile("gemm_pv_start_%=:" ::);

    if constexpr (!WARP_SPECIALIZED) {
      // clear out accumulators before GEMM
      initialize_accum_regs<0>();
      initialize_accum_regs<1>();

      if constexpr (GEMMINI_DMA) {
        thread_block_gemm_single_tile<
            float, MemLayout::K_major /* P matrix is row-major */,
            MemLayout::block_row_major, B_ROW, HEADDIM, B_COL,
            /*leading_dim_a=*/0, /*leading_dim_b=*/0,
            /*load_accum=*/true,
            /*write_to_smem=*/true>(
            smem_P, smem_V, smem_O /*load accum*/, smem_O, tid_in_warpgroup,
            threads_per_warpgroup, warpgroups_per_cluster,
            warpgroup_id_in_cluster);
      } else {
        thread_block_gemm_single_tile<float, MemLayout::K_major,
                                      MemLayout::MN_major, B_ROW, HEADDIM,
                                      B_COL,
                                      /*leading_dim_a=*/0, /*leading_dim_b=*/0,
                                      /*load_accum=*/true,
                                      /*write_to_smem=*/true>(
            smem_P, smem_V, smem_O /*load accum*/, smem_O, tid_in_warpgroup,
            threads_per_warpgroup, warpgroups_per_cluster,
            warpgroup_id_in_cluster);
        // FIXME: wrong but fast
        // thread_block_gemm_single_tile<float, MemLayout::MN_major,
        //                               MemLayout::MN_major,
        //                               B_ROW, HEADDIM, B_COL,
        //                               /*leading_dim_a=*/0,
        //                               /*leading_dim_b=*/0,
        //                               /*load_accum=*/true,
        //                               /*write_to_smem=*/true>(
        //     smem_P, smem_V, smem_O /*load accum*/, smem_O,
        //     tid_in_warpgroup, threads_per_warpgroup,
        //     warpgroups_per_cluster, warpgroup_id_in_cluster);
      }
    } else {
      // when warp-specialized, there's only enough warps to do 64x32 tile
      // size so we need to do 2 GEMM calls
      static_assert(B_ROW / 2 == 32,
                    "tile size assumption for warp-specialization not met");

      float *smem_P_half0 = smem_P;
      float *smem_P_half1 = (Q_IS_K_MAJOR || GEMMINI_DMA)
                                ? smem_P + (B_ROW / 2) * B_COL
                                : smem_P + (B_ROW / 2);
      float *smem_O_half0 = smem_O;
      float *smem_O_half1 = smem_O + (B_ROW / 2) * HEADDIM;

      // clear out accumulators before GEMM
      initialize_accum_regs<0>();
      initialize_accum_regs<1>();

      // split by rows into 2 chunks
      if constexpr (GEMMINI_DMA) {
        if constexpr (GEMMINI_DMA_FAST) {
          thread_block_gemm_single_tile<float, MemLayout::MN_major,
                                        MemLayout::MN_major, B_ROW / 2, HEADDIM,
                                        B_COL,
                                        /*leading_dim_a=*/0,
                                        /*leading_dim_b=*/0,
                                        /*load_accum=*/true,
                                        /*write_to_smem=*/true>(
              smem_P_half0, smem_V, smem_O_half0 /*load accum*/, smem_O_half0,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        } else {
          thread_block_gemm_single_tile<
              float, MemLayout::K_major /* P matrix is row-major */,
              MemLayout::block_row_major, B_ROW / 2, HEADDIM, B_COL,
              /*leading_dim_a=*/0,
              /*leading_dim_b=*/0,
              /*load_accum=*/true,
              /*write_to_smem=*/true>(
              smem_P_half0, smem_V, smem_O_half0 /*load accum*/, smem_O_half0,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        }
      } else if constexpr (Q_IS_K_MAJOR) {
        thread_block_gemm_single_tile<
            float, MemLayout::K_major, MemLayout::MN_major, B_ROW / 2, HEADDIM,
            B_COL, /*leading_dim_a=*/0, /*leading_dim_b=*/0,
            /*load_accum=*/true,
            /*write_to_smem=*/true>(
            smem_P_half0, smem_V, smem_O_half0 /*load accum*/, smem_O_half0,
            tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
            warpgroup_id_in_cluster);
      } else {
        thread_block_gemm_single_tile<
            float, MemLayout::MN_major, MemLayout::MN_major, B_ROW / 2, HEADDIM,
            B_COL, /*leading_dim_a=*/B_ROW, /*leading_dim_b=*/0,
            /*load_accum=*/true,
            /*write_to_smem=*/true>(
            smem_P_half0, smem_V, smem_O_half0 /*load accum*/, smem_O_half0,
            tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
            warpgroup_id_in_cluster);
      }

      initialize_accum_regs<0>();
      initialize_accum_regs<1>();

      if constexpr (GEMMINI_DMA) {
        if constexpr (GEMMINI_DMA_FAST) {
          thread_block_gemm_single_tile<float, MemLayout::MN_major,
                                        MemLayout::MN_major, B_ROW / 2, HEADDIM,
                                        B_COL,
                                        /*leading_dim_a=*/0,
                                        /*leading_dim_b=*/0,
                                        /*load_accum=*/true,
                                        /*write_to_smem=*/true>(
              smem_P_half1, smem_V, smem_O_half1 /*load accum*/, smem_O_half1,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        } else {
          thread_block_gemm_single_tile<
              float, MemLayout::K_major /* P matrix is row-major */,
              MemLayout::block_row_major, B_ROW / 2, HEADDIM, B_COL,
              /*leading_dim_a=*/0,
              /*leading_dim_b=*/0,
              /*load_accum=*/true,
              /*write_to_smem=*/true>(
              smem_P_half1, smem_V, smem_O_half1 /*load accum*/, smem_O_half1,
              tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
              warpgroup_id_in_cluster);
        }
      } else if constexpr (Q_IS_K_MAJOR) {
        thread_block_gemm_single_tile<
            float, MemLayout::K_major, MemLayout::MN_major, B_ROW / 2, HEADDIM,
            B_COL, /*leading_dim_a=*/0, /*leading_dim_b=*/0,
            /*load_accum=*/true,
            /*write_to_smem=*/true>(
            smem_P_half1, smem_V, smem_O_half1 /*load accum*/, smem_O_half1,
            tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
            warpgroup_id_in_cluster);
      } else {
        thread_block_gemm_single_tile<
            float, MemLayout::MN_major, MemLayout::MN_major, B_ROW / 2, HEADDIM,
            B_COL, /*leading_dim_a=*/B_ROW, /*leading_dim_b=*/0,
            /*load_accum=*/true,
            /*write_to_smem=*/true>(
            smem_P_half1, smem_V, smem_O_half1 /*load accum*/, smem_O_half1,
            tid_in_warpgroup, threads_per_warpgroup, warpgroups_per_cluster,
            warpgroup_id_in_cluster);
      }
    }

    threadblock_barrier(warpgroup_id_in_cluster, warps_per_warpgroup_per_core);

    asm volatile("gemm_pv_finish_%=:" ::);

    if constexpr (DEBUG) {
      if (warpgroup_id == 0) {
        // O after PV
        if (tile_k == 0) {
          thread_block_copy_tile<B_ROW, HEADDIM>(smem_O, gmem_tmp_d6, tid_in_warpgroup,
                                 threads_per_warpgroup,
                                 warpgroup_id_in_cluster);
        } else if (tile_k == 1) {
          thread_block_copy_tile<B_ROW, HEADDIM>(smem_O, gmem_tmp_d7, tid_in_warpgroup,
                                 threads_per_warpgroup,
                                 warpgroup_id_in_cluster);
        }

        threadblock_barrier(warpgroup_id_in_cluster,
                            warps_per_warpgroup_per_core);
      }
    }
  }

  asm volatile ("tile_loop_finish_%=:" :: );

  // wait for warpgroup 1 to finish, which called the global barrier before
  // entering the loop
  if (warpgroup_id == 0) {
    threadblock_barrier(global_barrier_id, warps_per_threadblock_per_core);
  }

  MARK_END();
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)KERNEL_ARG_DEV_MEM_ADDR;

  // FIXME:: use actuall seqlen/headdim
  const uint32_t problem_size = (B_ROW * B_COL) / (ELEM_PER_THREAD);
  const uint32_t hw_threads_per_cluster =
      CORES_PER_CLUSTER * vx_num_threads() * vx_num_warps();
  // prevent launching more threads than the necessary problem size
  // TODO: this does not take into account multiple clusters
  const uint32_t grid_size = (problem_size > hw_threads_per_cluster)
                                 ? hw_threads_per_cluster
                                 : problem_size;

#ifdef RADIANCE
  vx_spawn_tasks_cluster(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#else
  // NOTE: This kernel assumes contiguous thread scheduling for efficient shared
  // memory allocation, and therefore does not work with original vx_spawn_tasks
  vx_spawn_tasks_contiguous(grid_size, (vx_spawn_tasks_cb)kernel_body, arg);
#endif
  return 0;
}
