// enc.h: layout shared by the GPU kernel, the Rocket host and gen_ref.py for the tapeout
// transformer encoder layer (pre-LN, MX-FP8 GEMMs, bf16 activations).  See ref.py.
//
// All buffers live in GPU DRAM at device addresses below 0x1C00_0000 (the host sees device
// address X at 0x1_0000_0000 | X).  GEMM operands use the tapeout_gemm layouts:
//   A operand  codes [M][K] row-major, scales tile-blocked [M/TM][K/TK][TK/32][TM]
//   B operand  codes [K][N] row-major, scales tile-blocked [N/TN][K/TK][TK/32][TN]

#ifndef TAPEOUT_ENCODER_H
#define TAPEOUT_ENCODER_H

#include <stdint.h>

// ---- shape ---------------------------------------------------------------------------------
#define EN_S    256u    // tokens
#define EN_D    512u    // model width
#define EN_H    4u      // heads
#define EN_DH   128u    // head width
#define EN_F    2048u   // FFN width
#define EN_TM   128u    // GEMM tile
#define EN_TN   128u
#define EN_TK   256u
// Clusters the stages are split over: 1 = the one-cluster U250 system (a second cluster, if
// present, idles); 2 = the two-cluster tapeout config (FireAxe pair): GEMMs and fused stages by
// 128-row block, DMA-round SIMT stages by half the items, attention by heads (2 per cluster).
#ifndef EN_CLUSTERS
#define EN_CLUSTERS 1u
#endif    // 256: 82-91% on the four encoder GEMMs (board, 2026-10-12); 128: 80-88%
#define EN_LN_EPS_BITS 0x3727C5ACu   // 1e-5f

// ---- device addresses ------------------------------------------------------------------------
#define EN_X        0x18000000u   // x       bf16 [S][D]
#define EN_LN1_G    0x18040000u   // gamma1  f32 [D]
#define EN_LN1_B    0x18041000u   // beta1   f32 [D]
#define EN_LN2_G    0x18042000u   // gamma2  f32 [D]
#define EN_LN2_B    0x18043000u   // beta2   f32 [D]
#define EN_H1       0x18100000u   // LN1(x)  A operand codes [S][D]
#define EN_H1_S     0x18140000u   //         A operand scales
#define EN_WQKV     0x18200000u   // Wqkv    B operand codes [D][3D]
#define EN_WQKV_S   0x182C0000u   //         B operand scales
#define EN_QKV      0x18400000u   // qkv     bf16 [S][3D]
// attention operands per head h (FA_ENC layout, see kernels/fa_stable_fpga/kernel.cpp)
#define EN_FA_Q     0x18600000u   // [H][S/64][64][DH]  e4m3  (h stride S*DH)
#define EN_FA_QS    0x18680000u   // [H][S/64][DH/32][64] e8m0 (h stride S*DH/32)
#define EN_FA_KT    0x18700000u   // [H][DH][S]         e4m3  (h stride DH*S)
#define EN_FA_KTS   0x18780000u   // [H][DH/32][S]      e8m0
#define EN_FA_V     0x18800000u   // [H][S][DH]         e4m3
#define EN_FA_VS    0x18880000u   // [H][S/32][DH]      e8m0
#define EN_FA_O     0x18900000u   // [H][S/64][64][DH]  bf16  (h stride S*DH*2)
#define EN_OA       0x18A00000u   // MXq(o)  A operand codes [S][D] (o in [S][D] = heads side by side)
#define EN_OA_S     0x18A40000u   //         A operand scales
#define EN_WO       0x18B00000u   // Wo      B operand codes [D][D]
#define EN_WO_S     0x18B40000u   //         B operand scales
#define EN_ATT      0x18C00000u   // o @ Wo  bf16 [S][D]
#define EN_X1       0x18D00000u   // x1 = x + att  bf16 [S][D]
#define EN_H2       0x18E00000u   // LN2(x1) A operand codes [S][D]
#define EN_H2_S     0x18E40000u   //         A operand scales
#define EN_W1       0x19000000u   // W1      B operand codes [D][F]
#define EN_W1_S     0x19100000u   //         B operand scales
#define EN_FF       0x19200000u   // h2 @ W1 bf16 [S][F]; GELU(.) written back in place (bf16)
#define EN_G        0x19400000u   // MXq(GELU) A operand codes [S][F]
#define EN_G_S      0x19480000u   //         A operand scales
#define EN_W2       0x19600000u   // W2      B operand codes [F][D]
#define EN_W2_S     0x19700000u   //         B operand scales
#define EN_FF2      0x19800000u   // g @ W2  bf16 [S][D]
#define EN_Y        0x19900000u   // y = x1 + ff2  bf16 [S][D]
#define EN_SCRATCH  0x1BF00000u   // mesh CONFIG_SCALE_MEM target; never written
#define EN_ARGS     0x1F0F0000u   // struct ENArgs
#define EN_FA_ARGS  0x1F0F1000u   // struct FaEncArgs (FA_ENC_ARGS)

// ---- stages ----------------------------------------------------------------------------------
enum ENStage : uint32_t {
  EN_STAGE_LNQ1 = 1,    // h1 = MXq(LN1(x))
  EN_STAGE_QKV = 2,     // qkv = h1 @ Wqkv            (mesh, thread 0)
  EN_STAGE_SPLIT = 3,   // qkv -> per-head Q, K^T, V MX-FP8 operands in the FA_ENC layout
  EN_STAGE_FA = 4,      // one head of attention (FA_ENC, args at EN_FA_ARGS); launched per head
  EN_STAGE_OQ = 5,      // o (FA layout) -> MXq A operand
  EN_STAGE_OPROJ = 6,   // att = MXq(o) @ Wo          (mesh)
  EN_STAGE_RES1LN = 7,  // x1 = x + att; h2 = MXq(LN2(x1))
  EN_STAGE_UP = 8,      // ff = h2 @ W1               (mesh)
  EN_STAGE_GELUQ = 9,   // ff <- bf16(GELU(ff)); g = MXq(ff)
  EN_STAGE_DOWN = 10,   // ff2 = g @ W2               (mesh)
  EN_STAGE_RES2 = 11,   // y = x1 + ff2
  EN_STAGE_CLEAR = 12,  // zero SMEM (diagnostic)
  EN_STAGE_GENQKV = 13, // qkv <- the synthetic test pattern (host.cpp write_qkv_synthetic), on the GPU
  EN_STAGE_GENX = 14,   // x <- the test input (host.cpp write_inputs), on the GPU
  EN_STAGE_GEN = 15,    // ENArgs gen_*: fill a bf16 buffer with the synthetic pattern (gen_ref.gen_syn)
  EN_STAGE_UPGELU = 16, // g = MXq(GELU(h2 @ W1)) in one launch (ff stays in SMEM)
  EN_STAGE_QKVSPLIT = 17, // h1 @ Wqkv -> per-head Q, K^T, V operands in one launch (qkv stays in SMEM)
  EN_STAGE_GENW = 18,   // metasim: MX weight codes + scales at gen_addr / gen_saddr (ENArgs)
  EN_STAGE_DOWNRES = 19, // y = x1 + g @ W2 in one launch (ff2 stays in SMEM)
  EN_STAGE_OPROJRES = 20, // x1 = x + MXq(o) @ Wo in one launch (att stays in SMEM)
  EN_STAGE_LNQ2 = 21,   // h2 = MXq(LN2(x1)) when x1 is already in DRAM (after EN_STAGE_OPROJRES)
  EN_STAGE_DMAPROBE = 22,   // EN_TEST 22: 4-block vs 1-block move-in layout probe
  EN_STAGE_UPROBE = 23,     // EN_TEST 24: SIMT instruction throughput probe (all warps)
  EN_STAGE_FILL = 24,       // poison [gen_addr, + gen_elems bytes) (EN_HASH_OUT coverage checks)
  EN_STAGE_HASH = 25,       // position-sensitive hash of [gen_addr, + gen_elems bytes)
};
// EN_STAGE_DMAPROBE buffers: a 16 x 128 B source tile and the two SMEM images moved back out
#define EN_HASH_OUT  0x1A100000u   // EN_STAGE_HASH: one 64-byte line of lane partials per warp
#define EN_HPM       0x1A200000u   // EN_FX_HPM: perf-counter snapshots [cluster][core][2][32 words]
#define EN_HPM_N     24u           // mcycle, minstret, mhpmcounter3 .. 24
#define EN_PROBE_SRC 0x1A000000u
#define EN_PROBE_W   0x1A010000u   // via 4-block (64 B row) move-ins
#define EN_PROBE_N   0x1A020000u   // via 1-block (16 B row) move-ins
#define EN_PROBE_2   0x1A030000u   // via 2-block (32 B row) move-ins
#define EN_PROBE_M   0x1A038000u   // 2-block move-out of the narrow SMEM image (expected = source)
#define EN_PROBE_T   0x1A040000u   // 3 timing words, then a 64 KB move-out target at +0x10000

// FA_ENC argument block (must match FaEncArgs in kernels/fa_stable_fpga/kernel.cpp)
struct ENFaArgs { uint32_t q, qs, kt, kts, v, vs, o; };

struct ENArgs {
  uint32_t stage;
  uint32_t clear_lo, clear_hi;   // EN_STAGE_CLEAR byte range (0, 0: all of SMEM)
  uint32_t gen_addr, gen_elems, gen_stream;   // EN_STAGE_GEN, EN_STAGE_GENW
  uint32_t gen_saddr;                         // EN_STAGE_GENW scales
};

// printBuf slots (32..47 free): GPU thread-0 timestamps per launch
#define EN_PB_T_START 32u
#define EN_PB_T_END   33u
#define EN_PB_UPROBE  48u   // 48..60: EN_TEST 24 cycles per test (warp 0 of core 0)
#define EN_PB_PROF    40u   // 40..44: DMA stages, thread 0's cycles in move-in, barriers, work, move-out, issue (part of move-in)

// ---- test data: integer hash shared by host.cpp and gen_ref.py -----------------------------
// stream ids (hash input = index * 8 + stream)
#define EN_ST_X     0u
#define EN_ST_LN    1u
#define EN_ST_WQKV  2u
#define EN_ST_WO    3u
#define EN_ST_W1    4u
#define EN_ST_W2    5u
#define EN_ST_WSC   6u
#define EN_ST_GEN_FF  10u   // stage tests (EN_TEST 13): GELU input
#define EN_ST_GEN_O   11u   //   attention output (FA layout)
#define EN_ST_GEN_ATT 12u   //   out-projection output
#define EN_ST_GEN_FF2 13u   //   FFN-down output

#endif  // TAPEOUT_ENCODER_H
