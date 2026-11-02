// fa_stage.cpp: the stF24 MX-FP8 flash-attention kernel (kernels/fa_stable_fpga) compiled in
// FA_ENC mode for the encoder image.  One launch computes all 4 heads: 16 query tiles of 64 rows,
// each head's 4 tiles against that head's 256 keys (FA_ENC_TPH); head 0's operand and output
// addresses come from the FaEncArgs block at EN_FA_ARGS.  Its main() is renamed fa_image_main(); the encoder's main() calls it for the
// attention stage.  The define set is the one verified on the board at 37% steady state.

#include "enc.h"

#define FA_ENC 1
#if EN_CLUSTERS > 1
// two clusters: cluster c reads its own args block (EN_FA_ARGS + 0x100 c, host set_fa_args) and
// runs 2 heads = 8 query tiles
static inline uint32_t fa_enc_cluster() {
  uint32_t v;
  asm volatile("csrr %0, 0xCD0" : "=r"(v));
  return v;
}
#define FA_ENC_ARGS (EN_FA_ARGS + 0x100u * fa_enc_cluster())
#else
#define FA_ENC_ARGS EN_FA_ARGS
#endif
#define FA_FPGA 1
#define FA_MARK_GMEM_ADDR 0x1F100000
#if EN_CLUSTERS > 1
#define FA_MARK_CL_OFF (0x800u * fa_enc_cluster())   // cluster c's stamps at 0x1F100000 + 0x800 c
#endif
#ifndef EN_FA_FULLMARKS   // full per-stage stamps (~400 cycles each, so they inflate the tile)
#define FA_SP_MARK1 1
#endif
#define FA_EPILOGUE 1
#if EN_CLUSTERS > 1
#define FA_NT8 1        // 2 heads per cluster in one launch: 8 query tiles
#else
#define FA_NT16 1       // all 4 heads in one launch: 16 query tiles
#endif
#define FA_ENC_TPH 4    // tiles per head (S / 64)
#define FA_ENC_ECHO 1
#ifdef EN_FA_OQ
// EN_FA_OQ: attention's finalize quantizes O (bf16, normalized in SMEM) to the out-projection's MX
// A operand (EN_OA codes [S][D], EN_OA_S scales) and the o_quant stage is skipped.  Bit-identical
// codes: these helpers are copies of tapeout_encoder/kernel.cpp's (quant_group_bf16 path).
#define FA_ENC_OQ 1
#ifdef EN_FA_OQROT
#define FA_OQ_ROT 1    // rotated per-lane word order (SMEM subbank conflicts)
#endif
#ifdef EN_FA_OQFUSE
#define FA_OQ_FUSE 1   // finalize + quantize in one pass
#endif
static inline uint32_t oq_mask_lt(uint32_t a, uint32_t b) {
  uint32_t m;
  asm("sltu %0, %1, %2\n\tneg %0, %0" : "=r"(m) : "r"(a), "r"(b));
  return m;
}
static inline int oq_block_exp(uint32_t b) {
  const uint32_t e = (b >> 23) - 127u + (oq_mask_lt(0x700000u, b & 0x7FFFFFu) & 1u);
  return (int)(e & oq_mask_lt(0u, b));
}
static inline uint32_t oq_e4m3_swar2(uint32_t w, uint32_t p2) {
  const uint32_t a = w & 0x7FFF7FFFu;
  uint32_t t = a + 0x00070007u + ((a >> 4) & 0x00010001u);
  t = ((t >> 4) & 0x0FFF0FFFu) + p2;
  const uint32_t k = (t >> 12) & 0x00010001u;
  return ((t + 0x00080008u) | ((w >> 8) & 0x00800080u)) & ((k << 8) - k);
}
static inline uint32_t oq_fmax_h(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("fmax.h %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
static inline uint32_t oq_swar_p2(int e) { const uint32_t p = 0x1000u - (uint32_t)(8 * e + 968); return p | (p << 16); }
static inline uint32_t oq_swar_pack4(uint32_t r0, uint32_t r1) {
  return ((r0 | (r0 >> 8)) & 0xFFFFu) | ((r1 | (r1 >> 8)) << 16);
}
#endif
#ifdef EN_FA_DMAO
#define FA_ENC_DMAO 1    // O via SMEM + DMA move-out (experiment)
#endif
#ifdef EN_FA_KVOVL
#define FA_ENC_KVOVL 1   // V(h+1) reload under QK(t+1) at head switches (experiment)
#endif
#ifdef EN_FA_WIDE_A
#define FA_MVIN_WIDE_A 1
#endif
#ifdef EN_FA_WIDE_B
#define FA_MVIN_WIDE_B 1
#endif
#ifdef EN_FA_OVLQK
#define FA_ST_OVL_QK 1   // QK(t+1) under finalize(t) (FA_ST_NOOVL put-back)
#endif
#ifdef EN_FA_DMAO2
#define FA_ENC_DMAO2 1   // two-phase finalize: first half of O moves out under the second half
#endif
#ifdef EN_FA_DMAO_NOMV
#define FA_ENC_DMAO_NOMV 1
#endif
#ifdef EN_FA_SERQK
#define FA_ST_SER_QK 1   // drain QK(t+1) in S6a (finalize runs with the mesh idle)
#endif
#ifdef EN_STAGE_HPM   // counter snapshots at attention start / end (kernel.cpp en_hpm_snap)
extern void en_hpm_snap(uint32_t tid, uint32_t idx);
#define FA_HPM_SNAP(t, i) en_hpm_snap((t), (i))
#endif
#ifdef EN_FA_OQ_UNROLL
#define FA_OQ_UNROLL EN_FA_OQ_UNROLL   // finalize / convert inner loops unrolled n times
#endif
#ifdef EN_FA_KTEARLY
#define FA_ENC_KTEARLY 1   // K^T(h+1) loads under the softmax of the head's last tile
#endif
#ifdef EN_FA_PROSCL
#define FA_ENC_PROSCL 1    // prologue K / V scales from thread 16, under the K^T / V move-ins
#endif
#ifdef EN_FA_CVT_C
#define FA_CVT_C 1       // e4m3 packers in C, not one asm block each (flash_mx_impl.hpp)
#endif
#ifdef EN_FA_CVTXS
#define FA_SP_CVTXS 1    // SWAR P convert (25 instead of 34 instructions per output word)
#endif
#ifdef EN_FA_PROLOGUE_MARKS
#define FA_PROLOGUE_MARKS 1
#endif
#ifdef EN_FA_WIDE
#define FA_MVIN_WIDE 1   // 4-block-wide move-ins (experiment)
#endif
#define FA_ENC_PB_T_START EN_PB_T_START   // per-launch cycle stamps for the host
#define FA_ENC_PB_T_END   EN_PB_T_END
// stF24
#define FULL_ATTN2 1
#define FA_SP 1
#define FA_SP_QOVL 1
// FA_SP_LEANCFG is ON (the full per-gemm configure gives wrong output with this define set:
// 468k ppm in metasim enc_m7c).  OPEN: a second FA launch in the same session trips
// AccumulatorMem.scala:634 (metasim); see TAPEOUT_PERF_STATUS.md.  EN_FA_FULLCFG: experiments.
#ifndef EN_FA_FULLCFG
#define FA_SP_LEANCFG 1
#endif
#define FA_SP_QKACC 1
#define FA_SP_PKOVL 1
#define FA_SP_QSPLIT 1
#define FA_SP_WCNT 1
#define FA_SP_PAX 1
#define FA_SP_CVTX 1
#define FA_ST_NOOVL 1
#define FA_SP_ACCRS 1
#define FA_SP_PREPK 1
#define FA_ST_OVL_SCL 1
#define FA_ST_OVL_DMA 1
#define FA_SM_2P 1
#define FA_SM_2PRAW 1

#define main fa_image_main
#include "../fa_stable_fpga/kernel.cpp"
#undef main
