// llama_e2e: a whole Llama-family model (prefill chunk over a KV cache) on Radiance, every stage on the device.
//
// The data generator (gen_data.py) lays out the model and emits a PHASE PROGRAM: one 32-word descriptor per
// GEMM (gemmini) or row operation (Muon warps).  This kernel interprets it.  Phases are separated by a
// cross-cluster handshake through printBuf: the host writes GO = p + 1 to every cluster once all of them have
// posted DONE = p (1 cluster: the host writes GO = NP up front and the cluster runs straight through).  Each
// cluster's thread 0 stamps mcycle at the start and end of every phase into a DRAM table (LL_PERF_ADDR) that
// the host reads after the run; there are no RTL performance counters involved.
//
// Phase types (field layout in gen_data.py):
//   GEMM     C[bf16] = A[MX] x B[MX], batched; tiles (batch, m-tile, n-tile of 64) split contiguously over the
//            clusters; per tile KS k-steps with A and B double-buffered in the scratchpad, C stored from the
//            accumulator to DRAM (two accumulator halves alternate between tiles).
//   PRENORM  h = x + add in bf16 (written out);  MX(rmsnorm(h) * gamma)                         warp per row
//   ROPE     Q, K rope + MX (Q as the QK A operand, K into the K^T cache), V into the V cache    lane per item
//   SOFTMAX  causal softmax of S / sqrt(hd), rows normalized, MX (PV A operand)                 warp per row
//   EWQ      MX of a bf16 matrix                                                             lane per 32-group
//   SILU     MX of silu(G) * U                                                               lane per 32-group
//   FILL     pseudo-random E4M3 codes (mode 0) or a constant word (mode 1): synthetic weights, before timing
//
// MX: one E8M0 scale per 32 elements along K, e = floor(log2(amax)) (block max in [1, 2): the MX Gemmini's tile
// accumulator overflows on larger codes), codes by RNE (golden.fp8_code_rshift for |x| < 448).
#include <nightly/device.h>
#define NIGHTLY_MX_WAIT_READY
#define NIGHTLY_MX_RETIRE                    // loop retire counter: the ATTN phase's producer waits on it
#define NIGHTLY_MX_LOOPS_SMEM (8000u * 16u)  // the cluster's LOOP_WS count (above every phase's SMEM map)
#include <nightly/mx.h>
#include "llama_data.h"

constexpr uint32_t WARPS = MU_NUM_CORES * NIGHTLY_OCC;   // warps per cluster
constexpr uint32_t NC = NIGHTLY_CLUSTERS;
constexpr uint32_t PB_GO = 20, PB_DONE = 21;            // printBuf slots of the phase handshake
constexpr uint32_t BAR_PH = 1;
constexpr uint32_t RED_SM = 0x1C000;                     // per-warp 64 B reduction buffers (above the GEMM map)

static_assert(WARPS * 64 + RED_SM <= 0x20000, "SMEM budget");

static inline uint32_t opaque(uint32_t x) { asm volatile("" : "+r"(x)); return x; }
__attribute__((noinline, convergent)) static void bar(uint32_t id, uint32_t warps) { mu_barrier(id, warps); }

// ---- scalar helpers ---------------------------------------------------------------------------
static inline float f_bits(uint32_t u) { return __builtin_bit_cast(float, u); }
static inline uint32_t bits_f(float f) { return __builtin_bit_cast(uint32_t, f); }
static inline float lo(uint32_t w) { return f_bits(w << 16); }
static inline float hi(uint32_t w) { return f_bits(w & 0xFFFF0000u); }
static inline float bfv(uint32_t h16) { return f_bits(h16 << 16); }
static inline uint32_t bf16(float f) {                   // round to nearest even
  const uint32_t u = bits_f(f);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
static inline float bfr(float f) { return f_bits(bf16(f) << 16); }
static inline float fabs_(float f) { return f_bits(bits_f(f) & 0x7FFFFFFFu); }
static inline float fexp_bf(float x) {                    // Muon fexp.h on bf16(x), result widened to fp32
  const _Float16 r = mu_fexp(__builtin_bit_cast(_Float16, (uint16_t)bf16(x)));
  return f_bits((uint32_t)__builtin_bit_cast(uint16_t, r) << 16);
}

// Branch-free selects.  Base RV32 has no conditional move, so a C ternary on lane-varying data becomes a
// divergent branch (split / join); nested ones overflow the warp's 8-entry IPDOM stack.  The mask goes through
// opaque() so instcombine cannot turn the and/or back into a select.
static inline uint32_t mask_if(uint32_t c) { return opaque(0u - c); }      // c in {0, 1}
static inline uint32_t sel(uint32_t c, uint32_t a, uint32_t b) { const uint32_t m = mask_if(c); return (a & m) | (b & ~m); }
static inline uint32_t bit(bool c) { return opaque((uint32_t)c); }       // a compare result as 0 / 1

// E8M0 exponent of a 32-group: floor(log2(amax)), so the block maximum maps into [1, 2) (at least -126, amax = 0
// included).  The MX Gemmini accumulates a 16-deep tile in a reduced-range format: codes near the E4M3 maximum (448)
// overflow it to inf / NaN (golden.rand_fp8; mx_golden returns all NaN for 448-scaled blocks).  FA's requantizer uses
// the same [1, 2) convention.
static inline int32_t block_exp(float amax) {
  const int32_t e = (int32_t)((bits_f(amax) >> 23) & 0xFF) - 127;
  return (int32_t)sel((uint32_t)(e < -126), (uint32_t)-126, (uint32_t)e);
}
static inline float pow2_neg(int32_t e) { return f_bits((uint32_t)(127 - e) << 23); }

// fp32 -> e4m3, RNE, subnormals kept, for |x| <= 448 (block scaling keeps |x| < 2; no saturation needed).  Normal: round the mantissa to 3 bits in place and rebias the exponent 127 -> 7.  Subnormal
// (|x| < 2^-6): rne(|x| * 2^9) by the 2^23 float trick.  Bit-exact with golden.fp8_code_rshift in that range
// (gen_data.py checks it).  Needs -ffp-contract=off (a fused multiply-add would round differently).
static inline uint32_t fp8(float x) {
  const uint32_t u = bits_f(x), s = (u >> 24) & 0x80u, ax = u & 0x7FFFFFFFu;
  const uint32_t norm = ((ax + 0x7FFFFu + ((ax >> 20) & 1u)) >> 20) - (120u << 3);
  const uint32_t sub = bits_f(f_bits(ax) * 512.0f + 8388608.0f) - 0x4B000000u;
  return s | sel(bit(ax < 0x3C800000u), sub, norm);
}
static inline uint32_t pack4(float a, float b, float c, float d) {
  return fp8(a) | (fp8(b) << 8) | (fp8(c) << 16) | (fp8(d) << 24);
}

// A operand (pretiled [MT][KS][TM][TK] codes, [MT][KS][TK/32][TM] scales): addresses of row r, column k
static inline uint32_t a_code(uint32_t a, uint32_t TM, uint32_t TK, uint32_t KS, uint32_t r, uint32_t k) {
  return a + (((r / TM) * KS + k / TK) * TM + r % TM) * TK + k % TK;
}
static inline uint32_t a_scale(uint32_t asc, uint32_t TM, uint32_t TK, uint32_t KS, uint32_t r, uint32_t k) {
  return asc + (((r / TM) * KS + k / TK) * (TK / 32) + (k % TK) / 32) * TM + r % TM;
}

static inline volatile uint32_t *gw32(uint32_t a) { return (volatile uint32_t *)a; }
static inline volatile uint16_t *gw16(uint32_t a) { return (volatile uint16_t *)a; }
static inline volatile uint8_t *gw8(uint32_t a) { return (volatile uint8_t *)a; }

// =================================== GEMM (thread 0 of the cluster) ===================================
static uint32_t scale_sink[64] __attribute__((aligned(64)));   // dummy requantizer flush address
static uint32_t acc_parity = 0;                                // accumulator half of the next tile

struct GemmDesc {
  uint32_t a, asc, a_bs, asc_bs, b, bsc, b_bs, bsc_bs, c, c_bs, c_rs, c_ms, batch, MT, NT, KS, TM, TK;
};

// loads of step g (tile t0 + g / KS, k-step g % KS) into scratchpad buffer `buf`
__attribute__((always_inline)) static inline void gemm_loads(const GemmDesc &q, uint32_t t0, uint32_t g, uint32_t buf) {
  const uint32_t t = t0 + g / q.KS, s = g % q.KS;
  const uint32_t nt = t % q.NT, mt = (t / q.NT) % q.MT, bt = t / (q.NT * q.MT);
  const uint32_t A_ROWS = q.TM * q.TK / 16, B_ROWS = q.TK * LL_TN / 16;
  const uint32_t asc_step = q.TM * q.TK / 32, bsc_step = q.TK * LL_TN / 32;
  mx::set_ab_dram(mx::host_addr(q.a + bt * q.a_bs + (mt * q.KS + s) * q.TM * q.TK),
                  mx::host_addr(q.b + bt * q.b_bs + (s * q.NT + nt) * q.TK * LL_TN), q.TK, LL_TN);
  mx::loop_ws_spad(q.TM / 16, 4, q.TK / 16, buf * A_ROWS, 2 * A_ROWS + (buf + 1) * B_ROWS, 0, false, false,
                   {false, false, true, true, true});
  mx::load_scales(mx::host_addr(q.asc + bt * q.asc_bs + (mt * q.KS + s) * asc_step), asc_step, false,
                  buf ? mx::SF_BUF1 : 0);
  mx::load_scales(mx::host_addr(q.bsc + bt * q.bsc_bs + (nt * q.KS + s) * bsc_step), bsc_step, true,
                  buf ? mx::SF_BUF1 : 0);
}

// accumulator half `half` (tile t) -> C in DRAM, row-major bf16 with row stride c_rs
__attribute__((always_inline)) static inline void gemm_mvout(const GemmDesc &q, uint32_t t, uint32_t half) {
  const uint32_t nt = t % q.NT, mt = (t / q.NT) % q.MT, bt = t / (q.NT * q.MT);
  const uint64_t base = mx::host_addr(q.c + bt * q.c_bs + mt * q.c_ms + nt * LL_TN * 2);
  for (uint32_t i = 0; i < q.TM / 16; i++) {
    mx::mvout_acc(base + (uint64_t)i * 16 * q.c_rs, half * mx::ACC_HALF + i * 16, 16, 16, 0);
    mx::mvout_acc(base + (uint64_t)i * 16 * q.c_rs + 64, half * mx::ACC_HALF + i * 16, 16, 16, 1);
  }
}

// Fused GEMMs (d[20] = kind: 1 SILU, 2 ROPE, 3 PRENORM accumulation): while the GEMM thread runs, warps 1.. work on
// C tiles as they land.  The GEMM thread publishes, per tile, the LOOP_WS whose retirement proves the tile's mvout
// landed (the next loop issued after it), the retire counter itself once per k-step (so the workers poll SMEM, not the
// MMIO path the GEMM thread issues on), and DONE after its final fence.  Tiles split over clusters in pairs (one head;
// G_j, U_j).  All of it lives above every phase's SMEM map, next to the loop count.
enum : uint32_t { FUSE_SILU = 1, FUSE_ROPE = 2, FUSE_NORM = 3 };
constexpr uint32_t FUSE_DONE_SM = NIGHTLY_MX_LOOPS_SMEM + 4, FUSE_RET_SM = NIGHTLY_MX_LOOPS_SMEM + 8;
constexpr uint32_t FUSE_TAB_SM = NIGHTLY_MX_LOOPS_SMEM + 12, FUSE_SS_SM = 0x1FC00;   // FUSE_SS: PRENORM row sums
constexpr uint32_t FUSE_TAB_MAX = (FUSE_SS_SM - FUSE_TAB_SM) / 4;
constexpr uint32_t FUSE_STAGE_SM = 0x18000, FUSE_STAGE_BYTES = 0x1000;   // above the GEMM's A / B buffers (6144 rows)
static_assert(FUSE_STAGE_SM + (WARPS - 1) * FUSE_STAGE_BYTES <= NIGHTLY_MX_LOOPS_SMEM, "fused SMEM");

__attribute__((noinline)) static void gemm(const volatile uint32_t *d, uint32_t tb) {
  GemmDesc q;
  q.a = d[2]; q.asc = d[3]; q.a_bs = d[4]; q.asc_bs = d[5]; q.b = d[6]; q.bsc = d[7]; q.b_bs = d[8];
  q.bsc_bs = d[9]; q.c = d[10]; q.c_bs = d[11]; q.c_rs = d[12]; q.c_ms = d[13]; q.batch = d[14];
  q.MT = d[15]; q.NT = d[16]; q.KS = d[17]; q.TM = d[18]; q.TK = d[19];
  const uint32_t total = q.batch * q.MT * q.NT;
  const bool fused = d[20] != 0 && (d[20] != FUSE_NORM || NC == 1);   // fused: tile pairs stay on one cluster
  const uint32_t t0 = fused ? 2 * (total / 2 * tb / NC) : total * tb / NC;
  const uint32_t t1 = fused ? 2 * (total / 2 * (tb + 1) / NC) : total * (tb + 1) / NC;
  if (t0 == t1) {
    if (fused) mx::st32(FUSE_DONE_SM, 1);
    return;
  }
  const uint32_t I = q.TM / 16, KT = q.TK / 16;
  const uint32_t A_ROWS = q.TM * q.TK / 16, B_ROWS = q.TK * LL_TN / 16;
  mx::config_ld(q.TK, 0);
  mx::config_ld(LL_TN, 1);
  mx::config_st(q.c_rs);
  const uint32_t G = (t1 - t0) * q.KS;
  gemm_loads(q, t0, 0, 0);
  for (uint32_t g = 0; g < G; g++) {
    const uint32_t s = g % q.KS, buf = g & 1u;
    const bool last = s == q.KS - 1;
#if defined(LL_GEMM_FENCE) && LL_GEMM_FENCE
    // safe mode: everything issued so far (this step's loads included) has completed before its computes
    if (g > 0) mx::fence();
#endif
    // the config waits for this buffer's scale loads and is ordered with the computes
    mx::config_scale(I, 4, KT, buf, buf, mx::host_addr(scale_sink));
    mx::loop_ws_spad(I, 4, KT, buf * A_ROWS, 2 * A_ROWS + (buf + 1) * B_ROWS, 0, s > 0, last,
                     {true, true, true, false, true});
    // step g+1 into the other buffer: the reservation station orders the mvins after the computes of step
    // g-1 that read it; the scale loads wait until loop g is unrolled, when step g-1 has executed
    if (g + 1 < G) gemm_loads(q, t0, g + 1, buf ^ 1u);
    // the previous tile's stores go behind this tile's first compute
    if (s == 0 && g > 0) {
      const uint32_t t = t0 + g / q.KS - 1;
      gemm_mvout(q, t, (acc_parity + (t - t0)) & 1u);
      if (fused) mx::st32(FUSE_TAB_SM + 4 * (t - t0), mx::loops_issued() + 1);
    }
    if (fused) mx::st32(FUSE_RET_SM, mx::ld32(mx::RETIRED));
  }
  gemm_mvout(q, t1 - 1, (acc_parity + (t1 - 1 - t0)) & 1u);
  mx::fence();                                           // computes done, C stores acknowledged
  acc_parity ^= (t1 - t0) & 1u;
  if (fused) mx::st32(FUSE_DONE_SM, 1);
}

// =================================== Muon row operations ===================================
// A global load takes ~500+ cycles (L1 miss to DRAM): every op issues its loads in coalesced batches before using
// them, stages rows in SMEM for the later passes, and quantizes a 32-element MX group with the 16 lanes of a warp
// (2 elements per lane, cross-lane max through SMEM).  Data pointers are not volatile so loads can be scheduled.
constexpr uint32_t STAGE_SM = 0;                         // per-warp row stage, WARPS x 8 KiB (D <= 4096 bf16)
constexpr uint32_t STAGE_BYTES = 8192;
constexpr uint32_t GAMMA_SM = WARPS * STAGE_BYTES;       // gamma, 8 KiB
static_assert(GAMMA_SM + 8192 <= RED_SM, "SMEM map");
constexpr uint32_t CW_SM = RED_SM + WARPS * 64;          // per-warp 32-word code buffer
constexpr uint32_t SB_SM = GAMMA_SM + 8192, SB_BYTES = 2048;   // per-warp scale words of a 4-row block (<= 512 groups)
static_assert(SB_SM + WARPS * SB_BYTES <= RED_SM, "SMEM map");
static_assert(CW_SM + WARPS * 128 <= 0x20000, "SMEM map");

static inline const uint32_t *g32(uint32_t a) { return (const uint32_t *)a; }
static inline uint32_t *o32(uint32_t a) { return (uint32_t *)a; }
static inline volatile __shared uint32_t *sm32(uint32_t a) { return (volatile __shared uint32_t *)a; }

// warp-wide sum / max through the warp's SMEM reduction buffer (no shuffles on Muon)
static inline float warp_sum(volatile __shared float *red, uint32_t lane, float v) {
  red[lane] = v;
  mu_fence_smem();
  float s = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < MU_NUM_THREADS; j++) s += red[j];
  mu_fence_smem();                                       // every lane has read before the buffer is reused
  return s;
}
static inline float warp_max(volatile __shared float *red, uint32_t lane, float v) {
  red[lane] = v;
  mu_fence_smem();
  float m = red[0];
#pragma unroll
  for (uint32_t j = 1; j < MU_NUM_THREADS; j++) m = __builtin_fmaxf(m, red[j]);
  mu_fence_smem();
  return m;
}
// Global stores: on Muon, sub-word stores from several lanes of ONE store instruction into the same 32-bit word lose
// all but one lane's bytes (FireSim, llama_e2e muontest).  So codes are assembled into whole words through the warp's
// SMEM word buffer (`cw`, 32 words) and each word is stored by one lane; scale bytes are stored by lane 0 only.
static inline void store_byte_lane0(uint32_t lane, uint32_t addr, uint32_t v) {
  if (lane == 0) *(uint8_t *)addr = (uint8_t)v;
  else asm volatile("nop");
}
// MX-quantize one 32-group held 2 values per lane (lane l: elements 2l, 2l+1); codes at code_addr (32 contiguous
// bytes, 4-byte aligned), the scale byte at scale_addr
// Scale bytes: adjacent rows' scales are adjacent bytes of one word and are written by different warps at the same
// time, and those byte stores collide too.  So a warp owns blocks of 4 rows, collects the block's scale bytes in its
// SMEM scale words sb[group] (row rr in byte rr, lane 0 only) and stores whole words (sb_flush).
static inline void mx_group2(volatile __shared float *red, volatile __shared uint32_t *cw, uint32_t lane, float v0,
                             float v1, uint32_t code_addr, volatile __shared uint32_t *sbw, uint32_t rr) {
  const float amax = warp_max(red, lane, __builtin_fmaxf(fabs_(v0), fabs_(v1)));
  const int32_t e = block_exp(amax);
  const float sc = pow2_neg(e);
  cw[lane] = fp8(v0 * sc) | (fp8(v1 * sc) << 8);
  mu_fence_smem();
  if (lane < 8) *(uint32_t *)(code_addr + 4 * lane) = cw[2 * lane] | (cw[2 * lane + 1] << 16);
  else asm volatile("nop");
  mu_fence_smem();                                       // every lane has read cw before it is rewritten
  if (lane == 0) *sbw = (rr ? *sbw : 0u) | ((uint32_t)(e + 127) << (8 * rr));
  else asm volatile("nop");
}
// store the scale words of a 4-row block starting at row rb: sb[g] is group g0 + g's word, at a_scale(rb, 32 (g0 + g))
static inline void sb_flush(volatile __shared uint32_t *sb, uint32_t lane, uint32_t groups, uint32_t g0, uint32_t asc,
                            uint32_t TM, uint32_t TK, uint32_t KS, uint32_t rb) {
  mu_fence_smem();
  for (uint32_t g = lane; g < groups; g += MU_NUM_THREADS)
    *(uint32_t *)a_scale(asc, TM, TK, KS, rb, (g0 + g) * 32) = sb[g];
  mu_fence_smem();
}
// same with the group's elements at positions lane and lane + 16.  stride 1: a row (assembled into 8 words);
// stride 64: a K^T column (one byte per lane, each in its own word)
static inline void mx_group_split(volatile __shared float *red, volatile __shared uint32_t *cw, uint32_t lane, float v0,
                                  float v1, uint32_t code_addr, uint32_t stride, uint32_t scale_addr) {
  const float amax = warp_max(red, lane, __builtin_fmaxf(fabs_(v0), fabs_(v1)));
  const int32_t e = block_exp(amax);
  const float sc = pow2_neg(e);
  const uint32_t c0 = fp8(v0 * sc), c1 = fp8(v1 * sc);
  if (stride == 1) {
    cw[lane] = c0;
    cw[lane + 16] = c1;
    mu_fence_smem();
    if (lane < 8)
      *(uint32_t *)(code_addr + 4 * lane) = cw[4 * lane] | (cw[4 * lane + 1] << 8) | (cw[4 * lane + 2] << 16) |
                                            (cw[4 * lane + 3] << 24);
    else asm volatile("nop");
    mu_fence_smem();
  } else {
    *(uint8_t *)(code_addr + lane * stride) = (uint8_t)c0;
    *(uint8_t *)(code_addr + (lane + 16) * stride) = (uint8_t)c1;
  }
  store_byte_lane0(lane, scale_addr, (uint32_t)(e + 127));
}

// ---- Line ownership -------------------------------------------------------------------------------
// Two cores writing different bytes or words of the same cache line in one phase lose one core's writes (per-core
// write-back L0d; subword test T6 / T10).  Every row op therefore splits its output into units whose cache lines
// belong to that unit alone, and runs each unit on ONE core (owner own of nown = clusters x cores); that core's
// warps (wi of nwo) share the unit's rows.  Warps of one core may share lines (T5 passes).  Each phase ends with a
// fence, which flushes the L0d, so units of different phases do not interact.
// For an MX A operand ([MT][KS][TM][TK] codes, [MT][KS][TK/32][TM] scales) the unit is (m-tile, k-step): its codes
// are TM blocks of TK bytes, its scale lines hold 64 rows of one 32-column group of that k-step.

// ---- PRENORM: unit = (m-tile, k-step) ----------------------------------------------------------------
// h = bf16(x + add) (written to hout), then MX(rmsnorm(h) * gamma); layer 0 passes a zero `add`.  Each unit computes
// the full-row sum of squares of its rows itself (KS-fold extra reads, no cross-unit communication).
static inline uint32_t add_bf2(uint32_t x, uint32_t a) {
  return bf16(lo(x) + lo(a)) | (bf16(hi(x) + hi(a)) << 16);
}
static void prenorm(const volatile uint32_t *d, uint32_t tid, uint32_t tpb, uint32_t own, uint32_t nown, uint32_t wi,
                    uint32_t nwo, uint32_t lane, volatile __shared float *red, volatile __shared uint32_t *cw,
                    volatile __shared uint32_t *sb) {
  const uint32_t x = d[2], add = d[3], hout = d[4], gamma = d[5], D = d[8], M = d[9];
  const uint32_t A = d[6], ASC = d[7], TM = d[10], TK = d[11], KS = D / TK, GS = TK / 32, MT = M / TM;
  const float eps = f_bits(d[12]);
  const uint32_t W = D / 2;                              // words per row (a multiple of 128)
  volatile __shared uint32_t *gsm = sm32(GAMMA_SM);
  for (uint32_t i = tid; i < W; i += tpb) gsm[i] = g32(gamma)[i];
  mu_fence_smem();
  bar(2, tpb / MU_NUM_THREADS);
  for (uint32_t u = own; u < MT * KS; u += nown) {
    const uint32_t mt = u / KS, s = u % KS;
    for (uint32_t rb = mt * TM + 4 * wi; rb < (mt + 1) * TM; rb += 4 * nwo) {
      for (uint32_t rr = 0; rr < 4; rr++) {
        const uint32_t r = rb + rr;
        const uint32_t *X = g32(x + r * D * 2), *AD = g32(add + r * D * 2);
        uint32_t *H = o32(hout + r * D * 2);
        float ss = 0.f;
        for (uint32_t w0 = lane; w0 < W; w0 += 8 * MU_NUM_THREADS) {
          uint32_t xv[8], av[8];
#pragma unroll
          for (uint32_t k = 0; k < 8; k++) { xv[k] = X[w0 + k * MU_NUM_THREADS]; av[k] = AD[w0 + k * MU_NUM_THREADS]; }
#pragma unroll
          for (uint32_t k = 0; k < 8; k++) {
            const uint32_t hv = add_bf2(xv[k], av[k]);
            ss += lo(hv) * lo(hv) + hi(hv) * hi(hv);
          }
        }
        ss = warp_sum(red, lane, ss);
        const float inv = 1.0f / __builtin_sqrtf(ss / (float)D + eps);
        for (uint32_t gi = 0; gi < GS; gi++) {           // this k-step's groups: words g*16 + lane
          const uint32_t g = s * GS + gi, w = g * 16 + lane;
          const uint32_t hv = add_bf2(X[w], AD[w]), gv = gsm[w];
          H[w] = hv;                                     // h columns of this k-step (512 B per row)
          mx_group2(red, cw, lane, lo(hv) * inv * lo(gv), hi(hv) * inv * hi(gv), a_code(A, TM, TK, KS, r, g * 32),
                    sb + gi, rr);
        }
      }
      sb_flush(sb, lane, GS, s * GS, ASC, TM, TK, KS, rb);
    }
  }
}

// ---- Row ops, lane per group (needs the coalescer byte-merge and L0d dirty-mask RTL fixes) ------------------
// A warp owns whole rows.  A lane quantizes whole 32-groups (amax lane-local), puts the group's 8 code words in the
// warp's code buffer and stores its scale byte directly; the warp then stores the buffer with coalesced stores.
// SMEM indices are rotated by lane so the 16 lanes hit different banks.
constexpr uint32_t CB_GROUPS = SB_BYTES / 32;             // groups per code-buffer flush
static inline float silu_mul(float g, float u) { return g / (1.0f + fexp_bf(-g)) * u; }
struct F2 { float a, b; };

// pair(j), j < 16: elements 2j, 2j + 1 of the group (evaluated twice: amax pass, code pass); cbg: SMEM or global
template <typename P, typename W>
__attribute__((always_inline)) static inline void quant_group(P pair, uint32_t lane, W *cbg, uint32_t scale_addr) {
  float amax = 0.f;
#pragma unroll 4
  for (uint32_t jj = 0; jj < 16; jj++) {
    const F2 p = pair((jj + lane) & 15u);
    amax = __builtin_fmaxf(amax, __builtin_fmaxf(fabs_(p.a), fabs_(p.b)));
  }
  const int32_t e = block_exp(amax);
  const float sc = pow2_neg(e);
#pragma unroll 2
  for (uint32_t jj = 0; jj < 8; jj++) {
    const uint32_t j = (jj + (lane >> 1)) & 7u;
    const F2 p0 = pair(2 * j), p1 = pair(2 * j + 1);
    cbg[j] = pack4(p0.a * sc, p0.b * sc, p1.a * sc, p1.b * sc);
  }
  *(volatile uint8_t *)scale_addr = (uint8_t)(e + 127);
}

// code buffer (groups g0 .. g0 + ng of row r) -> the A operand, coalesced
static inline void cb_flush(volatile __shared uint32_t *cb, uint32_t lane, uint32_t ng, uint32_t g0, uint32_t A,
                            uint32_t TM, uint32_t TK, uint32_t KS, uint32_t r) {
  mu_fence_smem();
  for (uint32_t w = lane; w < ng * 8; w += MU_NUM_THREADS) *(uint32_t *)a_code(A, TM, TK, KS, r, 32 * g0 + 4 * w) = cb[w];
  mu_fence_smem();                                       // read before the buffer is rewritten
}

// PRENORM: h = bf16(x + add) (written to hout), then MX(rmsnorm(h) * gamma); warp per row, the row staged in SMEM
static void prenorm_rows(const volatile uint32_t *d, uint32_t tid, uint32_t tpb, uint32_t gw, uint32_t ngw,
                         uint32_t lane, volatile __shared float *red, volatile __shared uint32_t *stage,
                         volatile __shared uint32_t *cb) {
  const uint32_t x = d[2], add = d[3], hout = d[4], gamma = d[5], D = d[8], M = d[9];
  const uint32_t A = d[6], ASC = d[7], TM = d[10], TK = d[11], KS = D / TK, NG = D / 32;
  const float eps = f_bits(d[12]);
  const uint32_t W = D / 2;                              // words per row (a multiple of 128)
  volatile __shared uint32_t *gsm = sm32(GAMMA_SM);
  for (uint32_t i = tid; i < W; i += tpb) gsm[i] = g32(gamma)[i];
  mu_fence_smem();
  bar(2, tpb / MU_NUM_THREADS);
  // d[13] = 1: the GEMM before accumulated h and its row sums (norm_fused); only quantize (one cluster)
  const bool acc = NC == 1 && d[13] != 0;
  for (uint32_t r = gw; r < M; r += ngw) {
    const uint32_t *X = g32(x + r * D * 2), *AD = g32(add + r * D * 2);
    uint32_t *H = o32(hout + r * D * 2);
    float ss = 0.f;
    if (acc) {
      for (uint32_t w0 = lane; w0 < W; w0 += 8 * MU_NUM_THREADS) {
        uint32_t hv[8];
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) hv[k] = H[w0 + k * MU_NUM_THREADS];
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) stage[w0 + k * MU_NUM_THREADS] = hv[k];
      }
      mu_fence_smem();
      ss = ((volatile __shared float *)FUSE_SS_SM)[r];
    } else {
      for (uint32_t w0 = lane; w0 < W; w0 += 8 * MU_NUM_THREADS) {
        uint32_t xv[8], av[8];
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) { xv[k] = X[w0 + k * MU_NUM_THREADS]; av[k] = AD[w0 + k * MU_NUM_THREADS]; }
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          const uint32_t hv = add_bf2(xv[k], av[k]), w = w0 + k * MU_NUM_THREADS;
          H[w] = hv;
          stage[w] = hv;
          ss += lo(hv) * lo(hv) + hi(hv) * hi(hv);
        }
      }
      ss = warp_sum(red, lane, ss);                      // its fences also publish the stage
    }
    const float inv = 1.0f / __builtin_sqrtf(ss / (float)D + eps);
    for (uint32_t g0 = 0; g0 < NG; g0 += CB_GROUPS) {
      const uint32_t ng = NG - g0 < CB_GROUPS ? NG - g0 : CB_GROUPS;
      for (uint32_t g = g0 + lane; g < g0 + ng; g += MU_NUM_THREADS)
        quant_group([&](uint32_t j) __attribute__((always_inline)) {
                      const uint32_t w = 16 * g + j, hv = stage[w], gv = gsm[w];
                      return F2{lo(hv) * inv * lo(gv), hi(hv) * inv * hi(gv)};
                    },
                    lane, cb + 8 * (g - g0), a_scale(ASC, TM, TK, KS, r, 32 * g));
      cb_flush(cb, lane, ng, g0, A, TM, TK, KS, r);
    }
  }
}

// EWQ: MX of a bf16 matrix [M][K] (row stride rs elements); SILU: MX(silu(G) * U), G = columns [0, K), U = [K, 2K).
// Warp per row, CB_GROUPS groups per chunk: G (and U) staged in SMEM by coalesced loads.
template <bool SILU>
static void ewq_rows(const volatile uint32_t *d, uint32_t gw, uint32_t ngw, uint32_t lane,
                     volatile __shared uint32_t *stage, volatile __shared uint32_t *cb) {
  const uint32_t src = d[2], rs = d[3], K = d[5], M = d[6];
  const uint32_t A = d[7], ASC = d[8], TM = d[9], TK = d[10], KS = K / TK, NG = K / 32;
  constexpr uint32_t UOFF = CB_GROUPS * 16;              // U's words in the stage
  static_assert(2 * UOFF * 4 <= STAGE_BYTES, "stage");
  for (uint32_t t = gw; t < M; t += ngw) {
    for (uint32_t g0 = 0; g0 < NG; g0 += CB_GROUPS) {
      const uint32_t ng = NG - g0 < CB_GROUPS ? NG - g0 : CB_GROUPS, nw = ng * 16;
      const uint32_t *X = g32(src + (t * rs + g0 * 32) * 2), *U = g32(src + (t * rs + K + g0 * 32) * 2);
      for (uint32_t w0 = lane; w0 < nw; w0 += 8 * MU_NUM_THREADS) {
        uint32_t xv[8], uv[8];
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          const uint32_t w = w0 + k * MU_NUM_THREADS;
          xv[k] = w < nw ? X[w] : 0u;
          if (SILU) uv[k] = w < nw ? U[w] : 0u;
        }
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          const uint32_t w = w0 + k * MU_NUM_THREADS;
          if (w < nw) {
            stage[w] = xv[k];
            if (SILU) stage[UOFF + w] = uv[k];
          }
        }
      }
      mu_fence_smem();
      for (uint32_t g = g0 + lane; g < g0 + ng; g += MU_NUM_THREADS)
        quant_group([&](uint32_t j) __attribute__((always_inline)) {
                      const uint32_t w = 16 * (g - g0) + j, xv = stage[w];
                      if (!SILU) return F2{lo(xv), hi(xv)};
                      const uint32_t uv = stage[UOFF + w];
                      return F2{silu_mul(lo(xv), lo(uv)), silu_mul(hi(xv), hi(uv))};
                    },
                    lane, cb + 8 * (g - g0), a_scale(ASC, TM, TK, KS, t, 32 * g));
      cb_flush(cb, lane, ng, g0, A, TM, TK, KS, t);     // its trailing fence also orders the stage reuse
    }
  }
}

// A fused worker waits until tile ti (relative to the cluster's first) has landed.  Lane 0 polls SMEM.
static void wait_tile(uint32_t ti, uint32_t lane) {
  if (lane == 0) {
    while (true) {
      if (mx::ld32(FUSE_DONE_SM)) break;
      const uint32_t n = mx::ld32(FUSE_TAB_SM + 4 * ti);
      if (n && (int32_t)(mx::ld32(FUSE_RET_SM) - n) >= 0) break;
      for (uint32_t i = 0; i < 128; i++) asm volatile("nop");   // back off without CSR reads (they stall the core)
    }
  } else asm volatile("nop");
  mu_fence_smem();
}
static inline void fused_range(const volatile uint32_t *d, uint32_t tb, uint32_t &t0, uint32_t &t1) {
  const uint32_t total = d[14] * d[15] * d[16];
  t0 = 2 * (total / 2 * tb / NC);
  t1 = 2 * (total / 2 * (tb + 1) / NC);
}

// Fused SILU worker (gateup).  gu holds [G_j | U_j] per 64 columns (the generator interleaves Wgu's columns), so chunk
// j (64 outputs) needs C tiles 2j, 2j+1.  Unit = (chunk, 8 rows): G and U staged in SMEM, lane per group (row
// lane / 2, group lane % 2), codes out through the code buffer.  d[21..26]: H codes, scales, F, M, TM, TK.
static void silu_fused(const volatile uint32_t *d, uint32_t tb, uint32_t warp, uint32_t me, uint32_t nwk, uint32_t lane) {
  const uint32_t c = d[10], c_rs = d[12];
  const uint32_t A = d[21], ASC = d[22], F = d[23], M = d[24], TM = d[25], TK = d[26], KS = F / TK, RB = M / 8;
  uint32_t t0, t1;
  fused_range(d, tb, t0, t1);
  const uint32_t base = FUSE_STAGE_SM + (warp - 1) * FUSE_STAGE_BYTES;
  volatile __shared uint32_t *gs = sm32(base), *us = sm32(base + 1024), *cb = sm32(base + 2048);
  for (uint32_t u = t0 / 2 * RB + me; u < t1 / 2 * RB; u += nwk) {
    const uint32_t j = u / RB, r0 = (u % RB) * 8;
    wait_tile(2 * j + 1 - t0, lane);                     // U_j's tile (G_j's was stored before it)
    uint32_t gv[16], uv[16];
#pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
      const uint32_t w = lane + 16 * i, a = c + (r0 + w / 32) * c_rs + j * 256 + (w % 32) * 4;
      gv[i] = *(const uint32_t *)a;
      uv[i] = *(const uint32_t *)(a + 128);
    }
#pragma unroll
    for (uint32_t i = 0; i < 16; i++) { gs[lane + 16 * i] = gv[i]; us[lane + 16 * i] = uv[i]; }
    mu_fence_smem();
    const uint32_t r = lane / 2, k = lane % 2;
    quant_group([&](uint32_t jj) __attribute__((always_inline)) {
                  const uint32_t x = gs[32 * r + 16 * k + jj], y = us[32 * r + 16 * k + jj];
                  return F2{silu_mul(lo(x), lo(y)), silu_mul(hi(x), hi(y))};
                },
                lane, cb + 8 * lane, a_scale(ASC, TM, TK, KS, r0 + r, 64 * j + 32 * k));
    mu_fence_smem();
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) *(uint32_t *)(a_code(A, TM, TK, KS, r0 + i, 64 * j) + 4 * lane) = cb[16 * i + lane];
    mu_fence_smem();
  }
}

// Fused PRENORM accumulation (oproj -> norm2, down -> the next norm1 / final), one cluster only: h = bf16(x + C) is
// written to hout as C's tiles land, and each row's sum of squares is accumulated by the worker that owns the row
// (4-row blocks, block rb owned by worker rb % nwk; per-lane partials in the worker's SMEM), then reduced into
// FUSE_SS_SM.  The PRENORM phase that follows only quantizes (prenorm_rows, d[13] = 1).  d[21] x, d[22] hout.
static void norm_fused(const volatile uint32_t *d, uint32_t warp, uint32_t me, uint32_t nwk, uint32_t lane) {
  const uint32_t c = d[10], c_rs = d[12], x = d[21], hout = d[22], M = d[18], NT = d[16];
  const uint32_t RB = M / 4;
  volatile __shared float *ssp = (volatile __shared float *)(FUSE_STAGE_SM + (warp - 1) * FUSE_STAGE_BYTES);
  for (uint32_t rb = me, k = 0; rb < RB; rb += nwk, k++)
    for (uint32_t rr = 0; rr < 4; rr++) ssp[(4 * k + rr) * 16 + lane] = 0.f;
  for (uint32_t t = 0; t < NT; t++) {
    wait_tile(t, lane);
    for (uint32_t rb = me, k = 0; rb < RB; rb += nwk, k++) {
      uint32_t xv[8], av[8];
#pragma unroll
      for (uint32_t i = 0; i < 8; i++) {
        const uint32_t r = 4 * rb + i / 2, w = 32 * t + lane + 16 * (i % 2);
        xv[i] = g32(x + r * c_rs)[w];
        av[i] = g32(c + r * c_rs)[w];
      }
#pragma unroll
      for (uint32_t rr = 0; rr < 4; rr++) {
        float ss = 0.f;
#pragma unroll
        for (uint32_t h2 = 0; h2 < 2; h2++) {
          const uint32_t i = 2 * rr + h2, r = 4 * rb + rr, w = 32 * t + lane + 16 * h2;
          const uint32_t hv = add_bf2(xv[i], av[i]);
          o32(hout + r * c_rs)[w] = hv;
          ss += lo(hv) * lo(hv) + hi(hv) * hi(hv);
        }
        ssp[(4 * k + rr) * 16 + lane] = ssp[(4 * k + rr) * 16 + lane] + ss;
      }
    }
  }
  mu_fence_smem();
  for (uint32_t rb = me, k = 0; rb < RB; rb += nwk, k++) {
    if (lane < 4) {
      float ss = 0.f;
      for (uint32_t l = 0; l < 16; l++) ss += ssp[(4 * k + lane) * 16 + l];
      ((volatile __shared float *)FUSE_SS_SM)[4 * rb + lane] = ss;
    } else asm volatile("nop");
  }
}

// ---- ROPE: warp per head row (Q, K), warp per (KV head, 32 keys) for V ----------------------------
// Lane l holds pairs j = l + 16 i (i < NJ = hd / 32): element j (x1 c - x2 s) and j + hd/2 (x2 c + x1 s).
// Group gi (32 elements) of the lower half holds pairs 2 gi, 2 gi + 1 at positions lane, lane + 16.
template <uint32_t NJ>
__attribute__((always_inline)) static inline void rope_row(const uint16_t *src, const float *c, const float *sn,
                                                           float *lo_v, float *hi_v, uint32_t lane) {
  constexpr uint32_t half = NJ * 16;
  uint32_t x1[NJ], x2[NJ];
  float cv[NJ], sv[NJ];
#pragma unroll
  for (uint32_t i = 0; i < NJ; i++) {
    x1[i] = src[lane + 16 * i]; x2[i] = src[lane + 16 * i + half];
    cv[i] = c[lane + 16 * i]; sv[i] = sn[lane + 16 * i];
  }
#pragma unroll
  for (uint32_t i = 0; i < NJ; i++) {
    lo_v[i] = bfv(x1[i]) * cv[i] - bfv(x2[i]) * sv[i];
    hi_v[i] = bfv(x2[i]) * cv[i] + bfv(x1[i]) * sv[i];
  }
}
// Q unit = one head (its A rows and their scale lines); K unit = (KV head, 64-key block of the K^T cache: a K^T line
// holds 64 keys of one d).  The owner's warps take the unit's tokens.
template <uint32_t NJ>
static void rope_qk(const volatile uint32_t *d, uint32_t own, uint32_t nown, uint32_t wi, uint32_t nwo, uint32_t lane,
                    volatile __shared float *red, volatile __shared uint32_t *cw) {
  constexpr uint32_t hd = NJ * 32, half = NJ * 16;
  const uint32_t qkv = d[2], NQKV = d[3], Hq = d[4], Hkv = d[5], cosa = d[7], sina = d[8];
  const uint32_t qa = d[9], qasc = d[10], qa_bs = d[11], qasc_bs = d[12];
  const uint32_t kb = d[13], ksc = d[14], kb_bs = d[15], ksc_bs = d[16];
  const uint32_t P0 = d[21], M = d[22], SK = d[23], TM = d[24], fal = d[25];
  const uint32_t Gq = Hq / Hkv, Dq = Hq * hd;
  for (uint32_t h = own; h < Hq; h += nown) {
    const uint32_t row0 = (h % Gq) * M, aq = qa + (h / Gq) * qa_bs, asq = qasc + (h / Gq) * qasc_bs;
    for (uint32_t t = wi; t < M; t += nwo) {
      float lo_v[NJ], hi_v[NJ];
      rope_row<NJ>((const uint16_t *)(qkv + (t * NQKV + h * hd) * 2), (const float *)cosa + t * half,
                   (const float *)sina + t * half, lo_v, hi_v, lane);
      const uint32_t row = row0 + t;
#pragma unroll
      for (uint32_t gi = 0; gi < NJ / 2; gi++) {
        mx_group_split(red, cw, lane, lo_v[2 * gi], lo_v[2 * gi + 1], a_code(aq, TM, hd, 1, row, gi * 32), 1,
                       a_scale(asq, TM, hd, 1, row, gi * 32));
        mx_group_split(red, cw, lane, hi_v[2 * gi], hi_v[2 * gi + 1], a_code(aq, TM, hd, 1, row, half + gi * 32), 1,
                       a_scale(asq, TM, hd, 1, row, half + gi * 32));
      }
    }
  }
  const uint32_t nt0 = P0 / 64, nnt = (P0 + M - 1) / 64 - nt0 + 1;
  for (uint32_t u = own; u < Hkv * nnt; u += nown) {
    const uint32_t h = u / nnt, nt = nt0 + u % nnt;
    const uint32_t k0 = nt * 64 > P0 ? nt * 64 : P0, k1 = nt * 64 + 64 < P0 + M ? nt * 64 + 64 : P0 + M;
    for (uint32_t key = k0 + wi; key < k1; key += nwo) {
      const uint32_t t = key - P0, kc = key % 64;
      float lo_v[NJ], hi_v[NJ];
      rope_row<NJ>((const uint16_t *)(qkv + (t * NQKV + Dq + h * hd) * 2), (const float *)cosa + t * half,
                   (const float *)sina + t * half, lo_v, hi_v, lane);
      if (fal) {   // ATTN layout: K^T [hd][SK], scales [SK/128][hd/32][128]
        const uint32_t base = kb + h * kb_bs + key, sbase = ksc + h * ksc_bs + (key / 128) * (hd / 32) * 128 + key % 128;
#pragma unroll
        for (uint32_t gi = 0; gi < NJ / 2; gi++) {
          mx_group_split(red, cw, lane, lo_v[2 * gi], lo_v[2 * gi + 1], base + gi * 32 * SK, SK, sbase + gi * 128);
          mx_group_split(red, cw, lane, hi_v[2 * gi], hi_v[2 * gi + 1], base + (half + gi * 32) * SK, SK,
                         sbase + (half / 32 + gi) * 128);
        }
        continue;
      }
      const uint32_t base = kb + h * kb_bs + nt * hd * 64 + kc, sbase = ksc + h * ksc_bs + nt * (hd / 32) * 64 + kc;
#pragma unroll
      for (uint32_t gi = 0; gi < NJ / 2; gi++) {
        mx_group_split(red, cw, lane, lo_v[2 * gi], lo_v[2 * gi + 1], base + gi * 32 * 64, 64, sbase + gi * 64);
        mx_group_split(red, cw, lane, hi_v[2 * gi], hi_v[2 * gi + 1], base + (half + gi * 32) * 64, 64,
                       sbase + (half / 32 + gi) * 64);
      }
    }
  }
}
// V: warp per (KV head, 32 keys); lane per column dd = 16 i + lane, amax over the 32 keys is lane-local.  The 32 x 16
// code bytes of a column chunk go through the warp's stage (one word per byte), then lanes store whole words.
static void rope_v(const volatile uint32_t *d, uint32_t gw, uint32_t ngw, uint32_t lane, volatile __shared uint32_t *stage) {
  const uint32_t qkv = d[2], NQKV = d[3], Hq = d[4], Hkv = d[5], hd = d[6];
  const uint32_t vb = d[17], vsc = d[18], vb_bs = d[19], vsc_bs = d[20];
  const uint32_t P0 = d[21], M = d[22], SK = d[23], fal = d[25];
  const uint32_t Dq = Hq * hd, Dkv = Hkv * hd, KSv = SK / 256;
  for (uint32_t it = gw; it < Hkv * (M / 32); it += ngw) {
    const uint32_t h = it / (M / 32), tg = it % (M / 32), key0 = P0 + tg * 32;
    for (uint32_t d0 = 0; d0 < hd; d0 += MU_NUM_THREADS) {
      const uint32_t dd = d0 + lane;
      const uint16_t *src = (const uint16_t *)(qkv + ((tg * 32) * NQKV + Dq + Dkv + h * hd + dd) * 2);
      uint32_t v[32];
#pragma unroll
      for (uint32_t k = 0; k < 32; k++) v[k] = src[k * NQKV];
      float amax = 0.f;
#pragma unroll
      for (uint32_t k = 0; k < 32; k++) amax = __builtin_fmaxf(amax, fabs_(bfv(v[k])));
      const int32_t e = block_exp(amax);
      const float sc = pow2_neg(e);
#pragma unroll
      for (uint32_t k = 0; k < 32; k++) stage[k * 16 + lane] = fp8(bfv(v[k]) * sc);
      stage[512 + lane] = (uint32_t)(e + 127);
      mu_fence_smem();
      // 32 keys x 16 bytes = 128 words: lane writes words lane + 16 j (key (lane + 16 j) / 4, bytes 4 ((lane + 16 j) % 4))
      // ATTN layout: V [SK][hd], scales [SK/128][128/32][hd]; else the pretiled PV B operand
      const uint32_t blk = vb + h * vb_bs + ((key0 / 256) * (hd / 64) + d0 / 64) * 256 * 64 + d0 % 64;
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) {
        const uint32_t w = lane + 16 * j, k = w / 4, b0 = (w % 4) * 4;
        const uint32_t dst = fal ? vb + h * vb_bs + (key0 + k) * hd + d0 + b0 : blk + ((key0 + k) % 256) * 64 + b0;
        *(uint32_t *)dst = stage[k * 16 + b0] | (stage[k * 16 + b0 + 1] << 8) | (stage[k * 16 + b0 + 2] << 16) |
                           (stage[k * 16 + b0 + 3] << 24);
      }
      const uint32_t sdst = fal ? vsc + h * vsc_bs + (key0 / 128) * 4 * hd + ((key0 % 128) / 32) * hd + d0
                                : vsc + h * vsc_bs + ((d0 / 64) * KSv + key0 / 256) * 8 * 64 + ((key0 % 256) / 32) * 64 + d0 % 64;
      if (lane < 4)
        *(uint32_t *)(sdst + 4 * lane) = stage[512 + 4 * lane] | (stage[512 + 4 * lane + 1] << 8) |
                                         (stage[512 + 4 * lane + 2] << 16) | (stage[512 + 4 * lane + 3] << 24);
      else asm volatile("nop");
      mu_fence_smem();                                   // reads done before the next chunk rewrites the stage
    }
  }
}

// ---- ROPE for the ATTN layout (hd = 128), lane per group -------------------------------------------------------
// Q unit = (token, HC heads): the heads' bf16 inputs and the token's cos / sin staged in SMEM, a lane quantizes whole
// groups, codes go out through the code buffer.  K unit = (KV head, 64-key block): 4 keys per step (16 groups), codes
// into an SMEM row tile [64 keys][128 B] (pitch 33 words), then transposed into whole K^T lines.  V: rope_v.
constexpr uint32_t ROPE_BYTES = 12288;                   // per-warp SMEM region (words below)
static_assert(WARPS * ROPE_BYTES <= RED_SM, "SMEM map");
constexpr uint32_t RQ_XS = 0, RQ_CS = 1024, RQ_SN = 1088, RQ_CB = 1152;           // Q: up to 16 heads
constexpr uint32_t RK_XS = 0, RK_CS = 256, RK_SN = 512, RK_R = 768, RK_P = 33;    // K: 4 keys per step
static_assert((RK_R + 64 * RK_P) * 4 <= ROPE_BYTES && (RQ_CB + 512) * 4 <= ROPE_BYTES, "rope SMEM");

// group k (0..3) of a 128-element head row from its staged bf16 words x (64) and the token's cos / sin (64 floats):
// k < 2: x1 c - x2 s at positions 32 k + e; k >= 2: x2 c + x1 s at 32 (k - 2) + e, written x2 c - (-x1) s (exact)
__attribute__((always_inline)) static inline F2 rope_pair(volatile __shared uint32_t *x, volatile __shared uint32_t *cs,
                                                          volatile __shared uint32_t *sn, uint32_t k, uint32_t jj) {
  const uint32_t lo_half = bit(k < 2), o = 16 * (k & 1u) + jj, p = 32 * (k & 1u) + 2 * jj;
  const uint32_t xa = x[o], xb = x[32 + o];
  const uint32_t A = sel(lo_half, xa, xb), B = sel(lo_half, xb, xa ^ 0x80008000u);
  const float c0 = f_bits(cs[p]), c1 = f_bits(cs[p + 1]), s0 = f_bits(sn[p]), s1 = f_bits(sn[p + 1]);
  return F2{lo(A) * c0 - lo(B) * s0, hi(A) * c1 - hi(B) * s1};
}

// part (d[26]): 0 = Q and K, 1 = Q only, 2 = K only (the generator splits ROPE into timed Q / K / V phases)
static void rope_fa(const volatile uint32_t *d, uint32_t gw, uint32_t ngw, uint32_t lane, uint32_t warp, uint32_t part) {
  constexpr uint32_t hd = 128, half = 64;
  const uint32_t qkv = d[2], NQKV = d[3], Hq = d[4], Hkv = d[5], cosa = d[7], sina = d[8];
  const uint32_t qa = d[9], qasc = d[10], qa_bs = d[11], qasc_bs = d[12];
  const uint32_t kb = d[13], ksc = d[14], kb_bs = d[15], ksc_bs = d[16];
  const uint32_t P0 = d[21], M = d[22], SK = d[23], TMq = d[24];
  const uint32_t Gq = Hq / Hkv, Dq = Hq * hd, HC = Hq < 16 ? Hq : 16, NQU = M * (Hq / HC);
  const uint32_t base = warp * ROPE_BYTES;
  // ---- Q ----
  if (part != 2) {
    volatile __shared uint32_t *xs = sm32(base + 4 * RQ_XS), *cs = sm32(base + 4 * RQ_CS), *sn = sm32(base + 4 * RQ_SN);
    volatile __shared uint32_t *cb = sm32(base + 4 * RQ_CB);
    for (uint32_t u = gw; u < NQU; u += ngw) {
      const uint32_t t = u / (Hq / HC), h0 = (u % (Hq / HC)) * HC, nw = HC * 64;
      const uint32_t *X = g32(qkv + (t * NQKV + h0 * hd) * 2), *C = g32(cosa + t * half * 4), *S = g32(sina + t * half * 4);
      for (uint32_t w0 = lane; w0 < nw; w0 += 8 * MU_NUM_THREADS) {
        uint32_t xv[8];
#pragma unroll
        for (uint32_t i = 0; i < 8; i++) xv[i] = w0 + 16 * i < nw ? X[w0 + 16 * i] : 0u;
#pragma unroll
        for (uint32_t i = 0; i < 8; i++) if (w0 + 16 * i < nw) xs[w0 + 16 * i] = xv[i];
      }
      uint32_t cv[4], sv[4];
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) { cv[i] = C[lane + 16 * i]; sv[i] = S[lane + 16 * i]; }
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) { cs[lane + 16 * i] = cv[i]; sn[lane + 16 * i] = sv[i]; }
      mu_fence_smem();
      for (uint32_t gl = lane; gl < 4 * HC; gl += MU_NUM_THREADS) {
        const uint32_t hh = gl / 4, k = gl % 4, h = h0 + hh, row = (h % Gq) * M + t;
        quant_group([&](uint32_t jj) __attribute__((always_inline)) { return rope_pair(xs + 64 * hh, cs, sn, k, jj); },
                    lane, cb + 8 * gl, a_scale(qasc + (h / Gq) * qasc_bs, TMq, hd, 1, row, 32 * k));
      }
      mu_fence_smem();
      for (uint32_t w = lane; w < HC * 32; w += MU_NUM_THREADS) {
        const uint32_t h = h0 + w / 32, row = (h % Gq) * M + t;
        *(uint32_t *)(a_code(qa + (h / Gq) * qa_bs, TMq, hd, 1, row, 0) + 4 * (w % 32)) = cb[w];
      }
      mu_fence_smem();
    }
  }
  // ---- K ----
  if (part != 1) {
    volatile __shared uint32_t *xs = sm32(base + 4 * RK_XS), *cs = sm32(base + 4 * RK_CS), *sn = sm32(base + 4 * RK_SN);
    volatile __shared uint32_t *R = sm32(base + 4 * RK_R);
    const uint32_t NKB = M / 64;
    for (uint32_t u = (gw + NQU) % ngw; u < Hkv * NKB; u += ngw) {   // continue the Q units' warp rotation
      const uint32_t h = u / NKB, key0 = P0 + 64 * (u % NKB);
      for (uint32_t s = 0; s < 16; s++) {
        const uint32_t t0 = key0 - P0 + 4 * s;
        uint32_t xv[16], cv[16];
#pragma unroll
        for (uint32_t i = 0; i < 16; i++) {
          const uint32_t w = lane + 16 * i, kk = w / 64;
          xv[i] = g32(qkv + ((t0 + kk) * NQKV + Dq + h * hd) * 2)[w % 64];
        }
#pragma unroll
        for (uint32_t i = 0; i < 16; i++) xs[lane + 16 * i] = xv[i];
#pragma unroll
        for (uint32_t i = 0; i < 16; i++) { const uint32_t w = lane + 16 * i; cv[i] = g32(cosa + (t0 + w / 64) * half * 4)[w % 64]; }
#pragma unroll
        for (uint32_t i = 0; i < 16; i++) cs[lane + 16 * i] = cv[i];
#pragma unroll
        for (uint32_t i = 0; i < 16; i++) { const uint32_t w = lane + 16 * i; cv[i] = g32(sina + (t0 + w / 64) * half * 4)[w % 64]; }
#pragma unroll
        for (uint32_t i = 0; i < 16; i++) sn[lane + 16 * i] = cv[i];
        mu_fence_smem();
        const uint32_t kk = lane / 4, k = lane % 4, key = key0 + 4 * s + kk;
        quant_group([&](uint32_t jj) __attribute__((always_inline)) {
                      return rope_pair(xs + 64 * kk, cs + 64 * kk, sn + 64 * kk, k, jj);
                    },
                    lane, R + (4 * s + kk) * RK_P + 8 * k,
                    ksc + h * ksc_bs + (key / 128) * (hd / 32) * 128 + k * 128 + key % 128);
        mu_fence_smem();                                 // stage read before the next step rewrites it
      }
      // transpose: lane q builds the words of keys 4q .. 4q+3 for d = 4m .. 4m+3 from 4 row-tile words
      for (uint32_t m = 0; m < hd / 4; m++) {
        uint32_t r0 = R[(4 * lane) * RK_P + m], r1 = R[(4 * lane + 1) * RK_P + m];
        uint32_t r2 = R[(4 * lane + 2) * RK_P + m], r3 = R[(4 * lane + 3) * RK_P + m];
        const uint32_t dst = kb + h * kb_bs + 4 * m * SK + key0 + 4 * lane;
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          const uint32_t sh = 8 * i;
          *(uint32_t *)(dst + i * SK) = ((r0 >> sh) & 0xFFu) | (((r1 >> sh) & 0xFFu) << 8) | (((r2 >> sh) & 0xFFu) << 16) |
                                        (((r3 >> sh) & 0xFFu) << 24);
        }
      }
      mu_fence_smem();                                   // tile read before the next unit rewrites it
    }
  }
}

// V for the ATTN layout: unit = (KV head, 32 keys).  The 32 V rows are staged with coalesced loads ([32][64 words]),
// a lane takes 4 columns (amax over the 32 keys is lane-local), assembles each key's 4 codes into one word of an SMEM
// tile [32 keys][32 words], and the tile goes out as whole V rows.
static void rope_v_fa(const volatile uint32_t *d, uint32_t gw, uint32_t ngw, uint32_t lane, uint32_t warp) {
  constexpr uint32_t hd = 128;
  const uint32_t qkv = d[2], NQKV = d[3], Hq = d[4], Hkv = d[5];
  const uint32_t vb = d[17], vsc = d[18], vb_bs = d[19], vsc_bs = d[20];
  const uint32_t P0 = d[21], M = d[22];
  const uint32_t Dq = Hq * hd, Dkv = Hkv * hd;
  const uint32_t base = warp * ROPE_BYTES;
  volatile __shared uint32_t *vs = sm32(base), *T = sm32(base + 8192);
  static_assert(8192 + 4096 <= ROPE_BYTES, "rope V SMEM");
  for (uint32_t it = gw; it < Hkv * (M / 32); it += ngw) {
    const uint32_t h = it / (M / 32), tg = it % (M / 32), key0 = P0 + tg * 32;
    for (uint32_t w0 = lane; w0 < 32 * 64; w0 += 8 * MU_NUM_THREADS) {
      uint32_t v[8];
#pragma unroll
      for (uint32_t i = 0; i < 8; i++) {
        const uint32_t w = w0 + 16 * i;
        v[i] = g32(qkv + ((tg * 32 + w / 64) * NQKV + Dq + Dkv + h * hd) * 2)[w % 64];
      }
#pragma unroll
      for (uint32_t i = 0; i < 8; i++) vs[w0 + 16 * i] = v[i];
    }
    mu_fence_smem();
    for (uint32_t c4 = lane; c4 < hd / 4; c4 += MU_NUM_THREADS) {
      float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;
      for (uint32_t k = 0; k < 32; k++) {
        const uint32_t a = vs[k * 64 + 2 * c4], b = vs[k * 64 + 2 * c4 + 1];
        m0 = __builtin_fmaxf(m0, fabs_(lo(a))); m1 = __builtin_fmaxf(m1, fabs_(hi(a)));
        m2 = __builtin_fmaxf(m2, fabs_(lo(b))); m3 = __builtin_fmaxf(m3, fabs_(hi(b)));
      }
      const int32_t e0 = block_exp(m0), e1 = block_exp(m1), e2 = block_exp(m2), e3 = block_exp(m3);
      const float s0 = pow2_neg(e0), s1 = pow2_neg(e1), s2 = pow2_neg(e2), s3 = pow2_neg(e3);
      for (uint32_t k = 0; k < 32; k++) {
        const uint32_t a = vs[k * 64 + 2 * c4], b = vs[k * 64 + 2 * c4 + 1];
        T[k * 32 + c4] = fp8(lo(a) * s0) | (fp8(hi(a) * s1) << 8) | (fp8(lo(b) * s2) << 16) | (fp8(hi(b) * s3) << 24);
      }
      *(uint32_t *)(vsc + h * vsc_bs + (key0 / 128) * 4 * hd + ((key0 % 128) / 32) * hd + 4 * c4) =
          (uint32_t)(e0 + 127) | ((uint32_t)(e1 + 127) << 8) | ((uint32_t)(e2 + 127) << 16) | ((uint32_t)(e3 + 127) << 24);
    }
    mu_fence_smem();
    for (uint32_t w = lane; w < 32 * 32; w += MU_NUM_THREADS)
      *(uint32_t *)(vb + h * vb_bs + (key0 + w / 32) * hd + 4 * (w % 32)) = T[w];
    mu_fence_smem();                                     // tile read before the next unit rewrites the stage
  }
}

// ---- ROPE fused into the qkv GEMM (ATTN layout) ----------------------------------------------------------------
// A head's Q, K or V rows are qkv's C tiles 2h, 2h+1 of that section.  Units, in the order their tiles land: Q (head,
// 4 tokens), K (head, 16 keys), V (head, 32 keys, 32 columns); each fits the worker's 4 KB of SMEM.
// the ROPE descriptor's fields (read from DRAM in each unit: no pointer to a stack object on Muon)
#define ROPE_FIELDS(rd)                                                                                           \
  const uint32_t qkv = rd[2], NQKV = rd[3], Hq = rd[4], Hkv = rd[5], cosa = rd[7], sina = rd[8], qa = rd[9];       \
  const uint32_t qasc = rd[10], qa_bs = rd[11], qasc_bs = rd[12], kb = rd[13], ksc = rd[14], kb_bs = rd[15];       \
  const uint32_t ksc_bs = rd[16], vb = rd[17], vsc = rd[18], vb_bs = rd[19], vsc_bs = rd[20], P0 = rd[21];         \
  const uint32_t M = rd[22], SK = rd[23], TMq = rd[24];                                                           \
  (void)qkv; (void)NQKV; (void)Hq; (void)Hkv; (void)cosa; (void)sina; (void)qa; (void)qasc; (void)qa_bs;           \
  (void)qasc_bs; (void)kb; (void)ksc; (void)kb_bs; (void)ksc_bs; (void)vb; (void)vsc; (void)vb_bs; (void)vsc_bs;   \
  (void)P0; (void)M; (void)SK; (void)TMq
static __attribute__((noinline)) void rope_q_unit(const volatile uint32_t *rd, uint32_t h, uint32_t tq, uint32_t lane, uint32_t base) {
  constexpr uint32_t hd = 128;
  ROPE_FIELDS(rd);
  volatile __shared uint32_t *xs = sm32(base), *cs = sm32(base + 1024), *sn = sm32(base + 2048), *cb = sm32(base + 3072);
  const uint32_t Gq = Hq / Hkv;
  uint32_t xv[16], cv[16], sv[16];
#pragma unroll
  for (uint32_t i = 0; i < 16; i++) {
    const uint32_t w = lane + 16 * i, t = tq + w / 64;
    xv[i] = g32(qkv + (t * NQKV + h * hd) * 2)[w % 64];
    cv[i] = g32(cosa + t * 256)[w % 64];
    sv[i] = g32(sina + t * 256)[w % 64];
  }
#pragma unroll
  for (uint32_t i = 0; i < 16; i++) { xs[lane + 16 * i] = xv[i]; cs[lane + 16 * i] = cv[i]; sn[lane + 16 * i] = sv[i]; }
  mu_fence_smem();
  const uint32_t kk = lane / 4, k = lane % 4, row = (h % Gq) * M + tq + kk;
  quant_group([&](uint32_t jj) __attribute__((always_inline)) { return rope_pair(xs + 64 * kk, cs + 64 * kk, sn + 64 * kk, k, jj); },
              lane, cb + 8 * lane, a_scale(qasc + (h / Gq) * qasc_bs, TMq, hd, 1, row, 32 * k));
  mu_fence_smem();
#pragma unroll
  for (uint32_t i = 0; i < 8; i++) {
    const uint32_t w = lane + 16 * i, rw = (h % Gq) * M + tq + w / 32;
    *(uint32_t *)(a_code(qa + (h / Gq) * qa_bs, TMq, hd, 1, rw, 0) + 4 * (w % 32)) = cb[w];
  }
  mu_fence_smem();
}
static __attribute__((noinline)) void rope_k_unit(const volatile uint32_t *rd, uint32_t h, uint32_t k0, uint32_t lane, uint32_t base) {
  constexpr uint32_t hd = 128;
  ROPE_FIELDS(rd);
  volatile __shared uint32_t *xs = sm32(base), *cs = sm32(base + 512), *sn = sm32(base + 1024), *R = sm32(base + 1536);
  const uint32_t Dq = Hq * hd;
  for (uint32_t s = 0; s < 8; s++) {                     // 2 keys per step
    const uint32_t tk = k0 + 2 * s;
    uint32_t xv[8], cv[8], sv[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
      const uint32_t w = lane + 16 * i, t = tk + w / 64;
      xv[i] = g32(qkv + (t * NQKV + Dq + h * hd) * 2)[w % 64];
      cv[i] = g32(cosa + t * 256)[w % 64];
      sv[i] = g32(sina + t * 256)[w % 64];
    }
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) { xs[lane + 16 * i] = xv[i]; cs[lane + 16 * i] = cv[i]; sn[lane + 16 * i] = sv[i]; }
    mu_fence_smem();
    if (lane < 8) {
      const uint32_t kk = lane / 4, k = lane % 4, key = P0 + tk + kk;
      quant_group([&](uint32_t jj) __attribute__((always_inline)) { return rope_pair(xs + 64 * kk, cs + 64 * kk, sn + 64 * kk, k, jj); },
                  lane, R + (2 * s + kk) * 32 + 8 * k, ksc + h * ksc_bs + (key / 128) * (hd / 32) * 128 + k * 128 + key % 128);
    } else asm volatile("nop");
    mu_fence_smem();
  }
  // transpose: (m, q) -> the words of keys 4q .. 4q+3 for d = 4m .. 4m+3
#pragma unroll 2
  for (uint32_t i = 0; i < 8; i++) {
    const uint32_t idx = lane + 16 * i, m = idx / 4, q = idx % 4;
    const uint32_t r0 = R[(4 * q) * 32 + m], r1 = R[(4 * q + 1) * 32 + m], r2 = R[(4 * q + 2) * 32 + m], r3 = R[(4 * q + 3) * 32 + m];
    const uint32_t dst = kb + h * kb_bs + 4 * m * SK + P0 + k0 + 4 * q;
#pragma unroll
    for (uint32_t b = 0; b < 4; b++) {
      const uint32_t sh = 8 * b;
      *(uint32_t *)(dst + b * SK) = ((r0 >> sh) & 0xFFu) | (((r1 >> sh) & 0xFFu) << 8) | (((r2 >> sh) & 0xFFu) << 16) |
                                      (((r3 >> sh) & 0xFFu) << 24);
    }
  }
  mu_fence_smem();
}
static __attribute__((noinline)) void rope_v_unit(const volatile uint32_t *rd, uint32_t h, uint32_t g, uint32_t cq, uint32_t lane,
                                                  uint32_t base) {
  constexpr uint32_t hd = 128;
  ROPE_FIELDS(rd);
  volatile __shared uint32_t *vs = sm32(base), *T = sm32(base + 2048);
  const uint32_t Dq = Hq * hd, Dkv = Hkv * hd, key0 = P0 + 32 * g;
  for (uint32_t w0 = lane; w0 < 512; w0 += 8 * MU_NUM_THREADS) {
    uint32_t v[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
      const uint32_t w = w0 + 16 * i;
      v[i] = g32(qkv + ((32 * g + w / 16) * NQKV + Dq + Dkv + h * hd) * 2)[16 * cq + w % 16];
    }
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) vs[w0 + 16 * i] = v[i];
  }
  mu_fence_smem();
  if (lane < 8) {
    const uint32_t c4 = lane;
    float m0 = 0.f, m1 = 0.f, m2 = 0.f, m3 = 0.f;
    for (uint32_t k = 0; k < 32; k++) {
      const uint32_t a = vs[k * 16 + 2 * c4], b = vs[k * 16 + 2 * c4 + 1];
      m0 = __builtin_fmaxf(m0, fabs_(lo(a))); m1 = __builtin_fmaxf(m1, fabs_(hi(a)));
      m2 = __builtin_fmaxf(m2, fabs_(lo(b))); m3 = __builtin_fmaxf(m3, fabs_(hi(b)));
    }
    const int32_t e0 = block_exp(m0), e1 = block_exp(m1), e2 = block_exp(m2), e3 = block_exp(m3);
    const float s0 = pow2_neg(e0), s1 = pow2_neg(e1), s2 = pow2_neg(e2), s3 = pow2_neg(e3);
    for (uint32_t k = 0; k < 32; k++) {
      const uint32_t a = vs[k * 16 + 2 * c4], b = vs[k * 16 + 2 * c4 + 1];
      T[k * 8 + c4] = fp8(lo(a) * s0) | (fp8(hi(a) * s1) << 8) | (fp8(lo(b) * s2) << 16) | (fp8(hi(b) * s3) << 24);
    }
    *(uint32_t *)(vsc + h * vsc_bs + (key0 / 128) * 4 * hd + ((key0 % 128) / 32) * hd + 32 * cq + 4 * c4) =
        (uint32_t)(e0 + 127) | ((uint32_t)(e1 + 127) << 8) | ((uint32_t)(e2 + 127) << 16) | ((uint32_t)(e3 + 127) << 24);
  } else asm volatile("nop");
  mu_fence_smem();
#pragma unroll 4
  for (uint32_t i = 0; i < 16; i++) {
    const uint32_t w = lane + 16 * i;
    *(uint32_t *)(vb + h * vb_bs + (key0 + w / 8) * hd + 32 * cq + 4 * (w % 8)) = T[w];
  }
  mu_fence_smem();
}
// d[21]: the ROPE descriptor (a no-op phase after qkv)
static void rope_fused(const volatile uint32_t *d, uint32_t tb, uint32_t warp, uint32_t me, uint32_t nwk, uint32_t lane) {
  const volatile uint32_t *rd = gw32(d[21]);
  const uint32_t Hq = rd[4], Hkv = rd[5], M = rd[22];
  uint32_t t0, t1;
  fused_range(d, tb, t0, t1);
  const uint32_t base = FUSE_STAGE_SM + (warp - 1) * FUSE_STAGE_BYTES;
  const uint32_t NQ = Hq * (M / 4), NK = Hkv * (M / 16), NV = Hkv * (M / 32) * 4;
  for (uint32_t u = me; u < NQ + NK + NV; u += nwk) {
    // flat tests through opaque(): an else-if chain on one value becomes a switch, which UnifyLoopExits rejects
    const uint32_t isq = opaque(u < NQ), isk = opaque(u >= NQ) & opaque(u < NQ + NK), isv = opaque(u >= NQ + NK);
    const uint32_t vk = u - NQ, vv = u - NQ - NK;
    const uint32_t a0 = isq ? u / (M / 4) : isk ? vk / (M / 16) : vv / ((M / 32) * 4);
    const uint32_t a1 = isq ? (u % (M / 4)) * 4 : isk ? (vk % (M / 16)) * 16 : (vv / 4) % (M / 32);
    const uint32_t tile = (isq ? 0 : isk ? 2 * Hq : 2 * Hq + 2 * Hkv) + 2 * a0 + 1;
    if (opaque(tile >= t0) & opaque(tile < t1)) {        // else another cluster's head
      wait_tile(tile - t0, lane);
      if (isq) rope_q_unit(rd, a0, a1, lane, base);
      else asm volatile("nop");
      if (isk) rope_k_unit(rd, a0, a1, lane, base);
      else asm volatile("nop");
      if (isv) rope_v_unit(rd, a0, a1, vv % 4, lane, base);
      else asm volatile("nop");
    } else asm volatile("nop");
  }
}

// ---- SOFTMAX: warp per row, causal, normalized, MX --------------------------------------------------
static inline float masked(float v, uint32_t valid) { return f_bits(bits_f(v) & mask_if(valid)); }
// unit = (head group, m-tile, k-step): the k-step's P codes and scale lines.  Each unit computes its rows' full max
// and sum itself (KS-fold extra reads).
static void softmax_phase(const volatile uint32_t *d, uint32_t own, uint32_t nown, uint32_t wi, uint32_t nwo,
                          uint32_t lane, volatile __shared float *red, volatile __shared uint32_t *cw,
                          volatile __shared uint32_t *stage, volatile __shared uint32_t *sb) {
  const uint32_t s = d[2], s_bs = d[3], SK = d[4], R = d[5], batch = d[6];
  const uint32_t pa = d[7], pasc = d[8], pa_bs = d[9], pasc_bs = d[10];
  const float scale = f_bits(d[11]);
  const uint32_t P0 = d[12], M = d[13], TM = d[14], TK = d[15], KSp = SK / TK, W = SK / 2, GS = TK / 32;
  const uint32_t MTb = R / TM;
  const float NEG = f_bits(0xFF800000u);
  for (uint32_t u = own; u < batch * MTb * KSp; u += nown) {
    const uint32_t bt = u / (MTb * KSp), mt = (u / KSp) % MTb, ks = u % KSp;
    const uint32_t A = pa + bt * pa_bs, ASC = pasc + bt * pasc_bs;
    for (uint32_t rb = mt * TM + 4 * wi; rb < (mt + 1) * TM; rb += 4 * nwo) {
      for (uint32_t rr = 0; rr < 4; rr++) {
        const uint32_t r = rb + rr, qpos = P0 + r % M;
        const uint32_t *S = g32(s + bt * s_bs + r * SK * 2);
        float m = NEG;
        for (uint32_t w0 = lane; w0 < W; w0 += 8 * MU_NUM_THREADS) {   // W is a multiple of 128
          uint32_t sv[8];
#pragma unroll
          for (uint32_t k = 0; k < 8; k++) sv[k] = S[w0 + k * MU_NUM_THREADS];
#pragma unroll
          for (uint32_t k = 0; k < 8; k++) {
            const uint32_t w = w0 + k * MU_NUM_THREADS;
            stage[w] = sv[k];
            m = __builtin_fmaxf(m, f_bits(sel((uint32_t)(2 * w <= qpos), bits_f(lo(sv[k]) * scale), bits_f(NEG))));
            m = __builtin_fmaxf(m, f_bits(sel((uint32_t)(2 * w + 1 <= qpos), bits_f(hi(sv[k]) * scale), bits_f(NEG))));
          }
        }
        m = warp_max(red, lane, m);
        float sum = 0.f;
        for (uint32_t w = lane; w < W; w += MU_NUM_THREADS) {
          const uint32_t sv = stage[w];
          sum += masked(fexp_bf(lo(sv) * scale - m), (uint32_t)(2 * w <= qpos)) +
                 masked(fexp_bf(hi(sv) * scale - m), (uint32_t)(2 * w + 1 <= qpos));
        }
        sum = warp_sum(red, lane, sum);
        const float inv = 1.0f / sum;
        for (uint32_t gi = 0; gi < GS; gi++) {          // this k-step's groups: keys 32 g .., words 16 g + lane
          const uint32_t g = ks * GS + gi, w = g * 16 + lane, sv = stage[w];
          const float p0 = masked(fexp_bf(lo(sv) * scale - m) * inv, (uint32_t)(2 * w <= qpos));
          const float p1 = masked(fexp_bf(hi(sv) * scale - m) * inv, (uint32_t)(2 * w + 1 <= qpos));
          mx_group2(red, cw, lane, p0, p1, a_code(A, TM, TK, KSp, r, g * 32), sb + gi, rr);
        }
      }
      sb_flush(sb, lane, GS, ks * GS, ASC, TM, TK, KSp, rb);
    }
  }
}

// ---- EWQ / SILU: warp per row, a group per warp step ----------------------------------------------
// EWQ: MX of a bf16 matrix [M][K] (row stride rs elements); SILU: MX(silu(G) * U), G = columns [0, K),
// U = columns [K, 2K).  Four groups per step so 4 (8) loads per lane are in flight.
template <bool SILU>
static void ewq_phase(const volatile uint32_t *d, uint32_t own, uint32_t nown, uint32_t wi, uint32_t nwo, uint32_t lane,
                      volatile __shared float *red, volatile __shared uint32_t *cw, volatile __shared uint32_t *sb) {
  const uint32_t src = d[2], rs = d[3], K = d[5], M = d[6];
  const uint32_t A = d[7], ASC = d[8], TM = d[9], TK = d[10], KS = K / TK, MT = M / TM;   // TK = 256: 8 groups
  for (uint32_t u = own; u < MT * KS; u += nown) {
    const uint32_t mt = u / KS, s = u % KS;
    for (uint32_t rb = mt * TM + 4 * wi; rb < (mt + 1) * TM; rb += 4 * nwo) {
      for (uint32_t rr = 0; rr < 4; rr++) {
        const uint32_t t = rb + rr;
        const uint32_t *X = g32(src + (t * rs + s * 256) * 2), *U = g32(src + (t * rs + K + s * 256) * 2);
        uint32_t xv[8], uv[8];
#pragma unroll
        for (uint32_t gi = 0; gi < 8; gi++) {
          xv[gi] = X[gi * 16 + lane];
          if (SILU) uv[gi] = U[gi * 16 + lane];
        }
#pragma unroll
        for (uint32_t gi = 0; gi < 8; gi++) {
          const float a = SILU ? silu_mul(lo(xv[gi]), lo(uv[gi])) : lo(xv[gi]);
          const float b = SILU ? silu_mul(hi(xv[gi]), hi(uv[gi])) : hi(xv[gi]);
          mx_group2(red, cw, lane, a, b, a_code(A, TM, TK, KS, t, (s * 8 + gi) * 32), sb + gi, rr);
        }
      }
      sb_flush(sb, lane, 8, s * 8, ASC, TM, TK, KS, rb);
    }
  }
}

// ---- ACT: y = gelu(x), bf16 -> bf16 (the nightly gelu kernel's math) -------------------------------------
// Warp per 16-word chunk (one 64-byte line), lane per word: every line is written by one warp with whole-word
// stores (no sub-word stores, no line shared between cores).
static inline float exp2_poly(float t) {                  // 2^t, degree-5 minimax on the fraction
  t = __builtin_fminf(__builtin_fmaxf(t, -126.0f), 126.0f);
  int32_t n;
  asm("fcvt.w.s %0, %1, rdn" : "=r"(n) : "r"(t));
  const float f = t - (float)n;
  float p = 1.8775767e-3f;
  p = p * f + 8.9893397e-3f;
  p = p * f + 5.5826318e-2f;
  p = p * f + 2.4015361e-1f;
  p = p * f + 6.9315308e-1f;
  p = p * f + 9.9999994e-1f;
  return f_bits(bits_f(p) + ((uint32_t)n << 23));
}
static inline float gelu_f(float x) {
  const float z = 0.7978845608f * (x + 0.044715f * x * x * x);
  return x / (1.0f + exp2_poly(-2.0f * 1.4426950409f * z));
}
static void act_phase(const volatile uint32_t *d, uint32_t gw, uint32_t ngw, uint32_t lane) {
  const uint32_t *src = g32(d[2]);
  uint32_t *dst = o32(d[3]);
  const uint32_t chunks = d[4] / 16;                     // d[4]: words, a multiple of 16
  for (uint32_t c0 = gw; c0 < chunks; c0 += 4 * ngw) {
    uint32_t x[4];
#pragma unroll
    for (uint32_t u = 0; u < 4; u++) {
      const uint32_t c = c0 + u * ngw;
      x[u] = c < chunks ? src[c * 16 + lane] : 0u;
    }
#pragma unroll
    for (uint32_t u = 0; u < 4; u++) {
      const uint32_t c = c0 + u * ngw;
      if (c < chunks) dst[c * 16 + lane] = bf16(gelu_f(lo(x[u]))) | (bf16(gelu_f(hi(x[u]))) << 16);
      else asm volatile("nop");
    }
  }
}

// ---- FILL: synthetic weights (E4M3 codes with exponent field <= 7, |v| < 2) ---------------------------
static inline uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}
// warp per 16-word chunk (one 64-byte line), lane per word: no line written by two cores
static void fill_phase(const volatile uint32_t *d, uint32_t gw, uint32_t ngw, uint32_t lane) {
  const uint32_t base = d[2], nw = d[3] / 4, mode = d[4], val = d[5];
  uint32_t *P = o32(base);
  for (uint32_t c = gw; c * 16 < nw; c += ngw) {
    const uint32_t i = c * 16 + lane;
    if (i < nw) P[i] = mode ? val : (hash32(i ^ val) & 0xBFBFBFBFu);
    else asm volatile("nop");
  }
}

#include "fa_phase.h"

// =================================== interpreter ===================================
static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const uint32_t warps = tpb / MU_NUM_THREADS, warp = tid / MU_NUM_THREADS, lane = tid % MU_NUM_THREADS;
  const bool lead = tid == 0;
  const uint32_t gw = tb * WARPS + warp, ngw = NC * WARPS;
  const uint32_t gl = gw * MU_NUM_THREADS + lane, ngl = ngw * MU_NUM_THREADS;
  volatile __shared float *red = (volatile __shared float *)(RED_SM + warp * 64);
  volatile __shared uint32_t *stage = sm32(STAGE_SM + warp * STAGE_BYTES);
  volatile __shared uint32_t *cw = sm32(CW_SM + warp * 128);
  volatile __shared uint32_t *sb = sm32(SB_SM + warp * SB_BYTES);
  volatile uint32_t *perf = gw32(LL_PERF_ADDR);      // one 64-byte line per (phase, cluster)
  // line owner: block warp w runs on core w % cores; owner = (cluster, core), its warps are wi = warp / cores
  const uint32_t own = tb * MU_NUM_CORES + warp % MU_NUM_CORES, nown = NC * MU_NUM_CORES;
  const uint32_t wi = warp / MU_NUM_CORES, nwo = WARPS / MU_NUM_CORES;
  if (lead) {
    mx::loops_reset();                                   // before the first LOOP_WS
    mx::flush_tlb();
    mx::lut_disable();
    mx::config_ex(mx::FP8, mx::FP8, mx::BF16);
  } else asm volatile("nop");
  for (uint32_t p = 0; p < LL_NP; p++) {
    const volatile uint32_t *dsc = gw32(LL_PROG_ADDR + p * LL_PW * 4);
    if (lead) {
      while (nightly_pb_get(PB_GO) <= p) asm volatile("nop");
      perf[(p * NC + tb) * 16] = nightly_mcycle();
      if (tb == 0) nightly_heartbeat(p);
    } else asm volatile("nop");
    bar(BAR_PH, warps);
    // flat, one-level dispatch: an else-if chain nests a split per type on the IPDOM stack
    const uint32_t ty = opaque(dsc[0]);
    // fused GEMM: workers on warps 1.. (PRENORM accumulation only on one cluster)
    const uint32_t fk = opaque(ty == LL_T_GEMM) ? dsc[20] : 0u;
    const uint32_t fused = opaque(fk != 0) & opaque(fk != FUSE_NORM || NC == 1);
    if (fused) {                                         // clear the tile table before anyone polls it
      if (lead) {
        for (uint32_t i = 0; i < dsc[16]; i++) mx::st32(FUSE_TAB_SM + 4 * i, 0);
        mx::st32(FUSE_DONE_SM, 0);
        mu_fence_smem();
      } else asm volatile("nop");
      bar(4, warps);
    } else asm volatile("nop");
    if (opaque(ty == LL_T_GEMM) & (uint32_t)lead) gemm(dsc, tb);
    else asm volatile("nop");
    // workers: core 1's warps (1, 3, ..), off the GEMM thread's core (FireSim: 88.7% model util vs 85.4% with warps
    // 1..); -DLL_WORKERS_CORE1=0 uses warps 1..
#if !defined(LL_WORKERS_CORE1) || LL_WORKERS_CORE1
    const uint32_t wk = opaque(warp % MU_NUM_CORES == 1), wme = warp / MU_NUM_CORES, nwk = warps / MU_NUM_CORES;
#else
    const uint32_t wk = opaque(warp != 0), wme = warp - 1, nwk = warps - 1;
#endif
    if (fused & wk & opaque(fk == FUSE_SILU)) silu_fused(dsc, tb, warp, wme, nwk, lane);
    else asm volatile("nop");
    if (fused & wk & opaque(fk == FUSE_ROPE)) rope_fused(dsc, tb, warp, wme, nwk, lane);
    else asm volatile("nop");
    if (fused & wk & opaque(fk == FUSE_NORM)) norm_fused(dsc, warp, wme, nwk, lane);
    else asm volatile("nop");
#ifdef LL_OLD_ROWOPS
    if (opaque(ty == LL_T_PRENORM)) prenorm(dsc, tid, tpb, own, nown, wi, nwo, lane, red, cw, sb);
    else asm volatile("nop");
    if (opaque(ty == LL_T_EWQ)) ewq_phase<false>(dsc, own, nown, wi, nwo, lane, red, cw, sb);
    else asm volatile("nop");
    if (opaque(ty == LL_T_SILU)) ewq_phase<true>(dsc, own, nown, wi, nwo, lane, red, cw, sb);
    else asm volatile("nop");
#else
    if (opaque(ty == LL_T_PRENORM)) prenorm_rows(dsc, tid, tpb, gw, ngw, lane, red, stage, sb);
    else asm volatile("nop");
    if (opaque(ty == LL_T_EWQ)) ewq_rows<false>(dsc, gw, ngw, lane, stage, sb);
    else asm volatile("nop");
    if (opaque(ty == LL_T_SILU)) ewq_rows<true>(dsc, gw, ngw, lane, stage, sb);
    else asm volatile("nop");
#endif
    if (opaque(ty == LL_T_ROPE) & opaque(dsc[6] == 64)) rope_qk<2>(dsc, own, nown, wi, nwo, lane, red, cw);
    else asm volatile("nop");
    if (opaque(ty == LL_T_ROPE) & opaque(dsc[6] == 128) & opaque(dsc[25] == 0)) rope_qk<4>(dsc, own, nown, wi, nwo, lane, red, cw);
    else asm volatile("nop");
    // ATTN layout: d[26] = part (0 all, 1 Q, 2 K, 3 V)
    const uint32_t rfa = opaque(ty == LL_T_ROPE) & opaque(dsc[25] != 0), rpart = dsc[26];
    if (rfa & opaque(rpart != 3)) rope_fa(dsc, gw, ngw, lane, warp, rpart);
    else asm volatile("nop");
    if (rfa & opaque(rpart == 0 || rpart == 3)) rope_v_fa(dsc, gw, ngw, lane, warp);
    else asm volatile("nop");
    if (opaque(ty == LL_T_ROPE) & opaque(dsc[25] == 0)) rope_v(dsc, gw, ngw, lane, stage);
    else asm volatile("nop");
    if (opaque(ty == LL_T_SOFTMAX)) softmax_phase(dsc, own, nown, wi, nwo, lane, red, cw, stage, sb);
    else asm volatile("nop");
    if (opaque(ty == LL_T_FA)) fa::phase(dsc, tid, tpb, tb, acc_parity);
    else asm volatile("nop");
    if (opaque(ty == LL_T_FILL)) fill_phase(dsc, gw, ngw, lane);
    else asm volatile("nop");
    if (opaque(ty == LL_T_ACT)) act_phase(dsc, gw, ngw, lane);
    else asm volatile("nop");
    mu_fence();                                          // this thread's stores have landed
    bar(BAR_PH, warps);
    if (lead) {
      perf[(p * NC + tb) * 16 + 1] = nightly_mcycle();
      mu_fence();
      nightly_pb_put(PB_DONE, p + 1);
    } else asm volatile("nop");
  }
  nightly_kernel_end(tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
