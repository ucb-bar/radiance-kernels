// LIVE fp6 chain: a runtime SIMT fp6-e3m2-LUT ACTIVATION quantizer feeds the MX mesh.
//
//   X (bf16 activation, op1 output stand-in)
//     --[SIMT: bf16->fp6 code -> nearest-LUT-entry finder -> 4-bit index]-->
//   A_in (fp6 indices, HW layout) + A_lut (staged 16-entry fp6 LUT)
//     --[fp6 mxgemm, static fp6 weight B]--> C (bf16).
//
// This is the piece that was missing on-device: there was a runtime `quantize_fp4`
// but NO runtime fp6-LUT activation quantizer.  The two device functions below
// (dev_bf16_to_fp6_code + dev_fp6_nearest_finder) mirror the RTL BF16ScaleRoundToTiny
// (fp6 path) and FP6E3M2NearestFinder, and are validated bit-exact against the Python
// golden over all 65280 finite bf16 patterns.  Correctness: cyclotron verify_body vs the
// mx-hwlike bf16 golden that mirrors this exact quantizer -> tohost=0 (two-line pass).
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#ifndef MX_NUM_WARPS
#define MX_NUM_WARPS 2
#endif
extern "C" uint32_t __mu_num_warps = MX_NUM_WARPS;
#include "data"

// ---- runtime-produced A operand (mutable; filled by the SIMT quantizer) -------------
// A_in holds the fp6 nearest-LUT indices the finder computes at runtime.  A_lut (the fp6
// palette) is a fixed 16-entry table the mesh loads -- baked const in `data`, exactly as
// the hardware LUT SRAM is loaded once from a fixed palette (see NOTE at bottom on the
// runtime-staging variant).
// The runtime LUT staging (mission item a) is ON by default; -DBAKED_LUT uses the const
// palette instead (both pass tohost=0).
#ifndef BAKED_LUT
#define RUNTIME_LUT
#endif
static uint8_t A_in_buf[MATMUL_M / 2][MATMUL_K] __attribute__((aligned(64)));  // fp6 indices, HW row-pair layout (64B-aligned for word-stores)
static const uint8_t *A_in = &A_in_buf[0][0];       // mesh reads A operand through this
#ifdef RUNTIME_LUT
// Mission item (a): stage the per-row 16-entry fp6 LUT on-device at runtime.  8-word aligned
// + a guard word so the SIMT A_in byte-store stream cannot alias the triplet region.
static uint32_t A_lut_pad[8] __attribute__((aligned(64), used)) = {0};
static uint32_t A_lut[MATMUL_M / 2][3] __attribute__((aligned(64)));
#else
#define A_lut A_lut_baked                            // mesh loads the fixed baked palette
#endif

#include "mxgemm_lib_fused.hpp"
constexpr GemmConfig GEMM_CFG{ .TILE_M = MATMUL_M, .TILE_N = MATMUL_N, .TILE_K = 64,
                               .DATATYPE = GemmDatatype::FP6, .QUANT_OUTPUT = false };

// ===========================================================================
//  Runtime SIMT fp6-e3m2-LUT activation quantizer (the deliverable)
// ===========================================================================

// (1) bf16 bits -> fp6:e3m2 6-bit code.  Integer-only (libm-free) transliteration of
//     BF16ScaleRoundToTiny's fp6 path: RNE bf16->E4M2, then deterministic E4M2->fp6.
// BRANCHLESS single-return: the original's 7 nested early-
// `return`s overflow the Muon IPDOM stack on RTL (WarpScheduler.sv:793) before the finder
// runs.  dev_sel = mask-select (RISC-V has no cmov).  Bit-identical over all 65536 bf16.
// The C masked form `(a&m)|(b&~m)` is pattern-matched by the Muon LLVM backend back into a
// `select` and lowered to a DIVERGENT vx_split_n/beqz/vx_join (verified in the ELF), which
// deadlocks warp reconvergence on RTL at the quantizer's global loads.  Emitting the masked
// select as ONE opaque inline-asm block leaves no `select` IR to divergence-lower, so codegen
// is straight-line (zero vx_split_n).  cond is 0/1; returns a if cond else b.
static inline int dev_sel(int cond, int a, int b) {
  int m, r;
  asm("sub %[m], x0, %[c]\n\t"       // m = -(cond) -> 0 or 0xFFFFFFFF
      "xor %[r], %[b], %[a]\n\t"      // r = a^b
      "and %[r], %[r], %[m]\n\t"      // r = (a^b)&m
      "xor %[r], %[r], %[b]\n\t"      // r = b ^ ((a^b)&m) = m ? a : b
      : [m] "=&r"(m), [r] "=&r"(r)
      : [c] "r"(cond & 1), [a] "r"(a), [b] "r"(b));
  return r;
}
static inline uint8_t dev_bf16_to_fp6_code(uint16_t bits) {
  int sign = (bits >> 15) & 1;
  int E  = (bits >> 7) & 0xFF;
  int Mm =  bits       & 0x7F;
  int e = E - 127;
  int q = (Mm >> 5) & 3, r = (Mm >> 4) & 1;
  int sticky = (Mm & 0xF) ? 1 : 0, lsb = (Mm >> 5) & 1;
  int sig = q + (r & (sticky | lsb));
  int carry = (sig >= 4);
  int mant_e = dev_sel(carry, 0, sig);
  int exp_e  = dev_sel(carry, e + 1, e);
  int m_le1  = (mant_e <= 1);
  int iszero = (E == 0) | (e <= -7) | (exp_e <= -5);
  int ismax  = (E == 255) | (exp_e >= 5);
  int mag = (((exp_e + 3) & 7) << 2) | mant_e;
  mag = dev_sel(exp_e == -3, dev_sel(m_le1, 2, 3), mag);
  mag = dev_sel(exp_e == -4, dev_sel(m_le1, 1, 2), mag);
  mag = dev_sel(ismax, 0x1F, mag);
  int code = dev_sel(iszero, 0, (sign << 5) | mag);
  return (uint8_t)code;
}

// (2) fp6 code -> signed fixed-point (unit 2^-4).  Mirrors fp6ToFixedPoint in
//     FP6E3M2NearestFinder.scala.
static inline int dev_fp6_to_fixed(uint8_t code) {
  code &= 0x3F;
  int sign = (code >> 5) & 1, exp = (code >> 2) & 7, mant = code & 3;
  int is_zero = (exp == 0) & (mant == 0);
  int s_exp = dev_sel(exp == 0, -2, exp - 3);        // was `?:` -> divergent branch; now opaque
  int imp   = dev_sel(exp == 0,  0, 1);
  int sig   = (imp << 2) | mant;
  int shift = (s_exp + 2) & 7;
  int mag   = ((sig << shift) & 0x1FF) & 0xFF;
  return dev_sel(is_zero, 0, dev_sel(sign, -mag, mag));
}

// (3) nearest-LUT-entry finder -> 4-bit index.  Mirrors FP6E3M2NearestFinder.scala
//     (fixed-point |diff| masked to 9 bits, lower index wins on ties).
// BRANCHLESS: the `if (d < bd)` min-search over 16 unrolled
// steps overflows the Muon IPDOM reconvergence stack on RTL (WarpScheduler.sv:793 "ipdom
// stack is full" -> $stop) -- which is why this runtime quantizer was cyclotron-only.
// Arithmetic select is bit-identical (strict `<` = lower-index-wins) and uses zero IPDOM.
static inline int dev_abs9(int x) { int m = x >> 31; return ((x ^ m) - m) & 0x1FF; }
static inline uint8_t dev_fp6_nearest_finder(uint8_t in_code, const int lut_fixed[16]) {
  int fin = dev_fp6_to_fixed(in_code);
  int best = 0;
  int bd = dev_abs9(fin - lut_fixed[0]);
  #pragma unroll
  for (int i = 1; i < 16; i++) {
    int d  = dev_abs9(fin - lut_fixed[i]);
    int lt = (d < bd);                 // 0/1 (bare sltu, no branch); ties keep lower index
    best = dev_sel(lt, i, best);       // opaque asm select -> zero vx_split_n
    bd   = dev_sel(lt, d, bd);
  }
  return (uint8_t)best;
}

// SIMT body: (a) stage the canonical 16-entry fp6 LUT into A_lut (HW-packed, bit-compatible
// with load_lut()'s mesh read), (b) quantize every X activation element to its nearest LUT
// index and pack the row-pair nibble layout the mesh expects.
static void quantize_activation_body(void*, uint32_t tid, uint32_t tpb, uint32_t) {
  // fixed-point form of the LUT for the finder (per-lane, cheap).  Uses the SAME 16 codes
  // the mesh's A_lut was packed from -> the index the finder emits selects the value the
  // mesh will dequantize, so the chain is self-consistent by construction.
  int lut_fixed[16];
  #pragma unroll
  for (int i = 0; i < 16; i++) lut_fixed[i] = dev_fp6_to_fixed(CANON_LUT[i]);

  const uint32_t HWROWS = MATMUL_M / 2;
#ifdef RUNTIME_LUT
  // (a) stage the fp6 LUT into every A row-pair slot (HW-pack the 16 canonical 6-bit codes
  // into 3 words == pack_lut_hw_words / load_lut).  Single writer: the triplet stride is
  // non-power-of-two, and multi-thread coalescing of it drops words on this SIMT store model.
  if (tid == 0) {
    uint32_t l[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) l[i] = CANON_LUT[i];
    const uint32_t w0 = (l[0] | (l[1]<<6) | (l[2]<<12) | (l[3]<<18) | (l[4]<<24) | (l[5]<<30));
    const uint32_t w1 = ((l[5]>>2) | (l[6]<<4) | (l[7]<<10) | (l[8]<<16) | (l[9]<<22) | (l[10]<<28));
    const uint32_t w2 = ((l[10]>>4) | (l[11]<<2) | (l[12]<<8) | (l[13]<<14) | (l[14]<<20) | (l[15]<<26));
    for (uint32_t g = 0; g < HWROWS; g++) { A_lut[g][0]=w0; A_lut[g][1]=w1; A_lut[g][2]=w2; }
  }
#endif
  // quantize activations -> nearest 4-bit index, nibble-pack row pairs.
  const uint32_t ntasks = HWROWS * MATMUL_K;
  #pragma clang loop unroll(disable)
  for (uint32_t t = tid; t < ntasks; t += tpb) {
    const uint32_t i = t / MATMUL_K, k = t % MATMUL_K;
    uint8_t even = dev_fp6_nearest_finder(dev_bf16_to_fp6_code(X_bf16[2*i    ][k]), lut_fixed);
    uint8_t odd  = dev_fp6_nearest_finder(dev_bf16_to_fp6_code(X_bf16[2*i + 1][k]), lut_fixed);
    A_in_buf[i][k] = (uint8_t)((odd << 4) | (even & 0xF));
  }
  mu_fence();
}

static void matmul_body(void* raw_arg, uint32_t tid, uint32_t tpb, uint32_t tbid) {
  mxgemm<GEMM_CFG>(MATMUL_M, MATMUL_N, MATMUL_K,
                   reinterpret_cast<uint8_t*>(reinterpret_cast<uint32_t>((uint16_t*)C_raw)),
                   tid, tpb, tbid);
}

#define VERIFY_LANES (MU_NUM_THREADS * MU_NUM_CORES)
__global uint32_t lane_errors[VERIFY_LANES] = {0};
static void verify_body(void*, uint32_t tid, uint32_t tpb, uint32_t) {
  uint32_t e=0; for (uint32_t i=tid;i<VERIFY_COUNT;i+=tpb) if (C_raw[i]!=gold_raw[i]) e++;
  lane_errors[tid]=e; mu_fence();
}
static inline uint32_t hart_id(){uint32_t i;asm volatile("csrr %0, mhartid":"=r"(i)::"memory");return i;}

#ifdef QUANT_ONLY
// Debug: verify the runtime quantizer output (A_in_buf + A_lut) against the golden.
static void qverify_body(void*, uint32_t tid, uint32_t tpb, uint32_t) {
  uint32_t e=0;
#ifndef CHECK_LUT_ONLY
  for (uint32_t t=tid; t<(MATMUL_M/2)*MATMUL_K; t+=tpb)
    if (A_in_buf[t/MATMUL_K][t%MATMUL_K] != A_in_gold[t/MATMUL_K][t%MATMUL_K]) e++;
#endif
#ifndef CHECK_AIN_ONLY
  for (uint32_t g=tid; g<MATMUL_M/2; g+=tpb)
    for (int w=0; w<3; w++) if (A_lut[g][w] != A_lut_gold[w]) e++;
#endif
  lane_errors[tid]=e; mu_fence();
}
#endif
// Write-drain between the SIMT quantizer stores (A_in / staged A_lut) and the Gemmini
// operand DMA: without it the operand read backpressures against the still-draining SIMT
// stores and the mesh deadlocks (documented Muon write-drain hazard).  2000 suffices here.
#ifndef DRAIN_ITERS
#define DRAIN_ITERS 2000u
#endif
static inline void drain(){ for(volatile uint32_t d=0;d<DRAIN_ITERS;d++) asm volatile("":::"memory"); }
int main(){
  // Live chain: quantize activation to fp6 on device, drain, then run the fp6 matmul.
  mu_schedule(quantize_activation_body, nullptr, MX_NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES); mu_fence();          // write-drain: SIMT A/A_lut stores before Gemmini DMA reads
  drain(); mu_fence();
#ifdef QUANT_ONLY
  mu_schedule(qverify_body, nullptr, MX_NUM_WARPS); mu_barrier(0, MU_NUM_CORES);
  if(hart_id()!=0){for(;;){}}
  mu_fence(); uint32_t tt=0; for(uint32_t i=0;i<VERIFY_LANES;i++) tt+=lane_errors[i];
  uint32_t c2=tt?((tt<<1)|1u):0u; asm volatile(".insn i 0x73, 0, x0, %0, 0"::"r"(c2):"memory"); return 0;
#endif
  mu_schedule(matmul_body, nullptr, MX_NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
  mu_schedule(verify_body,nullptr,1); mu_barrier(0,MU_NUM_CORES);
  if(hart_id()!=0){for(;;){}}
  mu_fence(); uint32_t t=0; for(uint32_t i=0;i<VERIFY_LANES;i++) t+=lane_errors[i];
  uint32_t code=t?((t<<1)|1u):0u; asm volatile(".insn i 0x73, 0, x0, %0, 0"::"r"(code):"memory"); return 0;
}
