// SIMT MX-fp8 (e4m3) block quantizer: software replacement for the broken
// tapeout-330 hardware requantizer.
//
// Input : bf16 activation tile X[M][N] (M=64, N=512), stored as fp32 holding
//         bf16-exact values.
// Op    : for each contiguous 32-element block along N
//           amax  = max |x| in block
//           e     = ceil(log2(amax / 448))            (FP8_MAX = 448)
//           scale = 2^e, e8m0 code = e + 127 (clamped [0,254])
//           out_fp8[i] = quantize_e4m3(x[i] * 2^-e)   (round-to-nearest-even)
// Output: fp8 codes packed 4/uint32 (little-endian element order) + one e8m0
//         code per block.  Bit-exact vs a numpy golden using identical arithmetic.
//
// One thread == one 32-element block.  A block is 32 bytes = 8 word-aligned
// output words, so no two threads share an output word -> all stores are
// word-granular, race-free (no sub-word RMW hazard).
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"

// KERNEL_OCCUPANCY = warps spawned per core (mu_schedule -> vx_wspawn). This is
// deliberately NOT the header's NUM_WARPS (that is the hardware warp-slot count,
// fixed at 8 by VX_config.h). All spawned warps run the register-heavy quantize
// path and the 8 hardware slots share a 256-entry physical register file, so
// occupancy 8 (or 4) tips the rename union over 256 (RTL Rename.scala assert /
// cyclotron globalOverSubscription). Occupancy 2/core keeps a wide margin; the
// grid-stride loop still covers every block. Override with -DKERNEL_OCCUPANCY=N.
#ifndef KERNEL_OCCUPANCY
#define KERNEL_OCCUPANCY 2
#endif
extern "C" uint32_t __mu_num_warps = KERNEL_OCCUPANCY;

#include "data"

static inline uint32_t f32_bits(float f) {
  union { float f; uint32_t u; } c; c.f = f; return c.u;
}
static inline float bits_f32(uint32_t u) {
  union { float f; uint32_t u; } c; c.u = u; return c.f;
}

// round-half-to-even of (v >> sh), sh > 0
__attribute__((always_inline)) static inline uint32_t round_shift(uint32_t v, int sh) {
  if (sh >= 32) return 0u;  // shift >= word width: value underflows to 0 (avoids UB;
                            // a 24-bit significand is < 2^(sh-1) here, so it rounds to 0)
  uint32_t mask = (uint32_t(1) << sh) - 1u;
  uint32_t low = v & mask;
  uint32_t half = uint32_t(1) << (sh - 1);
  uint32_t q = v >> sh;
  if (low > half || (low == half && (q & 1u))) q++;
  return q;
}

// fp32 -> e4m3 code, round-to-nearest-even, saturate to +-448
__attribute__((always_inline)) static inline uint32_t f32_to_e4m3(float xf) {
  uint32_t x = f32_bits(xf);
  uint32_t sign = (x >> 31) & 1u;
  uint32_t absx = x & 0x7FFFFFFFu;
  uint32_t s = sign << 7;
  if (absx == 0u) return s;
  float axf = bits_f32(absx);
  if (axf >= 448.0f) return s | 0x7Eu;
  int e = int((absx >> 23) & 0xFFu) - 127;
  uint32_t man = absx & 0x7FFFFFu;
  uint32_t sig = (uint32_t(1) << 23) | man;   // value = sig * 2^(e-23)
  uint32_t out;
  if (e < -6) {                                // subnormal target (exp fixed -6)
    uint32_t q = round_shift(sig, 14 - e);     // value / 2^-9
    out = (q >= 8u) ? ((uint32_t(1) << 3) | (q - 8u)) : q;
  } else {                                      // normal target
    uint32_t r = round_shift(sig, 20);          // sig/2^20 in [8,16]
    int E = e;
    if (r == 16u) { r = 8u; E += 1; }
    int expf = E + 7;
    uint32_t mant = r - 8u;
    if (expf > 15 || (expf == 15 && mant == 7u)) return s | 0x7Eu;  // saturate
    out = (uint32_t(expf) << 3) | mant;
  }
  return s | out;
}

struct KernelArgs {
  __global float *X;       // [M*N] bf16-exact fp32
  __global uint32_t *out;  // [TOTAL_COUNT]: PACKED_COUNT fp8 words, then SCALE_COUNT scales
  uint32_t nblocks;
};

// One 32-element block, fully self-contained with all helpers inlined so it is a
// leaf (no calls): the register allocator keeps everything in caller-saved regs
// and the grid-stride loop below stays register-light. This split keeps the
// per-warp union of distinct architectural registers under the 256/8 rename
// budget (RTL Rename.scala / cyclotron globalOverSubscription), which a single
// fused kernel_body or a call-in-inner-loop structure both blew past.
__attribute__((noinline)) static void quantize_one_block(KernelArgs* a, uint32_t b) {
  // Derive the three block pointers here (not in the loop) so the grid-stride
  // loop in kernel_body doesn't strength-reduce into a dozen callee-saved stride
  // registers -> far smaller per-warp rename footprint.
  __global float* xb = a->X + b * 32u;
  __global uint32_t* ob = a->out + b * 8u;
  __global uint32_t* scl_slot = a->out + PACKED_COUNT + b;
  float amax = 0.0f;
  #pragma clang loop unroll(disable)
  for (uint32_t j = 0; j < 32u; j++) {
    float av = bits_f32(f32_bits(xb[j]) & 0x7FFFFFFFu);
    if (av > amax) amax = av;
  }
  float pow2;
  uint32_t scode;
  if (amax == 0.0f) { pow2 = 1.0f; scode = 0u; }
  else {
    // e = ceil(log2(amax / 448)), exact integer exponent extraction (448=1.75*2^8)
    uint32_t ab = f32_bits(amax);
    int e = (int((ab >> 23) & 0xFFu) - 127 - 8) + ((ab & 0x7FFFFFu) > 0x600000u ? 1 : 0);
    int sc = e + 127;
    scode = uint32_t(sc < 0 ? 0 : (sc > 254 ? 254 : sc));
    pow2 = bits_f32((uint32_t(127 - e)) << 23);   // 2^-e, exact power of two
  }
  *scl_slot = scode;
  // 8 output words per block, each packs 4 fp8 codes (little-endian order)
  #pragma clang loop unroll(disable)
  for (uint32_t w = 0; w < 8u; w++) {
    uint32_t word = 0u;
    #pragma clang loop unroll(disable)
    for (uint32_t k = 0; k < 4u; k++)
      word |= (f32_to_e4m3(xb[w * 4u + k] * pow2) & 0xFFu) << (8u * k);
    ob[w] = word;
  }
}

void kernel_body(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  auto* a = reinterpret_cast<KernelArgs*>(raw);
  const uint32_t nb = a->nblocks;
  #pragma clang loop unroll(disable)
  for (uint32_t b = tid; b < nb; b += tpb)
    quantize_one_block(a, b);
}

static KernelArgs kernel_args;
__global uint32_t out_all[TOTAL_COUNT] = {0};   // packed fp8 words, then e8m0 scales


// Verify in its own noinline frame, one tight loop over the two contiguous
// arrays, so its register footprint stays small on the main()-running warps
// (keeps the global rename union under 256).
__attribute__((noinline)) static uint32_t run_verify() {
  uint32_t errors = 0;
  for (uint32_t i = 0; i < TOTAL_COUNT; i++)
    if (out_all[i] != gold_all[i]) errors++;
  return errors;
}

int main() {
  kernel_args = {X_raw, out_all, NUM_BLOCKS};
  mu_schedule(kernel_body, &kernel_args, KERNEL_OCCUPANCY);
  mu_barrier(0, MU_NUM_CORES);

  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (mu_hart_id() != 0) { for (;;) {} }

  uint32_t errors = run_verify();
  uint32_t code = errors ? ((errors << 1) | 1u) : 0u;
  mu_tohost(code);
  return 0;
}
