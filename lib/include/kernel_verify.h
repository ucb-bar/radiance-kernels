// kernel_verify.h -- shared verify + tohost tail for the coverage/op kernels.
//
// The op kernels all end main() the same way: compare the computed output against the
// committed golden with an FP tolerance on hart 0, then write a tohost code (0 = pass,
// otherwise (errors<<1)|1). This header replaces that copy-pasted tail.
//
//   #include "kernel_verify.h"
//   ...
//   mu_verify_f32(out_raw, gold_raw, VERIFY_COUNT, TOLERANCE_REL, TOLERANCE_ABS);
//
#ifndef KERNEL_VERIFY_H
#define KERNEL_VERIFY_H

#include <stdint.h>

static inline float mu_fabsf(float x) { return x < 0.0f ? -x : x; }

static inline bool mu_close(float c, float g, float rel, float abs_) {
  return mu_fabsf(c - g) <= rel * mu_fabsf(g) + abs_;
}

static inline uint32_t mu_hart_id() {
  uint32_t id;
  asm volatile("csrr %0, mhartid" : "=r"(id)::"memory");
  return id;
}

// Emit a tohost code (0 == pass) via the tohost CSR write instruction.
static inline void mu_tohost(uint32_t code) {
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
}

// Serial verify of `n` fp32 outputs vs golden (hart 0 only), then tohost.
// Mirrors the per-kernel tail: only hart 0 verifies; other harts park.
// Operands live in the __global address space (the kernels' data arrays).
static inline void mu_verify_f32(const __global float *out, const __global float *gold,
                                 uint32_t n, float rel, float abs_) {
  if (mu_hart_id() != 0) { for (;;) {} }
  uint32_t errors = 0;
  for (uint32_t i = 0; i < n; i++)
    if (!mu_close(out[i], gold[i], rel, abs_)) errors++;
  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);
}

#endif // KERNEL_VERIFY_H
