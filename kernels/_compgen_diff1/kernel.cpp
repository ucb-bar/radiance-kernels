// Tier-2 self-verifying kernel: y = a + b over 4 f32 lanes.
// Compares against expected values baked in from CompGen's
// golden_outputs.pt and signals via $finish vs cycle-timeout.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#define N 4

struct Args {
  __global float* A;
  __global float* B;
  __global float* C;
  __global float* EXP;
  __global volatile uint32_t* mismatch;
};

static inline void add4_selfcheck(
    void* arg, uint32_t tid, uint32_t nthreads, uint32_t /*blk*/) {
  auto* a = reinterpret_cast<Args*>(arg);
  // Single-threaded computation for minimal sim footprint — one warp,
  // one active lane. N is tiny so the serialization cost is zero-
  // meaningful.
  if (tid == 0) {
    for (uint32_t i = 0; i < N; ++i) {
      a->C[i] = a->A[i] + a->B[i];
    }
    uint32_t bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
      // Bit-compare as u32 so NaN-vs-NaN and -0 vs +0 surface as
      // mismatches (CompGen's golden is from the PyTorch eager path
      // on CPU; we want bit-exact agreement).
      uint32_t got = __builtin_bit_cast(uint32_t, a->C[i]);
      uint32_t exp = __builtin_bit_cast(uint32_t, a->EXP[i]);
      if (got != exp) { bad = 1; break; }
    }
    if (bad) {
      *a->mismatch = 0xDEADBEEFu;
      for (;;) { /* stay alive so GPUResetAggregator never fires */ }
    }
    // fallthrough: single-lane warp goes idle on return; schedulers'
    // epilogue (_exit) issues `tmc x0`, all warps inactive, aggregator
    // counts 1024 idle cycles, $finish.
  }
}

__global float A_raw[N]   = { 0x1.3ad8020000000p+1f, -0x1.dc16a20000000p-2f, 0x1.31e3420000000p-1f, 0x1.4ca9ba0000000p-1f };
__global float B_raw[N]   = { -0x1.d149540000000p-2f, -0x1.65cfd00000000p+0f, -0x1.42ae4c0000000p-2f, 0x1.4ae9980000000p-4f };
__global float C_raw[N]   = { 0.0f, 0.0f, 0.0f, 0.0f };
__global float EXP_raw[N] = { 0x1.00aed80000000p+1f, -0x1.dcd5780000000p+0f, 0x1.2118380000000p-2f, 0x1.7606ec0000000p-1f };
__global volatile uint32_t mismatch_flag = 0u;

static Args args = { nullptr, nullptr, nullptr, nullptr, nullptr };

int main() {
  args.A = A_raw;
  args.B = B_raw;
  args.C = C_raw;
  args.EXP = EXP_raw;
  args.mismatch = &mismatch_flag;
  mu_schedule(add4_selfcheck, &args, 1);
  return 0;
}
