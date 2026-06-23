#pragma once

#include <vx_intrinsics.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <stdint.h>

// fmastore: a compute-bound microbenchmark for Muon's out-of-order *issue*.
//
// Each thread runs CHAINS independent depth-2 produce->consume chains, laid out
// BLOCKED (chain 0's pair, then chain 1's pair, ...), NOT software-interleaved.
// In a blocked layout the ready "produce" of chain i+1 sits *behind* the stalled
// "consume" of chain i, so only the hardware RS issuing out of order can reach
// it. An in-order machine (or RS depth 1) stalls and stays latency-bound. Thus a
// cycles-vs-RS-depth (or cycles-vs-CHAINS) sweep isolates the OOO-issue benefit,
// with no memory-latency confound (the GEMV problem) because the only long edge
// per chain is the FMA, and the store (when used) hits on-chip SMEM.
//
// CONSUMER selects what reads the produced value:
//   STORE: a `sh.shared` to SMEM. A store has no rd, so it creates no WAW/WAR at
//          admission (Hazard.scala gates on hasRd) and drains on the LSU port, not
//          the FMA port -- the cleanest dependent consumer.
//   FMADD: a dependent fmadd into a side accumulator (same FMA port as produce).
//
// Parameters (all -D overridable):
//   FMASTORE_CHAINS    K, number of independent chains          (default 4)
//   FMASTORE_CONSUMER  0=store to SMEM, 1=dependent fmadd       (default store)
//   FMASTORE_NUM_WARPS occupancy; 1 isolates ILP from TLP       (default 1)
//   FMASTORE_ITERS     loop trip count (runtime -> no fold)     (default 1024)

#define FMASTORE_CONSUMER_STORE 0
#define FMASTORE_CONSUMER_FMADD 1

#ifndef FMASTORE_CHAINS
#define FMASTORE_CHAINS 4
#endif

#ifndef FMASTORE_CONSUMER
#define FMASTORE_CONSUMER FMASTORE_CONSUMER_STORE
#endif

#ifndef FMASTORE_NUM_WARPS
#define FMASTORE_NUM_WARPS 1
#endif

#ifndef FMASTORE_ITERS
#define FMASTORE_ITERS 1024
#endif

extern "C" uint32_t __mu_num_warps = FMASTORE_NUM_WARPS;

struct FmaStoreArgs {
  __global _Float16* out;
  _Float16 m;     // multiplier   (runtime, so the recurrence cannot be folded)
  _Float16 b;     // addend
  _Float16 seed;  // base accumulator seed; per-(thread,chain) offset added
  uint32_t iters; // loop trip count (runtime)
};

// SMEM scratchpad base (address space 1).
__shared _Float16* const smem = reinterpret_cast<__shared _Float16*>(0x0);

template <uint32_t CHAINS, uint32_t CONSUMER>
static inline void fmastore_impl(
    void* arg,
    uint32_t tid_in_threadblock,
    uint32_t /*threads_per_threadblock*/,
    uint32_t /*threadblock_id*/) {
  static_assert(CHAINS > 0, "need at least one chain");
  auto* args = reinterpret_cast<FmaStoreArgs*>(arg);

  const _Float16 m = args->m;
  const _Float16 b = args->b;
  const _Float16 seed = args->seed;
  const uint32_t iters = args->iters;
  const uint32_t tid = tid_in_threadblock;

  // K independent accumulators with distinct, runtime-derived seeds: the
  // recurrence cannot be constant-folded and lanes are not uniform.
  _Float16 acc[CHAINS];
  #pragma unroll
  for (uint32_t i = 0; i < CHAINS; ++i)
    acc[i] = seed + static_cast<_Float16>(tid * CHAINS + i);

  // Side accumulators for the fmadd-consumer variant.
  _Float16 sink[CHAINS];
  #pragma unroll
  for (uint32_t i = 0; i < CHAINS; ++i)
    sink[i] = static_cast<_Float16>(0);

  // One SMEM slot per (thread, chain).
  __shared _Float16* const slot = smem + tid * CHAINS;

  for (uint32_t it = 0; it < iters; ++it) {
    #pragma unroll
    for (uint32_t i = 0; i < CHAINS; ++i) {
      // produce: depth-1 self-recurrent fmadd (rd == rs1). The WAW on acc[i] is
      // redundant with the RAW recurrence (same scoreboard counter) and the
      // blocking spaces successive writes 2K instructions apart, so it never
      // gates admission.
      acc[i] = acc[i] * m + b;

#if FMASTORE_CONSUMER == FMASTORE_CONSUMER_STORE
      // consume: store the fresh value to SMEM (sh.shared, no rd).
      slot[i] = acc[i];
#else
      // consume: dependent fmadd reading acc[i] into a side accumulator.
      sink[i] = acc[i] * m + sink[i];
#endif

      // Pin the BLOCKED order (emits no instructions):
      //   - launder acc[i]            -> produce_i / consume_i stay put
      //   - launder acc[(i+1)%CHAINS] -> produce_{i+1} cannot hoist above here
      //   - "memory" clobber          -> the SMEM store stays in place
      // Without this the compiler would software-pipeline (hoist the ready
      // produces up), which an in-order machine could also exploit -- defeating
      // the point of measuring OOO issue.
      if constexpr (CHAINS == 1) {
        asm volatile("" : "+r"(acc[i]) : : "memory");
      } else {
        asm volatile("" : "+r"(acc[i]), "+r"(acc[(i + 1) % CHAINS]) : : "memory");
      }
#if FMASTORE_CONSUMER == FMASTORE_CONSUMER_FMADD
      asm volatile("" : "+r"(sink[i]) : : "memory");
#endif
    }
  }

  // Observe the results so nothing is dead-code-eliminated.
  __global _Float16* out = args->out;
  #pragma unroll
  for (uint32_t i = 0; i < CHAINS; ++i) {
#if FMASTORE_CONSUMER == FMASTORE_CONSUMER_STORE
    out[tid * CHAINS + i] = acc[i];
#else
    out[tid * CHAINS + i] = acc[i] + sink[i];
#endif
  }
}

static inline void fmastore(
    void* arg,
    uint32_t tid_in_threadblock,
    uint32_t threads_per_threadblock,
    uint32_t threadblock_id) {
  fmastore_impl<FMASTORE_CHAINS, FMASTORE_CONSUMER>(
      arg, tid_in_threadblock, threads_per_threadblock, threadblock_id);
}

// Sized for the largest threadblock (all warps populated) x CHAINS.
__global _Float16 fmastore_out[MU_NUM_MAX_WARPS * MU_NUM_THREADS * FMASTORE_CHAINS] = {};

FmaStoreArgs fmastore_args = {
  .out = nullptr,
  .m = static_cast<_Float16>(0.5),   // |m| < 1 so acc -> fixed point b/(1-m), bounded
  .b = static_cast<_Float16>(0.5),
  .seed = static_cast<_Float16>(0.0),
  .iters = FMASTORE_ITERS,
};

int main() {
  fmastore_args.out = fmastore_out;
  mu_schedule(fmastore, &fmastore_args, FMASTORE_NUM_WARPS);
  return 0;
}
