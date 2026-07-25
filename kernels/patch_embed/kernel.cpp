// SigLIP patch/conv embedding (SmolVLA / SmolVLM2-500M vision tower).
//
// Conv2d(3->OC, kernel=16, stride=16) + bias + learned position embedding.
// stride == kernel => non-overlapping patches => conv is a patchify + GEMM:
//
//   patch p at grid (gy,gx), K = C*16*16 :
//     col[p,(c,ky,kx)] = image[c, gy*16+ky, gx*16+kx]
//     out[p, oc] = sum_K col[p,K] * W[oc,(c,ky,kx)] + bias[oc] + pos[p,oc]
//
// The NEW piece vs a plain GEMM is the strided patchify addressing (conv-as-GEMM)
// + per-channel bias + position-embedding add. One output (patch, oc) per thread,
// grid-stride. fp32. Self-verifying vs a numpy fp32 golden; tohost 0 = pass.

// SMEM-staged conv-as-GEMM. The naive one-output-per-thread version streams every
// (patch,oc) thread's full 768-elem weight row through the 4KB no-landing-pads l0d;
// within a 16-lane warp the lanes touch 16 DISTINCT 3KB weight rows at once (48KB),
// thrashing the l0d so hard the kernel is ~77x over compute-bound (cyclotron 3.8M
// cycles) -> Verilator needs ~8h and cannot retire in budget. Fix (no l0d $stop here,
// this is a THROUGHPUT fix): stage the patchified col matrix A[NP,K] (48KB, materialized
// once via the strided patchify gather) into the 128KB cluster SMEM, then remap threads
//   lane -> patch p (0..NP-1),  warp -> output channel oc (grid-stride over OC)
// so a warp's 16 lanes all read the SAME W[oc,k] (one broadcast global load, 16x less
// weight traffic, no thrash) and DISTINCT col[p,k] from SMEM (no DRAM). Accumulation
// stays in k=(c,ky,kx) order -> same fp32 result. The register-lean flat k-loop keeps
// first-writes low so occupancy 4 (8 warps) stays well under the 256 rename limit.
//
// NOTE: NUM_WARPS MUST be defined before mu_intrinsics.h (an #ifndef there defaults 8).
#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"

extern "C" uint32_t __mu_num_warps = NUM_WARPS;

#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-3f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 2.0e-4f
#endif

#include "data"

struct KernelArgs {
  __global float *image, *weight, *bias, *pos, *out;
};

// SMEM-staged: stage col matrix A[NP,K] (patchify gather) into SMEM, then each warp
// takes an output channel oc and its 16 lanes compute the 16 patches for that oc.
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t HW = PE_IMG * PE_IMG;
  const uint32_t nw = threads_per_threadblock / MU_NUM_THREADS;  // total warps (both cores)
  const uint32_t A_B = 0;                                        // SMEM base: col[p][k], NP*K f32

  // ---- stage: materialize col[p][k] = image[c, gy*16+ky, gx*16+kx] into SMEM ----
  const __global float* img = a->image;
  for (uint32_t i = tid_in_threadblock; i < PE_NP * PE_K; i += threads_per_threadblock) {
    const uint32_t p = i / PE_K, k = i % PE_K;
    const uint32_t c = k / (PE_PATCH * PE_PATCH);
    const uint32_t r = k % (PE_PATCH * PE_PATCH);
    const uint32_t ky = r / PE_PATCH, kx = r % PE_PATCH;
    const uint32_t gy = p / PE_GRID, gx = p % PE_GRID;
    const uint32_t iy = gy * PE_PATCH + ky, ix = gx * PE_PATCH + kx;
    const float v = img[c * HW + iy * PE_IMG + ix];
    store_shared(A_B + i * 4, 0, __builtin_bit_cast(uint32_t, v));
  }
  mu_barrier(0, nw);

  // ---- compute: warp -> oc (grid-stride), lane -> patch p ----
  const uint32_t lane = tid_in_threadblock % MU_NUM_THREADS;  // patch p in [0, NP)
  const uint32_t warp = tid_in_threadblock / MU_NUM_THREADS;
  if (lane < PE_NP) {
    const uint32_t colbase = A_B + lane * PE_K * 4;            // this lane's col[p][*]
    for (uint32_t oc = warp; oc < PE_OC; oc += nw) {
      const __global float* wrow = a->weight + (uint32_t)oc * PE_K;  // broadcast across lanes
      float acc = 0.0f;
      #pragma clang loop unroll(disable)
      for (uint32_t k = 0; k < PE_K; k++) {
        const float cv = __builtin_bit_cast(float, load32_shared(colbase + k * 4));
        acc += cv * wrow[k];
      }
      const uint32_t idx = lane * PE_OC + oc;
      a->out[idx] = acc + a->bias[oc] + a->pos[idx];
    }
  } else {
    asm volatile("nop");
  }
}

static KernelArgs kernel_args;


int main() {
  kernel_args = {image_raw, weight_raw, bias_raw, pos_raw, out_raw};
  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);

  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  mu_verify_f32(out_raw, gold_raw, VERIFY_COUNT, TOLERANCE_REL, TOLERANCE_ABS);
  return 0;
}
