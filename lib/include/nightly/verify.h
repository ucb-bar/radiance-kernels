/*
 * nightly/verify.h: GPU-side result check, run by a kernel after nightly_kernel_end().
 *
 * The host Rocket is far too slow to compare megabytes in RTL simulation (no FPU, one or two
 * cache misses in flight, ~10 instructions per element).  Instead every cluster compares the
 * WHOLE output tensor against the golden with all its threads, reduces the statistics in SMEM,
 * and posts them to its printBuf slots NV_SLOT.. (the host reads cluster 0, and may cross-check
 * cluster 1, which computed the same numbers independently).  The host still spot-checks a
 * sample itself (nightly_host_spot_check).  The check runs after the END stamp, so it is not
 * part of the kernel time.
 *
 * printBuf slots (per cluster):
 *   NV_SLOT+0  NV_DONE when posted          +4  bad (differ by more than one bf16 step)
 *   NV_SLOT+1  n                            +5  poison (output == NV_POISON16, never written)
 *   NV_SLOT+2  exact                        +6  nonfinite outputs
 *   NV_SLOT+3  within one bf16 step         +7  relative Frobenius error x 1e4 (0.01 % units)
 */
#ifndef NIGHTLY_VERIFY_H
#define NIGHTLY_VERIFY_H

#include <stdint.h>
#include <nightly/pb.h>

#define NV_SLOT 16u
#define NV_DONE 0x5EF1D0E5u
#define NV_POISON16 0xEEEEu
/* The check reads the whole tensor, including the parts other clusters write, so it must not start
 * before every cluster has finished: the host writes NV_GO_MAGIC to slot NV_GO once it has seen
 * every core's END stamp, and the check waits for it. */
#define NV_GO 24u
#define NV_GO_MAGIC 0x60C0FFEEu

#ifdef RADIANCE_DEVICE
#include <nightly/device.h>

#ifndef NV_SMEM
#define NV_SMEM 0u   /* SMEM scratch for the reduction: threads x 32 B, reused after the kernel */
#endif
#define NV_BAR 14u

static inline float nv_bf16(uint32_t b) { return __builtin_bit_cast(float, (b & 0xFFFFu) << 16); }

/* Compare n bf16 values (n even); `out`/`gold` are device addresses.  All threads call it. */
/* Optional per-tile diagnosis: with row length `cols` and tile shape tr x tc, the number of bad
 * elements of each of the first 16 tiles (row-major tile order) goes to printBuf 32..47. */
static __attribute__((noinline)) void nightly_verify_bf16(uint32_t out, uint32_t gold, uint32_t n,
                                                          uint32_t tid, uint32_t tpb,
                                                          uint32_t cols = 0, uint32_t tr = 0,
                                                          uint32_t tc = 0) {
  if (cols) { for (uint32_t t = 0; t < 16; t++) ((volatile __shared uint32_t *)(NV_SMEM + tpb * 32 + tid * 64))[t] = 0; }
  if (tid == 0) { while (nightly_pb_get(NV_GO) != NV_GO_MAGIC) asm volatile("nop"); }
  else asm volatile("nop");
  mu_barrier(NV_BAR, tpb / MU_NUM_THREADS);
  const volatile uint32_t *o32 = (const volatile uint32_t *)out;
  const uint32_t *g32 = (const uint32_t *)gold;
  uint32_t exact = 0, ulp1 = 0, bad = 0, poison = 0, nonfin = 0;
  float se = 0.f, sr = 0.f;
  for (uint32_t i = tid; i < n / 2; i += tpb) {
    const uint32_t ow = o32[i], gw = g32[i];
    for (uint32_t h = 0; h < 2; h++) {
      const uint32_t o = (ow >> (16 * h)) & 0xFFFFu, g = (gw >> (16 * h)) & 0xFFFFu;
      if (o == NV_POISON16) poison++;
      if (((o >> 7) & 0xFF) == 0xFF) nonfin++;
      if (o == g) exact++;
      else {
        const int d = (int)(o & 0x7FFF) - (int)(g & 0x7FFF);
        const bool same = (o >> 15) == (g >> 15);
        if ((same && (d == 1 || d == -1)) || (((o | g) & 0x7FFF) <= 0x0080)) ulp1++;
        else {
          bad++;
          if (cols) {
            const uint32_t e = 2 * i + h, r = e / cols, c = e % cols, t = (r / tr) * (cols / tc) + c / tc;
            if (t < 16) {
              volatile __shared uint32_t *tb = (volatile __shared uint32_t *)(NV_SMEM + tpb * 32 + tid * 64);
              tb[t] = tb[t] + 1;
            }
          }
        }
      }
      const float fo = nv_bf16(o), fg = nv_bf16(g);
      if (((o >> 7) & 0xFF) != 0xFF) { se += (fo - fg) * (fo - fg); }
      sr += fg * fg;
    }
  }
  volatile __shared uint32_t *red = (volatile __shared uint32_t *)(NV_SMEM + tid * 32);
  red[0] = exact; red[1] = ulp1; red[2] = bad; red[3] = poison; red[4] = nonfin;
  red[5] = __builtin_bit_cast(uint32_t, se); red[6] = __builtin_bit_cast(uint32_t, sr);
  mu_fence_smem();
  mu_barrier(NV_BAR, tpb / MU_NUM_THREADS);
  if (tid == 0) {
    uint32_t t[5] = {0, 0, 0, 0, 0};
    float tse = 0.f, tsr = 0.f;
    for (uint32_t k = 0; k < tpb; k++) {
      volatile __shared uint32_t *r = (volatile __shared uint32_t *)(NV_SMEM + k * 32);
      for (int q = 0; q < 5; q++) t[q] += r[q];
      tse += __builtin_bit_cast(float, (uint32_t)r[5]);
      tsr += __builtin_bit_cast(float, (uint32_t)r[6]);
    }
    const float rel = tsr > 0.f ? __builtin_sqrtf(tse / tsr) : 0.f;
    nightly_pb_put(NV_SLOT + 1, n);
    nightly_pb_put(NV_SLOT + 2, t[0]);
    nightly_pb_put(NV_SLOT + 3, t[1]);
    nightly_pb_put(NV_SLOT + 4, t[2]);
    nightly_pb_put(NV_SLOT + 5, t[3]);
    nightly_pb_put(NV_SLOT + 6, t[4]);
    nightly_pb_put(NV_SLOT + 7, (uint32_t)(rel * 10000.f + 0.5f));
    if (cols) {   /* diagnosis: bad elements per tile -> printBuf 32..47 */
      for (uint32_t t = 0; t < 16; t++) {
        uint32_t sum = 0;
        for (uint32_t k = 0; k < tpb; k++) sum += ((volatile __shared uint32_t *)(NV_SMEM + tpb * 32 + k * 64))[t];
        nightly_pb_put(32 + t, sum);
      }
    }
    nightly_pb_put(NV_SLOT + 0, NV_DONE);
  } else {
    asm volatile("nop");
  }
  mu_barrier(NV_BAR, tpb / MU_NUM_THREADS);
}
#endif /* RADIANCE_DEVICE */

#endif
