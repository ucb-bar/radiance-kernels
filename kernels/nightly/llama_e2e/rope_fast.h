// rope_fast.h: the unfused ROPE phase (Q and K), included by kernel.cpp and kernel_vla.cpp after their quant_group /
// add_bf2 / a_code / a_scale.  Unit = (head, 64-token block) for Q or (KV head, 64-key block) for K, one warp each
// (gw of ngw), so a unit's lines have one writer.
//   1. TB = 16 / (hd / 32) tokens at a time: the warp computes each token's head row (bias, rotation) with the lanes on
//      consecutive elements (coalesced loads, each element once) into fp32 in the stage;
//   2. lane per 32-group (TB x hd / 32 = 16 groups): quant_group from the stage.
//   Q: codes straight into the A rows ((h % Gq) M + t).
//   K: codes staged [key][64 d] (pitch 17 words), one 64-d pass at a time (hd 256 in four), then written as whole
//      64-byte lines: K^T [hd][SK] (ATTN layout, d[25] = 1) or the blocked [SK/64][hd][64] (d[25] = 0).  The old rope_qk
//      stored K one byte per lane per group, 64 scattered byte stores per line.
// ROT: x1 c - x2 s for d < hd/2, x2 c + x1 s above (rope_row's operations, so the codes are bit-identical); ROT = false:
// no rotation (SigLIP: bias + layout only).  BIAS: x = bf16(x + b) first, b = d[27] (bf16 [NQKV]).  d[29]: Gq when
// Hkv = 0.  Needs P0 and M multiples of 64 (the dispatcher falls back to rope_qk otherwise).
#ifndef LL_ROPE_FAST_H
#define LL_ROPE_FAST_H

// one token's head row into res[hd] (fp32): lane l takes pairs j = l + 16 i (elements j and j + hd / 2)
template <uint32_t NJ, bool BIAS, bool ROT>
__attribute__((always_inline)) static inline void rope_fast_row(const uint16_t *src, const uint16_t *bsrc, const float *cs,
                                                                const float *sn, uint32_t lane, volatile __shared float *res) {
  constexpr uint32_t half = NJ * 16;
  uint32_t x1[NJ], x2[NJ];
  float cv[ROT ? NJ : 1], sv[ROT ? NJ : 1];
#pragma unroll
  for (uint32_t i = 0; i < NJ; i++) {
    x1[i] = src[lane + 16 * i]; x2[i] = src[lane + 16 * i + half];
    if (ROT) { cv[i] = cs[lane + 16 * i]; sv[i] = sn[lane + 16 * i]; }
  }
  if (BIAS) {
#pragma unroll
    for (uint32_t i = 0; i < NJ; i++) {
      x1[i] = bf16(bfv(x1[i]) + bfv(bsrc[lane + 16 * i]));
      x2[i] = bf16(bfv(x2[i]) + bfv(bsrc[lane + 16 * i + half]));
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < NJ; i++) {
    const uint32_t j = lane + 16 * i;
    res[j] = ROT ? bfv(x1[i]) * cv[i] - bfv(x2[i]) * sv[i] : bfv(x1[i]);
    res[j + half] = ROT ? bfv(x2[i]) * cv[i] + bfv(x1[i]) * sv[i] : bfv(x2[i]);
  }
}

template <uint32_t NJ, bool BIAS, bool ROT>
static void rope_qk_fast(const volatile uint32_t *d, uint32_t gw, uint32_t ngw, uint32_t lane,
                         volatile __shared uint32_t *stage) {
  constexpr uint32_t hd = NJ * 32, half = NJ * 16, NG = hd / 32, TB = 16 / NG, PITCH = 17;
  constexpr uint32_t RES_WORDS = TB * hd;                // 512
  static_assert(TB >= 1 && TB * NG == 16, "rope_qk_fast: a lane per group");
  static_assert((RES_WORDS + 64 * PITCH) * 4 <= STAGE_BYTES, "rope_qk_fast stage");
  volatile __shared float *res = (volatile __shared float *)stage;
  volatile __shared uint32_t *cst = stage + RES_WORDS;
  const uint32_t qkv = d[2], NQKV = d[3], Hq = d[4], Hkv = d[5], cosa = d[7], sina = d[8];
  const uint32_t qa = d[9], qasc = d[10], qa_bs = d[11], qasc_bs = d[12];
  const uint32_t kb = d[13], ksc = d[14], kb_bs = d[15], ksc_bs = d[16];
  const uint32_t P0 = d[21], M = d[22], SK = d[23], TM = d[24], fal = d[25], bias = d[27];
  const uint32_t Gq = d[29] ? d[29] : Hq / Hkv, Dq = Hq * hd, NBLK = M / 64;
  const uint32_t tt = lane / NG, g = lane % NG;          // this lane's (token of the batch, group)
  auto rows = [&](uint32_t t0, uint32_t col0) __attribute__((always_inline)) {   // TB tokens' rows into res
    for (uint32_t b = 0; b < TB; b++) {
      const uint32_t t = t0 + b;
      rope_fast_row<NJ, BIAS, ROT>((const uint16_t *)(qkv + (t * NQKV + col0) * 2), (const uint16_t *)(bias + col0 * 2),
                                   (const float *)cosa + t * half, (const float *)sina + t * half, lane, res + b * hd);
    }
    mu_fence_smem();
  };
  auto pair = [&](uint32_t jj) __attribute__((always_inline)) {
    const uint32_t e = tt * hd + 32 * g + 2 * jj;
    return F2{res[e], res[e + 1]};
  };
  // ---- Q ----
  for (uint32_t u = gw; u < Hq * NBLK; u += ngw) {
    const uint32_t h = u / NBLK, ub = (u % NBLK) * 64;
    const uint32_t aq = qa + (h / Gq) * qa_bs, asq = qasc + (h / Gq) * qasc_bs;
    for (uint32_t t0 = ub; t0 < ub + 64; t0 += TB) {
      rows(t0, h * hd);
      const uint32_t row = (h % Gq) * M + t0 + tt;
      quant_group(pair, lane, (uint32_t *)a_code(aq, TM, hd, 1, row, 32 * g), a_scale(asq, TM, hd, 1, row, 32 * g));
      mu_fence_smem();                                   // res read before the next batch rewrites it
    }
  }
  // ---- K ----
  for (uint32_t u = gw; u < Hkv * NBLK; u += ngw) {
    const uint32_t h = u / NBLK, key0 = P0 + (u % NBLK) * 64, nt = key0 / 64;
    for (uint32_t p = 0; p < hd / 64; p++) {             // 64 d (2 groups) per pass
      for (uint32_t k0 = 0; k0 < 64; k0 += TB) {
        rows(key0 - P0 + k0, Dq + h * hd);
        if (g / 2 == p) {
          const uint32_t kk = k0 + tt, key = key0 + kk;
          const uint32_t sa = fal ? ksc + h * ksc_bs + (key / 128) * NG * 128 + g * 128 + key % 128
                                  : ksc + h * ksc_bs + nt * NG * 64 + g * 64 + kk;
          quant_group(pair, lane, cst + kk * PITCH + 8 * (g % 2), sa);
        } else asm volatile("nop");
        mu_fence_smem();
      }
      for (uint32_t it = lane; it < 64 * 16; it += MU_NUM_THREADS) {
        const uint32_t dl = it / 16, w = it % 16, sh = 8 * (dl % 4), col = dl / 4, dd = 64 * p + dl;
        const uint32_t b0 = (cst[(4 * w) * PITCH + col] >> sh) & 0xFFu, b1 = (cst[(4 * w + 1) * PITCH + col] >> sh) & 0xFFu;
        const uint32_t b2 = (cst[(4 * w + 2) * PITCH + col] >> sh) & 0xFFu, b3 = (cst[(4 * w + 3) * PITCH + col] >> sh) & 0xFFu;
        const uint32_t dst = fal ? kb + h * kb_bs + dd * SK + key0 + 4 * w : kb + h * kb_bs + nt * hd * 64 + dd * 64 + 4 * w;
        *(uint32_t *)dst = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
      }
      mu_fence_smem();                                   // code stage read before the next pass / unit rewrites it
    }
  }
}
#endif
