// rope_fast.h: the unfused ROPE phase (Q, K and V), included by kernel.cpp and kernel_vla.cpp after their quant_group /
// add_bf2 / a_code / a_scale.  Units: Q (head, 64-token block), K (KV head, 64-key block), V (KV head, 32-key block),
// one list dealt round-robin over the owners (own of nown = clusters x cores, Q first, then K, then V); the owner's
// warps (wi of nwo) split the unit, so a unit's lines have one writing core (warps of one core may share lines).
//   1. TB = 16 / (hd / 32) tokens at a time: the warp computes each token's head row (bias, rotation) with the lanes on
//      consecutive elements (coalesced loads, each element once) into fp32 in the stage;
//   2. lane per 32-group (TB x hd / 32 = 16 groups): quant_group from the stage.
//   Q: the warp takes batches of TB tokens (wi, wi + nwo, ...); codes straight into the A rows ((h % Gq) M + t).
//   K: the warp takes 16-key sub-blocks (wi, wi + nwo, ...); every group's codes staged [key][hd bytes] (pitch
//      hd / 4 + 1 words), then the warp writes its 4-key words of each K^T row: K^T [hd][SK] (ATTN layout, d[25] = 1)
//      or the blocked [SK/64][hd][64] (d[25] = 0).  CEN (Qwen2 key centering): the head's fp32 center (d[28]) is
//      subtracted after the rotation, as the fused rope_k_unit does.
//   V: lane per column; the unit's 16-column chunks are dealt over the warps (rope_v).
// ROT: x1 c - x2 s for d < hd/2, x2 c + x1 s above (rope_row's operations, so the codes are bit-identical); ROT = false:
// no rotation (SigLIP: bias + layout only).  BIAS: x = bf16(x + b) first, b = d[27] (bf16 [NQKV]).  d[29]: Gq when
// Hkv = 0.  Needs P0 and M multiples of 64 (the dispatcher falls back to rope_qk otherwise).
// d[31] = SB (batched SmolVLA expert, 0 = off): the M token rows are M / SB samples of SB rows (SB a multiple of 64);
// sample s's Q goes to batches s (Hq / Gq) + h / Gq (rows (h % Gq) SB + t % SB), its keys to positions P0 + t % SB of
// cache batch s Hkv + h.
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

// the first unit index >= 0 of a section that starts at flat unit `off`, for owner own (units go to owner U % nown)
static inline uint32_t rope_first(uint32_t own, uint32_t nown, uint32_t off) { return (own + nown - off % nown) % nown; }

template <uint32_t NJ, bool BIAS, bool ROT, bool CEN = false>
static void rope_qk_fast(const volatile uint32_t *d, uint32_t own, uint32_t nown, uint32_t wi, uint32_t nwo, uint32_t lane,
                         volatile __shared uint32_t *stage) {
  constexpr uint32_t hd = NJ * 32, half = NJ * 16, NG = hd / 32, TB = 16 / NG, KP = hd / 4 + 1;
  constexpr uint32_t RES_WORDS = TB * hd;                // 512
  static_assert(TB >= 1 && TB * NG == 16, "rope_qk_fast: a lane per group");
  static_assert((RES_WORDS + 16 * KP + (CEN ? hd : 0)) * 4 <= STAGE_BYTES, "rope_qk_fast stage");
  volatile __shared float *res = (volatile __shared float *)stage;
  volatile __shared uint32_t *cst = stage + RES_WORDS;   // K codes [16 keys][KP words]
  volatile __shared float *cen = (volatile __shared float *)(stage + RES_WORDS + 16 * KP);
  const uint32_t qkv = d[2], NQKV = d[3], Hq = d[4], Hkv = d[5], cosa = d[7], sina = d[8];
  const uint32_t qa = d[9], qasc = d[10], qa_bs = d[11], qasc_bs = d[12];
  const uint32_t kb = d[13], ksc = d[14], kb_bs = d[15], ksc_bs = d[16];
  const uint32_t P0 = d[21], M = d[22], SK = d[23], TM = d[24], fal = d[25], bias = d[27];
  const uint32_t Gq = d[29] ? d[29] : Hq / Hkv, Dq = Hq * hd, NBLK = M / 64;
  const uint32_t SB = d[31] ? d[31] : M, HG = Hq / Gq;   // rows per sample, query batches per sample
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
  const uint32_t NQ = Hq * NBLK;
  for (uint32_t u = own; u < NQ; u += nown) {
    const uint32_t h = u / NBLK, ub = (u % NBLK) * 64, qb = (ub / SB) * HG + h / Gq;
    const uint32_t aq = qa + qb * qa_bs, asq = qasc + qb * qasc_bs;
    for (uint32_t t0 = ub + wi * TB; t0 < ub + 64; t0 += nwo * TB) {
      rows(t0, h * hd);
      const uint32_t row = (h % Gq) * SB + (t0 + tt) % SB;
      quant_group(pair, lane, (uint32_t *)a_code(aq, TM, hd, 1, row, 32 * g), a_scale(asq, TM, hd, 1, row, 32 * g));
      mu_fence_smem();                                   // res read before the next batch rewrites it
    }
  }
  // ---- K ----
  for (uint32_t u = rope_first(own, nown, NQ); u < Hkv * NBLK; u += nown) {
    const uint32_t h = u / NBLK, tok0 = (u % NBLK) * 64, hb = (tok0 / SB) * Hkv + h, key0 = P0 + tok0 % SB, nt = key0 / 64;
    if (CEN) {
      for (uint32_t i = lane; i < hd; i += MU_NUM_THREADS) cen[i] = ((const float *)d[28])[h * hd + i];
      mu_fence_smem();
    }
    for (uint32_t sk = wi; sk < 4; sk += nwo) {          // 16-key sub-block
      for (uint32_t k0 = 16 * sk; k0 < 16 * sk + 16; k0 += TB) {
        rows(tok0 + k0, Dq + h * hd);
        const uint32_t kk = k0 + tt, key = key0 + kk;
        const uint32_t sa = fal ? ksc + hb * ksc_bs + (key / 128) * NG * 128 + g * 128 + key % 128
                                : ksc + hb * ksc_bs + nt * NG * 64 + g * 64 + kk;
        auto pc = [&](uint32_t jj) __attribute__((always_inline)) {
          const F2 r = pair(jj);
          if (!CEN) return r;
          const uint32_t e = 32 * g + 2 * jj;
          return F2{r.a - cen[e], r.b - cen[e + 1]};
        };
        quant_group(pc, lane, cst + (kk % 16) * KP + 8 * g, sa);
        mu_fence_smem();                                 // res read before the next batch rewrites it
      }
      // word (d, q): the codes of keys 4 q .. 4 q + 3 of the sub-block at d
      for (uint32_t it = lane; it < hd * 4; it += MU_NUM_THREADS) {
        const uint32_t dl = it / 4, q = it % 4, sh = 8 * (dl % 4), col = dl / 4, kq = 16 * sk + 4 * q;
        const uint32_t b0 = (cst[(4 * q) * KP + col] >> sh) & 0xFFu, b1 = (cst[(4 * q + 1) * KP + col] >> sh) & 0xFFu;
        const uint32_t b2 = (cst[(4 * q + 2) * KP + col] >> sh) & 0xFFu, b3 = (cst[(4 * q + 3) * KP + col] >> sh) & 0xFFu;
        const uint32_t dst = fal ? kb + hb * kb_bs + dl * SK + key0 + kq : kb + hb * kb_bs + nt * hd * 64 + dl * 64 + kq;
        *(uint32_t *)dst = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
      }
      mu_fence_smem();                                   // code tile read before the next sub-block rewrites it
    }
  }
}

// V: unit (KV head, 32 keys), after the Q and K units in the owners' rotation; lane per column dd = d0 + lane, the
// unit's 16-column chunks d0 dealt over the owner's warps; amax over the 32 keys is lane-local.  The 32 x 16 code
// bytes of a chunk go through the warp's stage (one word per byte), then lanes store whole words.
template <bool BIAS = false>
static void rope_v(const volatile uint32_t *d, uint32_t own, uint32_t nown, uint32_t wi, uint32_t nwo, uint32_t lane,
                   volatile __shared uint32_t *stage) {
  const uint32_t qkv = d[2], NQKV = d[3], Hq = d[4], Hkv = d[5], hd = d[6], bias = d[27];
  const uint32_t vb = d[17], vsc = d[18], vb_bs = d[19], vsc_bs = d[20];
  const uint32_t P0 = d[21], M = d[22], SK = d[23], fal = d[25];
  const uint32_t Dq = Hq * hd, Dkv = Hkv * hd, KSv = SK / 256;
  const uint32_t SB = d[31] ? d[31] : M;                 // d[31]: rows per sample (batched expert)
  for (uint32_t it = rope_first(own, nown, (Hq + Hkv) * (M / 64)); it < Hkv * (M / 32); it += nown) {
    const uint32_t h = it / (M / 32), tg = it % (M / 32), key0 = P0 + (tg * 32) % SB, hb = ((tg * 32) / SB) * Hkv + h;
    for (uint32_t d0 = MU_NUM_THREADS * wi; d0 < hd; d0 += MU_NUM_THREADS * nwo) {
      const uint32_t dd = d0 + lane;
      const uint16_t *src = (const uint16_t *)(qkv + ((tg * 32) * NQKV + Dq + Dkv + h * hd + dd) * 2);
      uint32_t v[32];
#pragma unroll
      for (uint32_t k = 0; k < 32; k++) v[k] = src[k * NQKV];
      if (BIAS) {   // the column's bias, bf16(v + b) as in rope_row
        const float b = bfv(((const uint16_t *)bias)[Dq + Dkv + h * hd + dd]);
#pragma unroll
        for (uint32_t k = 0; k < 32; k++) v[k] = bf16(bfv(v[k]) + b);
      }
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
      const uint32_t blk = vb + hb * vb_bs + ((key0 / 256) * (hd / 64) + d0 / 64) * 256 * 64 + d0 % 64;
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) {
        const uint32_t w = lane + 16 * j, k = w / 4, b0 = (w % 4) * 4;
        const uint32_t dst = fal ? vb + hb * vb_bs + (key0 + k) * hd + d0 + b0 : blk + ((key0 + k) % 256) * 64 + b0;
        *(uint32_t *)dst = stage[k * 16 + b0] | (stage[k * 16 + b0 + 1] << 8) | (stage[k * 16 + b0 + 2] << 16) |
                           (stage[k * 16 + b0 + 3] << 24);
      }
      const uint32_t sdst = fal ? vsc + hb * vsc_bs + (key0 / 128) * 4 * hd + ((key0 % 128) / 32) * hd + d0
                                : vsc + hb * vsc_bs + ((d0 / 64) * KSv + key0 / 256) * 8 * 64 + ((key0 % 256) / 32) * 64 + d0 % 64;
      if (lane < 4)
        *(uint32_t *)(sdst + 4 * lane) = stage[512 + 4 * lane] | (stage[512 + 4 * lane + 1] << 8) |
                                         (stage[512 + 4 * lane + 2] << 16) | (stage[512 + 4 * lane + 3] << 24);
      else asm volatile("nop");
      mu_fence_smem();                                   // reads done before the next chunk rewrites the stage
    }
  }
}
#endif
