// Built on Spike/riscv-isa-sim MX primitives (mx_fp_math.h), BSD-3-Clause. See NOTICE.
// Host reference ("golden") for the MX-Gemmini FP8 matmul, in HARDWARE semantics.
//
// WHY THIS EXISTS
// ---------------
// The obvious golden -- radiance-kernels' `fp8_matmul_model.py`
// (`matmul_outer_quantized_hwlike`, whose output is baked into the kernels'
// `C_out_bf16` headers) -- is an *idealized* model: it bf16-rounds after every single
// k. The real accelerator (and spike libgemmini, and RTL) instead accumulates through a
// 16-deep systolic array with a per-column precision schedule (acc_e/acc_m), scaling
// once per 32-element K group. The two disagree materially: real spike matches that
// ideal golden on only 434/4096 elements of a 64x64 tile (mean rel err ~2.5%). So the
// ideal model cannot gate correctness at any tight tolerance.
//
// This program reproduces the HARDWARE semantics. It is deliberately implemented
// independently of cyclotron's Rust co-model (it reuses spike's own `mx_fp_math.h`
// primitives), so it can serve as an out-of-band correctness gate for that model and
// for the MX-Gemmini kernels. Verified bit-exact against REAL spike
// (`spike --extension=gemmini`, via the gemmini-rocc-tests radiance_xcheck_* programs):
//   fp8 64x64x64  = 4096/4096,  fp4 128x128x128 = 16384/16384,
//   fp6 128x128x128 (LUT-indexed) = 16384/16384.
//
// Usage:
//   mx_golden M N K A.bin B.bin A_scales.bin B_scales.bin out_C.bin [fmt] [lutA.bin lutB.bin G]
//     fmt          : 0 = fp8 e4m3 (default), 2 = fp4 e2m1, 1 = fp6 e3m2 (LUT-indexed)
//     A.bin        : fp8: M*K uint8 [M][K].  sub-byte: (M/2)*K uint8, nibble-packed along M
//     B.bin        : fp8: K*N uint8 [K][N].  sub-byte: K*(N/2) uint8, nibble-packed along N
//     A_scales.bin : (K/32)*M uint8  e8m0 codes, [group][m]
//     B_scales.bin : (K/32)*N uint8  e8m0 codes, [group][n]
//     out_C.bin    : M*N  uint16  bf16, row-major [M][N]
#include "mx_fp_math.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <functional>
#include <cmath>
#include <cstring>

static const int DIM = 16;    // Gemmini systolic dimension
static const int GROUP = 32;  // MX block-scaling group size

// Per-column accumulator precision as partial sums flow down the 16-deep array.
// (spike libgemmini gemmini.cc::mx_loop_ws_spad)
static const int ACC_E[16] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8};
static const int ACC_M[16] = {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 7};
static const int PROD_E = 4, PROD_M = 3;

static std::vector<uint8_t> slurp(const char *path, size_t want) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "mx_golden: cannot open %s\n", path); exit(1); }
  std::vector<uint8_t> v(want);
  if (fread(v.data(), 1, want, f) != want) {
    fprintf(stderr, "mx_golden: short read on %s (want %zu)\n", path, want);
    exit(1);
  }
  fclose(f);
  return v;
}

int main(int argc, char **argv) {
  if (argc < 9) {
    fprintf(stderr,
            "usage: %s M N K A.bin B.bin A_scales.bin B_scales.bin out_C.bin\n", argv[0]);
    return 1;
  }
  using namespace mx;
  const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
  if (M % DIM || N % DIM || K % DIM || K % GROUP) {
    fprintf(stderr, "mx_golden: M,N,K must be multiples of %d and K of %d\n", DIM, GROUP);
    return 1;
  }
  const int TK = K / DIM, GK = K / GROUP;

  const int fmt = (argc >= 10) ? atoi(argv[9]) : 0;   // 0 fp8, 1 fp6, 2 fp4
  const bool sub = (fmt != 0);
  const int TILE = sub ? 32 : DIM;                     // rows/cols per PE tile
  // sub-byte operands are nibble-packed: A along M, B along N
  auto A = slurp(argv[4], sub ? (size_t)(M / 2) * K : (size_t)M * K);
  auto B = slurp(argv[5], sub ? (size_t)K * (N / 2) : (size_t)K * N);
  auto SA = slurp(argv[6], (size_t)GK * M);
  auto SB = slurp(argv[7], (size_t)GK * N);
  // fp6: 4-bit nibbles are indices into 16-entry LUTs of 6-bit e3m2 codes
  std::vector<uint8_t> LA, LB;
  int G = 0;
  if (fmt == 1) {
    if (argc < 13) { fprintf(stderr, "mx_golden: fp6 needs lutA.bin lutB.bin G\n"); return 1; }
    G = atoi(argv[12]);
    LA = slurp(argv[10], (size_t)(((M - 1) >> G) + 1) * 16);
    LB = slurp(argv[11], (size_t)(((N - 1) >> G) + 1) * 16);
  }
  const int TI = M / TILE, TJ = N / TILE;

  auto dec_a = [&](int i, int m, int k_outer, int kk) -> float {
    if (!sub) return fp8_e4m3_decode(A[(size_t)(i * TILE + m) * K + k_outer * DIM + kk]);
    const int row = (i * TILE + m) >> 1;                       // packed row
    uint8_t byte = A[(size_t)row * K + k_outer * DIM + kk];
    uint8_t nib = (m & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    if (fmt == 2) return fp4_e2m1_decode(nib);
    return fp6_e3m2_decode(LA[(size_t)((i * TILE + m) >> G) * 16 + nib]);
  };
  auto dec_b = [&](int j, int n, int k_outer, int kk) -> float {
    if (!sub) return fp8_e4m3_decode(B[(size_t)(k_outer * DIM + kk) * N + j * TILE + n]);
    const int col = (j * TILE + n) >> 1;                       // packed col
    uint8_t byte = B[(size_t)(k_outer * DIM + kk) * (N / 2) + col];
    uint8_t nib = (n & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    if (fmt == 2) return fp4_e2m1_decode(nib);
    return fp6_e3m2_decode(LB[(size_t)((j * TILE + n) >> G) * 16 + nib]);
  };

  std::vector<float> C((size_t)M * N, 0.0f);  // bf16-valued accumulator (mx_smem)

  for (int k_outer = 0; k_outer < TK; k_outer++) {
    const int group = (k_outer * DIM) / GROUP;
    for (int j = 0; j < TJ; j++) {
      for (int i = 0; i < TI; i++) {
        float Ct[32][32] = {};
        for (int kk = 0; kk < DIM; kk++) {
          float A_col[32], B_row[32];
          for (int r = 0; r < TILE; r++) A_col[r] = dec_a(i, r, k_outer, kk);
          for (int c = 0; c < TILE; c++) B_row[c] = dec_b(j, c, k_outer, kk);
          const int ae = ACC_E[kk], am = ACC_M[kk];
          for (int r = 0; r < TILE; r++)
            for (int c = 0; c < TILE; c++) {
              float p = mx_product_quantize_trunc(A_col[r] * B_row[c], PROD_E, PROD_M);
              float Cq = fp_quantize_rne(Ct[r][c], ae, am);
              float pq = fp_quantize_rne(p, ae, am);
              Ct[r][c] = fp_add_exact(Cq, pq, ae, am);
            }
        }
        // per-(row,col) e8m0 scale for this K group, then bf16 accumulate
        for (int r = 0; r < TILE; r++)
          for (int c = 0; c < TILE; c++) {
            const int m = i * TILE + r, n = j * TILE + c;
            int e = (int)SA[(size_t)group * M + m] + (int)SB[(size_t)group * N + n] - 127;
            uint8_t s_code = (e < 0) ? 0 : (e > 254 ? 254 : (uint8_t)e);
            float scaled = bf16_round(Ct[r][c] * fpe8m0_decode(s_code));
            float &acc = C[(size_t)m * N + n];
            acc = bf16_to_f32(f32_to_bf16_rne(bf16_accum_add(acc, scaled)));
          }
      }
    }
  }

  // Optional requant post-pass, selected by env (mirrors the accelerator's out_fmt):
  //   MX_OUT_FMT = 0 (fp8) | 1 (fp6, needs MX_LUT_C) | 2 (fp4);  3/unset = full bf16.
  //   MX_SCALES_OUT = path to write the per-row per-N-block e8m0 scale codes.
  //   MX_LUT_C = unpacked 16-code-per-LUT output LUT (fp6 only), MX_G = granularity.
  const char *ofmt_s = getenv("MX_OUT_FMT");
  const int out_fmt = ofmt_s ? atoi(ofmt_s) : 3;
  if (out_fmt != 3) {
    const int GROUP_OUT = 32, N_blocks = N / GROUP_OUT;
    const int log2_pmax = (out_fmt == 0) ? 8 : (out_fmt == 1 ? 4 : 2);
    std::vector<uint8_t> LC;
    int GC = 0;
    if (out_fmt == 1) {
      GC = getenv("MX_G") ? atoi(getenv("MX_G")) : 1;
      LC = slurp(getenv("MX_LUT_C"), (size_t)(((M - 1) >> GC) + 1) * 16);
    }
    std::vector<uint8_t> packed((size_t)M * N, 0), scales((size_t)M * N_blocks, 0);
    for (int m = 0; m < M; m++)
      for (int bi = 0; bi < N_blocks; bi++) {
        float vals[32], max_abs = 0.0f;
        for (int jj = 0; jj < GROUP_OUT; jj++) {
          vals[jj] = C[(size_t)m * N + bi * GROUP_OUT + jj];
          if (fabsf(vals[jj]) > max_abs) max_abs = fabsf(vals[jj]);
        }
        uint8_t sc = 0;
        if (max_abs != 0.0f) {
          int s2 = ((int)floorf(log2f(max_abs)) - log2_pmax) + 127;
          sc = (uint8_t)(s2 < 0 ? 0 : (s2 > 254 ? 254 : s2));
        }
        scales[(size_t)m * N_blocks + bi] = sc;
        float scale = fpe8m0_decode(sc);
        for (int jj = 0; jj < GROUP_OUT; jj++) {
          const int j = bi * GROUP_OUT + jj;
          if (out_fmt == 0) {
            packed[(size_t)m * N + j] = fp8_e4m3_to_code(vals[jj] / scale);
          } else {
            uint16_t sb = f32_to_bf16_rne(vals[jj] / scale);
            uint8_t code = (out_fmt == 1)
                ? (uint8_t)fp6e3m2_nearest_finder(bf16_bits_to_fp6_e3m2_code(sb), &LC[(size_t)(m >> GC) * 16])
                : bf16_bits_to_fp4_e2m1_code(sb);
            size_t bp = (size_t)(m >> 1) * N + j;   // HW-tiled: 2 codes per byte along M
            packed[bp] = (m & 1) ? ((packed[bp] & 0x0F) | ((code & 0xF) << 4))
                                 : ((packed[bp] & 0xF0) |  (code & 0xF));
          }
        }
      }
    FILE *pf = fopen(argv[8], "wb");
    fwrite(packed.data(), 1, (size_t)M * N, pf);
    fclose(pf);
    if (getenv("MX_SCALES_OUT")) {
      FILE *sf = fopen(getenv("MX_SCALES_OUT"), "wb");
      fwrite(scales.data(), 1, scales.size(), sf);
      fclose(sf);
    }
    return 0;
  }

  FILE *out = fopen(argv[8], "wb");
  if (!out) { fprintf(stderr, "mx_golden: cannot write %s\n", argv[8]); return 1; }
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      uint16_t v = f32_to_bf16_rne(C[(size_t)m * N + n]);
      fwrite(&v, sizeof(v), 1, out);
    }
  fclose(out);
  return 0;
}
