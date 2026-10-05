/*
 * nightly/host.h: Rocket (rv64) host side of the nightly launch protocol (see nightly/device.h).
 *
 * Include from host.cpp only.  The host Rocket has no FPU: keep host code integer-only (a double
 * in main() makes gcc spill fs0/fs1 and the host traps before its first print).
 *
 *   nightly_host_launch(n_clusters)  clear the END slots, write LAUNCH to every cluster
 *   nightly_host_wait(n_clusters)    wait for every core's END stamp; 1 = done, 0 = timeout
 *   nightly_host_report(...)         print per-cluster START/END and the kernel cycle count
 *   nightly_gpu_u16(dev_addr)        host pointer to GPU DRAM (device address | 0x1_0000_0000)
 *
 * Console output is buffered (nh_puts / nh_putu / nh_putx / nh_flush): HTIF output is slow.
 */
#ifndef NIGHTLY_HOST_H
#define NIGHTLY_HOST_H

#include <stdint.h>
#include <unistd.h>
#include <nightly/pb.h>
#include <nightly/verify.h>

extern "C" void exit(int) __attribute__((noreturn));

#define NIGHTLY_HOST_GPU_DRAM_BASE  0x100000000ull
#define NIGHTLY_HOST_ALL_FINISHED   0x41000008ull
#ifndef NIGHTLY_CORES_PER_CLUSTER
#define NIGHTLY_CORES_PER_CLUSTER 2
#endif

/* ---- buffered console ------------------------------------------------------------------- */
#ifndef NIGHTLY_HOST_OUTBUF
#define NIGHTLY_HOST_OUTBUF 8192
#endif
static char nh_buf[NIGHTLY_HOST_OUTBUF];
static unsigned nh_len = 0;
static inline void nh_flush(void) { if (nh_len) write(1, nh_buf, nh_len); nh_len = 0; }
static inline void nh_putc(char c) { if (nh_len == sizeof(nh_buf)) nh_flush(); nh_buf[nh_len++] = c; }
static inline void nh_puts(const char *s) { while (*s) nh_putc(*s++); }
static inline void nh_putu(uint64_t v) {
  char t[24]; int n = 0;
  do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v);
  while (n) nh_putc(t[--n]);
}
static inline void nh_puti(int64_t v) { if (v < 0) { nh_putc('-'); v = -v; } nh_putu((uint64_t)v); }
static inline void nh_putx(uint64_t v, int digits = 8) {
  static const char hex[] = "0123456789abcdef";
  nh_puts("0x");
  for (int i = digits - 1; i >= 0; i--) nh_putc(hex[(v >> (4 * i)) & 0xf]);
}
/* fixed point: v / 10^frac with `frac` digits */
static inline void nh_putfix(uint64_t v, int frac) {
  uint64_t p = 1; for (int i = 0; i < frac; i++) p *= 10;
  nh_putu(v / p); nh_putc('.');
  uint64_t r = v % p;
  for (int i = frac - 1; i >= 0; i--) { uint64_t q = 1; for (int j = 0; j < i; j++) q *= 10; nh_putc((char)('0' + (r / q) % 10)); }
}

/* ---- GPU memory and printBuf ------------------------------------------------------------ */
template <typename T = uint32_t>
static inline volatile T *nightly_gpu_ptr(uint32_t dev_addr) {
  return (volatile T *)(NIGHTLY_HOST_GPU_DRAM_BASE | (uint64_t)dev_addr);
}
static inline volatile uint64_t *nightly_pb(uint32_t cluster, uint32_t slot) {
  return (volatile uint64_t *)(NIGHTLY_PB_HOST_BASE(cluster) + 8ull * slot);
}
static inline uint32_t nightly_pb_read(uint32_t cluster, uint32_t slot) {
  return (uint32_t)*nightly_pb(cluster, slot);
}
static inline void nightly_pb_write(uint32_t cluster, uint32_t slot, uint32_t v) {
  *nightly_pb(cluster, slot) = ((uint64_t)v << 32) | v;
}
static inline uint64_t nightly_rdcycle(void) { uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c; }

/* ---- launch / wait ---------------------------------------------------------------------- */
static inline void nightly_host_launch(uint32_t n_clusters, uint32_t arg = 0) {
  for (uint32_t cl = 0; cl < n_clusters; cl++) {
    for (uint32_t c = 0; c < NIGHTLY_CORES_PER_CLUSTER; c++) {
      nightly_pb_write(cl, NIGHTLY_PB_END + c, 0);
      nightly_pb_write(cl, NIGHTLY_PB_START + c, 0);
    }
    nightly_pb_write(cl, NIGHTLY_PB_STATUS, 0);
    nightly_pb_write(cl, NIGHTLY_PB_ARG, arg);
  }
  asm volatile("fence" ::: "memory");
  for (uint32_t cl = 0; cl < n_clusters; cl++) nightly_pb_write(cl, NIGHTLY_PB_LAUNCH, NIGHTLY_LAUNCH_MAGIC);
  asm volatile("fence" ::: "memory");
}

static inline int nightly_host_wait(uint32_t n_clusters, uint64_t max_polls = 50000000ull) {
  uint32_t last_hb = 0;
  for (uint64_t i = 0; i < max_polls; i++) {
    if ((i & 1023u) == 0) {   /* heartbeat: print cluster 0's progress value when it changes */
      const uint32_t hb = nightly_pb_read(0, NIGHTLY_PB_HEARTBEAT);
      if (hb != last_hb) {
        last_hb = hb;
        nh_puts("heartbeat "); nh_putx(hb, 8); nh_puts(" host_cycle "); nh_putu(nightly_rdcycle()); nh_puts("\n");
        nh_flush();
      }
    }
    uint32_t done = 0;
    for (uint32_t cl = 0; cl < n_clusters; cl++)
      for (uint32_t c = 0; c < NIGHTLY_CORES_PER_CLUSTER; c++)
        done += nightly_pb_read(cl, NIGHTLY_PB_END + c) != 0;
    if (done == n_clusters * NIGHTLY_CORES_PER_CLUSTER) return 1;
  }
  return 0;
}

/* Kernel cycles = latest END - earliest START over all cores (GPU mcycle, 500 MHz). */
static inline uint32_t nightly_host_report(uint32_t n_clusters) {
  uint32_t t0 = 0xFFFFFFFFu, t1 = 0;
  for (uint32_t cl = 0; cl < n_clusters; cl++) {
    for (uint32_t c = 0; c < NIGHTLY_CORES_PER_CLUSTER; c++) {
      const uint32_t s = nightly_pb_read(cl, NIGHTLY_PB_START + c);
      const uint32_t e = nightly_pb_read(cl, NIGHTLY_PB_END + c);
      if (s < t0) t0 = s;
      if (e > t1) t1 = e;
      nh_puts("cluster "); nh_putu(cl); nh_puts(" core "); nh_putu(c);
      nh_puts(": start="); nh_putu(s); nh_puts(" end="); nh_putu(e);
      nh_puts(" cycles="); nh_putu(e - s); nh_puts("\n");
    }
  }
  const uint32_t cyc = t1 - t0;
  nh_puts("kernel_cycles="); nh_putu(cyc); nh_puts("\n");
  nh_flush();
  return cyc;
}

static inline void nightly_host_phases(uint32_t cluster, uint32_t n) {
  nh_puts("phases cluster "); nh_putu(cluster); nh_puts(":");
  for (uint32_t i = 0; i < n && i < 8; i++) { nh_putc(' '); nh_putu(nightly_pb_read(cluster, NIGHTLY_PB_PHASE + i)); }
  nh_puts("\n");
}

/* utilization in units of 0.01 % = useful / (peak * cycles) */
static inline uint64_t nightly_util_bp(uint64_t useful_ops, uint64_t peak_per_cycle, uint64_t cycles) {
  if (!cycles) return 0;
  return (useful_ops * 10000ull) / (peak_per_cycle * cycles);
}

/* ---- bf16 comparison (integer-only) ----------------------------------------------------- */
/* bf16 -> signed fixed point value * 2^scale_log2 as int64 (truncating), for error sums. */
static inline int64_t nh_bf16_to_fix(uint16_t b, int scale_log2) {
  const int sign = b >> 15, e = (b >> 7) & 0xFF;
  if (e == 0) return 0;           /* flush subnormals */
  if (e == 0xFF) return sign ? -(1ll << 62) : (1ll << 62);
  const int64_t mant = 0x80 | (b & 0x7F);   /* 1.m * 2^7 */
  const int sh = (e - 127) - 7 + scale_log2;
  int64_t v;
  if (sh >= 0) v = sh > 40 ? (1ll << 50) : (mant << sh);
  else v = sh < -40 ? 0 : (mant >> (-sh));
  return sign ? -v : v;
}

/* Relative Frobenius error sqrt(sum (g-r)^2 / sum r^2) in units of 1e-4 (i.e. 0.01 %),
 * computed with fixed point at 2^scale_log2 resolution.  Values must stay below ~2^20 / 2^scale.
 * Returns also the count of exact bf16 matches.                                              */
struct NhCmp { uint64_t err_bp; uint32_t exact; uint32_t n; uint32_t nonfinite; };
static inline uint64_t nh_isqrt(unsigned __int128 x) {
  if (x == 0) return 0;
  unsigned __int128 r = 0, bit = (unsigned __int128)1 << 126;
  while (bit > x) bit >>= 2;
  while (bit) {
    if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; } else r >>= 1;
    bit >>= 2;
  }
  return (uint64_t)r;
}
static inline NhCmp nh_compare_bf16(const volatile uint16_t *got, const uint16_t *ref, uint32_t n,
                                    int scale_log2 = 16) {
  unsigned __int128 se = 0, sr = 0;
  NhCmp c = {0, 0, n, 0};
  for (uint32_t i = 0; i < n; i++) {
    const uint16_t g = got[i], r = ref[i];
    if (g == r) c.exact++;
    if (((g >> 7) & 0xFF) == 0xFF) c.nonfinite++;
    const int64_t gv = nh_bf16_to_fix(g, scale_log2), rv = nh_bf16_to_fix(r, scale_log2);
    const int64_t d = gv - rv;
    se += (unsigned __int128)((__int128)d * d);
    sr += (unsigned __int128)((__int128)rv * rv);
  }
  if (sr == 0) { c.err_bp = se ? 1000000 : 0; return c; }
  /* sqrt(se/sr) * 1e4 = sqrt(se * 1e8 / sr) */
  c.err_bp = nh_isqrt((se * 100000000ull) / sr);
  return c;
}
static inline void nh_print_cmp(const char *name, const NhCmp &c) {
  nh_puts(name); nh_puts(": frob_rel_err="); nh_putfix(c.err_bp, 2); nh_puts("% exact=");
  nh_putu(c.exact); nh_puts("/"); nh_putu(c.n); nh_puts(" nonfinite="); nh_putu(c.nonfinite); nh_puts("\n");
}


/* ---- GPU-side verification results (nightly/verify.h) ---------------------------------- */
struct NhVerify { uint32_t done, n, exact, ulp1, bad, poison, nonfin, err_bp; };
static inline NhVerify nightly_host_read_verify(uint32_t cl) {
  NhVerify v;
  v.done = nightly_pb_read(cl, NV_SLOT + 0) == NV_DONE;
  v.n = nightly_pb_read(cl, NV_SLOT + 1);
  v.exact = nightly_pb_read(cl, NV_SLOT + 2);
  v.ulp1 = nightly_pb_read(cl, NV_SLOT + 3);
  v.bad = nightly_pb_read(cl, NV_SLOT + 4);
  v.poison = nightly_pb_read(cl, NV_SLOT + 5);
  v.nonfin = nightly_pb_read(cl, NV_SLOT + 6);
  v.err_bp = nightly_pb_read(cl, NV_SLOT + 7);
  return v;
}
/* clear the verify slots (call before launch) and wait for cluster `cl` to post them */
static inline void nightly_host_clear_verify(uint32_t n_clusters) {
  for (uint32_t cl = 0; cl < n_clusters; cl++) { nightly_pb_write(cl, NV_SLOT, 0); nightly_pb_write(cl, NV_GO, 0); }
}
static inline int nightly_host_wait_verify(uint32_t cl, uint64_t max_polls = 50000000ull) {
  for (uint64_t i = 0; i < max_polls; i++)
    if (nightly_pb_read(cl, NV_SLOT) == NV_DONE) return 1;
  return 0;
}
static inline void nightly_host_print_verify(const char *name, uint32_t cl, const NhVerify &v) {
  nh_puts(name); nh_puts(" (GPU check, cluster "); nh_putu(cl); nh_puts("): n="); nh_putu(v.n);
  nh_puts(" exact="); nh_putu(v.exact); nh_puts(" within1step="); nh_putu(v.ulp1);
  nh_puts(" bad="); nh_putu(v.bad); nh_puts(" poison="); nh_putu(v.poison);
  nh_puts(" nonfinite="); nh_putu(v.nonfin); nh_puts(" frob_rel_err="); nh_putfix(v.err_bp, 2);
  nh_puts("%\n");
}
/* Standard end-of-run check: GPU verify of every cluster (all must agree), plus a host spot
 * check of `samples` elements.  Returns 1 when the output passes: no poison, no non-finite,
 * relative Frobenius error <= max_err_bp (0.01 % units) and every `bad` count <= max_bad. */
static inline uint32_t nightly_host_spot_check(uint32_t out_dev, uint32_t gold_dev, uint32_t n,
                                               uint32_t samples);
static inline int nightly_host_check(const char *name, uint32_t n_clusters, uint32_t out_dev,
                                     uint32_t gold_dev, uint32_t n, uint32_t max_err_bp,
                                     uint32_t max_bad) {
  int pass = 1;
  NhVerify v0 = {};
  for (uint32_t cl = 0; cl < n_clusters; cl++) nightly_pb_write(cl, NV_GO, NV_GO_MAGIC);
  for (uint32_t cl = 0; cl < n_clusters; cl++) {
    if (!nightly_host_wait_verify(cl)) { nh_puts(name); nh_puts(": GPU verify timeout\n"); return 0; }
    const NhVerify v = nightly_host_read_verify(cl);
    nightly_host_print_verify(name, cl, v);
    if (cl == 0) v0 = v;
    else if (v.exact != v0.exact || v.bad != v0.bad || v.err_bp != v0.err_bp) {
      nh_puts("  clusters disagree\n"); pass = 0;
    }
  }
  const uint32_t samples = n < 128 ? n : 128;
  const uint32_t same = nightly_host_spot_check(out_dev, gold_dev, n, samples);
  nh_puts(name); nh_puts(" (host spot check): exact "); nh_putu(same); nh_puts("/"); nh_putu(samples); nh_puts("\n");
  if (v0.n != n || v0.poison || v0.nonfin || v0.err_bp > max_err_bp || v0.bad > max_bad) pass = 0;
  nh_flush();
  return pass;
}

/* Host spot check: `samples` evenly spaced elements, count exact matches (integer compare). */
static inline uint32_t nightly_host_spot_check(uint32_t out_dev, uint32_t gold_dev, uint32_t n,
                                               uint32_t samples) {
  const volatile uint16_t *o = nightly_gpu_ptr<uint16_t>(out_dev), *g = nightly_gpu_ptr<uint16_t>(gold_dev);
  uint32_t same = 0;
  const uint32_t step = n / samples ? n / samples : 1;
  for (uint32_t k = 0, i = 7 % step; k < samples && i < n; k++, i += step) same += o[i] == g[i];
  return same;
}

#endif
