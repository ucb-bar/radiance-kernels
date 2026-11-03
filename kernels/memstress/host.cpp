// memstress host: read the per-warp records the kernel left in GMEM and print them, one line each:
//   AGG <core> <warps> <sum_store> <sum_flush> <sum_load> <max_store> <max_load> <err>
// then DONE <n_warps>, and exit(0).  err nonzero means some lane read back wrong data.
//
// This runs only on the branch that has +gpu_finish_keeps_sim=1: the taped-out harness stops the
// simulation 1k cycles after the last warp dies, long before an HTIF write completes, so a run there
// yields only its own end time.
//
// The whole report is assembled in one buffer and written with a single HTIF write: an HTIF call
// costs thousands of target cycles and one write per field ran past the simulation's cycle budget.
#include <stdint.h>
#include <unistd.h>
#include <radiance.h>
extern "C" void exit(int) __attribute__((noreturn));
#define MS_REC_BASE 0x1F000000u
#define MS_ERR_BASE (MS_REC_BASE + 0x1000u)
#define MS_DONE_BASE (MS_REC_BASE + 0x2000u)
#define MS_DONE_MAGIC 0xD09E0000u
#define MS_MAX_REC 32
#ifndef MS_CORES
#define MS_CORES 0xFu
#endif
#ifndef MS_NWARPS
#define MS_NWARPS 8
#endif
// Give up polling at a CYCLE deadline, not a poll count: each poll reads GPU DRAM through the host
// alias and costs hundreds of cycles, so a count is impossible to size against the simulation's
// max-cycles.  Whatever has been published by the deadline is still printed.
#ifndef MS_DEADLINE
#define MS_DEADLINE 1050000ull
#endif

static inline uint64_t rdcycle64(void) { uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c; }

static char buf[4096];
static unsigned blen;

static void put(const char *s) { while (*s && blen < sizeof(buf) - 1) buf[blen++] = *s++; }
static void putnum(uint64_t v, char tail) {
  char t[24];
  int n = 0;
  if (!v) t[n++] = '0';
  while (v) { t[n++] = (char)('0' + v % 10); v /= 10; }
  while (n && blen < sizeof(buf) - 1) buf[blen++] = t[--n];
  if (blen < sizeof(buf) - 1) buf[blen++] = tail;
}

int main() {
  { const char m[] = "host: up\n"; write(1, m, sizeof(m) - 1); }
  volatile uint32_t *rec = (volatile uint32_t *)(RAD_HOST_GPU_DRAM_BASE | (uint64_t)MS_REC_BASE);
  volatile uint32_t *errs = (volatile uint32_t *)(RAD_HOST_GPU_DRAM_BASE | (uint64_t)MS_ERR_BASE);

  // Wait for the GPU to retire, then let the finish-triggered flush drain before reading.
  uint64_t polls = 0;
  uint32_t finished = 0;
  while (!finished && rdcycle64() < MS_DEADLINE) { finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED); polls++; }
  { const uint64_t u = rdcycle64() + 20000ull; while (rdcycle64() < u) asm volatile("" ::: "memory"); }
  asm volatile("fence" ::: "memory");
  put("host: polls=");
  putnum(polls, ' ');
  put("finished=");
  putnum(finished, '\n');
  write(1, buf, blen);   // flush before touching GPU DRAM, so a stalled read still leaves evidence
  blen = 0;

  // Print per-core aggregates, not per-warp records.  Every byte of an HTIF write costs hundreds of
  // target cycles, so a full per-warp dump ran past the simulation's cycle budget on most
  // configurations.  Per core: how many warps reported, the sum over warps of each phase's cycles
  // (the report divides to get the mean), the largest store and load phase seen, and whether any
  // lane read wrong data.
  unsigned n = 0;
  for (unsigned g = 0; g < 8; g++) {   // global core g = cluster*2 + core: up to 4 clusters
    if (((MS_CORES >> g) & 1u) == 0u) continue;
    unsigned warps = 0;
    uint32_t sum_store = 0, sum_flush = 0, sum_load = 0, max_store = 0, max_load = 0, err = 0;
    for (unsigned w = 0; w < MS_NWARPS; w++) {
      const unsigned r = g * 8u + w;
      volatile uint32_t *p = rec + r * 16;
      if (p[0] != (0xC0DE0000u | (g << 4) | w)) continue;
      const uint32_t st = p[2] - p[1], fl = p[3] - p[2], ld = p[5] - p[4];
      sum_store += st;
      sum_flush += fl;
      sum_load += ld;
      if (st > max_store) max_store = st;
      if (ld > max_load) max_load = ld;
      err |= errs[r];
      warps++;
      n++;
    }
    if (!warps) continue;
    put("AGG ");
    putnum(g, ' ');
    putnum(warps, ' ');
    putnum(sum_store, ' ');
    putnum(sum_flush, ' ');
    putnum(sum_load, ' ');
    putnum(max_store, ' ');
    putnum(max_load, ' ');
    putnum(err, '\n');
  }
  put("DONE ");
  putnum(n, '\n');
  write(1, buf, blen);
  exit(finished ? 0 : 30000);
}
