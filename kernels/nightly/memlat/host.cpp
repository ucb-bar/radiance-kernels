// memlat host: launch, wait, print the chaser's latencies and the loaders' bandwidth per cluster and level.
#include <nightly/host.h>
#include "memlat_data.h"

static const char *const LEVEL_NAME[MEMLAT_LEVELS] = {"idle", "muon", "dma", "muon+dma"};
static const uint32_t CHECK[2] = {MEMLAT_CHECK_C0, MEMLAT_CHECK_C1};

// cycles per load (x100) of n loads
static void put_lat(const char *name, uint32_t cyc, uint32_t n) {
  nh_puts(name); nh_putfix((uint64_t)cyc * 100 / n, 2);
}

int main() {
  nh_puts("memlat: N="); nh_putu(MEMLAT_N); nh_puts(" warm="); nh_putu(MEMLAT_WARM);
  nh_puts(" clusters="); nh_putu(NIGHTLY_CLUSTERS); nh_puts("\n"); nh_flush();
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int ok = nightly_host_wait(NIGHTLY_CLUSTERS);
  nightly_host_report(NIGHTLY_CLUSTERS);
  int good = ok;
  for (uint32_t cl = 0; cl < NIGHTLY_CLUSTERS; cl++) {
    nh_puts("memlat cluster "); nh_putu(cl);
    put_lat(": small(cached) ", nightly_pb_read(cl, 37), MEMLAT_SMALL);
    put_lat(" warm(seq again) ", nightly_pb_read(cl, 36), MEMLAT_N);
    const uint32_t chk = nightly_pb_read(cl, 38);
    nh_puts(" check "); nh_puts(chk == CHECK[cl] ? "ok" : "BAD"); nh_puts("\n");
    good &= chk == CHECK[cl];
    for (uint32_t lv = 0; lv < MEMLAT_LEVELS; lv++) {
      const uint32_t s = 16 + 5 * lv;
      const uint32_t seq = nightly_pb_read(cl, s), rnd = nightly_pb_read(cl, s + 1);
      const uint32_t loops = nightly_pb_read(cl, s + 2), dcyc = nightly_pb_read(cl, s + 3);
      const uint32_t lines = nightly_pb_read(cl, s + 4), lcyc = nightly_pb_read(cl, 40 + lv);
      nh_puts("memlat cluster "); nh_putu(cl); nh_puts(" level "); nh_putu(lv); nh_putc(' '); nh_puts(LEVEL_NAME[lv]);
      put_lat(": seq ", seq, MEMLAT_N);
      put_lat(" rand ", rnd, MEMLAT_N);
      nh_puts(" cyc/load  dma_B/cyc "); nh_putfix(dcyc ? (uint64_t)loops * 32768 * 100 / dcyc : 0, 2);
      nh_puts(" muon_B/cyc "); nh_putfix(lcyc ? (uint64_t)lines * 64 * 100 / lcyc : 0, 2);
      nh_puts(" level_cycles "); nh_putu(lcyc); nh_puts("\n");
    }
  }
  // bandwidth phase (kernel.cpp): window 0 Muon + DMA B only, window 1 Muon + DMA A and B
  static const uint32_t DMA_LOOP_BYTES[2] = {32768u, 65536u};
  uint64_t all[2] = {0, 0};
  for (uint32_t cl = 0; cl < NIGHTLY_CLUSTERS; cl++) {
    for (uint32_t w = 0; w < 2; w++) {
      const uint32_t s = 52 + 4 * w;
      const uint32_t tcyc = nightly_pb_read(cl, s), lines = nightly_pb_read(cl, s + 1);
      const uint32_t loops = nightly_pb_read(cl, s + 2), dcyc = nightly_pb_read(cl, s + 3);
      const uint64_t mbw = tcyc ? (uint64_t)lines * 64 * 100 / tcyc : 0;
      const uint64_t dbw = dcyc ? (uint64_t)loops * DMA_LOOP_BYTES[w] * 100 / dcyc : 0;
      all[w] += mbw + dbw;
      nh_puts("memlat cluster "); nh_putu(cl); nh_puts(w ? " stream dma_ab" : " stream dma_b ");
      nh_puts(": muon_B/cyc "); nh_putfix(mbw, 2); nh_puts(" dma_B/cyc "); nh_putfix(dbw, 2);
      nh_puts(" total_B/cyc "); nh_putfix(mbw + dbw, 2);
      nh_puts(" cycles "); nh_putu(tcyc); nh_puts(" lines "); nh_putu(lines); nh_puts(" loops "); nh_putu(loops);
      nh_puts(" dma_cycles "); nh_putu(dcyc); nh_puts("\n");
    }
  }
  nh_puts("memlat stream all clusters: dma_b total_B/cyc "); nh_putfix(all[0], 2);
  nh_puts(" dma_ab total_B/cyc "); nh_putfix(all[1], 2); nh_puts("\n");
  nh_puts(good ? "PASS\n" : "FAIL\n"); nh_flush();
  exit(good ? 0 : 1);
}
