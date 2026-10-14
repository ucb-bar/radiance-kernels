// LayerNorm host: launch, wait, compare Y against the golden (both in GPU DRAM).
#include <nightly/host.h>
#include "layernorm_data.h"

int main() {
  nh_puts("layernorm: rows="); nh_putu(LN_ROWS); nh_puts(" dim="); nh_putu(LN_DIM);
  nh_puts(" clusters="); nh_putu(NIGHTLY_CLUSTERS); nh_puts("\n"); nh_flush();
  nightly_host_clear_verify(NIGHTLY_CLUSTERS);
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int ok = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint32_t cyc = nightly_host_report(NIGHTLY_CLUSTERS);
  const uint32_t n = LN_ROWS * LN_DIM;
  const int good = nightly_host_check("Y", NIGHTLY_CLUSTERS, LN_Y_ADDR, LN_G_ADDR, n, 100, 0);
  nh_puts("bytes_per_cycle_x100="); nh_putu(cyc ? (4ull * n * 100ull) / cyc : 0); nh_puts("\n");
  const int pass = ok && good;
  nh_puts(pass ? "\nPASS\n" : "\nFAIL\n"); nh_flush();
  exit(pass ? 0 : 1);
}
