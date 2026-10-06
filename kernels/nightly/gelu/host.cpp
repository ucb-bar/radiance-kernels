// GELU host: launch, wait, compare Y against the golden (both in GPU DRAM).
#include <nightly/host.h>
#include "gelu_data.h"

int main() {
  nh_puts("gelu: N="); nh_putu(GELU_N); nh_puts(" clusters="); nh_putu(NIGHTLY_CLUSTERS); nh_puts("\n"); nh_flush();
  nightly_host_clear_verify(NIGHTLY_CLUSTERS);
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int ok = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint32_t cyc = nightly_host_report(NIGHTLY_CLUSTERS);
  const int good = nightly_host_check("Y", NIGHTLY_CLUSTERS, GELU_Y_ADDR, GELU_G_ADDR, GELU_N, 100, 0);
  nh_puts("bytes_per_cycle_x100="); nh_putu(cyc ? (4ull * GELU_N * 100ull) / cyc : 0); nh_puts("\n");
  const int pass = ok && good;
  nh_puts(pass ? "\nPASS\n" : "\nFAIL\n"); nh_flush();
  exit(pass ? 0 : 1);
}
