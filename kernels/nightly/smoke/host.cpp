#include <nightly/host.h>
#include "smoke.h"
#define A_ADDR 0x20000000u
#define B_ADDR 0x20100000u
#define C_ADDR 0x20200000u
int main() {
  nh_puts("smoke: host writes A,B; clusters="); nh_putu(NIGHTLY_CLUSTERS); nh_puts("\n"); nh_flush();
  volatile uint32_t *A = nightly_gpu_ptr(A_ADDR), *B = nightly_gpu_ptr(B_ADDR), *C = nightly_gpu_ptr(C_ADDR);
  for (uint32_t i = 0; i < SMOKE_N; i++) { A[i] = smoke_a(i); B[i] = smoke_b(i); C[i] = 0xDEADBEEFu; }
  const uint64_t t0 = nightly_rdcycle();
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int ok = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint64_t t1 = nightly_rdcycle();
  nh_puts("wait ok="); nh_putu(ok); nh_puts(" host_cycles="); nh_putu(t1 - t0); nh_puts("\n");
  nightly_host_report(NIGHTLY_CLUSTERS);
  uint32_t bad = 0, poison = 0, first = 0;
  for (uint32_t i = 0; i < SMOKE_N; i++) {
    const uint32_t g = C[i];
    if (g != smoke_a(i) + smoke_b(i)) { if (!bad) first = i; bad++; poison += g == 0xDEADBEEFu; }
  }
  nh_puts("check: bad="); nh_putu(bad); nh_puts(" poison="); nh_putu(poison);
  nh_puts(" first="); nh_putu(first); nh_puts(bad ? "\nFAIL\n" : "\nPASS\n"); nh_flush();
  exit(ok && !bad ? 0 : 1);
}
