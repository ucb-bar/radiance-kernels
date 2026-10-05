// MX GEMM host: launch, wait, compare C against the golden, report mesh utilization.
#include <nightly/host.h>
#include "gemm_data.h"

int main() {
  nh_puts(GEMM_FP4 ? "gemm_mx fp4" : "gemm_mx fp8");
  nh_puts(": M="); nh_putu(GEMM_M); nh_puts(" N="); nh_putu(GEMM_N); nh_puts(" K="); nh_putu(GEMM_K);
  nh_puts(" tile="); nh_putu(GEMM_TM); nh_putc('x'); nh_putu(GEMM_TN); nh_putc('x'); nh_putu(GEMM_TK);
  nh_puts(" clusters="); nh_putu(NIGHTLY_CLUSTERS); nh_puts("\n"); nh_flush();
  nightly_host_clear_verify(NIGHTLY_CLUSTERS);
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int ok = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint32_t cyc = nightly_host_report(NIGHTLY_CLUSTERS);
  nightly_host_phases(0, 2);
  const uint32_t mesh0 = nightly_pb_read(0, NIGHTLY_PB_PHASE + 0), mesh1 = nightly_pb_read(0, NIGHTLY_PB_PHASE + 1);
  const uint64_t macs = (uint64_t)GEMM_M * GEMM_N * GEMM_K;
  const uint64_t peak = GEMM_FP4 ? 1024 : 256;   // MACs per cycle per cluster
  nh_puts("mesh_util_kernel="); nh_putfix(nightly_util_bp(macs, peak * NIGHTLY_CLUSTERS, cyc), 2);
  nh_puts("% mesh_util_c0_first_to_last_compute=");
  nh_putfix(nightly_util_bp(macs / NIGHTLY_CLUSTERS, peak, mesh1 - mesh0), 2); nh_puts("%\n");
  const int good = nightly_host_check("C", NIGHTLY_CLUSTERS, GEMM_C_ADDR, GEMM_G_ADDR, GEMM_M * GEMM_N, 100, GEMM_M * GEMM_N / 100);
#if defined(GEMM_DIAG) && GEMM_DIAG
  nh_puts("bad per 128x64 tile (row-major tiles):");
  for (uint32_t t = 0; t < 16; t++) { nh_putc(' '); nh_putu(nightly_pb_read(0, 32 + t)); }
  nh_puts("\n"); nh_flush();
#endif
#if defined(GEMM_DUMP) && GEMM_DUMP
  {   // diagnosis: rows 0 and 77, first 8 columns, of every tile of the first 2x2 tiles
    const volatile uint16_t *c = nightly_gpu_ptr<uint16_t>(GEMM_C_ADDR);
    for (uint32_t t = 0; t < 4; t++)
      for (uint32_t r = 0; r < 128; r += 77) {
        nh_puts("dump t="); nh_putu(t); nh_puts(" r="); nh_putu(r); nh_putc(':');
        const uint32_t row = (t / 2) * GEMM_TM + r, col = (t % 2) * GEMM_TN;
        for (uint32_t k = 0; k < 8; k++) { nh_putc(' '); nh_putx(c[row * GEMM_N + col + k], 4); }
        nh_puts("\n");
      }
    nh_flush();
  }
#endif
  const int pass = ok && good;   // < 1 % Frobenius, < 1 % of elements off by more than one step
  nh_puts(pass ? "PASS\n" : "FAIL\n"); nh_flush();
  exit(pass ? 0 : 1);
}
