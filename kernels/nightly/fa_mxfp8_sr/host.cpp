// MXFP8 flash-attention host: launch, wait, check O against the kernel-order golden and the fp32
// reference, report mesh utilization (useful QK^T + PV MACs over peak 256 MAC/cycle/cluster).
#include <nightly/host.h>
#include "fa_data.h"

int main() {
  nh_puts("fa_mxfp8: H="); nh_putu(FA_H); nh_puts(" SQ="); nh_putu(FA_SQ); nh_puts(" SK="); nh_putu(FA_SK);
  nh_puts(" D="); nh_putu(FA_D); nh_puts(" BK="); nh_putu(FA_BK);
  nh_puts(" clusters="); nh_putu(NIGHTLY_CLUSTERS); nh_puts("\n"); nh_flush();
  nightly_host_clear_verify(NIGHTLY_CLUSTERS);
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int ok = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint32_t cyc = nightly_host_report(NIGHTLY_CLUSTERS);
  nightly_host_phases(0, 8);
  const uint32_t p0 = nightly_pb_read(0, NIGHTLY_PB_PHASE + 0), p1 = nightly_pb_read(0, NIGHTLY_PB_PHASE + 1);
  const uint64_t macs = 2ull * FA_H * FA_SQ * FA_SK * FA_D;
  nh_puts("mesh_util_kernel="); nh_putfix(nightly_util_bp(macs, 256ull * NIGHTLY_CLUSTERS, cyc), 2);
  nh_puts("% mesh_util_c0_loop="); nh_putfix(nightly_util_bp(macs / NIGHTLY_CLUSTERS, 256, p1 - p0), 2);
  nh_puts("%\n");
  const uint32_t n = FA_H * FA_SQ * FA_D;
  const int good = nightly_host_check("O vs kernel-order golden", NIGHTLY_CLUSTERS, FA_O_ADDR, FA_G_ADDR, n, 200, n);
  // accuracy context: output vs the fp32 reference, on a sample (host-side, integer-only)
  const uint32_t ns = 256, step = n / ns;
  const volatile uint16_t *O = nightly_gpu_ptr<uint16_t>(FA_O_ADDR);
  static uint16_t so[256], sr[256], sg[256];
  for (uint32_t k = 0; k < ns; k++) {
    so[k] = O[k * step]; sr[k] = nightly_gpu_ptr<uint16_t>(FA_R_ADDR)[k * step];
    sg[k] = nightly_gpu_ptr<uint16_t>(FA_G_ADDR)[k * step];
  }
  nh_print_cmp("O vs fp32 reference (sample)", nh_compare_bf16(so, sr, ns, 20));
  nh_print_cmp("golden vs fp32 reference (sample)", nh_compare_bf16(sg, sr, ns, 20));
#if defined(FA_DUMP) && FA_DUMP
  {   // diagnosis: per output row / per 16-column group, elements that differ from the golden
    const volatile uint16_t *G = nightly_gpu_ptr<uint16_t>(FA_G_ADDR);
    static uint32_t colbad[FA_D / 16];
    nh_puts("rowbad:");
    for (uint32_t r = 0; r < FA_H * FA_SQ; r++) {
      uint32_t nb = 0;
      for (uint32_t c = 0; c < FA_D; c++) {
        const uint16_t o = O[r * FA_D + c], g = G[r * FA_D + c];
        const int d = (int)(o & 0x7FFF) - (int)(g & 0x7FFF);
        if (o != g && !(d == 1 || d == -1)) { nb++; colbad[c / 16]++; }
      }
      nh_putc(' '); nh_putu(nb);
    }
    nh_puts("\ncolbad:");
    for (uint32_t k = 0; k < FA_D / 16; k++) { nh_putc(' '); nh_putu(colbad[k]); }
    nh_puts("\nrow0:");
    for (uint32_t c = 0; c < 8; c++) { nh_putc(' '); nh_putx(O[c], 4); nh_putc('/'); nh_putx(G[c], 4); }
    nh_puts("\n"); nh_flush();
  }
#endif
  const int pass = ok && good;   // < 2 % Frobenius vs the kernel-order golden
  nh_puts(pass ? "PASS\n" : "FAIL\n"); nh_flush();
  exit(pass ? 0 : 1);
}
