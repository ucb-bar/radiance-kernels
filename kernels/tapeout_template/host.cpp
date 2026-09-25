// host.cpp: Rocket (rv64) side of the tapeout template.
//
//   1. print a hello line;
//   2. write A, B, the argument block, and a poisoned C to GPU DRAM;
//   3. rad_host_run(): launch, wait for the epilogue, soft-reset the GPU;
//   4. check C.
//
// The host has no FPU: keep this file integer-only (see lib/include/rad_host.h).

#include <rad_host.h>
#include <stdint.h>

#include "template.h"

static uint32_t a_value(uint32_t i) { return i * 3u; }
static uint32_t b_value(uint32_t i) { return 0x10000u + i; }

static void write_inputs(void) {
  volatile TTArgs *args = reinterpret_cast<volatile TTArgs *>(rad_gpu_ptr(TT_ARGS_ADDR));
  args->a = TT_A_ADDR;
  args->b = TT_B_ADDR;
  args->c = TT_C_ADDR;
  args->n = TT_N;
  volatile uint32_t *A = rad_gpu_ptr(TT_A_ADDR);
  volatile uint32_t *B = rad_gpu_ptr(TT_B_ADDR);
  volatile uint32_t *C = rad_gpu_ptr(TT_C_ADDR);
  for (uint32_t i = 0; i < TT_N; i++) {
    A[i] = a_value(i);
    B[i] = b_value(i);
    C[i] = TT_POISON;   // so "never written" is distinguishable from "written wrong"
  }
}

int main() {
  rad_puts("Hello from Rocket: tapeout template, vecadd of "); rad_putu(TT_N); rad_puts(" words\n");
  rad_flush();

  // On the U250 the GPU starts executing whatever image is in DRAM at SoC reset, before this
  // program runs.  Hold it in reset so it cannot write into our buffers while we set them up.
  rad_host_gpu_soft_reset();
  write_inputs();
  const int ran = rad_host_run();

  volatile uint32_t *C = rad_gpu_ptr(TT_C_ADDR);
  uint32_t wrong = 0, poison = 0, first_bad = 0;
  for (uint32_t i = 0; i < TT_N; i++) {
    const uint32_t got = C[i];
    if (got == a_value(i) + b_value(i)) continue;
    if (wrong == 0) first_bad = i;
    wrong++;
    poison += got == TT_POISON;
  }
  rad_puts("check: wrong="); rad_putu(wrong); rad_puts("/"); rad_putu(TT_N);
  rad_puts(" still_poison="); rad_putu(poison);
  if (wrong) {
    rad_puts(" first_bad=C["); rad_putu(first_bad); rad_puts("]="); rad_putx(C[first_bad]);
    rad_puts(" expected "); rad_putx(a_value(first_bad) + b_value(first_bad));
  }
  const int pass = ran && wrong == 0;
  rad_puts(pass ? "\nPASS\n" : "\nFAIL\n");
  rad_flush();
  return pass ? 0 : 1;
}
