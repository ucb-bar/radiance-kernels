/*
 * rad_host.h: Rocket (rv64) host utilities for kernels on the taped-out Radiance part.
 *
 * Include from host.cpp only.  Provides:
 *   - buffered console output          rad_puts / rad_putu / rad_putx / rad_flush
 *   - GPU DRAM access and timing       rad_gpu_ptr / rad_rdcycle / rad_wait_cycles
 *   - one-call kernel execution        rad_host_run
 *
 * Host constraints these helpers respect, and host code built on them must respect too:
 *   - NO FLOATING POINT.  The host Rocket has no FPU (fpu = None), so every FP instruction traps,
 *     including the fsd/fld spills gcc puts in a prologue when a function uses a double.  The
 *     symptom is a host that prints nothing at all.
 *   - printf over HTIF is unbuffered, about 1 s per character on the U250.  rad_puts() and friends
 *     format into a buffer; rad_flush() emits it with one write(1, ...).
 *   - With the tapeout epilogue, all_finished and the per-core finished bits never assert.  Poll
 *     the printBuf postbox instead, which rad_host_run() does.
 */

#ifndef __RAD_HOST_H__
#define __RAD_HOST_H__

#define RAD_TAPEOUT_HOST 1
#include <rad_tapeout.h>
#include <radiance.h>
#include <stdint.h>
#include <unistd.h>

#ifndef RAD_HOST_NUM_CORES
#define RAD_HOST_NUM_CORES 2   /* the taped-out part: one cluster of two cores */
#endif

/* ---- buffered console output ---------------------------------------------------------------- */
#ifndef RAD_HOST_OUTBUF_BYTES
#define RAD_HOST_OUTBUF_BYTES 4096
#endif
static char rad_outbuf[RAD_HOST_OUTBUF_BYTES];
static unsigned rad_outlen = 0;

static inline void rad_putc(char ch) {
  if (rad_outlen < sizeof(rad_outbuf)) rad_outbuf[rad_outlen++] = ch;
}
static inline void rad_puts(const char *s) { while (*s) rad_putc(*s++); }
static inline void rad_putu(uint64_t v) {
  char t[20]; int n = 0;
  do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v);
  while (n) rad_putc(t[--n]);
}
static inline void rad_putx(uint32_t v) {
  static const char hex[] = "0123456789abcdef";
  rad_puts("0x");
  for (int i = 7; i >= 0; i--) rad_putc(hex[(v >> (4 * i)) & 0xf]);
}
static inline void rad_flush(void) { write(1, rad_outbuf, rad_outlen); rad_outlen = 0; }

/* ---- GPU DRAM access and timing ------------------------------------------------------------- */
/* Host view of a GPU (device) address: RAD_HOST_GPU_DRAM_BASE | addr. */
static inline volatile uint32_t *rad_gpu_ptr(uint32_t device_addr) {
  return (volatile uint32_t *)rad_device_to_host_address(device_addr);
}
static inline uint64_t rad_rdcycle(void) {
  uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c;
}
static inline void rad_wait_cycles(uint64_t n) {
  const uint64_t end = rad_rdcycle() + n;
  while (rad_rdcycle() < end) asm volatile("" ::: "memory");
}
static inline uint32_t rad_host_postbox(uint32_t slot) {
  return (uint32_t)*(volatile uint64_t *)RAD_PB_HOST(0, slot);
}

/* ---- kernel execution ----------------------------------------------------------------------- */
/* Run the loaded GPU image once and stop it.  Returns 1 if every core started THIS run and
 * finished through the tapeout epilogue, else 0.  Prints one status line.
 *   1. read each core's start generation (printBuf RAD_PB_EPI_START + core);
 *   2. release the GPU with a soft-reset edge (assert, hold, release);
 *   3. wait until every core has advanced its generation by one, which rad_tapeout_begin() does
 *      on the GPU;
 *   4. wait for every core's epilogue DONE (rad_host_wait_epilogue);
 *   5. soft-reset the GPU, which stops the cores the epilogue left parked.
 * Step 3 is what keeps a DONE left in the postbox by an earlier run from passing as this run.
 * Write all inputs to GPU DRAM before calling this.
 *
 * On the U250 the first run of an image loaded after a different image does not start
 * (started=0); the next run of the same image does.  Retrying inside the same program does not
 * help (measured).  Likely cause, not established: the GPU executes the image already in DRAM at
 * SoC reset, before the host loads the new one, and a soft reset does not invalidate the
 * instructions it cached.  Evidence: between two runs of one image the start generation advances
 * once more than the launches alone explain. */
static inline int rad_host_run(void) {
  uint32_t expect[RAD_HOST_NUM_CORES];
  for (int c = 0; c < RAD_HOST_NUM_CORES; c++)
    expect[c] = RAD_EPI_NEXT_GEN(rad_host_postbox(RAD_PB_EPI_START + c));

  asm volatile("fence" ::: "memory");
  *(volatile uint32_t *)RAD_HOST_GPU_RESET = 1u;
  rad_wait_cycles(200000);
  *(volatile uint32_t *)RAD_HOST_GPU_RESET = 0u;
  const uint64_t t_launch = rad_rdcycle();

  int started = 0;
  for (uint64_t i = 0; i < 2000000ull && started < RAD_HOST_NUM_CORES; i++) {
    started = 0;
    for (int c = 0; c < RAD_HOST_NUM_CORES; c++)
      started += rad_host_postbox(RAD_PB_EPI_START + c) == expect[c];
  }
  int done = 0;
  if (started == RAD_HOST_NUM_CORES) done = rad_host_wait_epilogue(RAD_HOST_NUM_CORES, 20000000ull);
  const uint64_t t_done = rad_rdcycle();
  rad_host_gpu_soft_reset();   /* stop the GPU whatever state it is in */

  rad_puts("rad_host_run: started="); rad_putu(started); rad_puts("/"); rad_putu(RAD_HOST_NUM_CORES);
  rad_puts(" done="); rad_putu(done); rad_puts("/"); rad_putu(RAD_HOST_NUM_CORES);
  for (int c = 0; c < RAD_HOST_NUM_CORES; c++) {
    rad_puts(" gen"); rad_putu(c); rad_puts("="); rad_putx(expect[c]);
    rad_puts(" trace"); rad_putu(c); rad_puts("="); rad_putx(rad_host_postbox(RAD_PB_EPI_TRACE + c));
  }
  rad_puts(" host_cycles="); rad_putu(t_done - t_launch); rad_puts("\n");
  rad_flush();
  return done == RAD_HOST_NUM_CORES;
}

#endif /* __RAD_HOST_H__ */
