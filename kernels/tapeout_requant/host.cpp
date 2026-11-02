// host.cpp: launches the requantizer probe and prints its postbox results (kernel.cpp).
#include <rad_host.h>
#include <stdint.h>

#include "rq.h"

static inline uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
  return x;
}
static inline uint32_t dma_word(uint32_t i) { return hash32(i * 4u + 11u); }

static void put(const char *name, uint32_t slot) {
  rad_puts(name); rad_puts("="); rad_putu(rad_host_postbox(slot)); rad_puts(" ");
}

int main() {
  rad_puts("tapeout_requant: blocks="); rad_putu(RQ_NB); rad_puts("\n");
  rad_flush();
  rad_host_gpu_soft_reset();
  volatile RQArgs *args = reinterpret_cast<volatile RQArgs *>(rad_gpu_ptr(RQ_ARGS));
  args->mode = RQ_MODE;
  if (RQ_MODE == RQ_MODE_DMA16 || RQ_MODE == RQ_MODE_DMA64) {
    volatile uint64_t *src = (volatile uint64_t *)rad_gpu_ptr(RQ_DMA_SRC);
    volatile uint64_t *dst = (volatile uint64_t *)rad_gpu_ptr(RQ_DMA_DST);
    for (uint32_t i = 0; i < RQ_DMA_BYTES / 8u; i++) {
      src[i] = dma_word(2u * i) | ((uint64_t)dma_word(2u * i + 1u) << 32);
      dst[i] = 0;
    }
  }
  const int ran = rad_host_run();
  put("status", RQ_PB_STATUS); put("t_feed", RQ_PB_T_FEED); put("t_done", RQ_PB_T_DONE);
  put("polls", RQ_PB_POLLS); rad_puts("\n");
  rad_puts("check: of "); rad_putu(RQ_NB * 32u); rad_puts(" codes, ");
  put("raw_vs_rne(rev)", RQ_PB_RAW_RNE); put("raw_vs_trunc", RQ_PB_RAW_TRUNC);
  put("raw_natural_vs_rne", RQ_PB_RAW_REV); put("fixed_vs_simt", RQ_PB_FIX_REF);
  put("fixed_far", RQ_PB_FIX_FAR); rad_puts("\n");
  rad_puts("first_bad="); rad_putu(rad_host_postbox(RQ_PB_FIRST_BAD));
  rad_puts(" got|model<<8|x<<16="); rad_putx(rad_host_postbox(RQ_PB_FIRST_GOT)); rad_puts("\n");
  rad_puts("out  :");
  for (uint32_t w = 0; w < 8; w++) { rad_puts(" "); rad_putx(rad_host_postbox(RQ_PB_DUMP + w)); }
  rad_puts("\nmodel:");
  for (uint32_t w = 0; w < 8; w++) { rad_puts(" "); rad_putx(rad_host_postbox(RQ_PB_DUMP + 8 + w)); }
  rad_puts("\n");
  volatile uint32_t *dump = rad_gpu_ptr(RQ_DUMP);
  for (uint32_t b = 0; b < RQ_DUMP_BLOCKS; b++) {
    rad_puts("D "); rad_putu(b); rad_puts(" src");
    for (uint32_t w = 0; w < 16; w++) { rad_puts(" "); rad_putx(dump[b * 16 + w]); }
    rad_puts(" out");
    for (uint32_t w = 0; w < 8; w++) { rad_puts(" "); rad_putx(dump[RQ_NB * 16 + b * 8 + w]); }
    rad_puts("\n");
    rad_flush();
  }
  if (RQ_MODE == RQ_MODE_GELU) {
    rad_puts("gelu cycles per element per warp x100 (6 warps at once): with fnexp.h+fdiv.h=");
    rad_putu(rad_host_postbox(RQ_PB_DUMP)); rad_puts(" without="); rad_putu(rad_host_postbox(RQ_PB_DUMP + 1)); rad_puts("\n");
  }
  if (RQ_MODE == RQ_MODE_ISSUE) {
    const char *names[11] = {"add", "fadd.s", "fmul.s", "fmul.h", "fsgnjx.s", "fmax.s", "lw.shared",
                             "fdiv.s", "fdiv.h", "fexp.h", "fmadd.s"};
    rad_puts("issue cycles per warp instruction x100:");
    for (uint32_t k = 0; k < 11; k++) { rad_puts(" "); rad_puts(names[k]); rad_puts("="); rad_putu(rad_host_postbox(RQ_PB_DUMP + k)); }
    rad_puts("\n");
  }
  const int pass = ran && rad_host_postbox(RQ_PB_STATUS) == 1u;
  rad_puts(pass ? "\nRAN\n" : "\nFAIL\n");
  rad_flush();
  return pass ? 0 : 1;
}
