#include <nightly/device.h>
#include "smoke.h"

// device addresses fixed so the host can find them without a symbol table
#define A_ADDR 0x20000000u
#define B_ADDR 0x20100000u
#define C_ADDR 0x20200000u

struct Args { const uint32_t *a, *b; uint32_t *c; uint32_t n; };
static Args args = {(const uint32_t *)A_ADDR, (const uint32_t *)B_ADDR, (uint32_t *)C_ADDR, SMOKE_N};

static void entry(void *raw, uint32_t tid, uint32_t tpb, uint32_t tb) {
  const Args *a = (const Args *)raw;
  const uint32_t gtid = tb * tpb + tid, gthreads = tpb * NIGHTLY_CLUSTERS;
  for (uint32_t i = gtid; i < a->n; i += gthreads) a->c[i] = a->a[i] + a->b[i];
  nightly_kernel_end(tid, tpb);
}

int main() {
  return nightly_main(entry, &args);
}
