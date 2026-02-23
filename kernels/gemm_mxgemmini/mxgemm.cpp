#include <stdint.h>
#include <radiance.h>
#include <vx_intrinsics.h>
#include <mu_intrinsics.h>

#include "include/gemmini.h"
#include "include/matmul_data.h"
#include "mxgemmini_mmio.h"

// TODO: cleanup
typedef uint8_t elem_t;   // A_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint8_t welem_t;  // B_in: lower 8 bits = fp8:e4m3, upper bits zero
typedef uint64_t  out_t;    // C_scaled: fp8:e4m3 (1 byte per output)

// global section that will store fp8 outputs from Gemmini
static uint64_t C_out_got[DIM][DIM] = {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF,};

inline int global_tid() {
  int gtid = 0;
  asm volatile ("csrr %0, mhartid" : "=r" (gtid));
  return gtid;
}

inline void load_scale_factors(volatile uint64_t *sf_mem, const uint8_t *scale_factors, int n) {
  uint64_t *dword_scale_factors = (uint64_t *)scale_factors;
  for (size_t i = 0; i < n / 8; i ++) {
    // do wide stores instead of individual 1-byte stores
    store64_shared(reinterpret_cast<uint32_t>(&sf_mem[i]), 0, dword_scale_factors[i]);
  }
}

/** Convert GPU-local GMEM address to CPU-addressable global address */
inline uint64_t convert_to_global(uint32_t addr) {
  return (static_cast<uint64_t>(addr) | RAD_HOST_GPU_DRAM_BASE);
}

inline int mxgemm() {
  gemmini_flush(0);
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);

  // We want 1 byte per output element in DRAM
  gemmini_extended_config_st(DIM * sizeof(out_t), NO_ACTIVATION, 1);

  // Load per-element scaling factors into the scale SRAM
  //
  // FIXME: currently the scale-factor memory only supports 8-byte writes,
  // preventing GPU to write SF since it only has 4-byte writes.
  // Currently, SF writes are all offloaded to the CPU.
  //
  // load_scale_factors(reinterpret_cast<volatile uint64_t *>(GEMMINI_SF_MEM_A), &A_scales_row[0][0], 32);
  // load_scale_factors(reinterpret_cast<volatile uint64_t *>(GEMMINI_SF_MEM_B), &B_scales_col[0][0], 32);

  // MVIN B as B^T for WS
  // note that gemmini needs CPU-global address for move-in
  gemmini_config_ld(DIM * sizeof(welem_t));
  gemmini_mvin(convert_to_global(reinterpret_cast<uint32_t>(B_in)), 1 * DIM);

  // MVIN A
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_mvin(convert_to_global(reinterpret_cast<uint32_t>(A_in)), 0 * DIM);

  // Preload + compute
  gemmini_config_ld(DIM * sizeof(welem_t));
  gemmini_preload(1 * DIM, (1u << (ADDR_LEN - 1)));
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_compute_preloaded(0 * DIM, GARBAGE_ADDR);

  // MVOUT scaled fp8 C
  gemmini_mvout(convert_to_global(reinterpret_cast<uint32_t>(C_out_got)), (1u << (ADDR_LEN - 1)));

  // synchronize with gemmini completion
  gemmini_fence();

  // read back C_out_got from DMEM to generate some traces
  uint64_t sum = 0;
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      uint64_t got = C_out_got[i][j];
      sum += got;
    }
  }

  return sum;
}

// TODO: use scheduler entry point
int main() {
  if (global_tid() != 0) {
    return 0;
  }

  // tmc: 1
  int res = mxgemm();

  return res;
}
