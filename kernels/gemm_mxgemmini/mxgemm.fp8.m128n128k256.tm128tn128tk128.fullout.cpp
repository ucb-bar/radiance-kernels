// FP8 128x128 output tile, K=256 over 2 K-tiles of TILE_K=128.
//
// This is the FEASIBLE replacement for mxgemm.fp8.m128n128k512.tm128tn128tk256.fullout.cpp.
// That kernel is impossible as written: with TILE_K=256 the double-buffered operands need
// 2048(A)+2048(B) rows per buffer, which leaves exactly ZERO free scratchpad rows for the
// 2048-row C accumulator -- so C landed on top of the A operand and the result was silently
// wrong (measured: 0/16384 elements correct). It never got caught because no data header
// existed for it, so it had never been buildable. GemmConfig::SPAD_DEST() now computes the
// C placement and static_asserts feasibility, so that config is a compile error.
//
// With TILE_K=128: A=B=1024 rows per buffer, C=2048 rows -> C fits in the gap between A_odd
// and B_odd (SPAD_DEST = 3072). Same total K, same output.
#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>

#include "mxgemm.data.fp8.m128n128k256.h"
static const uint8_t A_lut[64][16] = {0};
static const uint8_t B_lut[64][16] = {0};
static const uint8_t C_lut[64][16] = {0};

#include "mxgemm_lib.hpp"

constexpr GemmConfig C{
    .TILE_M = 128,
    .TILE_N = 128,
    .TILE_K = 128, // 256 is infeasible: no scratchpad room left for C
    .DATATYPE = GemmDatatype::FP8,
    .QUANT_OUTPUT = false,
};

void mxgemm_entry(void *arg, uint32_t tid_in_threadblock,
                  uint32_t threads_per_threadblock, uint32_t threadblock_id) {
    auto C_gmem = reinterpret_cast<uint8_t *>(0x40000000);

    mxgemm<C>(C.TILE_M, C.TILE_N, MATMUL_K, C_gmem, tid_in_threadblock,
              threads_per_threadblock, threadblock_id);
}

int main() {
    mu_schedule(mxgemm_entry, nullptr, 2);
    return 0;
}
