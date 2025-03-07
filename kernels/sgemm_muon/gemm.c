#include <stdint.h>

#define HW_TID(gtid) asm volatile ("csrr %0, mhartid" : "=r" (gtid))
#define N 64
#define NUM_THREADS 16
#define BLOCK_SIZE 16
#define REGISTER

uint32_t *A = (uint32_t *) 0xa0000000ULL;
uint32_t *B = (uint32_t *) 0xa1000000ULL;
uint32_t *C = (uint32_t *) 0xc0000000ULL;


#define INNER_KERNEL(i, j, k) \
    REGISTER uint32_t a0 = A[(i) * N + (k)]; \
    REGISTER uint32_t a1 = A[(i + 1) * N + (k)]; \
    REGISTER uint32_t a2 = A[(i + 2) * N + (k)]; \
    REGISTER uint32_t a3 = A[(i + 3) * N + (k)]; \
    REGISTER uint32_t b0 = B[(j) * N + (k)]; \
    REGISTER uint32_t b1 = B[(j + 1) * N + (k)]; \
    REGISTER uint32_t b2 = B[(j + 2) * N + (k)]; \
    REGISTER uint32_t b3 = B[(j + 3) * N + (k)]; \
    c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3; \
    c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3; \
    c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3; \
    c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;

void gemm() {
    uint32_t tid;
    HW_TID(tid);
    // Calculate the thread-specific row and column bounds
    uint32_t thread_rows = N / NUM_THREADS; // Rows handled by one thread
    uint32_t row_start = tid * thread_rows;
    uint32_t row_end = row_start + thread_rows;

    const uint32_t TILE_SIZE = 4;
    const uint32_t tile_columns = N / TILE_SIZE;
    uint32_t i_stride = NUM_THREADS >= tile_columns ? (NUM_THREADS / tile_columns) : 1;

    // Iterate over tiles of C
    for (uint32_t i = (tid / tile_columns) * TILE_SIZE; i < N; i += i_stride * TILE_SIZE) {
        for (uint32_t j = (tid % tile_columns) * TILE_SIZE; j < N; j += NUM_THREADS * TILE_SIZE) {
            // Initialize accumulators for the 4x4 tile
            REGISTER uint32_t c00 = 0, c01 = 0, c02 = 0, c03 = 0;
            REGISTER uint32_t c10 = 0, c11 = 0, c12 = 0, c13 = 0;
            REGISTER uint32_t c20 = 0, c21 = 0, c22 = 0, c23 = 0;
            REGISTER uint32_t c30 = 0, c31 = 0, c32 = 0, c33 = 0;

            // Compute the outer product for this tile
            for (uint32_t k = 0; k < N; k++) {
                INNER_KERNEL(i, j, k);
            }

            // Write back the results to C
            C[(i) * N + (j)] = c00; C[(i) * N + (j + 1)] = c01; C[(i) * N + (j + 2)] = c02; C[(i) * N + (j + 3)] = c03;
            C[(i + 1) * N + (j)] = c10; C[(i + 1) * N + (j + 1)] = c11; C[(i + 1) * N + (j + 2)] = c12; C[(i + 1) * N + (j + 3)] = c13;
            C[(i + 2) * N + (j)] = c20; C[(i + 2) * N + (j + 1)] = c21; C[(i + 2) * N + (j + 2)] = c22; C[(i + 2) * N + (j + 3)] = c23;
            C[(i + 3) * N + (j)] = c30; C[(i + 3) * N + (j + 1)] = c31; C[(i + 3) * N + (j + 2)] = c32; C[(i + 3) * N + (j + 3)] = c33;
        }
    }
}

__attribute__((section(".text.startup")))
int main() {
    asm volatile (".insn r %0, 1, 0, x0, %1, %2" :: "i"(0x0B), "r"(4), "r"(&gemm));
    gemm();
}
