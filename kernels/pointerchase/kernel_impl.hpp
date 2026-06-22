#include <mu_intrinsics.h>
#include <mu_schedule.h>

#include <stdint.h>

#ifndef POINTERCHASE_NUM_WARPS
#define POINTERCHASE_NUM_WARPS 8
#endif

#ifndef POINTERCHASE_ITERS
#define POINTERCHASE_ITERS 1024
#endif

#ifndef POINTERCHASE_UNROLL
#define POINTERCHASE_UNROLL 16
#endif

#define POINTERCHASE_BLOCK_NUM_WARPS MU_BLOCK_NUM_WARPS(POINTERCHASE_NUM_WARPS)

static_assert(POINTERCHASE_NUM_WARPS >= 1);
static_assert(POINTERCHASE_NUM_WARPS <= MU_NUM_MAX_WARPS);
static_assert(POINTERCHASE_ITERS > 0);
static_assert(POINTERCHASE_UNROLL > 0);
static_assert((POINTERCHASE_ITERS % POINTERCHASE_UNROLL) == 0);

extern "C" uint32_t __mu_num_warps = POINTERCHASE_NUM_WARPS;

struct PointerChaseNode {
  __global PointerChaseNode* next;
  uint32_t payload;
};

struct PointerChaseArgs {
  __global PointerChaseNode* head;
};

static inline volatile __global PointerChaseNode* pointerchase_next(
  volatile __global PointerChaseNode* node
) {
  __global PointerChaseNode* next = node->next;
  asm volatile("" : "+r"(next) :: "memory");
  return reinterpret_cast<volatile __global PointerChaseNode*>(next);
}

static inline void pointerchase(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<PointerChaseArgs*>(arg);
  (void)threads_per_threadblock;

  const uint32_t lane_id = tid_in_threadblock % MU_NUM_THREADS;
  const uint32_t warp_id = tid_in_threadblock / MU_NUM_THREADS;

  if (lane_id == 0) {
    volatile __global PointerChaseNode* node = args->head;

    #pragma unroll 1
    for (uint32_t i = 0; i < POINTERCHASE_ITERS; i += POINTERCHASE_UNROLL) {
      #pragma unroll
      for (uint32_t u = 0; u < POINTERCHASE_UNROLL; ++u) {
        node = pointerchase_next(node);
      }
    }

    asm volatile("" :: "r"(node) : "memory");
  }
}

PointerChaseArgs pointerchase_args = {
  .head = nullptr,
};

#include "data"

int main() {
  // self-referencing; always hits cache
  pointerchase_nodes[0].next = &pointerchase_nodes[0];

  pointerchase_args.head = &pointerchase_nodes[0];
  mu_schedule(pointerchase, &pointerchase_args, POINTERCHASE_NUM_WARPS);
  return 0;
}
