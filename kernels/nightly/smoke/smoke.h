// smoke: uint32 vecadd C = A + B over all warps of every cluster; checks the nightly launch
// protocol and host read-back.  A, B are initialised by the GPU image (static data).
#pragma once
#include <stdint.h>
#define SMOKE_N 16384u
#define SMOKE_OCC 8u
static inline uint32_t smoke_a(uint32_t i) { return i * 3u + 1u; }
static inline uint32_t smoke_b(uint32_t i) { return i * 7u + 2u; }
