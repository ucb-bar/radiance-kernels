#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#ifndef NEARN_NUM_WARPS
#define NEARN_NUM_WARPS 4
#endif

#ifndef NEARN_ILP
#define NEARN_ILP 1
#endif

extern "C" uint32_t __mu_num_warps = NEARN_NUM_WARPS;

struct LatLong {
  float lat;
  float lng;
};

struct NearnArgs {
  __global LatLong* locations;
  __global float* distances;
  uint32_t num_records;
  float lat;
  float lng;
};

template <uint32_t ILP>
static inline void nearn_impl(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  static_assert(ILP > 0);
  auto* args = reinterpret_cast<NearnArgs*>(arg);
  (void)threadblock_id;

  const uint32_t ilp_stride = threads_per_threadblock * ILP;
  const uint32_t ilp_span = threads_per_threadblock * (ILP - 1);
  uint32_t base = tid_in_threadblock;

  for (; base + ilp_span < args->num_records; base += ilp_stride) {
    float dx_vals[ILP];
    float dy_vals[ILP];
    float dist_vals[ILP];

    #pragma unroll
    for (uint32_t u = 0; u < ILP; ++u) {
      const uint32_t i = base + u * threads_per_threadblock;
      __global LatLong* lat_long = args->locations + i;
      dx_vals[u] = args->lat - lat_long->lat;
      dy_vals[u] = args->lng - lat_long->lng;
    }

    #pragma unroll
    for (uint32_t u = 0; u < ILP; ++u) {
      float dx = dx_vals[u];
      float dy = dy_vals[u];
      dist_vals[u] = sqrtf(dx * dx + dy * dy);
    }

    #pragma unroll
    for (uint32_t u = 0; u < ILP; ++u) {
      const uint32_t i = base + u * threads_per_threadblock;
      args->distances[i] = dist_vals[u];
    }
  }

  if constexpr (ILP > 1) {
    for (; base < args->num_records; base += threads_per_threadblock) {
      __global LatLong* lat_long = args->locations + base;
      float dx = args->lat - lat_long->lat;
      float dy = args->lng - lat_long->lng;
      args->distances[base] = sqrtf(dx * dx + dy * dy);
    }
  }
}

static inline void nearn(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  nearn_impl<NEARN_ILP>(
    arg, tid_in_threadblock, threads_per_threadblock, threadblock_id);
}

NearnArgs nearn_args = {
  .locations = nullptr,
  .distances = nullptr,
  .num_records = 0,
  .lat = 0.0f,
  .lng = 0.0f,
};

#include "data"

int main() {
  nearn_args.locations = locations_raw;
  nearn_args.distances = reinterpret_cast<__global float*>(distances_raw);
  nearn_args.num_records = num_records;
  nearn_args.lat = __builtin_bit_cast(float, query_lat_raw);
  nearn_args.lng = __builtin_bit_cast(float, query_lng_raw);
  mu_schedule(nearn, &nearn_args, NEARN_NUM_WARPS);
  return 0;
}
