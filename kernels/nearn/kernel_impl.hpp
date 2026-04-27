#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <shared_mem.h>

#include <math.h>
#include <stdint.h>

#ifndef NEARN_NUM_WARPS
#define NEARN_NUM_WARPS 4
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

static inline void nearn(
  void* arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  auto* args = reinterpret_cast<NearnArgs*>(arg);
  (void)threadblock_id;

  for (uint32_t i = tid_in_threadblock; i < args->num_records; i += threads_per_threadblock) {
    __global LatLong* lat_long = args->locations + i;
    float dx = args->lat - lat_long->lat;
    float dy = args->lng - lat_long->lng;
    args->distances[i] = sqrtf(dx * dx + dy * dy);
  }
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
