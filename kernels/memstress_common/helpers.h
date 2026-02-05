#ifndef RADIANCE_KERNELS_MEMSTRESS_HELPERS_H_
#define RADIANCE_KERNELS_MEMSTRESS_HELPERS_H_

#include <vx_intrinsics.h>
#include "common.h"

namespace memstress {

constexpr uint64_t kDefaultSrcBase = 0x90000000UL;
constexpr uint64_t kDefaultDstBase = 0x91000000UL;
constexpr uint64_t kDefaultAuxBase = 0x92000000UL;
constexpr uint32_t kDefaultLanesPerWarp = 16;
constexpr uint32_t kDefaultWarpsPerCore = 4;

inline uint32_t default_thread_count() {
  return kDefaultLanesPerWarp * kDefaultWarpsPerCore;
}

inline uint32_t hw_threads_per_core() {
  uint32_t lanes = vx_num_threads();
  uint32_t warps = vx_num_warps();
  if (lanes == 0) {
    lanes = kDefaultLanesPerWarp;
  }
  if (warps == 0) {
    warps = kDefaultWarpsPerCore;
  }
  return lanes * warps;
}

inline uint32_t resolve_threads(const kernel_arg_t* arg) {
  if (arg && arg->num_threads) {
    return arg->num_threads;
  }
  uint32_t threads = hw_threads_per_core();
  if (threads == 0) {
    threads = default_thread_count();
  }
  return threads;
}

inline uint32_t resolve_iterations(const kernel_arg_t* arg, uint32_t fallback) {
  return (arg && arg->iterations) ? arg->iterations : fallback;
}

inline uint32_t resolve_elems(const kernel_arg_t* arg, uint32_t fallback) {
  return (arg && arg->elements_per_thread) ? arg->elements_per_thread : fallback;
}

inline uint32_t resolve_stride(const kernel_arg_t* arg, uint32_t fallback) {
  return (arg && arg->stride_elems) ? arg->stride_elems : fallback;
}

inline uint32_t resolve_tile(const kernel_arg_t* arg, uint32_t fallback, uint32_t max_value) {
  uint32_t requested = fallback;
  if (arg && arg->tile_elems) {
    requested = arg->tile_elems;
  }
  if (requested > max_value) {
    requested = max_value;
  }
  return requested;
}

inline uint32_t resolve_smem_banks(const kernel_arg_t* arg, uint32_t fallback) {
  return (arg && arg->smem_banks) ? arg->smem_banks : fallback;
}

inline volatile uint32_t* as_u32_ptr(uint64_t addr, uint64_t fallback) {
  uint64_t chosen = addr ? addr : fallback;
  return reinterpret_cast<volatile uint32_t*>(chosen);
}

inline volatile uint32_t* src_ptr(const kernel_arg_t* arg) {
  return as_u32_ptr(arg ? arg->gmem_src : 0, kDefaultSrcBase);
}

inline volatile uint32_t* dst_ptr(const kernel_arg_t* arg) {
  return as_u32_ptr(arg ? arg->gmem_dst : 0, kDefaultDstBase);
}

inline volatile uint32_t* aux_ptr(const kernel_arg_t* arg) {
  return as_u32_ptr(arg ? arg->gmem_aux : 0, kDefaultAuxBase);
}

}  // namespace memstress

#endif  // RADIANCE_KERNELS_MEMSTRESS_HELPERS_H_
