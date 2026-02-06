#ifndef RADIANCE_KERNELS_TUNING_TUNE_ARGS_H_
#define RADIANCE_KERNELS_TUNING_TUNE_ARGS_H_

#include "../../memstress_common/helpers.h"

namespace tune {

inline kernel_arg_t* args() {
  return reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
}

inline uint32_t iterations(kernel_arg_t* arg, uint32_t fallback) {
  return memstress::resolve_iterations(arg, fallback);
}

inline volatile uint32_t* out_u32(kernel_arg_t* arg) {
  return memstress::dst_ptr(arg);
}

}

#endif
