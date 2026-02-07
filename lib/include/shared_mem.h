#pragma once

#include <VX_config.h>
#include <stdint.h>

#ifndef DEV_SMEM_START_ADDR
#define DEV_SMEM_START_ADDR SMEM_BASE_ADDR
#endif

#ifndef SMEM_SIZE
#define SMEM_SIZE (1u << SMEM_LOG_SIZE)
#endif

#define __global __attribute__((address_space(0)))
#define __shared __attribute__((address_space(1)))
