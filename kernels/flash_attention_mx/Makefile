PROJECT = kernel

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

FA_MX_MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(FA_MX_MU_OBJS): mxgemm_core.hpp flash_mx_impl.hpp include/fa_data.h

include ../common.mk

