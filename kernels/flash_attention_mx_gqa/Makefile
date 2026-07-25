PROJECT = kernel

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(MU_OBJS): mxgemm_core.hpp flash_mx_impl.hpp include/fa_data.h

include ../common.mk

# NOTE (garden port): dropped Richard's external gemmini_host_shim.h (-include) — our
# gemm_mxgemmini builds without it against the tapeout-330 headers.
