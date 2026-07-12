PROJECT = flash_attention_mx

MU_SRCS = flash_attention_mx.cpp
HOST_SRCS = host.cpp

FA_MX_MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(FA_MX_MU_OBJS): mxgemm_core.hpp include/fa_data.h

include ../common.mk

# gemmini.h (a host C reference header) references printf/exit/abs that the muon
# libc++ set doesn't expose; the shim only makes it parse (these fns aren't called
# on the GPU). Same fix the mxgemm build uses.
MU_CFLAGS += -include /scratch/yrh/ai-workspace/kernel-gen/toolchain-fix/gemmini_host_shim.h
