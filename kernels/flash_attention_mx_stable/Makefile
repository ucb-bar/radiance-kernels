PROJECT = kernel

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

FA_MX_MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(FA_MX_MU_OBJS): mxgemm_core.hpp flash_mx_impl.hpp include/fa_data.h

include ../common.mk


# gemmini.h (a host C reference header) references printf/exit/abs that the muon libc++ set does
# not expose; the shim only makes it parse (these fns are never called on the GPU).
MU_CFLAGS += -include /scratch/yrh/ai-workspace/kernel-gen/toolchain-fix/gemmini_host_shim.h

# Forward the GPU-side -D list to the rv64 host compile too.  host.cpp's MX-scale prefill is gated
# on FA_NOSCALES and MUST match the GPU build: the scale SRAM write port has one shared 2-beat
# pairing counter, so host-side and GPU-side scale writes may never both be enabled.
HOST_CXXFLAGS += $(FA_HOST_DEFS)
