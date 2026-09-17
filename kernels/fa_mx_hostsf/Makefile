PROJECT = flash_attention_mx

MU_SRCS = flash_attention_mx.cpp
HOST_SRCS = host.cpp

FA_MX_MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(FA_MX_MU_OBJS): mxgemm_core.hpp include/fa_data.h

include ../common.mk

# gemmini.h (a host C reference header) references printf/exit/abs that the muon
# libc++ set doesn't expose; the shim only makes it parse (these fns aren't called
# on the GPU). Same fix the mxgemm build uses.

# Forward the GPU-side -D list to the rv64 host compile too (empty unless the caller exports
# FA_HOST_DEFS -- /tmp/fa_build.sh does).  host.cpp's MX-scale prefill is gated on FA_NOSCALES
# and MUST match the GPU build: the scale SRAM write port has one shared 2-beat pairing
# counter, so host-side and GPU-side scale writes may never be enabled at the same time.
HOST_CXXFLAGS += $(FA_HOST_DEFS)
