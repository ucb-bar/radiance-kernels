PROJECT = kernel

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

FA_MX_MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(FA_MX_MU_OBJS): mxgemm_core.hpp flash_mx_impl.hpp include/fa_data.h

include ../common.mk


# gemmini.h (a host C reference header) references printf/exit/abs that the muon libc++ set does
# not expose; the shim only makes it parse (these fns are never called on the GPU).

# Forward the GPU-side -D list to the rv64 host compile too.  host.cpp's MX-scale prefill is gated
# on FA_NOSCALES and MUST match the GPU build: the scale SRAM write port has one shared 2-beat
# pairing counter, so host-side and GPU-side scale writes may never both be enabled.
HOST_CXXFLAGS += $(FA_HOST_DEFS)

# FA_LD=fa swaps in kernels/fa_stable_fpga/mu_link_fa.ld, which differs from the shared script in
# exactly two ways: `.bss : { *(.bss .bss.*) }` (the shared one drops -fdata-sections names, which
# is why C_scale_factors and schedule_context sit past the RW segment FileSiz and hold stale DRAM)
# and an early-text slot for FA_MAIN_EARLY.  common.mk appends -T with +=, so substitute rather
# than append or both scripts end up on the command line.
ifeq ($(FA_LD),fa)
MU_LDFLAGS := $(subst $(RADIANCE_LIB_PATH)/linker/mu_link.ld,/scratch/yrh/radiance-kernels/kernels/fa_stable_fpga/mu_link_fa.ld,$(MU_LDFLAGS))
endif
