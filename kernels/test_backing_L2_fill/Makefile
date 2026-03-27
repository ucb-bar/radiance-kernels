PROJECT = test_backing_L2_fill

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

# Replace the standard 4K-aligned linker script with a packed version
# so all kernel code fits in scratchpad page 0, leaving pages 1-2 for data.
MU_LDFLAGS := $(subst $(RADIANCE_LIB_PATH)/linker/mu_link.ld,mu_link_packed.ld,$(MU_LDFLAGS))
