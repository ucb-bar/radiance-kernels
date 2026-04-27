PROJECT = sfilter

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

ifneq ($(strip $(OCCUPANCIES)),)
define add_warp_variant
MU_VARIANTS += kernel_w$(1)
kernel_w$(1).mu.o: kernel.cpp data kernel_impl.hpp
kernel_w$(1).mu.o: MU_CFLAGS += -DSFILTER_NUM_WARPS=$(1)
endef

$(foreach w,$(OCCUPANCIES),$(eval $(call add_warp_variant,$(w))))
endif

include ../common.mk

data: sfilter.py
	python3 sfilter.py

kernel.mu.o: data kernel_impl.hpp
