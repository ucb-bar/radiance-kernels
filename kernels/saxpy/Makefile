PROJECT = saxpy

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

ifneq ($(strip $(SAXPY_ILP)),)
MU_CFLAGS += -DSAXPY_ILP=$(SAXPY_ILP)
endif

ifneq ($(strip $(OCCUPANCIES)),)
ifneq ($(strip $(ILPS)),)
define add_ilp_warp_variant
MU_VARIANTS += kernel_ilp$(1)_w$(2)
kernel_ilp$(1)_w$(2).mu.o: kernel.cpp data kernel_impl.hpp
kernel_ilp$(1)_w$(2).mu.o: MU_CFLAGS += -DSAXPY_ILP=$(1) -DSAXPY_NUM_WARPS=$(2)
endef

$(foreach i,$(ILPS),$(foreach w,$(OCCUPANCIES),$(eval $(call add_ilp_warp_variant,$(i),$(w)))))
else
define add_warp_variant
MU_VARIANTS += kernel_w$(1)
kernel_w$(1).mu.o: kernel.cpp data kernel_impl.hpp
kernel_w$(1).mu.o: MU_CFLAGS += -DSAXPY_NUM_WARPS=$(1)
endef

$(foreach w,$(OCCUPANCIES),$(eval $(call add_warp_variant,$(w))))
endif
endif

include ../common.mk

data: saxpy.py
	python3 saxpy.py

kernel.mu.o: data kernel_impl.hpp
