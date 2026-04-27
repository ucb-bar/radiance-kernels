PROJECT = vecadd

MU_SRCS = \
	kernel.cpp \
	kernel_ilp2.cpp \
	kernel_ilp4.cpp \
	kernel_ilp4_ou2.cpp \
	kernel_ilp8.cpp \
	kernel_ilp2_w8.cpp \
	kernel_ilp4_w2.cpp \
	kernel_ilp4_w8.cpp
HOST_SRCS = host.cpp

ifneq ($(strip $(OCCUPANCIES)),)
define add_warp_variant
MU_VARIANTS += kernel_w$(1)
kernel_w$(1).mu.o: kernel.cpp data kernel_impl.hpp
kernel_w$(1).mu.o: MU_CFLAGS += -DVECADD_NUM_WARPS=$(1)
endef

$(foreach w,$(OCCUPANCIES),$(eval $(call add_warp_variant,$(w))))
endif

include ../common.mk

data: vecadd.py extract_rtl_metrics.py
	python3 vecadd.py

$(addsuffix .mu.o,$(basename $(MU_SRCS))): data kernel_impl.hpp
