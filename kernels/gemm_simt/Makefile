PROJECT = gemm_simt

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

ifneq ($(strip $(TM)),)
MU_CFLAGS += -DTM=$(TM)
endif

ifneq ($(strip $(TN)),)
MU_CFLAGS += -DTN=$(TN)
endif

ifneq ($(strip $(GEMM_SIMT_NUM_WARPS)),)
MU_CFLAGS += -DGEMM_SIMT_NUM_WARPS=$(GEMM_SIMT_NUM_WARPS)
endif

ifneq ($(strip $(OCCUPANCIES)),)
ifeq ($(strip $(TMS)$(TNS)),)
define add_warp_variant
MU_VARIANTS += kernel_w$(1)
kernel_w$(1).mu.o: kernel.cpp data kernel_impl.hpp
kernel_w$(1).mu.o: MU_CFLAGS += -DGEMM_SIMT_NUM_WARPS=$(1)
endef

$(foreach w,$(OCCUPANCIES),$(eval $(call add_warp_variant,$(w))))
else
define add_tile_warp_variant
MU_VARIANTS += kernel_tm$(1)_tn$(2)_w$(3)
kernel_tm$(1)_tn$(2)_w$(3).mu.o: kernel.cpp data kernel_impl.hpp
kernel_tm$(1)_tn$(2)_w$(3).mu.o: MU_CFLAGS += -DTM=$(1) -DTN=$(2) -DGEMM_SIMT_NUM_WARPS=$(3)
endef

$(foreach m,$(TMS),$(foreach n,$(TNS),$(foreach w,$(OCCUPANCIES),$(eval $(call add_tile_warp_variant,$(m),$(n),$(w))))))
endif
endif

include ../common.mk

data: gemm.py
	python3 gemm.py

kernel.mu.o: data kernel_impl.hpp
