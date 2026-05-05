PROJECT = gemm_simt

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

ifneq ($(strip $(TM)),)
MU_CFLAGS += -DTM=$(TM)
endif

ifneq ($(strip $(TN)),)
MU_CFLAGS += -DTN=$(TN)
endif

ifneq ($(strip $(TMS)$(TNS)),)
define add_tile_variant
MU_VARIANTS += kernel_tm$(1)_tn$(2)
kernel_tm$(1)_tn$(2).mu.o: kernel.cpp data kernel_impl.hpp
kernel_tm$(1)_tn$(2).mu.o: MU_CFLAGS += -DTM=$(1) -DTN=$(2)
endef

$(foreach m,$(TMS),$(foreach n,$(TNS),$(eval $(call add_tile_variant,$(m),$(n)))))
endif

ifneq ($(strip $(CTA_TILE_SWEEP)),)
define add_cta_tile_variant
MU_VARIANTS += kernel_cta_tm$(1)_tn$(2)_w$(3)
kernel_cta_tm$(1)_tn$(2)_w$(3).mu.o: kernel.cpp data kernel_impl.hpp
kernel_cta_tm$(1)_tn$(2)_w$(3).mu.o: MU_CFLAGS += -DTM=$(1) -DTN=$(2)
endef

$(eval $(call add_cta_tile_variant,1,2,8))
$(eval $(call add_cta_tile_variant,1,4,4))
$(eval $(call add_cta_tile_variant,2,2,4))
$(eval $(call add_cta_tile_variant,2,4,2))
$(eval $(call add_cta_tile_variant,4,2,2))
$(eval $(call add_cta_tile_variant,2,8,1))
$(eval $(call add_cta_tile_variant,4,4,1))
$(eval $(call add_cta_tile_variant,8,2,1))
endif

include ../common.mk

data: gemm.py
	python3 gemm.py

kernel.mu.o: data kernel_impl.hpp
