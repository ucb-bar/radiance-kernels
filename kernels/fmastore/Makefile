PROJECT = fmastore

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

# Single-knob overrides for the base kernel.mu.o (and applied to all variants).
ifneq ($(strip $(FMASTORE_CHAINS)),)
MU_CFLAGS += -DFMASTORE_CHAINS=$(FMASTORE_CHAINS)
endif
ifneq ($(strip $(FMASTORE_CONSUMER)),)
MU_CFLAGS += -DFMASTORE_CONSUMER=$(FMASTORE_CONSUMER)
endif
ifneq ($(strip $(FMASTORE_NUM_WARPS)),)
MU_CFLAGS += -DFMASTORE_NUM_WARPS=$(FMASTORE_NUM_WARPS)
endif
ifneq ($(strip $(FMASTORE_ITERS)),)
MU_CFLAGS += -DFMASTORE_ITERS=$(FMASTORE_ITERS)
endif

# Sweep axes -> variants named kernel_<consumer>_k<K>_w<W>
#   K = FMASTORE_CHAINS (independent depth-2 chains)
#   W = FMASTORE_NUM_WARPS (occupancy)
CHAINS_LIST ?= 1 2 4 8
WARPS_LIST  ?= 1 2 4 8     # occupancy axis, store consumer only
FMADD_WARPS ?= 1           # fmadd consumer kept at single occupancy (comparison baseline)

store_CODE := 0
fmadd_CODE := 1

# (consumer, K, W)
define add_variant
MU_VARIANTS += kernel_$(1)_k$(2)_w$(3)
kernel_$(1)_k$(2)_w$(3).mu.o: kernel.cpp kernel_impl.hpp
kernel_$(1)_k$(2)_w$(3).mu.o: MU_CFLAGS += -DFMASTORE_CONSUMER=$($(1)_CODE) -DFMASTORE_CHAINS=$(2) -DFMASTORE_NUM_WARPS=$(3)
endef

# store: full CHAINS x WARPS occupancy matrix
$(foreach k,$(CHAINS_LIST),$(foreach w,$(WARPS_LIST),$(eval $(call add_variant,store,$(k),$(w)))))
# fmadd: CHAINS x FMADD_WARPS (default w1) -- comparison only, no occupancy sweep
$(foreach k,$(CHAINS_LIST),$(foreach w,$(FMADD_WARPS),$(eval $(call add_variant,fmadd,$(k),$(w)))))

include ../common.mk

kernel.mu.o: kernel_impl.hpp
