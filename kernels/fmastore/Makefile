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

# Sweep variants: CONSUMER x CHAINS -> kernel_<consumer>_k<K>
CHAINS_LIST ?= 1 2 4 8
CONSUMERS ?= store fmadd

store_CODE := 0
fmadd_CODE := 1

ifneq ($(strip $(CHAINS_LIST)),)
ifneq ($(strip $(CONSUMERS)),)
define add_variant
MU_VARIANTS += kernel_$(1)_k$(2)
kernel_$(1)_k$(2).mu.o: kernel.cpp kernel_impl.hpp
kernel_$(1)_k$(2).mu.o: MU_CFLAGS += -DFMASTORE_CONSUMER=$($(1)_CODE) -DFMASTORE_CHAINS=$(2)
endef
$(foreach c,$(CONSUMERS),$(foreach k,$(CHAINS_LIST),$(eval $(call add_variant,$(c),$(k)))))
endif
endif

include ../common.mk

kernel.mu.o: kernel_impl.hpp
