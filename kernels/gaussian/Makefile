PROJECT = gaussian

GAUSSIAN_TIMESTEPS = 0 1 2 3

MU_SRCS = \
	$(foreach t,$(GAUSSIAN_TIMESTEPS),fan1_t$(t).cpp fan2_t$(t).cpp)

HOST_SRCS = host.cpp

ifneq ($(strip $(OCCUPANCIES)),)
define add_fan1_variant
MU_VARIANTS += fan1_t$(1)_w$(2)
fan1_t$(1)_w$(2).mu.o: fan1_t$(1).cpp data kernel_impl.hpp
fan1_t$(1)_w$(2).mu.o: MU_CFLAGS += -DGAUSSIAN_NUM_WARPS=$(2)
endef

define add_fan2_variant
MU_VARIANTS += fan2_t$(1)_w$(2)
fan2_t$(1)_w$(2).mu.o: fan2_t$(1).cpp data kernel_impl.hpp
fan2_t$(1)_w$(2).mu.o: MU_CFLAGS += -DGAUSSIAN_NUM_WARPS=$(2)
endef

$(foreach t,$(GAUSSIAN_TIMESTEPS),$(foreach w,$(OCCUPANCIES),$(eval $(call add_fan1_variant,$(t),$(w)))))
$(foreach t,$(GAUSSIAN_TIMESTEPS),$(foreach w,$(OCCUPANCIES),$(eval $(call add_fan2_variant,$(t),$(w)))))
endif

include ../common.mk

data: gaussian.py kernel_impl.hpp verify_trace.py host.cpp Makefile
	python3 gaussian.py

$(addsuffix .mu.o,$(basename $(MU_SRCS))): %.mu.o: data %.cpp kernel_impl.hpp
	$(MU_CXX) $(MU_CFLAGS) -c $*.cpp -o $@
