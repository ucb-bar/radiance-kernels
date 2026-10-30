# rpbench: the llama_e2e ROPE phase on cyclotron (device-only ELF, no host).  See kernel.cpp.
#   make CASE=gemma2 DEFS="-DBENCH_MODE=3 -DSIM_NC=4"   -> build/1sm/rpbench.radiance.elf
#   make sweep                                          -> build/rp_<case>_m<mode>_nc<nc>_tb<tb>.elf (see README.md)
PROJECT   = rpbench
CASE     ?= gemma2
PY       ?= python3
MU_SRCS   = kernel.cpp
DEPS      = rp_data_$(CASE).h build/helpers.h build/rope_old.h ../rope_fast.h
BLOBS     = data/rp_$(CASE).region1.bin
OCC       = 4
MU_ADDR_HASH = 0
override DEFS += -DRP_CASE_H='"rp_data_$(CASE).h"'
include ../../nightly.mk

rp_data_%.h data/rp_%.region1.bin: gen_data.py
	$(PY) gen_data.py

# kernel.cpp's helpers and ROPE code as they are (scalar helpers, row-op SMEM map, add_bf2, quant_group, rope_qk,
# rope_fa, rope_v_fa, the fused rope_*_unit)
build/helpers.h: ../kernel.cpp
	mkdir -p build
	{ awk '/^\/\/ ---- scalar helpers/{f=1} /^\/\/ =+ GEMM/{f=0} f' $<; \
	  awk '/^\/\/ =+ Muon row operations/{f=1} /^\/\/ ---- Line ownership/{f=0} f' $<; \
	  awk '/^static inline uint32_t add_bf2/{f=1} f; f && /^}/{f=0}' $<; \
	  awk '/^\/\/ ---- Row ops, lane per group/{f=1} /^\/\/ PRENORM: h = bf16/{f=0} f' $<; \
	  awk '/^\/\/ ---- ROPE: warp per head row/{f=1} /^\/\/ d\[21\]: the ROPE descriptor/{f=0} f' $<; } > $@
	grep -q 'static void rope_fa' $@ && grep -q 'void quant_group' $@
# OLD: rope_fast.h and kernel.cpp's rope_v as committed (warp per unit, gw of ngw)
build/rope_old.h: ../rope_fast.h ../kernel.cpp
	mkdir -p build
	{ git show HEAD:./../rope_fast.h; git show HEAD:./../kernel.cpp | awk '/^static void rope_v\(/{f=1} f; f && /^}/{f=0}'; } > $@
	grep -q 'static void rope_v(' $@

# every case: the check (mode 3, 4 clusters) and the timings (modes 1, 2 of each cluster at 1, 2, 4 clusters)
CASES = gemma2 tinyllama deepseek llama7b
sweep:
	for c in $(CASES); do \
	  $(MAKE) CASE=$$c build/1sm/$(PROJECT).radiance.elf DEFS="-DBENCH_MODE=3 -DSIM_NC=4" && \
	    cp build/1sm/$(PROJECT).radiance.elf build/rp_$${c}_m3_nc4_tball.elf || exit 1; \
	  for m in 1 2; do for nc in 1 2 4; do tb=0; while [ $$tb -lt $$nc ]; do \
	    $(MAKE) CASE=$$c build/1sm/$(PROJECT).radiance.elf DEFS="-DBENCH_MODE=$$m -DSIM_NC=$$nc -DSIM_TB=$$tb" && \
	      cp build/1sm/$(PROJECT).radiance.elf build/rp_$${c}_m$${m}_nc$${nc}_tb$${tb}.elf || exit 1; \
	    tb=$$((tb + 1)); done; done; done; \
	  $(MAKE) CASE=$$c build/1sm/$(PROJECT).radiance.elf DEFS="-DBENCH_MODE=0 -DSIM_NC=1 -DSIM_TB=0" && \
	    cp build/1sm/$(PROJECT).radiance.elf build/rp_$${c}_m0_nc1_tb0.elf || exit 1; \
	done
.PHONY: sweep
