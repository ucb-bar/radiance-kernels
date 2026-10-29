# smbench: the llama_e2e SOFTMAX phases on cyclotron (device-only ELF, no host).  See kernel.cpp.
#   make bench    -> build/smbench_mode{0,1,2,3}.elf
PROJECT   = smbench
MU_SRCS   = kernel.cpp
DEPS      = sm_data.h build/softmax_vla.h build/softmax_llm.h
BLOBS     = data/sm.region1.bin
OCC       = 4
MU_ADDR_HASH = 0
include ../../nightly.mk

sm_data.h data/sm.region1.bin &: gen_data.py
	python3 gen_data.py

# the SOFTMAX section of kernel_vla.cpp (masked, quant_half, sm_logit, softmax_phase), tested as it is
build/softmax_vla.h: ../kernel_vla.cpp
	mkdir -p build
	awk '/^\/\/ ---- SOFTMAX:/{f=1} /^\/\/ ---- EWQ \/ SILU/{f=0} f' $< > $@
	grep -q 'static void softmax_phase' $@
build/softmax_llm.h: ../kernel.cpp
	mkdir -p build
	awk '/^\/\/ ---- SOFTMAX:/{f=1} /^\/\/ ---- EWQ \/ SILU/{f=0} f' $< > $@
	grep -q 'static void softmax_bf' $@

bench:
	for m in 0 1 2 3; do $(MAKE) build/1sm/$(PROJECT).radiance.elf DEFS="-DBENCH_MODE=$$m $(BENCH_DEFS)" && \
	  cp build/1sm/$(PROJECT).radiance.elf build/smbench_mode$$m.elf && \
	  cp build/1sm/$(PROJECT).radiance.dump build/smbench_mode$$m.dump || exit 1; done
.PHONY: bench
