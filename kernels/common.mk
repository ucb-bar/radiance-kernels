TOOLDIR ?= /opt

RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
STARTUP_ADDR ?= 0x80000000

RISCV_PREFIX ?= riscv32-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)
RISCV64_PREFIX ?= riscv64-unknown-elf
RISCV64_TOOLCHAIN_PATH ?= $(RISCV)

RADIANCE_LIB_PATH ?= $(realpath ../../lib)
RADIANCE_INCLUDE_PATH ?= $(RADIANCE_LIB_PATH)/include
GEMMINI_SW_PATH ?= $(realpath ../../lib/mxgemmini)
SOC_DIR ?= $(realpath ../../soc)

LLVM_MUON ?= $(realpath ../../llvm/llvm-muon-stride)
# llvm-muon predates -riscv-stack-word-stride and rejects it outright, so the interleaved
# stack needs the newer install.  Point LLVM_MUON back at .../llvm-muon and set
# MU_STACK_WORD_STRIDE=1 together to build the old way.

MU_CC  = $(LLVM_MUON)/bin/clang
MU_CXX = $(LLVM_MUON)/bin/clang++
MU_OBJDUMP  = $(LLVM_MUON)/bin/llvm-objdump
MU_OBJCOPY  = $(LLVM_MUON)/bin/llvm-objcopy

# MU_CFLAGS += --sysroot=$(LLVM_MUON)
MU_CFLAGS += --sysroot=$(RISCV_SYSROOT)
MU_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -nodefaultlibs
MU_CFLAGS += -Xclang -target-feature -Xclang +vortex
MU_CFLAGS += -march=rv32im_zfinx_zhinx -mabi=ilp32
MU_CFLAGS += -O3 -std=c++20
MU_CFLAGS += -mcmodel=medany -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections
MU_CFLAGS += -mllvm -inline-threshold=262144
MU_CFLAGS += -I$(RADIANCE_INCLUDE_PATH) -I$(GEMMINI_SW_PATH)
# Shared MX mesh library (one copy for the gemm/gemv/ws kernels; flash kernels keep their
# own same-dir variant, which takes precedence for `#include "mxgemm_lib.hpp"`).
MU_CFLAGS += -I$(RADIANCE_LIB_PATH)/mxgemm
MU_CFLAGS += -DRADIANCE -DRADIANCE_DEVICE -DNDEBUG -DLLVM_VORTEX
# Force-include the abs() disambiguation shim ahead of gemmini.h (see the header) so the
# MX kernels compile against an unmodified gemmini submodule.
MU_CFLAGS += -include $(RADIANCE_INCLUDE_PATH)/gemmini_abs_shim.h

# Extra device-side flags from the caller (e.g. an RTL verify harness passes -DDRAIN_ITERS=200000 so
# the harness can let stores drain before verifying on RTL). This was referenced by the
# gate scripts but never consumed here, silently making the flag a no-op.
MU_CFLAGS += $(EXTRA_MU_CFLAGS)

# The muon LLVM ships libc++ headers but no C library headers. Kernels that pull in
# <math.h>/<stdlib.h> (e.g. anything including gemmini.h -> the mxgemmini kernels)
# therefore fail to compile: libc++'s <math.h>/<stdlib.h> wrappers #include_next the
# C headers and find nothing (FP_NORMAL, ldiv_t, ... undeclared).
#
# Point MU_LIBC_INCLUDE at a newlib include dir to supply them. It MUST be added with
# -idirafter (not -I/-isystem) so it is searched *after* libc++, otherwise libc++
# rejects the C <stdint.h> being found ahead of its own.
# Override on the command line or in the environment for other machines.
MU_LIBC_INCLUDE ?= $(realpath $(dir $(RISCV64_TOOLCHAIN_PATH))/riscv-tools/$(RISCV64_PREFIX)/include)
ifneq ($(MU_LIBC_INCLUDE),)
MU_CFLAGS += -idirafter $(MU_LIBC_INCLUDE)
endif

# Lane-interleaved stack.  Must match MU_STACK_WORD_STRIDE in lib/ (mu_start.S and the archive):
# a kernel built with a different stride than libmuonrt.a will have lanes overwrite each other's
# stack slots.  Set MU_STACK_WORD_STRIDE=1 on both sides for the old thread-major layout.
MU_STACK_WORD_STRIDE ?= 16
ifneq ($(MU_STACK_WORD_STRIDE),1)
MU_CFLAGS += -mllvm -riscv-stack-word-stride=$(MU_STACK_WORD_STRIDE)
endif

MU_LDFLAGS += -nodefaultlibs -nostartfiles -Wl,-Bstatic,-T,$(RADIANCE_LIB_PATH)/linker/mu_link.ld,-z,norelro -fuse-ld=lld
# RAD_TAPEOUT=1 links the TAPEOUT runtime: mu_schedule ends in rad_tapeout_epilogue() instead of
# letting a core assert `finished`.  On the taped-out part the finish edge fires the L0d flush unit
# (MuonTile.scala:371-374), which wedges the L0d, and at occupancy >= 2 `finished` never asserts at
# all, so the entire output stays stranded in cache -- measured on both U250 boards 2026-09-21.
# The host side must poll the printBuf postbox and soft-reset; see rad_host_wait_epilogue() in
# lib/include/rad_tapeout.h, and note that all_finished will NEVER assert by design.
# VERIFIED 2026-09-22 on the U250 tapeout bitstream fe4dc316: RAD_TAPEOUT=1 gives done_cores=2,
# soft reset asserted and still_poison=0/4096, with no kernel edit at all.  It used to lose 64-160
# words; that was NOT a writeback defect but a hang -- the epilogue opened with an all-warp
# mu_fence() (only one warp per core may fence on this part) and never reached its drains, so the
# words that did reach DRAM were ordinary capacity eviction and the rest was residue.  Fixed in
# rad_tapeout.h, together with -mllvm -vortex-branch-divergence=1 for the tapeout objects.
# NOTE: output is correct but not bit-identical to the kernel-source call site (code_sumabs
# 3651795 vs 3651536, 360 vs 364 exact halfwords) -- both at the MX-FP8 floor, cause not yet
# established.  Default stays off until that is understood.
RAD_TAPEOUT ?= 0
ifeq ($(RAD_TAPEOUT),1)
MU_LDFLAGS += $(RADIANCE_LIB_PATH)/libmuonrt-tapeout.a $(RADIANCE_LIB_PATH)/tohost.S
else
MU_LDFLAGS += $(RADIANCE_LIB_PATH)/libmuonrt.a $(RADIANCE_LIB_PATH)/tohost.S
endif

ifdef MU_USE_LIBC
# Link in libc + compiler builtins; not sure why it doesn't know about them already
MU_LDFLAGS += -L$(LLVM_MUON)/lib/riscv32-unknown-elf -lc -lm -Wl,$(LLVM_MUON)/lib/clang/18/lib/riscv32-unknown-elf/libclang_rt.builtins.a
endif

HOST_TOOLCHAIN_PREFIX ?= $(RISCV64_TOOLCHAIN_PATH)/bin/$(RISCV64_PREFIX)
HOST_CC ?= $(HOST_TOOLCHAIN_PREFIX)-gcc
HOST_CXX ?= $(HOST_TOOLCHAIN_PREFIX)-g++
HOST_AS ?= $(HOST_TOOLCHAIN_PREFIX)-as
HOST_LD ?= $(HOST_TOOLCHAIN_PREFIX)-ld
HOST_LINK ?= $(HOST_CC)
HOST_OBJDUMP ?= $(HOST_TOOLCHAIN_PREFIX)-objdump
HOST_OBJCOPY ?= $(HOST_TOOLCHAIN_PREFIX)-objcopy
HOST_READELF ?= readelf

# NOTE (bitten twice: fpga_bringup and fa_stable_fpga): this is `?=`, so setting HOST_CFLAGS or
# HOST_CXXFLAGS in the ENVIRONMENT REPLACES it wholesale and silently drops -march/-mabi/-I.
# To add host-side defines, append AFTER including this file (e.g. HOST_CXXFLAGS += $(FB_HOST_DEFS)).
HOST_CFLAGS ?= -march=rv64imafd -mabi=lp64d -mcmodel=medany -ffreestanding -fno-common -fno-builtin-printf \
	       -I$(RADIANCE_INCLUDE_PATH) -I$(GEMMINI_SW_PATH)
HOST_CXXFLAGS ?= $(HOST_CFLAGS)
HOST_LDFLAGS ?= -static -specs=htif_nano.specs
HOST_LIBS ?=

PROJECT ?= kernel

# MU_SRCS are entrypoint sources that provide main()
# MU_SRC_DEPS are optional shared/common sources linked into every radiance target
ifneq ($(strip $(MU_SRCS)),)
BASE_RADIANCE_TARGETS := $(addsuffix .radiance.elf,$(basename $(MU_SRCS)))
else
BASE_RADIANCE_TARGETS := $(addsuffix .radiance.elf,$(PROJECT))
endif

VARIANT_RADIANCE_TARGETS := $(addsuffix .radiance.elf,$(MU_VARIANTS))
RADIANCE_TARGETS := $(BASE_RADIANCE_TARGETS) $(VARIANT_RADIANCE_TARGETS)
BINARIES := $(RADIANCE_TARGETS)
OBJDUMPS := $(patsubst %.elf,%.dump,$(RADIANCE_TARGETS))
MU_LIB_OBJS := $(sort $(addsuffix .mu.o,$(basename $(MU_SRC_DEPS))))

ifneq ($(strip $(HOST_SRCS)),)
SOC_TARGETS := $(patsubst %.radiance.elf,%.soc.elf,$(RADIANCE_TARGETS))
BINARIES += $(SOC_TARGETS)
OBJDUMPS += $(patsubst %.elf,%.dump,$(SOC_TARGETS))
endif

.DEFAULT_GOAL := all
all: $(BINARIES) $(OBJDUMPS)

%.radiance.dump: %.radiance.elf
	$(MU_OBJDUMP) -D $< > $@
%.soc.dump: %.soc.elf
	$(HOST_OBJDUMP) -D $< > $@

OBJCOPY_FLAGS ?= "LOAD,ALLOC,DATA,CONTENTS"
# BINFILES ?=  args.bin input.a.bin input.b.bin input.c.bin
BINFILES ?=
# Optional object files to be linked into *.radiance.elf, e.g. kernel argument
# tensors
MU_BIN_OBJS ?=

%.mu.o: %.cpp
	$(MU_CXX) $(MU_CFLAGS) -c $< -o $@

ifneq ($(strip $(MU_VARIANTS)),)
VARIANT_MU_OBJS := $(addsuffix .mu.o,$(MU_VARIANTS))

$(VARIANT_MU_OBJS):
	$(MU_CXX) $(MU_CFLAGS) -c $(firstword $(filter %.cpp,$^)) -o $@
endif

%.ll: %.cpp
	$(MU_CXX) $(MU_CFLAGS) -S -emit-llvm $< -o $@

%.radiance.elf: %.mu.o $(MU_LIB_OBJS) $(MU_BIN_OBJS) $(BINFILES)
	$(MU_CXX) $(MU_CFLAGS) $< $(MU_LIB_OBJS) $(MU_BIN_OBJS) $(MU_LDFLAGS) -o $@
	@for bin in $(BINFILES); do \
		sec=$$(echo $$bin | sed 's/\.bin$$//'); \
		echo "-$(MU_OBJCOPY) --update-section .$$sec=$$bin $@"; \
		$(MU_OBJCOPY) --set-section-flags .input.a=$(OBJCOPY_FLAGS) $@; \
		$(MU_OBJCOPY) --update-section .$$sec=$$bin $@ || true; \
	done

ifneq ($(strip $(HOST_SRCS)),)
HOST_OBJS := $(addsuffix .host.o,$(basename $(HOST_SRCS)))
# don't delete HOST_OBJS and cause rebuilds of *.soc.elf
.SECONDARY: $(HOST_OBJS)
endif

%.host.o: %.c
	$(HOST_CC) $(HOST_CFLAGS) -c $< -o $@
%.host.o: %.cc
	$(HOST_CXX) $(HOST_CXXFLAGS) -c $< -o $@
%.host.o: %.cpp
	$(HOST_CXX) $(HOST_CXXFLAGS) -c $< -o $@
%.host.o: %.S
	$(HOST_CC) $(HOST_CFLAGS) -c $< -o $@
%.host.o: %.s
	$(HOST_CC) $(HOST_CFLAGS) -c $< -o $@

%.soc.elf: %.radiance.elf $(HOST_OBJS) $(SOC_DIR)/fuse_rv32_into_rv64.sh $(SOC_DIR)/start.S
	RV32_ELF="$<" OUT="$@" \
	RV64_START="$(SOC_DIR)/start.S" RV64_MAIN= \
	RV64_OBJS="$(HOST_OBJS)" RV64_CFLAGS="$(HOST_CFLAGS)" \
	RV64_LDFLAGS="$(HOST_LDFLAGS)" RV64_LIBS="$(HOST_LIBS)" \
	CC="$(HOST_CC)" LD="$(HOST_LD)" RV64_LINK="$(HOST_LINK)" OBJCOPY="$(HOST_OBJCOPY)" READELF="$(HOST_READELF)" \
	$(SOC_DIR)/fuse_rv32_into_rv64.sh

clean:
	rm -rf *.o
	rm -rf *.host.o
	rm -rf $(BINARIES) $(OBJDUMPS)

clean-all: clean
	rm -rf *.o
	rm -rf *.elf
	rm -rf *.dump
