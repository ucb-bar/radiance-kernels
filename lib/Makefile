XLEN ?= 32
TOOLDIR ?= /opt

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv64-gnu-toolchain
CFLAGS += -march=rv64imafd -mabi=lp64d
else
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
CFLAGS += -march=rv32im_zfinx -mabi=ilp32
endif

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

LLVM_VORTEX ?= $(TOOLDIR)/llvm-vortex
LLVM_MUON ?= $(realpath ../llvm/llvm-muon-stride)
# llvm-muon predates -riscv-stack-word-stride and rejects it outright, so the interleaved
# stack needs the newer install.  Point LLVM_MUON back at .../llvm-muon and set
# MU_STACK_WORD_STRIDE=1 together to build the old way.

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex -mllvm -vortex-branch-divergence=0
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0/$(RISCV_PREFIX) 
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0
#LLVM_CFLAGS += -Wl,-L$(RISCV_TOOLCHAIN_PATH)/lib/gcc/$(RISCV_PREFIX)/9.2.0
#LLVM_CFLAGS += --rtlib=libgcc

#CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
#CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
#DP  = $(LLVM_VORTEX)/bin/llvm-objdump
#CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

# CC  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
# CXX = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-g++
AR  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc-ar
# DP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
# CP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

VX_CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
VX_CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
VX_DP  = $(LLVM_VORTEX)/bin/llvm-objdump 
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

MU_CC  = $(LLVM_MUON)/bin/clang $(LLVM_CFLAGS)
MU_CXX = $(LLVM_MUON)/bin/clang++ $(LLVM_CFLAGS)
MU_DP  = $(LLVM_MUON)/bin/llvm-objdump 
MU_CP  = $(LLVM_MUON)/bin/llvm-objcopy

CFLAGS += -O3 -mcmodel=medany -fno-exceptions -fdata-sections -ffunction-sections
CFLAGS += -I./include
CFLAGS += -DXLEN_$(XLEN)

VX_CFLAGS := $(CFLAGS)
VX_CFLAGS += -mllvm -inline-threshold=262144
VX_CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(GEMMINI_SW_PATH)
VX_CFLAGS += -DNDEBUG -DLLVM_VORTEX

MU_CFLAGS := $(VX_CFLAGS)
# Lane-interleaved stack: every stack byte offset is scaled by the stride, and mu_start.S gives lane
# L a stack pointer 4*L bytes into its warp's region, so the sixteen lanes of one stack word share a
# single 64 B line instead of four.  The two must match: a kernel built with this flag must link an
# archive built with it, and MU_STACK_WORD_STRIDE in mu_start.S must be the same number.
MU_STACK_WORD_STRIDE ?= 16
ifneq ($(MU_STACK_WORD_STRIDE),1)
MU_CFLAGS += -mllvm -riscv-stack-word-stride=$(MU_STACK_WORD_STRIDE)
endif
# extra device-side defines for the runtime build (e.g. -DRAD_EPI_PROBE_O=...)
MU_CFLAGS += $(MU_EXTRA_DEFS)
ifeq (1,0)
endif

PROJECT = libmuonrt libvortexrt

VX_SRCS = ./src/vx_start.S ./src/vx_syscalls.c ./src/tinyprintf.c ./src/vx_spawn.c
MU_SRCS = ./src/mu_start.S ./src/mu_schedule.cpp ./src/vx_syscalls.c

VX_OBJS := $(addsuffix .o, $(notdir $(VX_SRCS)))
MU_OBJS := $(addsuffix .o, $(notdir $(MU_SRCS)))

VX_TARGET := $(addsuffix .dump, $(PROJECT))
MU_TARGET := $(addsuffix .dump, $(PROJECT))

all: $(MU_TARGET) $(VX_TARGET)

%.dump: %.a
	$(MU_DP) -D $< > $@

%.S.o: src/%.S
	$(MU_CC) $(MU_CFLAGS) -c $< -o $@

%.cpp.o: src/%.cpp include/vx_spawn.h
	$(MU_CXX) $(MU_CFLAGS) -c $< -o $@

%.c.o: src/%.c include/vx_spawn.h
	$(MU_CC) $(MU_CFLAGS) -c $< -o $@

libvortexrt.a: $(VX_OBJS)
	$(AR) rcs $@ $^

libmuonrt.a: $(MU_OBJS)
	$(AR) rcs $@ $^

# TAPEOUT VARIANT.  Identical sources built with -DMU_TAPEOUT_EPILOGUE, which makes
# mu_schedule_standalone() end in rad_tapeout_epilogue() instead of letting the core assert
# `finished`.  Select it from a kernel with RAD_TAPEOUT=1 (see kernels/common.mk).
# Separate object dir so the two archives can coexist without a clean between them.
MU_OBJS_TO := $(addprefix tapeout-, $(MU_OBJS))
# rad_tapeout_epilogue() is kernel-shaped code -- every postbox write and both drains sit under a
# DIVERGENT guard (`if (l == 0u)`), and the fences and vx_bar between them must run with the full
# thread mask.  LLVM_CFLAGS turns branch-divergence lowering OFF for the runtime, which is fine for
# the rest of mu_schedule but leaves the epilogue's warps unreconverged: measured 2026-09-22, all
# four warps entered the epilogue (postbox slots 40..43 stamped) and NONE got past the opening
# mu_fence(), so the drains never ran and the output stayed stranded.  Kernels are built without
# this flag, which is exactly why the same source works when inlined into a kernel.  Turn it back
# on for the tapeout objects only.
MU_CFLAGS_TO := $(MU_CFLAGS) -mllvm -vortex-branch-divergence=1 $(MU_TAPEOUT_EXTRA)
tapeout-%.S.o: src/%.S
	$(MU_CC) $(MU_CFLAGS_TO) -DMU_TAPEOUT_EPILOGUE -c $< -o $@
tapeout-%.cpp.o: src/%.cpp include/vx_spawn.h
	$(MU_CXX) $(MU_CFLAGS_TO) -DMU_TAPEOUT_EPILOGUE -c $< -o $@
tapeout-%.c.o: src/%.c include/vx_spawn.h
	$(MU_CC) $(MU_CFLAGS_TO) -DMU_TAPEOUT_EPILOGUE -c $< -o $@
libmuonrt-tapeout.a: $(MU_OBJS_TO)
	$(AR) rcs $@ $^

.depend: $(SRCS)
	$(MU_CC) $(MU_CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.a *.o *.dump .depend
