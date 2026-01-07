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
LLVM_MUON ?= $(realpath ../llvm/llvm-muon)

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

PROJECT = libmuonrt libvortexrt

VX_SRCS = ./src/vx_start.S ./src/vx_syscalls.c ./src/tinyprintf.c ./src/vx_spawn.c
MU_SRCS = ./src/mu_start.S ./src/vx_syscalls.c ./src/vx_spawn.c

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

.depend: $(SRCS)
	$(MU_CC) $(MU_CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.a *.o *.dump .depend
