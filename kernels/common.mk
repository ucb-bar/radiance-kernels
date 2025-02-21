XLEN ?= 32

TOOLDIR ?= /opt

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv64-gnu-toolchain
VX_CFLAGS += -march=rv64imafd -mabi=lp64d
STARTUP_ADDR ?= 0x180000000
else
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
STARTUP_ADDR ?= 0x80000000
endif

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

VORTEX_KN_PATH ?= $(realpath ../../lib)
GEMMINI_SW_PATH ?= $(realpath ../../lib/gemmini)

LLVM_VORTEX ?= $(TOOLDIR)/llvm-vortex

LLVM_MUON ?= $(TOOLDIR)/llvm-muon

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex

#LLVM_CFLAGS += -mllvm -vortex-branch-divergence=2
#LLVM_CFLAGS += -mllvm -print-after-all 
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0/$(RISCV_PREFIX) 
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0
#LLVM_CFLAGS += -Wl,-L$(RISCV_TOOLCHAIN_PATH)/lib/gcc/$(RISCV_PREFIX)/9.2.0
#LLVM_CFLAGS += --rtlib=libgcc

VX_CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
VX_CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
VX_DP  = $(LLVM_VORTEX)/bin/llvm-objdump 
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

MU_CC  = $(LLVM_MUON)/bin/clang $(LLVM_CFLAGS)
MU_CXX = $(LLVM_MUON)/bin/clang++ $(LLVM_CFLAGS)
MU_DP  = $(LLVM_MUON)/bin/llvm-objdump 
MU_CP  = $(LLVM_MUON)/bin/llvm-objcopy

VX_CFLAGS += -v -O3 -std=c++17
VX_CFLAGS += -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -fdata-sections -ffunction-sections
# comment out below for regression/basic, which uses GCC that doesn't
# understand these flags
VX_CFLAGS += -mllvm -inline-threshold=262144
VX_CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(GEMMINI_SW_PATH)
VX_CFLAGS += -DNDEBUG -DLLVM_VORTEX

MU_CFLAGS := $(VX_CFLAGS)
MU_CFLAGS += -fuse-ld=lld

VX_LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR)
MU_LDFLAGS := $(VX_LDFLAGS)
VX_LDFLAGS += $(VORTEX_KN_PATH)/libvortexrt.a
MU_LDFLAGS += $(VORTEX_KN_PATH)/libvortexrtmuon.a $(VORTEX_KN_PATH)/tohost.S

# CONFIG is supplied from the command line to differentiate ELF files with custom suffixes
CONFIGEXT = $(if $(CONFIG),.$(CONFIG),)

all: kernel.radiance.dump kernel.radiance$(CONFIGEXT).dump kernel.vortex.dump

kernel.vortex.dump: kernel.vortex.elf
	$(VX_DP) -D kernel.vortex.elf > kernel.vortex.dump
kernel.radiance.dump: kernel.radiance.elf
	$(MU_DP) -D kernel.radiance.elf > kernel.radiance.dump

ifneq ($(CONFIG),)
kernel.radiance$(CONFIGEXT).dump: kernel.radiance$(CONFIGEXT).elf
	$(MU_DP) -D kernel.radiance$(CONFIGEXT).elf > kernel.radiance$(CONFIGEXT).dump
endif

OBJCOPY_FLAGS ?= "LOAD,ALLOC,DATA,CONTENTS"
BINFILES :=  args.bin input.a.bin input.b.bin input.c.bin

kernel.vortex.elf: $(VX_SRCS) $(VX_INCLUDES) $(BINFILES)
	$(VX_CXX) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -DRADIANCE -o $@
	$(VX_CP) --set-section-flags .operand.a=$(OBJCOPY_FLAGS) $@
	$(VX_CP) --set-section-flags .operand.b=$(OBJCOPY_FLAGS) $@
	$(VX_CP) --set-section-flags .operand.c=$(OBJCOPY_FLAGS) $@
	$(VX_CP) --set-section-flags .args=$(OBJCOPY_FLAGS) $@
	$(VX_CP) --update-section .operand.a=input.a.bin $@ || true
	$(VX_CP) --update-section .operand.b=input.b.bin $@ || true
	$(VX_CP) --update-section .operand.c=input.c.bin $@ || true
	$(VX_CP) --update-section .args=args.bin $@ || true

kernel.radiance.elf: $(VX_SRCS) $(VX_INCLUDES) $(BINFILES)
	$(MU_CXX) $(MU_CFLAGS) $(VX_SRCS) $(MU_LDFLAGS) -DRADIANCE -S
	$(MU_CXX) $(MU_CFLAGS) $(VX_SRCS) $(MU_LDFLAGS) -DRADIANCE -c
	$(MU_CXX) $(MU_CFLAGS) $(VX_SRCS) $(MU_LDFLAGS) -DRADIANCE -o $@
	# $(MU_CP) --set-section-flags .operand.a=$(OBJCOPY_FLAGS) $@
	# $(MU_CP) --set-section-flags .operand.b=$(OBJCOPY_FLAGS) $@
	# $(MU_CP) --set-section-flags .operand.c=$(OBJCOPY_FLAGS) $@
	# $(MU_CP) --set-section-flags .args=$(OBJCOPY_FLAGS) $@
	# $(MU_CP) --update-section .operand.a=input.a.bin $@ || true
	# $(MU_CP) --update-section .operand.b=input.b.bin $@ || true
	# $(MU_CP) --update-section .operand.c=input.c.bin $@ || true
	# $(MU_CP) --update-section .args=args.bin $@ || true

ifneq ($(CONFIG),)
kernel.radiance$(CONFIGEXT).elf: kernel.radiance.elf
	cp $< $@
endif

clean:
	rm -rf *.o

clean-all: clean
	rm -rf kernel*.elf kernel*.dump
