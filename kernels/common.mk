TOOLDIR ?= /opt

RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
VX_CFLAGS += -march=rv32im_zfinx -mabi=ilp32
STARTUP_ADDR ?= 0x80000000

RISCV_PREFIX ?= riscv32-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

VORTEX_KN_PATH ?= $(realpath ../../lib)
GEMMINI_SW_PATH ?= $(realpath ../../lib/gemmini)

LLVM_VORTEX ?= $(TOOLDIR)/llvm-vortex

LLVM_MUON ?= $(realpath ../../llvm/llvm-muon)

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -nodefaultlibs
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
VX_CFLAGS += -mcmodel=medany -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections
VX_CFLAGS += -mllvm -inline-threshold=262144
VX_CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(GEMMINI_SW_PATH)
VX_CFLAGS += -DNDEBUG -DLLVM_VORTEX

MU_CFLAGS := $(VX_CFLAGS)

VX_LDFLAGS += -nostartfiles -Wl,-Bstatic,-T,$(VORTEX_KN_PATH)/linker/vx_link32.ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR)
MU_LDFLAGS := -fuse-ld=lld $(VX_LDFLAGS)
VX_LDFLAGS += $(VORTEX_KN_PATH)/libvortexrt.a
MU_LDFLAGS += $(VORTEX_KN_PATH)/libmuonrt.a $(VORTEX_KN_PATH)/tohost.S

# CONFIG is supplied from the command line to differentiate ELF files with custom suffixes
CONFIGEXT = $(if $(CONFIG),.$(CONFIG),)

all: kernel.radiance.dump kernel.radiance$(CONFIGEXT).dump # kernel.vortex.dump

kernel.vortex.dump: kernel.vortex.elf
	$(VX_DP) -D kernel.vortex.elf > kernel.vortex.dump
kernel.radiance.dump: kernel.radiance.elf
	$(MU_DP) -D kernel.radiance.elf > kernel.radiance.dump

ifneq ($(CONFIG),)
kernel.radiance$(CONFIGEXT).dump: kernel.radiance$(CONFIGEXT).elf
	$(MU_DP) -D kernel.radiance$(CONFIGEXT).elf > kernel.radiance$(CONFIGEXT).dump
endif

OBJCOPY_FLAGS ?= "LOAD,ALLOC,DATA,CONTENTS"
BINFILES ?=  args.bin input.a.bin input.b.bin input.c.bin

# kernel.vortex.elf: $(VX_SRCS) $(VX_INCLUDES) $(BINFILES)
# 	$(VX_CXX) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -DRADIANCE -o $@
# 
# 	@for bin in $(BINFILES); do \
# 		sec=$$(echo $$bin | sed 's/\.bin$$//'); \
# 		echo "-$(VX_CP) --update-section .$$sec=$$bin $@"; \
# 		$(VX_CP) --set-section-flags .input.a=$(OBJCOPY_FLAGS) $@; \
# 		$(VX_CP) --update-section .$$sec=$$bin $@ || true; \
# 	done

kernel.radiance.elf: $(VX_SRCS) $(VX_INCLUDES) $(BINFILES)
	$(MU_CXX) $(MU_CFLAGS) $(VX_SRCS) -DRADIANCE -S
	$(MU_CXX) $(MU_CFLAGS) $(VX_SRCS) -DRADIANCE -c
	$(MU_CXX) $(MU_CFLAGS) $(VX_SRCS) $(MU_LDFLAGS) -DRADIANCE -o $@
	@for bin in $(BINFILES); do \
		sec=$$(echo $$bin | sed 's/\.bin$$//'); \
		echo "-$(MU_CP) --update-section .$$sec=$$bin $@"; \
		$(MU_CP) --set-section-flags .input.a=$(OBJCOPY_FLAGS) $@; \
		$(MU_CP) --update-section .$$sec=$$bin $@ || true; \
	done

ifneq ($(CONFIG),)
kernel.radiance$(CONFIGEXT).elf: kernel.radiance.elf
	cp $< $@
endif

clean:
	rm -rf *.o
	rm -rf *.elf
	rm -rf *.dump

clean-all: clean
	rm -rf kernel*.elf kernel*.dump
