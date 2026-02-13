TOOLDIR ?= /opt

RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
STARTUP_ADDR ?= 0x80000000

RISCV_PREFIX ?= riscv32-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)
RISCV64_PREFIX ?= riscv64-unknown-elf
RISCV64_TOOLCHAIN_PATH ?= $(RISCV)

VORTEX_KN_PATH ?= $(realpath ../../lib)
GEMMINI_SW_PATH ?= $(realpath ../../lib/gemmini)
SOC_DIR ?= $(realpath ../../soc)

LLVM_MUON ?= $(realpath ../../llvm/llvm-muon)

MU_CC  = $(LLVM_MUON)/bin/clang
MU_CXX = $(LLVM_MUON)/bin/clang++
MU_DP  = $(LLVM_MUON)/bin/llvm-objdump 
MU_CP  = $(LLVM_MUON)/bin/llvm-objcopy

CPU_TOOLCHAIN_PREFIX ?= $(RISCV64_TOOLCHAIN_PATH)/bin/$(RISCV64_PREFIX)
CPU_CC ?= $(CPU_TOOLCHAIN_PREFIX)-gcc
CPU_CXX ?= $(CPU_TOOLCHAIN_PREFIX)-g++
CPU_AS ?= $(CPU_TOOLCHAIN_PREFIX)-as
CPU_LD ?= $(CPU_TOOLCHAIN_PREFIX)-ld
CPU_LINK ?= $(CPU_CC)
CPU_DP ?= $(CPU_TOOLCHAIN_PREFIX)-objdump
CPU_OBJCOPY ?= $(CPU_TOOLCHAIN_PREFIX)-objcopy
CPU_READELF ?= readelf

MU_CFLAGS += --sysroot=$(LLVM_MUON)
MU_CFLAGS += -Xclang -target-feature -Xclang +vortex
MU_CFLAGS += -march=rv32im_zfinx -mabi=ilp32
MU_CFLAGS += -O3 -std=c++17
MU_CFLAGS += -mcmodel=medany -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections
MU_CFLAGS += -mllvm -inline-threshold=262144
MU_CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(GEMMINI_SW_PATH)
MU_CFLAGS += -DNDEBUG -DLLVM_VORTEX

MU_LDFLAGS += -nodefaultlibs -nostartfiles -Wl,-Bstatic,-T,$(VORTEX_KN_PATH)/../muon-isa-tests/env/link.ld,-z,norelro -fuse-ld=lld
MU_LDFLAGS += $(VORTEX_KN_PATH)/libmuonrt.a $(VORTEX_KN_PATH)/tohost.S

ifdef MU_USE_LIBC
# Link in libc + compiler builtins; not sure why it doesn't know about them already
MU_LDFLAGS += -L$(LLVM_MUON)/lib/riscv32-unknown-elf -lc -lm -Wl,$(LLVM_MUON)/lib/clang/18/lib/riscv32-unknown-elf/libclang_rt.builtins.a
endif

CPU_CFLAGS ?= -march=rv64imafd -mabi=lp64d -mcmodel=medany -ffreestanding -fno-common -fno-builtin-printf
CPU_CXXFLAGS ?= $(CPU_CFLAGS)
CPU_LDFLAGS ?= -static -specs=htif_nano.specs
CPU_LIBS ?=

# CONFIG is supplied from the command line to differentiate ELF files with custom suffixes
CONFIGEXT = $(if $(CONFIG),.$(CONFIG),)

PROJECT ?= kernel
BINARIES := $(addsuffix .radiance.elf,$(PROJECT))
OBJDUMPS := $(addsuffix .radiance.dump,$(PROJECT))

ifneq ($(strip $(CPU_SRCS)),)
BINARIES += $(addsuffix .soc.elf,$(PROJECT))
OBJDUMPS += $(addsuffix .soc.dump,$(PROJECT))
endif

all: $(BINARIES) $(OBJDUMPS)

%.radiance.dump: %.radiance.elf
	$(MU_DP) -D $< > $@
%.soc.dump: %.soc.elf
	$(CPU_DP) -D $< > $@

ifneq ($(CONFIG),)
%.radiance$(CONFIGEXT).dump: %.radiance$(CONFIGEXT).elf
	$(MU_DP) -D %.radiance$(CONFIGEXT).elf > $@
endif

OBJCOPY_FLAGS ?= "LOAD,ALLOC,DATA,CONTENTS"
# BINFILES ?=  args.bin input.a.bin input.b.bin input.c.bin
BINFILES ?=

%.radiance.elf: $(MU_SRCS) $(BINFILES)
	$(MU_CXX) $(MU_CFLAGS) $(MU_SRCS) -DRADIANCE -S
	$(MU_CXX) $(MU_CFLAGS) $(MU_SRCS) -DRADIANCE -c
	$(MU_CXX) $(MU_CFLAGS) $(MU_SRCS) $(MU_LDFLAGS) -DRADIANCE -o $@
	@for bin in $(BINFILES); do \
		sec=$$(echo $$bin | sed 's/\.bin$$//'); \
		echo "-$(MU_CP) --update-section .$$sec=$$bin $@"; \
		$(MU_CP) --set-section-flags .input.a=$(OBJCOPY_FLAGS) $@; \
		$(MU_CP) --update-section .$$sec=$$bin $@ || true; \
	done

ifneq ($(CONFIG),)
%.radiance$(CONFIGEXT).elf: %.radiance.elf
	cp $< $@
endif

ifneq ($(strip $(CPU_SRCS)),)
CPU_OBJS := $(addsuffix .cpu.o,$(basename $(CPU_SRCS)))

%.cpu.o: %.c
	$(CPU_CC) $(CPU_CFLAGS) -c $< -o $@
%.cpu.o: %.cc
	$(CPU_CXX) $(CPU_CXXFLAGS) -c $< -o $@
%.cpu.o: %.cpp
	$(CPU_CXX) $(CPU_CXXFLAGS) -c $< -o $@
%.cpu.o: %.S
	$(CPU_CC) $(CPU_CFLAGS) -c $< -o $@
%.cpu.o: %.s
	$(CPU_CC) $(CPU_CFLAGS) -c $< -o $@

%.soc.elf: %.radiance.elf $(CPU_OBJS) $(SOC_DIR)/fuse_rv32_into_rv64.sh $(SOC_DIR)/start.S
	RV32_ELF="$<" OUT="$@" \
	RV64_START="$(SOC_DIR)/start.S" RV64_MAIN= \
	RV64_OBJS="$(CPU_OBJS)" RV64_CFLAGS="$(CPU_CFLAGS)" \
	RV64_LDFLAGS="$(CPU_LDFLAGS)" RV64_LIBS="$(CPU_LIBS)" \
	CC="$(CPU_CC)" LD="$(CPU_LD)" RV64_LINK="$(CPU_LINK)" OBJCOPY="$(CPU_OBJCOPY)" READELF="$(CPU_READELF)" \
	$(SOC_DIR)/fuse_rv32_into_rv64.sh
endif

clean:
	rm -rf *.o
	rm -rf *.cpu.o
	rm -rf $(BINARIES) $(OBJDUMPS)

clean-all: clean
	rm -rf *.o
	rm -rf *.elf
	rm -rf *.dump
