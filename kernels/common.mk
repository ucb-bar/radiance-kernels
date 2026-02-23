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

LLVM_MUON ?= $(realpath ../../llvm/llvm-muon)

MU_CC  = $(LLVM_MUON)/bin/clang
MU_CXX = $(LLVM_MUON)/bin/clang++
MU_OBJDUMP  = $(LLVM_MUON)/bin/llvm-objdump
MU_OBJCOPY  = $(LLVM_MUON)/bin/llvm-objcopy

HOST_TOOLCHAIN_PREFIX ?= $(RISCV64_TOOLCHAIN_PATH)/bin/$(RISCV64_PREFIX)
HOST_CC ?= $(HOST_TOOLCHAIN_PREFIX)-gcc
HOST_CXX ?= $(HOST_TOOLCHAIN_PREFIX)-g++
HOST_AS ?= $(HOST_TOOLCHAIN_PREFIX)-as
HOST_LD ?= $(HOST_TOOLCHAIN_PREFIX)-ld
HOST_LINK ?= $(HOST_CC)
HOST_OBJDUMP ?= $(HOST_TOOLCHAIN_PREFIX)-objdump
HOST_OBJCOPY ?= $(HOST_TOOLCHAIN_PREFIX)-objcopy
HOST_READELF ?= readelf

MU_CFLAGS += --sysroot=$(LLVM_MUON)
MU_CFLAGS += -Xclang -target-feature -Xclang +vortex
MU_CFLAGS += -march=rv32im_zfinx_zhinx -mabi=ilp32
MU_CFLAGS += -O3 -std=c++17
MU_CFLAGS += -mcmodel=medany -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections
MU_CFLAGS += -mllvm -inline-threshold=262144
MU_CFLAGS += -I$(RADIANCE_INCLUDE_PATH) -I$(GEMMINI_SW_PATH)
MU_CFLAGS += -DRADIANCE -DRADIANCE_DEVICE -DNDEBUG -DLLVM_VORTEX

MU_LDFLAGS += -nodefaultlibs -nostartfiles -Wl,-Bstatic,-T,$(RADIANCE_LIB_PATH)/linker/mu_link.ld,-z,norelro -fuse-ld=lld
MU_LDFLAGS += $(RADIANCE_LIB_PATH)/libmuonrt.a $(RADIANCE_LIB_PATH)/tohost.S

ifdef MU_USE_LIBC
# Link in libc + compiler builtins; not sure why it doesn't know about them already
MU_LDFLAGS += -L$(LLVM_MUON)/lib/riscv32-unknown-elf -lc -lm -Wl,$(LLVM_MUON)/lib/clang/18/lib/riscv32-unknown-elf/libclang_rt.builtins.a
endif

HOST_CFLAGS ?= -march=rv64imafd -mabi=lp64d -mcmodel=medany -ffreestanding -fno-common -fno-builtin-printf \
	       -I$(RADIANCE_INCLUDE_PATH) -I$(GEMMINI_SW_PATH)
HOST_CXXFLAGS ?= $(HOST_CFLAGS)
HOST_LDFLAGS ?= -static -specs=htif_nano.specs
HOST_LIBS ?=

# CONFIG is supplied from the command line to differentiate ELF files with custom suffixes
CONFIGEXT = $(if $(CONFIG),.$(CONFIG),)

PROJECT ?= kernel

# MU_SRCS are entrypoint sources that provide main()
# MU_SRC_DEPS are optional shared/common sources linked into every radiance target
ifneq ($(strip $(MU_SRCS)),)
RADIANCE_TARGETS := $(addsuffix .radiance.elf,$(basename $(MU_SRCS)))
else
RADIANCE_TARGETS := $(addsuffix .radiance.elf,$(PROJECT))
endif

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

ifneq ($(CONFIG),)
%.radiance$(CONFIGEXT).dump: %.radiance$(CONFIGEXT).elf
	$(MU_OBJDUMP) -D %.radiance$(CONFIGEXT).elf > $@
endif

OBJCOPY_FLAGS ?= "LOAD,ALLOC,DATA,CONTENTS"
# BINFILES ?=  args.bin input.a.bin input.b.bin input.c.bin
BINFILES ?=
# Optional object files to be linked into *.radiance.elf, e.g. kernel argument
# tensors
MU_BIN_OBJS ?=

%.mu.o: %.cpp
	$(MU_CXX) $(MU_CFLAGS) -c $< -o $@

%.radiance.elf: %.mu.o $(MU_LIB_OBJS) $(MU_BIN_OBJS) $(BINFILES)
	$(MU_CXX) $(MU_CFLAGS) $< $(MU_LIB_OBJS) $(MU_BIN_OBJS) $(MU_LDFLAGS) -o $@
	@for bin in $(BINFILES); do \
		sec=$$(echo $$bin | sed 's/\.bin$$//'); \
		echo "-$(MU_OBJCOPY) --update-section .$$sec=$$bin $@"; \
		$(MU_OBJCOPY) --set-section-flags .input.a=$(OBJCOPY_FLAGS) $@; \
		$(MU_OBJCOPY) --update-section .$$sec=$$bin $@ || true; \
	done

ifneq ($(CONFIG),)
%.radiance$(CONFIGEXT).elf: %.radiance.elf
	cp $< $@
endif

ifneq ($(strip $(HOST_SRCS)),)
HOST_OBJS := $(addsuffix .host.o,$(basename $(HOST_SRCS)))
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
