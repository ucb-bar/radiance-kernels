# nightly.mk: build rules for the kernels under kernels/nightly (nightly RTL, see
# lib/include/nightly/README.md).
#
# A kernel Makefile sets PROJECT, MU_SRCS (one device source with main()), HOST_SRCS, optional
# DEPS (generated headers), then includes this file.  Every kernel is built once per cluster
# count, into its own directory so the objects never mix:
#
#   build/1sm/$(PROJECT).soc.elf   NIGHTLY_CLUSTERS=1  (RadianceHBMConfig, cluster 0 only)
#   build/2sm/$(PROJECT).soc.elf   NIGHTLY_CLUSTERS=2  (RadianceHBMConfig, both clusters)
#
# `make` builds both.  `make SMS=1` builds one.  Extra defines: DEFS="-DFOO=1 -DBAR".
# With NIGHTLY_CLUSTERS=1 the host launches cluster 0 only; cluster 1 waits for LAUNCH forever.
#
# MU_ADDR_HASH (default 1): RadianceHBMConfig hashes GPU memory over the 4 L2 slices / DRAM
# channels (radiance WithGPUAddressHash).  SimDRAM +loadmem and FireSim LoadMem write DRAM
# directly, so the GPU load segments are scrambled into the hashed layout before the fuse step
# (soc/scramble_gpu_elf.py).  Use MU_ADDR_HASH=0 only for a loader that goes through the hash
# (TSI, RadianceHBMTSIConfig) or for an unhashed config.  The flag is part of the DEFS stamp, so
# switching it rebuilds.
#
# BLOBS: raw binary files named <anything>.regionN.bin (N = 0..3) are linked into section
# .regionN of the device image (see lib/linker/mu_link_nightly.ld for the addresses).  Use them
# for input tensors and goldens; the generator emits a header with the offsets.

NIGHTLY_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
RK_ROOT     := $(realpath $(NIGHTLY_DIR)/../..)

TOOLDIR ?= /opt
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
RISCV_SYSROOT := $(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf
RISCV64_TOOLCHAIN_PATH ?= $(RISCV)
LLVM_MUON ?= $(RK_ROOT)/llvm/llvm-muon
LIB := $(RK_ROOT)/lib
SOC := $(RK_ROOT)/soc

MU_CXX := $(LLVM_MUON)/bin/clang++
MU_OBJDUMP := $(LLVM_MUON)/bin/llvm-objdump
MU_STACK_WORD_STRIDE ?= 16

MU_CFLAGS := --sysroot=$(RISCV_SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -nodefaultlibs \
  -Xclang -target-feature -Xclang +vortex -march=rv32im_zfinx_zhinx -mabi=ilp32 \
  -O3 -fno-math-errno -std=c++20 -mcmodel=medany -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections \
  -I$(LIB)/include -I$(NIGHTLY_DIR)/common \
  -DRADIANCE -DRADIANCE_DEVICE -DNDEBUG -DLLVM_VORTEX \
  -mllvm -riscv-stack-word-stride=$(MU_STACK_WORD_STRIDE)
MU_LIBC_INCLUDE ?= $(realpath $(dir $(RISCV64_TOOLCHAIN_PATH))/riscv-tools/riscv64-unknown-elf/include)
ifneq ($(MU_LIBC_INCLUDE),)
MU_CFLAGS += -idirafter $(MU_LIBC_INCLUDE)
endif
MU_CFLAGS += $(KERNEL_MU_CFLAGS)

# Register budget.  Muon's renamer allocates a physical register on a warp's FIRST write of an
# architectural register and never frees it; a core has 256 (Rename.scala:123 asserts on
# overflow).  So warps_per_core * registers_per_warp must stay <= 256.  OCC is the warps per
# core the kernel runs (nightly_main's occupancy and the boot spawn count, __mu_num_warps), and
# the compiler is capped to 256/OCC allocatable GPRs (x0 is never allocated).
OCC ?= 8
GPRS := $(if $(filter 1 2,$(OCC)),128,$(if $(filter 3 4,$(OCC)),64,32))
MU_CFLAGS += -DNIGHTLY_OCC=$(OCC) -mllvm -riscv-max-allocatable-gprs=$(GPRS)

MU_LINKER_SCRIPT ?= $(LIB)/linker/mu_link_nightly.ld
MU_LDFLAGS := -nodefaultlibs -nostartfiles -Wl,-Bstatic,-T,$(MU_LINKER_SCRIPT),-z,norelro -fuse-ld=lld \
  $(LIB)/libmuonrt.a $(LIB)/tohost.S
ifdef MU_USE_LIBC
MU_LDFLAGS += -L$(LLVM_MUON)/lib/riscv32-unknown-elf -lc -lm \
  -Wl,$(LLVM_MUON)/lib/clang/18/lib/riscv32-unknown-elf/libclang_rt.builtins.a
endif

HOST_PREFIX := $(RISCV64_TOOLCHAIN_PATH)/bin/riscv64-unknown-elf
HOST_CC  := $(HOST_PREFIX)-gcc
HOST_CXX := $(HOST_PREFIX)-g++
HOST_LD  := $(HOST_PREFIX)-ld
HOST_OBJCOPY := $(HOST_PREFIX)-objcopy
HOST_OBJDUMP := $(HOST_PREFIX)-objdump
HOST_CFLAGS := -march=rv64imafd -mabi=lp64d -mcmodel=medany -ffreestanding -fno-common \
  -fno-builtin-printf -O2 -I$(LIB)/include -I$(NIGHTLY_DIR)/common $(KERNEL_HOST_CFLAGS)
HOST_LDFLAGS := -static -specs=htif_nano.specs

SMS ?= 1 2

# Generator-argument stamp: data rules depend on $(GEN_STAMP), which is rewritten whenever
# GEN_ARGS (the data generator's arguments) changes, so `make GEN_ARGS=...` regenerates.
GEN_STAMP := build/gen_args.stamp
$(shell mkdir -p build; printf '%s\n' "$(GEN_ARGS)" | cmp -s - $(GEN_STAMP) || printf '%s\n' "$(GEN_ARGS)" > $(GEN_STAMP))

# DEFS stamp: the objects depend on $(DEFS_STAMP), rewritten whenever DEFS changes, so
# `make DEFS=...` recompiles even when no source changed.
MU_ADDR_HASH ?= 1
MU_ADDR_HASH_ARGS := --base 0x0 --size 0x80000000 --unit 32 --slices 4
DEFS_STAMP := build/defs.stamp
STAMP_TEXT := $(DEFS) MU_ADDR_HASH=$(MU_ADDR_HASH)
$(shell mkdir -p build; printf '%s\n' "$(STAMP_TEXT)" | cmp -s - $(DEFS_STAMP) || printf '%s\n' "$(STAMP_TEXT)" > $(DEFS_STAMP))

MU_OBJCOPY := $(LLVM_MUON)/bin/llvm-objcopy
# one object per blob: build/blobs/<name>.o holds <name>.bin in section .regionN
blob_obj = build/blobs/$(basename $(notdir $(1))).o
blob_sec = $(patsubst .%,%,$(suffix $(basename $(notdir $(1)))))
BLOB_OBJS := $(foreach b,$(BLOBS),$(call blob_obj,$(b)))
define BLOB_RULE
$(call blob_obj,$(1)): $(1)
	mkdir -p build/blobs
	$(MU_OBJCOPY) -I binary -O elf32-littleriscv \
	  --rename-section .data=.$(call blob_sec,$(1)),alloc,load,contents,data $$< $$@
endef
$(foreach b,$(BLOBS),$(eval $(call BLOB_RULE,$(b))))

.DEFAULT_GOAL := all
all: $(foreach s,$(SMS),build/$(s)sm/$(PROJECT).soc.elf)

define SM_RULES
build/$(1)sm/$(PROJECT).mu.o: $(MU_SRCS) $(DEPS) $(DEFS_STAMP) $(NIGHTLY_DIR)/nightly.mk $(wildcard $(LIB)/include/nightly/*.h) | build/$(1)sm
	$(MU_CXX) $(MU_CFLAGS) -DNIGHTLY_CLUSTERS=$(1) $(DEFS) -c $(firstword $(MU_SRCS)) -o $$@
build/$(1)sm/$(PROJECT).radiance.elf: build/$(1)sm/$(PROJECT).mu.o $(BLOB_OBJS)
	$(MU_CXX) $(MU_CFLAGS) $$< $(BLOB_OBJS) $(MU_LDFLAGS) -o $$@
	$(MU_OBJDUMP) -d $$@ > build/$(1)sm/$(PROJECT).radiance.dump
build/$(1)sm/$(PROJECT).host.o: $(HOST_SRCS) $(DEPS) $(DEFS_STAMP) | build/$(1)sm
	$(HOST_CXX) $(HOST_CFLAGS) -DNIGHTLY_CLUSTERS=$(1) $(DEFS) -c $(firstword $(HOST_SRCS)) -o $$@
build/$(1)sm/$(PROJECT).fuse.elf: build/$(1)sm/$(PROJECT).radiance.elf $(DEFS_STAMP) $(SOC)/scramble_gpu_elf.py
ifeq ($(MU_ADDR_HASH),1)
	python3 $(SOC)/scramble_gpu_elf.py $$< $$@ $(MU_ADDR_HASH_ARGS) > /dev/null
else
	cp $$< $$@
endif
build/$(1)sm/$(PROJECT).soc.elf: build/$(1)sm/$(PROJECT).fuse.elf build/$(1)sm/$(PROJECT).host.o
	cd build/$(1)sm && RV32_ELF=$(PROJECT).fuse.elf OUT=$(PROJECT).soc.elf \
	  RV64_START=$(SOC)/start.S RV64_MAIN= RV64_OBJS=$(PROJECT).host.o \
	  RV64_CFLAGS="$(HOST_CFLAGS)" RV64_LDFLAGS="$(HOST_LDFLAGS)" RV64_LIBS= \
	  CC=$(HOST_CC) LD=$(HOST_LD) RV64_LINK=$(HOST_CC) OBJCOPY=$(HOST_OBJCOPY) READELF=readelf \
	  $(SOC)/fuse_rv32_into_rv64.sh > /dev/null
ifeq ($(MU_ADDR_HASH),1)
	echo "unit=32 slices=4 size=0x80000000 gpu_base=0x100000000" > $$@.addrhash
	$(HOST_OBJCOPY) --add-section .radiance_addr_hash=$$@.addrhash $$@
	rm -f $$@.addrhash
endif
build/$(1)sm:
	mkdir -p $$@
endef
$(foreach s,$(sort 1 2 $(SMS)),$(eval $(call SM_RULES,$(s))))

clean:
	rm -rf build

.PHONY: all clean
