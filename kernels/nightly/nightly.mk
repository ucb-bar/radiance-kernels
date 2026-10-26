# nightly.mk: build rules for the kernels under kernels/nightly (nightly RTL, see
# lib/include/nightly/README.md).
#
# A kernel Makefile sets PROJECT, MU_SRCS (one device source with main()), HOST_SRCS, optional
# DEPS (generated headers), then includes this file.  Every kernel is built once per cluster
# count, into its own directory so the objects never mix:
#
#   build/1sm/$(PROJECT).soc.elf   NIGHTLY_CLUSTERS=1  (cluster 0 only)
#   build/2sm/$(PROJECT).soc.elf   NIGHTLY_CLUSTERS=2  (clusters 0 and 1)
#   build/4sm/$(PROJECT).soc.elf   NIGHTLY_CLUSTERS=4  (4-cluster configs, e.g. RadianceHBMConfig)
#
# `make` builds 1sm and 2sm.  `make SMS="1 2 4"` adds 4sm.  Extra defines: DEFS="-DFOO=1 -DBAR".
# The host launches clusters 0 .. NIGHTLY_CLUSTERS-1; the other clusters wait for LAUNCH forever.
#
# MU_ADDR_HASH (default 1): the hashed HBM configs hash GPU memory over the GPU DRAM channels
# (radiance WithGPUAddressHash).  The defaults below match radiance WithRadianceHBM's defaults (4 GPU
# L2 slices, one channel each): Radiance2SMHBMConfig, RadianceE4M3MxGemmini*HBMConfig,
# RadianceFP4HBMConfig and the FireSim *HBMRadianceConfig targets.  RadianceHBMConfig (4 SMs, 8 slices
# x 2 channels) needs MU_ADDR_HASH_SLICES=16 MU_ADDR_HASH_CPS=2 MU_ADDR_HASH_CSS=128.
# SimDRAM +loadmem and FireSim LoadMem write DRAM directly, so the GPU load segments are scrambled
# into the hashed layout before the fuse step (soc/scramble_gpu_elf.py).  Use MU_ADDR_HASH=0 only for a loader that goes through the hash
# (TSI, RadianceHBMTSIConfig) or for an unhashed config.  MU_ADDR_HASH_SLICES (default 4) is the
# number of parts the hash spreads GPU memory over and MU_ADDR_HASH_CPS (default 1) the DRAM channels
# per GPU L2 slice, MU_ADDR_HASH_CSS (default 0) the per-channel set stride (--chan-set-stride);
# all must match the RTL.  MU_ADDR_HASH_SCRAMBLER overrides the scrambler script.
# All are part of the DEFS stamp, so switching any rebuilds.
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

# freestanding libstdc++ headers (<type_traits> in mu_intrinsics.h)
MU_CXX_INCLUDE ?= $(lastword $(sort $(wildcard $(MU_LIBC_INCLUDE)/c++/*)))
ifneq ($(MU_CXX_INCLUDE),)
MU_CFLAGS += -idirafter $(MU_CXX_INCLUDE) -idirafter $(MU_CXX_INCLUDE)/riscv64-unknown-elf
endif

# Register budget.  Muon's renamer allocates a physical register on a warp's FIRST write of an
# architectural register and never frees it; a core has 256 (Rename.scala:123 asserts on
# overflow).  So warps_per_core * registers_per_warp must stay <= 256.  OCC is the warps per
# core the kernel runs (nightly_main's occupancy and the boot spawn count, __mu_num_warps), and
# the compiler is capped to 256/OCC allocatable GPRs (x0 is never allocated).
OCC ?= 8
GPRS := $(if $(filter 1 2,$(OCC)),128,$(if $(filter 3 4,$(OCC)),64,32))
MU_CFLAGS += -DNIGHTLY_OCC=$(OCC) -mllvm -riscv-max-allocatable-gprs=$(GPRS)

# The Muon runtime, built here when missing or older than its sources (a fresh checkout has no lib/libmuonrt.a):
# llvm-muon's llvm-ar (no riscv32 gcc-ar needed) and the freestanding libc / libstdc++ headers the nightly headers use.
LIB_SRCS := $(wildcard $(LIB)/src/*) $(wildcard $(LIB)/include/*.h) $(wildcard $(LIB)/include/nightly/*.h)
$(LIB)/libmuonrt.a: $(LIB_SRCS)
	$(MAKE) -C $(LIB) libmuonrt.a LLVM_MUON=$(LLVM_MUON) AR=$(LLVM_MUON)/bin/llvm-ar \
	  RISCV_TOOLCHAIN_PATH=$(RISCV_TOOLCHAIN_PATH) MU_STACK_WORD_STRIDE=$(MU_STACK_WORD_STRIDE) \
	  MU_EXTRA_DEFS="$(if $(MU_LIBC_INCLUDE),-idirafter $(MU_LIBC_INCLUDE)) $(if $(MU_CXX_INCLUDE),-idirafter $(MU_CXX_INCLUDE) -idirafter $(MU_CXX_INCLUDE)/riscv64-unknown-elf)"

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
MU_ADDR_HASH_SLICES ?= 4
MU_ADDR_HASH_CPS ?= 1
MU_ADDR_HASH_CSS ?= 0
MU_ADDR_HASH_SCRAMBLER ?= $(SOC)/scramble_gpu_elf.py
MU_ADDR_HASH_ARGS := --base 0x0 --size 0x80000000 --unit 32 --slices $(MU_ADDR_HASH_SLICES) \
  $(if $(filter-out 1,$(MU_ADDR_HASH_CPS)),--channels-per-slice $(MU_ADDR_HASH_CPS)) \
  $(if $(filter-out 0,$(MU_ADDR_HASH_CSS)),--chan-set-stride $(MU_ADDR_HASH_CSS))
MU_ADDR_HASH_TEXT := unit=32 slices=$(MU_ADDR_HASH_SLICES)$(if $(filter-out 1,$(MU_ADDR_HASH_CPS)), cps=$(MU_ADDR_HASH_CPS))$(if $(filter-out 0,$(MU_ADDR_HASH_CSS)), css=$(MU_ADDR_HASH_CSS))
DEFS_STAMP := build/defs.stamp
STAMP_TEXT := $(DEFS) MU_ADDR_HASH=$(MU_ADDR_HASH) MU_ADDR_HASH_SLICES=$(MU_ADDR_HASH_SLICES) MU_ADDR_HASH_CPS=$(MU_ADDR_HASH_CPS) MU_ADDR_HASH_CSS=$(MU_ADDR_HASH_CSS)
$(shell mkdir -p build; printf '%s\n' "$(STAMP_TEXT)" | cmp -s - $(DEFS_STAMP) || printf '%s\n' "$(STAMP_TEXT)" > $(DEFS_STAMP))

MU_OBJCOPY := $(LLVM_MUON)/bin/llvm-objcopy
# one object per blob: $(BLOB_DIR)/<name>.o holds <name>.bin in section .regionN.  A project that builds several
# data sets with the same blob names (llama_e2e: one per model) sets BLOB_DIR per data set, or a build whose data is
# up to date links the objects of whichever data set was converted last.
BLOB_DIR ?= build/blobs
blob_obj = $(BLOB_DIR)/$(basename $(notdir $(1))).o
blob_sec = $(patsubst .%,%,$(suffix $(basename $(notdir $(1)))))
BLOB_OBJS := $(foreach b,$(BLOBS),$(call blob_obj,$(b)))
define BLOB_RULE
$(call blob_obj,$(1)): $(1)
	mkdir -p $(BLOB_DIR)
	$(MU_OBJCOPY) -I binary -O elf32-littleriscv \
	  --rename-section .data=.$(call blob_sec,$(1)),alloc,load,contents,data $$< $$@
endef
$(foreach b,$(BLOBS),$(eval $(call BLOB_RULE,$(b))))

.DEFAULT_GOAL := all
all: $(foreach s,$(SMS),build/$(s)sm/$(PROJECT).soc.elf)

define SM_RULES
build/$(1)sm/$(PROJECT).mu.o: $(MU_SRCS) $(DEPS) $(DEFS_STAMP) $(NIGHTLY_DIR)/nightly.mk $(wildcard $(LIB)/include/nightly/*.h) | build/$(1)sm
	$(MU_CXX) $(MU_CFLAGS) -DNIGHTLY_CLUSTERS=$(1) $(DEFS) -c $(firstword $(MU_SRCS)) -o $$@
build/$(1)sm/$(PROJECT).radiance.elf: build/$(1)sm/$(PROJECT).mu.o $(BLOB_OBJS) $(LIB)/libmuonrt.a
	$(MU_CXX) $(MU_CFLAGS) $$< $(BLOB_OBJS) $(MU_LDFLAGS) -o $$@
	$(MU_OBJDUMP) -d $$@ > build/$(1)sm/$(PROJECT).radiance.dump
build/$(1)sm/$(PROJECT).host.o: $(HOST_SRCS) $(DEPS) $(DEFS_STAMP) $(NIGHTLY_DIR)/nightly.mk $(wildcard $(LIB)/include/nightly/*.h) | build/$(1)sm
	$(HOST_CXX) $(HOST_CFLAGS) -DNIGHTLY_CLUSTERS=$(1) $(DEFS) -c $(firstword $(HOST_SRCS)) -o $$@
build/$(1)sm/$(PROJECT).fuse.elf: build/$(1)sm/$(PROJECT).radiance.elf $(DEFS_STAMP) $(MU_ADDR_HASH_SCRAMBLER)
ifeq ($(MU_ADDR_HASH),1)
	python3 $(MU_ADDR_HASH_SCRAMBLER) $$< $$@ $(MU_ADDR_HASH_ARGS) > /dev/null
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
	echo "$(MU_ADDR_HASH_TEXT) size=0x80000000 gpu_base=0x100000000" > $$@.addrhash
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
