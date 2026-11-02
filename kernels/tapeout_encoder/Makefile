PROJECT = tapeout_encoder

MU_SRCS = kernel.cpp
MU_SRC_DEPS = fa_stage.cpp ln_stage.cpp
HOST_SRCS = host.cpp

include ../common.mk

HOST_CXXFLAGS += $(EN_HOST_DEFS)

# ref_data.h is generated: python3 gen_ref.py
kernel.mu.o: enc.h
fa_stage.mu.o: enc.h ../fa_stable_fpga/kernel.cpp ../fa_stable_fpga/mxgemm_core.hpp ../fa_stable_fpga/flash_mx_impl.hpp
host.host.o: enc.h ref_data.h

# The SIMT stages are unrolled; capping the allocator at 64 registers keeps each stage's per-warp
# register union under the renamer budget at EN_OCC = 3 (runs/tapeout-perf/scripts/regs.py).
# fa_stage.mu.o keeps the FA kernel's verified code generation.
# FA_MU_CFLAGS adds flags to fa_stage.mu.o only (e.g. -mllvm -muon-hazard-rename-used-only: the FA
# object allocates from 128 registers, so renaming into unused ones would exceed the renamer budget).
fa_stage.mu.o: MU_CFLAGS += $(FA_MU_CFLAGS)
kernel.mu.o: MU_CFLAGS += -mllvm -riscv-max-allocatable-gprs=64
# ln_stage.mu.o: the LN stages at 4 warps per core (EN_LN_OCC4) must write <= 54 registers.
# (-riscv-max-allocatable-gprs accepts only 32 / 64 / 128, so x54-x63 are reserved instead; a
# per-function target("reserve-xN") attribute is ignored by this LLVM)
ln_stage.mu.o: MU_CFLAGS += -mllvm -riscv-max-allocatable-gprs=64 \
	$(foreach r,54 55 56 57 58 59 60 61 62 63,-Xclang -target-feature -Xclang +reserve-x$(r))
ln_stage.mu.o: kernel.cpp enc.h

# EN_FA_FIRST=1: link fa_stage.mu.o ahead of kernel.mu.o, so the attention code's address does not
# move when kernel.mu.o changes size (the attention output depends on that address: hzm2 vs hzm2a,
# workspace-notes/MUON_LLVM_HAZARD.md).
ifeq ($(EN_FA_FIRST),1)
# EN_KERNEL_ALIGN=1 adds kernel_pad.S between the two: the kernel code starts at a 16 KB boundary.
EN_PAD_OBJ := $(if $(filter 1,$(EN_KERNEL_ALIGN)),kernel_pad.mu.o)
kernel_pad.mu.o: kernel_pad.S
	$(MU_CC) $(MU_CFLAGS) -c $< -o $@
kernel.radiance.elf: kernel.mu.o $(MU_LIB_OBJS) $(MU_BIN_OBJS) $(BINFILES) $(EN_PAD_OBJ)
	$(MU_CXX) $(MU_CFLAGS) fa_stage.mu.o $(EN_PAD_OBJ) $< $(filter-out fa_stage.mu.o,$(MU_LIB_OBJS)) $(MU_BIN_OBJS) $(MU_LDFLAGS) -o $@
endif
