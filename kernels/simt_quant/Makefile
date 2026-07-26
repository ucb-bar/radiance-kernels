PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

# Register-rename budget: the 8 hardware warp slots share a 256-entry physical
# register file. kernel.cpp already partitions the hot path via always_inline /
# noinline attributes and a low occupancy (KERNEL_OCCUPANCY=2); this also caps
# common.mk's -inline-threshold=262144 so the compiler can't re-fuse the helpers
# back into the block loop and re-inflate the per-warp first-write union.
MU_CFLAGS += -mllvm -inline-threshold=64
