PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

# Rein in the inliner: mxgemm_lib is heavily inlined and every warp
# slot runs main(), so summed first-writes hit the RTL Rename.scala physical-regfile assert.
MU_CFLAGS += -mllvm -inline-threshold=256
