PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

# Match the MX harness inliner cap to stay under the 256-entry physical register file
# (mxgemm_lib is heavily inlined; every warp slot runs main()).
MU_CFLAGS += -mllvm -inline-threshold=256
