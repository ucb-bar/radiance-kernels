PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

# The MX harness sits right at the 256-entry physical register file limit: mxgemm_lib is
# heavily inlined (common.mk sets -inline-threshold=262144) and every one of the 8 warp
# slots executes main(), so the summed first-writes hit RTL Rename.scala's assert.
# Rein the inliner in for this kernel only.
MU_CFLAGS += -mllvm -inline-threshold=256
