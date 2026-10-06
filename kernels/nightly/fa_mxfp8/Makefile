# MXFP8 flash attention.  make ARGS="--H 2 --SQ 128 --SK 1024" (defaults; --SQ 256 gives 4 query
# tiles per cluster on 2 SMs).  DEFS="-DFA_ILV=0" assigns tiles to clusters in blocks.
# Measured (RTL sim, mesh utilization over the kernel): 1 SM 67.1 %; 2 SM 54.0 % (SQ 128),
# 62.3 % (SQ 256).
PROJECT   = fa_mxfp8
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = fa_data.h
BLOBS     = data/fa.region1.bin data/fa.region2.bin data/fa.region3.bin
OCC       = 4
ARGS     ?=
GEN_ARGS := $(ARGS)
KERNEL_MU_CFLAGS = -ffp-contract=off
include ../nightly.mk
fa_data.h data/fa.region1.bin data/fa.region2.bin data/fa.region3.bin &: gen_data.py $(GEN_STAMP)
	python3 gen_data.py $(GEN_ARGS)
