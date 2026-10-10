# fa_mxfp8 with SPAD_REQUANT (needs has_spad_requant in the Radiance build): make (default: H=2 SQ=128 SK=1024 d=128,
# keys in natural order).  ARGS="..." passes gen_data.py options (always with --novperm).
PROJECT   = fa_mxfp8_sr
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = fa_data.h
BLOBS     = data/fa.region1.bin data/fa.region2.bin data/fa.region3.bin
OCC       = 4
ARGS     ?=
GEN_ARGS := --novperm $(ARGS)
KERNEL_MU_CFLAGS = -ffp-contract=off -DNIGHTLY_MX_WAIT_READY -DNIGHTLY_MX_RETIRE
# Built for the unhashed RadianceE4M3MxGemmini*Config: plain ELF.  fa_mxfp8_sr_hbm builds the same kernel for the
# *HBMConfig variants (split L2, hashed GPU memory).
MU_ADDR_HASH ?= 0
include ../nightly.mk
fa_data.h data/fa.region1.bin data/fa.region2.bin data/fa.region3.bin &: gen_data.py $(GEN_STAMP)
	python3 gen_data.py $(GEN_ARGS)
