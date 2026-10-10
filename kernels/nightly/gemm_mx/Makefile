# MX GEMM.  The data (and so the ELF) depends on the format and shape:
#   make FMT=fp8|fp4 ARGS="--M 512 --N 512 --K 512"   -> build/{1,2}sm/gemm_mx.soc.elf
#
# Measured configurations (RTL sim, RadianceHBMConfig with the GPU address hash, mesh utilization
# over the kernel / from the first to the last compute):
#   make FMT=fp8 ARGS="--M 512 --N 512 --K 512 --TK 256 --pretile --bdup"
#     1 SM 87.6 % / 92.3 %, 2 SM 80.5 % / 91.2 %
#   make FMT=fp4 ARGS="--M 512 --N 512 --K 1024 --TK 512 --pretile --bdup"
#     1 SM 73.4 % / 80.0 %, 2 SM 60.9 % / 75.9 % (needs the MX Gemmini, not the E4M3 one that
#     RadianceHBMConfig now has)
PROJECT   = gemm_mx
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = gemm_data.h
BLOBS     = data/gemm.region1.bin data/gemm.region2.bin data/gemm.region3.bin
FMT  ?= fp8
ARGS ?=
GEN_ARGS := --fmt $(FMT) $(ARGS)
include ../nightly.mk
gemm_data.h data/gemm.region1.bin data/gemm.region2.bin data/gemm.region3.bin &: gen_data.py $(GEN_STAMP)
	python3 gen_data.py $(GEN_ARGS)
