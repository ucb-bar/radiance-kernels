# MX GEMM.  The data (and so the ELF) depends on the format and shape:
#   make FMT=fp8|fp4 ARGS="--M 512 --N 512 --K 512"   -> build/{1,2}sm/gemm_mx.soc.elf
#
# Measured configurations (RTL sim, mesh utilization over the kernel):
#   make FMT=fp8 ARGS="--M 512 --N 512 --K 512 --TK 256 --pretile --bdup"
#     1 SM 76.5 %, 2 SM 67.5 %; with DEFS="-DGEMM_MANAGED=1": 1 SM 77.4 %, 2 SM 66.8 %
#   make FMT=fp4 ARGS="--M 512 --N 512 --K 1024 --TK 512 --pretile --bdup"
#     1 SM 61.9 %, 2 SM 47.1 % (operand DMA bound)
#   M=1024 on 2 SM: fp8 72.2 %, fp4 50.8 %
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
