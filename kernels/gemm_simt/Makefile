PROJECT = gemm_simt

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: gemm.py
	python3 gemm.py

kernel.mu.o: data
