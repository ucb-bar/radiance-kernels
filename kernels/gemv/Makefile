PROJECT = gemv

MU_SRCS = kernel.cpp

include ../common.mk

data: gemv.py
	python3 gemv.py

kernel.mu.o: data
