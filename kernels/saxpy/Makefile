PROJECT = saxpy

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: saxpy.py
	python3 saxpy.py

kernel.mu.o: data
