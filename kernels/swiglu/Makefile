PROJECT = swiglu

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: swiglu.py
	python3 swiglu.py

kernel.mu.o: data
