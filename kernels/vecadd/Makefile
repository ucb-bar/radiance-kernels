PROJECT = vecadd

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: vecadd.py
	python3 vecadd.py

kernel.mu.o: data
