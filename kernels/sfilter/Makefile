PROJECT = sfilter

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: sfilter.py
	python3 sfilter.py

kernel.mu.o: data
