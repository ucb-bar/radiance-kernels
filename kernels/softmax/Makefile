PROJECT = softmax

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: softmax.py
	python3 softmax.py

kernel.mu.o: data
