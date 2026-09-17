PROJECT = pointerchase

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: pointerchase.py
	python3 pointerchase.py

kernel.mu.o: data kernel_impl.hpp
