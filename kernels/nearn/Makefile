PROJECT = nearn

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: nearn.py
	python3 nearn.py

kernel.mu.o: data
