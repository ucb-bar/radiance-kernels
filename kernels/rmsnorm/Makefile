PROJECT = rmsnorm

MU_SRCS = rmsnorm.cpp
MU_USE_LIBC = 1

HOST_SRCS = host.cpp

include ../common.mk

rmsnorm.mu.o: data
data: rmsnorm.py
	python3 rmsnorm.py
