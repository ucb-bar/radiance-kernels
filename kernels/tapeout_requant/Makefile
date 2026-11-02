PROJECT = tapeout_requant

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

HOST_CXXFLAGS += $(RQ_HOST_DEFS)
kernel.mu.o host.host.o: rq.h
