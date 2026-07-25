PROJECT = kernel
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(MU_OBJS): data

include ../common.mk
