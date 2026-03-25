PROJECT = mxgemm

MU_SRCS = mxgemm.cpp mxgemm_singletile.cpp \
	  mxgemm.fp8.singletile.tm128tn128tk256.fullout.cpp \
	  mxgemm.fp8.singletile.tm64tn64tk64.fullout.cpp \
	  mxgemm.fp6.singletile.tm128tn128tk512.fullout.cpp \
	  requant.cpp
HOST_SRCS = host.cpp

MXGEMM_MU_OBJS := $(addsuffix .mu.o,$(basename $(MU_SRCS)))
$(MXGEMM_MU_OBJS): mxgemm_lib.hpp

include ../common.mk
