PROJECT = mxgemm

MU_SRCS = mxgemm.cpp mxgemm_singletile.cpp \
	  mxgemm.fp8.singletile.tm128tn128tk256.fullout.cpp \
	  mxgemm.fp8.singletile.tm64tn64tk64.fullout.cpp \
	  mxgemm.fp6.singletile.tm128tn128tk512.fullout.cpp
HOST_SRCS = host.cpp

mxgemm.mu.o: mxgemm_lib.hpp
# %.mu.o: mxgemm_lib.hpp

include ../common.mk
