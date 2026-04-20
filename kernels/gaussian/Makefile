PROJECT = gaussian

MU_SRCS = \
	fan1_t0.cpp \
	fan2_t0.cpp \
	fan1_t1.cpp \
	fan2_t1.cpp \
	fan1_t2.cpp \
	fan2_t2.cpp \
	fan1_t3.cpp \
	fan2_t3.cpp

HOST_SRCS = host.cpp

include ../common.mk

data: gaussian.py fan1_kernel.inc fan2_kernel.inc verify_trace.py host.cpp Makefile
	python3 gaussian.py

$(addsuffix .mu.o,$(basename $(MU_SRCS))): data
