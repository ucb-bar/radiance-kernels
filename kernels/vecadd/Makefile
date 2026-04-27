PROJECT = vecadd

MU_SRCS = \
	kernel.cpp \
	kernel_ilp2.cpp \
	kernel_ilp4.cpp \
	kernel_ilp4_ou2.cpp \
	kernel_ilp8.cpp \
	kernel_ilp2_w8.cpp \
	kernel_ilp4_w2.cpp \
	kernel_ilp4_w8.cpp
HOST_SRCS = host.cpp

include ../common.mk

data: vecadd.py kernel_common.inc extract_rtl_metrics.py
	python3 vecadd.py

$(addsuffix .mu.o,$(basename $(MU_SRCS))): data
