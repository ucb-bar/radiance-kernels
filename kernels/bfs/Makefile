PROJECT = bfs

MU_SRCS = \
	bfs1_l0.cpp \
	bfs2_l0.cpp \
	bfs1_l1.cpp \
	bfs2_l1.cpp \
	bfs1_l2.cpp \
	bfs2_l2.cpp

HOST_SRCS = host.cpp

include ../common.mk

data: bfs.py bfs1_kernel.inc bfs2_kernel.inc verify_trace.py host.cpp Makefile
	python3 bfs.py

$(addsuffix .mu.o,$(basename $(MU_SRCS))): data
