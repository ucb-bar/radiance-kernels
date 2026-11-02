PROJECT = tapeout_gemm

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

# Host-side knobs, appended after common.mk (it uses ?=, so an environment HOST_CXXFLAGS would
# replace -march/-mabi).  TG_HOST_DEFS="-DTG_MODE=1 -DTG_REPS=16"
HOST_CXXFLAGS += $(TG_HOST_DEFS)

# gemm_data.h is generated: python3 gen_data.py M N K TM TN TK
kernel.mu.o: gemm.h
host.host.o: gemm.h gemm_data.h
