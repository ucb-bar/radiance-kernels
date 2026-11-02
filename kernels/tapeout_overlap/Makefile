PROJECT = tapeout_overlap

MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp

include ../common.mk

# Host-side knobs: OV_HOST_DEFS="-DOV_HOST_REPS=8"
HOST_CXXFLAGS += $(OV_HOST_DEFS)

# ov_data.h is generated: python3 gen_data.py 128 128 128 128 128 128 ov_data.h
kernel.mu.o: ov.h ../tapeout_gemm/mxg.hpp
host.host.o: ov.h ov_data.h
