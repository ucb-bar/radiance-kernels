PROJECT = flops

MU_SRCS = flops_fp16.cpp flops_fp32.cpp flops_int.cpp flops_exp_fp16.cpp \
	  bandwidth_gmem_lw_l0hit.cpp bandwidth_gmem_sw_l0hit.cpp \
	  bandwidth_smem_lw.cpp bandwidth_smem_sw.cpp
HOST_SRCS = host.cpp

flops_fp16.mu.o: lib.h
flops_fp32.mu.o: lib.h
flops_int.mu.o: lib.h
flops_exp_fp16.mu.o: lib.h

include ../common.mk
