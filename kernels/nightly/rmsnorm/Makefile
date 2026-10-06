PROJECT   = rmsnorm
MU_SRCS   = kernel.cpp
HOST_SRCS = host.cpp
DEPS      = rmsnorm_data.h
BLOBS     = data/rmsnorm.region1.bin data/rmsnorm.region2.bin data/rmsnorm.region3.bin
include ../nightly.mk
rmsnorm_data.h data/rmsnorm.region1.bin data/rmsnorm.region2.bin data/rmsnorm.region3.bin &: gen_data.py $(GEN_STAMP)
	python3 gen_data.py $(GEN_ARGS)
